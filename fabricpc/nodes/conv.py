"""
Convolutional nodes for JAX predictive coding networks.
"""

from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.activations import ReLUActivation, IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeParams, NodeState, NodeInfo


class Conv2DNode(NodeBase):
    """
    2D Convolutional node using JAX's lax.conv_general_dilated.

    Expects inputs in NHWC format (batch, height, width, channels).
    Output shape should be specified as (H_out, W_out, C_out).

    Parameters:
        kernel_size: Tuple[int, int] - Kernel dimensions (kH, kW)
        stride: Tuple[int, int] - Stride (default: (1, 1))
        padding: str - "VALID" or "SAME" (default: "SAME")
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        name: str,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv2D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Conv2D fan_in = C_in * kH * kW (kernel receptive field)."""
        kernel_size = config.get("kernel_size", (1, 1))
        C_in = source_shape[-1]  # NHWC: channels last
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize convolution kernels and biases.

        Kernel shape: (kH, kW, C_in, C_out)
        Bias shape: (1, 1, 1, C_out) for NHWC broadcasting
        """
        if config is None:
            config = {}
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]  # Last dim is channels (NHWC)

        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.05)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]  # Input channels from source

            kernel_param_shape = (
                kernel_size[0],
                kernel_size[1],
                in_channels,
                out_channels,
            )

            weights_dict[edge_key] = initialize(
                keys[i], kernel_param_shape, weight_init
            )

        # Initialize bias
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias = jnp.zeros((1, 1, 1, out_channels))
        else:
            bias = jnp.array([])

        return NodeParams(weights=weights_dict, biases={"b": bias} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass using JAX conv2d.

        Computes: conv2d(x, kernel) + bias -> activation -> error -> energy
        """
        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        # Accumulate convolution outputs from all inputs
        pre_activation = jnp.zeros((batch_size, *out_shape))

        for edge_key, x in inputs.items():
            kernel = params.weights[edge_key]
            # Use JAX's lax.conv_general_dilated for the convolution
            conv_out = jax.lax.conv_general_dilated(
                x,  # input: NHWC
                kernel,  # kernel: HWIO
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            pre_activation = pre_activation + conv_out

        # Add bias if present
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation
        activation = node_info.activation  # ActivationBase instance
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Compute error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


class Pool2DNode(NodeBase):
    """
    2D Pooling node (Average or Max) using JAX's lax.reduce_window.

    Expects inputs in NHWC format (batch, height, width, channels).

    Parameters:
        window_shape: Tuple[int, int] - Window dimensions (wH, wW)
        stride: Tuple[int, int] - Stride (default: same as window_shape)
        padding: str - "VALID" or "SAME" (default: "SAME")
        pooling_type: str - "avg" or "max" (default: "avg")
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        name: str,
        window_shape: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None,
        padding: str = "SAME",
        pooling_type: str = "avg",
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        **kwargs,
    ):
        if stride is None:
            stride = window_shape

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            window_shape=window_shape,
            stride=stride,
            padding=padding,
            pooling_type=pooling_type,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        window_shape = config.get("window_shape")
        stride = config.get("stride")
        padding = config.get("padding", "SAME")
        pooling_type = config.get("pooling_type", "avg")

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        # Accumulate inputs
        x_sum = None
        for edge_key, x in inputs.items():
            if x_sum is None:
                x_sum = x
            else:
                x_sum = x_sum + x

        # Window shape and stride for lax.reduce_window
        # Dimensions: (batch, height, width, channels)
        window_dims = (1, window_shape[0], window_shape[1], 1)
        stride_dims = (1, stride[0], stride[1], 1)

        if pooling_type == "avg":
            pool_out = jax.lax.reduce_window(
                x_sum, 0.0, jax.lax.add, window_dims, stride_dims, padding
            )
            # Normalize by window size
            window_size = window_shape[0] * window_shape[1]
            pool_out = pool_out / window_size
        elif pooling_type == "max":
            pool_out = jax.lax.reduce_window(
                x_sum, -jnp.inf, jax.lax.max, window_dims, stride_dims, padding
            )
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(pool_out, activation.config)

        # Compute error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(
            pre_activation=pool_out,
            z_mu=z_mu,
            error=error,
        )

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state
