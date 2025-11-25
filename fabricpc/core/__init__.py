"""Core JAX predictive coding components."""

# Type definitions
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    EdgeInfo,
    SlotInfo,
)

# Activation functions
from fabricpc.core.activations import (
    get_activation,
    get_activation_fn,
    get_activation_deriv,
    sigmoid,
    sigmoid_deriv,
    relu,
    relu_deriv,
    tanh,
    tanh_deriv,
    identity,
    identity_deriv,
    leaky_relu,
    leaky_relu_deriv,
    hard_tanh,
    hard_tanh_deriv,
    ACTIVATIONS,
)

# Inference functions
from fabricpc.core.inference import (
    compute_node_projection,
    compute_latent_gradients,
    compute_all_projections,
    compute_errors,
    inference_step,
    run_inference,
)

# Initialization utilities
from fabricpc.core.initialization import (
    initialize_weights,
    initialize_state_values,
    parse_state_init_config,
    get_default_weight_init,
    get_default_state_init,
)

__all__ = [
    # Types
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "NodeInfo",
    "EdgeInfo",
    "SlotInfo",
    # Activation functions
    "get_activation",
    "get_activation_fn",
    "get_activation_deriv",
    "sigmoid",
    "sigmoid_deriv",
    "relu",
    "relu_deriv",
    "tanh",
    "tanh_deriv",
    "identity",
    "identity_deriv",
    "leaky_relu",
    "leaky_relu_deriv",
    "hard_tanh",
    "hard_tanh_deriv",
    "ACTIVATIONS",
    # Inference
    "compute_node_projection",
    "compute_latent_gradients",
    "compute_all_projections",
    "compute_errors",
    "inference_step",
    "run_inference",
    # Initialization
    "initialize_weights",
    "initialize_state_values",
    "parse_state_init_config",
    "get_default_weight_init",
    "get_default_state_init",
]
