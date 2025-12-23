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
)

# Inference functions
from fabricpc.core.inference import (
    gather_inputs,
    inference_step,
    run_inference,
)

# Initializer registry
from fabricpc.core.initializers import (
    InitializerBase,
    register_initializer,
    get_initializer_class,
    list_initializer_types,
    initialize,
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
    # Inference
    "gather_inputs",
    "inference_step",
    "run_inference",
    # Initialization (backward compatible)
    "initialize_weights",
    "initialize_state_values",
    "parse_state_init_config",
    "get_default_weight_init",
    "get_default_state_init",
    # Initializer registry
    "InitializerBase",
    "register_initializer",
    "get_initializer_class",
    "list_initializer_types",
    "initialize",
]
