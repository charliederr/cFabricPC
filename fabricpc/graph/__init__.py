"""
JAX graph for predictive coding networks.
"""

from fabricpc.graph.graph_net import (
    build_graph_structure,
    initialize_params,
    initialize_state,
    create_pc_graph,
    set_latents_to_clamps,
)

from fabricpc.graph.state_initializer import (
    StateInitBase,
    register_state_init,
    get_state_init_class,
    list_state_init_types,
    initialize_graph_state,
    get_default_graph_state_init,
)

__all__ = [
    # Graph construction
    "build_graph_structure",
    "initialize_params",
    "initialize_state",
    "create_pc_graph",
    "set_latents_to_clamps",
    # State initializer registry
    "StateInitBase",
    "register_state_init",
    "get_state_init_class",
    "list_state_init_types",
    "initialize_graph_state",
    "get_default_graph_state_init",
]
