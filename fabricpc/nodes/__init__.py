"""
Node types for JAX predictive coding networks.
"""

from fabricpc.nodes.base import (
    SlotSpec,
    Slot,
    NodeBase,
)
from fabricpc.nodes.linear import LinearNode, LinearExplicitGrad

from typing import Type
def get_node_class_from_type(node_type: str) -> Type[NodeBase]:
    """
    Args:
        node_type: Type of node ("linear", "transformer", etc.)
    Returns:
        Node class (not instance, since nodes are collections of static methods)
    Raises:
        ValueError: If node_type is not recognized
    """

    node_types = {
        "linear": LinearNode,
        "linear_explicit_grad": LinearExplicitGrad,
    }

    if node_type.lower() not in node_types:
        raise ValueError(
            f"Unknown node type '{node_type}'. "
            f"Supported types: {list(node_types.keys())}"
        )
    return node_types[node_type.lower()]

__all__ = [
    "SlotSpec",
    "Slot",
    "NodeBase",
    "get_node_class_from_type",
    "LinearNode",
    "LinearExplicitGrad",
]
