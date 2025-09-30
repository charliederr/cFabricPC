"""
FabricPC: A flexible, performant predictive coding framework.

FabricPC implements predictive coding networks using a pattern of nodes (state variables),
wires (edges/connections), and iterative updates for inference and learning.
"""

__version__ = "0.1.0"

# Core abstractions
from fabricpc.core.base_pc import PCNet
from fabricpc.core.graph_pc import (
    LinearPCNode,
    PCNodeBase,
    EdgeId,
    create_node_from_config,
)

# Layer implementations
from fabricpc.core.sequential_pc import PCDenseLayer

# Activation functions
from fabricpc.core.activation_functions import get_activation

# Models
from fabricpc.models.graph_net import PCGraphNet
from fabricpc.models.sequential_mlp import PC_MLP

# Optimizers
from fabricpc.training.optimizers import instantiate_optimizer

__all__ = [
    # Core
    "PCNet",
    "PCNodeBase",
    "LinearPCNode",
    "EdgeId",
    "create_node_from_config",
    # Layers
    "PCDenseLayer",
    # Activations
    "get_activation",
    # Models
    "PCGraphNet",
    "PC_MLP",
    # Training
    "instantiate_optimizer",
]
