"""
Initialization utilities for JAX predictive coding networks.

This module provides backward-compatible wrappers that delegate to the new
registry-based initialization system in fabricpc.core.initializers.

For new code, prefer using fabricpc.core.initializers directly:
    from fabricpc.core.initializers import initialize, get_initializer_class

The functions in this module are maintained for backward compatibility.
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp


# ==============================================================================
# WEIGHT INITIALIZATION (Backward Compatible)
# ==============================================================================

def initialize_weights(
    config: Dict[str, Any],
    key: jax.Array,
    shape: Tuple[int, ...]
) -> jnp.ndarray:
    """
    Initialize weight array based on configuration.

    This function provides backward compatibility by delegating to
    the new Initializer registry.

    Args:
        config: Weight initialization configuration dict
        key: JAX random key
        shape: Shape of weight array (fan_in, fan_out)

    Returns:
        Initialized weight array

    Supported types:
        - zeros: All zeros
        - ones: All ones
        - uniform: Uniform distribution in [min, max]
        - normal: Normal distribution with mean and std
        - xavier: Xavier/Glorot initialization
        - kaiming: Kaiming/He initialization

    Example:
        >>> config = {"type": "normal", "mean": 0, "std": 0.05}
        >>> W = initialize_weights(config, key, (784, 256))
    """
    from fabricpc.core.initializers import initialize
    return initialize(key, shape, config)


def get_default_weight_init() -> Dict[str, Any]:
    """Get default weight initialization config (normal with std=0.05)."""
    from fabricpc.core.initializers import get_default_weight_init as _get_default
    return _get_default()


# ==============================================================================
# STATE INITIALIZATION (Backward Compatible)
# ==============================================================================

def initialize_state_values(
    config: Dict[str, Any],
    key: jax.Array,
    shape: Tuple[int, ...]
) -> jnp.ndarray:
    """
    Initialize state array based on configuration.

    This function provides backward compatibility by delegating to
    the new Initializer registry.

    Args:
        config: State initialization configuration dict
        key: JAX random key
        shape: Shape of state array (batch_size, dim)

    Returns:
        Initialized state array

    Supported types:
        - zeros: All zeros
        - uniform: Uniform distribution in [min, max]
        - normal: Normal distribution with mean and std

    Example:
        >>> config = {"type": "normal", "mean": 0, "std": 0.01}
        >>> z = initialize_state_values(config, key, (32, 256))
    """
    from fabricpc.core.initializers import initialize
    return initialize(key, shape, config)


def parse_state_init_config(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Parse state initialization config.

    DEPRECATED: This function is maintained for backward compatibility.
    New code should use fabricpc.graph.state_initializer directly.

    Args:
        config: State initialization configuration

    Returns:
        Tuple of (init_method, fallback_config)
        - init_method: "zeros", "uniform", "normal", or "feedforward"
        - fallback_config: Config for fallback initialization

    Example:
        >>> config = {"type": "feedforward", "fallback": {"type": "normal", "std": 0.01}}
        >>> method, fallback = parse_state_init_config(config)
        >>> method
        'feedforward'
    """
    if not isinstance(config, dict):
        raise ValueError(f"State init config must be a dict, got {type(config)}")

    init_type = config.get("type", "feedforward").lower()

    if init_type == "zeros":
        return "zeros", {}

    elif init_type == "uniform":
        return "uniform", config

    elif init_type == "normal":
        return "normal", config

    elif init_type == "feedforward":
        fallback = config.get("fallback", {"type": "normal", "mean": 0.0, "std": 0.05})
        return "feedforward", fallback

    elif init_type == "distribution":
        # New format - map to equivalent old format
        default_init = config.get("default_initializer", {"type": "normal"})
        return default_init.get("type", "normal"), default_init

    else:
        raise ValueError(
            f"Unknown state initialization type: '{init_type}'. "
            f"Supported: 'zeros', 'uniform', 'normal', 'feedforward', 'distribution'"
        )


def get_default_state_init() -> Dict[str, Any]:
    """Get default state initialization config."""
    from fabricpc.core.initializers import get_default_state_init as _get_default
    return _get_default()