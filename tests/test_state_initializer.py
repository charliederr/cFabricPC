#!/usr/bin/env python3
"""
Test suite for the State Initializer system.

Tests:
- DistributionStateInit with graph-level config
- DistributionStateInit with node-level override
- FeedforwardStateInit requires params
- FeedforwardStateInit topological propagation
- Clamp handling in both strategies
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.graph.graph_net import create_pc_graph
from fabricpc.graph.state_initializer import (
    StateInitBase,
    register_state_init,
    get_state_init_class,
    list_state_init_types,
    unregister_state_init,
    initialize_graph_state,
    StateInitRegistrationError,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_graph_config():
    """Simple 3-layer graph config for testing."""
    return {
        "node_list": [
            {"name": "input", "shape": (784,), "type": "linear"},
            {"name": "hidden", "shape": (128,), "type": "linear", "activation": {"type": "relu"}},
            {"name": "output", "shape": (10,), "type": "linear"},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }


class TestStateInitRegistry:
    """Test suite for state initializer registry operations."""

    def test_list_state_init_types(self):
        """Test listing registered state init types."""
        types = list_state_init_types()

        # Should include built-in types
        assert "distribution" in types
        assert "feedforward" in types

    def test_get_state_init_class(self):
        """Test getting state init class by type."""
        dist_class = get_state_init_class("distribution")

        assert dist_class is not None
        assert hasattr(dist_class, "initialize_state")
        assert hasattr(dist_class, "CONFIG_SCHEMA")

    def test_get_state_init_class_case_insensitive(self):
        """Test that type lookup is case-insensitive."""
        assert get_state_init_class("Distribution") == get_state_init_class("distribution")
        assert get_state_init_class("FEEDFORWARD") == get_state_init_class("feedforward")

    def test_get_unknown_type_raises(self):
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            get_state_init_class("nonexistent_state_init")


class TestDistributionStateInit:
    """Test suite for DistributionStateInit."""

    def test_distribution_init_graph_level_config(self, simple_graph_config, rng_key):
        """Test distribution init with graph-level default initializer."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={
                "type": "distribution",
                "default_initializer": {"type": "normal", "std": 0.1}
            }
        )

        # Verify state structure
        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes

        # Verify shapes
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        # Hidden node should be initialized with normal distribution
        # (not clamped, so should have non-zero values with std ~0.1)
        hidden_std = jnp.std(state.nodes["hidden"].z_latent)
        assert hidden_std > 0.05 and hidden_std < 0.2

    def test_distribution_init_with_zeros(self, simple_graph_config, rng_key):
        """Test distribution init with zeros initializer."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={
                "type": "distribution",
                "default_initializer": {"type": "zeros"}
            }
        )

        # Hidden should be all zeros
        assert jnp.all(state.nodes["hidden"].z_latent == 0.0)

    def test_distribution_init_node_level_override(self, rng_key):
        """Test distribution init with node-level latent_init override."""
        config = {
            "node_list": [
                {"name": "input", "shape": (32,), "type": "linear"},
                {
                    "name": "hidden",
                    "shape": (16,),
                    "type": "linear",
                    "activation": {"type": "relu"},
                    "latent_init": {"type": "uniform", "min": -1.0, "max": 1.0}
                },
                {"name": "output", "shape": (8,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "hidden", "slot": "in"},
                {"source_name": "hidden", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        params, structure = create_pc_graph(config, rng_key)

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 32))
        y = jax.random.normal(rng_key, (batch_size, 8))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={
                "type": "distribution",
                "default_initializer": {"type": "zeros"}  # Default to zeros
            }
        )

        # Hidden should be uniform(-1, 1) due to node-level override
        assert jnp.all(state.nodes["hidden"].z_latent >= -1.0)
        assert jnp.all(state.nodes["hidden"].z_latent <= 1.0)
        # Should not be all zeros
        assert not jnp.all(state.nodes["hidden"].z_latent == 0.0)


class TestFeedforwardStateInit:
    """Test suite for FeedforwardStateInit."""

    def test_feedforward_init_requires_params(self, simple_graph_config, rng_key):
        """Test that feedforward init raises error without params."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        with pytest.raises(ValueError, match="requires params"):
            initialize_graph_state(
                structure, batch_size, rng_key, clamps,
                state_init_config={"type": "feedforward"},
                params=None  # No params provided
            )

    def test_feedforward_init_with_params(self, simple_graph_config, rng_key):
        """Test feedforward init propagates through network."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )

        # Verify state structure
        assert state.batch_size == batch_size
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        # Hidden should have z_latent = z_mu (feedforward propagation)
        # So z_latent should not be zero for a feedforward init
        assert not jnp.allclose(state.nodes["hidden"].z_latent, 0.0)

    def test_feedforward_init_topological_order(self, rng_key):
        """Test feedforward init processes nodes in topological order."""
        # Create a deeper network to test ordering
        config = {
            "node_list": [
                {"name": "input", "shape": (32,), "type": "linear"},
                {"name": "h1", "shape": (16,), "type": "linear", "activation": {"type": "relu"}},
                {"name": "h2", "shape": (16,), "type": "linear", "activation": {"type": "relu"}},
                {"name": "h3", "shape": (8,), "type": "linear", "activation": {"type": "relu"}},
                {"name": "output", "shape": (4,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "h1", "slot": "in"},
                {"source_name": "h1", "target_name": "h2", "slot": "in"},
                {"source_name": "h2", "target_name": "h3", "slot": "in"},
                {"source_name": "h3", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        params, structure = create_pc_graph(config, rng_key)

        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 32))
        y = jax.random.normal(rng_key, (batch_size, 4))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )

        # All intermediate nodes should have non-trivial values
        for name in ["h1", "h2", "h3"]:
            assert not jnp.allclose(state.nodes[name].z_latent, 0.0), \
                f"Node {name} should have non-zero z_latent after feedforward init"

    def test_feedforward_init_with_custom_fallback(self, simple_graph_config, rng_key):
        """Test feedforward init with custom fallback initializer."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={
                "type": "feedforward",
                "fallback": {"type": "zeros"}  # Use zeros fallback
            },
            params=params
        )

        # Should still work with zeros fallback
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)


class TestClampHandling:
    """Test clamp handling in state initialization."""

    def test_distribution_init_respects_clamps(self, simple_graph_config, rng_key):
        """Test that distribution init respects clamped values."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 5.0  # Specific clamped value
        y = jnp.ones((batch_size, 10)) * -3.0  # Specific clamped value
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "distribution"}
        )

        # Clamped nodes should have exact clamped values
        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)

    def test_feedforward_init_respects_clamps(self, simple_graph_config, rng_key):
        """Test that feedforward init respects clamped values."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 2.0
        y = jnp.ones((batch_size, 10)) * -1.0
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )

        # Clamped nodes should have exact clamped values
        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)

    def test_partial_clamps(self, simple_graph_config, rng_key):
        """Test initialization with only some nodes clamped."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 3.0
        clamps = {"input": x}  # Only input clamped

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )

        # Input should be clamped
        assert jnp.allclose(state.nodes["input"].z_latent, x)

        # Other nodes should be initialized but not clamped
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_initialize_graph_state_function(self, simple_graph_config, rng_key):
        """Test that initialize_graph_state convenience function works correctly."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        # Use the convenience function
        state = initialize_graph_state(
            structure, batch_size, rng_key,
            clamps=clamps,
            params=params
        )

        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes


class TestCustomStateInit:
    """Test custom state initializer registration."""

    def test_register_custom_state_init(self, simple_graph_config, rng_key):
        """Test registering and using a custom state initializer."""
        @register_state_init("test_custom_state")
        class TestCustomStateInit(StateInitBase):
            CONFIG_SCHEMA = {
                "fill_value": {"type": (int, float), "default": 99.0}
            }

            @staticmethod
            def initialize_state(structure, batch_size, rng_key, clamps, config, params=None):
                from fabricpc.core.types import GraphState, NodeState

                fill_value = config.get("fill_value", 99.0)
                node_state_dict = {}

                for node_name, node_info in structure.nodes.items():
                    shape = (batch_size, *node_info.shape)

                    if node_name in clamps:
                        z_latent = clamps[node_name]
                    else:
                        z_latent = jnp.full(shape, fill_value)

                    node_state_dict[node_name] = NodeState(
                        z_latent=z_latent,
                        z_mu=jnp.zeros(shape),
                        error=jnp.zeros(shape),
                        energy=jnp.zeros((batch_size,)),
                        pre_activation=jnp.zeros(shape),
                        latent_grad=jnp.zeros(shape),
                        substructure={},
                    )

                return GraphState(nodes=node_state_dict, batch_size=batch_size)

        try:
            # Verify registration
            assert "test_custom_state" in list_state_init_types()

            params, structure = create_pc_graph(simple_graph_config, rng_key)

            batch_size = 2
            x = jax.random.normal(rng_key, (batch_size, 784))
            y = jax.random.normal(rng_key, (batch_size, 10))
            clamps = {"input": x, "output": y}

            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps,
                state_init_config={"type": "test_custom_state", "fill_value": 42.0}
            )

            # Hidden should be filled with 42.0
            assert jnp.all(state.nodes["hidden"].z_latent == 42.0)

            # Clamped nodes should have clamped values
            assert jnp.allclose(state.nodes["input"].z_latent, x)
            assert jnp.allclose(state.nodes["output"].z_latent, y)
        finally:
            unregister_state_init("test_custom_state")


class TestStateInitDeterminism:
    """Test that state initialization is deterministic."""

    def test_distribution_init_deterministic(self, simple_graph_config, rng_key):
        """Test distribution init is deterministic with same key."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        clamps = {}

        state1 = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "distribution"}
        )
        state2 = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "distribution"}
        )

        # Should produce identical results
        assert jnp.allclose(
            state1.nodes["hidden"].z_latent,
            state2.nodes["hidden"].z_latent
        )

    def test_feedforward_init_deterministic(self, simple_graph_config, rng_key):
        """Test feedforward init is deterministic with same key."""
        params, structure = create_pc_graph(simple_graph_config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state1 = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )
        state2 = initialize_graph_state(
            structure, batch_size, rng_key, clamps,
            state_init_config={"type": "feedforward"},
            params=params
        )

        # Should produce identical results
        assert jnp.allclose(
            state1.nodes["hidden"].z_latent,
            state2.nodes["hidden"].z_latent
        )
