"""
Split-CIFAR10 Columnar Continual Learning Example

Implements the PDF-driven first serious CIFAR protocol:
- 5 tasks of 2 classes each
- shallow shared visual stem before columns
- sparse support over 40 columns (4 shared + adaptive + reserve)
- task-local-head semantics via active-class masking
- no replay, light CIFAR augmentation only
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer, NormalInitializer, initialize

from fabricpc.continual.config import make_cifar10_protocol_config, ExperimentConfig
from fabricpc.continual.nodes import ColumnNode, ComposerNode
from fabricpc.continual.data_cifar import build_split_cifar10_loaders
from fabricpc.continual.trainer import SequentialTrainer
from fabricpc.continual.utils import (
    plot_accuracy_curves,
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    print_summary_table,
    save_summaries_json,
    save_accuracy_matrix,
    create_run_directory,
    save_experiment_config,
)


class SpatialAveragePoolNode(NodeBase):
    """Average-pool an NHWC feature map to a lower spatial resolution."""

    def __init__(
        self,
        shape,
        name,
        kernel_size=(2, 2),
        stride=(2, 2),
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(params, inputs, state: NodeState, node_info: NodeInfo):
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        kernel_size = node_info.node_config.get("kernel_size", (2, 2))
        stride = node_info.node_config.get("stride", kernel_size)
        pooled = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, kernel_size[0], kernel_size[1], 1),
            window_strides=(1, stride[0], stride[1], 1),
            padding="VALID",
        )
        pooled = pooled / float(kernel_size[0] * kernel_size[1])
        activation = node_info.activation
        z_mu = type(activation).forward(pooled, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pooled, z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


class Conv2DNode(NodeBase):
    """Minimal NHWC conv node reused from the CIFAR demos."""

    def __init__(
        self,
        shape,
        name,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
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
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.05)
        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)
        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_param_shape = (
                kernel_size[0],
                kernel_size[1],
                in_channels,
                out_channels,
            )
            weights_dict[edge_key] = initialize(
                keys[i], kernel_param_shape, weight_init
            )
        bias = jnp.zeros((1, 1, 1, out_channels))
        return NodeParams(weights=weights_dict, biases={"b": bias})

    @staticmethod
    def forward(params, inputs, state: NodeState, node_info: NodeInfo):
        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        pre_activation = jnp.zeros((batch_size, *out_shape))
        for edge_key, x in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = jax.lax.conv_general_dilated(
                x,
                kernel,
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            pre_activation = pre_activation + conv_out
        pre_activation = pre_activation + params.biases["b"]
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


def create_network_structure(config: ExperimentConfig):
    """Create a CIFAR10 continual graph with a shallow shared visual stem."""
    pixels = IdentityNode(
        shape=(32, 32, 3), activation=IdentityActivation(), name="pixels"
    )

    stem1 = Conv2DNode(
        shape=(32, 32, 32),
        name="stem1",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )
    stem2 = Conv2DNode(
        shape=(16, 16, 64),
        name="stem2",
        kernel_size=(3, 3),
        stride=(2, 2),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )
    stem3 = Conv2DNode(
        shape=(16, 16, 64),
        name="stem3",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )
    region_pool = SpatialAveragePoolNode(
        shape=(8, 8, 64),
        name="region_pool",
        kernel_size=(2, 2),
        stride=(2, 2),
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
    )

    columns = ColumnNode(
        shape=(config.columns.num_columns, config.columns.memory_dim),
        name="columns",
        num_shells=3,
        shell_sizes=config.shell_demotion_transweave.shell_sizes,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )
    aggregator = ComposerNode(
        shape=(config.columns.aggregator_dim,),
        name="aggregator",
        num_heads=4,
        num_layers=1,
        num_tasks=config.num_tasks,
        gate_temp=0.5,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )
    output = Linear(
        shape=(config.num_output_classes,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
        weight_init=XavierInitializer(),
    )

    return graph(
        nodes=[pixels, stem1, stem2, stem3, region_pool, columns, aggregator, output],
        edges=[
            Edge(source=pixels, target=stem1.slot("in")),
            Edge(source=stem1, target=stem2.slot("in")),
            Edge(source=stem2, target=stem3.slot("in")),
            Edge(source=stem3, target=region_pool.slot("in")),
            Edge(source=region_pool, target=columns.slot("in")),
            Edge(source=columns, target=aggregator.slot("in")),
            Edge(source=aggregator, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split-CIFAR10 Columnar Continual Learning"
    )
    parser.add_argument(
        "--quick-smoke", action="store_true", help="Run a quick smoke test"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/split_cifar10_columnar",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs per task"
    )
    parser.add_argument(
        "--num-columns", type=int, default=None, help="Override total columns"
    )
    parser.add_argument(
        "--memory-dim", type=int, default=None, help="Override column width"
    )
    parser.add_argument(
        "--causal-scale", type=float, default=None, help="Override causal scale"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Split-CIFAR10 Columnar Continual Learning")
    print("=" * 60)

    config = make_cifar10_protocol_config(quick_smoke=args.quick_smoke)
    config.seed = args.seed

    if args.epochs is not None:
        config.training.epochs_per_task = args.epochs
    if args.num_columns is not None:
        config.columns.num_columns = args.num_columns
        config.columns.adaptive_columns = max(
            config.columns.num_columns
            - config.columns.shared_columns
            - config.columns.reserve_columns,
            0,
        )
    if args.memory_dim is not None:
        config.columns.memory_dim = args.memory_dim
    if args.causal_scale is not None:
        config.support.causal_max_effective_scale = args.causal_scale

    print("\nConfiguration:")
    print(f"  Epochs per task: {config.training.epochs_per_task}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Num tasks: {config.num_tasks}")
    print(f"  Task pairs: {config.task_pairs}")
    print(f"  Columns: {config.columns.num_columns}")
    print(f"  Shared columns: {config.columns.shared_columns}")
    print(f"  Adaptive columns: {config.columns.adaptive_columns}")
    print(f"  Reserve columns: {config.columns.reserve_columns}")
    print(f"  Top-k nonshared: {config.columns.topk_nonshared}")
    print(f"  Memory dim: {config.columns.memory_dim}")
    print(f"  Task-local head: {config.training.task_local_head}")
    print(f"  Light augmentation: {config.training.use_light_augmentation}")
    print(f"  Causal scale: {config.support.causal_max_effective_scale}")

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(config.seed)
    init_key, train_key = jax.random.split(master_key)

    print("\nCreating network...")
    structure = create_network_structure(config)
    params = initialize_params(structure, init_key)
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Nodes: {len(structure.nodes)}")
    print(f"  Edges: {len(structure.edges)}")
    print(f"  Parameters: {total_params:,}")

    optimizer = optax.adamw(
        config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    trainer = SequentialTrainer(
        structure=structure,
        config=config,
        params=params,
        optimizer=optimizer,
        rng_key=train_key,
    )

    print("\nLoading Split-CIFAR10 data...")
    tasks = build_split_cifar10_loaders(config, data_root="./data")
    print(f"  Tasks: {len(tasks)}")
    for task in tasks:
        print(
            f"    Task {task.task_id}: classes {task.classes}, "
            f"train batches: {len(task.train_loader)}, test batches: {len(task.test_loader)}"
        )

    run_dir = create_run_directory(
        args.output_dir, "split_cifar10_columnar", config.seed
    )
    print(f"\nOutput directory: {run_dir}")
    save_experiment_config(config, run_dir / "config.json")

    print("\n" + "=" * 60)
    print("Starting Sequential Training")
    print("=" * 60)

    start_time = time.time()
    for task_data in tasks:
        trainer.train_task(task_data, verbose=True)
        checkpoint_path = run_dir / f"checkpoint_task_{task_data.task_id}.npz"
        trainer.save_checkpoint(str(checkpoint_path))
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    summaries = trainer.summaries
    print_summary_table(summaries)
    print(f"\nWall-clock run time: {total_time:.1f}s")
    print(f"Average wall time per task: {total_time / max(len(summaries), 1):.1f}s")

    accuracy_matrix = trainer.accuracy_matrix()
    print("\nAccuracy Matrix:")
    print(accuracy_matrix)
    print(f"\nAverage forgetting: {trainer.get_forgetting_metric():.4f}")

    save_summaries_json(summaries, run_dir / "summaries.json")
    save_accuracy_matrix(accuracy_matrix, run_dir / "accuracy_matrix.csv")

    try:
        print("\nGenerating plots...")
        plot_accuracy_curves(
            summaries, save_path=run_dir / "accuracy_curves.png", show=False
        )
        plot_accuracy_matrix(
            accuracy_matrix, save_path=run_dir / "accuracy_matrix.png", show=False
        )
        plot_forgetting_analysis(
            accuracy_matrix, save_path=run_dir / "forgetting_analysis.png", show=False
        )
        print(f"Plots saved to {run_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print(f"\nResults saved to: {run_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
