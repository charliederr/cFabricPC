"""
Split-CIFAR-10 HiBaCaML (ColBa) Continual Learning Example

Implements the scaling protocol described in hibacaml_agi26.pdf for CIFAR-10:
- 5 binary tasks (task-incremental)
- Shared convolutional visual stem (3 conv layers + pooling)
- Columnar architecture with N_col=40 and k_total=9
- HiBaCaML-optimized hyperparameters

Usage:
    python examples/split_cifar10_hibacaml.py
    python examples/split_cifar10_hibacaml.py --quick-smoke
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import jax
import optax
import numpy as np

from fabricpc.nodes import Linear, IdentityNode, Conv2DNode, Pool2DNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SoftmaxActivation,
    ReLUActivation,
    IdentityActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.initializers import (
    XavierInitializer,
    NormalInitializer,
)

from fabricpc.continual.config import make_config, ExperimentConfig
from fabricpc.continual.nodes import (
    ColumnNode,
    ComposerNode,
    PartitionedAggregator,
    create_visual_stem,
)
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


def create_hibacaml_cifar10_structure(config: ExperimentConfig):
    """
    FabricPC graph structure for Split-CIFAR-10 HiBaCaML.
    """
    # 1. Input (NHWC)
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # 2. Shared Visual Stem
    # PDF: Conv(3, 32, 3x3) -> Conv(32, 64, 3x3, stride 2) -> Conv(64, 64, 3x3)
    # followed by light patch pooling to 64-96 tokens.
    # pooling 16x16 to 8x8 (patch_pool_size=2) gives 64 tokens.
    stem_nodes, stem_edges, pool_node = create_visual_stem(
        input_node,
        stem_channels=(32, 64, 64),
        patch_pool_size=2,
        activation=ReLUActivation(),
    )

    # 3. Columns
    # PDF: N_col=40, d_m=32. Each column retains R=3 microcolumns {K, L, B}.
    # (Microcolumns are currently handled as a single larger memory_dim for simplicity)
    num_columns = config.columns.num_columns
    memory_dim = config.columns.memory_dim

    columns = ColumnNode(
        shape=(num_columns, memory_dim),
        name="columns",
        num_shells=3,
        shell_sizes=config.shell_demotion_transweave.shell_sizes,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
    )

    # 4. Aggregator (ComposerNode or PartitionedAggregator)
    aggregator_dim = config.columns.aggregator_dim
    use_partitioned = config.columns.use_partitioned_aggregator
    use_attention = config.columns.use_attention_aggregator

    if use_partitioned:
        shared_dim = config.columns.partitioned_shared_dim
        task_dim = config.columns.partitioned_task_dim
        aggregator = PartitionedAggregator(
            shape=(shared_dim + task_dim,),
            name="aggregator",
            num_tasks=config.num_tasks,
            shared_columns=config.columns.shared_columns,
            topk_nonshared=config.columns.topk_nonshared,
            shared_dim=shared_dim,
            task_dim=task_dim,
            memory_dim=memory_dim,
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            weight_init=NormalInitializer(mean=0.0, std=0.02),
        )
    elif use_attention:
        aggregator = ComposerNode(
            shape=(aggregator_dim,),
            name="aggregator",
            num_heads=config.columns.attention_num_heads,
            num_layers=config.columns.attention_num_layers,
            num_tasks=config.num_tasks,
            gate_temp=0.5,
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            weight_init=NormalInitializer(mean=0.0, std=0.02),
        )
    else:
        aggregator = Linear(
            shape=(aggregator_dim,),
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            name="aggregator",
            weight_init=XavierInitializer(),
            flatten_input=True,
        )

    # 5. Output
    output = Linear(
        shape=(config.num_output_classes,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
        weight_init=XavierInitializer(),
    )

    # Combine all
    nodes = [input_node] + stem_nodes + [columns, aggregator, output]
    edges = stem_edges + [
        Edge(source=pool_node, target=columns.slot("in")),
        Edge(source=columns, target=aggregator.slot("in")),
        Edge(source=aggregator, target=output.slot("in")),
    ]

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output),
        # Using NormClip inference for stability in deep vision networks
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=20, max_norm=1.0),
    )

    return structure


def _configure_hibacaml_cifar10(
    config: ExperimentConfig,
    quick_smoke: bool,
) -> ExperimentConfig:
    """Apply HIBACAML scaling protocol for Split-CIFAR-10."""
    config.num_tasks = 5
    config.num_output_classes = 10
    config.task_pairs = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))

    # PDF: N_col = 40, N_shared = 4, N_adaptive = 30, N_reserve = 6
    # Active support: k_nonshared = 5 (total 9)
    config.columns.num_columns = 40
    config.columns.shared_columns = 4
    config.columns.topk_nonshared = 5
    config.columns.memory_dim = 32  # d_m = 32

    # Shell sizes: |S1|=10, |S2|=20, |S3|=30
    config.shell_demotion_transweave.shell_sizes = (10, 20, 30)

    # Aggregator settings
    config.columns.aggregator_dim = 256
    config.columns.partitioned_shared_dim = 64
    config.columns.partitioned_task_dim = 64

    # Support settings
    config.support.topk_nonshared = 5
    config.support.causal_max_effective_scale = 0.5

    # Data format
    config.training.tensor_format = "NHWC"

    if quick_smoke:
        config.training.fast_dev_max_train_batches = 2
        config.training.fast_dev_max_test_batches = 2
        config.training.epochs_per_task = 1
        # Scale down for speed
        config.columns.num_columns = 12
        config.columns.shared_columns = 2
        config.columns.topk_nonshared = 2
        config.columns.memory_dim = 8
        config.shell_demotion_transweave.shell_sizes = (2, 4, 6)

    return config


def main():
    parser = argparse.ArgumentParser(description="Split-CIFAR-10 HiBaCaML Example")
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run quick smoke test",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="pc",
        choices=["pc", "backprop"],
        help="Training mode (default: pc)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Use PartitionedAggregator (true architectural isolation)",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use ComposerNode (attention aggregator)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/split_cifar10_hibacaml",
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Split-CIFAR-10 HiBaCaML (ColBa) Protocol")
    print("=" * 60)

    config = make_config(quick_smoke=args.quick_smoke)
    config.seed = args.seed
    config.training.training_mode = args.training_mode

    config = _configure_hibacaml_cifar10(config, args.quick_smoke)

    if args.partitioned:
        config.columns.use_partitioned_aggregator = True
        config.columns.use_attention_aggregator = False
    elif args.attention:
        config.columns.use_attention_aggregator = True
        config.columns.use_partitioned_aggregator = False
        config.training.learning_rate = 0.0003  # Transformer-like LR

    print("\nHiBaCaML Configuration:")
    print(
        f"  Columns: {config.columns.num_columns} (shared: {config.columns.shared_columns})"
    )
    print(f"  Active support: {config.columns.topk_nonshared} (+ shared)")
    print(f"  Memory dim (d_m): {config.columns.memory_dim}")
    print(f"  Shell sizes: {config.shell_demotion_transweave.shell_sizes}")
    print(f"  Partitioned: {config.columns.use_partitioned_aggregator}")
    print(f"  Attention: {config.columns.use_attention_aggregator}")

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(config.seed)
    init_key, train_key = jax.random.split(master_key)

    print("\nBuilding HiBaCaML graph with visual stem...")
    structure = create_hibacaml_cifar10_structure(config)
    params = initialize_params(structure, init_key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Nodes: {len(structure.nodes)}")
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

    print("\nLoading Split-CIFAR-10 data...")
    tasks = build_split_cifar10_loaders(config, data_root="./data")

    run_dir = create_run_directory(args.output_dir, "cifar10_hibacaml", config.seed)
    save_experiment_config(config, run_dir / "config.json")

    print("\nStarting Sequential Training...")
    start_time = time.time()
    for task_data in tasks:
        trainer.train_task(task_data, verbose=True)
        trainer.save_checkpoint(
            str(run_dir / f"checkpoint_task_{task_data.task_id}.npz")
        )
    total_time = time.time() - start_time

    print(f"\nTraining Complete in {total_time:.1f}s")
    print_summary_table(trainer.summaries)

    acc_matrix = trainer.accuracy_matrix()
    save_summaries_json(trainer.summaries, run_dir / "summaries.json")
    save_accuracy_matrix(acc_matrix, run_dir / "accuracy_matrix.csv")

    print(f"\nFinal average accuracy: {np.mean(acc_matrix[-1]):.4f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
