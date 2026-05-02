# cFabricPC Project Status & Architecture

This file tracks the current state of the `cFabricPC` repository, its core architecture, and ongoing development tasks.

## Current Version: 0.3.0 (2026-04-17)

### Core Architecture
cFabricPC is a JAX-based predictive coding library organized around three main pillars:
1. **Nodes**: Independent computational units (latents, projections, activations).
2. **Wires (Edges)**: Connections between nodes defining the graph topology.
3. **Updates**: Local Hebbian learning and iterative inference algorithms.

### Key Features
- **muPC Scaling**: Support for arbitrary DAG topologies with automatic variance scaling.
- **Composable Nodes**: Includes `Linear`, `TransformerBlock`, `StorkeyHopfield`, `IdentityNode`, `SkipConnection`, and `LinearResidual`.
- **Continual Learning**: A comprehensive `fabricpc.continual` package for task-sequential training (Split-MNIST, Split-CIFAR), including replay, support selection, causal guidance, and TransWeave.
- **JAX-Native**: High-performance inference and learning utilizing JAX's autodiff and GPU acceleration.
- **Experiment Framework**: Statistical A/B testing and experiment tracking integration (Aim).

### Repository Status
- **Development Branch**: The current branch is actively being used for development.
- **Core Engine**: Stable and verified for deep networks (100+ layers) using muPC.
- **Continual Learning**: Feature-rich, implementing V20.2b-style research ideas (exact audits, causal guidance, replay-backed support).
- **Tests**: Extensive test suite in `tests/` covering core mechanics, nodes, and continual features.

## Ongoing Work & Goals
- [x] Integrate/Verify Hopfield nodes in continual learning pipelines. (Verified in cFabricPC core)
- [x] Implement HiBaCaML scaling protocol for Split-CIFAR-10.
    - [x] Add `Conv2DNode` and `Pool2DNode` to core.
    - [x] Generalize CIFAR loaders for Split-CIFAR-10.
    - [x] Implement convolutional visual stem.
    - [x] Create benchmark example `split_cifar10_hibacaml.py`.
- [ ] Improve JAX-native performance of the continual learning layer (currently heavily Python/NumPy dependent).
- [ ] Standardize documentation across all `fabricpc.continual` modules.

## Architectural Notes
- **Visual Stem**: Shared 3-layer conv front-end ($3 \rightarrow 32 \rightarrow 64 \rightarrow 64$) with pooling to reduce spatial redundancy for CIFAR-10.
- **Columnar Architecture**: N_col=40 for CIFAR-10, supporting high-capacity memory with sparse selection.
- **Training Stability**: Using `InferenceSGDNormClip` and `opt_state` preservation across tasks for deep vision networks.
- **Graph Construction**: Uses an object-oriented builder API (`fabricpc.builder`).

---
*Last updated: 2026-05-01*
