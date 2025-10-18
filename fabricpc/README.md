# FabricPC

**A flexible, performant predictive coding framework**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and learning algorithms

## Quick Start
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ../experiments/mnist/demo_minimal.py

## From Scratch
pwd > ~/fabricpc-code.location
cd
python3 -m venv pc-venv
cd pc-venv
. bin/activate
pip install --upgrade pip
cd $(< ~/fabricpc-code.location)
pip install -r ../requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ../experiments/mnist/demo_minimal.py
