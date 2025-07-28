# Decentralized Sensor Network Localization

Implementation of the paper "Decentralized Sensor Network Localization using Matrix-Parametrized Proximal Splittings" by Peter Barkley and Robert L. Bassett (2025).

## Overview

This project implements a novel decentralized algorithm for sensor network localization (SNL) that uses only noisy pairwise distance measurements between sensors. The algorithm leverages matrix-parametrized proximal splitting methods and the Sinkhorn-Knopp algorithm for efficient distributed computation.

### Key Features

- **Decentralized computation**: Each sensor runs on its own MPI process and only communicates with neighbors
- **Matrix-parametrized proximal splitting (MPS)**: Novel optimization algorithm that outperforms ADMM
- **Sinkhorn-Knopp parameter selection**: Efficient, decentralized matrix parameter computation
- **OARS integration**: Optional support for advanced matrix parameter generation methods
- **Early termination**: Automatic detection of optimal solution before full convergence
- **Comprehensive experiments**: Reproduce all results from the paper

### Algorithm Overview

The SNL problem estimates sensor positions given:
- A small set of anchor nodes with known positions
- Noisy distance measurements between nearby sensors
- Communication constraints (sensors can only talk to neighbors)

The MPS algorithm solves a semidefinite programming (SDP) relaxation of this problem using:
1. **2-Block design**: Splits computation into two parallel blocks
2. **Proximal operators**: Efficiently handles non-smooth objectives and constraints
3. **Distributed Sinkhorn-Knopp**: Computes optimal matrix parameters without centralization

## Installation

### Prerequisites

- Python 3.8 or higher
- MPI implementation (OpenMPI or MPICH)
- C compiler (for mpi4py)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/DecentralizedLocale.git
cd DecentralizedLocale

# Run setup script
./setup.sh

# Activate environment
source activate_env.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install MPI (macOS)
brew install open-mpi

# Install MPI (Ubuntu)
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# Test installation
python test_installation.py
```

## Usage

### Basic Usage

Run a comparison between MPS and ADMM:

```bash
# Run with 30 sensors (requires 30 MPI processes)
./run_snl.sh

# Run with fewer processes (sensors will be distributed)
./run_snl.sh -n 10

# Run quick test with 20 experiments
./run_snl.sh -e 20
```

### Experiment Types

```bash
# Algorithm comparison (default)
./run_snl.sh -t comparison -e 50

# Parameter sensitivity study
./run_snl.sh -t parameter

# Early termination analysis (300 experiments)
./run_snl.sh -t early -n 30

# Single experiment with detailed output
./run_snl.sh -t single -v
```

### Command Line Options

```
Options:
    -n, --processes NUM      Number of MPI processes (default: 30)
    -e, --experiments NUM    Number of experiments to run (default: 50)
    -t, --type TYPE         Experiment type: comparison, parameter, early, single
    -o, --output DIR        Output directory (default: results)
    -v, --verbose           Verbose output
    -h, --help              Show help message
```

### Python API

```python
from snl_main import SNLProblem, DistributedSNL

# Configure problem
problem = SNLProblem(
    n_sensors=30,
    n_anchors=6,
    communication_range=0.7,
    noise_factor=0.05,
    seed=42
)

# Initialize solver
snl = DistributedSNL(problem)
snl.generate_network()

# Compare algorithms
results = snl.compare_algorithms()
```

### Using OARS Integration

```python
from snl_main_oars import EnhancedDistributedSNL

# Create enhanced solver with OARS support
snl = EnhancedDistributedSNL(problem, use_oars=True)
snl.generate_network()

# Compare different matrix generation methods
results = snl.compare_algorithms_with_oars(
    matrix_methods=["Sinkhorn-Knopp", "MinSLEM", "MaxConnectivity", "MinResist"]
)

# Run with specific OARS method
mps_results = snl.matrix_parametrized_splitting_oars("MinSLEM")
```

## Expected Results

Based on the paper, you should observe:

| Metric | MPS | ADMM | Ratio |
|--------|-----|------|-------|
| Mean Relative Error | ~0.08 | ~0.16 | ~2.0x |
| Convergence (iterations) | <200 | >400 | ~2.0x |
| Early Termination Success | 64% | N/A | - |

### Convergence Comparison

- MPS converges to the relaxation solution error level in <200 iterations
- ADMM requires >400 iterations for similar accuracy
- MPS maintains approximately half the error of ADMM throughout iterations

### Early Termination

- Early termination (based on objective value) improves accuracy in ~64% of cases
- Average improvement when better: ~5-10%
- Reduces computational cost significantly

## Project Structure

```
DecentralizedLocale/
├── snl_main.py              # Main implementation with MPS and ADMM algorithms
├── proximal_operators.py    # Proximal operator implementations
├── run_experiments.py       # Experiment runner and analysis
├── visualize_results.py     # Visualization tools
├── setup.sh                 # Environment setup script
├── run_snl.sh              # Experiment execution script
├── requirements.txt         # Python dependencies
├── results/                # Experiment results (created by scripts)
│   └── logs/              # Execution logs
└── figures/                # Generated visualizations
```

## Troubleshooting

### MPI Issues

**Error: "MPI not found"**
```bash
# macOS
brew install open-mpi

# Ubuntu
sudo apt-get install openmpi-bin libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
```

**Error: "mpi4py installation failed"**
```bash
# Ensure MPI compiler wrapper is available
which mpicc

# Install with explicit compiler
MPICC=mpicc pip install mpi4py --no-cache-dir
```

### Memory Issues

For large sensor networks (>100 sensors), you may need to:
- Reduce the number of experiments
- Run on a cluster with more memory
- Use fewer MPI processes (sensors will be distributed)

### Convergence Issues

If algorithms don't converge:
- Check that the communication graph is connected
- Verify noise levels aren't too high (>20%)
- Ensure sufficient anchors (at least 3 for 2D)

## Algorithm Details

### Proximal Operators

The implementation uses two types of proximal operators:

1. **Prox of g_i**: Handles the distance measurement objective using ADMM
2. **Prox of indicator**: Projects onto the positive semidefinite cone

### Matrix Parameters

Matrix parameters can be generated using several methods:

1. **Sinkhorn-Knopp** (default): Efficient distributed algorithm
2. **OARS Methods** (optional):
   - **MinSLEM**: Minimize second-largest eigenvalue magnitude
   - **MaxConnectivity**: Maximize algebraic connectivity (Fiedler value)
   - **MinResist**: Minimize total effective resistance
   - **MinSpectralDifference**: Minimize spectral difference

All methods ensure the matrix parameters satisfy:
- Z ⪰ W (Z dominates W in PSD order)
- null(W) = span(1) (connected graph property)
- diag(Z) = 2·1 (scaling condition)
- 1ᵀZ1 = 0 (sum-to-zero property)

### Communication Pattern

Each sensor only communicates with:
- Neighbors within communication range
- For matrix parameter computation
- For dual variable updates

## Citation

If you use this code in your research, please cite:

```bibtex
@article{barkley2025decentralized,
  title={Decentralized Sensor Network Localization using Matrix-Parametrized Proximal Splittings},
  author={Barkley, Peter and Bassett, Robert L},
  journal={arXiv preprint arXiv:2503.13403},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Naval Postgraduate School Operations Research Department
- Office of Naval Research (awards N0001425WX00069 and N0001425GI01512)

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact the authors: peter.barkley@nps.edu, robert.bassett@nps.edu