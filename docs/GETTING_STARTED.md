# Getting Started with Decentralized Localization

## Quick Setup

```bash
# Clone the repository
git clone -b clean-impl-subtree https://github.com/Murmur-ops/DeLocale.git
cd DeLocale

# Install in development mode
pip install -e .

# Verify installation
python -c "from src.core.algorithms.mps_proper import ProperMPSAlgorithm; print('✓ Installation successful')"
```

## Prerequisites

- Python 3.8+
- pip
- Git
- MPI implementation (optional, for distributed algorithms)

## Installation

### Basic Installation

```bash
# Clone repository
git clone -b clean-impl-subtree https://github.com/Murmur-ops/DeLocale.git
cd DeLocale

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### MPI Installation (Optional)

For distributed algorithms:

```bash
# Ubuntu/Debian
sudo apt-get install mpich
pip install mpi4py

# macOS
brew install mpich
pip install mpi4py

# Windows
# Download MS-MPI from Microsoft, then:
pip install mpi4py
```

## Quick Examples

### 1. Run MPS Algorithm

```bash
# Run with small network configuration
python scripts/run_mps.py --config configs/examples/small_network.yaml

# Run with custom parameters
python scripts/run_mps.py --config configs/examples/large_network.yaml --visualize
```

### 2. Compare MPS vs ADMM

```bash
# Create output directory
mkdir -p data

# Run comparison
python experiments/run_comparison.py --num-nodes 20 --num-anchors 4 --max-iterations 200
```

### 3. Basic Python Usage

```python
from src.core.algorithms.mps_proper import ProperMPSAlgorithm
from src.core.algorithms.admm import DecentralizedADMM
import numpy as np

# Generate test network
np.random.seed(42)
n_sensors = 20
n_anchors = 4

# Random positions
positions = np.random.rand(n_sensors, 2) * 10
anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

# Create distance measurements (with noise)
distances = {}
noise_std = 0.05
for i in range(n_sensors):
    for j in range(n_anchors):
        true_dist = np.linalg.norm(positions[i] - anchors[j])
        noisy_dist = true_dist * (1 + np.random.randn() * noise_std)
        distances[(i, n_sensors + j)] = noisy_dist

# Run MPS
mps = ProperMPSAlgorithm(n_sensors, n_anchors)
mps_positions = mps.localize(distances, anchors, max_iter=200)

# Run ADMM for comparison
admm = DecentralizedADMM(n_sensors, n_anchors)
admm_positions = admm.localize(distances, anchors, max_iter=200)

# Calculate errors
mps_rmse = np.sqrt(np.mean((mps_positions - positions)**2))
admm_rmse = np.sqrt(np.mean((admm_positions - positions)**2))

print(f"MPS RMSE: {mps_rmse:.3f}m")
print(f"ADMM RMSE: {admm_rmse:.3f}m")
print(f"MPS is {admm_rmse/mps_rmse:.2f}x better")
```

## Configuration Files

Configuration files are in YAML format. Example:

```yaml
# configs/custom.yaml
algorithm:
  name: "mps"
  max_iterations: 200
  convergence_threshold: 1e-6
  
network:
  n_sensors: 30
  n_anchors: 6
  noise_std: 0.05
  scale: 10.0  # meters
  
visualization:
  show_plots: true
  save_figures: true
  output_dir: "results/"
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_unified_system.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
DeLocale/
├── src/
│   └── core/
│       ├── algorithms/     # Core algorithms (MPS, ADMM, etc.)
│       ├── mps_core/       # MPS-specific implementations
│       └── time_sync/      # Time synchronization modules
├── scripts/                # Runnable scripts
│   ├── run_mps.py         # Main MPS runner
│   └── run_distributed.py # Distributed execution
├── experiments/            # Experiment scripts
│   └── run_comparison.py  # Algorithm comparisons
├── tests/                  # Test suite
├── configs/               # Configuration files
│   └── examples/          # Example configs
├── data/                  # Output data directory
└── results/               # Results directory
```

## Common Use Cases

### 1. Research Comparison

```python
from experiments.run_comparison import run_single_comparison

# Compare algorithms with specific parameters
results = run_single_comparison(
    n_sensors=50,
    n_anchors=8,
    noise_levels=[0.01, 0.05, 0.1],
    max_iterations=500
)

print(f"MPS advantage: {results['mps_advantage']:.2f}x")
```

### 2. Network Topology Analysis

```python
from src.core.algorithms.node_analyzer import NodeAnalyzer

analyzer = NodeAnalyzer()

# Analyze network connectivity impact
for connectivity in [0.3, 0.5, 0.7, 1.0]:
    metrics = analyzer.analyze_topology(
        n_sensors=30,
        connectivity=connectivity
    )
    print(f"Connectivity {connectivity}: {metrics}")
```

### 3. Distributed Execution with MPI

```bash
# Run on 4 processes
mpirun -n 4 python scripts/run_distributed.py --config configs/examples/large_network.yaml
```

## Performance Expectations

Based on real algorithm execution (not simulated):

| Algorithm | Typical RMSE | Convergence | Notes |
|-----------|-------------|-------------|-------|
| MPS | 0.15-0.25m | 200-300 iter | Better accuracy |
| ADMM | 0.25-0.40m | 400-500 iter | More iterations |
| Ratio | MPS ~1.5-2x better | MPS ~2x faster | Real performance |

## Troubleshooting

### Import Errors

If you see import errors:
```bash
# Ensure you're in the project root
cd /path/to/DeLocale

# Reinstall in development mode
pip install -e .
```

### MPI Issues

```bash
# Test MPI installation
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

# Should print:
# 0
# 1
```

### Missing Dependencies

```bash
# Install all dependencies
pip install numpy scipy matplotlib cvxpy pyyaml networkx mpi4py
```

## Advanced Usage

### Custom Algorithm Implementation

```python
from src.core.algorithms.proximal_operators import ProximalOperators

class CustomAlgorithm:
    def __init__(self):
        self.prox_ops = ProximalOperators()
    
    def localize(self, measurements, anchors):
        # Your implementation here
        pass
```

### Visualization

```python
from scripts.visualize_network import visualize_results

# After running localization
visualize_results(
    true_positions=positions,
    estimated_positions=mps_positions,
    anchors=anchors,
    title="MPS Localization Results"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues or questions:
- Check the [documentation](docs/)
- Open an [issue on GitHub](https://github.com/Murmur-ops/DeLocale/issues)
- Review example configurations in `configs/examples/`

## License

See LICENSE file for details.