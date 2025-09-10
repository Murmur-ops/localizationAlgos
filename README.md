# Decentralized Sensor Network Localization

A high-performance implementation of the Matrix-Parametrized Proximal Splitting (MPS) algorithm for sensor network localization, based on the paper arXiv:2503.13403v1.

## Features

- **MPS Algorithm**: State-of-the-art semidefinite relaxation approach for sensor localization
- **Distributed Execution**: MPI support for large-scale networks (100+ sensors)
- **YAML Configuration**: Flexible configuration system with inheritance and overrides  
- **Multiple Solvers**: ADMM inner solver with warm-starting capabilities
- **Carrier Phase Support**: Millimeter-level accuracy with phase measurements

## Performance

- Relative error: 0.14-0.18 (approaching paper's 0.05-0.10)
- Scales to 200+ sensors with MPI distribution
- Convergence in 200-500 iterations typically
- RMSE: ~0.09 meters in unit square for 30-sensor networks
- Carrier phase: 0.14mm RMSE achieved in S-band testing

## Installation

### Requirements
- Python 3.8+
- NumPy, SciPy
- MPI (optional, for distributed execution)

### Setup
```bash
# Clone repository
git clone https://github.com/Murmur-ops/DelocaleClean.git
cd DelocaleClean

# Install dependencies
pip install -r requirements.txt

# Run setup script
./setup.sh
```

## Quick Start

### Basic Usage
```python
from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import create_network_data

# Load configuration
loader = ConfigLoader()
config = loader.load_config("configs/default.yaml")

# Generate network
network = create_network_data(
    n_sensors=30,
    n_anchors=6,
    dimension=2,
    communication_range=0.3,
    measurement_noise=0.05
)

# Run algorithm (see examples for full implementation)
```

### Command Line
```bash
# Single process execution
python scripts/run_mps_mpi.py --config configs/default.yaml

# Distributed execution with MPI
mpirun -n 4 python scripts/run_mps_mpi.py --config configs/mpi/mpi_medium.yaml

# With parameter overrides
python scripts/run_mps_mpi.py --config configs/default.yaml \
  --override network.n_sensors=50 algorithm.max_iterations=300
```

## Project Structure

```
DelocaleClean/
├── src/
│   └── core/
│       └── mps_core/         # Core MPS algorithm
│           ├── algorithm_sdp.py
│           ├── mps_distributed.py
│           ├── config_loader.py
│           └── ...
├── configs/                  # YAML configurations
├── scripts/                  # Executable scripts
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Configuration

The system uses YAML configuration files with comprehensive parameter documentation. Example configurations:

- `configs/default.yaml` - Standard settings
- `configs/high_accuracy.yaml` - Maximum precision
- `configs/fast_convergence.yaml` - Real-time applications
- `configs/noisy_measurements.yaml` - Robust to noise
- `configs/distributed_large.yaml` - Large-scale MPI

See documentation for detailed parameter reference.

## Documentation

Detailed documentation is available in the `docs/` directory:
- Algorithm details and mathematical formulation
- Implementation architecture
- Configuration guide
- MPI distributed execution
- API reference

## Testing

```bash
# Run test suite
python -m pytest tests/

# Test specific configuration
python tests/test_yaml_config.py

# Test MPI functionality
python tests/test_mpi_simple.py
```

## Author

Max Burnett

## References

Based on: "Matrix-Parametrized Proximal Splitting for Sensor Network Localization" (arXiv:2503.13403v1)

## License

MIT License - See LICENSE file for details.