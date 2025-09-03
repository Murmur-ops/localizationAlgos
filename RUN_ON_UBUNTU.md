# Running on Ubuntu/Linux Systems

## Quick Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DecentralizedLocale
```

2. **Install system dependencies:**
```bash
# Update package list
sudo apt-get update

# Install Python and pip
sudo apt-get install python3 python3-pip python3-venv

# Install MPI (for distributed execution)
sudo apt-get install mpich libmpich-dev

# Install build tools (for compiling dependencies)
sudo apt-get install build-essential
```

3. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

4. **Test installation:**
```bash
python3 test_installation.py
```

## Running Simulations

### Basic MPS Algorithm
```bash
python3 scripts/run_mps.py --config configs/quick_test.yaml
```

### S-Band Precision Simulation
```bash
cd src/simulation/src
python3 run_phase_sync_simulation.py
```

### Algorithm Comparison
```bash
python3 experiments/run_comparison.py
```

### Distributed Execution (MPI)
```bash
# IMPORTANT: Use python3 explicitly with mpirun
mpirun -n 2 python3 scripts/run_distributed.py

# Or for larger networks
mpirun -n 4 python3 scripts/run_distributed.py --config configs/distributed_large.yaml
```

## Common Issues and Fixes

### 1. MPI Execution Error
If you get "mpirun was unable to launch the specified application":
- **Solution:** Use `python3` explicitly: `mpirun -n 2 python3 script.py`

### 2. Import Errors
If modules can't be found:
- **Solution:** Ensure you're in the project root directory
- Activate the virtual environment: `source venv/bin/activate`

### 3. Permission Denied
If scripts aren't executable:
```bash
chmod +x scripts/*.py
chmod +x experiments/*.py
```

### 4. Missing Dependencies
If packages are missing:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. No MPI Support
If MPI isn't installed:
```bash
# Debian/Ubuntu
sudo apt-get install mpich libmpich-dev

# Then reinstall mpi4py
pip install --no-cache-dir mpi4py
```

## Configuration Files

- `configs/quick_test.yaml` - Fast testing (10 sensors)
- `configs/research_comparison.yaml` - Algorithm comparison
- `configs/distributed_large.yaml` - Large-scale distributed
- `configs/high_accuracy.yaml` - Maximum precision
- `configs/sband_precision.yaml` - S-band millimeter accuracy

## Expected Performance

### Quick Test
- Time: < 1 second
- RMSE: ~0.2 units
- Iterations: ~100

### S-Band Precision
- Time: ~5 seconds for 10 trials
- RMSE: **0.14 ± 0.01 mm** (meets S-band requirement of <15mm)
- Success Rate: 100%

### Distributed Large Scale
- Network: 100 sensors, 10 anchors
- Time: ~10-30 seconds (depends on cores)
- Scalability: Near-linear with MPI processes

## Directory Structure
```
DecentralizedLocale/
├── configs/           # YAML configuration files
├── scripts/          # Main execution scripts
├── src/              # Source code
│   ├── core/         # Core algorithms
│   ├── simulation/   # Simulation tools
│   └── emulation/    # Hardware emulation
├── experiments/      # Comparison studies
├── results/          # Output directory
└── data/            # Input data
```

## Python Version Compatibility

Tested with:
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- MPI4PY 3.0+

## For Development

To contribute or modify:

1. Create a new branch:
```bash
git checkout -b feature-name
```

2. Run tests after changes:
```bash
python3 test_installation.py
```

3. Check S-band performance:
```bash
python3 src/simulation/src/run_phase_sync_simulation.py
```

## Support

For issues or questions:
- Check the logs in `results/` directory
- Run `python3 test_installation.py` for diagnostics
- Ensure all dependencies are installed with correct versions