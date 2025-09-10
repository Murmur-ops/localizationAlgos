# Setup Guide: Decentralized Localization System

## Quick Start (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/DecentralizedLocale.git
cd DecentralizedLocale/CleanImplementation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Your First Simulation
```bash
cd simulation/src
python run_phase_sync_simulation.py
```

Expected output:
```
CARRIER PHASE SYNCHRONIZATION SIMULATION
Mean RMSE: 0.14 ± 0.01 mm
Meets S-band requirement (<15.0 mm): YES ✓
```

## Complete Installation

### Prerequisites

- Python: 3.8 or higher
- pip: Latest version
- Optional: MPI implementation (for distributed algorithms)
  - macOS: `brew install mpich`
  - Ubuntu: `sudo apt-get install mpich`
  - Windows: Use Microsoft MPI

### Step-by-Step Installation

#### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Core Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Verify Installation
```bash
python -c "import numpy, scipy, matplotlib, cvxpy, yaml, networkx; print('Core packages: OK')"
python -c "from mpi4py import MPI; print('MPI support: OK')"
```

#### 4. Test Components

Test Simulation (Ideal Hardware):
```bash
cd src/simulation
python run_phase_sync_simulation.py
# Expected: RMSE ~0.14mm
```

Test Emulation (Python Timing):
```bash
cd src/emulation
python test_python_timing_limits.py
# Expected: Shows ~41ns timer resolution
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CVXPy Installation Issues
```bash
# If cvxpy fails, try:
pip install --upgrade pip
pip install cvxpy clarabel
```

#### 2. MPI4py Installation Issues
```bash
# Install MPI first:
# macOS:
brew install mpich

# Ubuntu/Debian:
sudo apt-get install mpich

# Then install mpi4py:
pip install mpi4py
```

#### 3. Matplotlib Backend Issues
If plots don't display:
```python
# Add to your script:
import matplotlib
matplotlib.use('Agg')  # For saving only
# or
matplotlib.use('TkAgg')  # For display
```

#### 4. ImportError for Shared Modules
```bash
# Run from CleanImplementation directory:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Running Different Components

### 1. Simulation (Theoretical Performance)
```bash
cd simulation/src

# Run with default config
python run_phase_sync_simulation.py

# Run with custom config
python run_phase_sync_simulation.py --config ../config/custom.yaml
```

### 2. Emulation (Real Timing Constraints)
```bash
cd emulation/src

# Test timing limits
python test_python_timing_limits.py

# Test time synchronization
python -m time_sync.twtt
```

### 3. MPS Algorithm Tests
```bash
# Run distributed MPS
mpirun -n 4 python run_distributed.py

# Run single-node MPS
python run_mps.py
```

## Configuration Files

### Modify Simulation Parameters
Edit `simulation/config/phase_sync_sim.yaml`:
```yaml
network:
  n_sensors: 30  # Increase sensors
  scale_meters: 100.0  # Larger network

carrier_phase:
  phase_noise_milliradians: 0.5  # Better hardware
```

### Modify Emulation Parameters
Edit `emulation/config/time_sync_emulation.yaml`:
```yaml
synchronization:
  algorithm: "twtt"  # or "consensus", "frequency"
  num_exchanges: 20  # More exchanges for better accuracy
```

## Expected Performance

### Simulation (Ideal Hardware)
- Technology: Carrier phase @ 2.4 GHz
- Ranging Accuracy: 0.02mm
- Localization RMSE: 0.1-0.2mm
- Meets S-band: ✓

### Emulation (Python Timing)
- Technology: Computer clock
- Timer Resolution: ~41ns
- Distance Uncertainty: ~12m
- Localization RMSE: 600-1000mm
- Meets S-band: ✗

## System Requirements

### Minimum
- OS: Windows 10, macOS 10.14, Ubuntu 18.04
- Python: 3.8+
- RAM: 4GB
- Disk: 500MB

### Recommended
- OS: macOS 12+, Ubuntu 20.04+
- Python: 3.10+
- **RAM**: 8GB+
- **CPU**: Multi-core for MPI
- **Disk**: 1GB (for results/visualizations)

## Development Setup

### For Contributing
```bash
# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Check style
flake8 .
```

### IDE Setup
- **VS Code**: Install Python extension
- **PyCharm**: Open as Python project
- **Jupyter**: For interactive exploration

## Next Steps

1. **Explore Simulation**: Understand theoretical performance
2. **Test Emulation**: See real timing constraints
3. **Modify Configs**: Try different parameters
4. **Read Documentation**:
   - `simulation/README.md`
   - `emulation/README.md`
   - `PROJECT_STRUCTURE.md`

## Support

For issues or questions:
1. Check documentation in respective directories
2. Review `PROJECT_STRUCTURE.md`
3. See analysis reports (`*_report.md` files)

## Quick Test Script

Save as `test_installation.py`:
```python
#!/usr/bin/env python3
"""Test if installation is working correctly"""

print("Testing installation...")

# Test imports
try:
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import cvxpy as cp
    import yaml
    import networkx as nx
    from mpi4py import MPI
    print("✓ All packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test basic functionality
try:
    # Numpy
    a = np.array([1, 2, 3])
    
    # Scipy
    from scipy.linalg import norm
    n = norm(a)
    
    # CVXPy
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x**2), [x >= 1])
    
    # YAML
    config = yaml.safe_load("test: 123")
    
    # NetworkX
    G = nx.Graph()
    G.add_edge(0, 1)
    
    print("✓ Basic functionality works")
    print("\nInstallation successful! You can now run the simulations.")
    
except Exception as e:
    print(f"✗ Functionality error: {e}")
    exit(1)
```

Run with: `python test_installation.py`