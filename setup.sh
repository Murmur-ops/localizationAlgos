#!/bin/bash
# Setup script for DecentralizedLocale on Linux/Ubuntu systems

echo "========================================"
echo "DecentralizedLocale Setup Script"
echo "========================================"

# Check Python version
echo -n "Checking Python version... "
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
# Use awk for version comparison (more portable than bc)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "OK (Python $python_version)"
else
    echo "ERROR: Python 3.8+ required (found $python_version)"
    exit 1
fi

# Check for MPI
echo -n "Checking for MPI installation... "
if command -v mpirun &> /dev/null; then
    echo "OK ($(mpirun --version 2>&1 | head -1))"
else
    echo "NOT FOUND"
    echo ""
    echo "To install MPI on Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install mpich libmpich-dev"
    echo ""
    echo "To install MPI on macOS:"
    echo "  brew install mpich"
    echo ""
    read -p "Continue without MPI? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p results/figures
mkdir -p results/data
mkdir -p data
mkdir -p logs
mkdir -p visualizations

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py
chmod +x experiments/*.py
chmod +x src/simulation/src/*.py

# Run verification
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"

python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')

try:
    import scipy
    print(f'✓ SciPy {scipy.__version__}')
except ImportError as e:
    print(f'✗ SciPy: {e}')

try:
    import matplotlib
    print(f'✓ Matplotlib {matplotlib.__version__}')
except ImportError as e:
    print(f'✗ Matplotlib: {e}')

try:
    import yaml
    print(f'✓ PyYAML')
except ImportError as e:
    print(f'✗ PyYAML: {e}')

try:
    import mpi4py
    print(f'✓ mpi4py {mpi4py.__version__}')
except ImportError as e:
    print(f'✗ mpi4py: {e}')

try:
    import networkx as nx
    print(f'✓ NetworkX {nx.__version__}')
except ImportError as e:
    print(f'✗ NetworkX: {e}')

try:
    import cvxpy as cp
    print(f'✓ CVXPY {cp.__version__}')
except ImportError as e:
    print(f'✗ CVXPY: {e}')
"

# Test imports of main modules
echo ""
echo "Testing module imports..."
python3 -c "
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

try:
    from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig
    print('✓ MPS Core modules')
except ImportError as e:
    print(f'✗ MPS Core: {e}')

try:
    from src.core.algorithms.mps_proper import ProperMPSAlgorithm
    print('✓ Algorithm modules')
except ImportError as e:
    print(f'✗ Algorithms: {e}')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Quick test commands:"
echo "  python scripts/run_mps.py --config configs/quick_test.yaml"
echo "  python experiments/run_comparison.py"
echo "  python src/simulation/src/run_phase_sync_simulation.py"
echo ""
echo "For distributed execution (requires MPI):"
echo "  mpirun -n 2 python scripts/run_distributed.py"
echo ""