#!/bin/bash
# Setup script for Decentralized Sensor Network Localization
# This script creates the environment and installs dependencies

set -e  # Exit on error

echo "Setting up Decentralized Sensor Network Localization environment..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install mpi4py>=3.1.0
pip install pandas>=1.3.0
pip install seaborn>=0.11.0
pip install networkx>=2.6.0
pip install tqdm>=4.62.0
pip install pytest>=6.2.0

# Optional: Install MOSEK (for comparison with SDP solutions)
echo "Note: MOSEK is optional. To install:"
echo "  1. Get academic license from https://www.mosek.com/products/academic-licenses/"
echo "  2. Run: pip install mosek"

# Optional: Install CVXPY for SDP formulations
read -p "Install CVXPY for SDP formulations? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing CVXPY..."
    pip install cvxpy>=1.2.0
fi

# Check MPI installation
echo "Checking MPI installation..."
if command -v mpirun &> /dev/null; then
    mpi_version=$(mpirun --version 2>&1 | head -n 1)
    echo "MPI found: $mpi_version ✓"
else
    echo "Warning: MPI not found. Install OpenMPI or MPICH:"
    echo "  macOS: brew install open-mpi"
    echo "  Ubuntu: sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev"
    echo "  CentOS: sudo yum install openmpi openmpi-devel"
fi

# Test MPI4PY
echo "Testing mpi4py installation..."
if python -c "from mpi4py import MPI; print('mpi4py version:', MPI.Get_version())" 2>/dev/null; then
    echo "mpi4py installation successful ✓"
else
    echo "Error: mpi4py installation failed"
    echo "Try: MPICC=mpicc pip install mpi4py --no-cache-dir"
    exit 1
fi

# Clone and install OARS repository
echo "Installing OARS (Optimal Algorithms for Resolvent Splitting)..."
if [ ! -d "oars" ]; then
    echo "Cloning OARS repository..."
    git clone https://github.com/peterbarkley/oars.git
    cd oars
    pip install -e .
    cd ..
    echo "OARS installed successfully ✓"
else
    echo "OARS directory already exists"
    cd oars
    git pull
    pip install -e . --upgrade
    cd ..
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p results
mkdir -p results/logs
mkdir -p figures
mkdir -p data

# Create a simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test installation of required packages"""

import sys

def test_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name}")
        return False

print("Testing package imports...")
all_good = True

# Test required packages
all_good &= test_import("numpy")
all_good &= test_import("scipy")
all_good &= test_import("matplotlib")
all_good &= test_import("mpi4py")
all_good &= test_import("pandas")
all_good &= test_import("seaborn")
all_good &= test_import("networkx")
all_good &= test_import("tqdm")

# Test optional packages
print("\nOptional packages:")
test_import("cvxpy")
test_import("mosek")

# Test MPI
print("\nTesting MPI functionality...")
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f"MPI initialized: rank {comm.rank} of {comm.size}")
except Exception as e:
    print(f"MPI test failed: {e}")
    all_good = False

if all_good:
    print("\n✓ All required packages installed successfully!")
    sys.exit(0)
else:
    print("\n✗ Some packages are missing. Please check the errors above.")
    sys.exit(1)
EOF

chmod +x test_installation.py

# Run installation test
echo -e "\nRunning installation test..."
python test_installation.py

# Create environment file for easy activation
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Source this file to activate the environment
source venv/bin/activate
echo "Decentralized SNL environment activated"
echo "Run experiments with: ./run_snl.sh"
EOF

chmod +x activate_env.sh

echo -e "\n=========================================="
echo "Setup complete!"
echo "To activate the environment: source activate_env.sh"
echo "To run experiments: ./run_snl.sh"
echo "To run tests: pytest tests/"
echo "=========================================="