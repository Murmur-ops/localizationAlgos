#!/bin/bash
# Unix/Linux/macOS script to run the sensor localization example

echo "=========================================="
echo "Decentralized Sensor Network Localization"
echo "=========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3 first"
    exit 1
fi

# Check/Install dependencies
echo "Checking dependencies..."
if ! python3 -c "import numpy" &> /dev/null; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
fi

# Check if MPI and mpi4py are available
MPI_AVAILABLE=false
if command -v mpirun &> /dev/null; then
    # MPI command exists, now check for mpi4py
    if python3 -c "import mpi4py" &> /dev/null 2>&1; then
        MPI_AVAILABLE=true
        echo "✓ MPI and mpi4py detected"
    else
        echo "⚠ MPI found but mpi4py not installed"
        echo "  To install: pip3 install mpi4py"
    fi
else
    echo "⚠ MPI not found"
fi

# Run appropriate version
if [ "$MPI_AVAILABLE" = true ]; then
    echo
    echo "Running MPI optimized version (fast)..."
    echo
    mpirun -np 4 python3 snl_mpi_optimized.py
else
    echo
    echo "MPI setup incomplete. To enable fast MPI version:"
    echo "─────────────────────────────────────────────"
    echo "macOS:"
    echo "  brew install mpich"
    echo "  pip3 install mpi4py"
    echo ""
    echo "Ubuntu/Linux:"
    echo "  sudo apt-get install mpich"
    echo "  pip3 install mpi4py"
    echo ""
    echo "Windows:"
    echo "  Option 1: Use WSL2 with Ubuntu"
    echo "  Option 2: Install MS-MPI and then pip install mpi4py"
    echo "─────────────────────────────────────────────"
    echo
    echo "For now, running single-machine version..."
    echo "⚠ WARNING: This version is significantly slower (30-60 seconds)"
    echo
    
    # Check if user wants to continue with slow version
    read -p "Continue with slow version? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 simple_example.py
    else
        echo "Exiting. Please install MPI for better performance."
        exit 0
    fi
fi

echo
echo "=========================================="
echo "Example complete!"
echo "To generate figures, run:"
echo "  python3 generate_figures.py"
echo "=========================================="
echo