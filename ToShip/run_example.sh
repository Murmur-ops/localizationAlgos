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

# Run the example
echo
echo "Running sensor localization example..."
echo
python3 simple_example.py

echo
echo "=========================================="
echo "Example complete! Check the generated images:"
echo "- simple_example_results.png"
echo "- simple_example_convergence.png"
echo "=========================================="
echo