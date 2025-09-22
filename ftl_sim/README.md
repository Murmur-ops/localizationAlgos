# FTL (Frequency-Time-Localization) Simulation Framework

## Overview

FTL is a distributed consensus-based system for joint position and time synchronization. It enables a network of nodes to collaboratively determine their positions and synchronize their clocks without central coordination, achieving sub-centimeter accuracy under ideal conditions.

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:Murmur-ops/localizationAlgos.git
cd localizationAlgos/ftl_sim

# Install dependencies
pip install numpy scipy matplotlib pyyaml
```

### Run a Simple Demo

```bash
# Basic 5x5 grid simulation
python demos/run_ftl_grid.py

# Custom parameters
python demos/run_ftl_grid.py --n_nodes 10 --seed 42

# Without visualization (headless mode)
python demos/run_ftl_grid.py --no-viz

# Using configuration file
python demos/run_ftl_grid.py --config configs/scene.yaml
```

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Quick validation
python tests/test_crlb_validation.py
```

## Key Features

- **Distributed Architecture**: No central server required
- **Joint Estimation**: Simultaneously estimates position (x,y), clock bias, clock drift, and carrier frequency offset
- **Sub-centimeter Accuracy**: Achieves 0.9 cm RMSE under ideal conditions
- **Scalable**: Tested with 30+ node networks
- **Robust**: Handles multipath, NLOS conditions, and clock imperfections

## System Components

### Core Modules (`ftl/`)
- **Signal Processing**: IEEE 802.15.4z HRP-UWB waveforms
- **Channel Modeling**: Saleh-Valenzuela multipath model
- **Clock Models**: TCXO/OCXO with Allan variance
- **Factor Graph**: ToA, TDOA, TWR measurements
- **Distributed Consensus**: Consensus-Gauss-Newton algorithm

### State Vector
Each node maintains a 5D state:
```python
[x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
```
- Position in meters
- Clock bias in nanoseconds
- Clock drift in parts-per-billion
- Carrier frequency offset in parts-per-million

## Example Results

With a 30-node network over 50×50m area:
- **Position RMSE**: 0.9 cm (ideal conditions)
- **Convergence**: 50-500 iterations
- **Processing Time**: ~0.5-2 seconds

## Configuration

Edit `configs/scene.yaml` to customize:
- Network topology (grid/random)
- Number of nodes and anchors
- Signal parameters (bandwidth, SNR)
- Channel conditions (indoor/outdoor)
- Solver parameters

## Documentation

For detailed information, see:
- `COMPREHENSIVE_PROJECT_DOCUMENTATION.md` - Full system documentation
- `FTL_SYSTEM_END_TO_END_REPORT.md` - Technical details
- `30_NODE_PERFORMANCE_REPORT.md` - Performance analysis
- `CONSENSUS_AUDIT.md` - Distributed consensus implementation

## Project Structure

```
ftl_sim/
├── ftl/                    # Core simulation modules
│   ├── consensus/          # Distributed consensus implementation
│   ├── signal.py           # Waveform generation
│   ├── channel.py          # Multipath propagation
│   ├── factors_scaled.py   # Measurement factors
│   └── solver_scaled.py    # Optimization solver
├── demos/                  # Demonstration scripts
│   ├── run_ftl_grid.py     # Main demo
│   └── quick_position_plot.py
├── tests/                  # Unit tests (97 tests)
├── configs/                # Configuration files
└── docs/                   # Additional documentation
```

## Performance Notes

Best results require:
- Non-collinear anchor placement (avoid all anchors on a line)
- Good network connectivity (communication range > node spacing)
- High SNR (20+ dB for cm-level accuracy)
- Proper initialization (within 1-2m of true positions)

## License

See repository root for license information.

## Citation

If you use this code in research, please cite:
```
FTL Simulation Framework
https://github.com/Murmur-ops/localizationAlgos
```