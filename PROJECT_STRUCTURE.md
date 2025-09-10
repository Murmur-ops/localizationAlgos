# Project Structure: Decentralized Localization System (Refactored)

## Overview

This project has been reorganized for better maintainability, clearer separation of concerns, and production-readiness. The new structure follows Python best practices with distinct directories for source code, tests, documentation, and configuration.

## Directory Structure

```
CleanImplementation/
│
├── src/                    # All production source code
│   ├── simulation/         # Ideal carrier phase synchronization
│   │   ├── config/        # Simulation configurations
│   │   ├── src/           # Simulation implementation
│   │   └── README.md      # Simulation documentation
│   │
│   ├── emulation/         # Python timing constraints
│   │   ├── config/        # Emulation configurations
│   │   ├── src/           # Emulation implementation
│   │   └── README.md      # Emulation documentation
│   │
│   ├── hardware/          # Hardware interface layer
│   │   ├── interfaces/    # Hardware abstraction
│   │   ├── config/        # Hardware configs
│   │   └── docs/          # Integration guides
│   │
│   ├── core/              # Shared core algorithms
│   │   ├── algorithms/    # MPS, OARS, etc.
│   │   ├── mps_core/      # MPS implementation
│   │   ├── graph_theoretic/ # Graph algorithms
│   │   ├── visualization/ # Plotting utilities
│   │   └── utils/         # Common utilities
│   │
│   └── cli/               # Command-line interface
│       ├── __init__.py
│       └── main.py        # CLI entry point
│
├── tests/                 # All test files
│   ├── test_*.py         # Unit and integration tests
│   └── conftest.py       # Pytest configuration
│
├── docs/                  # All documentation
│   ├── *.md              # Analysis reports and docs
│   └── api/              # API documentation
│
├── configs/               # All configuration files
│   ├── *.yaml            # YAML configurations
│   └── examples/         # Example configs
│
├── scripts/               # Utility scripts
│   ├── run_*.py          # Run scripts
│   ├── simulate_*.py     # Simulation scripts
│   └── visualize_*.py    # Visualization scripts
│
├── experiments/           # Research experiments
│   └── demo_*.py         # Demo scripts
│
├── results/               # Output results
│   ├── *.png             # Visualizations
│   └── *.json            # Numerical results
│
├── setup.py              # Package installation
├── pytest.ini            # Testing configuration
├── requirements.txt      # Dependencies
├── README.md            # Main documentation
├── SETUP_GUIDE.md       # Installation guide
└── PROJECT_STRUCTURE.md # This file
```

## Key Improvements

### 1. Clean Root Directory
- Only essential files at root level
- Clear separation of code, tests, and documentation
- Easy to navigate and understand

### 2. Proper Python Package Structure
- All source code under `src/`
- Installable via `pip install -e .`
- Clear module hierarchy

### 3. Unified Testing Framework
- All tests in dedicated `tests/` directory
- Configured with pytest
- Easy to run: `pytest`

### 4. Command-Line Interface
```bash
# Run simulation
python -m cli.main simulate

# Test timing limits
python -m cli.main emulate --test timing

# Run benchmarks
python -m cli.main benchmark --component all

# Generate visualization
python -m cli.main visualize results.json --plot-type network
```

### 5. Better Documentation
- Comprehensive README for each component
- API documentation
- Clear separation of analysis reports

## Component Descriptions

### Core Components (`src/core/`)
- algorithms/: Core localization algorithms (MPS, belief propagation)
- mps_core/: Message Passing Scheme implementation
- graph_theoretic/: Graph-based localization
- visualization/: Unified plotting and visualization tools
- utils/: Shared utilities and helpers

### Main Applications

#### Simulation (`src/simulation/`)
- Purpose: Theoretical performance with ideal carrier phase
- Technology: Carrier phase @ 2.4 GHz
- RMSE: 0.1-0.2mm
- Meets S-band: ✓

#### Emulation (`src/emulation/`)
- Purpose: Real constraints with Python timing
- Technology: Computer clock
- RMSE: 600-1000mm
- Meets S-band: ✗

#### Hardware (`src/hardware/`)
- Purpose: Interface for real RF hardware
- Status: Ready for integration
- **Supports**: RF phase, UWB, GPS modules

## Quick Start

### Installation
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_mps.py

# Run with coverage
pytest --cov=src
```

### Using the CLI
```bash
# Show help
python -m cli.main --help

# Run simulation
python -m cli.main simulate

# Test timing
python -m cli.main emulate --test timing

# Benchmark all components
python -m cli.main benchmark --component all
```

### Direct Module Usage
```python
# Import core algorithms
from src.core.mps_core.algorithm import MPSAlgorithm
from src.core.visualization.network_plots import NetworkVisualizer

# Run simulation
from src.simulation.src.run_phase_sync_simulation import CarrierPhaseSimulation
sim = CarrierPhaseSimulation("config.yaml")
sim.run()
```

## Configuration System

All components use YAML configuration:
```yaml
# Example: configs/simulation.yaml
network:
  n_sensors: 30
  n_anchors: 4
  
carrier_phase:
  frequency_ghz: 2.4
  phase_noise_milliradians: 1.0
  
visualization:
  style: "publication"
  save_plots: true
```

## Development Workflow

1. **Make changes** in appropriate `src/` subdirectory
2. **Write tests** in `tests/`
3. **Run tests**: `pytest`
4. **Check style**: `flake8 src tests`
5. **Format code**: `black src tests`
6. **Build docs**: `cd docs && make html`

## Key Results Summary

| Approach | Technology | Localization RMSE | Meets S-band? |
|----------|-----------|-------------------|---------------|
| **Simulation** | Carrier phase @ 2.4 GHz | 0.1-0.2mm | ✓ |
| **Emulation** | Python timer | 600-1000mm | ✗ |
| **No sync** | 5% noise model | 14,500mm | ✗ |

## Next Steps

1. **Hardware Integration**: Implement interfaces in `src/hardware/`
2. **Performance Optimization**: Port critical paths to C/Rust
3. **Field Testing**: Deploy with actual S-band hardware
4. **Documentation**: Generate full API documentation