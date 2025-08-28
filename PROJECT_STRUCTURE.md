# Project Structure: Decentralized Localization System

## Overview

This project is organized into three distinct components:

1. **Simulation** - Theoretical performance with ideal hardware
2. **Emulation** - Real constraints using computer's clock  
3. **Hardware Ready** - Interface for actual RF hardware

## Directory Structure

```
CleanImplementation/
│
├── simulation/              # Ideal carrier phase synchronization
│   ├── config/             # YAML configuration files
│   │   └── phase_sync_sim.yaml
│   ├── src/                # Simulation source code
│   │   ├── run_phase_sync_simulation.py
│   │   ├── simulate_ideal_phase_sync.py
│   │   └── simulate_nanzer_plus_mps.py
│   ├── visualizations/     # Generated plots
│   └── README.md          # Detailed documentation
│
├── emulation/              # Python timing limitations
│   ├── config/            # YAML configurations
│   │   └── time_sync_emulation.yaml
│   ├── src/               # Time-based implementation
│   │   ├── test_python_timing_limits.py
│   │   └── time_sync/     # TWTT, frequency, consensus
│   ├── results/           # Test results
│   └── README.md         # Documentation
│
├── hardware_ready/         # Future hardware interface
│   ├── interfaces/        # Hardware abstraction layer
│   ├── config/           # Hardware configurations
│   ├── docs/             # Integration guides
│   └── README.md        # Hardware requirements
│
├── shared/                # Common components
│   ├── algorithms/       # Core MPS, OARS, etc.
│   ├── visualization/    # Unified plotting module
│   │   └── network_plots.py
│   └── utils/           # Common utilities
│
└── PROJECT_STRUCTURE.md  # This file
```

## Component Descriptions

### 1. Simulation (`simulation/`)

**Purpose**: Demonstrates theoretical performance with ideal carrier phase synchronization

**Key Features**:
- Implements Nanzer paper's carrier phase approach
- Uses floating-point to simulate picosecond-level precision
- YAML-based configuration
- Comprehensive visualization
- **Expected RMSE: 0.1-0.2mm** (meets S-band requirement)

**Run Example**:
```bash
cd simulation/src
python run_phase_sync_simulation.py
```

### 2. Emulation (`emulation/`)

**Purpose**: Shows real limitations using computer's clock

**Key Features**:
- Uses `time.perf_counter_ns()` (~41ns resolution)
- Implements TWTT, frequency sync, consensus
- Demonstrates why time-based approach fails
- **Achieved RMSE: 600-1000mm** (doesn't meet requirements)

**Run Example**:
```bash
cd emulation/src
python test_python_timing_limits.py
```

### 3. Hardware Ready (`hardware_ready/`)

**Purpose**: Interface layer for real RF hardware integration

**Key Features**:
- Hardware abstraction layer (HAL)
- Support for multiple hardware types:
  - RF phase measurement modules
  - UWB rangers
  - GPS-disciplined oscillators
- Configuration templates
- Integration documentation

**Future Use**:
```python
from hardware_ready.interfaces import HardwareInterface

hw = HardwareInterface(type='rf_phase')
distance = hw.measure_distance(node_i, node_j)
```

### 4. Shared (`shared/`)

**Purpose**: Common algorithms and utilities

**Key Components**:
- **algorithms/**: Core MPS, OARS matrices, proximal operators
- **visualization/**: Unified plotting module with consistent style
- **utils/**: Common utilities

## Key Results Summary

| Approach | Technology | Ranging Accuracy | Localization RMSE | Meets S-band? |
|----------|------------|------------------|-------------------|---------------|
| **Simulation** | Carrier phase @ 2.4 GHz | 0.02mm | 0.1-0.2mm | ✓ |
| **Emulation** | Python timer | 600mm | 600-1000mm | ✗ |
| No sync | 5% noise model | 50mm @ 1m | 14,500mm | ✗ |
| GPS time | Hardware GPS | 30-50mm | 30-50mm | ✗ |

## Configuration System

All components use YAML configuration files for easy parameter tuning:

```yaml
# Example: simulation/config/phase_sync_sim.yaml
network:
  n_sensors: 20
  n_anchors: 4
  scale_meters: 10.0
  
carrier_phase:
  frequency_ghz: 2.4
  phase_noise_milliradians: 1.0
  
visualization:
  style: "publication"
  save_plots: true
```

## Visualization System

Unified visualization module provides:
- Network topology plots (true vs estimated positions)
- Convergence curves (RMSE and objective function)
- Error distribution histograms
- Spatial error heatmaps
- RMSE comparison bar charts

## Quick Start Guide

1. **See theoretical performance**:
   ```bash
   cd simulation/src
   python run_phase_sync_simulation.py
   ```

2. **Test Python timing limits**:
   ```bash
   cd emulation/src
   python test_python_timing_limits.py
   ```

3. **View all documentation**:
   - `simulation/README.md` - Carrier phase theory
   - `emulation/README.md` - Timing limitations
   - `hardware_ready/README.md` - Hardware requirements

## Key Insights

1. **Algorithms are correct** - MPS works with accurate measurements
2. **Carrier phase is key** - Milliradian phase → millimeter ranging
3. **Python timing is limiting** - 41ns resolution → 12m uncertainty
4. **Hardware is required** - Need RF phase measurement for S-band

## Next Steps

1. **Hardware Integration**: Implement HAL for real RF modules
2. **Field Testing**: Deploy with actual S-band hardware
3. **Optimization**: Port critical paths to C/FPGA
4. **Scaling**: Test with larger networks (100+ nodes)