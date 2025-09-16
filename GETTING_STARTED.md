# Getting Started Guide

Welcome to the Decentralized Localization System! This guide will help you quickly get up and running.

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/Murmur-ops/DelocaleClean.git
cd DelocaleClean

# Install dependencies
pip install -r requirements.txt

# For visualization and analysis
pip install matplotlib scikit-learn networkx
```

## Quick Start Examples

### 1. Basic 10-Node Indoor Localization

The simplest demo - 10 nodes in a 10×10m room with excellent performance:

```bash
# Run the 10-node demo (4 anchors, 6 unknowns)
python demo_10_nodes.py

# Expected output:
# - RMSE: ~0.01m (1cm accuracy!)
# - Visualization saved to spread_spectrum_analysis.png
```

**What it demonstrates:**
- Spread spectrum signal generation
- Realistic channel modeling with multipath
- Smart trilateration initialization
- Sub-centimeter accuracy in ideal conditions

### 2. Challenging 30-Node Large Area

Test the system's limits with sparse anchors:

```bash
# Run 30-node demo (4 anchors, 26 unknowns over 50×50m)
python demo_30_nodes_large.py

# Expected output:
# - RMSE: ~20m (poor due to sparse anchors)
# - Shows bimodal performance
# - Visualization saved to 30_node_localization_results.png
```

**What it reveals:**
- Initialization matters more than algorithm
- Need ~15% nodes as anchors for reliability
- Local minima challenges with sparse networks

### 3. Analyze System Performance

Understand why certain nodes fail:

```bash
# Analyze failure patterns
python analyze_30_node_failures.py

# This will show:
# - Connectivity analysis
# - Comparison of initialization methods
# - Which nodes fail and why
```

### 4. Test Spread Spectrum Signals

Visualize the RF waveforms:

```bash
# Generate detailed signal analysis
python visualize_spread_spectrum.py

# Creates spread_spectrum_analysis.png showing:
# - Gold code autocorrelation
# - Spectrum spreading
# - Sub-sample interpolation
```

## Configuration Files

### configs/10_node_demo.yaml
Perfect for testing in small areas with good anchor coverage:
- 10×10m area
- 4 corner anchors
- Indoor LOS propagation
- Expected RMSE: <5cm

### configs/30_node_large.yaml
Stress test for large areas with minimal anchors:
- 50×50m area
- Only 4 corner anchors
- Urban propagation with 15% NLOS
- Expected RMSE: 10-20m (challenging!)

### configs/test_config.yaml
Minimal configuration for unit tests

## Key Concepts

### Initialization Strategies

The system supports multiple initialization methods:

```python
# 1. Smart Trilateration (good with many anchors)
init_trilat = smart_initialization(unknowns, measurements, anchors)

# 2. Center initialization (surprisingly effective!)
init_center = np.ones(len(unknowns) * 2) * center_value

# 3. MDS-based (uses all measurements)
init_mds = mds_initialization(measurements, anchors, unknowns)
```

### Channel Models

Realistic RF propagation modeling:

```python
from src.channel.propagation import RangingChannel, PropagationType

# Configure channel
channel = RangingChannel(config)

# Generate measurement with multipath/NLOS
measurement = channel.generate_measurement(
    true_distance=10.0,
    prop_type=PropagationType.LOS,  # or NLOS, OBSTRUCTED
    environment='indoor'  # or 'urban', 'suburban'
)
```

### Distributed ADMM Solver

```python
from src.localization.robust_solver import RobustLocalizer

solver = RobustLocalizer(dimension=2, huber_delta=1.0)
positions, info = solver.solve(
    initial_positions,
    measurements,
    anchor_positions
)
```

## Understanding Results

### Good Performance Indicators
- RMSE < 1m for indoor environments
- Median error < 0.5m
- >80% nodes with <1m error
- Convergence in <20 iterations

### Warning Signs
- Bimodal error distribution
- Large difference between mean and median error
- <50% nodes with good accuracy
- Need >50 iterations to converge

## Advanced Features

### 1. Test Different SNR Levels

Modify channel configuration to test robustness:

```yaml
channel:
  path_loss_exponent: 3.5  # Higher = more loss
  nlos_probability: 0.3    # 30% NLOS measurements
  nlos_bias_mean_m: 5.0    # Larger NLOS errors
```

### 2. Distributed Consensus

Test time synchronization:

```bash
python demo_distributed_consensus.py
```

### 3. Performance Analysis

Generate detailed reports:

```bash
# Analyze measurement quality
python analyze_10_node_results.py

# Investigate convergence
python investigate_results.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install numpy scipy matplotlib pyyaml
   ```

2. **Poor Localization Results**
   - Check anchor coverage (need >15% as anchors)
   - Verify measurements are connected
   - Try different initialization methods

3. **Convergence Issues**
   - Increase max_iterations in config
   - Adjust huber_delta for outlier handling
   - Check for disconnected network components

## Learn More

- **[LOCALIZATION_PERFORMANCE_REPORT.md](LOCALIZATION_PERFORMANCE_REPORT.md)** - Detailed performance analysis
- **[LOCALIZATION_CHALLENGES.md](LOCALIZATION_CHALLENGES.md)** - Future research directions
- **[docs/](docs/)** - Technical specifications

## Tips for Success

1. **Start Small**: Begin with 10_node_demo to understand the system
2. **Anchor Placement Matters**: Corners work for convex areas, but need more for complex shapes
3. **Initialization is Key**: Try multiple methods and pick best result
4. **Dense > Sparse**: More measurements always help
5. **Check Connectivity**: Ensure all nodes can reach 3+ anchors (directly or indirectly)

## Contributing

Found a bug or want to add features?
- Open an issue on GitHub
- Submit a pull request
- Check ROADMAP.md for planned features

Happy localizing!