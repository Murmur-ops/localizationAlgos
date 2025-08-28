# Getting Started with Decentralized Localization

## Overview

This repository implements advanced decentralized localization algorithms including time synchronization, frequency tracking, and distributed optimization. The codebase demonstrates both the theoretical correctness of these algorithms and the practical limitations of software-based timing.

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip package manager
- Git for version control
- MPI implementation (for distributed algorithms)

### Python Dependencies
```bash
pip install numpy scipy matplotlib cvxpy pyyaml mpi4py networkx
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DecentralizedLocale.git
cd DecentralizedLocale/CleanImplementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import algorithms; print('Installation successful!')"
```

## Quick Start

### 1. Basic Localization (No Synchronization)

```python
from algorithms.mps_advanced import AdvancedMPSAlgorithm
import numpy as np

# Create algorithm instance
mps = AdvancedMPSAlgorithm(n_sensors=20, n_anchors=4, noise_std=0.05)

# Generate test network
positions, measurements, anchors = mps.generate_test_network()

# Run localization
estimated_positions = mps.localize(measurements, anchors)

# Calculate error
rmse = mps.calculate_rmse(positions, estimated_positions)
print(f"Localization RMSE: {rmse:.3f}m")
```

### 2. Time Synchronization with TWTT

```python
from algorithms.time_sync.twtt import RealTWTT

# Initialize TWTT synchronization
twtt = RealTWTT(num_nodes=10)

# Perform synchronization
offsets = twtt.synchronize_network(reference_node=0)

# Check achieved accuracy
print(f"Sync accuracy: {twtt.achieved_accuracy_ns:.1f} nanoseconds")
```

### 3. GPS-Disciplined Anchor Simulation

```python
from test_gps_anchors_fixed import GPSDisciplinedNetwork

# Create network with GPS anchors
network = GPSDisciplinedNetwork(
    n_sensors=20,
    n_anchors=6, 
    network_scale=10.0,  # 10m scale
    gps_time_sync_ns=15.0  # GPS timing accuracy
)

# Run comparison
results = network.run_localization_comparison()
print(f"Improvement with GPS: {results['improvement']:.2f}x")
```

## Examples

### Example 1: Measuring Python Timing Limitations

```bash
python test_python_timing_limits.py
```

This will show you the fundamental timing limitations of Python:
- Timer resolution (~41-50ns)
- Sleep accuracy
- Measurement noise
- Implications for localization

### Example 2: Distributed MPS with MPI

```bash
# Run on 4 processes
mpirun -n 4 python test_distributed_mps.py
```

Configuration via `mps_config.yaml`:
```yaml
algorithm:
  name: "distributed_mps"
  n_sensors: 20
  n_anchors: 4
  noise_std: 0.05
  
distributed:
  consensus_rounds: 10
  convergence_threshold: 0.001
```

### Example 3: Complete Synchronization Pipeline

```python
from algorithms.time_sync import RealTWTT, RealFrequencySync, RealClockConsensus
import numpy as np

# Step 1: Time synchronization
twtt = RealTWTT(num_nodes=10)
time_offsets = twtt.synchronize_network(reference_node=0)

# Step 2: Frequency tracking
freq_sync = RealFrequencySync(num_nodes=10)
for _ in range(100):
    freq_sync.track_frequency_drift(duration_seconds=0.1)
drift_rates = freq_sync.get_drift_estimates()

# Step 3: Consensus clock
consensus = RealClockConsensus(num_nodes=10)
consensus.set_initial_offsets(time_offsets)
final_offsets = consensus.achieve_consensus(rounds=50)

print(f"Final sync accuracy: {np.std(final_offsets):.1f} ns")
```

### Example 4: Analyzing Network Topology Impact

```python
import numpy as np
from algorithms.mps_advanced import AdvancedMPSAlgorithm

# Test different network densities
for connectivity in [0.3, 0.5, 0.7, 0.9]:
    mps = AdvancedMPSAlgorithm(n_sensors=20, n_anchors=4)
    
    # Generate network with specific connectivity
    positions, measurements, anchors = mps.generate_test_network(
        connectivity=connectivity
    )
    
    # Run localization
    estimated = mps.localize(measurements, anchors)
    rmse = mps.calculate_rmse(positions, estimated)
    
    print(f"Connectivity {connectivity:.1f}: RMSE = {rmse:.3f}m")
```

### Example 5: Visualizing Results

```python
from visualize_comparison import create_performance_comparison
import matplotlib.pyplot as plt

# Run comparison across scales
scales = [1, 10, 100]  # meters
results = {}

for scale in scales:
    network = GPSDisciplinedNetwork(
        n_sensors=20,
        n_anchors=6,
        network_scale=scale
    )
    results[scale] = network.run_localization_comparison()

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, scale in enumerate(scales):
    ax = axes[idx]
    data = results[scale]
    
    # Plot true vs estimated positions
    ax.scatter(data['positions'][:, 0], data['positions'][:, 1], 
              label='True', alpha=0.6)
    ax.scatter(data['anchors'][:, 0], data['anchors'][:, 1], 
              marker='^', s=100, label='Anchors')
    
    ax.set_title(f'Scale: {scale}m\nRMSE: {data["rmse_gps"]:.3f}m')
    ax.legend()
    ax.axis('equal')

plt.tight_layout()
plt.show()
```

## Understanding the Results

### Timing Limitations
- **Python Resolution**: ~41-50ns minimum timer resolution
- **Distance Impact**: 1ns = 30cm ranging error
- **Practical Limit**: ~15m ranging accuracy with software timing

### When Synchronization Helps
For 5% measurement noise:
- **Helps**: Distances > 12m (sync error < percentage error)
- **Hurts**: Distances < 12m (sync error > percentage error)
- **GPS Anchors**: Always help by providing hardware timing reference

### Performance Expectations

| Scenario | Expected RMSE | Notes |
|----------|---------------|-------|
| No sync, 5% noise | 14.5m | Baseline |
| Software sync | 15-20m | Worse for small networks |
| GPS anchors | 3-5cm | Hardware timing bypass |
| Ideal hardware | <1cm | Requires dedicated timing |

## Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Performance Tests
```bash
python test_performance.py
```

### Hardware Limitation Tests
```bash
python test_python_timing_limits.py
python test_gps_anchors_fixed.py
```

## Troubleshooting

### Common Issues

1. **ImportError for MPI**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install mpich
   pip install mpi4py
   
   # macOS
   brew install mpich
   pip install mpi4py
   ```

2. **CVX Solver Issues**
   ```bash
   pip install cvxpy clarabel
   ```

3. **Timer Resolution Warning**
   - This is expected! Python cannot achieve nanosecond precision
   - See `timing_precision_analysis_report.md` for details

4. **MPI Process Errors**
   ```bash
   # Test with fewer processes
   mpirun -n 2 python your_script.py
   
   # Check MPI installation
   mpirun --version
   ```

## Key Files to Explore

- `algorithms/time_sync/twtt.py` - Two-way time transfer implementation
- `algorithms/mps_advanced.py` - Advanced MPS localization
- `test_python_timing_limits.py` - Demonstrates fundamental limitations
- `timing_precision_analysis_report.md` - Comprehensive analysis report
- `SYNCHRONIZATION_REALITY_CHECK.md` - Honest assessment of capabilities

## Contributing

When contributing, please ensure:
1. All measurements use real timestamps (no mock data)
2. Document any hardware assumptions
3. Be transparent about limitations
4. Include tests demonstrating actual performance

## Citation

If you use this code in your research, please cite:
```bibtex
@article{nanzer2017precise,
  title={Precise millimeter-wave time transfer for distributed coherent aperture},
  author={Nanzer, Jeffrey A},
  journal={IEEE Transactions on Microwave Theory and Techniques},
  year={2017}
}
```

## License

This project is for educational and research purposes. See LICENSE file for details.

## Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the analysis reports (`*_report.md` files)
3. Open an issue on GitHub with details about your setup