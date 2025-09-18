# Configuration Files for FTL Consensus System

This directory contains YAML configuration files for different scenarios and use cases.

## Available Configurations

### 1. `ideal_30node.yaml`
**Purpose**: Achieve maximum accuracy under ideal conditions
- **Performance**: 0.9 cm RMSE
- **Iterations**: ~457
- **Use Case**: Surveying, high-precision mapping
- **Key Parameters**:
  - Consensus gain μ = 0.05
  - 30 nodes, 5 anchors
  - 1 cm measurement noise
  - 25m communication range

### 2. `ideal_30node_fast.yaml`
**Purpose**: Fast convergence for real-time applications
- **Performance**: 1.0 cm RMSE
- **Iterations**: 49
- **Use Case**: Real-time tracking, robotics
- **Key Parameters**:
  - Consensus gain μ = 0.10
  - Adaptive parameters
  - Early stopping enabled

### 3. `scene.yaml` (existing)
**Purpose**: General network configuration
- Various network sizes and topologies
- Configurable for different experiments

## Usage

### Running a Configuration
```bash
# For Python script
python run_yaml_config.py configs/ideal_30node.yaml

# For analysis
python analyze_network.py --config configs/ideal_30node.yaml

# For batch experiments
python batch_experiments.py --configs configs/*.yaml
```

### Loading in Python
```python
import yaml

with open('configs/ideal_30node.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
consensus_gain = config['consensus']['parameters_accurate']['consensus_gain']
n_nodes = config['scene']['n_nodes']
```

## Configuration Structure

### Main Sections

1. **scene**: Network topology and node placement
   - Area dimensions
   - Number of nodes/anchors
   - Communication range
   - Node placement strategy

2. **signal**: Waveform parameters
   - Carrier frequency
   - Bandwidth
   - Signal type (HRP_UWB, Zadoff-Chu)

3. **channel**: Propagation environment
   - Path loss model
   - Multipath settings
   - Noise levels

4. **clocks**: Oscillator and timing
   - Clock quality (Allan deviation)
   - Initial errors
   - SCO settings

5. **measurements**: Ranging configuration
   - Measurement type (ToA/TDoA)
   - Noise levels
   - NLOS handling

6. **consensus**: Algorithm parameters
   - Consensus gain (μ)
   - Step size (α)
   - Convergence criteria
   - Iteration limits

7. **initialization**: Initial estimates
   - Position initialization method
   - Clock initialization

8. **output**: Results and visualization
   - Metrics to compute
   - Plot settings
   - File formats

## Parameter Guidelines

### For Accuracy (Survey/Mapping)
- Consensus gain: 0.01-0.05
- Max iterations: 300-500
- Step size: 0.2-0.3
- Strict tolerances: 1e-5 to 1e-6

### For Speed (Real-time)
- Consensus gain: 0.10-0.20
- Max iterations: 50-100
- Step size: 0.4-0.5
- Relaxed tolerances: 1e-4

### For Robustness (Noisy)
- Consensus gain: 0.01-0.02
- Smaller step size: 0.1-0.2
- Damping: 1e-3 to 1e-2
- Variance inflation for NLOS

## Creating Custom Configurations

### Template
```yaml
# Custom configuration
scene:
  name: "My Network"
  n_nodes: 20
  n_anchors: 4
  # ... other scene parameters

consensus:
  parameters:
    consensus_gain: 0.08  # Tune this
    step_size: 0.35       # And this
    # ... other parameters

# Include only sections you want to override
# Missing sections will use defaults
```

### Inheritance
You can inherit from base configurations:
```yaml
# Inherit from ideal_30node.yaml
base_config: "ideal_30node.yaml"

# Override specific parameters
consensus:
  parameters_accurate:
    consensus_gain: 0.07  # Different μ
```

## Validation

### Check Configuration
```python
from ftl.config import validate_config

config = yaml.safe_load(open('my_config.yaml'))
errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```

### Performance Benchmarks
Each configuration includes expected performance:
```yaml
benchmarks:
  expected:
    rmse_cm: 0.9
    iterations: 457
  thresholds:
    max_rmse_cm: 2.0  # Fail if worse
```

## Tips

1. **Start with ideal configurations** and gradually add realism
2. **Test consensus gain** in range [0.01, 0.20]
3. **Use adaptive parameters** for difficult networks
4. **Enable early stopping** for real-time applications
5. **Increase damping** for oscillation issues
6. **Add center anchor** for 50×50m or larger areas

## Common Issues and Solutions

| Issue | Solution | Config Parameter |
|-------|----------|-----------------|
| Slow convergence | Increase μ | `consensus_gain: 0.10` |
| Oscillation | Decrease α | `step_size: 0.2` |
| Poor accuracy | Decrease μ | `consensus_gain: 0.03` |
| Numerical issues | Increase damping | `damping_lambda: 1e-3` |
| Timeout | Relax tolerances | `gradient_tol: 1e-4` |

## References

- Performance analysis: `30_NODE_PERFORMANCE_REPORT.md`
- Algorithm details: `CONSENSUS_AUDIT.md`
- System overview: `FTL_SYSTEM_END_TO_END_REPORT.md`