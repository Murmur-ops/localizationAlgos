# YAML Configuration Guide

## Quick Start

The easiest way to get started is to copy one of the templates:

```bash
# For quick testing
cp configs/templates/quick_test.yaml configs/my_config.yaml

# For research comparison
cp configs/templates/research_comparison.yaml configs/my_config.yaml

# Run with your config
python scripts/run_mps.py --config configs/my_config.yaml
```

## Template Files

### Available Templates

1. **`configs/template.yaml`** - Master template with ALL options and detailed comments
2. **`configs/templates/quick_test.yaml`** - Quick testing (runs in <5 seconds)
3. **`configs/templates/research_comparison.yaml`** - MPS vs ADMM comparison
4. **`configs/templates/distributed_large.yaml`** - MPI distributed execution
5. **`configs/templates/high_accuracy.yaml`** - Maximum accuracy configuration

### Example Configurations

- **`configs/default.yaml`** - Balanced default settings
- **`configs/examples/small_network.yaml`** - 10 sensors, quick convergence
- **`configs/examples/large_network.yaml`** - 100 sensors, distributed
- **`configs/examples/20_nodes_8_anchors.yaml`** - Specific network size

## Key Parameters

### Network Size
```yaml
network:
  n_sensors: 30    # Number of unknown nodes
  n_anchors: 6     # Number of GPS/known nodes
```

**Guidelines:**
- Small: 5-20 sensors (testing)
- Medium: 20-50 sensors (typical research)
- Large: 50-200 sensors (distributed recommended)
- Very Large: 200+ sensors (require MPI)

### Measurement Noise
```yaml
measurements:
  noise_factor: 0.05  # 5% multiplicative noise
```

**Typical Values:**
- 0.01-0.02: Ultra-wideband (UWB)
- 0.03-0.05: Good RF conditions
- 0.05-0.10: Typical wireless
- 0.10-0.20: Challenging environment

### Algorithm Parameters

#### MPS Parameters
```yaml
algorithm:
  gamma: 0.99    # Consensus mixing (0.9-0.999)
  alpha: 1.0     # Step size (0.1-2.0)
```

**Tuning Guide:**
- **Oscillating?** → Decrease alpha, increase gamma
- **Too slow?** → Increase alpha, decrease gamma slightly
- **Not converging?** → Increase max_iterations
- **Unstable?** → Increase gamma to 0.995+

#### ADMM Parameters
```yaml
algorithm:
  rho: 1.0          # Penalty parameter
  admm_alpha: 1.5   # Over-relaxation
```

### Convergence Settings
```yaml
algorithm:
  max_iterations: 500     # Stop after this many
  tolerance: 0.00001      # Stop if change < this
  patience: 50            # Stop if no improvement
```

## Common Use Cases

### 1. Quick Algorithm Test
```yaml
# Use: configs/templates/quick_test.yaml
network:
  n_sensors: 10
  n_anchors: 4
algorithm:
  max_iterations: 100
  tolerance: 0.001
```

### 2. Paper Reproduction
```yaml
# Use: configs/templates/research_comparison.yaml
measurements:
  seed: 42  # Reproducible
validation:
  compute_crlb: true
  monte_carlo_runs: 100
```

### 3. Real Deployment
```yaml
network:
  n_sensors: 50
  communication_range: 0.2  # Realistic sparse
measurements:
  noise_factor: 0.08
  outlier_probability: 0.05
algorithm:
  adaptive: true  # Auto-tune
```

### 4. Maximum Accuracy
```yaml
# Use: configs/templates/high_accuracy.yaml
algorithm:
  gamma: 0.999
  alpha: 0.5
  max_iterations: 2000
  tolerance: 0.000001
```

## Performance Impact

| Parameter | Impact on Speed | Impact on Accuracy |
|-----------|----------------|-------------------|
| n_sensors | High (quadratic) | Neutral |
| n_anchors | Low | High (positive) |
| gamma ↑ | Slower convergence | More stable |
| alpha ↑ | Faster (may oscillate) | Can decrease |
| max_iterations ↑ | Slower | Higher |
| tolerance ↓ | Slower | Higher |

## Advanced Features

### Distributed Execution (MPI)
```yaml
mpi:
  enable: true
  n_processes: 8
  consensus_rounds: 10
```

Run with:
```bash
mpirun -n 8 python scripts/run_distributed.py --config your_config.yaml
```

### Time Synchronization
```yaml
time_sync:
  enable: true
  protocol: "twtt"
  drift_ppm: 20
  initial_offset_ns: 1000
```

### Visualization
```yaml
visualization:
  show_plots: true     # Display plots
  animate: true        # Animated convergence
  style: "paper"       # Publication quality
output:
  save_plots: true
  plot_format: "pdf"   # Vector graphics
  plot_dpi: 300        # High resolution
```

### Experimental Features
```yaml
experimental:
  robust_estimator: "huber"  # Outlier rejection
  multi_resolution: true      # Coarse-to-fine
  hierarchical: true          # For very large networks
  use_gpu: true              # GPU acceleration (if available)
```

## Troubleshooting

### Problem: Algorithm not converging
**Solution:**
```yaml
algorithm:
  gamma: 0.995          # Increase stability
  alpha: 0.8            # Reduce step size
  max_iterations: 1000  # More iterations
```

### Problem: Too slow
**Solution:**
```yaml
algorithm:
  tolerance: 0.001      # Relax tolerance
  patience: 20          # Earlier stopping
output:
  save_interval: 0      # Don't save intermediate
  verbose: false        # Reduce logging
```

### Problem: Poor accuracy
**Solution:**
```yaml
network:
  n_anchors: 8          # More anchors
  communication_range: 0.4  # Better connectivity
algorithm:
  tolerance: 0.000001   # Tighter tolerance
validation:
  compute_crlb: true    # Compare to bound
```

### Problem: Memory issues
**Solution:**
```yaml
output:
  save_history: false   # Don't store history
mpi:
  enable: true          # Distribute computation
experimental:
  hierarchical: true    # Process in chunks
```

## Creating Custom Configurations

1. **Start with a template:**
   ```bash
   cp configs/template.yaml configs/custom.yaml
   ```

2. **Remove unnecessary options** (defaults will be used)

3. **Focus on key parameters:**
   - Network size (n_sensors, n_anchors)
   - Noise level (noise_factor)
   - Algorithm parameters (gamma, alpha)
   - Convergence (max_iterations, tolerance)

4. **Test with small network first:**
   ```yaml
   network:
     n_sensors: 10  # Start small
   ```

5. **Scale up gradually**

## Command Line Override

You can override any parameter from command line:

```bash
# Override network size
python scripts/run_mps.py --config base.yaml \
  --set network.n_sensors=50 network.n_anchors=10

# Override algorithm
python scripts/run_mps.py --config base.yaml \
  --set algorithm.name=admm algorithm.rho=2.0

# Override output
python scripts/run_mps.py --config base.yaml \
  --set output.verbose=false output.save_plots=false
```

## Best Practices

1. **Always use seeds during development** for reproducibility
2. **Start with templates** rather than writing from scratch
3. **Change one parameter at a time** when tuning
4. **Save configurations** that work well
5. **Use descriptive names** for config files
6. **Comment your changes** in the YAML file
7. **Version control** your configurations
8. **Validate against CRLB** when possible
9. **Run multiple seeds** for publication results
10. **Document parameter choices** in your research

## Examples Directory Structure

```
configs/
├── template.yaml                 # Master template
├── default.yaml                  # Default configuration
├── examples/
│   ├── small_network.yaml      # 10 nodes
│   ├── large_network.yaml      # 100 nodes
│   └── 20_nodes_8_anchors.yaml # Specific setup
└── templates/
    ├── quick_test.yaml          # Quick testing
    ├── research_comparison.yaml # Research
    ├── distributed_large.yaml   # MPI/distributed
    └── high_accuracy.yaml       # Maximum accuracy
```