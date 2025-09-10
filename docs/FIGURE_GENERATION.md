# Figure Generation Guide

## Overview

This guide explains how to generate publication-quality figures using YAML configuration files with the DecentralizedLocale system.

## Quick Start

### 1. Generate Figures from a Single Configuration

```bash
python scripts/generate_figures.py --config configs/quick_test.yaml
```

This creates:
- `figures/network_and_convergence.png` - Network topology and convergence plot
- `figures/performance_analysis.png` - Performance metrics and parameters

### 2. Compare Multiple Configurations

```bash
python scripts/generate_figures.py --compare configs/quick_test.yaml configs/high_accuracy.yaml configs/sband_precision.yaml
```

This creates:
- `figures/configuration_comparison.png` - Side-by-side comparison of all configs

### 3. Generate Figures for All Configurations

```bash
python scripts/generate_figures.py --all
```

## Using run_distributed.py with Visualization

The distributed MPS script now supports figure generation:

```bash
# Run with visualization
mpirun -n 4 python3 scripts/run_distributed.py --config configs/distributed_large.yaml --visualize

# Without MPI (single process)
python3 scripts/run_distributed.py --config configs/quick_test.yaml --visualize
```

This creates:
- `results/distributed_results.png` - Convergence and final network positions

## Using run_mps.py with Visualization

```bash
python scripts/run_mps.py --config configs/quick_test.yaml --visualize
```

## Available Configuration Files

| Config File | Description | Sensors | Use Case |
|------------|-------------|---------|----------|
| `quick_test.yaml` | Fast testing | 10 | Development/testing |
| `research_comparison.yaml` | Algorithm comparison | 30 | Research benchmarks |
| `distributed_large.yaml` | Large-scale MPI | 100 | Scalability testing |
| `high_accuracy.yaml` | Maximum precision | 40 | Accuracy benchmarks |
| `sband_precision.yaml` | S-band millimeter accuracy | 30 | Carrier phase sync |

## Figure Types Generated

### 1. Network Visualization
- Node positions (sensors and anchors)
- Communication links
- Node IDs

### 2. Convergence Analysis
- Objective function over iterations
- Log-scale convergence plot
- Convergence point marker

### 3. Performance Metrics
- RMSE history
- Error statistics (mean, max, min)
- Algorithm parameters
- Results summary

### 4. Comparison Plots
- Multi-config convergence
- RMSE comparison
- Iteration count comparison
- Summary table

## Customizing Figure Generation

### Output Directory

```bash
# Save to custom directory
python scripts/generate_figures.py --config configs/quick_test.yaml --output-dir my_figures/

# Default is 'figures/'
```

### Figure Quality Settings

Edit `scripts/generate_figures.py` to modify:
- DPI: Change `dpi=150` to higher values for publication (300-600)
- Figure size: Modify `figsize=(14, 6)` tuples
- Font sizes: Adjust `fontsize` parameters

### Color Schemes

Default colors:
- Blue: Sensor nodes
- Red: Anchor nodes
- Gray: Communication links
- Green: Convergence marker

## YAML Configuration for Figures

To control figure generation through YAML:

```yaml
output:
  save_results: true
  output_dir: "results/"
  save_plots: true  # Enable figure generation
  plot_format: "png"  # or "pdf", "svg"
  plot_dpi: 300      # Figure resolution
  
visualization:
  show_plots: true
  figure_size: [14, 6]
  color_scheme: "default"
```

## Batch Processing

Create a script for batch figure generation:

```bash
#!/bin/bash
# batch_figures.sh

configs=(
    "configs/quick_test.yaml"
    "configs/high_accuracy.yaml"
    "configs/distributed_large.yaml"
)

for config in "${configs[@]}"; do
    echo "Processing $config..."
    python scripts/generate_figures.py --config "$config" --output-dir "figures/$(basename $config .yaml)/"
done

# Generate comparison
python scripts/generate_figures.py --compare "${configs[@]}" --output-dir "figures/comparison/"
```

## S-Band Visualization

For S-band carrier phase results:

```bash
# Run simulation and generate figures
python src/simulation/src/run_phase_sync_simulation.py

# Visualize results
python scripts/visualize_sband_results.py
```

This creates:
- `results/sband_performance_visualization.png` - Comprehensive S-band results
- `results/sband_performance_visualization.pdf` - Publication-ready PDF

## Directory Structure

After running figure generation:

```
figures/
├── network_and_convergence.png
├── performance_analysis.png
├── configuration_comparison.png
└── distributed_results.png

results/
├── figures/
│   └── *.png
├── data/
│   └── *.json
└── sband_performance_visualization.png
```

## Troubleshooting

### No Figures Generated
- Check matplotlib is installed: `pip install matplotlib`
- Verify output directory permissions
- Check YAML configuration has valid paths

### MPI Visualization Issues
- Only rank 0 generates figures
- Ensure `--visualize` flag is used
- Check output directory exists

### Memory Issues with Large Networks
- Reduce DPI: `dpi=100`
- Generate figures separately for each config
- Use `--no-save` to skip data saving

## Examples

### Example 1: Quick Test with Figures
```bash
python scripts/run_mps.py --config configs/quick_test.yaml --visualize
```

### Example 2: Distributed with MPI
```bash
mpirun -n 4 python3 scripts/run_distributed.py --config configs/distributed_large.yaml --visualize
```

### Example 3: Compare All Configs
```bash
python scripts/generate_figures.py --all --output-dir comparison_results/
```

### Example 4: S-band Performance
```bash
cd src/simulation/src
python3 run_phase_sync_simulation.py
cd ../../..
python scripts/visualize_sband_results.py
```

## Advanced Usage

### Custom Figure Generation

Create your own visualization script:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_figures import load_config, dict_to_config
from src.core.mps_core.algorithm import MPSAlgorithm

# Load and run
config_dict = load_config('configs/my_config.yaml')
config = dict_to_config(config_dict)
algorithm = MPSAlgorithm(config)
results = algorithm.run()

# Custom plotting
import matplotlib.pyplot as plt
# Your custom visualization code here
```

## Performance Tips

1. **For large networks (>100 nodes)**: Use lower DPI initially
2. **For publication**: Use PDF format with 300+ DPI
3. **For presentations**: Use PNG with 150 DPI
4. **For web**: Use SVG format

## Clean Output

The cleaned-up figure structure:
- `figures/` - Main figure output directory
- `results/figures/` - Algorithm-specific figures
- `visualizations/` - Temporary/working figures

All redundant directories have been removed:
- ~~figures_actual/~~
- ~~figures_improved/~~
- ~~figures_real_data/~~
- ~~visualization/~~