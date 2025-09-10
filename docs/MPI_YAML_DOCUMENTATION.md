# MPS Algorithm: MPI and YAML Configuration Documentation

## Table of Contents
1. [Overview](#overview)
2. [YAML Configuration System](#yaml-configuration-system)
3. [MPI Distributed Execution](#mpi-distributed-execution)
4. [Quick Start Guide](#quick-start-guide)
5. [Configuration Reference](#configuration-reference)
6. [MPI Performance Tuning](#mpi-performance-tuning)
7. [Troubleshooting](#troubleshooting)

## Overview

The MPS (Matrix-Parametrized Proximal Splitting) algorithm now supports:
- **YAML Configuration**: Flexible, hierarchical configuration management
- **MPI Distribution**: Scalable parallel execution across multiple processes
- **Command-line Interface**: Easy execution with parameter overrides

### Key Features
- Configuration inheritance and composition
- Environment variable substitution
- Mathematical expression evaluation
- Dynamic load balancing for MPI
- Checkpoint/restart capability
- Performance benchmarking tools

## YAML Configuration System

### Basic Configuration Structure

```yaml
# Example: configs/default.yaml
network:
  n_sensors: 30         # Number of sensors to localize
  n_anchors: 6          # Number of anchors (known positions)
  dimension: 2          # 2D or 3D localization
  communication_range: 0.3  # Communication radius
  scale: 1.0           # Network scale in meters

measurements:
  noise_factor: 0.05    # 5% measurement noise
  carrier_phase: false  # Enable millimeter accuracy
  outlier_probability: 0.0

algorithm:
  gamma: 0.999         # Proximal gradient parameter
  alpha: 10.0          # Step size
  max_iterations: 1000
  tolerance: 1e-6

mpi:
  enable: false        # Enable MPI distribution
  buffer_size_kb: 1024
```

### Configuration Inheritance

Configurations can extend from base configs:

```yaml
# configs/mpi/mpi_large.yaml
extends: ../default.yaml  # Inherit from default

network:
  n_sensors: 100  # Override specific values
  n_anchors: 15

mpi:
  enable: true
  async_communication: true
```

### Environment Variables

Use environment variables in configs:

```yaml
network:
  n_sensors: ${MPS_N_SENSORS:30}  # Default to 30 if not set
  
output:
  output_dir: ${MPS_OUTPUT_DIR:results/}
```

### Mathematical Expressions

Evaluate expressions in configs:

```yaml
network:
  n_anchors: eval: int(0.2 * 100)  # 20% of sensors
  
algorithm:
  alpha: eval: 2 * pi  # Use mathematical constants
```

## MPI Distributed Execution

### Architecture

The MPI implementation uses a block distribution strategy:

```
Sensors: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Process 0: [0, 1, 2, 3, 4]
Process 1: [5, 6, 7, 8, 9]
```

Key components:
- **Block Distribution**: Sensors divided evenly among processes
- **Sequential Dependencies**: L matrix handled sequentially
- **Consensus Updates**: W matrix via AllReduce operations
- **Asynchronous Communication**: Optional for performance

### Running with MPI

Basic command:
```bash
mpirun -n 4 python scripts/run_mps_mpi.py --config configs/mpi/mpi_medium.yaml
```

With overrides:
```bash
mpirun -n 8 python scripts/run_mps_mpi.py \
  --config configs/default.yaml \
  --override network.n_sensors=100 algorithm.max_iterations=500
```

Benchmark mode:
```bash
mpirun -n 16 python scripts/run_mps_mpi.py \
  --config configs/mpi/mpi_benchmark.yaml \
  --benchmark --profile
```

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install mpi4py numpy scipy pyyaml
```

### 2. Test MPI Installation

```bash
python test_mpi_simple.py
```

### 3. Run Small Example

```bash
# Single process (no MPI)
python scripts/run_mps_mpi.py --config configs/default.yaml

# With MPI (4 processes)
mpirun -n 4 python scripts/run_mps_mpi.py --config configs/mpi/mpi_small.yaml
```

### 4. Run Large Network

```bash
# Large network with 16 MPI processes
mpirun -n 16 python scripts/run_mps_mpi.py \
  --config configs/mpi/mpi_large.yaml \
  --output-dir results/large_run/
```

## Configuration Reference

### Pre-built Configurations

| Config File | Purpose | Network Size | Runtime | Use Case |
|------------|---------|--------------|---------|----------|
| `default.yaml` | Standard single-process | 30 sensors | 5-10s | Development |
| `mpi/mpi_small.yaml` | Quick MPI testing | 20 sensors | 2-5s | Testing |
| `mpi/mpi_medium.yaml` | Standard MPI | 50 sensors | 10-20s | Production |
| `mpi/mpi_large.yaml` | Large-scale | 100 sensors | 30-60s | Research |
| `mpi/mpi_benchmark.yaml` | Performance testing | Variable | Variable | Benchmarking |
| `high_accuracy.yaml` | Maximum accuracy | 40 sensors | 30-60s | High precision |
| `fast_convergence.yaml` | Quick results | 30 sensors | 2-5s | Real-time |
| `large_network.yaml` | Very large networks | 200 sensors | 60-120s | Large deployments |
| `noisy_measurements.yaml` | Robustness testing | 35 sensors | 15-30s | Noisy environments |

### Key Parameters

#### Network Parameters
- `n_sensors`: Number of sensors to localize
- `n_anchors`: Number of anchors with known positions
- `dimension`: Spatial dimension (2 or 3)
- `communication_range`: Max communication distance (fraction of scale)
- `scale`: Network scale in meters
- `topology`: Network topology ("random", "grid", "cluster")

#### Algorithm Parameters
- `gamma`: Proximal gradient parameter (0 < γ < 1)
- `alpha`: Step size / over-relaxation parameter
- `max_iterations`: Maximum iterations
- `tolerance`: Convergence tolerance
- `use_2block`: Use 2-block structure
- `adaptive_alpha`: Enable adaptive step size

#### MPI Parameters
- `enable`: Enable MPI distribution
- `async_communication`: Use asynchronous communication
- `buffer_size_kb`: MPI buffer size in KB
- `collective_operations`: Use collective ops (AllReduce)
- `load_balancing`: Load balancing strategy ("block", "cyclic", "dynamic")

### Command-line Options

```bash
scripts/run_mps_mpi.py [options]

Options:
  --config CONFIG [CONFIG ...]  Configuration file(s) to load
  --override KEY=VALUE         Override configuration parameters
  --output-dir DIR            Output directory for results
  --no-save                   Don't save results
  --profile                   Enable performance profiling
  --benchmark                 Run in benchmark mode
  --validate-only            Only validate configuration
  --verbose                   Enable verbose output
  --quiet                    Suppress output except errors
  --plot                     Generate plots after completion
```

## MPI Performance Tuning

### Process Count Selection

Choose process count based on:
- **Network size**: ~10-20 sensors per process
- **Hardware**: Don't exceed physical cores
- **Communication overhead**: More processes = more overhead

Example scaling:
```
20 sensors  -> 2-4 processes
50 sensors  -> 4-8 processes
100 sensors -> 8-16 processes
200 sensors -> 16-32 processes
```

### Configuration Tuning

For best MPI performance:

```yaml
mpi:
  async_communication: true    # Overlap computation/communication
  buffer_size_kb: 2048        # Larger for big networks
  collective_operations: true  # Efficient global ops
  load_balancing: "dynamic"   # For heterogeneous systems

algorithm:
  gamma: 0.995  # Slightly conservative for distributed
  alpha: 2.0    # Smaller than single-process
```

### Benchmarking

Run benchmarks to find optimal settings:

```bash
# Test different process counts
for n in 2 4 8 16; do
  mpirun -n $n python scripts/run_mps_mpi.py \
    --config configs/mpi/mpi_benchmark.yaml \
    --benchmark
done
```

## Troubleshooting

### Common Issues

#### 1. MPI Import Error
```
ImportError: No module named 'mpi4py'
```
**Solution**: Install mpi4py:
```bash
pip install mpi4py
```

#### 2. MPI Runtime Error
```
mpirun: command not found
```
**Solution**: Install MPI implementation:
```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt-get install openmpi-bin openmpi-common

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
```

#### 3. Configuration Not Found
```
FileNotFoundError: configs/default.yaml
```
**Solution**: Run from project root:
```bash
cd /path/to/DecentralizedLocale
mpirun -n 4 python scripts/run_mps_mpi.py --config configs/default.yaml
```

#### 4. Memory Issues with Large Networks
```
MemoryError: Unable to allocate array
```
**Solution**: Use more MPI processes or reduce network size:
```yaml
algorithm:
  parallel_proximal: true  # Reduce memory usage
  
mpi:
  enable: true
  buffer_size_kb: 512  # Smaller buffers
```

#### 5. Convergence Issues
```
Warning: Algorithm did not converge
```
**Solution**: Adjust algorithm parameters:
```yaml
algorithm:
  gamma: 0.999  # More conservative
  alpha: 1.0    # Smaller step size
  max_iterations: 2000  # More iterations
```

### Performance Debugging

Enable detailed timing:
```bash
mpirun -n 4 python scripts/run_mps_mpi.py \
  --config configs/default.yaml \
  --profile --verbose
```

Check MPI communication patterns:
```bash
export OMPI_MCA_btl_base_verbose=100
mpirun -n 4 python scripts/run_mps_mpi.py --config configs/mpi/mpi_small.yaml
```

### Getting Help

1. Check configuration is valid:
```bash
python scripts/run_mps_mpi.py --config configs/default.yaml --validate-only
```

2. Run simple test:
```bash
python test_mpi_simple.py
```

3. Enable verbose output:
```bash
mpirun -n 2 python scripts/run_mps_mpi.py \
  --config configs/mpi/mpi_small.yaml --verbose
```

## Examples

### Example 1: High Accuracy Localization

```bash
# For maximum accuracy with carrier phase
python scripts/run_mps_mpi.py \
  --config configs/high_accuracy.yaml \
  --output-dir results/high_accuracy/
```

### Example 2: Large Network with MPI

```bash
# 200 sensors with 16 MPI processes
mpirun -n 16 python scripts/run_mps_mpi.py \
  --config configs/large_network.yaml \
  --override mpi.enable=true \
  --output-dir results/large_network/
```

### Example 3: Noisy Environment Testing

```bash
# Test with 15% noise and 10% outliers
python scripts/run_mps_mpi.py \
  --config configs/noisy_measurements.yaml \
  --plot  # Generate plots to analyze noise effects
```

### Example 4: Custom Configuration

Create `my_config.yaml`:
```yaml
extends: configs/default.yaml

network:
  n_sensors: 60
  n_anchors: 9
  communication_range: 0.25

measurements:
  noise_factor: 0.02
  carrier_phase: true  # Millimeter accuracy

algorithm:
  gamma: 0.998
  alpha: 8.0
  max_iterations: 800

output:
  output_dir: "results/my_experiment/"
```

Run:
```bash
mpirun -n 6 python scripts/run_mps_mpi.py --config my_config.yaml
```

## Performance Results

Typical performance with MPS algorithm:

| Network Size | Processes | Runtime | Relative Error | RMSE |
|-------------|-----------|---------|----------------|------|
| 20 sensors | 1 | 3s | 0.12 | 0.08 |
| 20 sensors | 4 | 1.5s | 0.12 | 0.08 |
| 50 sensors | 1 | 15s | 0.14 | 0.10 |
| 50 sensors | 8 | 5s | 0.14 | 0.10 |
| 100 sensors | 1 | 60s | 0.16 | 0.12 |
| 100 sensors | 16 | 15s | 0.16 | 0.12 |
| 200 sensors | 16 | 45s | 0.18 | 0.15 |
| 200 sensors | 32 | 25s | 0.18 | 0.15 |

## Summary

The MPS algorithm implementation now provides:

1. **Flexible Configuration**: YAML-based with inheritance and overrides
2. **Scalable Execution**: MPI distribution for large networks
3. **Easy Usage**: Simple command-line interface
4. **Performance Tools**: Benchmarking and profiling capabilities
5. **Robustness**: Handles noise, outliers, and large networks

For production use, start with the provided configurations and tune based on your specific requirements. Use MPI for networks with >50 sensors for significant speedup.