# Getting Started with Decentralized Sensor Network Localization

Welcome! This package contains a complete implementation of the distributed sensor network localization algorithm from Barkley & Bassett (2025).

## 📋 Quick Start (5 minutes)

### 1. Check Your System
```bash
python check_windows_compatibility.py
```

### 2. Install Dependencies
```bash
pip install numpy scipy matplotlib
```

### 3. Run Your First Simulation
```bash
# For small networks (no MPI needed)
python snl_threaded_standalone.py
```

## 📁 What's Included

| File | Description | When to Use |
|------|-------------|-------------|
| `snl_threaded_standalone.py` | Single-machine version | Testing, <50 sensors |
| `snl_main_full.py` | Complete reference implementation | Understanding the algorithm |
| `snl_mpi_optimized.py` | Production MPI version | Large networks, real deployments |
| `generate_figures.py` | Visualization tools | Analyzing results |
| `crlb_analysis.py` | Performance comparison | Evaluating accuracy |

## 🚀 Running Different Configurations

### Option 1: Quick Test (No Setup Required)
```bash
python snl_threaded_standalone.py
```
This runs a 30-sensor network with 6 anchors. Expected output:
```
Starting distributed sensor network localization...
Generated network with 30 sensors and 6 anchors
Average connectivity: 5.2 neighbors per sensor
Running MPS algorithm...
Iteration 50: Error = 0.0031, Objective = 0.245
Converged! Final RMSE: 0.0031
```

### Option 2: Custom Network Size
Edit the parameters at the top of any script:
```python
# In snl_threaded_standalone.py, lines 850-860
problem = SNLProblem(
    n_sensors=50,          # Increase sensors
    n_anchors=10,          # More anchors for better accuracy
    communication_range=0.4,  # Larger range = more connections
    noise_factor=0.10      # 10% measurement noise
)
```

### Option 3: Production MPI Version
First install MPI (see Platform-Specific Setup below), then:
```bash
# Run with 4 processes
mpirun -np 4 python snl_mpi_optimized.py

# Custom configuration
mpirun -np 8 python snl_mpi_optimized.py --sensors 500 --anchors 50
```

## 💻 Platform-Specific Setup

### Windows
```bash
# Option 1: Use threading version (no setup)
python snl_threaded_standalone.py

# Option 2: Install MPI for full performance
# Download MS-MPI from Microsoft, then:
pip install mpi4py
mpiexec -n 4 python snl_mpi_optimized.py
```

### macOS
```bash
# Install MPI
brew install mpich
pip install mpi4py

# Run MPI version
mpirun -np 4 python snl_mpi_optimized.py
```

### Linux
```bash
# Install MPI
sudo apt-get install mpich
pip install mpi4py

# Run MPI version
mpirun -np 4 python snl_mpi_optimized.py
```

## 📊 Generating Visualizations

```bash
# Generate all figures
python generate_figures.py

# Analyze CRLB performance
python crlb_analysis.py
```

This creates:
- `sensor_network_topology.png` - Network structure
- `algorithm_convergence.png` - Convergence comparison
- `crlb_comparison.png` - Performance vs theoretical limit
- `localization_results.png` - Before/after positions

## 🎯 Understanding the Results

### Key Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **CRLB Efficiency**: How close to theoretical optimal (80-85% is excellent)
- **Iterations**: Typically converges in 50-100 iterations

### Expected Performance
| Noise Level | Expected RMSE | CRLB Efficiency |
|-------------|---------------|-----------------|
| 1%          | ~0.6mm        | 85%             |
| 5%          | ~3mm          | 83%             |
| 10%         | ~6mm          | 82%             |
| 20%         | ~12mm         | 80%             |

## 🔧 Common Modifications

### 1. Change Network Topology
```python
# Grid layout instead of random
n = int(np.sqrt(n_sensors))
x = np.linspace(0.1, 0.9, n)
y = np.linspace(0.1, 0.9, n)
xx, yy = np.meshgrid(x, y)
true_positions = np.column_stack([xx.ravel(), yy.ravel()])
```

### 2. Adjust Algorithm Parameters
```python
problem = SNLProblem(
    gamma=0.99,        # Stability (0.9-0.999)
    alpha_mps=20.0,    # Convergence speed (5-50)
    tol=1e-5,          # Precision (1e-3 to 1e-6)
    max_iter=200       # Maximum iterations
)
```

### 3. Different Distance Measurements
```python
# Add custom noise model
def custom_noise(true_distance):
    # Example: 5% + 0.1m constant error
    return true_distance * (1 + 0.05 * np.random.randn()) + 0.1
```

## 📈 Performance Tips

1. **For Best Accuracy**: Use more anchors (10-20% of total sensors)
2. **For Faster Convergence**: Increase connectivity (larger communication_range)
3. **For Robustness**: Use gamma = 0.999 (very stable)
4. **For Speed**: Use fewer iterations with early termination

## 🐛 Troubleshooting

### "Import Error: mpi4py"
- Solution: Use `snl_threaded_standalone.py` or install MPI

### "Timeout" or Very Slow
- Threading version is slow for >50 sensors
- Use MPI version for better performance

### Poor Accuracy
- Check connectivity: Average should be >4 neighbors
- Ensure anchors are well-distributed (corners + center)
- Verify noise levels are realistic

## 📚 Next Steps

1. **Understand the Algorithm**: Read `snl_main_full.py` with detailed comments
2. **Visualize Your Results**: Use `generate_figures.py`
3. **Scale Up**: Try MPI version with larger networks
4. **Real Hardware**: See main documentation for hardware integration

## 💡 Example: Complete Workflow

```bash
# 1. Test installation
python check_windows_compatibility.py

# 2. Run small test
python snl_threaded_standalone.py

# 3. Generate visualizations
python generate_figures.py

# 4. Try larger network (if MPI installed)
mpirun -np 4 python snl_mpi_optimized.py

# 5. Analyze performance
python crlb_analysis.py
```

## Need Help?

- Implementation details: See comments in source files
- Algorithm theory: Check the paper references
- Performance issues: Try MPI version
- Bug reports: Include full error message and system info

Happy localizing! 🎯