# Quick Reference Card - Decentralized SNL

## 🚀 Instant Start
```bash
pip install numpy scipy matplotlib
python simple_example.py
```

## 📊 Key Commands

| Task | Command |
|------|---------|
| Run example | `python simple_example.py` |
| Small network | `python snl_threaded_standalone.py` |
| Large network | `mpirun -np 4 python snl_mpi_optimized.py` |
| Generate figures | `python generate_figures.py` |
| Check system | `python check_windows_compatibility.py` |

## 🎛️ Key Parameters

```python
SNLProblem(
    n_sensors=30,        # Sensors to find
    n_anchors=6,         # Known positions  
    communication_range=0.3,  # Radio range
    noise_factor=0.05,   # 5% noise
    gamma=0.999,         # Stability (0.9-0.999)
    alpha_mps=10.0       # Speed (5-50)
)
```

## 📈 Performance Targets

- **Efficiency**: 80-85% of theoretical optimal
- **Convergence**: 50-100 iterations typical
- **Accuracy**: RMSE ≈ noise_factor × range

## 🔧 Quick Modifications

**More sensors:**
```python
problem.n_sensors = 100
```

**Better accuracy:**
```python
problem.n_anchors = 20  # More anchors
problem.gamma = 0.999   # More stable
```

**Faster convergence:**
```python
problem.alpha_mps = 20.0  # Higher speed
problem.tol = 1e-3       # Lower precision
```

## 💡 Tips

1. **Anchors**: Use 10-20% of total sensors
2. **Connectivity**: Aim for 4+ neighbors average
3. **Placement**: Put anchors at corners + center
4. **MPI**: Use for networks >50 sensors

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| "Too slow" | Use MPI version |
| "Poor accuracy" | Add more anchors |
| "Won't converge" | Increase gamma to 0.999 |
| "Import error" | Run `pip install -r requirements.txt` |

---
*80-85% CRLB efficiency • Scales to 1000+ sensors • Fully distributed*