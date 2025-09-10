# MPS Algorithm Convergence Analysis Report

## Executive Summary
Fixed broadcasting error and implemented multiple convergence improvements for the MPS (Matrix-Parametrized Proximal Splitting) algorithm. Added seed-based reproducibility, smoothing mechanisms, and escape strategies. While the algorithm finds good solutions (~16.6% error) early, it consistently gets stuck at local minima (~79% error).

## Issues Addressed

### 1. Broadcasting Error (FIXED ✅)
**Problem**: Shape mismatch (4,4) into shape (2,) when handling lifted matrix variables  
**Solution**: Properly extract position vectors from lifted matrices in `_prox_objective` method
```python
# Handle lifted matrix variables correctly
if v.ndim == 2 and v.shape[0] > self.d:
    position = v[:self.d, -1] if v.shape[1] > self.d else v[:self.d, 0]
```

### 2. Noisy Convergence (IMPROVED ✅)
**Problem**: Oscillating convergence plots made it hard to assess progress  
**Solution**: Implemented exponential moving average (EMA) smoothing with α=0.1
- Smoothed objectives and errors for cleaner visualization
- Adaptive parameter tuning to reduce oscillations
- Momentum-based updates for dampening

### 3. Reproducibility (FIXED ✅)
**Problem**: Results varied between runs, making comparison difficult  
**Solution**: Added fixed seed support (default: 42) in network generation
- Consistent network topology across all tests
- Reproducible error measurements
- Identical results verified across multiple runs

### 4. Local Minima (PARTIALLY ADDRESSED ⚠️)
**Problem**: Algorithm gets stuck at ~79% error despite finding 16.6% solution early  
**Solutions Attempted**:
- Random perturbation (5% noise when stuck for 50 iterations)
- Simulated annealing (temperature-based acceptance of worse solutions)
- Parameter shocking (3x alpha boost for 20 iterations)
- Smart restart (from perturbed best position after 150 iterations stuck)

**Status**: Escape mechanisms implemented but not triggering effectively

## Test Results

### Configuration Comparison (Fixed Seed: 42)

| Configuration | Best Error | Final Error | Iterations | Time | Status |
|--------------|------------|-------------|------------|------|---------|
| Default | 16.61% | 80.34% | 1000 | 5.66s | Stuck in local minimum |
| Stable Convergence | 16.61% | 78.78% | 511 | 3.66s | Converged but stuck |
| Escape Minima | 16.61% | 78.78% | 1500 | 12.76s | Failed to escape |

### Key Findings
1. **Consistent Early Success**: All configurations find ~16.6% error solution in first few iterations
2. **Universal Plateau**: All configurations get stuck at ~78-80% error
3. **Escape Failure**: Despite aggressive escape mechanisms, cannot recover the good solution
4. **Early Convergence**: Stable config converges at iteration 511 (early stopping triggered)

## Files Modified

### Core Algorithm
- `/src/core/mps_core/mps_distributed.py`
  - Fixed broadcasting error in `_prox_objective`
  - Added smoothing (EMA) for objectives and errors
  - Implemented adaptive alpha and momentum
  - Added escape mechanisms (perturbation, annealing, shock, restart)

### Visualization
- `/src/core/mps_core/visualization.py`
  - Removed legend from error distribution plot
  - Added support for smoothed vs raw convergence plots

### Configuration
- `/configs/default.yaml` - Added seed parameter
- `/configs/stable_convergence.yaml` - Conservative parameters with smoothing
- `/configs/escape_minima.yaml` - Aggressive escape strategies

### Infrastructure
- `/scripts/run_mps_mpi.py` - Added seed support for reproducibility
- `/src/core/mps_core/mps_full_algorithm.py` - Added seed parameter to network generation

## Convergence Behavior Analysis

### Phase 1: Initial Success (Iterations 0-10)
- Algorithm quickly finds good solution (~16.6% error)
- Likely due to good initialization or lucky proximal evaluation

### Phase 2: Degradation (Iterations 10-50)
- Solution quality rapidly degrades to ~80% error
- Consensus updates may be pulling away from good solution
- High gamma (0.98-0.999) emphasizes proximal over consensus

### Phase 3: Plateau (Iterations 50+)
- Algorithm stuck in local minimum
- Small improvements (<0.1% per 100 iterations)
- Escape mechanisms fail to trigger or are ineffective

## Recommendations

### Short-term
1. **Debug Escape Triggers**: Add logging to verify when escape mechanisms activate
2. **Tune Thresholds**: Current stuck_threshold (50) may be too high
3. **Memory Mechanism**: Store top-K best solutions, not just single best

### Long-term
1. **Multi-start Strategy**: Run multiple instances with different initializations
2. **Hybrid Approach**: Use different algorithms for exploration vs exploitation
3. **Problem-specific Tuning**: Parameters may need adjustment for this network topology

## Technical Details

### Broadcasting Fix
The lifted matrix structure in MPS uses matrices of dimension (d+|N_i|+1) × (d+|N_i|+1) where:
- d = spatial dimension (2 for 2D)
- |N_i| = number of neighbors for sensor i
- Extra dimensions for consensus variables

The fix properly extracts the d-dimensional position vector from these larger matrices.

### Smoothing Implementation
Exponential Moving Average with α=0.1:
```python
smoothed_value = α * current + (1-α) * previous_smoothed
```
Provides ~10 iteration lag but much cleaner convergence visualization.

### Escape Mechanisms
1. **Perturbation**: Add Gaussian noise N(0, 0.05) to positions
2. **Annealing**: Accept worse solutions with probability exp(-ΔE/T)
3. **Parameter Shock**: Temporarily multiply alpha by 3.0
4. **Smart Restart**: Return to perturbed best position

## Conclusion
Successfully fixed critical bugs and improved convergence stability. The algorithm consistently finds good solutions early but cannot maintain them. The escape mechanisms need further tuning or a different approach to overcome the strong local minimum at ~79% error. The reproducible testing framework (seed=42) enables systematic comparison of future improvements.