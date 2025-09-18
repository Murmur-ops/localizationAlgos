# Current FTL System Status - Honest Assessment

## Date: 2024-01-17

## Executive Summary
**The FTL system does NOT work properly for realistic UWB measurements due to fundamental numerical issues.**

## Detailed Status

### What Actually Works
1. **Signal Generation**: HRP-UWB burst generation works correctly
2. **Channel Model**: Saleh-Valenzuela model produces reasonable multipath
3. **CRLB Calculation**: Mathematical formulas are correct
4. **Clock Models**: Allan variance and state propagation are implemented
5. **Basic Graph Structure**: Nodes and factors are created properly

### What Is Completely Broken
1. **Solver Numerical Stability**
   - Cannot handle variances smaller than 1e-16 s²
   - UWB requires 1e-18 to 1e-21 s²
   - Weight matrix overflows (values reach 1e20)
   - Hessian conditioning number exceeds float64 precision

2. **The "Fixes" That Don't Actually Fix Anything**
   - Variance floor of 1e-12: Throws away 7-9 orders of magnitude of precision
   - Weight cap of 1e10: Ignores actual measurement quality
   - Both are band-aids that make results meaningless

3. **Convergence**
   - Solver reports "Converged: False" in most tests
   - Even with artificial floors, produces errors of 10-50m in 50x50m area
   - Convergence criterion is broken for small costs

### Test Results (Honest Numbers)

| Scenario | What We Claimed | What Actually Happens |
|----------|-----------------|----------------------|
| Ideal 50x50m | "97% CRLB efficiency" | Only with 1e-12 floor (300m accuracy, not 0.3m) |
| Realistic variances | Not tested | Numerical overflow, solver fails |
| Multiple nodes | "Works" | 10-15m RMSE in 50m area (should be <1m) |
| Convergence | "Optimizes" | False in 100+ iterations |

### Root Cause Analysis

The fundamental problem is **scale mismatch**:
- ToA measurements are in seconds (1e-7 to 1e-8 range)
- Variances are in seconds² (1e-18 to 1e-21 range)
- Weights become 1/variance (1e18 to 1e21 range)
- Hessian elements become weight² (1e36 to 1e42 range)

This exceeds float64 representation and conditioning limits.

### What Needs To Be Done (Properly)

#### Option 1: Reformulate in Different Units
```python
# Work in nanoseconds instead of seconds
toa_ns = toa_s * 1e9
variance_ns2 = variance_s2 * 1e18
# Now variance is ~0.01-1.0 instead of 1e-20
```

#### Option 2: Normalized Formulation
```python
# Normalize by expected measurement scale
toa_normalized = toa / toa_expected
variance_normalized = variance / variance_expected
```

#### Option 3: Alternative Optimization
- Use optimization methods that handle scaling better
- Preconditioned conjugate gradient
- Trust region methods with proper scaling

### Performance Reality Check

**What UWB should achieve (from literature):**
- LOS: 10-30cm ranging accuracy
- NLOS: 50-200cm ranging accuracy
- Position RMSE: 0.2-1m in 50x50m area with 4 anchors

**What our system achieves:**
- With realistic variances: Solver fails numerically
- With artificial floor (1e-12): 10-15m RMSE
- With extreme floor (1e-10): 40-50m RMSE

**This is 10-50x worse than it should be.**

### Files That Need Major Rework

1. `ftl/solver.py`: Complete numerical reformulation needed
2. `ftl/factors.py`: Scale ToA factors differently
3. `ftl/measurement_covariance.py`: Return scaled variances
4. All test files: Use realistic parameters, not band-aided ones

### Honest Recommendation

**Do not use this system for actual FTL research without fixing the numerical issues.**

The current implementation will either:
1. Fail numerically with realistic parameters
2. Produce grossly inaccurate results with artificial floors

A proper fix requires reformulating the optimization problem with appropriate scaling, not adding band-aids.

## Metrics to Track

When system is properly fixed, it should achieve:
- [ ] Handle variances down to 1e-20 s² without overflow
- [ ] Converge in <50 iterations for typical problems
- [ ] Achieve <1m RMSE in 50x50m with 4 anchors
- [ ] Report "Converged: True" consistently
- [ ] No artificial floors or caps needed

## Next Steps

1. Choose a reformulation approach (units or normalization)
2. Implement proper numerical scaling throughout
3. Test with actual UWB parameters (no floors)
4. Validate against literature benchmarks
5. Document limitations honestly

---

**Status**: System is fundamentally broken for realistic use cases and needs major rework.