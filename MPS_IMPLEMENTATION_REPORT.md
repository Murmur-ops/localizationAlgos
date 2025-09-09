# MPS Algorithm Implementation Report

## Summary
This report documents the challenges encountered while implementing the Matrix-Parametrized Proximal Splitting (MPS) algorithm from the paper "Decentralized Sensor Network Localization: A Matrix-Parametrized Proximal Splittings Approach" (arXiv:2503.13403v1).

## Current Status

### What Works
- **ADMM Implementation**: Successfully achieves ~39% relative error as reported in the paper
- **Network Generation**: Correctly generates sensor networks with noisy distance measurements
- **Basic MPS Structure**: Implemented the core components (proximal operators, consensus matrices)

### What Doesn't Work
- **MPS Convergence**: Current implementation achieves 65-120% relative error vs expected 5-10%
- **Algorithm 1**: The implicit fixed-point equation `x^k = J_αF(v^k + Lx^k)` causes numerical instability

## Implementation Attempts

### 1. Initial Implementation (`algorithm.py`)
- **Approach**: Simplified ADMM-like structure with X, Y, U variables
- **Result**: 65% error - not following paper's Algorithm 1
- **Issue**: Missing consensus variables v and proper matrix structure

### 2. Faithful Algorithm 1 (`algorithm_v2.py`, `algorithm_corrected.py`)
- **Approach**: Exact implementation of paper's Algorithm 1
- **Result**: NaN values, numerical divergence
- **Issues**:
  - Implicit fixed-point solver unstable
  - Matrix relationships Z = 2I - L - L^T difficult to satisfy
  - Zero-sum constraint on v causes drift

### 3. Stable Simplified Version (`algorithm_stable.py`)
- **Approach**: Alternating proximal steps and consensus averaging
- **Result**: 88-130% error, quick convergence but poor accuracy
- **Issue**: Lost essential mathematical structure from paper

## Key Technical Challenges

### 1. Implicit Fixed-Point Equation
The paper requires solving:
```
x^k = J_αF(v^k + Lx^k)
```
This implicit equation requires iterative solving at each step, causing:
- Computational overhead
- Numerical instability
- Accumulation of errors

### 2. Matrix Structure
The paper specifies:
- Z, W matrices must be positive semidefinite
- L must satisfy: Z = 2I - L - L^T
- All must preserve consensus properties

Creating matrices that simultaneously satisfy all constraints proved challenging.

### 3. Lifted Variable Structure
The paper uses a "lifted" formulation with:
- Position variables X
- Gram matrix variables for PSD constraints
- Complex block structure

The simplified 2-block structure we implemented may be insufficient.

### 4. Parameter Sensitivity
The algorithm is highly sensitive to:
- α (proximal step size): Paper uses 10.0
- γ (consensus mixing): Paper uses 0.999
- Network scale: Affects numerical conditioning

## Comparison with ADMM

| Metric | ADMM (Working) | MPS (Current) | MPS (Expected) |
|--------|---------------|---------------|----------------|
| Relative Error | 39% | 65-120% | 5-10% |
| Convergence | Stable | Unstable | Should be stable |
| Iterations | 500 | 20-500 | ~100 |
| Implementation | Standard | Simplified | Full Algorithm 1 |

## Recommendations

1. **Further Research Needed**:
   - The paper may have implementation details not fully documented
   - The "lifted" variable structure may be more complex than described
   - Numerical preconditioning may be required

2. **Practical Approach**:
   - Use ADMM for production (working correctly)
   - Continue MPS research separately
   - Consider contacting paper authors for clarification

3. **Next Steps**:
   - Review paper's theoretical foundations more carefully
   - Examine if there are published implementations
   - Consider alternative proximal splitting methods

## Conclusion

While we successfully implemented the network generation and ADMM baseline, the MPS Algorithm 1 requires additional work to achieve the paper's reported performance. The simplified version converges but doesn't achieve the expected accuracy, suggesting fundamental mathematical structure is missing from our implementation.

The core challenge is balancing numerical stability with mathematical fidelity to the paper's algorithm. The implicit fixed-point equation and complex matrix relationships make direct implementation challenging without additional implementation details from the authors.