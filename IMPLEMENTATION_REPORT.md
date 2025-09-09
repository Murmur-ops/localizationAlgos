# Comprehensive MPS Algorithm Implementation Report

## Executive Summary

This report documents the extensive effort to implement the Matrix-Parametrized Proximal Splitting (MPS) algorithm from the paper "Decentralized Sensor Network Localization: A Matrix-Parametrized Proximal Splittings Approach" (arXiv:2503.13403v1). While the ADMM baseline performs as expected (~39% relative error), the MPS algorithm implementation achieves 54-108% error versus the paper's reported 5-10%, indicating fundamental implementation challenges.

## Implementation Overview

### Files Created

1. **Core Implementations**:
   - `src/core/mps_core/algorithm.py` - Main algorithm (simplified version)
   - `src/core/mps_core/algorithm_v2.py` - Faithful Algorithm 1 attempt
   - `src/core/mps_core/algorithm_corrected.py` - With numerical stability fixes
   - `src/core/mps_core/algorithm_correct.py` - Improved stability version
   - `src/core/mps_core/algorithm_stable.py` - Simplified stable version
   - `src/core/mps_core/algorithm_oars.py` - OARS framework-based
   - `src/core/mps_core/algorithm_final.py` - Final best attempt

2. **Experimental Scripts**:
   - `scripts/section3_numerical_experiments.py` - Full Section 3 experiments
   - `scripts/extract_mps_results.py` - Results extraction without plotting
   - Various test scripts for validation

3. **Documentation**:
   - `MPS_IMPLEMENTATION_REPORT.md` - Technical challenges documentation
   - `IMPLEMENTATION_REPORT.md` - This comprehensive report

## Algorithm 1 from Paper

The paper specifies Algorithm 1 as:
```
1. Initialize: v⁰ ∈ H^p with Σᵢ vᵢ⁰ = 0
2. Repeat:
   - Solve x^k = J_αF(v^k + Lx^k) for x^k
   - v^(k+1) = v^k - γWx^k
```

Where:
- Z, W ∈ S⁺ᵖ (positive semidefinite matrices)
- L is lower triangular such that Z = 2I - L - L^T
- v maintains zero-sum constraint
- J_αF is the resolvent (proximal operator)

## Implementation Approaches

### 1. Initial Simplified Implementation
**File**: `algorithm.py`
**Approach**: ADMM-like structure with X, Y, U variables
**Result**: 65% error
**Issues**: 
- Missing consensus variables v
- No implicit equation solver
- Wrong update structure

### 2. Faithful Algorithm 1 Implementation
**Files**: `algorithm_v2.py`, `algorithm_corrected.py`
**Approach**: Exact implementation of paper's Algorithm 1
```python
# Step 1: Solve implicit equation
x = solve_implicit_equation(v)  # x = J_αF(v + Lx)

# Step 2: Update consensus
v = v - gamma * W @ x
```
**Result**: NaN values, numerical divergence
**Issues**:
- Implicit fixed-point equation unstable
- Matrix constraint Z = 2I - L - L^T difficult to satisfy
- Zero-sum constraint causes drift

### 3. Stable Simplified Version
**File**: `algorithm_stable.py`
**Approach**: Alternating proximal and consensus steps
```python
positions_prox = apply_distance_constraints(positions)
positions_consensus = apply_consensus(positions)
positions = gamma * positions_prox + (1-gamma) * positions_consensus
```
**Result**: 88-130% error
**Issues**: Lost essential mathematical structure

### 4. OARS Framework Integration
**Files**: `algorithm_oars.py`, `algorithm_final.py`
**Approach**: Using OARS library structure from https://github.com/peterbarkley/oars.git
```python
# OARS pattern from serial.py
for block_idx in range(n_blocks):
    y = all_v[i] - sum(Z[i,j]*all_x[j] for j < i)
    all_x[i] = resolvent.prox(y, alpha)

for block_idx in range(n_blocks):
    wx[i] = sum(W[i,j]*all_x[j] for j in range(n))
    all_v[i] = all_v[i] - gamma*wx[i]
```
**Result**: 54-108% error (best achieved)
**Progress**: Proper matrix structure but still not matching paper

## Key Technical Discoveries

### 1. Matrix Structure Requirements
- **Z Matrix**: Must be structured for implicit equation y = v - Z*x
- **W Matrix**: Controls consensus updates
- **L Matrix**: Lower triangular with Z = 2I - L - L^T
- **Challenge**: Satisfying all constraints while maintaining numerical stability

### 2. Implicit Fixed-Point Equation
The core challenge is solving:
```
x^k = J_αF(v^k + Lx^k)
```
This requires:
- Fixed-point iteration within each algorithm step
- Careful relaxation parameters for stability
- Computational overhead that may not be documented

### 3. Block Structure
The paper uses a "lifted" formulation with:
- Position variables X ∈ R^(n×d)
- Gram matrix variables for PSD constraints
- Complex block coupling not fully specified

### 4. OARS Insights
The OARS library provides:
- Correct sequential resolvent application
- Prebuilt matrix patterns (getTwoBlockSimilar, getFull)
- Proper consensus variable updates

However, OARS is for general convex optimization, not specifically sensor localization.

## Performance Comparison

| Algorithm | Expected Error | Achieved Error | Status |
|-----------|---------------|----------------|--------|
| ADMM | ~40% | 39% | ✓ Working |
| MPS | 5-10% | 54-108% | ✗ Not matching |

### ADMM Success
- Correctly implements decentralized consensus
- Achieves expected ~39% error from paper
- Stable convergence in 500 iterations

### MPS Challenges
- Converges too quickly (7-20 iterations)
- High error (54-108% vs expected 5-10%)
- Numerical instability in faithful implementation

## Missing Components Analysis

### 1. Undocumented Implementation Details
The paper likely has:
- Specific matrix construction for sensor networks
- Preconditioning or scaling strategies
- Additional constraints or regularization

### 2. PSD Constraint Implementation
The lifted formulation with Gram matrices for PSD constraints is mentioned but not detailed:
- How to construct the lifted variables
- How to project onto PSD cone efficiently
- Integration with distance constraints

### 3. Network-Specific Optimizations
The general OARS framework may need adaptations for sensor networks:
- Topology-aware matrix construction
- Distance-specific proximal operators
- Anchor constraint prioritization

## Recommendations

### 1. Short-term (Production Use)
- **Use ADMM**: Working correctly with expected performance
- **Document limitations**: MPS requires further research
- **Monitor convergence**: ADMM is stable and reliable

### 2. Medium-term (Research)
- **Contact authors**: Request implementation details or code
- **Study related papers**: Look for similar MPS applications
- **Experiment with parameters**: Systematic grid search for α, γ

### 3. Long-term (Full Implementation)
- **Theoretical analysis**: Deeper understanding of lifted formulation
- **Custom matrix design**: Network-specific Z and W construction
- **Numerical optimization**: Preconditioning for implicit equation

## Conclusion

Despite extensive implementation efforts including:
- 7 different algorithm versions
- Integration with OARS framework
- Multiple matrix construction approaches
- Various numerical stability techniques

The MPS algorithm does not achieve the paper's reported 5-10% error. The implementation achieves 54-108% error at best, suggesting:

1. **Critical details are missing** from the paper's algorithm description
2. **The lifted formulation** requires specific structure not documented
3. **Network-specific optimizations** are needed beyond general OARS

The ADMM implementation works correctly and should be used for production. The MPS algorithm requires either:
- Additional information from the paper's authors
- Significant additional research into the mathematical structure
- Access to the original implementation for comparison

## Technical Artifacts

All code has been thoroughly documented with:
- Clear docstrings and comments
- Test functions for validation
- Performance metrics tracking
- Numerical stability safeguards

The implementation represents a comprehensive effort to recreate the paper's results and provides a solid foundation for future work once additional details become available.