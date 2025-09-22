# Action Plan for Technical Fixes

Based on ChatGPT's critique, here are the required fixes prioritized by impact.

## Critical Fixes (Must Fix)

### 1. Add Clock Drift Term to ToA Factor
**File:** `ftl/factors_scaled.py`
**Issue:** Missing drift contribution in range model
**Fix:**
```python
def residual(self, xi, xj, delta_t=1.0):
    # ... existing code ...
    di_ppb = xi[3]  # drift in ppb
    dj_ppb = xj[3]

    # Add drift contribution
    drift_contribution = (dj_ppb - di_ppb) * delta_t * self.c * 1e-9
    predicted_range = geometric_range + clock_contribution + drift_contribution
```

### 2. Fix Whitening to Proper SRIF
**File:** `ftl/factors_scaled.py`
**Issue:** Current "whitening" just divides by std, not true square-root information
**Fix:**
```python
class ScaledFactor:
    def __init__(self, covariance):
        if np.isscalar(covariance):
            self.L = 1.0 / np.sqrt(covariance)  # Square root of information
        else:
            # For matrix covariance
            self.L = np.linalg.cholesky(np.linalg.inv(covariance))

    def whiten(self, residual):
        return self.L @ residual  # Proper whitening
```

### 3. Document Observability Constraints
**File:** New file `OBSERVABILITY_AND_GAUGES.md`
**Content:**
- Absolute time/frequency unobservable without disciplined anchor
- SE(2) gauge fixing requirements
- Minimum anchor requirements for observability

### 4. Verify CRLB Formula Consistency
**Files:** All files using CRLB
**Action:**
- Audit all CRLB uses
- Ensure using β_rms form everywhere
- Remove any f_c terms if found
- Document units clearly

## Important Fixes (Should Fix)

### 5. Clarify Performance Claims
**File:** `30_NODE_PERFORMANCE_REPORT.md` and others
**Action:**
- Qualify ALL performance numbers with:
  - SNR level
  - Network configuration (nodes, anchors, connectivity)
  - Ideal vs realistic conditions
  - Consensus vs centralized

### 6. Remove Misleading Comments
**File:** Various
**Action:**
- Remove "J^T J is wrong" comment
- Fix any actual shape bugs
- Document that for scalar residual, J^T J = outer(J,J)

### 7. Standardize Units Documentation
**Action:**
- Create unit convention document
- Meters for all distances
- Nanoseconds for time
- ppb/ppm for frequency
- Document conversions explicitly

## Nice to Have (Future Work)

### 8. Add Robust Weighting
**File:** `ftl/consensus/consensus_node.py`
**Enhancement:**
```python
def huber_weight(r_whitened, delta=1.0):
    """Huber M-estimator weight"""
    if abs(r_whitened) <= delta:
        return 1.0
    else:
        return delta / abs(r_whitened)
```

### 9. Add Per-Iteration Instrumentation
**Enhancement:**
- Log gradient norm
- Log consensus residuals
- Log LM lambda
- Export to CSV for analysis

### 10. Add GDOP Calculation
**Enhancement:**
- Compute geometric dilution of precision per node
- Use for performance prediction
- Include in reports

## Implementation Priority

### Phase 1 (Immediate)
1. Add drift term to ToA factor
2. Document observability constraints
3. Clarify performance claims

### Phase 2 (Next Sprint)
4. Implement proper SRIF whitening
5. Standardize units documentation
6. Remove misleading comments

### Phase 3 (Future)
7. Add robust weighting
8. Per-iteration instrumentation
9. GDOP calculation

## Testing Requirements

For each fix:
1. Unit test the changed component
2. Integration test with full system
3. Compare performance before/after
4. Document any performance changes

## Acceptance Criteria

Per ChatGPT's suggestion:
- Consensus RMSE within 1.2× centralized at various SNRs
- Anchor-starvation test: interior nodes ≤ 1.5× baseline RMSE
- Whitened residuals ~ N(0,1) via Q-Q plot

---

## Summary

ChatGPT identified **10 technical issues**, of which:
- **4 are critical** (drift term, whitening, observability, CRLB)
- **3 are important** (performance claims, comments, units)
- **3 are enhancements** (robustness, instrumentation, GDOP)

Most issues are valid and require fixing for production readiness.