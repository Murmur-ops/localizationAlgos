# Observability and Gauge Constraints in FTL Localization

## Executive Summary

This document describes the fundamental observability constraints and gauge freedoms inherent in the FTL (Frequency-Time-Localization) system. Understanding these constraints is critical for proper system initialization and interpreting estimation results.

---

## 1. Unobservable States

### 1.1 Absolute Time Reference
**Constraint**: The absolute time offset of the entire network is unobservable.

**Mathematical Formulation**:
- If all clock biases are shifted by constant Δt: `b_i → b_i + Δt` for all nodes i
- Range measurements remain unchanged: `d_ij = ||p_i - p_j|| + c(b_j - b_i)`
- The term `c(b_j - b_i)` is invariant under global time shift

**Implication**: At least one node must have its clock bias fixed (typically to zero) to establish a time reference.

### 1.2 Absolute Frequency Reference
**Constraint**: The absolute frequency offset of the entire network is unobservable.

**Mathematical Formulation**:
- If all CFOs are shifted by constant Δf: `f_i → f_i + Δf` for all nodes i
- CFO difference measurements remain unchanged: `Δf_ij = f_i - f_j`
- Only relative frequency offsets between nodes are observable

**Implication**: At least one node must have its CFO fixed (typically to zero) to establish a frequency reference.

### 1.3 Global Position and Orientation (SE(2) Gauge Freedom)
**Constraint**: The absolute position and orientation of the entire network in 2D space is unobservable.

**Degrees of Freedom**:
1. **Translation** (2 DOF): Global shift in x and y
2. **Rotation** (1 DOF): Global rotation about any point

**Mathematical Formulation**:
For any rigid transformation T ∈ SE(2):
- If all positions transform as: `p_i → T(p_i)`
- Inter-node distances remain unchanged: `||T(p_i) - T(p_j)|| = ||p_i - p_j||`

**Implication**: Need at least 3 constraints to fix SE(2) gauge:
- 2 coordinates for one anchor (fixes translation)
- 1 coordinate for a second anchor (fixes rotation)

---

## 2. Minimum Anchor Requirements

### 2.1 For 2D Positioning

**Theoretical Minimum**: 3 scalar constraints
- Option A: 2 anchors with known positions (4 constraints, 1 redundant)
- Option B: 1 anchor with position + 1 anchor with x-coordinate
- Option C: 1 anchor with position + bearing to another node

**Practical Minimum**: 3 non-collinear anchors
- Provides redundancy for robustness
- Enables outlier detection
- Improves geometric dilution of precision (GDOP)

### 2.2 For Joint Position-Time Estimation

**Minimum Requirements**:
- 3 anchors with known positions (fixes SE(2))
- 1 anchor with known clock bias (fixes time gauge)
- Clock drift can float if measurements are instantaneous

**Recommended Configuration**:
- 4+ anchors with known positions
- 1 disciplined anchor with b=0, d=0 (time reference)
- 1 disciplined anchor with f=0 (frequency reference)

### 2.3 Critical Anchor Geometry Constraint

**IMPORTANT**: Anchors must not be collinear.

**Why Collinearity Fails**:
- Collinear anchors leave one dimension unconstrained
- Creates ambiguity: reflection across the anchor line
- Leads to convergence failure or wrong solutions

**Solution**: Always include at least one anchor off the line formed by others (e.g., center anchor in rectangular layout).

---

## 3. Observable vs Unobservable Subspaces

### 3.1 Observable Quantities
- **Relative positions**: Inter-node distances
- **Relative clock biases**: Time differences between nodes
- **Relative clock drifts**: Drift rate differences
- **Relative CFOs**: Frequency differences
- **Network geometry**: Shape and scale (not position/orientation)

### 3.2 Unobservable Quantities
- **Absolute position**: Where the network is located
- **Absolute orientation**: Which way the network faces
- **Absolute time**: When t=0 occurs
- **Absolute frequency**: Reference carrier frequency
- **Common mode drift**: If all clocks drift together

---

## 4. Disciplined Anchors

### 4.1 Definition
A **disciplined anchor** has its clock synchronized to an external reference (e.g., GPS, atomic clock).

### 4.2 Properties
```python
disciplined_anchor = {
    'position': [x, y],      # Known from survey
    'clock_bias': 0.0,       # Synchronized to reference
    'clock_drift': 0.0,      # Compensated drift
    'cfo': 0.0               # Frequency locked
}
```

### 4.3 Benefits
- Breaks gauge freedoms
- Provides absolute time reference
- Enables network-wide synchronization
- Improves estimation accuracy

---

## 5. Practical Implementation Guidelines

### 5.1 Initialization Strategy

**Step 1**: Fix gauge constraints
```python
# Fix one anchor completely (position + clock)
anchor_0 = {
    'position': [0, 0],
    'is_fixed': True,
    'clock_bias': 0.0,
    'clock_drift': 0.0,
    'cfo': 0.0
}

# Fix positions of other anchors
for i in range(1, n_anchors):
    anchors[i]['position'] = known_positions[i]
    anchors[i]['is_fixed_position'] = True
```

**Step 2**: Initialize unknown nodes
```python
# Use trilateration or MDS for initial positions
# Set clocks to small random values near zero
for node in unknown_nodes:
    node['position'] = trilateration_estimate
    node['clock_bias'] = np.random.normal(0, 10)  # nanoseconds
    node['clock_drift'] = np.random.normal(0, 1)   # ppb
```

### 5.2 Factor Graph Construction

**Include Prior Factors** for gauge fixing:
```python
# Strong prior on disciplined anchor
graph.add_prior(
    node_id=0,
    position=[0, 0],
    clock_bias=0.0,
    variance=1e-10  # Very small variance = strong constraint
)

# Weaker priors on other anchors (position only)
for anchor in other_anchors:
    graph.add_position_prior(
        node_id=anchor.id,
        position=anchor.position,
        variance=1e-6
    )
```

### 5.3 Observability Check

**Test for sufficient constraints**:
```python
def check_observability(anchors, measurements):
    # Check spatial constraints
    n_position_constraints = 2 * len(anchors)
    assert n_position_constraints >= 3, "Need at least 3 position constraints"

    # Check anchor geometry
    if len(anchors) >= 2:
        assert not are_collinear(anchors), "Anchors must not be collinear"

    # Check time constraints
    time_anchors = [a for a in anchors if a.clock_fixed]
    assert len(time_anchors) >= 1, "Need at least one time reference"

    # Check connectivity
    assert is_graph_connected(nodes, measurements), "Graph must be connected"

    return True
```

---

## 6. Common Pitfalls and Solutions

### 6.1 Pitfall: All anchors on a line
**Symptom**: Convergence failure, reflection ambiguity
**Solution**: Add off-line anchor (e.g., center of area)

### 6.2 Pitfall: No time reference
**Symptom**: Clock biases drift together, large common-mode error
**Solution**: Fix at least one clock bias to zero

### 6.3 Pitfall: Over-constraining
**Symptom**: Conflicting constraints, high residuals at anchors
**Solution**: Use soft constraints (priors) instead of hard fixes

### 6.4 Pitfall: Weak connectivity
**Symptom**: Islands of nodes, poor information flow
**Solution**: Ensure sufficient communication range, add relays

---

## 7. Mathematical Formulation

### 7.1 Observability Matrix

The system is observable if the observability matrix has full rank:

```
O = [H; H*F; H*F²; ...]
```

Where:
- H: Measurement Jacobian
- F: State transition matrix

For the FTL system with state x = [p, b, d, f]:
- Rank deficiency = number of gauge freedoms
- Expected rank deficiency = 4 (2 for SE(2), 1 for time, 1 for frequency)

### 7.2 Fisher Information Matrix

The Fisher Information Matrix reveals unobservable directions:

```python
FIM = Σ_k J_k^T Σ_k^(-1) J_k
```

Eigenanalysis:
- Zero eigenvalues → unobservable directions
- Small eigenvalues → poorly observable
- Eigenvectors → directions in state space

---

## 8. Summary

### Key Takeaways

1. **Gauge freedoms are fundamental** - not bugs but inherent properties
2. **Minimum 3 non-collinear anchors** for robust 2D positioning
3. **At least one disciplined anchor** for time/frequency reference
4. **Collinear anchors must be avoided** - always include off-line anchor
5. **Use soft constraints (priors)** rather than hard fixes when possible

### Design Recommendations

For production deployment:
- Use 4+ anchors with good geometric distribution
- Include 1-2 disciplined anchors with GPS/atomic reference
- Place anchors at area perimeter + center
- Ensure every unknown node has path to 2+ anchors
- Monitor observability metrics during operation

---

## References

1. "Observability Analysis of Collaborative Localization" - IEEE Trans. Robotics
2. "Gauge Freedom in Network Localization" - SIAM J. Optimization
3. "Fisher Information and Cramer-Rao Bounds" - Statistical Signal Processing
4. "Clock Synchronization in Wireless Networks" - IEEE Communications

---

*Document Version: 1.0*
*Last Updated: 2024*
*System: FTL Consensus Simulation Framework*