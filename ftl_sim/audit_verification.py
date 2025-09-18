#!/usr/bin/env python3
"""
Verify the issues identified in the audit
"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver
from ftl.factors_scaled import ToAFactorMeters

print("=" * 60)
print("AUDIT VERIFICATION SCRIPT")
print("=" * 60)

# 1. Check variance/std handling
print("\n1. VARIANCE vs STD HANDLING CHECK:")
print("-" * 40)

# Create a factor with known variance
range_meas = 10.0  # meters
range_std = 0.1   # 10cm standard deviation
range_var = range_std ** 2  # 0.01 m²

factor = ToAFactorMeters(0, 1, range_meas, range_var)
print(f"Input: std={range_std}m, var={range_var}m²")
print(f"Factor stored variance: {factor.variance}")
print(f"Factor computed std: {np.sqrt(factor.variance)}")

# Check weight computation
weight = 1.0 / factor.variance
print(f"Weight = 1/variance = {weight}")
print(f"Expected weight for 10cm std: {1.0/(0.1**2)} = 100")

if abs(weight - 100) < 1e-10:
    print("✓ Variance handling is CORRECT")
else:
    print("✗ Variance handling is INCORRECT")

# 2. Check regularization impact
print("\n2. REGULARIZATION IMPACT CHECK:")
print("-" * 40)

solver = SquareRootSolver()

# Create a simple problem where y is barely observable
solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(1, np.array([10.0, 0.01, 0.0, 0.0, 0.0]), is_anchor=True)  # Almost collinear
solver.add_node(2, np.array([5.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=False)

solver.add_toa_factor(0, 2, 5.0, 0.01)
solver.add_toa_factor(1, 2, 5.0, 0.01)

# Build system
J_wh, r_wh, _, _ = solver._build_whitened_system()
J_scaled, _ = solver._apply_state_scaling(J_wh)
H = J_scaled.T @ J_scaled

diag_H = np.diag(H)
print(f"Hessian diagonal (before regularization):")
print(f"  x:    {diag_H[0]:.2e}")
print(f"  y:    {diag_H[1]:.2e}")
print(f"  bias: {diag_H[2]:.2e}")
print(f"  drift: {diag_H[3]:.2e}")
print(f"  cfo:  {diag_H[4]:.2e}")

# Apply current regularization
min_diag = 1e-6
diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
lambda_lm = 1e-4
H_damped = H + lambda_lm * np.diag(diag_regularized)

diag_damped = np.diag(H_damped)
print(f"\nDiagonal after regularization (lambda={lambda_lm}):")
print(f"  x:    {diag_damped[0]:.2e}")
print(f"  y:    {diag_damped[1]:.2e} (was {diag_H[1]:.2e})")

# Check if observable states are affected
y_increase = (diag_damped[1] - diag_H[1]) / diag_H[1] * 100 if diag_H[1] > 0 else np.inf
print(f"\nY-direction diagonal increased by {y_increase:.1f}%")

if y_increase < 1:
    print("✓ Regularization minimally affects observable states")
else:
    print(f"⚠ Regularization affects observable states by {y_increase:.1f}%")

# 3. Check state scaling justification
print("\n3. STATE SCALING CHECK:")
print("-" * 40)

scales = solver.config.get_default_state_scale()
print(f"Default scales: {scales}")
print(f"  Position (m):     {scales[0]}")
print(f"  Position (m):     {scales[1]}")
print(f"  Bias (ns):        {scales[2]}")
print(f"  Drift (ppb):      {scales[3]}")
print(f"  CFO (ppm):        {scales[4]}")

# Typical magnitudes
typical_pos = 10.0  # 10m
typical_bias = 10.0  # 10ns
typical_drift = 100.0  # 100ppb
typical_cfo = 10.0  # 10ppm

scaled_values = np.array([
    typical_pos * scales[0],
    typical_pos * scales[1],
    typical_bias * scales[2],
    typical_drift * scales[3],
    typical_cfo * scales[4]
])

print(f"\nTypical scaled values:")
print(f"  Position: {scaled_values[0]:.1f}")
print(f"  Bias:     {scaled_values[2]:.1f}")
print(f"  Drift:    {scaled_values[3]:.1f}")
print(f"  CFO:      {scaled_values[4]:.1f}")

if np.all(scaled_values > 0.1) and np.all(scaled_values < 100):
    print("✓ Scaling produces reasonable magnitudes")
else:
    print("⚠ Scaling may need adjustment")

# 4. Check for bias in simple problem
print("\n4. BIAS CHECK (SIMPLE TRILATERATION):")
print("-" * 40)

np.random.seed(42)
n_trials = 50
errors_x = []
errors_y = []

for _ in range(n_trials):
    solver = SquareRootSolver()

    # Perfect square of anchors
    solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
    solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
    solver.add_node(2, np.array([10.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)
    solver.add_node(3, np.array([0.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)

    # True position at center
    true_pos = np.array([5.0, 5.0])

    # Initial guess
    solver.add_node(4, np.array([4.0, 6.0, 0.0, 0.0, 0.0]), is_anchor=False)

    # Add perfect measurements (no noise) to isolate algorithmic bias
    for anchor_id in range(4):
        anchor = solver.nodes[anchor_id].state
        true_range = np.linalg.norm(true_pos - anchor[:2])
        solver.add_toa_factor(anchor_id, 4, true_range, 0.01**2)

    result = solver.optimize(verbose=False)
    if result.converged:
        est = result.estimates[4]
        errors_x.append(est[0] - true_pos[0])
        errors_y.append(est[1] - true_pos[1])

mean_error_x = np.mean(errors_x) if errors_x else np.nan
mean_error_y = np.mean(errors_y) if errors_y else np.nan

print(f"Results from {len(errors_x)}/{n_trials} converged trials:")
print(f"Mean error: x={mean_error_x*100:.3f}cm, y={mean_error_y*100:.3f}cm")

if abs(mean_error_x) < 0.001 and abs(mean_error_y) < 0.001:
    print("✓ No algorithmic bias detected")
else:
    print(f"⚠ Possible algorithmic bias: ({mean_error_x*100:.3f}, {mean_error_y*100:.3f})cm")

# 5. Check condition number with realistic problem
print("\n5. CONDITION NUMBER CHECK:")
print("-" * 40)

solver = SquareRootSolver()

# 4 anchors, well-conditioned geometry
for i in range(4):
    angle = i * np.pi / 2
    x = 10 * np.cos(angle)
    y = 10 * np.sin(angle)
    solver.add_node(i, np.array([x, y, 0.0, 0.0, 0.0]), is_anchor=True)

solver.add_node(4, np.array([0.0, 0.0, 5.0, 10.0, 1.0]), is_anchor=False)

# Add measurements
for i in range(4):
    solver.add_toa_factor(i, 4, 10.0, 0.01)

# Add clock prior for drift/CFO observability
solver.add_clock_prior(4, 0.0, 0.0, 100.0, 2500.0)

J_wh, r_wh, _, _ = solver._build_whitened_system()
J_scaled, _ = solver._apply_state_scaling(J_wh)
H = J_scaled.T @ J_scaled

cond = np.linalg.cond(H)
print(f"Condition number (with clock prior): {cond:.2e}")

if cond < 1e8:
    print("✓ Good conditioning")
elif cond < 1e12:
    print("⚠ Marginal conditioning but workable")
else:
    print("✗ Poor conditioning")

print("\n" + "=" * 60)
print("AUDIT VERIFICATION COMPLETE")
print("=" * 60)