#!/usr/bin/env python3
"""Debug gain ratio calculation"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver

solver = SquareRootSolver()

# Simple problem
solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)

# Measurements place unknown at (5, 0)
solver.add_toa_factor(0, 2, 5.0, 0.01)
solver.add_toa_factor(1, 2, 5.0, 0.01)

# Build system
J_wh, r_wh, node_to_idx, idx_to_node = solver._build_whitened_system()
J_scaled, S_mat = solver._apply_state_scaling(J_wh)

# Current cost
cost = solver._compute_cost(r_wh)
print(f"Initial cost: {cost:.6f}")

# Form system
H = J_scaled.T @ J_scaled
g = J_scaled.T @ r_wh

print(f"\nGradient g: {g}")
print(f"Gradient norm: {np.linalg.norm(g, ord=np.inf):.2e}")

# Add damping
lambda_lm = 1e-4
diag_H = np.diag(H)
min_diag = 1e-6
diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
H_damped = H + lambda_lm * np.diag(diag_regularized)

# Solve
delta_scaled = np.linalg.solve(H_damped, g)
print(f"\nDelta (scaled): {delta_scaled}")

# Predicted decrease using DAMPED Hessian (correct formula)
predicted_decrease_damped = np.dot(g, delta_scaled) - 0.5 * np.dot(delta_scaled, H_damped @ delta_scaled)
print(f"\nPredicted decrease (with damped H): {predicted_decrease_damped:.6f}")

# Predicted decrease using UNDAMPED Hessian (current incorrect formula)
predicted_decrease_undamped = -np.dot(g, delta_scaled) - 0.5 * np.dot(delta_scaled, H @ delta_scaled)
print(f"Predicted decrease (with undamped H): {predicted_decrease_undamped:.6f}")

# Apply update
delta_unscaled = S_mat @ delta_scaled
print(f"\nDelta (unscaled): {delta_unscaled}")

# Apply to node
if 2 in node_to_idx:
    idx = node_to_idx[2]
    print(f"\nBefore update: {solver.nodes[2].state}")
    solver.nodes[2].state -= delta_unscaled[idx:idx+5]
    print(f"After update: {solver.nodes[2].state}")

# Compute new cost
J_wh_new, r_wh_new, _, _ = solver._build_whitened_system()
new_cost = solver._compute_cost(r_wh_new)
print(f"\nNew cost: {new_cost:.6f}")

actual_decrease = cost - new_cost
print(f"Actual decrease: {actual_decrease:.6f}")

# Gain ratios
if predicted_decrease_damped > 0:
    gain_damped = actual_decrease / predicted_decrease_damped
else:
    gain_damped = 0

if predicted_decrease_undamped > 0:
    gain_undamped = actual_decrease / predicted_decrease_undamped
else:
    gain_undamped = 0

print(f"\nGain ratio (damped): {gain_damped:.3f}")
print(f"Gain ratio (undamped): {gain_undamped:.3f}")

# The correct gain ratio should be close to 1 for good steps
print("\nAnalysis:")
if gain_damped > 0.25:
    print(f"✓ Step would be ACCEPTED with correct formula (gain={gain_damped:.3f})")
else:
    print(f"✗ Step would be REJECTED with correct formula (gain={gain_damped:.3f})")

if gain_undamped > 0.25:
    print(f"✓ Step would be ACCEPTED with current formula (gain={gain_undamped:.3f})")
else:
    print(f"✗ Step would be REJECTED with current formula (gain={gain_undamped:.3f})")