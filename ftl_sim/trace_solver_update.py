#!/usr/bin/env python3
"""Trace through solver update step"""

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

print("Initial state:", solver.nodes[2].state)

# One iteration manually
J_wh, r_wh, node_to_idx, idx_to_node = solver._build_whitened_system()
J_scaled, S_mat = solver._apply_state_scaling(J_wh)

H = J_scaled.T @ J_scaled
g = J_scaled.T @ r_wh

print(f"\nGradient g: {g}")

# Add damping
lambda_lm = 1e-4
diag_H = np.diag(H)
min_diag = 1e-6
diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
H_damped = H + lambda_lm * np.diag(diag_regularized)

# Solve
delta_scaled = np.linalg.solve(H_damped, g)
print(f"Delta (scaled): {delta_scaled}")

# Unscale
delta_unscaled = S_mat @ delta_scaled
print(f"Delta (unscaled): {delta_unscaled}")

# This is what should happen in solver:
print(f"\nNode 2 in node_to_idx? {2 in node_to_idx}")
print(f"node_to_idx[2] = {node_to_idx[2]}")

# Apply update manually
if 2 in node_to_idx:
    idx = node_to_idx[2]
    print(f"Updating node 2 at index {idx}")
    print(f"Current state: {solver.nodes[2].state}")
    print(f"Update: -{delta_unscaled[idx:idx+5]}")
    solver.nodes[2].state -= delta_unscaled[idx:idx+5]
    print(f"New state: {solver.nodes[2].state}")
