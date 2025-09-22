#!/usr/bin/env python3
"""Debug Hessian issue"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver

solver = SquareRootSolver()

# Simple setup
solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)

solver.add_toa_factor(0, 2, 5.0, 0.01)
solver.add_toa_factor(1, 2, 5.0, 0.01)

# Build system
J_wh, r_wh, node_to_idx, idx_to_node = solver._build_whitened_system()

# Apply scaling
J_scaled, S_mat = solver._apply_state_scaling(J_wh)

print("J_wh shape:", J_wh.shape)
print("J_scaled shape:", J_scaled.shape)
print("S_mat diagonal:", np.diag(S_mat))

# Form Hessian
H = J_scaled.T @ J_scaled
g = J_scaled.T @ r_wh

print("\nHessian shape:", H.shape)
print("Hessian diagonal:", np.diag(H))
print("Hessian condition number:", np.linalg.cond(H))

# Try to solve
lambda_lm = 1e-4
H_damped = H + lambda_lm * np.diag(np.diag(H))
print("\nDamped Hessian diagonal:", np.diag(H_damped))

try:
    delta = np.linalg.solve(H_damped, g)
    print("Solution delta:", delta)
except np.linalg.LinAlgError as e:
    print(f"Failed to solve: {e}")
    
    # Check eigenvalues
    eigenvalues = np.linalg.eigvals(H_damped)
    print("Eigenvalues:", eigenvalues)
