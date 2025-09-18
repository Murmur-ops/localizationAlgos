#!/usr/bin/env python3
"""Debug why solver doesn't move"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver

solver = SquareRootSolver()

# Simple 2-anchor, 1-unknown
solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
solver.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)  # Initial guess

# True position is (5, 0) - on the line between anchors
# Distances: 5m and 5m
solver.add_toa_factor(0, 2, 5.0, 0.01)  # 10cm std
solver.add_toa_factor(1, 2, 5.0, 0.01)

print("Initial state of unknown:", solver.nodes[2].state)

# Try one iteration manually
J_wh, r_wh, node_to_idx, idx_to_node = solver._build_whitened_system()
print(f"\nWhitened residual shape: {r_wh.shape}")
print(f"Whitened residual: {r_wh}")
print(f"Whitened Jacobian shape: {J_wh.shape}")
print(f"Whitened Jacobian:\n{J_wh}")

# Check if node_to_idx is correct
print(f"\nNode to index mapping: {node_to_idx}")

result = solver.optimize(verbose=True)
print(f"\nFinal state of unknown: {result.estimates[2]}")
print(f"Converged: {result.converged}")
print(f"Reason: {result.convergence_reason}")
