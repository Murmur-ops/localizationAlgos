#!/usr/bin/env python3
"""
Test if solver works after fixes
"""

import numpy as np
from ftl.solver import FactorGraph

np.random.seed(42)

print("Testing solver with fixed numerical issues...")
print("="*60)

# Simple 4-anchor, 1-unknown setup
area_size = 20.0
anchors = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
unknown_true = (area_size/2, area_size/2)

# Test with realistic variance (after fix, min is 1e-12)
graph = FactorGraph()

# Add anchors
for i, (x, y) in enumerate(anchors):
    graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
    
# Add unknown with initial guess
initial = np.array([unknown_true[0] + 5, unknown_true[1] + 5, 0, 0, 0])
graph.add_node(4, initial, is_anchor=False)

# Add measurements with realistic variance
for i in range(4):
    anchor_pos = np.array(anchors[i])
    unknown_pos = np.array(unknown_true)
    dist = np.linalg.norm(anchor_pos - unknown_pos)
    toa = dist / 3e8
    
    # Use variance that will hit the 1e-12 floor
    # This corresponds to ~30cm range accuracy
    variance = 1e-12  # 1 ns²
    graph.add_toa_factor(i, 4, toa, variance)

# Optimize with verbose output
print("\nOptimizing with variance = 1e-12 s² (after fix)...")
result = graph.optimize(max_iterations=100, verbose=True, tolerance=1e-6)

# Check results
est_pos = result.estimates[4][:2]
pos_error = np.linalg.norm(est_pos - unknown_true)

print(f"\nResults:")
print(f"  Initial error: {np.linalg.norm(initial[:2] - unknown_true):.3f} m")
print(f"  Final error: {pos_error:.3f} m")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.iterations}")

print("\n" + "="*60)
if pos_error < 0.1:
    print("✓ SOLVER WORKING CORRECTLY!")
else:
    print("✗ Solver still has issues")
