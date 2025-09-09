#!/usr/bin/env python3
"""
Simple test of the MPS algorithm components
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.algorithm import MPSAlgorithm

# Test parameters
n_sensors = 10
n_anchors = 3
np.random.seed(42)

print("="*60)
print("SIMPLE MPS ALGORITHM TEST")
print("="*60)

# Generate positions
positions = np.random.uniform(0, 1, (n_sensors, 2))
anchor_positions = np.random.uniform(0, 1, (n_anchors, 2))

print(f"\nTest configuration:")
print(f"  Sensors: {n_sensors}")
print(f"  Anchors: {n_anchors}")
print(f"  Dimension: 2D")

# Create distance matrix
n_total = n_sensors + n_anchors
all_positions = np.vstack([positions, anchor_positions])
distance_matrix = np.zeros((n_total, n_total))

for i in range(n_total):
    for j in range(i+1, n_total):
        dist = np.linalg.norm(all_positions[i] - all_positions[j])
        # Add 5% noise
        noisy_dist = dist * (1 + 0.05 * np.random.randn())
        distance_matrix[i, j] = noisy_dist
        distance_matrix[j, i] = noisy_dist

print("\nRunning MPS algorithm...")

# Run algorithm with reduced iterations
mps = MPSAlgorithm(
    n_sensors=n_sensors,
    n_anchors=n_anchors,
    distance_matrix=distance_matrix,
    anchor_positions=anchor_positions,
    gamma=0.999,
    alpha=10.0,
    max_iterations=100  # Quick test
)

result = mps.solve()

print(f"\nResults:")
print(f"  Converged: {result['converged']}")
print(f"  Iterations: {result['iterations']}")
print(f"  Final residual: {result['residual']:.6f}")
print(f"  RMSE: {result['rmse']:.4f}")

# Compute errors
errors = np.linalg.norm(result['positions'] - positions, axis=1)
print(f"\nPosition errors:")
print(f"  Mean: {np.mean(errors):.4f}")
print(f"  Max: {np.max(errors):.4f}")
print(f"  Min: {np.min(errors):.4f}")

# Show relative error
network_diameter = np.max(distance_matrix)
relative_error = result['rmse'] / network_diameter
print(f"\nRelative error: {relative_error:.2%} of network diameter")

print("\n✓ Test completed successfully")