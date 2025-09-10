#!/usr/bin/env python3
"""
Quick test of MPS algorithm with reduced iterations
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.mps_full_algorithm import MatrixParametrizedProximalSplitting, NetworkData

# Generate test network
print("Generating test network...")
np.random.seed(42)
n_sensors = 10
n_anchors = 3
n_total = n_sensors + n_anchors

# Generate positions
positions = np.random.uniform(0, 1, (n_total, 2))

# Compute true distances
true_distances = np.zeros((n_total, n_total))
for i in range(n_total):
    for j in range(i+1, n_total):
        dist = np.linalg.norm(positions[i] - positions[j])
        true_distances[i, j] = dist
        true_distances[j, i] = dist

# Add noise
print("Adding measurement noise...")
measured_distances = true_distances.copy()
mask = true_distances > 0
noise = 1 + 0.05 * np.random.randn(*true_distances.shape)
measured_distances[mask] *= noise[mask]

# Create network data
network_data = NetworkData(
    n_sensors=n_sensors,
    n_anchors=n_anchors,
    true_positions=positions,
    true_distances=true_distances,
    measured_distances=measured_distances,
    communication_range=0.5
)

# Run MPS with fewer iterations
print("\nRunning MPS algorithm (quick test)...")
mps = MatrixParametrizedProximalSplitting(
    network_data,
    gamma=0.999,
    alpha=10.0,
    max_iterations=50,  # Reduced for quick test
    tolerance=1e-4,
    verbose=True
)

# Solve
result = mps.solve()

print("\n" + "="*60)
print("QUICK TEST RESULTS")
print("="*60)
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Final RMSE: {result['rmse']:.4f}")
print(f"Relative error: {result['relative_error']:.2%}")
print(f"Consensus residual: {result['consensus_residual']:.6f}")

# Show position errors
if result['position_estimates'] is not None:
    errors = np.linalg.norm(
        result['position_estimates'] - network_data.true_positions[:n_sensors],
        axis=1
    )
    print(f"\nPer-node position errors:")
    for i, err in enumerate(errors):
        print(f"  Node {i}: {err:.4f}")
    print(f"\nMean error: {np.mean(errors):.4f}")
    print(f"Max error: {np.max(errors):.4f}")