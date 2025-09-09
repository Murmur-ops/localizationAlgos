#!/usr/bin/env python3
"""Validate MPS algorithm with paper's exact setup."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)

def create_paper_network():
    """Create network exactly as in paper Section 3."""
    np.random.seed(42)
    
    n_sensors = 30
    n_anchors = 6
    
    # Generate positions
    sensor_positions = np.random.uniform(0, 1, (n_sensors, 2))
    anchor_positions = np.random.uniform(0, 1, (n_anchors, 2))
    
    # Build adjacency
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Communication range 0.7, max 7 neighbors
    for i in range(n_sensors):
        neighbors = []
        for j in range(n_sensors):
            if i != j:
                dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
                if dist < 0.7:
                    neighbors.append((j, dist))
        
        # Limit to 7 neighbors
        if len(neighbors) > 7:
            neighbors.sort(key=lambda x: x[1])
            neighbors = neighbors[:7]
        
        for j, true_dist in neighbors:
            adjacency[i, j] = adjacency[j, i] = 1
            if (min(i,j), max(i,j)) not in distance_measurements:
                # Add 5% noise
                noisy_dist = true_dist * (1 + 0.05 * np.random.randn())
                distance_measurements[(min(i,j), max(i,j))] = noisy_dist
    
    # Anchor connections
    anchor_connections = {i: [] for i in range(n_sensors)}
    for i in range(n_sensors):
        for k in range(n_anchors):
            dist = np.linalg.norm(sensor_positions[i] - anchor_positions[k])
            if dist < 0.7:
                anchor_connections[i].append(k)
                noisy_dist = dist * (1 + 0.05 * np.random.randn())
                distance_measurements[(i, k)] = noisy_dist
    
    return NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=sensor_positions,
        measurement_variance=0.05**2
    )

def validate():
    """Run validation test."""
    
    print("MPS Algorithm Validation")
    print("=" * 60)
    
    # Create network
    network = create_paper_network()
    print(f"Network: {30} sensors, {6} anchors")
    print(f"Communication range: 0.7")
    print(f"Noise: 5%")
    
    # Paper's exact parameters
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        dimension=2,
        gamma=0.999,      # Paper's value
        alpha=10.0,       # Paper's value
        max_iterations=200,
        tolerance=1e-6,
        communication_range=0.7,
        verbose=False,
        early_stopping=True,
        early_stopping_window=50,
        admm_iterations=100,
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=True,
        parallel_proximal=False,
        use_2block=True,
        adaptive_alpha=False
    )
    
    print(f"\nParameters:")
    print(f"  γ = {config.gamma}")
    print(f"  α = {config.alpha}")
    print(f"  ADMM iterations = {config.admm_iterations}")
    print(f"  ADMM ρ = {config.admm_rho}")
    
    # Run algorithm
    print("\nRunning algorithm...")
    mps = MatrixParametrizedProximalSplitting(config, network)
    
    # Track convergence
    errors = []
    for k in range(50):
        stats = mps.run_iteration(k)
        errors.append(stats['position_error'])
        
        if k % 10 == 0:
            rel_error = np.linalg.norm(mps.X - network.true_positions, 'fro') / \
                       np.linalg.norm(network.true_positions, 'fro')
            print(f"  Iteration {k}: rel_error={rel_error:.4f}, "
                  f"pos_error={stats['position_error']:.4f}")
    
    # Final metrics
    final_positions = mps.X
    true_positions = network.true_positions
    
    rel_error = np.linalg.norm(final_positions - true_positions, 'fro') / \
               np.linalg.norm(true_positions, 'fro')
    
    mean_dist = np.mean([np.linalg.norm(final_positions[i] - true_positions[i])
                        for i in range(30)])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Relative Error: {rel_error:.4f}")
    print(f"Mean Distance: {mean_dist:.4f}")
    
    print("\nPaper reports: 0.05-0.10")
    print(f"Our result: {rel_error:.4f}")
    
    if rel_error <= 0.10:
        print("\n✓✓✓ SUCCESS! Matches paper's performance!")
    elif rel_error <= 0.15:
        print("\n✓✓ Close to paper's performance")
    else:
        print("\n✓ Needs more tuning")
    
    # Show convergence trend
    print("\nConvergence trend:")
    print(f"  Start: {errors[0]:.4f}")
    print(f"  End:   {errors[-1]:.4f}")
    if errors[-1] < errors[0]:
        print(f"  ✓ Improving ({100*(errors[0]-errors[-1])/errors[0]:.1f}% reduction)")
    else:
        print(f"  ✗ Not improving")

if __name__ == "__main__":
    validate()