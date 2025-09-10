#!/usr/bin/env python3
"""
Test MPS algorithm in properly normalized [0,1] space
This should match the paper's results
"""

import numpy as np
import sys
sys.path.append('.')

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)


def test_normalized():
    """Test in normalized [0,1] space as the paper does"""
    
    print("="*60)
    print("NORMALIZED SPACE TEST")
    print("="*60)
    print()
    
    # Create network in [0,1] x [0,1] space
    n_sensors = 9  # 3x3 grid
    n_anchors = 4  # corners
    
    # Positions in normalized space
    positions = []
    for i in range(3):
        for j in range(3):
            positions.append([i/2, j/2])  # 0, 0.5, 1.0
    
    true_positions = np.array(positions)
    anchor_positions = np.array([[0,0], [1,0], [0,1], [1,1]])
    
    print(f"Network: {n_sensors} sensors, {n_anchors} anchors")
    print(f"Space: [0,1] x [0,1] normalized")
    print()
    
    # Generate measurements in normalized space
    # Paper uses σ=0.1 which means 10% relative error
    relative_noise = 0.01  # Start with 1% to test
    
    np.random.seed(42)
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Sensor-to-sensor
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= 0.8:  # Within normalized range
                adjacency[i,j] = 1
                adjacency[j,i] = 1
                
                # Add relative noise
                noise = np.random.normal(0, relative_noise * true_dist)
                noisy_dist = true_dist + noise
                distance_measurements[(i,j)] = noisy_dist
    
    # Anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 1.0:
                connected.append(a)
                noise = np.random.normal(0, relative_noise * true_dist)
                distance_measurements[(i, n_sensors + a)] = true_dist + noise
        anchor_connections[i] = connected
    
    print(f"Measurements: {len(distance_measurements)}")
    print(f"Relative noise: {relative_noise*100:.1f}%")
    print()
    
    # Create network data
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=(relative_noise)**2
    )
    
    # Test with paper's parameters
    config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.99,  # Paper's value
        alpha=1.0,   # No arbitrary scaling!
        max_iterations=1000,
        tolerance=1e-8,
        verbose=False,
        early_stopping=True,
        early_stopping_window=50,
        admm_iterations=20,
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=False,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=False
    )
    
    print("Running MPS in normalized space...")
    print(f"  gamma: {config.gamma}")
    print(f"  alpha: {config.alpha} (no scaling!)")
    print()
    
    # Run MPS
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    result = mps.run()
    
    # Calculate RMSE in normalized space
    final_positions = result['final_positions']
    errors = np.linalg.norm(final_positions - true_positions, axis=1)
    rmse_normalized = np.sqrt(np.mean(errors**2))
    
    # Convert to physical units (assume 1m x 1m physical space)
    physical_scale = 1.0  # meters
    rmse_meters = rmse_normalized * physical_scale
    rmse_mm = rmse_meters * 1000
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    print()
    print(f"RMSE (normalized): {rmse_normalized:.6f}")
    print(f"RMSE (meters): {rmse_meters:.6f}")
    print(f"RMSE (mm): {rmse_mm:.2f}")
    print()
    
    # Per-sensor errors
    print("Per-sensor errors (mm):")
    for i, error in enumerate(errors):
        error_mm = error * physical_scale * 1000
        print(f"  Sensor {i}: {error_mm:.2f}mm")
    
    print()
    print("="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Paper achieves: ~40mm")
    print(f"We achieve: {rmse_mm:.2f}mm")
    print(f"Ratio: {rmse_mm/40:.1f}x")
    
    if rmse_mm < 100:
        print("\n✓ SUCCESS: Within same order of magnitude as paper!")
    else:
        print(f"\n⚠ Still off by {rmse_mm/40:.1f}x")
    
    # Also test with higher noise like paper
    print("\n" + "="*60)
    print("TEST WITH PAPER'S NOISE LEVEL")
    print("="*60)
    
    # Regenerate with σ=0.1 (10% relative noise)
    relative_noise_paper = 0.1
    distance_measurements_paper = {}
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= 0.8:
                noise = np.random.normal(0, relative_noise_paper * true_dist)
                distance_measurements_paper[(i,j)] = true_dist + noise
    
    for i in range(n_sensors):
        for a in anchor_connections[i]:
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            noise = np.random.normal(0, relative_noise_paper * true_dist)
            distance_measurements_paper[(i, n_sensors + a)] = true_dist + noise
    
    network_data_paper = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements_paper,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=(relative_noise_paper)**2
    )
    
    print(f"Relative noise: {relative_noise_paper*100:.1f}% (same as paper)")
    print()
    
    mps_paper = MatrixParametrizedProximalSplitting(config, network_data_paper)
    result_paper = mps_paper.run()
    
    final_positions_paper = result_paper['final_positions']
    rmse_paper = np.sqrt(np.mean(np.linalg.norm(final_positions_paper - true_positions, axis=1)**2))
    rmse_paper_mm = rmse_paper * physical_scale * 1000
    
    print(f"RMSE with paper's noise: {rmse_paper_mm:.2f}mm")
    print(f"Paper achieves: ~40mm")
    print(f"Ratio: {rmse_paper_mm/40:.1f}x")
    
    return rmse_mm, rmse_paper_mm


if __name__ == "__main__":
    rmse_low_noise, rmse_paper_noise = test_normalized()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"With 1% noise: {rmse_low_noise:.2f}mm")
    print(f"With 10% noise: {rmse_paper_noise:.2f}mm")
    print(f"Paper (10% noise): 40mm")
    
    if rmse_paper_noise < 100:
        print("\n✓ Algorithm works correctly in normalized space!")
    else:
        print("\n⚠ Still have implementation issues to fix")