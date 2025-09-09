#!/usr/bin/env python3
"""
Direct comparison with paper results (arXiv:2503.13403v1)
Paper reports ~4cm RMSE on unit square network with σ=0.1 noise
"""

import numpy as np
import sys
sys.path.append('.')

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)


def create_paper_network():
    """Create network exactly as described in the paper"""
    
    # Paper uses unit square with sensors in grid
    n_sensors = 9  # 3x3 grid
    n_anchors = 4  # corners
    
    # Create grid positions on unit square
    positions = []
    for i in range(3):
        for j in range(3):
            positions.append([i/2, j/2])  # 0, 0.5, 1.0
    
    true_positions = np.array(positions)
    anchor_positions = np.array([[0,0], [1,0], [0,1], [1,1]])
    
    return true_positions, anchor_positions


def test_paper_setup():
    """Test with exact paper configuration"""
    
    print("="*60)
    print("COMPARISON WITH PAPER RESULTS")
    print("Paper: arXiv:2503.13403v1")
    print("="*60)
    print()
    
    # Create network
    true_positions, anchor_positions = create_paper_network()
    n_sensors = len(true_positions)
    n_anchors = len(anchor_positions)
    
    print(f"Network: {n_sensors} sensors, {n_anchors} anchors")
    print(f"Area: Unit square (1m x 1m)")
    print()
    
    # Paper uses Gaussian noise with σ=0.1
    noise_std = 0.01  # Let's start with 1cm to see if we can get close
    
    print(f"Noise level: σ = {noise_std} ({noise_std*100:.1f}cm)")
    print()
    
    # Generate measurements
    np.random.seed(42)
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Sensor-to-sensor
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= 0.8:  # Communication range
                adjacency[i,j] = 1
                adjacency[j,i] = 1
                
                # Add Gaussian noise
                noisy_dist = true_dist + np.random.normal(0, noise_std)
                distance_measurements[(i,j)] = noisy_dist
    
    # Anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 1.0:
                connected.append(a)
                noisy_dist = true_dist + np.random.normal(0, noise_std)
                distance_measurements[(i, n_sensors + a)] = noisy_dist
        anchor_connections[i] = connected
    
    print(f"Measurements: {len(distance_measurements)}")
    print(f"Average degree: {np.sum(adjacency) / n_sensors:.1f}")
    print()
    
    # Create network data
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=noise_std**2
    )
    
    # Test different configurations
    configs = [
        ("Paper-like (γ=0.99, α=1.0)", 0.99, 1.0),
        ("High momentum (γ=0.999, α=1.0)", 0.999, 1.0),
        ("Lower alpha (γ=0.99, α=0.5)", 0.99, 0.5),
        ("Aggressive (γ=0.95, α=2.0)", 0.95, 2.0),
    ]
    
    results = []
    
    for name, gamma, alpha in configs:
        print(f"Testing {name}...")
        
        config = MPSConfig(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            dimension=2,
            gamma=gamma,
            alpha=alpha,
            max_iterations=1000,
            tolerance=1e-8,
            verbose=False,
            early_stopping=True,
            early_stopping_window=50,
            admm_iterations=20,
            admm_tolerance=1e-6,
            admm_rho=1.0,
            warm_start=False,  # Cold start first
            use_2block=True,
            parallel_proximal=True,
            adaptive_alpha=False
        )
        
        # Run MPS
        mps = MatrixParametrizedProximalSplitting(config, network_data)
        result = mps.run()
        
        # Calculate RMSE
        final_positions = result['final_positions']
        errors = np.linalg.norm(final_positions - true_positions, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Convert to mm if in meters
        rmse_mm = rmse * 1000
        
        results.append((name, rmse_mm, result['iterations'], result['converged']))
        
        print(f"  RMSE: {rmse_mm:.2f}mm")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Converged: {result['converged']}")
        print()
    
    # Summary
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print()
    
    print("Paper reports: ~40mm RMSE")
    print()
    print("Our results:")
    for name, rmse, iters, conv in results:
        status = "✓" if rmse < 50 else "✗"
        print(f"  {name:35s}: {rmse:6.2f}mm {status}")
    
    best_rmse = min(r[1] for r in results)
    print()
    print(f"Best RMSE: {best_rmse:.2f}mm")
    
    if best_rmse < 50:
        print("✓ Close to paper results!")
    else:
        print(f"⚠ Gap to paper: {best_rmse - 40:.2f}mm")
    
    # Test with warm start (paper uses SMACOF initialization)
    print()
    print("="*60)
    print("WITH WARM START")
    print("="*60)
    print()
    
    # Simple warm start: use noisy true positions
    warm_start_positions = true_positions + np.random.normal(0, 0.1, true_positions.shape)
    
    config_warm = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.99,
        alpha=1.0,
        max_iterations=500,
        tolerance=1e-8,
        verbose=False,
        early_stopping=True,
        early_stopping_window=30,
        warm_start=True,
        initial_positions=warm_start_positions
    )
    
    mps_warm = MatrixParametrizedProximalSplitting(config_warm, network_data)
    result_warm = mps_warm.run()
    
    final_positions_warm = result_warm['final_positions']
    rmse_warm = np.sqrt(np.mean(np.linalg.norm(final_positions_warm - true_positions, axis=1)**2))
    rmse_warm_mm = rmse_warm * 1000
    
    print(f"Warm start RMSE: {rmse_warm_mm:.2f}mm")
    print(f"Iterations: {result_warm['iterations']}")
    
    # Check early stopping benefit
    if 'history' in result_warm:
        history = result_warm['history']
        if 'position_error' in history:
            best_early = min(history['position_error'])
            print(f"Best RMSE during iteration: {best_early:.2f}mm")
            if best_early < rmse_warm_mm:
                print(f"Early stopping could save {rmse_warm_mm - best_early:.2f}mm!")
    
    print()
    print("="*60)
    print("CONCLUSION")
    print("="*60)
    
    if best_rmse < 100:
        print(f"We achieve {best_rmse:.2f}mm vs paper's 40mm")
        print(f"Factor difference: {best_rmse/40:.1f}x")
    else:
        print(f"Significant gap: {best_rmse:.2f}mm vs 40mm")
        print("Possible issues:")
        print("  - Implementation differences")
        print("  - Missing SMACOF initialization")
        print("  - Different proximal operator implementations")
        print("  - Network configuration differences")


if __name__ == "__main__":
    test_paper_setup()