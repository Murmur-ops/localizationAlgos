#!/usr/bin/env python3
"""
Test MPS with correct alpha value (not 10.0!)
"""

import numpy as np
import sys
sys.path.append('.')

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)


def test_with_correct_params():
    """Test with paper's actual parameters"""
    
    print("="*60)
    print("TEST WITH CORRECT PARAMETERS")
    print("="*60)
    print()
    
    # Setup exactly as paper
    n_sensors = 9
    n_anchors = 4
    
    # Normalized positions
    positions = []
    for i in range(3):
        for j in range(3):
            positions.append([i/2, j/2])
    
    true_positions = np.array(positions)
    anchor_positions = np.array([[0,0], [1,0], [0,1], [1,1]])
    
    # Generate measurements
    np.random.seed(42)
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Use paper's noise level
    noise_std = 0.01  # This is absolute noise in normalized space
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= 0.8:
                adjacency[i,j] = 1
                adjacency[j,i] = 1
                # Absolute noise (not relative)
                distance_measurements[(i,j)] = true_dist + np.random.normal(0, noise_std)
    
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 1.0:
                connected.append(a)
                distance_measurements[(i, n_sensors + a)] = true_dist + np.random.normal(0, noise_std)
        anchor_connections[i] = connected
    
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=noise_std**2
    )
    
    # Test different alpha values
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"Testing different alpha values...")
    print(f"Paper uses alpha ≈ 1.0")
    print()
    
    results = []
    
    for alpha in alphas:
        config = MPSConfig(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            dimension=2,
            gamma=0.99,  # Paper value
            alpha=alpha,  # TEST THIS
            max_iterations=500,
            tolerance=1e-8,
            verbose=False,
            early_stopping=True,
            early_stopping_window=30,
            admm_iterations=20,
            admm_tolerance=1e-6,
            admm_rho=1.0,
            warm_start=False,
            use_2block=True,
            parallel_proximal=False,  # Simpler for testing
            adaptive_alpha=False  # Don't adapt - use fixed value
        )
        
        mps = MatrixParametrizedProximalSplitting(config, network_data)
        result = mps.run()
        
        final_positions = result['final_positions']
        rmse = np.sqrt(np.mean(np.linalg.norm(final_positions - true_positions, axis=1)**2))
        rmse_mm = rmse * 1000
        
        results.append((alpha, rmse_mm, result['iterations'], result['converged']))
        
        print(f"  α = {alpha:4.1f}: RMSE = {rmse_mm:7.2f}mm, iter = {result['iterations']:3d}, conv = {result['converged']}")
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    
    best_alpha, best_rmse, _, _ = min(results, key=lambda x: x[1])
    
    print(f"Best alpha: {best_alpha}")
    print(f"Best RMSE: {best_rmse:.2f}mm")
    print(f"Paper achieves: ~40mm")
    print(f"Ratio: {best_rmse/40:.1f}x")
    
    if best_rmse < 100:
        print("\n✓ SUCCESS! Within same order of magnitude as paper!")
        print(f"The problem was alpha = 10.0 instead of {best_alpha}!")
    else:
        print(f"\n⚠ Still off. Need to check proximal operators.")
    
    # Test with even smaller alphas
    if best_rmse > 100:
        print("\nTrying smaller alphas...")
        small_alphas = [0.01, 0.05, 0.001]
        
        for alpha in small_alphas:
            config.alpha = alpha
            mps = MatrixParametrizedProximalSplitting(config, network_data)
            result = mps.run()
            
            final_positions = result['final_positions']
            rmse = np.sqrt(np.mean(np.linalg.norm(final_positions - true_positions, axis=1)**2))
            rmse_mm = rmse * 1000
            
            print(f"  α = {alpha:6.3f}: RMSE = {rmse_mm:7.2f}mm")
            
            if rmse_mm < best_rmse:
                best_alpha = alpha
                best_rmse = rmse_mm
    
    print()
    print(f"Final best: α = {best_alpha}, RMSE = {best_rmse:.2f}mm")
    
    return best_alpha, best_rmse


if __name__ == "__main__":
    best_alpha, best_rmse = test_with_correct_params()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if best_rmse < 100:
        print(f"✓ Algorithm works with α = {best_alpha}")
        print(f"  Achieves {best_rmse:.2f}mm vs paper's 40mm")
        print(f"  The default α = 10.0 was the problem!")
    else:
        print(f"⚠ Even with α = {best_alpha}, still get {best_rmse:.2f}mm")
        print("  Need to check:")
        print("  - Proximal operator implementation")
        print("  - Matrix structure initialization")
        print("  - ADMM solver convergence")