#!/usr/bin/env python3
"""
Check for accounting/conversion errors in the MPS implementation
"""

import numpy as np
import sys
sys.path.append('.')

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)


def test_simple_case():
    """Test with the simplest possible case to check accounting"""
    
    print("="*60)
    print("SIMPLE TEST CASE - CHECK ACCOUNTING")
    print("="*60)
    print()
    
    # Simplest case: 3 sensors in a line, 2 anchors at ends
    true_positions = np.array([
        [0.0, 0.0],  # Sensor 0 (at anchor 0)
        [0.5, 0.0],  # Sensor 1 (middle)
        [1.0, 0.0],  # Sensor 2 (at anchor 1)
    ])
    
    anchor_positions = np.array([
        [0.0, 0.0],  # Anchor 0
        [1.0, 0.0],  # Anchor 1
    ])
    
    print("Setup:")
    print(f"  Sensors: {true_positions}")
    print(f"  Anchors: {anchor_positions}")
    print()
    
    # Perfect measurements (no noise)
    adjacency = np.ones((3, 3)) - np.eye(3)
    distance_measurements = {}
    
    # Sensor-to-sensor distances
    for i in range(3):
        for j in range(i+1, 3):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            distance_measurements[(i,j)] = true_dist
            print(f"  Distance {i}-{j}: {true_dist:.3f}")
    
    # Anchor connections
    anchor_connections = {
        0: [0],  # Sensor 0 connects to anchor 0
        1: [],   # Sensor 1 no direct anchor connection
        2: [1],  # Sensor 2 connects to anchor 1
    }
    
    # Add anchor distances
    distance_measurements[(0, 3)] = 0.0  # Sensor 0 to anchor 0
    distance_measurements[(2, 4)] = 0.0  # Sensor 2 to anchor 1
    
    print("\nAnchor connections:")
    print(f"  Sensor 0 -> Anchor 0: 0.0")
    print(f"  Sensor 2 -> Anchor 1: 0.0")
    print()
    
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=1e-10  # Nearly zero noise
    )
    
    # Run with minimal iterations to see initial behavior
    config = MPSConfig(
        n_sensors=3,
        n_anchors=2,
        dimension=2,
        gamma=0.99,
        alpha=1.0,
        max_iterations=10,
        tolerance=1e-8,
        verbose=True,  # See what's happening
        early_stopping=False,
        admm_iterations=10,
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=False,
        use_2block=True,
        parallel_proximal=False,
        adaptive_alpha=False
    )
    
    print("Running MPS for 10 iterations...")
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    
    # Check initial state
    print("\nInitial estimates:")
    print(f"  X: {mps.X}")
    print(f"  Y diagonal: {np.diag(mps.Y)}")
    print()
    
    result = mps.run()
    
    final_positions = result['final_positions']
    
    print("\nFinal positions after 10 iterations:")
    for i in range(3):
        print(f"  Sensor {i}: {final_positions[i]} (true: {true_positions[i]})")
    
    # Calculate errors
    errors = []
    for i in range(3):
        error = np.linalg.norm(final_positions[i] - true_positions[i])
        errors.append(error)
        print(f"  Error {i}: {error:.6f} ({error*1000:.3f}mm)")
    
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    print(f"\nRMSE: {rmse:.6f} ({rmse*1000:.2f}mm)")
    
    # Check if positions are in wrong units
    print("\n" + "="*60)
    print("UNIT CHECK")
    print("="*60)
    
    # Check if there's a scaling issue
    scales_to_test = [1.0, 10.0, 50.0, 100.0, 1000.0]
    
    print("\nTrying different scale interpretations:")
    for scale in scales_to_test:
        scaled_rmse = rmse * scale
        print(f"  Scale {scale:5.0f}: RMSE = {scaled_rmse*1000:.2f}mm")
        if 30 <= scaled_rmse*1000 <= 50:
            print(f"    ^ This matches the paper's 40mm!")
    
    # Check if errors are actually in different units
    print("\nChecking if final_positions are in different units:")
    avg_position = np.mean([np.linalg.norm(p) for p in final_positions])
    print(f"  Average position magnitude: {avg_position:.6f}")
    print(f"  Expected (0-1 range): ~0.5")
    
    if avg_position > 10:
        print(f"  WARNING: Positions seem to be scaled by {avg_position/0.5:.1f}x")
    
    return rmse


def test_distance_computation():
    """Check how distances are computed from the matrix Y"""
    
    print("\n" + "="*60)
    print("DISTANCE COMPUTATION CHECK")
    print("="*60)
    print()
    
    # Create a simple Y matrix
    Y = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    print("Test Y matrix:")
    print(Y)
    print()
    
    # Compute distances using the formula from the paper
    # d_ij = sqrt(Y_ii + Y_jj - 2*Y_ij)
    
    print("Distances from Y:")
    for i in range(3):
        for j in range(i+1, 3):
            dist_sq = Y[i,i] + Y[j,j] - 2*Y[i,j]
            dist = np.sqrt(max(0, dist_sq))
            print(f"  d_{i}{j} = sqrt({Y[i,i]:.2f} + {Y[j,j]:.2f} - 2*{Y[i,j]:.2f}) = {dist:.3f}")
    
    # Check if Y = X*X^T
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0]
    ])
    
    Y_from_X = X @ X.T
    
    print("\nIf X (positions) =")
    print(X)
    print("\nThen Y = X*X^T =")
    print(Y_from_X)
    
    print("\nDistances from X directly:")
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(X[i] - X[j])
            print(f"  d_{i}{j} = ||X_{i} - X_{j}|| = {dist:.3f}")
    
    print("\nDistances from Y_from_X:")
    for i in range(3):
        for j in range(i+1, 3):
            dist_sq = Y_from_X[i,i] + Y_from_X[j,j] - 2*Y_from_X[i,j]
            dist = np.sqrt(max(0, dist_sq))
            print(f"  d_{i}{j} = sqrt(Y_ii + Y_jj - 2*Y_ij) = {dist:.3f}")
    
    print("\n✓ Distance formula is correct when Y = X*X^T")


def check_history():
    """Check the convergence history for accounting errors"""
    
    print("\n" + "="*60)
    print("CONVERGENCE HISTORY CHECK")
    print("="*60)
    print()
    
    # Simple test case
    true_positions = np.array([[0, 0], [1, 0], [0.5, 0.866]])  # Equilateral triangle
    anchor_positions = np.array([[0, 0], [1, 0]])
    
    adjacency = np.ones((3, 3)) - np.eye(3)
    distance_measurements = {}
    
    for i in range(3):
        for j in range(i+1, 3):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            # Add small noise
            distance_measurements[(i,j)] = true_dist + np.random.normal(0, 0.001)
    
    anchor_connections = {0: [0], 1: [1], 2: []}
    distance_measurements[(0, 3)] = 0.001
    distance_measurements[(1, 4)] = 0.001
    
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=1e-6
    )
    
    config = MPSConfig(
        n_sensors=3,
        n_anchors=2,
        dimension=2,
        gamma=0.99,
        alpha=0.1,  # Small alpha
        max_iterations=100,
        tolerance=1e-8,
        verbose=False,
        early_stopping=False,
        admm_iterations=20,
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=False,
        use_2block=True,
        parallel_proximal=False,
        adaptive_alpha=False
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    result = mps.run()
    
    # Check history
    if 'history' in result and 'position_error' in result['history']:
        errors = result['history']['position_error']
        
        print(f"Error progression (first 10 iterations):")
        for i in range(min(10, len(errors))):
            # Check if error is in wrong units
            error_mm = errors[i]
            error_m = errors[i] / 1000 if errors[i] > 100 else errors[i]
            print(f"  Iter {i}: {errors[i]:.3f} (mm? or {error_m:.6f}m?)")
        
        print(f"\nFinal error: {errors[-1]:.3f}")
        
        # Check if there's a 1000x scaling issue
        if errors[-1] > 100:
            print(f"  If this is meters not mm: {errors[-1]/1000:.6f}m = {errors[-1]/1000*1000:.2f}mm")
            print("  ^ This would match the paper!")


if __name__ == "__main__":
    # Test 1: Simple case
    rmse = test_simple_case()
    
    # Test 2: Distance computation
    test_distance_computation()
    
    # Test 3: Check history
    check_history()
    
    print("\n" + "="*60)
    print("ACCOUNTING ERROR ANALYSIS")
    print("="*60)
    
    print("\nPossible issues found:")
    print("1. Check if 'position_error' in history is already in mm not meters")
    print("2. Check if the RMSE calculation is using wrong units")
    print("3. Check if final_positions are scaled incorrectly")
    print("4. Check if the paper's '40mm' is actually '40 units' in normalized space")