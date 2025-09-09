#!/usr/bin/env python3
"""
Detailed test to understand MPS algorithm behavior and debug issues.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)

def create_simple_network():
    """Create the simplest possible network: 2 sensors, 1 anchor"""
    n_sensors = 2
    n_anchors = 1
    dimension = 2
    
    # True positions
    true_positions = np.array([
        [0.0, 0.0],   # Sensor 0
        [1.0, 0.0],   # Sensor 1
    ])
    
    anchor_positions = np.array([
        [0.5, 0.5]    # Anchor
    ])
    
    # Full connectivity
    adjacency_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    
    # Exact distances (no noise for debugging)
    distance_measurements = {}
    distance_measurements[(0, 1)] = 1.0  # Distance between sensors
    distance_measurements[(0, 0)] = np.linalg.norm(true_positions[0] - anchor_positions[0])
    distance_measurements[(1, 0)] = np.linalg.norm(true_positions[1] - anchor_positions[0])
    
    anchor_connections = {0: [0], 1: [0]}
    
    return NetworkData(
        adjacency_matrix=adjacency_matrix,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=0.0  # No noise
    )

def analyze_algorithm_behavior():
    """Analyze the algorithm behavior step by step"""
    print("=" * 60)
    print("Detailed MPS Algorithm Analysis")
    print("=" * 60 + "\n")
    
    # Create simple network
    network_data = create_simple_network()
    
    # Configure algorithm with minimal settings
    config = MPSConfig(
        n_sensors=2,
        n_anchors=1,
        dimension=2,
        gamma=0.5,           # Moderate step size for debugging
        alpha=1.0,           # Simple proximal parameter
        max_iterations=5,    # Just a few iterations
        tolerance=1e-6,
        communication_range=2.0,
        verbose=False,
        early_stopping=False,
        admm_iterations=10,
        use_2block=True,
        parallel_proximal=False,
        adaptive_alpha=False  # Disable adaptation for debugging
    )
    
    # Initialize algorithm
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    
    print("Initial state:")
    print(f"  True positions:\n{network_data.true_positions}")
    print(f"  Initial X:\n{mps.X}")
    print(f"  Initial error: {np.linalg.norm(mps.X - network_data.true_positions):.6f}")
    
    # Analyze W and Z matrices
    print(f"\nMatrix parameters:")
    print(f"  Z matrix shape: {mps.Z.shape}")
    print(f"  W matrix shape: {mps.W.shape}")
    print(f"  L matrix shape: {mps.L.shape}")
    
    # Check W matrix properties
    print(f"\nW matrix properties:")
    ones = np.ones(mps.W.shape[0])
    W_ones = mps.W @ ones
    print(f"  null(W) check (W*1): max = {np.max(np.abs(W_ones)):.6f}")
    
    eigenvalues_W = np.linalg.eigvals(mps.W)
    print(f"  W eigenvalues range: [{np.min(np.real(eigenvalues_W)):.4f}, "
          f"{np.max(np.real(eigenvalues_W)):.4f}]")
    
    # Check Z-W properties
    Z_minus_W = mps.Z - mps.W
    eigenvalues_ZW = np.linalg.eigvals(Z_minus_W)
    print(f"  Z-W min eigenvalue: {np.min(np.real(eigenvalues_ZW)):.6f}")
    
    print("\nRunning iterations:")
    print("-" * 50)
    
    for k in range(5):
        # Store state before iteration
        x_before = [x.copy() for x in mps.x]
        v_before = [v.copy() for v in mps.v]
        
        # Run iteration
        stats = mps.run_iteration(k)
        
        print(f"\nIteration {k}:")
        print(f"  Objective: {stats['objective']:.6f}")
        print(f"  Position error: {stats['position_error']:.6f}")
        print(f"  Consensus error: {stats['consensus_error']:.6f}")
        print(f"  PSD violation: {stats['psd_violation']:.6f}")
        
        # Analyze changes
        x_change = sum(np.linalg.norm(mps.x[i] - x_before[i], 'fro') 
                      for i in range(len(mps.x)))
        v_change = sum(np.linalg.norm(mps.v[i] - v_before[i], 'fro') 
                      for i in range(len(mps.v)))
        
        print(f"  Total x change: {x_change:.6f}")
        print(f"  Total v change: {v_change:.6f}")
        
        # Show current position estimates
        print(f"  Current X:\n{mps.X}")
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)

def test_without_2block():
    """Test without 2-block structure to isolate issues"""
    print("\nTesting without 2-block structure:")
    print("-" * 40)
    
    network_data = create_simple_network()
    
    config = MPSConfig(
        n_sensors=2,
        n_anchors=1,
        dimension=2,
        gamma=0.5,
        alpha=1.0,
        max_iterations=10,
        tolerance=1e-6,
        communication_range=2.0,
        verbose=False,
        early_stopping=False,
        admm_iterations=10,
        use_2block=False,  # Disable 2-block
        parallel_proximal=False,
        adaptive_alpha=False
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    
    print(f"Initial error: {np.linalg.norm(mps.X - network_data.true_positions):.6f}")
    
    for k in range(10):
        stats = mps.run_iteration(k)
        if k % 2 == 0:
            print(f"Iteration {k}: pos_err={stats['position_error']:.6f}, "
                  f"obj={stats['objective']:.6f}")
    
    print(f"Final error: {np.linalg.norm(mps.X - network_data.true_positions):.6f}")

def main():
    np.random.seed(42)
    
    try:
        analyze_algorithm_behavior()
        test_without_2block()
        print("\n✓ Analysis completed")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())