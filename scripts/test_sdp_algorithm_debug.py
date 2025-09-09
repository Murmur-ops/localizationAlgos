#!/usr/bin/env python3
"""
Debug version of SDP algorithm to find dimension issues
"""

import numpy as np
import sys
sys.path.append('/Users/maxburnett/Documents/DecentralizedLocale')

from src.core.mps_core.algorithm_sdp import SDPBasedMPSAlgorithm, MPSConfig

def test_with_debug():
    """Test SDP algorithm with debugging"""
    
    config = MPSConfig(
        n_sensors=10,  # Smaller for debugging
        n_anchors=3,
        verbose=True,
        seed=42,
        max_iterations=2,  # Just 2 iterations for debugging
        admm_iterations=2   # Minimal ADMM iterations
    )
    
    algo = SDPBasedMPSAlgorithm(config)
    algo.generate_network()
    
    print(f"Network generated: {config.n_sensors} sensors, {config.n_anchors} anchors")
    
    # Initialize variables
    X, Y = algo.initialize_lifted_variables()
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Check distance measurements
    n_measurements = len(algo.distance_measurements)
    n_anchor_measurements = sum(len(d) for d in algo.anchor_distances.values())
    print(f"Distance measurements: {n_measurements//2} pairs")
    print(f"Anchor measurements: {n_anchor_measurements}")
    
    # Test one sensor's ADMM setup
    for i in range(config.n_sensors):
        neighbors = [j for j in range(config.n_sensors) if (i, j) in algo.distance_measurements]
        anchors = [k for k in algo.anchor_distances.get(i, {}).keys()] if i in algo.anchor_distances else []
        
        print(f"\nSensor {i}:")
        print(f"  Neighbors: {neighbors} (count: {len(neighbors)})")
        print(f"  Anchors: {anchors} (count: {len(anchors)})")
        
        if len(neighbors) == 0 and len(anchors) == 0:
            print("  Skipping - no connections")
            continue
        
        # Setup problem matrices
        matrices = algo.prox_ops.admm_solver.setup_problem_matrices(
            i, neighbors, anchors, config.dimension
        )
        
        print(f"  Matrix dimensions:")
        print(f"    K: {matrices['K'].shape}")
        print(f"    n_constraints: {matrices['n_constraints']}")
        print(f"    vec_dim: {matrices['vec_dim']}")
        
        # Check if dimensions match
        expected_constraints = len(neighbors) + len(anchors)
        if matrices['n_constraints'] != expected_constraints:
            print(f"  ERROR: n_constraints mismatch! Expected {expected_constraints}, got {matrices['n_constraints']}")
        
        # Try to create the constraint vector
        distances_sensors = {j: algo.distance_measurements[(i, j)] for j in neighbors}
        distances_anchors = algo.anchor_distances.get(i, {})
        
        c = np.zeros(matrices['n_constraints'])
        print(f"  c shape: {c.shape}")
        
        # Check anchors are valid
        for k in anchors:
            if k >= config.n_anchors:
                print(f"  WARNING: Anchor index {k} >= n_anchors {config.n_anchors}")
        
        # Break after first few sensors for debugging
        if i >= 2:
            break
    
    print("\nAttempting to run algorithm...")
    try:
        results = algo.run()
        print(f"Success! Final error: {results['final_relative_error']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_debug()