#!/usr/bin/env python3
"""
Test script to debug dimension issues in SDP algorithm
"""

import numpy as np
import sys
sys.path.append('/Users/maxburnett/Documents/DecentralizedLocale')

from src.core.mps_core.proximal_sdp import ProximalADMMSolver

def test_dimensions():
    """Test matrix dimensions in ADMM solver"""
    
    # Create solver
    solver = ProximalADMMSolver()
    
    # Test case
    sensor_idx = 0
    neighbors = [1, 2, 3]  # 3 neighbors
    anchors = [0, 1]  # 2 anchors
    dimension = 2
    
    # Setup matrices
    matrices = solver.setup_problem_matrices(sensor_idx, neighbors, anchors, dimension)
    
    print("Matrix dimensions:")
    print(f"  D shape: {matrices['D'].shape}")
    print(f"  N shape: {matrices['N'].shape}")
    print(f"  M shape: {matrices['M'].shape}")
    print(f"  K shape: {matrices['K'].shape}")
    print(f"  vec_dim: {matrices['vec_dim']}")
    print(f"  n_constraints: {matrices['n_constraints']}")
    
    # Expected dimensions
    n_neighbors = len(neighbors)
    n_anchors = len(anchors)
    vec_dim = 1 + 2*n_neighbors + dimension  # 1 + 2*3 + 2 = 9
    n_constraints = n_neighbors + n_anchors  # 3 + 2 = 5
    
    print("\nExpected:")
    print(f"  vec_dim: {vec_dim}")
    print(f"  n_constraints: {n_constraints}")
    
    # Check K matrix construction
    print("\nK matrix structure:")
    print(f"  K should be ({n_constraints}, {vec_dim})")
    print(f"  K actual: {matrices['K'].shape}")
    
    # Test vector sizes
    w_prev = np.zeros(vec_dim)
    c = np.zeros(n_constraints)
    lambda_admm = np.zeros(n_constraints)
    y = np.zeros(n_constraints)
    
    print("\nVector dimensions:")
    print(f"  w_prev: {w_prev.shape}")
    print(f"  c: {c.shape}")
    print(f"  lambda_admm: {lambda_admm.shape}")
    print(f"  y: {y.shape}")
    
    # Test the problematic operation
    try:
        test = lambda_admm + c - y
        print(f"  lambda + c - y: {test.shape} ✓")
    except Exception as e:
        print(f"  lambda + c - y: ERROR - {e}")
    
    # Test K.T @ (lambda + c - y)
    try:
        test2 = matrices['K'].T @ (lambda_admm + c - y)
        print(f"  K.T @ (lambda + c - y): {test2.shape} ✓")
    except Exception as e:
        print(f"  K.T @ (lambda + c - y): ERROR - {e}")

if __name__ == "__main__":
    test_dimensions()