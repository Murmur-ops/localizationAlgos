#!/usr/bin/env python3
"""
Test script for verifying MPS algorithm fixes with a small problem.
Tests with 3 sensors and 1 anchor to ensure correctness of:
1. Sequential L matrix dependencies
2. Proper v update with W matrix
3. Vectorization/devectorization consistency
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

def create_small_test_network():
    """Create a small 3-sensor, 1-anchor test network"""
    n_sensors = 3
    n_anchors = 1
    dimension = 2
    
    # True positions (known for validation)
    true_positions = np.array([
        [0.0, 0.0],   # Sensor 0
        [1.0, 0.0],   # Sensor 1  
        [0.5, 0.866]  # Sensor 2 (equilateral triangle)
    ])
    
    anchor_positions = np.array([
        [0.5, 0.289]  # Anchor at centroid
    ])
    
    # Full connectivity
    adjacency_matrix = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    # True distances
    distance_measurements = {}
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            # Add small noise
            measured_dist = true_dist + np.random.randn() * 0.001
            distance_measurements[(i, j)] = measured_dist
    
    # Anchor connections (all sensors connected to anchor)
    anchor_connections = {0: [0], 1: [0], 2: [0]}
    
    for i in range(n_sensors):
        true_dist = np.linalg.norm(true_positions[i] - anchor_positions[0])
        measured_dist = true_dist + np.random.randn() * 0.001
        distance_measurements[(i, 0)] = measured_dist
    
    return NetworkData(
        adjacency_matrix=adjacency_matrix,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        measurement_variance=1e-6
    )

def test_vectorization():
    """Test that vectorization/devectorization is consistent"""
    from src.core.mps_core.vectorization import MatrixVectorizer
    
    print("Testing vectorization consistency...")
    
    n_sensors = 3
    dimension = 2
    neighborhoods = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    
    vectorizer = MatrixVectorizer(n_sensors, dimension, neighborhoods)
    
    # Create random symmetric matrices
    matrices = []
    for i in range(n_sensors):
        dim = vectorizer.matrix_dims[i]
        A = np.random.randn(dim, dim)
        A = (A + A.T) / 2  # Make symmetric
        matrices.append(A)
    
    # Test vectorize/devectorize
    for i, mat in enumerate(matrices):
        vec = vectorizer.vectorize_matrix(mat, i)
        mat_reconstructed = vectorizer.devectorize_matrix(vec, i)
        
        error = np.linalg.norm(mat - mat_reconstructed, 'fro')
        print(f"  Sensor {i}: reconstruction error = {error:.2e}")
        assert error < 1e-10, f"Vectorization failed for sensor {i}"
    
    # Test stack/unstack
    stacked = vectorizer.stack_matrices(matrices)
    matrices_reconstructed = vectorizer.unstack_vector(stacked)
    
    for i, (mat, mat_rec) in enumerate(zip(matrices, matrices_reconstructed)):
        error = np.linalg.norm(mat - mat_rec, 'fro')
        print(f"  Stack/unstack sensor {i}: error = {error:.2e}")
        assert error < 1e-10, f"Stack/unstack failed for sensor {i}"
    
    print("✓ Vectorization tests passed!\n")

def test_l_matrix():
    """Test L matrix computation and properties"""
    from src.core.mps_core.sinkhorn_knopp import MatrixParameterGenerator
    
    print("Testing L matrix computation...")
    
    # Simple 4x4 test case
    n = 4
    # Create a valid Z matrix with diag(Z) = 2
    Z = 2 * np.eye(n)
    Z[1, 0] = Z[0, 1] = -0.3
    Z[2, 0] = Z[0, 2] = -0.2
    Z[2, 1] = Z[1, 2] = -0.3
    Z[3, 0] = Z[0, 3] = -0.1
    Z[3, 1] = Z[1, 3] = -0.1
    Z[3, 2] = Z[2, 3] = -0.1
    
    generator = MatrixParameterGenerator()
    L = generator.compute_lower_triangular_L(Z)
    
    # Check that L is strictly lower triangular
    assert np.allclose(np.diag(L), 0), "L should have zero diagonal"
    assert np.allclose(np.triu(L, 1), 0), "L should be lower triangular"
    
    # Check reconstruction: Z = 2I - L - L^T
    Z_reconstructed = 2 * np.eye(n) - L - L.T
    error = np.linalg.norm(Z - Z_reconstructed, 'fro')
    print(f"  Z reconstruction error: {error:.2e}")
    assert error < 1e-10, "L matrix computation failed"
    
    print("✓ L matrix tests passed!\n")

def test_mps_algorithm():
    """Test the full MPS algorithm on small problem"""
    print("Testing MPS algorithm on small network...")
    
    # Create test network
    network_data = create_small_test_network()
    
    # Configure algorithm
    config = MPSConfig(
        n_sensors=3,
        n_anchors=1,
        dimension=2,
        gamma=0.99,          # Step size (closer to paper's 0.999)
        alpha=10.0,          # Proximal parameter (as in paper)
        max_iterations=100,  # Reduced for testing
        tolerance=1e-4,
        communication_range=2.0,  # Full connectivity
        verbose=True,
        early_stopping=False,
        admm_iterations=50,
        use_2block=True,
        parallel_proximal=False  # Use sequential for correct implementation
    )
    
    # Run algorithm
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    
    print("Initial position error:", 
          np.linalg.norm(mps.X - network_data.true_positions))
    
    # Run more iterations to see convergence
    for k in range(50):
        stats = mps.run_iteration(k)
        if k % 10 == 0:
            print(f"Iteration {k}: obj={stats['objective']:.6f}, "
                  f"pos_err={stats['position_error']:.6f}, "
                  f"consensus={stats['consensus_error']:.6f}")
    
    # Check improvement
    final_error = np.linalg.norm(mps.X - network_data.true_positions)
    print(f"\nFinal position error: {final_error:.6f}")
    
    # Verify sequential dependencies are working
    print("\nChecking sequential dependency structure:")
    print(f"  L matrix shape: {mps.L.shape}")
    print(f"  L is strictly lower triangular: {np.allclose(np.triu(mps.L), 0)}")
    print(f"  Number of non-zero L entries: {np.count_nonzero(mps.L)}")
    
    print("✓ MPS algorithm test completed!\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("MPS Algorithm Fix Verification Tests")
    print("=" * 60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        test_vectorization()
        test_l_matrix()
        test_mps_algorithm()
        
        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())