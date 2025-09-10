"""
Test script for full implementation
Verifies L matrix operations and tracking functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from snl_main import SNLProblem
from snl_main_full import FullDistributedSNL
import json
import os


def test_L_matrix_computation():
    """Test that L matrix satisfies Z = 2I - L - L^T"""
    print("Testing L matrix computation...")
    
    # Test matrices
    test_cases = [
        # Simple 2x2
        np.array([[2, -0.5], [-0.5, 2]]),
        # 3x3 case
        np.array([[2, -0.3, -0.2], [-0.3, 2, -0.4], [-0.2, -0.4, 2]]),
    ]
    
    for i, Z in enumerate(test_cases):
        n = Z.shape[0]
        print(f"\nTest case {i+1}: {n}x{n} matrix")
        print(f"Z = \n{Z}")
        
        # Compute L
        snl = FullDistributedSNL(SNLProblem())
        L = snl._compute_L_from_Z(Z)
        print(f"L = \n{L}")
        
        # Verify Z = 2I - L - L^T
        Z_reconstructed = 2 * np.eye(n) - L - L.T
        print(f"Z_reconstructed = \n{Z_reconstructed}")
        
        # Check if close
        error = np.linalg.norm(Z - Z_reconstructed, 'fro')
        print(f"Reconstruction error: {error:.2e}")
        
        assert error < 1e-10, f"L matrix computation failed, error = {error}"
    
    print("\n✓ L matrix computation test passed!")


def test_tracking_functionality():
    """Test that metrics are properly tracked"""
    print("\nTesting tracking functionality...")
    
    # Small test problem
    problem = SNLProblem(
        n_sensors=6,
        n_anchors=3,
        communication_range=0.8,
        noise_factor=0.05,
        max_iter=50,
        seed=42
    )
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        if comm.size > 1:
            # Run with MPI
            snl = FullDistributedSNL(problem)
            snl.generate_network()
            
            # Run MPS
            mps_results = snl.matrix_parametrized_splitting_full()
            
            if comm.rank == 0:
                print(f"\nMPS Results:")
                print(f"  Iterations: {snl.mps_state.iteration}")
                print(f"  Objective history length: {len(snl.mps_state.objective_history)}")
                print(f"  Error history length: {len(snl.mps_state.error_history)}")
                print(f"  Early termination: {snl.mps_state.early_termination_triggered}")
                
                assert len(snl.mps_state.objective_history) > 0, "No objective history recorded"
                assert len(snl.mps_state.error_history) > 0, "No error history recorded"
                assert snl.mps_state.iteration > 0, "No iterations recorded"
                
                print("\n✓ Tracking test passed!")
        else:
            print("  Single process mode - using simulation")
            # Use threaded simulation for testing
            from snl_simulation import ThreadedSNLSimulator
            
            sim = ThreadedSNLSimulator(problem)
            sim.generate_network(seed=42)
            
            # Test Sinkhorn-Knopp
            edge_weights = sim.run_distributed_sinkhorn_knopp()
            print(f"  Sinkhorn-Knopp completed")
            
            # Test MPS
            mps_results = sim.run_mps(max_iter=50)
            error = sim.compute_error(mps_results)
            print(f"  MPS error: {error:.6f}")
            
            sim.shutdown()
            print("\n✓ Simulation test passed!")
            
    except ImportError:
        print("  MPI not available - skipping MPI tests")


def plot_convergence_comparison(mps_history, admm_history, save_path="convergence_test.png"):
    """Plot convergence curves for verification"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Objective convergence
    ax1.semilogy(mps_history['objective_history'], 'b-', label='MPS', linewidth=2)
    ax1.semilogy(admm_history['objective_history'], 'r--', label='ADMM', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Objective Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error convergence
    if mps_history.get('error_history') and admm_history.get('error_history'):
        ax2.semilogy(mps_history['error_history'], 'b-', label='MPS', linewidth=2)
        ax2.semilogy(admm_history['error_history'], 'r--', label='ADMM', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")


def run_full_test():
    """Run complete test suite"""
    print("="*60)
    print("Running Full Implementation Tests")
    print("="*60)
    
    # Test 1: L matrix computation
    test_L_matrix_computation()
    
    # Test 2: Tracking functionality
    test_tracking_functionality()
    
    # Test 3: Full comparison (if multiple processes available)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        if comm.size >= 10:
            print("\nRunning full comparison test...")
            
            problem = SNLProblem(
                n_sensors=20,
                n_anchors=4,
                communication_range=0.7,
                noise_factor=0.05,
                max_iter=200,
                seed=42
            )
            
            snl = FullDistributedSNL(problem)
            snl.generate_network()
            
            comparison = snl.compare_algorithms_full()
            
            if comm.rank == 0 and comparison:
                # Save results
                os.makedirs('test_results', exist_ok=True)
                
                with open('test_results/full_comparison.json', 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
                
                # Plot convergence
                plot_convergence_comparison(
                    comparison['mps'],
                    comparison['admm'],
                    'test_results/convergence_comparison.png'
                )
                
                print("\n✓ Full comparison test complete!")
                print(f"  Results saved to test_results/")
        else:
            if comm.rank == 0:
                print(f"\nSkipping full comparison (need at least 10 processes, have {comm.size})")
                
    except ImportError:
        print("\nMPI not available for full comparison test")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    run_full_test()