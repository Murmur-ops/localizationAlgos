#!/usr/bin/env python3
"""
Simple example of using the sensor network localization algorithm
No MPI required - runs on a single machine
"""

import numpy as np
import matplotlib.pyplot as plt
from snl_threaded_standalone import SNLProblem, ThreadedSNLFull

def main():
    """Run a simple localization example"""
    
    print("=== Simple Sensor Network Localization Example ===\n")
    
    # Step 1: Configure the problem
    print("Step 1: Setting up network configuration...")
    problem = SNLProblem(
        n_sensors=20,          # 20 sensors to localize
        n_anchors=4,           # 4 anchors with known positions
        d=2,                   # 2D localization
        communication_range=0.4,  # 40% of area width
        noise_factor=0.05,     # 5% measurement noise
        gamma=0.999,           # Algorithm stability parameter
        alpha_mps=10.0,        # Algorithm speed parameter
        max_iter=100,          # Maximum iterations
        seed=42                # For reproducible results
    )
    print(f"  - Sensors: {problem.n_sensors}")
    print(f"  - Anchors: {problem.n_anchors}")
    print(f"  - Noise: {problem.noise_factor*100}%")
    
    # Step 2: Create and run solver
    print("\nStep 2: Initializing distributed solver...")
    solver = ThreadedSNLFull(problem)
    
    print("\nStep 3: Generating random sensor network...")
    solver.generate_network(seed=42)
    
    # Get initial positions (random initialization)
    initial_positions = np.zeros((problem.n_sensors, problem.d))
    for i, sensor in solver.sensors.items():
        initial_positions[i] = sensor.sensor_data.X
    
    # Calculate initial error
    initial_errors = []
    for i in range(problem.n_sensors):
        error = np.linalg.norm(initial_positions[i] - solver.true_positions[i])
        initial_errors.append(error)
    initial_rmse = np.sqrt(np.mean(np.array(initial_errors)**2))
    print(f"  - Initial RMSE: {initial_rmse:.4f}")
    
    print("\nStep 4: Running MPS localization algorithm...")
    print("  (Note: Threading version is slower than MPI)")
    print("  Progress: Running... (this may take 10-30 seconds)")
    
    # Run algorithm
    mps_results = solver.matrix_parametrized_splitting_threaded()
    
    # Get final positions
    final_positions = np.zeros((problem.n_sensors, problem.d))
    for i, (X, Y) in mps_results.items():
        final_positions[i] = X
    
    # Calculate final error
    final_errors = []
    for i in range(problem.n_sensors):
        error = np.linalg.norm(final_positions[i] - solver.true_positions[i])
        final_errors.append(error)
    final_rmse = np.sqrt(np.mean(np.array(final_errors)**2))
    
    print(f"\n✓ Algorithm converged in {solver.mps_state.iteration} iterations")
    print(f"  - Final RMSE: {final_rmse:.4f}")
    print(f"  - Improvement: {(initial_rmse - final_rmse)/initial_rmse * 100:.1f}%")
    
    # Step 5: Visualize results
    print("\nStep 5: Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Initial state
    ax1.set_title('Initial Network State')
    ax1.scatter(solver.anchor_positions[:, 0], solver.anchor_positions[:, 1], 
               c='red', s=200, marker='^', label='Anchors', zorder=5)
    ax1.scatter(solver.true_positions[:, 0], solver.true_positions[:, 1], 
               c='green', s=100, alpha=0.6, label='True Positions')
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], 
               c='blue', s=50, marker='x', label='Initial Estimates')
    
    # Draw initial errors
    for i in range(problem.n_sensors):
        ax1.plot([solver.true_positions[i, 0], initial_positions[i, 0]], 
                [solver.true_positions[i, 1], initial_positions[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: Final results
    ax2.set_title('After MPS Localization')
    ax2.scatter(solver.anchor_positions[:, 0], solver.anchor_positions[:, 1], 
               c='red', s=200, marker='^', label='Anchors', zorder=5)
    ax2.scatter(solver.true_positions[:, 0], solver.true_positions[:, 1], 
               c='green', s=100, alpha=0.6, label='True Positions')
    ax2.scatter(final_positions[:, 0], final_positions[:, 1], 
               c='blue', s=100, alpha=0.8, label='MPS Estimates')
    
    # Draw final errors (much smaller)
    for i in range(problem.n_sensors):
        ax2.plot([solver.true_positions[i, 0], final_positions[i, 0]], 
                [solver.true_positions[i, 1], final_positions[i, 1]], 
                'gray', alpha=0.5, linewidth=1)
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add text boxes with metrics
    ax1.text(0.02, 0.98, f'RMSE: {initial_rmse:.3f}', 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    ax2.text(0.02, 0.98, f'RMSE: {final_rmse:.3f}', 
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('simple_example_results.png', dpi=150, bbox_inches='tight')
    print("  - Saved visualization to 'simple_example_results.png'")
    
    # Show convergence
    if solver.mps_state.error_history:
        fig2, ax = plt.subplots(figsize=(8, 6))
        iterations = range(len(solver.mps_state.error_history))
        ax.semilogy(iterations, solver.mps_state.error_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (log scale)')
        ax.set_title('Algorithm Convergence')
        ax.grid(True, alpha=0.3)
        plt.savefig('simple_example_convergence.png', dpi=150, bbox_inches='tight')
        print("  - Saved convergence plot to 'simple_example_convergence.png'")
    
    print("\n=== Example Complete! ===")
    print("\nNext steps:")
    print("1. For faster execution, use the MPI version:")
    print("   mpirun -np 4 python3 snl_mpi_optimized.py")
    print("2. Try changing the number of sensors or anchors")
    print("3. Adjust the noise_factor to see how it affects accuracy")
    print("4. Modify communication_range to change network connectivity")
    
    # Clean up
    solver.shutdown()

if __name__ == "__main__":
    main()