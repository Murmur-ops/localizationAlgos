#!/usr/bin/env python3
"""
Use the COMPLETE MatrixParametrizedProximalSplitting implementation
with the exact configuration from Section 3 of the paper.

THIS IS THE FULL IMPLEMENTATION WITH:
- Lifted variables (matrix S^i)
- ADMM inner solver
- 2-Block structure with PSD constraints
- Sinkhorn-Knopp matrix generation
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FULL implementation - NOT the simple one!
from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)
import matplotlib.pyplot as plt
import time


def generate_paper_network_data(seed=42):
    """
    Generate network EXACTLY as described in Section 3 of the paper.
    """
    np.random.seed(seed)
    
    n_sensors = 30  # Paper's n
    n_anchors = 6   # Paper's m
    
    # Generate positions in [0,1]²
    sensor_positions = np.random.uniform(0, 1, (n_sensors, 2))
    anchor_positions = np.random.uniform(0, 1, (n_anchors, 2))
    
    # Build adjacency and distance measurements
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Build neighborhoods: "select sensors with distance less than 0.7"
    for i in range(n_sensors):
        neighbors_found = []
        for j in range(n_sensors):
            if i != j:
                true_dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
                if true_dist < 0.7:
                    neighbors_found.append((j, true_dist))
        
        # "up to a maximum of 7"
        if len(neighbors_found) > 7:
            np.random.shuffle(neighbors_found)
            neighbors_found = neighbors_found[:7]
        
        for j, true_dist in neighbors_found:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
            
            # Apply noise: d̃ij = d⁰ij(1 + 0.05εij)
            if (min(i,j), max(i,j)) not in distance_measurements:
                epsilon = np.random.randn()
                noisy_dist = true_dist * (1 + 0.05 * epsilon)
                distance_measurements[(min(i,j), max(i,j))] = noisy_dist
    
    # Add anchor connections
    anchor_connections = {i: [] for i in range(n_sensors)}
    for i in range(n_sensors):
        for k in range(n_anchors):
            true_dist = np.linalg.norm(sensor_positions[i] - anchor_positions[k])
            if true_dist < 0.7:
                anchor_connections[i].append(k)
                epsilon = np.random.randn()
                noisy_dist = true_dist * (1 + 0.05 * epsilon)
                # Use special key format for anchors
                distance_measurements[(i, k)] = noisy_dist
    
    # Create NetworkData object
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=sensor_positions,
        measurement_variance=(0.05)**2
    )
    
    return network_data


def run_full_mps_algorithm(network_data, max_iterations=500):
    """
    Run the COMPLETE MPS algorithm with paper's exact parameters.
    """
    
    # Configure EXACTLY as in paper Section 3
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        dimension=2,
        gamma=0.999,        # Paper's γ
        alpha=10.0,         # Paper's α
        max_iterations=max_iterations,
        tolerance=1e-6,
        communication_range=0.7,
        scale=1.0,
        verbose=False,
        early_stopping=False,  # We'll track manually
        early_stopping_window=100,
        admm_iterations=100,    # Inner ADMM iterations
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=False,
        parallel_proximal=False,
        use_2block=True,        # Paper uses 2-Block
        adaptive_alpha=False,   # Use fixed α from paper
        carrier_phase_mode=False
    )
    
    print("\n" + "="*70)
    print("RUNNING FULL MPS IMPLEMENTATION")
    print("="*70)
    print("\nConfiguration (Section 3):")
    print(f"  Sensors: {config.n_sensors}")
    print(f"  Anchors: {config.n_anchors}")
    print(f"  γ = {config.gamma}")
    print(f"  α = {config.alpha}")
    print(f"  2-Block: {config.use_2block}")
    print(f"  ADMM iterations: {config.admm_iterations}")
    
    # Create the FULL algorithm instance
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    
    print("\n✓ Full implementation initialized with:")
    print("  - Lifted variables (matrix S^i)")
    print("  - ADMM inner solver")
    print("  - 2-Block structure")
    print("  - Sinkhorn-Knopp matrix generation")
    
    # Run the algorithm
    print("\nRunning algorithm...")
    start_time = time.time()
    result = mps.run(max_iterations=max_iterations)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.1f} seconds")
    
    return result, mps


def calculate_metrics(result, network_data):
    """Calculate the metrics used in the paper."""
    
    # Get final positions
    final_positions = result['final_positions']
    true_positions = network_data.true_positions
    
    # Calculate relative error: ||X̂ - X⁰||_F / ||X⁰||_F
    relative_error = np.linalg.norm(final_positions - true_positions, 'fro') / \
                    np.linalg.norm(true_positions, 'fro')
    
    # Mean distance from true locations: 1/n Σ||X̂i - X⁰i||₂
    distances = [np.linalg.norm(final_positions[i] - true_positions[i]) 
                 for i in range(len(true_positions))]
    mean_distance = np.mean(distances)
    
    return relative_error, mean_distance


def plot_results(result, network_data):
    """Create visualization similar to paper's figures."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: True positions
    ax = axes[0]
    ax.scatter(network_data.true_positions[:, 0], 
              network_data.true_positions[:, 1], 
              c='blue', s=50, label='True sensors')
    ax.scatter(network_data.anchor_positions[:, 0],
              network_data.anchor_positions[:, 1],
              c='red', s=100, marker='s', label='Anchors')
    ax.set_title('True Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Estimated positions
    ax = axes[1]
    final_pos = result['final_positions']
    ax.scatter(final_pos[:, 0], final_pos[:, 1], 
              c='green', s=50, label='Estimated')
    ax.scatter(network_data.anchor_positions[:, 0],
              network_data.anchor_positions[:, 1],
              c='red', s=100, marker='s', label='Anchors')
    ax.set_title('Estimated Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error vectors
    ax = axes[2]
    for i in range(len(network_data.true_positions)):
        true = network_data.true_positions[i]
        est = final_pos[i]
        ax.arrow(true[0], true[1], 
                est[0]-true[0], est[1]-true[1],
                head_width=0.01, head_length=0.01, 
                fc='red', ec='red', alpha=0.5)
    ax.scatter(network_data.true_positions[:, 0], 
              network_data.true_positions[:, 1], 
              c='blue', s=30, alpha=0.5)
    ax.set_title('Error Vectors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Full MPS Algorithm Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('full_mps_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_monte_carlo(n_trials=10):
    """Run multiple trials to get statistics."""
    
    print("\n" + "="*70)
    print("MONTE CARLO EVALUATION")
    print("="*70)
    print(f"\nRunning {n_trials} trials...")
    
    relative_errors = []
    mean_distances = []
    iterations_list = []
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}:")
        
        # Generate network
        network_data = generate_paper_network_data(seed=42+trial)
        
        # Run algorithm
        result, mps = run_full_mps_algorithm(network_data, max_iterations=200)
        
        # Calculate metrics
        rel_error, mean_dist = calculate_metrics(result, network_data)
        
        relative_errors.append(rel_error)
        mean_distances.append(mean_dist)
        iterations_list.append(result.get('iterations', 200))
        
        print(f"  Relative error: {rel_error:.4f}")
        print(f"  Mean distance: {mean_dist:.4f}")
        print(f"  Iterations: {result.get('iterations', 'max')}")
    
    # Statistics
    print("\n" + "-"*70)
    print("STATISTICS:")
    print(f"  Relative Error:")
    print(f"    Mean: {np.mean(relative_errors):.4f}")
    print(f"    Std:  {np.std(relative_errors):.4f}")
    print(f"    Min:  {np.min(relative_errors):.4f}")
    print(f"    Max:  {np.max(relative_errors):.4f}")
    
    return relative_errors


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("FULL MPS IMPLEMENTATION - PAPER SECTION 3")
    print("="*70)
    print("\nUsing the COMPLETE implementation with all components:")
    print("  ✓ Lifted variables")
    print("  ✓ ADMM inner solver")
    print("  ✓ 2-Block structure")
    print("  ✓ Sinkhorn-Knopp matrices")
    
    # Single run with visualization
    print("\n1. SINGLE RUN WITH VISUALIZATION")
    network_data = generate_paper_network_data(seed=42)
    result, mps = run_full_mps_algorithm(network_data, max_iterations=500)
    
    # Calculate and display metrics
    rel_error, mean_dist = calculate_metrics(result, network_data)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n✓ Relative Error: {rel_error:.4f}")
    print(f"✓ Mean Distance: {mean_dist:.4f}")
    print(f"✓ Iterations: {result.get('iterations', 'max')}")
    
    # Compare with paper
    print("\n" + "-"*70)
    print("COMPARISON WITH PAPER:")
    print(f"  Paper reports: 0.05-0.10 relative error")
    print(f"  Our result:    {rel_error:.4f}")
    
    if rel_error <= 0.10:
        print("\n  ✓✓✓ SUCCESS! We match the paper's reported performance!")
    elif rel_error <= 0.15:
        print("\n  ✓✓ Good - Close to paper's performance")
    else:
        print("\n  ✓ Reasonable - May need parameter tuning")
    
    # Visualize
    plot_results(result, network_data)
    
    # Monte Carlo for statistics
    print("\n2. MONTE CARLO EVALUATION")
    relative_errors = run_monte_carlo(n_trials=5)
    
    mean_rel_error = np.mean(relative_errors)
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if mean_rel_error <= 0.10:
        print(f"\n✓✓✓ FULL SUCCESS!")
        print(f"Mean relative error: {mean_rel_error:.4f}")
        print(f"This MATCHES the paper's reported 0.05-0.10 range!")
    else:
        print(f"\nMean relative error: {mean_rel_error:.4f}")
        print(f"Paper reports: 0.05-0.10")
        print(f"Gap: {mean_rel_error - 0.10:.4f}")
    
    print("\nThe FULL implementation with lifted variables and ADMM")
    print("is what the paper actually uses and reports results for.")


if __name__ == "__main__":
    main()