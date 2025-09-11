#!/usr/bin/env python3
"""
Working Distributed Consensus Demo
Shows distributed achieving similar accuracy to centralized
"""

import numpy as np
import matplotlib.pyplot as plt
from src.localization.true_decentralized import TrueDecentralizedSystem


def simple_centralized(anchors, unknown_true, measurements):
    """Simple centralized least squares"""
    # For each unknown, solve using all anchor measurements
    results = []
    
    for u_idx, true_pos in enumerate(unknown_true):
        # Build least squares problem
        A = []
        b = []
        
        for a_idx, anchor_pos in enumerate(anchors):
            # Measurement from unknown to anchor
            meas_dist = measurements[len(anchors) + u_idx][a_idx]
            
            # Initial guess (center of room)
            pos = np.array([5.0, 5.0])
            
            # Iterative refinement
            for _ in range(10):
                diff = pos - anchor_pos
                est_dist = np.linalg.norm(diff)
                if est_dist > 1e-6:
                    gradient = diff / est_dist
                    error = est_dist - meas_dist
                    pos -= 0.1 * gradient * error
            
        results.append(pos)
    
    return np.array(results)


def main():
    print("=" * 60)
    print("WORKING DISTRIBUTED CONSENSUS DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Setup: 3 anchors, 2 unknowns in 10x10m room
    anchors = np.array([
        [0, 0],
        [10, 0],
        [5, 10]
    ])
    
    unknowns = np.array([
        [3, 4],
        [7, 6]
    ])
    
    all_positions = np.vstack([anchors, unknowns])
    n_nodes = len(all_positions)
    
    # Generate noisy measurements
    noise_std = 0.05
    true_distances = np.zeros((n_nodes, n_nodes))
    measured_distances = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(all_positions[i] - all_positions[j])
            meas_dist = true_dist + np.random.normal(0, noise_std)
            true_distances[i, j] = true_distances[j, i] = true_dist
            measured_distances[i, j] = measured_distances[j, i] = meas_dist
    
    print(f"\nScenario:")
    print(f"  3 anchors, 2 unknowns")
    print(f"  Room: 10x10m")
    print(f"  Noise: {noise_std*100:.0f}% ({noise_std}m std)")
    
    # CENTRALIZED
    print("\n--- Centralized (Simple Trilateration) ---")
    cent_results = simple_centralized(anchors, unknowns, measured_distances)
    
    cent_errors = []
    for i, true_pos in enumerate(unknowns):
        error = np.linalg.norm(cent_results[i] - true_pos)
        cent_errors.append(error)
        print(f"  Node {i+3}: error = {error:.3f}m")
    
    cent_rmse = np.sqrt(np.mean(np.array(cent_errors) ** 2))
    print(f"  RMSE: {cent_rmse:.3f}m")
    
    # DISTRIBUTED (using correct implementation)
    print("\n--- Distributed (True Local Information Only) ---")
    
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add anchors
    for i in range(3):
        system.add_node(i, anchors[i], is_anchor=True)
    
    # Add unknowns with random initial positions
    for i in range(2):
        initial = np.array([5.0, 5.0]) + np.random.randn(2)
        system.add_node(3 + i, initial, is_anchor=False)
    
    # Add measurements (edges)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            system.add_edge(i, j, measured_distances[i, j], variance=noise_std**2)
    
    # Run distributed algorithm
    final_positions, info = system.run(max_iterations=50, convergence_threshold=1e-4)
    
    dist_errors = []
    for i in range(2):
        node_id = 3 + i
        est_pos = final_positions[node_id]
        true_pos = unknowns[i]
        error = np.linalg.norm(est_pos - true_pos)
        dist_errors.append(error)
        print(f"  Node {node_id}: error = {error:.3f}m")
    
    dist_rmse = np.sqrt(np.mean(np.array(dist_errors) ** 2))
    print(f"  RMSE: {dist_rmse:.3f}m")
    print(f"  Converged in {info['iterations']} iterations")
    
    # RESULTS
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Centralized RMSE: {cent_rmse:.3f}m")
    print(f"Distributed RMSE: {dist_rmse:.3f}m")
    print(f"Ratio (Dist/Cent): {dist_rmse/cent_rmse:.2f}x")
    
    if dist_rmse < cent_rmse * 2:
        print("✅ SUCCESS: Distributed achieves comparable accuracy!")
    else:
        print("⚠️ Distributed needs tuning")
    
    # VISUALIZATION
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Setup
    plt.subplot(1, 2, 1)
    plt.scatter(anchors[:, 0], anchors[:, 1], marker='^', s=200, c='red', label='Anchors')
    plt.scatter(unknowns[:, 0], unknowns[:, 1], marker='o', s=100, c='blue', label='True positions')
    for i in range(3):
        plt.text(anchors[i, 0], anchors[i, 1] - 0.5, f'A{i}', ha='center')
    for i in range(2):
        plt.text(unknowns[i, 0], unknowns[i, 1] + 0.5, f'U{i+3}', ha='center')
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Network Setup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 2: Results
    plt.subplot(1, 2, 2)
    plt.scatter(anchors[:, 0], anchors[:, 1], marker='^', s=200, c='red')
    plt.scatter(unknowns[:, 0], unknowns[:, 1], marker='o', s=100, c='blue', alpha=0.5, label='True')
    
    # Centralized estimates
    for i in range(2):
        plt.scatter(cent_results[i, 0], cent_results[i, 1], 
                   marker='s', s=100, c='green', 
                   label='Centralized' if i == 0 else '')
    
    # Distributed estimates
    for i in range(2):
        pos = final_positions[3 + i]
        plt.scatter(pos[0], pos[1], 
                   marker='x', s=100, c='orange',
                   label='Distributed' if i == 0 else '')
    
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Localization Results\nCent: {cent_rmse:.3f}m, Dist: {dist_rmse:.3f}m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('working_distributed.png', dpi=150)
    plt.show()
    
    print("\n✅ Results saved to working_distributed.png")


if __name__ == "__main__":
    main()