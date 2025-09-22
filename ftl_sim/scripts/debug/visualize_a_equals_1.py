#!/usr/bin/env python3
"""
Visualize FTL consensus performance when a=1.0
Shows ideal vs actual vs estimated positions
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

def run_a_equals_1_visualization():
    """Run consensus with a=1.0 and create detailed visualization"""

    np.random.seed(42)  # For reproducibility

    # Parameters
    error_scale = 1.0  # a = 1.0
    area_size = 50
    n_nodes = 30
    n_anchors = 5
    n_unknowns = 25
    comm_range = 25
    measurement_noise = 0.01  # 1 cm
    init_noise = 0.5  # 50 cm

    # Ideal positions
    ideal_anchors = np.array([
        [0, 0],
        [area_size, 0],
        [area_size, area_size],
        [0, area_size],
        [area_size/2, area_size/2]
    ])

    # Ideal unknown positions (grid)
    grid_size = int(np.ceil(np.sqrt(n_unknowns)))
    margin = 5
    x = np.linspace(margin, area_size-margin, grid_size)
    y = np.linspace(margin, area_size-margin, grid_size)

    ideal_unknowns = []
    for xi in x:
        for yi in y:
            ideal_unknowns.append([xi, yi])
            if len(ideal_unknowns) >= n_unknowns:
                break
        if len(ideal_unknowns) >= n_unknowns:
            break
    ideal_unknowns = np.array(ideal_unknowns[:n_unknowns])

    # Apply position errors: x = x_ideal + 1.0 * uniform(1, 10)
    unknown_errors = error_scale * np.random.uniform(1, 10, ideal_unknowns.shape)
    actual_unknowns = np.clip(ideal_unknowns + unknown_errors, 0, area_size)

    # True positions (where nodes actually are)
    true_positions = np.vstack([ideal_anchors, actual_unknowns])

    print("="*70)
    print("RUNNING FTL CONSENSUS WITH a=1.0")
    print("="*70)
    print(f"Position errors: x = x_ideal + 1.0 * uniform(1, 10) meters")
    print(f"Average position error introduced: {np.mean(np.linalg.norm(unknown_errors, axis=1)):.1f} cm")
    print(f"Max position error introduced: {np.max(np.linalg.norm(unknown_errors, axis=1)):.1f} cm")

    # Create solver
    config = ConsensusGNConfig(
        max_iterations=500,
        consensus_gain=0.05,
        step_size=0.3,
        gradient_tol=1e-5,
        step_tol=1e-6,
        verbose=True
    )
    cgn = ConsensusGaussNewton(config)

    # Add nodes
    for i in range(n_nodes):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = ideal_anchors[i]
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Initial guess with noise
            state[:2] = actual_unknowns[i-n_anchors] + np.random.normal(0, init_noise, 2)
            cgn.add_node(i, state, is_anchor=False)

    # Add measurements based on actual positions
    n_edges = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)
                n_edges += 1
                meas_range = dist + np.random.normal(0, measurement_noise)
                cgn.add_measurement(ToAFactorMeters(i, j, meas_range, measurement_noise**2))

    print(f"\nNetwork: {n_edges} edges, avg degree: {2*n_edges/n_nodes:.1f}")

    # Set true positions for evaluation
    cgn.set_true_positions({i: actual_unknowns[i-n_anchors] for i in range(n_anchors, n_nodes)})

    # Run optimization
    print("\nRunning optimization...")
    results = cgn.optimize()

    # Extract estimated positions
    estimated_positions = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        estimated_positions[i] = cgn.nodes[i].state[:2]

    # Calculate errors
    actual_errors = []
    for i in range(n_unknowns):
        idx = i + n_anchors
        estimated = estimated_positions[idx]
        actual = actual_unknowns[i]
        actual_errors.append(np.linalg.norm(estimated - actual))

    rmse_actual = np.sqrt(np.mean(np.array(actual_errors)**2))

    print(f"\nResults:")
    print(f"  RMSE: {rmse_actual*100:.2f} cm")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Network with actual positions
    ax = axes[0]
    # Draw edges
    for i, j in cgn.edges:
        ax.plot([true_positions[i, 0], true_positions[j, 0]],
                [true_positions[i, 1], true_positions[j, 1]],
                'gray', alpha=0.2, linewidth=0.5)
    # Plot nodes
    ax.scatter(ideal_anchors[:, 0], ideal_anchors[:, 1],
               s=200, c='red', marker='s', label='Anchors', zorder=5)
    ax.scatter(actual_unknowns[:, 0], actual_unknowns[:, 1],
               s=100, c='blue', marker='o', alpha=0.6, label='Actual Unknown Positions', zorder=4)
    ax.scatter(ideal_unknowns[:, 0], ideal_unknowns[:, 1],
               s=50, c='gray', marker='x', alpha=0.5, label='Ideal Grid Positions', zorder=3)

    # Draw displacement vectors
    for i in range(n_unknowns):
        ax.arrow(ideal_unknowns[i, 0], ideal_unknowns[i, 1],
                 actual_unknowns[i, 0] - ideal_unknowns[i, 0],
                 actual_unknowns[i, 1] - ideal_unknowns[i, 1],
                 head_width=0.5, head_length=0.3, fc='yellow', ec='orange', alpha=0.3)

    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Position Errors Applied (a=1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Estimated vs Actual
    ax = axes[1]
    # Draw edges based on estimated positions
    for i, j in cgn.edges:
        if i >= n_anchors and j >= n_anchors:
            ax.plot([estimated_positions[i, 0], estimated_positions[j, 0]],
                    [estimated_positions[i, 1], estimated_positions[j, 1]],
                    'gray', alpha=0.2, linewidth=0.5)

    ax.scatter(ideal_anchors[:, 0], ideal_anchors[:, 1],
               s=200, c='red', marker='s', label='Anchors', zorder=5)
    ax.scatter(actual_unknowns[:, 0], actual_unknowns[:, 1],
               s=100, c='blue', marker='o', alpha=0.6, label='Actual Positions', zorder=4)
    ax.scatter(estimated_positions[n_anchors:, 0], estimated_positions[n_anchors:, 1],
               s=100, c='green', marker='^', label='Estimated Positions', zorder=5)

    # Error vectors from actual to estimated
    for i in range(n_unknowns):
        idx = i + n_anchors
        ax.plot([actual_unknowns[i, 0], estimated_positions[idx, 0]],
                [actual_unknowns[i, 1], estimated_positions[idx, 1]],
                'r-', alpha=0.5, linewidth=1)

    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Estimated vs Actual (RMSE: {rmse_actual*100:.1f} cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Error distribution
    ax = axes[2]
    # Histogram
    counts, bins, patches = ax.hist(np.array(actual_errors)*100, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(rmse_actual*100, color='red', linestyle='--', linewidth=2, label=f'RMSE: {rmse_actual*100:.1f} cm')
    ax.axvline(np.mean(actual_errors)*100, color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(actual_errors)*100:.1f} cm')
    ax.set_xlabel('Localization Error (cm)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"Min: {np.min(actual_errors)*100:.1f} cm\nMax: {np.max(actual_errors)*100:.1f} cm\nStd: {np.std(actual_errors)*100:.1f} cm"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('FTL Consensus with Large Position Errors (a = 1.0)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('consensus_a_equals_1.png', dpi=150, bbox_inches='tight')
    plt.show()

    return estimated_positions, actual_unknowns, ideal_unknowns

if __name__ == "__main__":
    estimated, actual, ideal = run_a_equals_1_visualization()
    print("\nVisualization saved to consensus_a_equals_1.png")