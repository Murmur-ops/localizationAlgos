#!/usr/bin/env python3
"""
MDS-based initialization for localization
Uses all pairwise measurements to create initial positions
"""

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.localization.robust_solver import MeasurementEdge
from demo_30_nodes_large import load_config, generate_measurements, run_localization
from analyze_30_node_failures import run_with_initial_positions


def mds_initialization(measurements: List[MeasurementEdge],
                       anchors: Dict[int, np.ndarray],
                       unknowns: Dict[int, np.ndarray],
                       max_range: float = 50.0) -> np.ndarray:
    """
    Initialize positions using Multidimensional Scaling (MDS)

    This uses ALL measurements (node-to-node and anchor) to find
    positions that best preserve measured distances.
    """

    # Get all node IDs
    all_nodes = sorted(list(anchors.keys()) + list(unknowns.keys()))
    n_nodes = len(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    # Build distance matrix
    # Initialize with max_range for missing measurements
    dist_matrix = np.full((n_nodes, n_nodes), max_range)
    np.fill_diagonal(dist_matrix, 0)

    # Fill in measured distances
    n_measurements = 0
    for m in measurements:
        i = node_to_idx[m.node_i]
        j = node_to_idx[m.node_j]
        dist_matrix[i, j] = m.distance
        dist_matrix[j, i] = m.distance
        n_measurements += 1

    print(f"MDS using {n_measurements} measurements for {n_nodes} nodes")
    print(f"Distance matrix completeness: {100*n_measurements*2/(n_nodes*(n_nodes-1)):.1f}%")

    # Apply MDS
    mds = MDS(n_components=2,
             dissimilarity='precomputed',
             normalized_stress='auto',
             max_iter=500,
             random_state=42)

    positions_mds = mds.fit_transform(dist_matrix)

    print(f"MDS stress (lower is better): {mds.stress_:.3f}")

    # Extract anchor positions from MDS result
    anchor_positions_mds = []
    anchor_positions_true = []

    for aid in sorted(anchors.keys()):
        idx = node_to_idx[aid]
        anchor_positions_mds.append(positions_mds[idx])
        anchor_positions_true.append(anchors[aid])

    anchor_positions_mds = np.array(anchor_positions_mds)
    anchor_positions_true = np.array(anchor_positions_true)

    # Find optimal rotation and translation to align MDS with true anchors
    # Using Procrustes analysis

    # Center both sets
    centroid_mds = np.mean(anchor_positions_mds, axis=0)
    centroid_true = np.mean(anchor_positions_true, axis=0)

    anchors_centered_mds = anchor_positions_mds - centroid_mds
    anchors_centered_true = anchor_positions_true - centroid_true

    # Find optimal rotation using SVD
    H = anchors_centered_mds.T @ anchors_centered_true
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Find scale factor
    scale = np.sqrt(np.sum(anchors_centered_true**2) / np.sum(anchors_centered_mds**2))

    # Apply transformation to all MDS positions
    positions_aligned = scale * (positions_mds - centroid_mds) @ R.T + centroid_true

    # Extract unknown positions
    unknown_positions = []
    for uid in sorted(unknowns.keys()):
        idx = node_to_idx[uid]
        unknown_positions.extend(positions_aligned[idx])

    return np.array(unknown_positions)


def compare_initializations():
    """Compare different initialization strategies"""

    print("="*70)
    print("COMPARING INITIALIZATION STRATEGIES")
    print("="*70)

    # Load config and generate measurements
    config = load_config()
    measurements, anchors, unknowns, _ = generate_measurements(config)

    # 1. MDS initialization
    print("\n" + "="*70)
    print("MDS INITIALIZATION")
    print("="*70)

    init_mds = mds_initialization(measurements, anchors, unknowns)

    # Visualize MDS initial positions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot MDS initial positions
    ax = axes[0]
    anchor_pos = np.array(list(anchors.values()))
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=300, c='red', marker='^', label='Anchors', zorder=5)

    unknown_ids = sorted(unknowns.keys())
    for i, uid in enumerate(unknown_ids):
        true_pos = unknowns[uid]
        init_pos = init_mds[i*2:(i+1)*2]

        ax.scatter(true_pos[0], true_pos[1], s=100, c='green', marker='o', alpha=0.5)
        ax.scatter(init_pos[0], init_pos[1], s=50, c='blue', marker='x')
        ax.plot([true_pos[0], init_pos[0]], [true_pos[1], init_pos[1]], 'k--', alpha=0.3, linewidth=0.5)

    ax.set_title('MDS Initial Positions', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    ax.legend()

    # Run optimization with MDS init
    results_mds, rmse_mds = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_mds, "MDS"
    )

    # Plot MDS final results
    ax = axes[1]
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=300, c='red', marker='^', label='Anchors', zorder=5)

    for uid in unknown_ids:
        true_pos = unknowns[uid]
        est_pos = results_mds[uid]['estimated']
        error = results_mds[uid]['error']

        color = 'green' if error < 1 else ('orange' if error < 5 else 'red')
        ax.scatter(true_pos[0], true_pos[1], s=100, c='lightblue', marker='o', alpha=0.5)
        ax.scatter(est_pos[0], est_pos[1], s=50, c=color, marker='x')
        ax.plot([true_pos[0], est_pos[0]], [true_pos[1], est_pos[1]], 'k--', alpha=0.3, linewidth=0.5)

    ax.set_title(f'MDS Final Results (RMSE: {rmse_mds:.2f}m)', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)

    # 2. Compare with trilateration
    from demo_30_nodes_large import smart_initialization
    init_trilat = smart_initialization(unknowns, measurements, anchors)
    results_trilat, rmse_trilat = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_trilat, "Trilateration"
    )

    # 3. Compare with center init
    init_center = np.ones(len(unknowns) * 2) * 25
    results_center, rmse_center = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_center, "Center"
    )

    # Plot comparison bar chart
    ax = axes[2]
    methods = ['MDS', 'Trilateration', 'Center']
    rmse_values = [rmse_mds, rmse_trilat, rmse_center]
    colors = ['blue', 'orange', 'green']

    bars = ax.bar(methods, rmse_values, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE (m)', fontsize=12)
    ax.set_title('Initialization Method Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}m', ha='center', va='bottom', fontsize=11)

    plt.suptitle('MDS-based Initialization for 30-Node Localization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mds_initialization_results.png', dpi=150, bbox_inches='tight')

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"MDS:           {rmse_mds:.2f}m RMSE")
    print(f"Trilateration: {rmse_trilat:.2f}m RMSE")
    print(f"Center:        {rmse_center:.2f}m RMSE")

    # Analyze MDS initial position quality
    print("\n" + "="*70)
    print("MDS INITIAL POSITION QUALITY")
    print("="*70)

    init_errors = []
    for i, uid in enumerate(unknown_ids):
        true_pos = unknowns[uid]
        init_pos = init_mds[i*2:(i+1)*2]
        error = np.linalg.norm(init_pos - true_pos)
        init_errors.append(error)

    print(f"Initial position RMSE: {np.sqrt(np.mean(np.array(init_errors)**2)):.2f}m")
    print(f"Initial position median error: {np.median(init_errors):.2f}m")
    print(f"Initial positions < 10m error: {sum(e < 10 for e in init_errors)}/{len(init_errors)}")

    plt.show()
    print(f"\nSaved results to: mds_initialization_results.png")


if __name__ == "__main__":
    compare_initializations()