#!/usr/bin/env python3
"""
Monte Carlo simulation to assess RMSE as a function of center initialization position
Tests how sensitive the localization is to the choice of "center" for initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from demo_30_nodes_large import load_config, generate_measurements


def run_with_center_init(config, measurements, anchors, unknowns, center_x, center_y):
    """Run localization with all nodes initialized at (center_x, center_y)"""

    # Initialize solver
    solver = RobustLocalizer(
        dimension=2,
        huber_delta=config['solver']['huber_delta']
    )
    solver.max_iterations = 30  # Limit iterations for speed
    solver.convergence_threshold = config['system']['convergence_threshold']

    # Create center initialization
    n_unknowns = len(unknowns)
    initial_positions = np.empty(n_unknowns * 2)
    for i in range(n_unknowns):
        initial_positions[i*2] = center_x
        initial_positions[i*2 + 1] = center_y

    # Filter measurements
    filtered = [m for m in measurements if m.node_i in unknowns or m.node_j in unknowns]

    # Remap IDs
    unknown_ids = sorted(unknowns.keys())
    id_mapping = {aid: aid for aid in anchors}
    for i, uid in enumerate(unknown_ids):
        id_mapping[uid] = len(anchors) + i

    remapped_measurements = []
    for m in filtered:
        remapped = MeasurementEdge(
            node_i=id_mapping[m.node_i],
            node_j=id_mapping[m.node_j],
            distance=m.distance,
            quality=m.quality,
            variance=m.variance
        )
        remapped_measurements.append(remapped)

    remapped_anchors = {id_mapping[aid]: anchors[aid] for aid in anchors}

    # Solve
    try:
        optimized_positions, info = solver.solve(
            initial_positions,
            remapped_measurements,
            remapped_anchors
        )

        # Calculate RMSE
        errors = []
        for i, uid in enumerate(unknown_ids):
            est_pos = optimized_positions[i*2:(i+1)*2]
            true_pos = unknowns[uid]
            error = np.linalg.norm(est_pos - true_pos)
            errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        return rmse

    except Exception as e:
        print(f"Failed at center ({center_x:.1f}, {center_y:.1f}): {e}")
        return 100.0  # Return large error for failed cases


def monte_carlo_simulation():
    """Run Monte Carlo simulation over different center positions"""

    print("="*70)
    print("MONTE CARLO SIMULATION: CENTER INITIALIZATION SENSITIVITY")
    print("="*70)

    # Load configuration
    config = load_config()

    # Grid of center positions to test
    center_x_values = np.linspace(0, 50, 11)  # 11 points from 0 to 50 (every 5m)
    center_y_values = np.linspace(0, 50, 11)

    # Number of random seeds per center position
    n_seeds = 3  # Reduced for speed

    # Storage for results
    rmse_grid = np.zeros((len(center_y_values), len(center_x_values)))
    std_grid = np.zeros((len(center_y_values), len(center_x_values)))

    print(f"\nTesting {len(center_x_values)} x {len(center_y_values)} = {len(center_x_values)*len(center_y_values)} center positions")
    print(f"With {n_seeds} random seeds each = {len(center_x_values)*len(center_y_values)*n_seeds} total runs")
    print("\nThis will take a few minutes...\n")

    # Progress tracking
    total_runs = len(center_y_values) * len(center_x_values)
    run_count = 0

    # Pre-generate measurements for all seeds (same across all centers)
    all_measurements = []
    print("Pre-generating measurements for all seeds...")
    for seed in range(n_seeds):
        np.random.seed(42 + seed)
        measurements, anchors, unknowns, _ = generate_measurements(config)
        all_measurements.append((measurements, anchors, unknowns))

    print(f"\nTesting centers...")
    for i, cy in enumerate(center_y_values):
        for j, cx in enumerate(center_x_values):
            rmse_values = []

            # Test with each pre-generated measurement set
            for seed in range(n_seeds):
                measurements, anchors, unknowns = all_measurements[seed]

                # Run localization with center at (cx, cy)
                rmse = run_with_center_init(config, measurements, anchors, unknowns, cx, cy)
                rmse_values.append(rmse)

            # Store mean and std
            rmse_grid[i, j] = np.mean(rmse_values)
            std_grid[i, j] = np.std(rmse_values)

            run_count += 1
            if run_count % 10 == 0:
                print(f"Progress: {run_count}/{total_runs} positions tested...")

    return center_x_values, center_y_values, rmse_grid, std_grid


def visualize_results(cx_vals, cy_vals, rmse_grid, std_grid):
    """Create visualization of RMSE as function of center position"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. RMSE heatmap
    ax = axes[0, 0]
    im1 = ax.imshow(rmse_grid, extent=[0, 50, 0, 50], origin='lower',
                    cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=30)

    # Mark anchors
    anchors_x = [0, 50, 50, 0]
    anchors_y = [0, 0, 50, 50]
    ax.scatter(anchors_x, anchors_y, s=200, c='blue', marker='^',
               edgecolors='black', linewidth=2, label='Anchors', zorder=5)

    # Mark best center
    min_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    best_cx = cx_vals[min_idx[1]]
    best_cy = cy_vals[min_idx[0]]
    ax.scatter(best_cx, best_cy, s=200, c='white', marker='*',
               edgecolors='black', linewidth=2, label=f'Best: ({best_cx:.0f},{best_cy:.0f})', zorder=5)

    plt.colorbar(im1, ax=ax, label='RMSE (m)')
    ax.set_xlabel('Center X (m)')
    ax.set_ylabel('Center Y (m)')
    ax.set_title('RMSE vs Center Position')
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)

    # 2. Log-scale RMSE (to see fine structure)
    ax = axes[0, 1]
    im2 = ax.imshow(rmse_grid + 0.1, extent=[0, 50, 0, 50], origin='lower',
                    cmap='RdYlGn_r', aspect='equal', norm=LogNorm(vmin=0.1, vmax=30))
    ax.scatter(anchors_x, anchors_y, s=200, c='blue', marker='^',
               edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(best_cx, best_cy, s=200, c='white', marker='*',
               edgecolors='black', linewidth=2, zorder=5)
    plt.colorbar(im2, ax=ax, label='RMSE (m, log scale)')
    ax.set_xlabel('Center X (m)')
    ax.set_ylabel('Center Y (m)')
    ax.set_title('RMSE (Log Scale)')
    ax.grid(True, alpha=0.3)

    # 3. Standard deviation
    ax = axes[1, 0]
    im3 = ax.imshow(std_grid, extent=[0, 50, 0, 50], origin='lower',
                    cmap='viridis', aspect='equal')
    ax.scatter(anchors_x, anchors_y, s=200, c='red', marker='^',
               edgecolors='white', linewidth=2, zorder=5)
    plt.colorbar(im3, ax=ax, label='Std Dev (m)')
    ax.set_xlabel('Center X (m)')
    ax.set_ylabel('Center Y (m)')
    ax.set_title('Standard Deviation Across Seeds')
    ax.grid(True, alpha=0.3)

    # 4. Cross sections through center
    ax = axes[1, 1]

    # Horizontal slice through y=25
    center_y_idx = len(cy_vals) // 2
    ax.plot(cx_vals, rmse_grid[center_y_idx, :], 'b-',
            label=f'Horizontal (y={cy_vals[center_y_idx]:.0f}m)', linewidth=2)

    # Vertical slice through x=25
    center_x_idx = len(cx_vals) // 2
    ax.plot(cy_vals, rmse_grid[:, center_x_idx], 'r-',
            label=f'Vertical (x={cx_vals[center_x_idx]:.0f}m)', linewidth=2)

    # Diagonal slice
    diag_rmse = np.diag(rmse_grid)
    diag_pos = np.linspace(0, 50, len(diag_rmse))
    ax.plot(diag_pos, diag_rmse, 'g--',
            label='Diagonal', linewidth=2)

    ax.set_xlabel('Position (m)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE Cross Sections')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])

    plt.suptitle('Monte Carlo Analysis: RMSE vs Center Initialization Position\n' +
                 f'30 nodes, 4 anchors, 50×50m area, 5 seeds per position',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def main():
    """Run the Monte Carlo simulation and analysis"""

    # Run simulation
    cx_vals, cy_vals, rmse_grid, std_grid = monte_carlo_simulation()

    # Analysis
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)

    # Find best and worst centers
    min_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    max_idx = np.unravel_index(np.argmax(rmse_grid), rmse_grid.shape)

    best_rmse = rmse_grid[min_idx]
    worst_rmse = rmse_grid[max_idx]
    best_cx = cx_vals[min_idx[1]]
    best_cy = cy_vals[min_idx[0]]
    worst_cx = cx_vals[max_idx[1]]
    worst_cy = cy_vals[max_idx[0]]

    print(f"\nBest center position:  ({best_cx:.1f}, {best_cy:.1f}) -> RMSE = {best_rmse:.2f}m")
    print(f"Worst center position: ({worst_cx:.1f}, {worst_cy:.1f}) -> RMSE = {worst_rmse:.2f}m")
    print(f"True geometric center: (25.0, 25.0) -> RMSE = {rmse_grid[10, 10]:.2f}m")

    # Count good vs bad initializations
    good_count = np.sum(rmse_grid < 1.0)
    medium_count = np.sum((rmse_grid >= 1.0) & (rmse_grid < 10.0))
    bad_count = np.sum(rmse_grid >= 10.0)
    total_count = rmse_grid.size

    print(f"\nPerformance distribution:")
    print(f"  Excellent (<1m):   {good_count}/{total_count} ({100*good_count/total_count:.1f}%)")
    print(f"  Medium (1-10m):    {medium_count}/{total_count} ({100*medium_count/total_count:.1f}%)")
    print(f"  Poor (>10m):       {bad_count}/{total_count} ({100*bad_count/total_count:.1f}%)")

    # Analyze symmetry
    print(f"\nSymmetry analysis:")
    center_idx = len(cx_vals) // 2

    # Check if center is best
    if min_idx[0] == center_idx and min_idx[1] == center_idx:
        print("  Geometric center (25,25) IS the optimal initialization!")
    else:
        print(f"  Geometric center (25,25) is NOT optimal")
        print(f"  Optimal is offset by ({best_cx-25:.1f}, {best_cy-25:.1f})m")

    # Visualize
    fig = visualize_results(cx_vals, cy_vals, rmse_grid, std_grid)
    plt.savefig('monte_carlo_center_init_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: monte_carlo_center_init_results.png")

    plt.show()


if __name__ == "__main__":
    main()