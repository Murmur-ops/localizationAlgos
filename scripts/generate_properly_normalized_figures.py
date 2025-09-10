#!/usr/bin/env python3
"""
Generate figures with proper normalization based on actual network measurements.
Uses distributed distance consensus for realistic normalization.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig
from src.core.consensus.distance_consensus import DistanceConsensus
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
import networkx as nx

# Professional color palette
COLORS = {
    'sensor': '#3498db',
    'anchor': '#e74c3c', 
    'edge': '#95a5a6',
    'true': '#2c3e50',
    'estimated': '#27ae60',
    'error': '#e67e22',
    'primary': '#34495e',
    'grid': '#ecf0f1'
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5


def setup_network_with_consensus():
    """
    Create network and run distance consensus for proper normalization
    """
    
    # Generate network
    config = MPSConfig(
        n_sensors=20,
        n_anchors=4,
        scale=10.0,  # 10 meter × 10 meter physical space
        communication_range=0.35,  # 3.5 meters
        noise_factor=0.01,
        gamma=0.999,
        alpha=0.5,
        max_iterations=500,
        tolerance=1e-6,
        dimension=2,
        seed=42
    )
    
    mps = MPSAlgorithm(config)
    mps.generate_network()
    
    # Build adjacency matrix from actual measurements
    n = config.n_sensors
    adjacency = np.zeros((n, n))
    for (i, j), _ in mps.distance_measurements.items():
        if i < n and j < n:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    
    # Run distance consensus
    consensus = DistanceConsensus(n, adjacency)
    
    # Add all measured distances
    for (i, j), dist in mps.distance_measurements.items():
        if i < n and j < n:
            consensus.add_distance_measurement(i, j, dist)
    
    # Run consensus protocol
    consensus_results = consensus.run_consensus()
    
    # Get normalization factor
    max_measured = consensus_results['global_max_distance']
    
    # Also calculate true maximum for comparison
    true_max = 0
    for i in range(n):
        for j in range(i+1, n):
            true_dist = np.linalg.norm(
                np.array(mps.true_positions[i]) - np.array(mps.true_positions[j])
            )
            true_max = max(true_max, true_dist)
    
    # Run localization
    result = mps.run()
    
    return mps, result, consensus_results, max_measured, true_max


def figure1_network_topology_normalized():
    """
    Network topology with proper normalization annotations
    """
    
    mps, _, consensus_results, max_measured, true_max = setup_network_with_consensus()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    n_sensors = mps.config.n_sensors
    positions = mps.true_positions
    anchor_pos = mps.anchor_positions
    
    # --- Left: Physical units ---
    ax = ax1
    
    # Draw edges
    for (i, j), dist in mps.distance_measurements.items():
        if i < n_sensors and j < n_sensors:
            ax.plot([positions[i][0], positions[j][0]], 
                   [positions[i][1], positions[j][1]],
                   color=COLORS['edge'], linewidth=1, alpha=0.4, zorder=1)
    
    # Draw sensors
    sensor_x = [positions[i][0] for i in range(n_sensors)]
    sensor_y = [positions[i][1] for i in range(n_sensors)]
    
    ax.scatter(sensor_x, sensor_y, s=150, c=COLORS['sensor'],
              edgecolors=COLORS['primary'], linewidth=2,
              label=f'Sensors (n={n_sensors})', zorder=3, alpha=0.9)
    
    # Draw anchors
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=400, c=COLORS['anchor'], marker='s',
              edgecolors=COLORS['primary'], linewidth=3,
              label=f'Anchors (m={len(anchor_pos)})', zorder=5)
    
    # Labels and styling
    ax.set_xlabel('X Position (meters)', fontweight='bold')
    ax.set_ylabel('Y Position (meters)', fontweight='bold')
    ax.set_title('Physical Coordinates', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Add info box
    info_text = (f'Network Size: {mps.config.scale:.1f}m × {mps.config.scale:.1f}m\n'
                f'Comm Range: {mps.config.communication_range * mps.config.scale:.1f}m\n'
                f'Max Distance: {true_max:.2f}m')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                edgecolor=COLORS['primary'], linewidth=2, alpha=0.95)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', bbox=props)
    
    # --- Right: Normalized by consensus ---
    ax = ax2
    
    # Normalize positions
    norm_factor = 1.0 / max_measured
    norm_positions = {i: positions[i] * norm_factor for i in range(n_sensors)}
    norm_anchors = anchor_pos * norm_factor
    
    # Draw edges
    for (i, j), dist in mps.distance_measurements.items():
        if i < n_sensors and j < n_sensors:
            ax.plot([norm_positions[i][0], norm_positions[j][0]], 
                   [norm_positions[i][1], norm_positions[j][1]],
                   color=COLORS['edge'], linewidth=1, alpha=0.4, zorder=1)
    
    # Draw sensors
    sensor_x = [norm_positions[i][0] for i in range(n_sensors)]
    sensor_y = [norm_positions[i][1] for i in range(n_sensors)]
    
    ax.scatter(sensor_x, sensor_y, s=150, c=COLORS['sensor'],
              edgecolors=COLORS['primary'], linewidth=2, zorder=3, alpha=0.9)
    
    # Draw anchors
    ax.scatter(norm_anchors[:, 0], norm_anchors[:, 1],
              s=400, c=COLORS['anchor'], marker='s',
              edgecolors=COLORS['primary'], linewidth=3, zorder=5)
    
    # Labels and styling
    ax.set_xlabel('Normalized Position (max measured distance = 1)', fontweight='bold')
    ax.set_ylabel('Normalized Position', fontweight='bold')
    ax.set_title('Consensus-Normalized Coordinates', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add consensus info
    consensus_text = (f'Consensus Results:\n'
                     f'Converged: {consensus_results["converged"]}\n'
                     f'Iterations: {consensus_results["iterations"]}\n'
                     f'Max (measured): {max_measured:.2f}m\n'
                     f'Max (true): {true_max:.2f}m\n'
                     f'Error: {abs(max_measured-true_max)/true_max*100:.1f}%')
    
    ax.text(0.02, 0.98, consensus_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', bbox=props)
    
    fig.suptitle('Network Topology: Physical vs Distributed Normalization',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def figure2_localization_with_proper_units():
    """
    Localization results with clear unit labeling
    """
    
    mps, result, consensus_results, max_measured, true_max = setup_network_with_consensus()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    n_sensors = mps.config.n_sensors
    true_pos = np.array([mps.true_positions[i] for i in range(n_sensors)])
    est_pos = np.array([result['estimated_positions'][i] for i in range(n_sensors)])
    anchor_pos = mps.anchor_positions
    
    # Calculate errors in meters
    errors_m = np.array([np.linalg.norm(est_pos[i] - true_pos[i]) 
                        for i in range(n_sensors)])
    
    # --- Top left: Physical coordinates (meters) ---
    ax = axes[0, 0]
    
    # Plot true and estimated
    ax.scatter(true_pos[:, 0], true_pos[:, 1],
              s=100, c=COLORS['true'], marker='o', alpha=0.6,
              label='True', zorder=2)
    ax.scatter(est_pos[:, 0], est_pos[:, 1],
              s=100, c=COLORS['estimated'], marker='^', alpha=0.6,
              label='Estimated', zorder=3)
    
    # Error vectors
    for i in range(n_sensors):
        ax.plot([true_pos[i, 0], est_pos[i, 0]],
               [true_pos[i, 1], est_pos[i, 1]],
               color=COLORS['error'], alpha=0.5, linewidth=1, zorder=1)
    
    # Anchors
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=400, c=COLORS['anchor'], marker='s',
              edgecolors=COLORS['primary'], linewidth=3,
              label='Anchors', zorder=4)
    
    ax.set_xlabel('X (meters)', fontweight='bold')
    ax.set_ylabel('Y (meters)', fontweight='bold')
    ax.set_title('Localization in Physical Units', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add RMSE in meters
    rmse_m = np.sqrt(np.mean(errors_m**2))
    ax.text(0.98, 0.02, f'RMSE: {rmse_m:.3f}m\n({rmse_m*1000:.1f}mm)',
           transform=ax.transAxes, ha='right', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor=COLORS['primary'], linewidth=2))
    
    # --- Top right: Normalized coordinates ---
    ax = axes[0, 1]
    
    norm_factor = 1.0 / max_measured
    norm_true = true_pos * norm_factor
    norm_est = est_pos * norm_factor
    norm_anchors = anchor_pos * norm_factor
    
    ax.scatter(norm_true[:, 0], norm_true[:, 1],
              s=100, c=COLORS['true'], marker='o', alpha=0.6,
              label='True', zorder=2)
    ax.scatter(norm_est[:, 0], norm_est[:, 1],
              s=100, c=COLORS['estimated'], marker='^', alpha=0.6,
              label='Estimated', zorder=3)
    
    # Anchors
    ax.scatter(norm_anchors[:, 0], norm_anchors[:, 1],
              s=400, c=COLORS['anchor'], marker='s',
              edgecolors=COLORS['primary'], linewidth=3, zorder=4)
    
    ax.set_xlabel('Normalized X (max dist = 1)', fontweight='bold')
    ax.set_ylabel('Normalized Y', fontweight='bold')
    ax.set_title('Localization in Normalized Units', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add normalized RMSE
    rmse_norm = rmse_m * norm_factor
    ax.text(0.98, 0.02, f'Normalized RMSE: {rmse_norm:.4f}\n({rmse_norm*100:.2f}% of max)',
           transform=ax.transAxes, ha='right', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=COLORS['primary'], linewidth=2))
    
    # --- Bottom left: Error distribution (meters) ---
    ax = axes[1, 0]
    
    ax.hist(errors_m, bins=15, color=COLORS['error'], alpha=0.7,
           edgecolor=COLORS['primary'], linewidth=2)
    ax.axvline(rmse_m, color='red', linestyle='--', linewidth=2,
              label=f'RMSE: {rmse_m:.3f}m')
    ax.set_xlabel('Localization Error (meters)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom right: Error as percentage ---
    ax = axes[1, 1]
    
    # Error as percentage of different references
    errors_pct_network = (errors_m / mps.config.scale) * 100  # % of network size
    errors_pct_max = (errors_m / max_measured) * 100  # % of max measured distance
    errors_pct_comm = (errors_m / (mps.config.communication_range * mps.config.scale)) * 100  # % of comm range
    
    x_labels = ['% of Network\nSize', '% of Max\nDistance', '% of Comm\nRange']
    data = [errors_pct_network, errors_pct_max, errors_pct_comm]
    
    bp = ax.boxplot(data, labels=x_labels, patch_artist=True,
                    boxprops=dict(facecolor=COLORS['sensor'], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.set_ylabel('Error (%)', fontweight='bold')
    ax.set_title('Relative Error Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(i+1, mean_val, f'{mean_val:.1f}%', ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Localization Performance with Proper Units',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def figure3_convergence_with_units():
    """
    Convergence analysis with clear unit specifications
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Run multiple trials
    n_trials = 20
    all_histories = []
    all_scales = []
    
    for trial in range(n_trials):
        config = MPSConfig(
            n_sensors=20, n_anchors=4,
            scale=10.0,  # 10m × 10m
            communication_range=0.35,
            noise_factor=0.01,
            gamma=0.999, alpha=0.5,
            max_iterations=500,
            tolerance=1e-6,
            dimension=2, seed=trial
        )
        
        mps = MPSAlgorithm(config)
        mps.generate_network()
        result = mps.run()
        
        if 'rmse_history' in result:
            all_histories.append(result['rmse_history'])
            all_scales.append(config.scale)
    
    # --- Top left: RMSE in meters ---
    ax = axes[0, 0]
    
    for history, scale in zip(all_histories[:5], all_scales[:5]):  # Show first 5
        iterations = np.arange(len(history)) * 10
        rmse_m = np.array(history)  # Already in meters
        ax.plot(iterations, rmse_m, alpha=0.5, linewidth=1)
    
    # Add median
    max_len = max(len(h) for h in all_histories)
    padded = []
    for h in all_histories:
        padded.append(h + [h[-1]] * (max_len - len(h)))
    
    median_rmse = np.median(np.array(padded), axis=0)
    iterations = np.arange(len(median_rmse)) * 10
    
    ax.plot(iterations, median_rmse, color='red', linewidth=3,
           label='Median', zorder=10)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('RMSE (meters)', fontweight='bold')
    ax.set_title('Convergence in Physical Units', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 500])
    
    # --- Top right: RMSE as percentage ---
    ax = axes[0, 1]
    
    for history, scale in zip(all_histories[:5], all_scales[:5]):
        iterations = np.arange(len(history)) * 10
        rmse_pct = (np.array(history) / scale) * 100
        ax.plot(iterations, rmse_pct, alpha=0.5, linewidth=1)
    
    median_pct = (median_rmse / all_scales[0]) * 100
    ax.plot(iterations, median_pct, color='red', linewidth=3,
           label='Median', zorder=10)
    
    ax.axhline(y=4, color='green', linestyle='--', linewidth=2,
              label='4% Target', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('RMSE (% of network size)', fontweight='bold')
    ax.set_title('Convergence as Percentage', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 10])
    
    # --- Bottom left: Final errors in different units ---
    ax = axes[1, 0]
    
    final_rmse_m = [h[-1] for h in all_histories]
    final_rmse_mm = [r * 1000 for r in final_rmse_m]
    final_rmse_pct = [(r / all_scales[0]) * 100 for r in final_rmse_m]
    
    x_pos = [1, 2, 3]
    labels = ['Millimeters', 'Meters', 'Percentage']
    
    # Create separate boxplots
    bp1 = ax.boxplot([final_rmse_mm], positions=[1], widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor='lightblue'))
    
    # Use secondary axis for different scales
    ax2 = ax.twinx()
    bp2 = ax2.boxplot([final_rmse_m], positions=[2], widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    bp3 = ax3.boxplot([final_rmse_pct], positions=[3], widths=0.6,
                      patch_artist=True, boxprops=dict(facecolor='lightyellow'))
    
    ax.set_xlabel('Unit Type', fontweight='bold')
    ax.set_ylabel('Error (mm)', fontweight='bold', color='blue')
    ax2.set_ylabel('Error (m)', fontweight='bold', color='green')
    ax3.set_ylabel('Error (%)', fontweight='bold', color='orange')
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_title('Final Error in Different Units', fontweight='bold')
    
    # --- Bottom right: Summary table ---
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary statistics
    summary_data = [
        ['Metric', 'Value', 'Units'],
        ['', '', ''],
        ['Mean RMSE', f'{np.mean(final_rmse_m):.3f}', 'meters'],
        ['Mean RMSE', f'{np.mean(final_rmse_mm):.1f}', 'millimeters'],
        ['Mean RMSE', f'{np.mean(final_rmse_pct):.2f}', '% of network'],
        ['', '', ''],
        ['Network Size', f'{all_scales[0]:.1f} × {all_scales[0]:.1f}', 'meters'],
        ['Comm Range', f'{0.35 * all_scales[0]:.1f}', 'meters'],
        ['', '', ''],
        ['Best Trial', f'{min(final_rmse_mm):.1f}', 'mm'],
        ['Worst Trial', f'{max(final_rmse_mm):.1f}', 'mm'],
    ]
    
    # Create table
    table = ax.table(cellText=summary_data, loc='center',
                    cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics', fontweight='bold', fontsize=14, pad=20)
    
    fig.suptitle('Convergence Analysis with Proper Units',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def main():
    """Generate all properly normalized figures"""
    
    print("="*70)
    print("GENERATING PROPERLY NORMALIZED FIGURES")
    print("="*70)
    
    print("\n1. Creating network topology figure...")
    fig1 = figure1_network_topology_normalized()
    fig1.savefig('figure1_network_proper_normalization.png', 
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved: figure1_network_proper_normalization.png")
    
    print("\n2. Creating localization results figure...")
    fig2 = figure2_localization_with_proper_units()
    fig2.savefig('figure2_localization_proper_units.png',
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved: figure2_localization_proper_units.png")
    
    print("\n3. Creating convergence analysis figure...")
    fig3 = figure3_convergence_with_units()
    fig3.savefig('figure3_convergence_proper_units.png',
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved: figure3_convergence_proper_units.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("FIGURES WITH PROPER NORMALIZATION COMPLETE")
    print("="*70)
    print("\n✓ Physical units clearly labeled (meters)")
    print("✓ Normalized units explained (max distance = 1)")
    print("✓ Consensus-based normalization demonstrated")
    print("✓ Multiple unit perspectives shown")
    print("\n✓ Distributed normalization protocol implemented!")


if __name__ == "__main__":
    main()