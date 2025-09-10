#!/usr/bin/env python3
"""
Generate three separate, focused figures:
1. Network topology
2. Estimated vs true positions
3. Convergence rate
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx

# Professional color palette
COLORS = {
    'sensor': '#3498db',      # Bright blue
    'anchor': '#e74c3c',      # Red
    'edge': '#95a5a6',        # Gray
    'true': '#2c3e50',        # Dark blue-gray
    'estimated': '#27ae60',   # Green
    'error': '#e67e22',       # Orange
    'primary': '#34495e',     # Dark gray
    'grid': '#ecf0f1',        # Light gray
    'success': '#2ecc71',     # Light green
    'convergence': '#9b59b6'  # Purple
}

# Set clean matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = COLORS['primary']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.color'] = COLORS['grid']
plt.rcParams['grid.linewidth'] = 0.8


def setup_network():
    """Create and run MPS algorithm to get network data."""
    
    config = MPSConfig(
        n_sensors=20,
        n_anchors=4,
        scale=1.0,
        communication_range=0.35,
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
    result = mps.run()
    
    return mps, result


def figure1_network_topology():
    """
    Figure 1: Network Topology
    Shows the communication graph structure and anchor placement
    """
    
    mps, _ = setup_network()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get positions
    n_sensors = mps.config.n_sensors
    positions = {i: mps.true_positions[i] for i in range(n_sensors)}
    anchor_pos = mps.anchor_positions
    
    # Create networkx graph for better visualization
    G = nx.Graph()
    for i in range(n_sensors):
        G.add_node(i, pos=positions[i])
    
    # Add edges based on communication range
    for (i, j), _ in mps.distance_measurements.items():
        if i < n_sensors and j < n_sensors:
            G.add_edge(i, j)
    
    # Draw edges first (bottom layer)
    for edge in G.edges():
        i, j = edge
        x_coords = [positions[i][0], positions[j][0]]
        y_coords = [positions[i][1], positions[j][1]]
        ax.plot(x_coords, y_coords, color=COLORS['edge'], 
                linewidth=1, alpha=0.4, zorder=1)
    
    # Draw sensors
    sensor_x = [positions[i][0] for i in range(n_sensors)]
    sensor_y = [positions[i][1] for i in range(n_sensors)]
    
    # Calculate node degrees for sizing
    degrees = dict(G.degree())
    node_sizes = [100 + degrees[i] * 20 for i in range(n_sensors)]
    
    scatter = ax.scatter(sensor_x, sensor_y, s=node_sizes, 
                        c=[degrees[i] for i in range(n_sensors)],
                        cmap='YlOrRd', vmin=0, vmax=max(degrees.values()),
                        edgecolors=COLORS['primary'], linewidth=2,
                        zorder=3, alpha=0.9)
    
    # Add sensor labels
    for i in range(n_sensors):
        ax.annotate(f'{i}', xy=positions[i], 
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=8,
                   color='white', weight='bold', zorder=4)
    
    # Draw anchors
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
              s=400, c=COLORS['anchor'], marker='s',
              edgecolors=COLORS['primary'], linewidth=3,
              zorder=5, label='Anchors')
    
    # Add anchor labels
    for i, pos in enumerate(anchor_pos):
        ax.annotate(f'A{i}', xy=pos,
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=11,
                   color='white', weight='bold', zorder=6)
    
    # Highlight communication range with circles
    for i in range(0, n_sensors, 5):  # Show for every 5th sensor
        circle = Circle(positions[i], mps.config.communication_range,
                       fill=False, edgecolor=COLORS['sensor'],
                       linestyle='--', linewidth=1, alpha=0.2)
        ax.add_patch(circle)
    
    # Styling
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Network Topology and Communication Graph', 
                fontweight='bold', fontsize=18, pad=20)
    
    ax.set_xlim([-0.15, 1.15])
    ax.set_ylim([-0.15, 1.15])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for node degree
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Node Degree (Connections)', fontweight='bold', fontsize=11)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], s=200, c=COLORS['sensor'], 
                   edgecolors=COLORS['primary'], linewidth=2,
                   label=f'Sensors (n={n_sensors})'),
        plt.scatter([], [], s=400, c=COLORS['anchor'], marker='s',
                   edgecolors=COLORS['primary'], linewidth=3,
                   label=f'Anchors (m={len(anchor_pos)})'),
        plt.Line2D([0], [0], color=COLORS['edge'], linewidth=2,
                  alpha=0.4, label=f'Communication Links')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=False, shadow=True, framealpha=0.95)
    
    # Add info box
    info_text = (f'Communication Range: {mps.config.communication_range:.2f}m\n'
                f'Total Edges: {G.number_of_edges()}\n'
                f'Avg Degree: {np.mean(list(degrees.values())):.1f}')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                edgecolor=COLORS['primary'], linewidth=2, alpha=0.95)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=props)
    
    plt.tight_layout()
    return fig


def figure2_estimated_vs_true():
    """
    Figure 2: Estimated vs True Positions
    Shows the localization results with error visualization
    """
    
    mps, result = setup_network()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    n_sensors = mps.config.n_sensors
    true_pos = np.array([mps.true_positions[i] for i in range(n_sensors)])
    est_pos = np.array([result['estimated_positions'][i] for i in range(n_sensors)])
    anchor_pos = mps.anchor_positions
    
    # Calculate errors
    errors = np.array([np.linalg.norm(est_pos[i] - true_pos[i]) 
                      for i in range(n_sensors)])
    errors_mm = errors * 1000  # Convert to mm
    
    # --- Left panel: Overlay of true and estimated ---
    
    # Draw error vectors first (bottom layer)
    for i in range(n_sensors):
        if errors[i] > 0.005:  # Only show visible errors
            ax1.plot([true_pos[i, 0], est_pos[i, 0]], 
                    [true_pos[i, 1], est_pos[i, 1]],
                    color=COLORS['error'], alpha=0.5, linewidth=1.5,
                    zorder=1)
    
    # True positions
    ax1.scatter(true_pos[:, 0], true_pos[:, 1],
               s=120, c=COLORS['true'], marker='o',
               edgecolors='white', linewidth=2,
               label='True Positions', zorder=2, alpha=0.8)
    
    # Estimated positions
    ax1.scatter(est_pos[:, 0], est_pos[:, 1],
               s=120, c=COLORS['estimated'], marker='^',
               edgecolors='white', linewidth=2,
               label='Estimated Positions', zorder=3, alpha=0.8)
    
    # Anchors
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
               s=400, c=COLORS['anchor'], marker='s',
               edgecolors=COLORS['primary'], linewidth=3,
               label='Anchors', zorder=4)
    
    # Styling
    ax1.set_xlabel('X Position (m)', fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontweight='bold')
    ax1.set_title('Position Estimates Overlay', fontweight='bold', fontsize=16)
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    
    # Add RMSE info
    rmse = np.sqrt(np.mean(errors**2)) * 1000
    info_text = f'RMSE: {rmse:.1f} mm\nMean Error: {np.mean(errors_mm):.1f} mm\nMax Error: {np.max(errors_mm):.1f} mm'
    props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['success'] if rmse < 50 else COLORS['error'],
                alpha=0.2, edgecolor=COLORS['primary'], linewidth=2)
    ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes,
            fontsize=11, horizontalalignment='right',
            verticalalignment='bottom', bbox=props, weight='bold')
    
    # --- Right panel: Error heatmap ---
    
    # Create scatter plot with error coloring
    scatter = ax2.scatter(true_pos[:, 0], true_pos[:, 1],
                         s=300, c=errors_mm, cmap='RdYlGn_r',
                         vmin=0, vmax=np.percentile(errors_mm, 95),
                         edgecolors=COLORS['primary'], linewidth=2,
                         zorder=2)
    
    # Add error circles proportional to error
    for i in range(n_sensors):
        circle = Circle(true_pos[i], errors[i]*5,  # Scale for visibility
                       fill=False, edgecolor=COLORS['error'],
                       linestyle='--', linewidth=1, alpha=0.5)
        ax2.add_patch(circle)
    
    # Add sensor numbers
    for i in range(n_sensors):
        ax2.annotate(f'{i}', xy=true_pos[i],
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=9,
                    color='white', weight='bold', zorder=3)
    
    # Anchors
    ax2.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
               s=400, c=COLORS['anchor'], marker='s',
               edgecolors=COLORS['primary'], linewidth=3,
               zorder=4)
    
    # Styling
    ax2.set_xlabel('X Position (m)', fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontweight='bold')
    ax2.set_title('Localization Error Distribution', fontweight='bold', fontsize=16)
    ax2.set_xlim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02, fraction=0.046)
    cbar.set_label('Error (mm)', fontweight='bold')
    
    # Add statistics box
    percentiles = np.percentile(errors_mm, [25, 50, 75, 90])
    stats_text = (f'Error Statistics\n' + '─'*15 + '\n'
                 f'25th: {percentiles[0]:.1f} mm\n'
                 f'50th: {percentiles[1]:.1f} mm\n'
                 f'75th: {percentiles[2]:.1f} mm\n'
                 f'90th: {percentiles[3]:.1f} mm')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                edgecolor=COLORS['primary'], linewidth=2, alpha=0.95)
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)
    
    fig.suptitle('Localization Performance Analysis', 
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def figure3_convergence_rate():
    """
    Figure 3: Convergence Rate
    Shows the algorithm convergence over iterations
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Run multiple trials for statistics
    n_trials = 30
    all_rmse_histories = []
    all_obj_histories = []
    
    for trial in range(n_trials):
        config = MPSConfig(
            n_sensors=20, n_anchors=4, scale=1.0,
            communication_range=0.35, noise_factor=0.01,
            gamma=0.999, alpha=0.5,
            max_iterations=500, tolerance=1e-6,
            dimension=2, seed=trial
        )
        
        mps = MPSAlgorithm(config)
        mps.generate_network()
        result = mps.run()
        
        if 'rmse_history' in result:
            all_rmse_histories.append(result['rmse_history'])
        if 'objective_history' in result:
            all_obj_histories.append(result['objective_history'])
    
    # --- Top left: RMSE convergence ---
    ax = axes[0, 0]
    
    # Prepare data
    max_len = max(len(h) for h in all_rmse_histories)
    padded_rmse = []
    for h in all_rmse_histories:
        h_mm = [val * 1000 for val in h]  # Convert to mm
        padded = h_mm + [h_mm[-1]] * (max_len - len(h_mm))
        padded_rmse.append(padded)
    
    rmse_array = np.array(padded_rmse)
    iterations = np.arange(len(rmse_array[0])) * 10
    
    # Calculate statistics
    median_rmse = np.median(rmse_array, axis=0)
    q1_rmse = np.percentile(rmse_array, 25, axis=0)
    q3_rmse = np.percentile(rmse_array, 75, axis=0)
    min_rmse = np.min(rmse_array, axis=0)
    max_rmse = np.max(rmse_array, axis=0)
    
    # Plot
    ax.plot(iterations, median_rmse, color=COLORS['convergence'], 
           linewidth=3, label='Median', zorder=3)
    ax.fill_between(iterations, q1_rmse, q3_rmse, 
                    alpha=0.3, color=COLORS['convergence'], 
                    label='IQR (25-75%)', zorder=2)
    ax.fill_between(iterations, min_rmse, max_rmse, 
                    alpha=0.1, color=COLORS['convergence'], 
                    label='Min-Max', zorder=1)
    
    # Target line
    ax.axhline(y=40, color=COLORS['success'], linestyle='--', 
              linewidth=2, label='Target (40mm)', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('RMSE (mm)', fontweight='bold')
    ax.set_title('RMSE Convergence', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, max(100, np.max(q3_rmse[:50]))])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # --- Top right: Convergence rate (log scale) ---
    ax = axes[0, 1]
    
    # Plot on log scale
    ax.semilogy(iterations, median_rmse, color=COLORS['convergence'],
               linewidth=3, label='RMSE')
    ax.fill_between(iterations, q1_rmse, q3_rmse,
                    alpha=0.3, color=COLORS['convergence'])
    
    # Add convergence rate annotation
    # Find linear region in log scale (iterations 50-200)
    if len(iterations) > 200:
        idx_start, idx_end = 5, 20  # iterations 50-200
        log_rmse = np.log(median_rmse[idx_start:idx_end])
        x_fit = iterations[idx_start:idx_end]
        rate = np.polyfit(x_fit, log_rmse, 1)[0]
        
        ax.text(0.6, 0.8, f'Convergence Rate:\n{abs(rate):.4f} per iteration',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=COLORS['primary'], linewidth=2))
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('RMSE (mm, log scale)', fontweight='bold')
    ax.set_title('Convergence Rate Analysis', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 500])
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='upper right')
    
    # --- Bottom left: Iteration to convergence ---
    ax = axes[1, 0]
    
    # Find convergence iteration for each trial (first time below 50mm)
    convergence_iters = []
    for rmse_hist in all_rmse_histories:
        rmse_mm = [val * 1000 for val in rmse_hist]
        conv_idx = next((i for i, val in enumerate(rmse_mm) if val < 50), len(rmse_mm))
        convergence_iters.append(conv_idx * 10)
    
    # Histogram
    ax.hist(convergence_iters, bins=20, color=COLORS['convergence'],
           alpha=0.7, edgecolor=COLORS['primary'], linewidth=2)
    
    mean_conv = np.mean(convergence_iters)
    ax.axvline(mean_conv, color=COLORS['error'], linestyle='-',
              linewidth=2, label=f'Mean: {mean_conv:.0f} iterations')
    
    ax.set_xlabel('Iterations to Convergence (<50mm)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Convergence Time Distribution', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right')
    
    # --- Bottom right: Final error distribution ---
    ax = axes[1, 1]
    
    final_rmses = [h[-1] * 1000 for h in all_rmse_histories]
    
    # Violin plot
    parts = ax.violinplot([final_rmses], positions=[1], widths=0.6,
                          showmeans=True, showmedians=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['convergence'])
        pc.set_alpha(0.7)
    
    # Style the other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor(COLORS['primary'])
            parts[partname].set_linewidth(2)
    
    # Add individual points
    y_jitter = np.random.normal(1, 0.04, len(final_rmses))
    ax.scatter(y_jitter, final_rmses, alpha=0.4, s=30, 
              color=COLORS['primary'])
    
    # Target line
    ax.axhline(y=40, color=COLORS['success'], linestyle='--',
              linewidth=2, label='Target (40mm)', alpha=0.7)
    
    # Statistics
    stats_text = (f'Mean: {np.mean(final_rmses):.1f} mm\n'
                 f'Std: {np.std(final_rmses):.1f} mm\n'
                 f'Min: {np.min(final_rmses):.1f} mm\n'
                 f'Max: {np.max(final_rmses):.1f} mm')
    
    ax.text(1.4, np.max(final_rmses) * 0.9, stats_text, 
           fontsize=10, bbox=dict(boxstyle='round', facecolor='white',
                                 edgecolor=COLORS['primary'], linewidth=2))
    
    ax.set_ylabel('Final RMSE (mm)', fontweight='bold')
    ax.set_title('Final Error Distribution', fontweight='bold', fontsize=14)
    ax.set_xticks([1])
    ax.set_xticklabels([f'{n_trials} Trials'])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right')
    
    fig.suptitle('Algorithm Convergence Analysis', 
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def main():
    """Generate all three separate figures."""
    
    print("="*70)
    print("GENERATING SEPARATE FOCUSED FIGURES")
    print("="*70)
    
    print("\n1. Creating network topology figure...")
    fig1 = figure1_network_topology()
    fig1.savefig('figure1_network_topology.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   ✓ Saved: figure1_network_topology.png")
    
    print("\n2. Creating estimated vs true positions figure...")
    fig2 = figure2_estimated_vs_true()
    fig2.savefig('figure2_estimated_vs_true.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   ✓ Saved: figure2_estimated_vs_true.png")
    
    print("\n3. Creating convergence rate figure...")
    fig3 = figure3_convergence_rate()
    fig3.savefig('figure3_convergence_rate.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   ✓ Saved: figure3_convergence_rate.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("SEPARATE FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print("\n✓ Figure 1: Network topology with communication graph")
    print("✓ Figure 2: Estimated vs true positions with error visualization")
    print("✓ Figure 3: Convergence rate analysis with statistics")
    print("\nAll figures are publication-ready with consistent styling")


if __name__ == "__main__":
    main()