#!/usr/bin/env python3
"""
Visualization of estimated vs actual positions for 20 nodes with 8 anchors
Shows the localization performance visually
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from mps_core import MPSAlgorithm, MPSConfig


def visualize_estimated_vs_actual():
    """Create detailed visualization of estimated vs actual positions"""
    
    # Configuration for 20 nodes with 8 anchors
    config = MPSConfig(
        n_sensors=20,
        n_anchors=8,
        communication_range=0.35,
        noise_factor=0.05,
        gamma=0.98,
        alpha=1.2,
        max_iterations=400,
        tolerance=0.00005,
        dimension=2,
        seed=2024  # Fixed seed for reproducibility
    )
    
    # Run the algorithm to get positions
    print("Running MPS algorithm for 20 nodes with 8 anchors...")
    mps = MPSAlgorithm(config)
    mps.generate_network()
    results = mps.run()
    
    print(f"Algorithm converged: {results['converged']}")
    print(f"Final RMSE: {results['final_rmse']:.4f}")
    
    # Extract positions
    true_positions = mps.true_positions
    estimated_positions = results['final_positions']
    anchor_positions = mps.anchor_positions
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle('Estimated vs Actual Positions - 20 Nodes with 8 Anchors\n'
                 f'RMSE: {results["final_rmse"]:.4f} | '
                 f'Converged in {results["iterations"]} iterations',
                 fontsize=16, fontweight='bold')
    
    # Subplot 1: True Positions with Network Topology
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot communication links
    for i in range(config.n_sensors):
        for j in range(i+1, config.n_sensors):
            if mps.adjacency[i, j] > 0:
                x = [true_positions[i][0], true_positions[j][0]]
                y = [true_positions[i][1], true_positions[j][1]]
                ax1.plot(x, y, 'gray', alpha=0.2, linewidth=0.5)
    
    # Plot true sensor positions
    for i in range(config.n_sensors):
        ax1.scatter(true_positions[i][0], true_positions[i][1],
                   c='blue', s=100, alpha=0.8, zorder=5)
        ax1.text(true_positions[i][0], true_positions[i][1], str(i),
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Plot anchors
    ax1.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
               c='red', s=200, marker='^', alpha=0.9, zorder=6,
               label='Anchors', edgecolors='darkred', linewidth=2)
    
    for k in range(config.n_anchors):
        ax1.text(anchor_positions[k, 0], anchor_positions[k, 1], f'A{k}',
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('True Positions & Network', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    
    # Subplot 2: Estimated Positions
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot estimated sensor positions
    for i in range(config.n_sensors):
        ax2.scatter(estimated_positions[i][0], estimated_positions[i][1],
                   c='green', s=100, alpha=0.8, zorder=5)
        ax2.text(estimated_positions[i][0], estimated_positions[i][1], str(i),
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Plot anchors (same positions)
    ax2.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
               c='red', s=200, marker='^', alpha=0.9, zorder=6,
               label='Anchors', edgecolors='darkred', linewidth=2)
    
    for k in range(config.n_anchors):
        ax2.text(anchor_positions[k, 0], anchor_positions[k, 1], f'A{k}',
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Estimated Positions', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    
    # Subplot 3: Overlay - True vs Estimated
    ax3 = plt.subplot(2, 3, 3)
    
    # Plot error vectors first (so they're behind)
    for i in range(config.n_sensors):
        x = [true_positions[i][0], estimated_positions[i][0]]
        y = [true_positions[i][1], estimated_positions[i][1]]
        error = np.linalg.norm(true_positions[i] - estimated_positions[i])
        
        # Color code error lines
        if error < 0.1:
            color = 'green'
            alpha = 0.3
        elif error < 0.2:
            color = 'orange'
            alpha = 0.4
        else:
            color = 'red'
            alpha = 0.5
        
        ax3.plot(x, y, color=color, alpha=alpha, linewidth=2)
    
    # Plot true positions
    for i in range(config.n_sensors):
        ax3.scatter(true_positions[i][0], true_positions[i][1],
                   c='blue', s=80, alpha=0.6, marker='o', label='True' if i==0 else '')
    
    # Plot estimated positions
    for i in range(config.n_sensors):
        ax3.scatter(estimated_positions[i][0], estimated_positions[i][1],
                   c='green', s=100, alpha=0.8, marker='x',
                   label='Estimated' if i==0 else '')
        
    # Plot anchors
    ax3.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
               c='red', s=200, marker='^', alpha=0.9,
               label='Anchors', edgecolors='darkred', linewidth=2)
    
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_title('True (○) vs Estimated (×) Positions', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim([-0.1, 1.1])
    ax3.set_ylim([-0.1, 1.1])
    
    # Subplot 4: Error Distribution (Bar Chart)
    ax4 = plt.subplot(2, 3, 4)
    
    errors = []
    for i in range(config.n_sensors):
        error = np.linalg.norm(true_positions[i] - estimated_positions[i])
        errors.append(error)
    
    # Sort sensors by error for better visualization
    sorted_indices = np.argsort(errors)
    sorted_errors = [errors[i] for i in sorted_indices]
    
    colors = []
    for e in sorted_errors:
        if e < 0.1:
            colors.append('green')
        elif e < 0.2:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = ax4.bar(range(config.n_sensors), sorted_errors, color=colors, alpha=0.7)
    
    # Add sensor IDs on x-axis
    ax4.set_xticks(range(config.n_sensors))
    ax4.set_xticklabels([str(sorted_indices[i]) for i in range(config.n_sensors)], fontsize=8)
    
    # Add threshold lines
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='OK (<0.2)')
    
    # Add average line
    avg_error = np.mean(errors)
    ax4.axhline(y=avg_error, color='blue', linestyle='-', linewidth=2,
               label=f'Avg: {avg_error:.3f}')
    
    ax4.set_xlabel('Sensor ID (sorted by error)', fontsize=12)
    ax4.set_ylabel('Position Error', fontsize=12)
    ax4.set_title('Error Distribution by Sensor', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: 2D Error Heatmap
    ax5 = plt.subplot(2, 3, 5)
    
    # Create a grid for interpolation
    grid_x = np.linspace(0, 1, 50)
    grid_y = np.linspace(0, 1, 50)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate errors onto grid (simple nearest neighbor)
    grid_errors = np.zeros_like(grid_X)
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            point = np.array([grid_X[j, i], grid_Y[j, i]])
            # Find nearest sensor
            min_dist = float('inf')
            nearest_error = 0
            for k in range(config.n_sensors):
                dist = np.linalg.norm(point - true_positions[k])
                if dist < min_dist:
                    min_dist = dist
                    nearest_error = errors[k]
            grid_errors[j, i] = nearest_error
    
    # Plot heatmap
    im = ax5.contourf(grid_X, grid_Y, grid_errors, levels=15, cmap='RdYlGn_r', alpha=0.7)
    plt.colorbar(im, ax=ax5, label='Error')
    
    # Overlay sensor positions
    for i in range(config.n_sensors):
        ax5.scatter(true_positions[i][0], true_positions[i][1],
                   c='blue', s=50, edgecolors='white', linewidth=1)
        ax5.text(true_positions[i][0], true_positions[i][1], str(i),
                ha='center', va='center', fontsize=6, color='white')
    
    # Plot anchors
    ax5.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
               c='red', s=100, marker='^', edgecolors='white', linewidth=2)
    
    ax5.set_xlabel('X', fontsize=12)
    ax5.set_ylabel('Y', fontsize=12)
    ax5.set_title('Spatial Error Distribution', fontsize=14, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # Subplot 6: Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean(np.square(errors)))
    max_error = max(errors)
    min_error = min(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    within_01 = sum(1 for e in errors if e < 0.1)
    within_02 = sum(1 for e in errors if e < 0.2)
    
    # Find worst and best sensors
    worst_sensor = np.argmax(errors)
    best_sensor = np.argmin(errors)
    
    stats_text = f"""
    LOCALIZATION STATISTICS
    
    Overall Performance:
    • RMSE:           {rmse:.4f}
    • Mean Error:     {avg_error:.4f}
    • Median Error:   {median_error:.4f}
    • Std Deviation:  {std_error:.4f}
    
    Range:
    • Min Error:      {min_error:.4f} (Sensor {best_sensor})
    • Max Error:      {max_error:.4f} (Sensor {worst_sensor})
    
    Success Rates:
    • Within 0.1:     {within_01}/{config.n_sensors} ({100*within_01/config.n_sensors:.1f}%)
    • Within 0.2:     {within_02}/{config.n_sensors} ({100*within_02/config.n_sensors:.1f}%)
    
    Algorithm:
    • Iterations:     {results['iterations']}
    • Converged:      {results['converged']}
    • Final Obj:      {results['final_objective']:.4f}
    """
    
    ax6.text(0.05, 0.5, stats_text, ha='left', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.axis('off')
    ax6.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = Path("results/20_nodes/estimated_vs_actual_positions.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print position comparison
    print("\n" + "="*60)
    print("POSITION COMPARISON (Sample of 5 sensors):")
    print("="*60)
    print(f"{'Sensor':<8} {'True Position':<20} {'Estimated Position':<20} {'Error':<10}")
    print("-"*60)
    
    for i in range(min(5, config.n_sensors)):
        true_pos = true_positions[i]
        est_pos = estimated_positions[i]
        error = np.linalg.norm(true_pos - est_pos)
        print(f"{i:<8} ({true_pos[0]:.3f}, {true_pos[1]:.3f})"
              f"{'':5} ({est_pos[0]:.3f}, {est_pos[1]:.3f})"
              f"{'':5} {error:.4f}")
    
    print("="*60)
    
    plt.show()
    
    return fig, results


if __name__ == "__main__":
    fig, results = visualize_estimated_vs_actual()