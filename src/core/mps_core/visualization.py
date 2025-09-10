"""
Visualization module for MPS algorithm results
Generates separate figure windows for network, convergence, and position comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def generate_figures(results: Dict[str, Any], 
                    network_data: Any,
                    config: Dict[str, Any],
                    save_path: Optional[str] = None) -> None:
    """
    Generate three separate figure windows:
    1. Network topology with communication links
    2. True vs estimated positions
    3. Convergence curves (objective and error)
    
    Args:
        results: Algorithm results dictionary
        network_data: Network data with true positions and adjacency
        config: Configuration dictionary
        save_path: Optional path to save figures
    """
    
    # Only 2D visualization supported
    if config['network']['dimension'] != 2:
        logger.warning("Visualization only supported for 2D networks")
        return
    
    try:
        # Set interactive mode for separate windows
        plt.ion()
        
        # Generate each figure
        fig1 = plot_network_topology(network_data, config)
        fig2 = plot_position_comparison(results, network_data, config)
        fig3 = plot_convergence_curves(results, config)
        
        # Save if requested
        if save_path:
            fig1.savefig(f"{save_path}_network.png", dpi=150, bbox_inches='tight')
            fig2.savefig(f"{save_path}_positions.png", dpi=150, bbox_inches='tight')
            fig3.savefig(f"{save_path}_convergence.png", dpi=150, bbox_inches='tight')
            logger.info(f"Figures saved to {save_path}_*.png")
        
        # Show all figures
        plt.show(block=True)
        
    except Exception as e:
        logger.error(f"Error generating figures: {e}")


def plot_network_topology(network_data: Any, config: Dict[str, Any]) -> plt.Figure:
    """
    Plot network topology showing sensors, anchors, and communication links
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    n_sensors = config['network']['n_sensors']
    n_anchors = config['network']['n_anchors']
    scale = config['network'].get('scale', 1.0)
    # Scale positions for display (internal are in [0,1])
    positions = network_data.true_positions * scale
    
    # Plot communication links
    adjacency = network_data.adjacency_matrix
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            if adjacency[i, j] > 0:
                ax.plot([positions[i, 0], positions[j, 0]], 
                       [positions[i, 1], positions[j, 1]], 
                       'gray', alpha=0.3, linewidth=0.5)
    
    # Plot anchor-sensor connections
    if hasattr(network_data, 'anchor_positions'):
        anchor_pos = network_data.anchor_positions * scale
        for i in range(n_sensors):
            for a in range(n_anchors):
                key = (i, n_sensors + a)
                if key in network_data.distance_measurements:
                    ax.plot([positions[i, 0], anchor_pos[a, 0]], 
                           [positions[i, 1], anchor_pos[a, 1]], 
                           'green', alpha=0.2, linewidth=0.5, linestyle='--')
    
    # Plot sensors
    ax.scatter(positions[:, 0], positions[:, 1], 
              c='blue', s=100, alpha=0.8, edgecolors='navy',
              linewidth=2, label=f'Sensors (n={n_sensors})', zorder=5)
    
    # Plot anchors
    if hasattr(network_data, 'anchor_positions'):
        ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
                  c='red', s=200, alpha=0.9, marker='^', edgecolors='maroon',
                  linewidth=2, label=f'Anchors (n={n_anchors})', zorder=6)
    
    # Add sensor labels
    for i in range(n_sensors):
        ax.annotate(str(i), (positions[i, 0], positions[i, 1]), 
                   fontsize=8, ha='center', va='center', color='white', weight='bold')
    
    # Add anchor labels
    if hasattr(network_data, 'anchor_positions'):
        for a in range(n_anchors):
            ax.annotate(f'A{a}', (anchor_pos[a, 0], anchor_pos[a, 1]), 
                       fontsize=10, ha='center', va='center', color='white', weight='bold')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Network Topology and Communication Links', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits with padding
    scale = config['network'].get('scale', 1.0)
    ax.set_xlim(-0.1*scale, 1.1*scale)
    ax.set_ylim(-0.1*scale, 1.1*scale)
    
    fig.suptitle(f"Network: {n_sensors} sensors, {n_anchors} anchors, "
                f"Range: {config['network']['communication_range']}", 
                fontsize=12)
    
    return fig


def plot_position_comparison(results: Dict[str, Any], 
                            network_data: Any, 
                            config: Dict[str, Any]) -> plt.Figure:
    """
    Plot true vs estimated positions with error vectors
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    
    n_sensors = config['network']['n_sensors']
    scale = config['network'].get('scale', 1.0)
    true_pos = network_data.true_positions * scale
    
    if 'best_positions' not in results or results['best_positions'] is None:
        # If no results yet, just show true positions
        ax = fig.add_subplot(111)
        ax.scatter(true_pos[:, 0], true_pos[:, 1], 
                  c='blue', s=100, alpha=0.8, label='True positions')
        ax.set_title('True Positions (No estimates available yet)', fontsize=14)
    else:
        est_pos = results['best_positions'][:n_sensors] * scale  # Scale and get only sensor positions
        
        # Create two subplots
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Left plot: Overlay of true and estimated
        ax1.scatter(true_pos[:, 0], true_pos[:, 1], 
                   c='blue', s=100, alpha=0.6, label='True', 
                   edgecolors='navy', linewidth=2)
        ax1.scatter(est_pos[:, 0], est_pos[:, 1], 
                   c='red', s=80, alpha=0.6, label='Estimated',
                   marker='x', linewidth=2)
        
        # Draw error vectors
        for i in range(n_sensors):
            ax1.arrow(true_pos[i, 0], true_pos[i, 1],
                     est_pos[i, 0] - true_pos[i, 0],
                     est_pos[i, 1] - true_pos[i, 1],
                     color='gray', alpha=0.5, width=0.001,
                     head_width=0.01, head_length=0.01)
        
        ax1.set_xlabel('X Position (m)', fontsize=12)
        ax1.set_ylabel('Y Position (m)', fontsize=12)
        ax1.set_title('True vs Estimated Positions', fontsize=14, weight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right plot: Error distribution
        errors = np.linalg.norm(est_pos - true_pos, axis=1)
        
        ax2.hist(errors, bins=20, alpha=0.7, color='purple', edgecolor='purple')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax2.axvline(np.median(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(errors):.4f}')
        
        ax2.set_xlabel('Position Error (m)', fontsize=12)
        ax2.set_ylabel('Number of Sensors', fontsize=12)
        ax2.set_title('Error Distribution', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        # Get scale and communication range from config
        scale = config['network'].get('scale', 1.0)
        comm_range = config['network'].get('communication_range', 0.3)
        physical_comm_radius = comm_range * scale
        
        # Normalized RMSE (as in papers)
        rmse_normalized = results.get('best_error', 0)
        
        # Physical RMSE in meters
        rmse_meters = rmse_normalized * physical_comm_radius
        
        # Per-sensor errors in meters
        errors_meters = errors * physical_comm_radius
        
        stats_text = (f"Normalized RMSE: {rmse_normalized:.4f} ({rmse_normalized*100:.1f}%)\n"
                     f"Physical RMSE: {rmse_meters:.4f} m\n"
                     f"Max Error: {np.max(errors_meters):.4f} m\n"
                     f"Min Error: {np.min(errors_meters):.4f} m\n"
                     f"Comm. Radius: {physical_comm_radius:.1f} m")
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    fig.suptitle(f"Position Comparison - Iteration {results.get('iterations', 0)}", 
                fontsize=14)
    
    return fig


def plot_convergence_curves(results: Dict[str, Any], 
                           config: Dict[str, Any]) -> plt.Figure:
    """
    Plot convergence curves for objective function and RMSE
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Get data
    objectives = results.get('objectives', [])
    errors = results.get('errors', [])
    
    if not objectives and not errors:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No convergence data available yet', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Create iteration array
    log_interval = config['performance'].get('log_interval', 10)
    iterations = range(0, len(objectives) * log_interval, log_interval)
    
    # Left plot: Objective function
    ax1 = fig.add_subplot(121)
    if objectives:
        ax1.semilogy(iterations, objectives, 'b-', linewidth=2, label='Objective')
        ax1.fill_between(iterations, objectives, alpha=0.3)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Objective Value (log scale)', fontsize=12)
        ax1.set_title('Objective Function Convergence', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend()
        
        # Mark best iteration
        best_idx = np.argmin(objectives)
        ax1.scatter(iterations[best_idx], objectives[best_idx], 
                   color='red', s=100, zorder=5, 
                   label=f'Best: {objectives[best_idx]:.2e}')
    
    # Right plot: RMSE
    ax2 = fig.add_subplot(122)
    if errors:
        ax2.semilogy(iterations, errors, 'r-', linewidth=2, label='RMSE')
        ax2.fill_between(iterations, errors, alpha=0.3, color='red')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('RMSE (log scale)', fontsize=12)
        ax2.set_title('Position Error Convergence', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend()
        
        # Mark best iteration
        best_idx = np.argmin(errors)
        ax2.scatter(iterations[best_idx], errors[best_idx], 
                   color='green', s=100, zorder=5,
                   label=f'Best: {errors[best_idx]:.4f}')
        
        # Add convergence rate if enough points
        if len(errors) > 10:
            # Estimate convergence rate from last 20% of iterations
            start_idx = int(0.8 * len(errors))
            x = np.array(list(range(start_idx, len(errors))))
            y = np.log(errors[start_idx:])
            if len(x) > 1:
                rate = np.polyfit(x, y, 1)[0]
                ax2.text(0.98, 0.02, f'Conv. rate: {-rate:.4f}', 
                        transform=ax2.transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                        fontsize=10)
    
    # Overall title
    fig.suptitle(f"Convergence Analysis - {config['algorithm']['name']} Algorithm\n"
                f"γ={config['algorithm']['gamma']}, α={config['algorithm']['alpha']}", 
                fontsize=14)
    
    return fig


def plot_results_separate_windows(results: Dict[str, Any], 
                                 config: Dict[str, Any], 
                                 network_data: Any) -> None:
    """
    Generate plots in separate windows (wrapper for backwards compatibility)
    
    Args:
        results: Algorithm results
        config: Configuration dictionary
        network_data: Network data object
    """
    generate_figures(results, network_data, config)