#!/usr/bin/env python3
"""
Generate figures from actual MPI simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import json
import pickle
import os
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def load_simulation_data():
    """Load simulation results from saved files"""
    
    # Try to load pickle file first (complete data)
    if os.path.exists('mpi_simulation_results.pkl'):
        with open('mpi_simulation_results.pkl', 'rb') as f:
            return pickle.load(f)
    
    # Fall back to JSON summary
    elif os.path.exists('mpi_simulation_summary.json'):
        print("Warning: Using summary data only. Run simulation first for complete results.")
        with open('mpi_simulation_summary.json', 'r') as f:
            return json.load(f)
    
    else:
        print("Error: No simulation results found!")
        print("Please run: mpirun -n 4 python snl_mpi_optimized.py")
        sys.exit(1)

def generate_network_visualization(data):
    """Visualize the actual sensor network from simulation"""
    
    # Extract data
    true_positions = data.get('true_positions', {})
    final_positions = data.get('final_positions', {})
    anchor_positions = np.array(data.get('anchor_positions', []))
    params = data.get('problem_params', {})
    comm_range = params.get('communication_range', 0.3)
    
    if not true_positions or not final_positions:
        print("Warning: Position data not available for network visualization")
        return
    
    # Convert positions to arrays
    n_sensors = len(true_positions)
    true_pos_array = np.array([true_positions[i] for i in range(n_sensors)])
    final_pos_array = np.array([final_positions[i] for i in range(n_sensors)])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: True network topology
    ax1.set_title('True Sensor Network', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    
    # Draw communication links based on true positions
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            dist = np.linalg.norm(true_pos_array[i] - true_pos_array[j])
            if dist <= comm_range:
                ax1.plot([true_pos_array[i, 0], true_pos_array[j, 0]], 
                        [true_pos_array[i, 1], true_pos_array[j, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
    
    # Plot true positions and anchors
    ax1.scatter(true_pos_array[:, 0], true_pos_array[:, 1], 
               c='blue', s=100, alpha=0.7, edgecolors='darkblue', 
               linewidth=2, label='True Sensor Positions')
    ax1.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
               c='red', s=200, marker='^', alpha=0.9, edgecolors='darkred', 
               linewidth=2, label='Anchors')
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Localization results
    ax2.set_title('Localization Results', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect('equal')
    
    # Plot anchors
    ax2.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
               c='red', s=200, marker='^', alpha=0.9, 
               edgecolors='darkred', linewidth=2, label='Anchors', zorder=5)
    
    # Plot true positions
    ax2.scatter(true_pos_array[:, 0], true_pos_array[:, 1], 
               c='green', s=100, alpha=0.5, 
               edgecolors='darkgreen', linewidth=2, label='True Positions', zorder=3)
    
    # Plot estimated positions
    ax2.scatter(final_pos_array[:, 0], final_pos_array[:, 1], 
               c='blue', s=100, alpha=0.7, marker='o', 
               edgecolors='darkblue', linewidth=2, label='MPI Estimates', zorder=4)
    
    # Draw error lines
    for i in range(n_sensors):
        ax2.plot([true_pos_array[i, 0], final_pos_array[i, 0]], 
                [true_pos_array[i, 1], final_pos_array[i, 1]], 
                'gray', alpha=0.5, linewidth=1)
    
    # Calculate and display RMSE
    rmse = np.sqrt(np.mean(np.sum((true_pos_array - final_pos_array)**2, axis=1)))
    
    ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}', 
             transform=ax2.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('actual_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: actual_network_visualization.png")

def generate_convergence_plots(data):
    """Generate actual convergence plots from simulation"""
    
    results = data.get('results', data)
    objectives = results.get('objectives', [])
    errors = results.get('errors', [])
    
    if not objectives or not errors:
        print("Warning: Convergence data not available")
        return
    
    # Create iteration indices (every 10 iterations)
    iterations = np.arange(0, len(objectives) * 10, 10)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Objective value convergence
    ax1.set_title('MPI Algorithm Convergence (Actual)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.semilogy(iterations, objectives, 'b-', linewidth=2, label='MPI-MPS', marker='o', markersize=4)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, iterations[-1])
    
    # Localization error
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Localization Error (RMSE)')
    ax2.semilogy(iterations, errors, 'b-', linewidth=2, label='MPI-MPS', marker='o', markersize=4)
    
    # Add final error value
    final_error = errors[-1]
    ax2.axhline(y=final_error, color='green', linestyle=':', linewidth=2, 
                label=f'Final Error: {final_error:.4f}')
    
    # Mark convergence point if available
    converged = results.get('converged', False)
    if converged:
        conv_iter = results.get('iterations', len(objectives) * 10)
        ax2.axvline(x=conv_iter, color='orange', linestyle='-.', 
                    linewidth=2, label=f'Converged at iter {conv_iter}')
    
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, iterations[-1])
    
    plt.tight_layout()
    plt.savefig('actual_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: actual_convergence.png")

def generate_performance_summary(data):
    """Generate performance summary visualization"""
    
    results = data.get('results', data)
    params = data.get('problem_params', {})
    
    # Create figure with summary statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Timing breakdown (if available)
    timing_stats = results.get('timing_stats', {})
    if timing_stats:
        categories = list(timing_stats.keys())
        times = list(timing_stats.values())
        total_time = sum(times)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        wedges, texts, autotexts = ax1.pie(times, labels=categories, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax1.set_title('Time Distribution', fontsize=12, fontweight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, 'Timing data not available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Problem statistics
    ax2.axis('off')
    stats_text = f"""Problem Configuration:
    
    Sensors: {params.get('n_sensors', 'N/A')}
    Anchors: {params.get('n_anchors', 'N/A')}
    Comm. Range: {params.get('communication_range', 'N/A')}
    Noise Factor: {params.get('noise_factor', 'N/A')}
    
    Results:
    Converged: {results.get('converged', 'N/A')}
    Iterations: {results.get('iterations', 'N/A')}
    Final Error: {results.get('final_error', results.get('errors', ['N/A'])[-1]):.4f}
    Total Time: {data.get('total_time', 'N/A'):.2f}s
    """
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax2.set_title('Simulation Summary', fontsize=12, fontweight='bold')
    
    # 3. Error reduction over iterations
    errors = results.get('errors', [])
    if errors:
        iterations = np.arange(0, len(errors) * 10, 10)
        ax3.plot(iterations, errors, 'b-', linewidth=2)
        ax3.fill_between(iterations, errors, alpha=0.3)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Localization Error')
        ax3.set_title('Error Reduction', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add percentage improvement
        if len(errors) > 1:
            improvement = (errors[0] - errors[-1]) / errors[0] * 100
            ax3.text(0.95, 0.95, f'Improvement: {improvement:.1f}%', 
                    transform=ax3.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'Error data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Objective value reduction
    objectives = results.get('objectives', [])
    if objectives:
        iterations = np.arange(0, len(objectives) * 10, 10)
        ax4.semilogy(iterations, objectives, 'r-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Objective Value (log scale)')
        ax4.set_title('Objective Minimization', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Objective data not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('actual_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: actual_performance_summary.png")

def generate_error_distribution(data):
    """Generate error distribution visualization"""
    
    true_positions = data.get('true_positions', {})
    final_positions = data.get('final_positions', {})
    
    if not true_positions or not final_positions:
        print("Warning: Position data not available for error distribution")
        return
    
    # Calculate per-sensor errors
    n_sensors = len(true_positions)
    errors = []
    for i in range(n_sensors):
        true_pos = np.array(true_positions[i])
        final_pos = np.array(final_positions[i])
        error = np.linalg.norm(true_pos - final_pos)
        errors.append(error)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of errors
    ax1.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='darkblue')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax1.axvline(np.median(errors), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(errors):.4f}')
    ax1.set_xlabel('Localization Error')
    ax1.set_ylabel('Number of Sensors')
    ax1.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot and statistics
    ax2.boxplot(errors, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Localization Error')
    ax2.set_title('Error Statistics', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""Statistics:
    Min: {np.min(errors):.4f}
    Max: {np.max(errors):.4f}
    Mean: {np.mean(errors):.4f}
    Std: {np.std(errors):.4f}
    25%: {np.percentile(errors, 25):.4f}
    50%: {np.percentile(errors, 50):.4f}
    75%: {np.percentile(errors, 75):.4f}
    """
    
    ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('actual_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: actual_error_distribution.png")

def main():
    """Generate all figures from actual simulation data"""
    
    print("Loading simulation results...")
    print("="*50)
    
    # Load simulation data
    data = load_simulation_data()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures_actual', exist_ok=True)
    os.chdir('figures_actual')
    
    print("Generating figures from actual simulation data...")
    print("="*50)
    
    # Generate all figures
    generate_network_visualization(data)
    generate_convergence_plots(data)
    generate_performance_summary(data)
    generate_error_distribution(data)
    
    print("="*50)
    print("All figures generated successfully!")
    print("Check the 'figures_actual' directory for:")
    print("  - actual_network_visualization.png")
    print("  - actual_convergence.png")
    print("  - actual_performance_summary.png")
    print("  - actual_error_distribution.png")

if __name__ == "__main__":
    main()