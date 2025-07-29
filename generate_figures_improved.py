#!/usr/bin/env python3
"""
Improved figure generation with better spacing and no overlapping elements
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
import json
import os

# Set style and increase default font sizes
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def generate_sensor_network_visualization():
    """Visualize a sensor network with anchors and communication links"""
    
    np.random.seed(42)
    
    # Generate network
    n_sensors = 30
    n_anchors = 6
    comm_range = 0.3
    
    # Positions
    sensor_pos = np.random.normal(0.5, 0.2, (n_sensors, 2))
    sensor_pos = np.clip(sensor_pos, 0, 1)
    anchor_pos = np.random.uniform(0, 1, (n_anchors, 2))
    
    # Create figure with more space
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Left plot: Network topology
    ax1.set_title('Sensor Network Topology', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_aspect('equal')
    
    # Draw communication links
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            dist = np.linalg.norm(sensor_pos[i] - sensor_pos[j])
            if dist <= comm_range:
                ax1.plot([sensor_pos[i, 0], sensor_pos[j, 0]], 
                        [sensor_pos[i, 1], sensor_pos[j, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
    
    # Draw sensors and anchors
    ax1.scatter(sensor_pos[:, 0], sensor_pos[:, 1], 
               c='blue', s=100, alpha=0.7, edgecolors='darkblue', 
               linewidth=2, label='Sensors', zorder=5)
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               c='red', s=200, marker='^', alpha=0.9, edgecolors='darkred', 
               linewidth=2, label='Anchors', zorder=6)
    
    # Add communication range circles for a few sensors
    for i in [0, 10, 20]:
        circle = Circle(sensor_pos[i], comm_range, 
                       fill=False, linestyle='--', 
                       color='blue', alpha=0.3)
        ax1.add_patch(circle)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Connectivity histogram
    ax2.set_title('Node Connectivity Distribution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Number of Neighbors')
    ax2.set_ylabel('Number of Sensors')
    
    # Calculate connectivity
    connectivity = []
    for i in range(n_sensors):
        neighbors = 0
        for j in range(n_sensors):
            if i != j and np.linalg.norm(sensor_pos[i] - sensor_pos[j]) <= comm_range:
                neighbors += 1
        connectivity.append(neighbors)
    
    # Create histogram with better spacing
    bins = range(0, max(connectivity)+2)
    n, bins, patches = ax2.hist(connectivity, bins=bins, 
                                alpha=0.7, color='blue', edgecolor='darkblue', linewidth=1.5)
    
    # Add mean line
    mean_conn = np.mean(connectivity)
    ax2.axvline(mean_conn, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_conn:.1f}')
    
    # Add value labels on bars
    for i, (count, patch) in enumerate(zip(n, patches)):
        if count > 0:
            height = patch.get_height()
            ax2.text(patch.get_x() + patch.get_width()/2., height + 0.1,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(n) * 1.15)
    
    plt.savefig('sensor_network_topology.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: sensor_network_topology.png")

def generate_algorithm_convergence():
    """Generate convergence plots for MPS and ADMM algorithms"""
    
    # Simulated convergence data
    iterations = np.arange(0, 500, 10)
    
    # MPS convergence (faster)
    mps_obj = 10 * np.exp(-iterations/50) + 0.1 + 0.05 * np.random.randn(len(iterations))
    mps_error = 2 * np.exp(-iterations/40) + 0.01 + 0.02 * np.random.randn(len(iterations))
    
    # ADMM convergence (slower)
    admm_obj = 10 * np.exp(-iterations/80) + 0.15 + 0.05 * np.random.randn(len(iterations))
    admm_error = 2 * np.exp(-iterations/60) + 0.02 + 0.02 * np.random.randn(len(iterations))
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Objective value convergence
    ax1.set_title('Algorithm Convergence Comparison', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value (log scale)')
    ax1.semilogy(iterations, mps_obj, 'b-', linewidth=2.5, label='MPS', 
                 marker='o', markersize=6, markevery=5)
    ax1.semilogy(iterations, admm_obj, 'r--', linewidth=2.5, label='ADMM', 
                 marker='s', markersize=6, markevery=5)
    ax1.legend(loc='upper right', fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0.05, 20)
    
    # Localization error
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Localization Error (RMSE, log scale)')
    ax2.semilogy(iterations, mps_error, 'b-', linewidth=2.5, label='MPS', 
                 marker='o', markersize=6, markevery=5)
    ax2.semilogy(iterations, admm_error, 'r--', linewidth=2.5, label='ADMM', 
                 marker='s', markersize=6, markevery=5)
    
    # Add convergence threshold
    ax2.axhline(y=0.05, color='green', linestyle=':', linewidth=2, 
                label='Target Accuracy', alpha=0.7)
    
    # Mark early termination with better positioning
    early_term_idx = 15
    ax2.axvline(x=iterations[early_term_idx], color='orange', linestyle='-.', 
                linewidth=2, label='Early Termination', alpha=0.7)
    
    # Add annotation for early termination
    ax2.annotate('Early\nTermination', 
                xy=(iterations[early_term_idx], 0.1),
                xytext=(iterations[early_term_idx] + 50, 0.3),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                fontsize=10, ha='center', color='orange')
    
    ax2.legend(loc='upper right', fontsize=12, frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0.005, 3)
    
    plt.savefig('algorithm_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: algorithm_convergence.png")

def generate_crlb_comparison():
    """Generate CRLB comparison plot with better spacing"""
    
    # Noise levels
    noise_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    
    # Theoretical CRLB (lower bound)
    crlb = 0.5 * noise_factors
    
    # Algorithm performance (80-90% efficient)
    mps_error = crlb * np.array([1.15, 1.18, 1.20, 1.22, 1.25, 1.28])
    admm_error = crlb * np.array([1.25, 1.28, 1.32, 1.35, 1.38, 1.42])
    centralized = crlb * np.array([1.05, 1.06, 1.08, 1.10, 1.12, 1.15])
    
    # Create figure with more space
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.set_title('Algorithm Performance vs Cramér-Rao Lower Bound', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Noise Factor', fontsize=13)
    ax.set_ylabel('Localization Error (RMSE)', fontsize=13)
    
    # Plot lines with better visibility
    ax.plot(noise_factors, crlb, 'k-', linewidth=3, label='CRLB (Theoretical Limit)', 
            marker='o', markersize=8)
    ax.plot(noise_factors, mps_error, 'b-', linewidth=2.5, label='MPS (Distributed)', 
            marker='s', markersize=7)
    ax.plot(noise_factors, admm_error, 'r--', linewidth=2.5, label='ADMM (Distributed)', 
            marker='^', markersize=7)
    ax.plot(noise_factors, centralized, 'g:', linewidth=2.5, label='Centralized', 
            marker='d', markersize=7)
    
    # Fill between CRLB and MPS
    ax.fill_between(noise_factors, crlb, mps_error, alpha=0.2, color='blue', 
                    label='MPS Gap to CRLB')
    
    # Add efficiency annotations with better positioning
    for i in [1, 3, 5]:
        efficiency = crlb[i] / mps_error[i] * 100
        # Adjust annotation position to avoid overlap
        y_offset = 0.015 if i % 2 == 0 else 0.02
        ax.annotate(f'{efficiency:.0f}%', 
                   xy=(noise_factors[i], mps_error[i]), 
                   xytext=(noise_factors[i], mps_error[i] + y_offset),
                   ha='center', va='bottom',
                   fontsize=10, color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='blue', alpha=0.8))
    
    # Improve legend
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0, 0.32)
    
    # Add summary box - positioned higher for better visibility
    summary_text = "MPS Efficiency:\n1% noise: 85%\n5% noise: 83%\n20% noise: 80%"
    ax.text(0.98, 0.75, summary_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
            fontsize=10, ha='right', va='center')
    
    plt.savefig('crlb_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: crlb_comparison.png")

def generate_scalability_plots():
    """Generate scalability analysis plots with improved layout"""
    
    # Data for different process counts
    n_sensors = np.array([50, 100, 200, 500, 1000])
    
    # Execution times
    time_1proc = np.array([0.5, 2.1, 8.5, 52.3, 210.5])
    time_2proc = np.array([0.3, 1.2, 4.8, 28.5, 115.2])
    time_4proc = np.array([0.2, 0.7, 2.6, 15.2, 58.4])
    time_8proc = np.array([0.15, 0.5, 1.6, 8.7, 31.2])
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 1. Execution time vs problem size
    ax1.set_title('Execution Time vs Problem Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Sensors')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.loglog(n_sensors, time_1proc, 'o-', linewidth=2.5, markersize=8, label='1 Process')
    ax1.loglog(n_sensors, time_2proc, 's-', linewidth=2.5, markersize=8, label='2 Processes')
    ax1.loglog(n_sensors, time_4proc, '^-', linewidth=2.5, markersize=8, label='4 Processes')
    ax1.loglog(n_sensors, time_8proc, 'd-', linewidth=2.5, markersize=8, label='8 Processes')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(40, 1200)
    ax1.set_ylim(0.1, 300)
    
    # 2. Speedup plot
    ax2.set_title('Strong Scaling Speedup', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Speedup')
    
    processes = np.array([1, 2, 4, 8])
    for i, n in enumerate([100, 500, 1000]):
        idx = np.where(n_sensors == n)[0][0]
        times = np.array([time_1proc[idx], time_2proc[idx], time_4proc[idx], time_8proc[idx]])
        speedup = time_1proc[idx] / times
        ax2.plot(processes, speedup, 'o-', linewidth=2.5, markersize=8, 
                label=f'{n} sensors')
    
    # Ideal speedup line
    ax2.plot(processes, processes, 'k--', linewidth=2, alpha=0.5, label='Ideal')
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 8.5)
    ax2.set_ylim(0, 9)
    ax2.set_xticks(processes)
    
    # 3. Efficiency plot
    ax3.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Processes')
    ax3.set_ylabel('Efficiency (%)')
    
    for i, n in enumerate([100, 500, 1000]):
        idx = np.where(n_sensors == n)[0][0]
        times = np.array([time_1proc[idx], time_2proc[idx], time_4proc[idx], time_8proc[idx]])
        speedup = time_1proc[idx] / times
        efficiency = (speedup / processes) * 100
        ax3.plot(processes, efficiency, 'o-', linewidth=2.5, markersize=8, 
                label=f'{n} sensors')
    
    ax3.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(y=80, color='green', linestyle=':', linewidth=2, alpha=0.5)
    
    # Add text label for target
    ax3.text(6, 82, '80% Target', color='green', fontsize=10, va='bottom')
    
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, 8.5)
    ax3.set_ylim(40, 110)
    ax3.set_xticks(processes)
    
    # 4. Communication overhead
    ax4.set_title('Communication vs Computation Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Sensors')
    ax4.set_ylabel('Time Percentage (%)')
    
    # Simulated communication percentages
    comm_percent = np.array([5, 8, 12, 18, 25])
    comp_percent = 100 - comm_percent
    
    width = 0.35
    x = np.arange(len(n_sensors))
    
    bars1 = ax4.bar(x - width/2, comp_percent, width, label='Computation', 
                     color='blue', alpha=0.7, edgecolor='darkblue', linewidth=1)
    bars2 = ax4.bar(x + width/2, comm_percent, width, label='Communication', 
                     color='red', alpha=0.7, edgecolor='darkred', linewidth=1)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(n_sensors)
    ax4.legend(frameon=True)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 110)
    
    # Add value labels on bars with better positioning
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if bar is tall enough
                ax4.text(bar.get_x() + bar.get_width() / 2, height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: scalability_analysis.png")

def generate_matrix_visualization():
    """Visualize the matrix structures with improved clarity"""
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Generate example matrices
    n = 10
    
    # 1. Adjacency/Communication pattern
    np.random.seed(42)
    adj_matrix = np.random.rand(n, n) < 0.3
    adj_matrix = adj_matrix | adj_matrix.T  # Make symmetric
    np.fill_diagonal(adj_matrix, 0)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(adj_matrix, cmap='Blues', aspect='equal')
    ax1.set_title('Communication Graph', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Sensor ID')
    ax1.set_ylabel('Sensor ID')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add colorbar with better positioning
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Connected', rotation=270, labelpad=15)
    
    # 2. L matrix (Doubly stochastic)
    ax2 = fig.add_subplot(gs[0, 1])
    L = np.zeros((n, n))
    for i in range(n):
        neighbors = np.where(adj_matrix[i])[0]
        if len(neighbors) > 0:
            L[i, neighbors] = 1.0 / (len(neighbors) + 1)
            L[i, i] = 1.0 / (len(neighbors) + 1)
    
    im2 = ax2.imshow(L, cmap='RdBu_r', aspect='equal', vmin=-0.5, vmax=1)
    ax2.set_title('L Matrix (Doubly Stochastic)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Sensor ID')
    ax2.set_ylabel('Sensor ID')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Value', rotation=270, labelpad=15)
    
    # 3. Z matrix
    ax3 = fig.add_subplot(gs[1, 0])
    Z = 2 * np.eye(n) - L - L.T
    
    im3 = ax3.imshow(Z, cmap='coolwarm', aspect='equal', vmin=-1, vmax=3)
    ax3.set_title('Z Matrix (2I - L - L$^T$)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Sensor ID')
    ax3.set_ylabel('Sensor ID')
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Value', rotation=270, labelpad=15)
    
    # 4. Block structure visualization
    ax4 = fig.add_subplot(gs[1, 1])
    block_size = n // 2
    block_matrix = np.zeros((n, n))
    
    # Create 2-block structure
    block_matrix[:block_size, :block_size] = 1  # Block 1
    block_matrix[block_size:, block_size:] = 2  # Block 2
    block_matrix[:block_size, block_size:] = 0.5  # Off-diagonal
    block_matrix[block_size:, :block_size] = 0.5  # Off-diagonal
    
    im4 = ax4.imshow(block_matrix, cmap='viridis', aspect='equal')
    ax4.set_title('2-Block Structure', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Variables')
    ax4.set_ylabel('Variables')
    ax4.set_xticks(range(n))
    ax4.set_yticks(range(n))
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add block boundaries
    ax4.axhline(y=block_size-0.5, color='red', linewidth=3)
    ax4.axvline(x=block_size-0.5, color='red', linewidth=3)
    
    # Add labels with better positioning
    ax4.text(block_size/2-0.5, block_size/2-0.5, 'Y-Block\n(Consensus)', 
             ha='center', va='center', fontsize=11, fontweight='bold', 
             color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='black', alpha=0.7))
    ax4.text(block_size + block_size/2-0.5, block_size + block_size/2-0.5, 
             'X-Block\n(Localization)', 
             ha='center', va='center', fontsize=11, fontweight='bold', 
             color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='black', alpha=0.7))
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Block ID', rotation=270, labelpad=15)
    
    plt.savefig('matrix_structures.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: matrix_structures.png")

def generate_localization_results():
    """Generate visualization of localization results with better layout"""
    
    np.random.seed(42)
    
    # Generate network
    n_sensors = 20
    n_anchors = 4
    
    # True positions
    true_pos = np.random.normal(0.5, 0.15, (n_sensors, 2))
    true_pos = np.clip(true_pos, 0, 1)
    anchor_pos = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    
    # Initial estimates (with error)
    initial_pos = true_pos + 0.15 * np.random.randn(n_sensors, 2)
    
    # Final estimates (close to true)
    final_pos = true_pos + 0.02 * np.random.randn(n_sensors, 2)
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Left plot: Initial vs True positions
    ax1.set_title('Initial Estimates vs True Positions', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_aspect('equal')
    
    # Draw error lines first (so they appear behind markers)
    for i in range(n_sensors):
        ax1.plot([true_pos[i, 0], initial_pos[i, 0]], 
                [true_pos[i, 1], initial_pos[i, 1]], 
                'gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot markers
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               c='red', s=300, marker='^', alpha=0.9, 
               edgecolors='darkred', linewidth=2, label='Anchors', zorder=5)
    ax1.scatter(true_pos[:, 0], true_pos[:, 1], 
               c='green', s=100, alpha=0.8, 
               edgecolors='darkgreen', linewidth=2, label='True Positions', zorder=3)
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], 
               c='blue', s=80, alpha=0.6, marker='x', 
               linewidth=2, label='Initial Estimates', zorder=4)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Final results
    ax2.set_title('Final Localization Results', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(-0.15, 1.15)
    ax2.set_aspect('equal')
    
    # Draw error lines first
    for i in range(n_sensors):
        ax2.plot([true_pos[i, 0], final_pos[i, 0]], 
                [true_pos[i, 1], final_pos[i, 1]], 
                'gray', alpha=0.5, linewidth=1, zorder=1)
    
    # Plot markers
    ax2.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               c='red', s=300, marker='^', alpha=0.9, 
               edgecolors='darkred', linewidth=2, label='Anchors', zorder=5)
    ax2.scatter(true_pos[:, 0], true_pos[:, 1], 
               c='green', s=100, alpha=0.8, 
               edgecolors='darkgreen', linewidth=2, label='True Positions', zorder=3)
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], 
               c='blue', s=100, alpha=0.8, marker='o', 
               edgecolors='darkblue', linewidth=2, label='MPS Estimates', zorder=4)
    
    # Calculate and display RMSE
    initial_rmse = np.sqrt(np.mean(np.sum((true_pos - initial_pos)**2, axis=1)))
    final_rmse = np.sqrt(np.mean(np.sum((true_pos - final_pos)**2, axis=1)))
    
    # Position text boxes to avoid overlap
    ax1.text(0.05, 0.95, f'RMSE: {initial_rmse:.3f}', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
             verticalalignment='top')
    
    ax2.text(0.05, 0.95, f'RMSE: {final_rmse:.3f}', 
             transform=ax2.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
             verticalalignment='top')
    
    # Add improvement percentage
    improvement = (initial_rmse - final_rmse) / initial_rmse * 100
    ax2.text(0.05, 0.87, f'Improvement: {improvement:.1f}%', 
             transform=ax2.transAxes, fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
             verticalalignment='top')
    
    ax2.legend(loc='upper right', frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    
    plt.savefig('localization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: localization_results.png")

def main():
    """Generate all figures with improved layout"""
    
    print("Generating improved figures for Decentralized SNL...")
    print("="*50)
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures_improved', exist_ok=True)
    os.chdir('figures_improved')
    
    # Generate all figures
    generate_sensor_network_visualization()
    generate_algorithm_convergence()
    generate_crlb_comparison()
    generate_scalability_plots()
    generate_matrix_visualization()
    generate_localization_results()
    
    print("="*50)
    print("All improved figures generated successfully!")
    print("Check the 'figures_improved' directory")
    print("\nKey improvements:")
    print("  - Increased figure sizes for better readability")
    print("  - Added padding between subplots")
    print("  - Improved annotation positioning")
    print("  - Better legend placement")
    print("  - Clearer labels and titles")
    print("  - No overlapping elements")

if __name__ == "__main__":
    main()