"""
Generate figures to visualize the algorithm performance
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Import our standalone implementation
from snl_threaded_standalone import SNLProblem, ThreadedSNLFull

def generate_network_visualization():
    """Generate network topology and localization results visualization"""
    print("Generating network visualization...")
    
    # Create a small network for clear visualization
    problem = SNLProblem(
        n_sensors=15,
        n_anchors=4,
        communication_range=0.5,
        noise_factor=0.05,
        max_iter=200,
        seed=42
    )
    
    # Run the algorithm
    snl = ThreadedSNLFull(problem)
    snl.generate_network(seed=42)
    
    print("Running MPS algorithm...")
    mps_results = snl.matrix_parametrized_splitting_threaded()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Network topology
    ax1.set_title('Sensor Network Topology', fontsize=14, fontweight='bold')
    
    # Draw communication range circles for anchors
    for i in range(problem.n_anchors):
        circle = Circle(snl.anchor_positions[i], problem.communication_range, 
                       fill=False, linestyle='--', color='green', alpha=0.3)
        ax1.add_patch(circle)
    
    # Draw edges
    adjacency, _, _ = snl._build_network_data()
    for i in range(problem.n_sensors):
        for j in range(i+1, problem.n_sensors):
            if adjacency[i, j] > 0:
                ax1.plot([snl.true_positions[i, 0], snl.true_positions[j, 0]],
                        [snl.true_positions[i, 1], snl.true_positions[j, 1]],
                        'k-', alpha=0.2, linewidth=0.5)
    
    # Draw anchor connections
    for i in range(problem.n_sensors):
        for k in snl.sensors[i].sensor_data.anchor_neighbors:
            ax1.plot([snl.true_positions[i, 0], snl.anchor_positions[k, 0]],
                    [snl.true_positions[i, 1], snl.anchor_positions[k, 1]],
                    'g-', alpha=0.2, linewidth=0.5)
    
    # Plot positions
    ax1.scatter(snl.true_positions[:, 0], snl.true_positions[:, 1], 
               c='blue', s=100, label='True sensor positions', zorder=5)
    ax1.scatter(snl.anchor_positions[:, 0], snl.anchor_positions[:, 1], 
               c='green', s=200, marker='s', label='Anchors', zorder=5)
    
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # Right plot: Localization results
    ax2.set_title('Localization Results', fontsize=14, fontweight='bold')
    
    # Extract estimated positions
    estimated = np.array([mps_results[i][0] for i in range(problem.n_sensors)])
    
    # Plot true vs estimated
    ax2.scatter(snl.true_positions[:, 0], snl.true_positions[:, 1], 
               c='blue', s=100, label='True positions', alpha=0.5)
    ax2.scatter(estimated[:, 0], estimated[:, 1], 
               c='red', s=100, marker='x', label='MPS estimates', linewidth=2)
    ax2.scatter(snl.anchor_positions[:, 0], snl.anchor_positions[:, 1], 
               c='green', s=200, marker='s', label='Anchors')
    
    # Draw error lines
    for i in range(problem.n_sensors):
        ax2.plot([snl.true_positions[i, 0], estimated[i, 0]],
                [snl.true_positions[i, 1], estimated[i, 1]],
                'r-', alpha=0.5, linewidth=1)
    
    # Add error statistics
    errors = [np.linalg.norm(snl.true_positions[i] - estimated[i]) 
              for i in range(problem.n_sensors)]
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    ax2.text(0.02, 0.98, f'Avg error: {avg_error:.4f}\nMax error: {max_error:.4f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('figures/network_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/network_visualization.png")
    
    # Cleanup
    snl.shutdown()
    plt.close()


def generate_convergence_comparison():
    """Generate convergence comparison between MPS and ADMM"""
    print("\nGenerating convergence comparison...")
    
    # Run both algorithms
    problem = SNLProblem(
        n_sensors=20,
        n_anchors=5,
        communication_range=0.6,
        noise_factor=0.05,
        max_iter=300,
        seed=42
    )
    
    snl = ThreadedSNLFull(problem)
    snl.generate_network(seed=42)
    
    print("Running algorithm comparison...")
    comparison = snl.compare_algorithms_threaded()
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Objective convergence
    ax1.semilogy(comparison['mps']['objective_history'], 'b-', 
                label='MPS', linewidth=2)
    ax1.semilogy(comparison['admm']['objective_history'], 'r--', 
                label='ADMM', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Objective Function Convergence', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark early termination
    if comparison['mps']['early_termination']:
        mps_iters = comparison['mps']['iterations']
        ax1.axvline(x=mps_iters, color='blue', linestyle=':', 
                   label=f'MPS early termination (iter {mps_iters})')
    
    # Error convergence
    ax2.semilogy(comparison['mps']['error_history'], 'b-', 
                label='MPS', linewidth=2)
    ax2.semilogy(comparison['admm']['error_history'], 'r--', 
                label='ADMM', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Localization Error Convergence', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance comparison bar chart
    categories = ['Iterations', 'Final Error\n(×0.01)', 'Runtime (s)']
    mps_values = [
        comparison['mps']['iterations'],
        comparison['mps']['final_error'] * 100,  # Scale for visibility
        comparison['mps']['total_time']
    ]
    admm_values = [
        comparison['admm']['iterations'],
        comparison['admm']['final_error'] * 100,  # Scale for visibility
        comparison['admm']['total_time']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, mps_values, width, label='MPS', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, admm_values, width, label='ADMM', color='red', alpha=0.7)
    
    ax3.set_ylabel('Value')
    ax3.set_title('Algorithm Performance Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Convergence rate analysis
    ax4.text(0.5, 0.9, 'Performance Summary', 
             transform=ax4.transAxes, ha='center', fontsize=16, fontweight='bold')
    
    summary_text = f"""
MPS Algorithm:
• Iterations: {comparison['mps']['iterations']}
• Final error: {comparison['mps']['final_error']:.6f}
• Runtime: {comparison['mps']['total_time']:.2f}s
• Early termination: {'Yes' if comparison['mps']['early_termination'] else 'No'}

ADMM Algorithm:
• Iterations: {comparison['admm']['iterations']}
• Final error: {comparison['admm']['final_error']:.6f}
• Runtime: {comparison['admm']['total_time']:.2f}s

Comparison:
• Error ratio (ADMM/MPS): {comparison['error_ratio']:.2f}×
• Speed ratio (ADMM/MPS): {comparison['speedup']:.2f}×
• Iteration ratio: {comparison['iteration_ratio']:.2f}×

Conclusion: MPS converges {comparison['iteration_ratio']:.1f}× faster
with {comparison['error_ratio']:.1f}× better accuracy than ADMM.
"""
    
    ax4.text(0.5, 0.45, summary_text, transform=ax4.transAxes, 
             ha='center', va='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/convergence_comparison.png")
    
    # Cleanup
    snl.shutdown()
    plt.close()


def generate_matrix_structure_visualization():
    """Visualize the 2-Block matrix structure"""
    print("\nGenerating matrix structure visualization...")
    
    # Small example for clarity
    n = 6
    
    # Create example matrices
    np.random.seed(42)
    
    # Generate a simple doubly stochastic matrix
    A = np.random.rand(n, n)
    A = A + A.T  # Make symmetric
    # Row normalize
    A = A / A.sum(axis=1, keepdims=True)
    # Column normalize (approximate)
    A = A / A.sum(axis=0, keepdims=True)
    
    # Create Z = 2I - A
    I = np.eye(n)
    Z = 2*I - A
    
    # Compute L from Z
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = (2 - Z[i, i]) / 2
    for i in range(n):
        for j in range(i):
            L[i, j] = -Z[i, j]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot A (doubly stochastic)
    im1 = ax1.imshow(A, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Doubly Stochastic Matrix A', fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1)
    
    # Add grid
    for i in range(n+1):
        ax1.axhline(i-0.5, color='gray', linewidth=0.5)
        ax1.axvline(i-0.5, color='gray', linewidth=0.5)
    
    # Plot Z = 2I - A
    im2 = ax2.imshow(Z, cmap='RdBu_r', vmin=-1, vmax=2)
    ax2.set_title('Matrix Z = 2I - A', fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2)
    
    for i in range(n+1):
        ax2.axhline(i-0.5, color='gray', linewidth=0.5)
        ax2.axvline(i-0.5, color='gray', linewidth=0.5)
    
    # Plot L (lower triangular)
    im3 = ax3.imshow(L, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title('Lower Triangular Matrix L', fontweight='bold')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3)
    
    for i in range(n+1):
        ax3.axhline(i-0.5, color='gray', linewidth=0.5)
        ax3.axvline(i-0.5, color='gray', linewidth=0.5)
    
    # Verify Z = 2I - L - L^T
    Z_reconstructed = 2*I - L - L.T
    im4 = ax4.imshow(np.abs(Z - Z_reconstructed), cmap='Reds', vmin=0, vmax=0.1)
    ax4.set_title('Reconstruction Error |Z - (2I - L - L^T)|', fontweight='bold')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im4, ax=ax4)
    
    for i in range(n+1):
        ax4.axhline(i-0.5, color='gray', linewidth=0.5)
        ax4.axvline(i-0.5, color='gray', linewidth=0.5)
    
    # Add text
    max_error = np.max(np.abs(Z - Z_reconstructed))
    ax4.text(0.5, -0.15, f'Max reconstruction error: {max_error:.2e}',
             transform=ax4.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/matrix_structure.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/matrix_structure.png")
    plt.close()


def generate_algorithm_flow_diagram():
    """Generate a visual representation of the algorithm flow"""
    print("\nGenerating algorithm flow diagram...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define components
    components = {
        'init': (5, 9, 'Initialize:\n• Generate network\n• Compute distances'),
        'sinkhorn': (5, 7.5, 'Distributed\nSinkhorn-Knopp\n→ Doubly stochastic A'),
        'matrix': (5, 6, 'Compute matrices:\nZ = 2I - A\nL from Z = 2I - L - L^T'),
        'block1': (2, 4, 'Block 1:\nCompute prox_gi\n(sensor positions)'),
        'block2': (8, 4, 'Block 2:\nCompute prox_PSD\n(Gram matrices)'),
        'sync': (5, 2.5, 'Synchronize\n& Exchange'),
        'update': (5, 1, 'Update:\nX, Y, dual variables'),
        'check': (5, -0.5, 'Check convergence\n& early termination')
    }
    
    # Draw boxes
    for key, (x, y, text) in components.items():
        if key in ['block1', 'block2']:
            color = 'lightblue'
        elif key == 'sinkhorn':
            color = 'lightgreen'
        elif key == 'check':
            color = 'lightyellow'
        else:
            color = 'lightgray'
            
        rect = mpatches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color,
                                      edgecolor='black',
                                      linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        ('init', 'sinkhorn'),
        ('sinkhorn', 'matrix'),
        ('matrix', 'block1'),
        ('matrix', 'block2'),
        ('block1', 'sync'),
        ('block2', 'sync'),
        ('sync', 'update'),
        ('update', 'check')
    ]
    
    for start, end in arrows:
        x1, y1, _ = components[start]
        x2, y2, _ = components[end]
        
        if start == 'matrix' and end in ['block1', 'block2']:
            # Split arrow
            ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        else:
            ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Loop back arrow
    ax.annotate('', xy=(2, 4.5), xytext=(3, -0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=-0.5"))
    ax.text(0.5, 2, 'Iterate', ha='center', color='red', fontweight='bold')
    
    # Add title and labels
    ax.text(5, 10, 'Matrix-Parametrized Proximal Splitting (MPS) Algorithm',
            ha='center', fontsize=16, fontweight='bold')
    
    ax.text(2, 3.3, '2-Block\nParallel\nStructure', ha='center', 
            fontsize=9, style='italic', color='blue')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 10.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/algorithm_flow.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/algorithm_flow.png")
    plt.close()


def main():
    """Generate all figures"""
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    print("="*60)
    print("Generating Figures for Decentralized SNL")
    print("="*60)
    
    # Generate each figure
    generate_network_visualization()
    generate_convergence_comparison()
    generate_matrix_structure_visualization()
    generate_algorithm_flow_diagram()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("Check the 'figures' directory for:")
    print("  • network_visualization.png - Network topology and localization results")
    print("  • convergence_comparison.png - MPS vs ADMM performance comparison")
    print("  • matrix_structure.png - 2-Block matrix structure visualization")
    print("  • algorithm_flow.png - Algorithm flow diagram")
    print("="*60)


if __name__ == "__main__":
    main()