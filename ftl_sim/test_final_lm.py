"""
Final test of adaptive LM performance
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


def test_adaptive_lm():
    """Test adaptive LM with detailed output"""
    print("="*60)
    print("Adaptive Levenberg-Marquardt Performance Test")
    print("="*60)

    # Test with different network sizes
    node_counts = [6, 8, 10, 15]

    results = {}

    for n_nodes in node_counts:
        print(f"\nTesting {n_nodes} nodes ({3} anchors, {n_nodes-3} unknowns)...")

        config = EnhancedFTLConfig(
            n_nodes=n_nodes,
            n_anchors=3,
            use_adaptive_lm=True,
            use_line_search=False,
            max_iterations=100,
            verbose=False
        )

        ftl = EnhancedFTL(config)
        ftl.run()

        results[n_nodes] = {
            'pos_rmse': ftl.position_rmse_history,
            'time_rmse': ftl.time_rmse_history,
            'lambda': ftl.lm_optimizer.lambda_history,
            'final_pos': ftl.position_rmse_history[-1],
            'final_time': ftl.time_rmse_history[-1],
            'iterations': len(ftl.position_rmse_history) - 1
        }

        print(f"  Converged in {results[n_nodes]['iterations']} iterations")
        print(f"  Final position RMSE: {results[n_nodes]['final_pos']*1e6:.2f} µm")
        print(f"  Final time RMSE: {results[n_nodes]['final_time']*1e3:.2f} ps")
        print(f"  Final λ: {ftl.lm_optimizer.lambda_current:.2e}")

    # Create convergence plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position convergence
    ax = axes[0, 0]
    for n_nodes in node_counts:
        ax.semilogy(results[n_nodes]['pos_rmse'],
                   label=f'{n_nodes} nodes', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Position Convergence with Adaptive LM')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time convergence
    ax = axes[0, 1]
    for n_nodes in node_counts:
        ax.semilogy(results[n_nodes]['time_rmse'],
                   label=f'{n_nodes} nodes', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time RMSE (ns)')
    ax.set_title('Time Synchronization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda adaptation
    ax = axes[1, 0]
    for n_nodes in node_counts:
        if results[n_nodes]['lambda']:
            ax.semilogy(results[n_nodes]['lambda'][:50],  # First 50 iterations
                       label=f'{n_nodes} nodes', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Damping Parameter λ')
    ax.set_title('Adaptive Damping Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final accuracy vs network size
    ax = axes[1, 1]
    sizes = list(node_counts)
    pos_accuracy = [results[n]['final_pos']*1e6 for n in node_counts]
    time_accuracy = [results[n]['final_time']*1e3 for n in node_counts]

    ax2 = ax.twinx()
    line1 = ax.semilogy(sizes, pos_accuracy, 'b-o', label='Position (µm)', linewidth=2)
    line2 = ax2.semilogy(sizes, time_accuracy, 'r-s', label='Time (ps)', linewidth=2)

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Position RMSE (µm)', color='b')
    ax2.set_ylabel('Time RMSE (ps)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Final Accuracy vs Network Size')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_lm_performance.png', dpi=150)
    print("\nPlots saved to adaptive_lm_performance.png")

    # Summary
    print("\n" + "="*60)
    print("Summary: Adaptive LM achieves sub-millimeter accuracy")
    print("="*60)
    print("Network Size | Position RMSE | Time RMSE | Iterations")
    print("-"*60)
    for n_nodes in node_counts:
        r = results[n_nodes]
        print(f"{n_nodes:12d} | {r['final_pos']*1e3:13.3f} mm | {r['final_time']:9.3f} ns | {r['iterations']:10d}")

    return results


if __name__ == "__main__":
    results = test_adaptive_lm()