#!/usr/bin/env python3
"""
Test if the fixes improve localization performance.
Compare before and after the mathematical corrections.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    create_network_data
)
import matplotlib.pyplot as plt


def test_performance():
    """Test algorithm performance with fixes."""
    
    print("Testing MPS Algorithm Performance with Fixes")
    print("=" * 60)
    
    # Create test networks of increasing size
    test_configs = [
        {'n_sensors': 5, 'n_anchors': 2, 'name': 'Small'},
        {'n_sensors': 10, 'n_anchors': 3, 'name': 'Medium'},
        {'n_sensors': 20, 'n_anchors': 4, 'name': 'Large'},
    ]
    
    results = {}
    
    for test in test_configs:
        print(f"\n{test['name']} Network ({test['n_sensors']} sensors, {test['n_anchors']} anchors)")
        print("-" * 40)
        
        # Create network
        network = create_network_data(
            n_sensors=test['n_sensors'],
            n_anchors=test['n_anchors'],
            dimension=2,
            communication_range=0.7,
            measurement_noise=0.05,  # 5% noise as in paper
            carrier_phase=False
        )
        
        # Test with paper's parameters
        config = MPSConfig(
            n_sensors=test['n_sensors'],
            n_anchors=test['n_anchors'],
            dimension=2,
            gamma=0.999,  # Paper value
            alpha=10.0,   # Paper value
            max_iterations=200,
            tolerance=1e-6,
            communication_range=0.7,
            verbose=False,
            early_stopping=True,
            early_stopping_window=50,
            admm_iterations=100,
            admm_tolerance=1e-6,
            admm_rho=1.0,
            warm_start=True,
            parallel_proximal=False,
            use_2block=True,
            adaptive_alpha=False
        )
        
        # Run algorithm
        mps = MatrixParametrizedProximalSplitting(config, network)
        
        # Track convergence
        errors = []
        objectives = []
        consensus_errors = []
        
        for k in range(100):
            stats = mps.run_iteration(k)
            errors.append(stats['position_error'])
            objectives.append(stats['objective'])
            consensus_errors.append(stats['consensus_error'])
            
            if k % 20 == 0:
                rel_error = np.linalg.norm(mps.X - network.true_positions, 'fro') / \
                           np.linalg.norm(network.true_positions, 'fro')
                print(f"  Iter {k:3d}: rel_error={rel_error:.4f}, "
                      f"obj={stats['objective']:.4f}, "
                      f"consensus={stats['consensus_error']:.4f}")
        
        # Final metrics
        final_rel_error = np.linalg.norm(mps.X - network.true_positions, 'fro') / \
                         np.linalg.norm(network.true_positions, 'fro')
        
        results[test['name']] = {
            'errors': errors,
            'objectives': objectives,
            'consensus': consensus_errors,
            'final_error': final_rel_error,
            'improvement': (errors[0] - errors[-1]) / errors[0] * 100
        }
        
        print(f"\n  Final relative error: {final_rel_error:.4f}")
        print(f"  Improvement: {results[test['name']]['improvement']:.1f}%")
        
        # Check monotonic convergence
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        print(f"  Monotonic convergence: {'✓ Yes' if is_monotonic else '✗ No'}")
    
    return results


def plot_convergence(results):
    """Plot convergence curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for name, data in results.items():
        # Position errors
        axes[0].plot(data['errors'], label=name, linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Position Error')
        axes[0].set_title('Position Error Convergence')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_yscale('log')
        
        # Objectives
        axes[1].plot(data['objectives'], label=name, linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Objective')
        axes[1].set_title('Objective Function')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Consensus errors
        axes[2].plot(data['consensus'], label=name, linewidth=2)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Consensus Error')
        axes[2].set_title('Consensus Error')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_yscale('log')
    
    plt.suptitle('MPS Algorithm Convergence with Fixes')
    plt.tight_layout()
    plt.savefig('mps_fixes_convergence.png', dpi=150, bbox_inches='tight')
    print("\nConvergence plots saved to mps_fixes_convergence.png")


def compare_with_paper_target():
    """Compare with paper's reported performance."""
    
    print("\n" + "=" * 60)
    print("Comparison with Paper Target")
    print("=" * 60)
    
    # Paper reports 0.05-0.10 relative error for 30 sensors, 6 anchors
    print("\nCreating paper's exact setup: 30 sensors, 6 anchors...")
    
    network = create_network_data(
        n_sensors=30,
        n_anchors=6,
        dimension=2,
        communication_range=0.7,
        measurement_noise=0.05,
        carrier_phase=False
    )
    
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=500,  # More iterations
        tolerance=1e-6,
        communication_range=0.7,
        verbose=False,
        early_stopping=False,  # Run full iterations
        admm_iterations=100,
        admm_tolerance=1e-6,
        admm_rho=1.0,
        warm_start=True,
        parallel_proximal=False,
        use_2block=True,
        adaptive_alpha=False
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network)
    
    # Run and track
    checkpoints = [50, 100, 200, 300, 400, 500]
    for k in range(500):
        stats = mps.run_iteration(k)
        
        if k+1 in checkpoints:
            rel_error = np.linalg.norm(mps.X - network.true_positions, 'fro') / \
                       np.linalg.norm(network.true_positions, 'fro')
            print(f"Iteration {k+1:3d}: relative error = {rel_error:.4f}")
    
    final_rel_error = np.linalg.norm(mps.X - network.true_positions, 'fro') / \
                     np.linalg.norm(network.true_positions, 'fro')
    
    print(f"\nFinal relative error: {final_rel_error:.4f}")
    print(f"Paper target: 0.05-0.10")
    
    if final_rel_error <= 0.10:
        print("✓✓✓ SUCCESS! Matches paper's performance!")
    elif final_rel_error <= 0.15:
        print("✓✓ Close to paper's performance")
    elif final_rel_error <= 0.30:
        print("✓ Significant improvement, but more tuning needed")
    else:
        print("✗ Still needs work")
    
    return final_rel_error


def main():
    """Run all tests."""
    
    print("\n" + "=" * 60)
    print("MPS Algorithm Performance Test with Mathematical Fixes")
    print("=" * 60)
    print("\nFixes applied:")
    print("  1. L matrix factor-of-2 correction")
    print("  2. Proper 2-block SK construction")
    print("  3. Vectorization with √2 scaling")
    print("  4. Complete LAD+Tikhonov ADMM")
    print("  5. Per-node PSD projection")
    print("  6. Zero-sum warm-start")
    print("  7. Second communication step")
    print("  8. Proper early stopping")
    
    # Test on multiple network sizes
    results = test_performance()
    
    # Plot convergence
    plot_convergence(results)
    
    # Compare with paper
    final_error = compare_with_paper_target()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, data in results.items():
        print(f"\n{name} Network:")
        print(f"  Final error: {data['final_error']:.4f}")
        print(f"  Improvement: {data['improvement']:.1f}%")
    
    print(f"\nPaper's network (30 sensors):")
    print(f"  Final error: {final_error:.4f}")
    print(f"  Target: 0.05-0.10")
    
    improvement_factor = 0.745 / final_error  # Previous error was ~0.745
    print(f"\nImprovement over previous implementation: {improvement_factor:.2f}x")


if __name__ == "__main__":
    main()