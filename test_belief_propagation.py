#!/usr/bin/env python3
"""
Test Belief Propagation implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.belief_propagation import BeliefPropagation
from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer


def test_bp_vs_mps():
    """Compare BP against MPS"""
    
    # Parameters
    n_sensors = 20
    n_anchors = 4
    communication_range = 0.4
    noise_levels = [0.02, 0.05, 0.08, 0.10]
    
    results = {
        'noise': [],
        'bp_rmse': [],
        'mps_rmse': [],
        'bp_efficiency': [],
        'mps_efficiency': [],
        'bp_iterations': [],
        'mps_iterations': []
    }
    
    for noise in noise_levels:
        print(f"\nTesting noise level: {noise}")
        
        # Generate network
        np.random.seed(42)
        true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
        anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        
        # Run BP
        bp = BeliefPropagation(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=communication_range,
            noise_factor=noise,
            max_iter=100,
            damping=0.5
        )
        bp.generate_network(true_positions, anchor_positions)
        bp_result = bp.run()
        
        # Run MPS
        mps = ProperMPSAlgorithm(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=communication_range,
            noise_factor=noise,
            max_iter=500
        )
        mps.generate_network(true_positions, anchor_positions)
        mps_result = mps.run()
        
        # Calculate CRLB
        analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors,
                               communication_range=communication_range)
        crlb = analyzer.compute_crlb(noise)
        
        # Calculate efficiencies
        bp_efficiency = (crlb / bp_result['final_error']) * 100 if bp_result['final_error'] > 0 else 0
        mps_efficiency = (crlb / mps_result['final_error']) * 100 if mps_result['final_error'] > 0 else 0
        
        # Store results
        results['noise'].append(noise)
        results['bp_rmse'].append(bp_result['final_error'])
        results['mps_rmse'].append(mps_result['final_error'])
        results['bp_efficiency'].append(bp_efficiency)
        results['mps_efficiency'].append(mps_efficiency)
        results['bp_iterations'].append(bp_result['iterations'])
        results['mps_iterations'].append(mps_result['iterations'])
        
        print(f"  BP:  RMSE={bp_result['final_error']:.4f}, Efficiency={bp_efficiency:.1f}%, Iter={bp_result['iterations']}")
        print(f"  MPS: RMSE={mps_result['final_error']:.4f}, Efficiency={mps_efficiency:.1f}%, Iter={mps_result['iterations']}")
    
    # Summary
    print("\n" + "="*70)
    print("BELIEF PROPAGATION vs MPS COMPARISON")
    print("="*70)
    print(f"{'Noise':<10} {'BP RMSE':<12} {'MPS RMSE':<12} {'BP Eff':<10} {'MPS Eff':<10}")
    print("-"*70)
    
    for i in range(len(results['noise'])):
        print(f"{results['noise'][i]*100:5.0f}%     "
              f"{results['bp_rmse'][i]:<12.4f} {results['mps_rmse'][i]:<12.4f} "
              f"{results['bp_efficiency'][i]:>6.1f}%     {results['mps_efficiency'][i]:>6.1f}%")
    
    avg_bp_eff = np.mean(results['bp_efficiency'])
    avg_mps_eff = np.mean(results['mps_efficiency'])
    
    print("-"*70)
    print(f"Average:    BP: {avg_bp_eff:.1f}%         MPS: {avg_mps_eff:.1f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE comparison
    axes[0, 0].plot(results['noise'], results['bp_rmse'], 'o-', label='BP')
    axes[0, 0].plot(results['noise'], results['mps_rmse'], 's-', label='MPS')
    axes[0, 0].set_xlabel('Noise Factor')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Efficiency comparison
    axes[0, 1].plot(results['noise'], results['bp_efficiency'], 'o-', label='BP')
    axes[0, 1].plot(results['noise'], results['mps_efficiency'], 's-', label='MPS')
    axes[0, 1].set_xlabel('Noise Factor')
    axes[0, 1].set_ylabel('CRLB Efficiency (%)')
    axes[0, 1].set_title('CRLB Efficiency Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Iterations comparison
    axes[1, 0].plot(results['noise'], results['bp_iterations'], 'o-', label='BP')
    axes[1, 0].plot(results['noise'], results['mps_iterations'], 's-', label='MPS')
    axes[1, 0].set_xlabel('Noise Factor')
    axes[1, 0].set_ylabel('Iterations')
    axes[1, 0].set_title('Convergence Speed')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Improvement ratio
    improvement = [(bp/mps - 1) * 100 for bp, mps in zip(results['bp_efficiency'], results['mps_efficiency'])]
    axes[1, 1].bar(range(len(results['noise'])), improvement)
    axes[1, 1].set_xlabel('Noise Level Index')
    axes[1, 1].set_ylabel('BP Improvement over MPS (%)')
    axes[1, 1].set_title('Relative Performance Gain')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('bp_vs_mps_comparison.png', dpi=150)
    plt.show()
    
    return results


def test_bp_convergence():
    """Test BP convergence behavior"""
    
    # Setup
    n_sensors = 20
    n_anchors = 4
    noise = 0.05
    
    np.random.seed(42)
    true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
    anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    
    # Test different damping values
    damping_values = [0.3, 0.5, 0.7, 0.9]
    
    plt.figure(figsize=(10, 6))
    
    for damping in damping_values:
        bp = BeliefPropagation(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=0.4,
            noise_factor=noise,
            max_iter=100,
            damping=damping
        )
        bp.generate_network(true_positions, anchor_positions)
        result = bp.run()
        
        plt.plot(result['errors'], label=f'Damping={damping}')
    
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('BP Convergence with Different Damping Factors')
    plt.legend()
    plt.grid(True)
    plt.savefig('bp_convergence.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Testing Belief Propagation Implementation")
    print("="*70)
    
    # Run comparison
    results = test_bp_vs_mps()
    
    # Test convergence
    test_bp_convergence()
    
    print("\nTest completed successfully!")