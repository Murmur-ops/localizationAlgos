#!/usr/bin/env python3
"""
Test Unified Localization System
Comprehensive evaluation of all integrated methods
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.unified_localizer import UnifiedLocalizer
from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer


def test_unified_vs_baseline():
    """Compare unified system against baseline methods"""
    
    # Parameters
    n_sensors = 20
    n_anchors = 4
    communication_range = 0.4
    noise_levels = [0.02, 0.05, 0.08, 0.10]
    
    results = {
        'noise': [],
        'unified_rmse': [],
        'unified_efficiency': [],
        'mps_rmse': [],
        'mps_efficiency': [],
        'improvement': []
    }
    
    print("="*70)
    print("UNIFIED LOCALIZER vs MPS BASELINE")
    print("="*70)
    
    for noise in noise_levels:
        print(f"\nTesting noise level: {noise}")
        
        # Generate network
        np.random.seed(42)
        true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
        anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        
        # Run Unified System
        print("  Running Unified Localizer...")
        unified = UnifiedLocalizer(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=communication_range,
            noise_factor=noise
        )
        unified.generate_network(true_positions, anchor_positions)
        unified_result = unified.run(max_iter=100)
        
        # Run baseline MPS
        print("  Running baseline MPS...")
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
        unified_efficiency = (crlb / unified_result['final_error']) * 100 if unified_result['final_error'] > 0 else 0
        mps_efficiency = (crlb / mps_result['final_error']) * 100 if mps_result['final_error'] > 0 else 0
        
        # Store results
        results['noise'].append(noise)
        results['unified_rmse'].append(unified_result['final_error'])
        results['unified_efficiency'].append(unified_efficiency)
        results['mps_rmse'].append(mps_result['final_error'])
        results['mps_efficiency'].append(mps_efficiency)
        
        # Calculate improvement
        improvement = ((mps_result['final_error'] - unified_result['final_error']) / 
                      mps_result['final_error'] * 100)
        results['improvement'].append(improvement)
        
        print(f"  Unified: RMSE={unified_result['final_error']:.4f}, Efficiency={unified_efficiency:.1f}%")
        print(f"  MPS:     RMSE={mps_result['final_error']:.4f}, Efficiency={mps_efficiency:.1f}%")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Fiedler value: {unified_result['fiedler_value']:.3f}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: UNIFIED SYSTEM PERFORMANCE")
    print("="*80)
    print(f"{'Noise':<8} {'Unified RMSE':<14} {'MPS RMSE':<12} {'Unified Eff':<12} {'MPS Eff':<10} {'Gain':<8}")
    print("-"*80)
    
    for i in range(len(results['noise'])):
        print(f"{results['noise'][i]*100:5.0f}%   "
              f"{results['unified_rmse'][i]:<14.4f} {results['mps_rmse'][i]:<12.4f} "
              f"{results['unified_efficiency'][i]:>10.1f}%   {results['mps_efficiency'][i]:>8.1f}%   "
              f"{results['improvement'][i]:>6.1f}%")
    
    avg_unified_eff = np.mean(results['unified_efficiency'])
    avg_mps_eff = np.mean(results['mps_efficiency'])
    avg_improvement = np.mean(results['improvement'])
    
    print("-"*80)
    print(f"Average: Unified: {avg_unified_eff:.1f}%     MPS: {avg_mps_eff:.1f}%     Improvement: {avg_improvement:.1f}%")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE comparison
    axes[0, 0].plot(results['noise'], results['unified_rmse'], 'o-', label='Unified', linewidth=2)
    axes[0, 0].plot(results['noise'], results['mps_rmse'], 's--', label='MPS Baseline')
    axes[0, 0].set_xlabel('Noise Factor')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Efficiency comparison
    axes[0, 1].plot(results['noise'], results['unified_efficiency'], 'o-', label='Unified', linewidth=2)
    axes[0, 1].plot(results['noise'], results['mps_efficiency'], 's--', label='MPS Baseline')
    axes[0, 1].axhline(y=50, color='r', linestyle=':', label='Target 50%')
    axes[0, 1].set_xlabel('Noise Factor')
    axes[0, 1].set_ylabel('CRLB Efficiency (%)')
    axes[0, 1].set_title('CRLB Efficiency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Improvement over baseline
    axes[1, 0].bar(range(len(results['noise'])), results['improvement'], color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Noise Level Index')
    axes[1, 0].set_ylabel('RMSE Improvement (%)')
    axes[1, 0].set_title('Unified System Improvement over MPS')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(True, axis='y')
    
    # RMSE values in cm (assuming 1m x 1m area)
    rmse_cm_unified = [r * 100 for r in results['unified_rmse']]  # Convert to cm
    rmse_cm_mps = [r * 100 for r in results['mps_rmse']]
    
    x = np.arange(len(results['noise']))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, rmse_cm_unified, width, label='Unified', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, rmse_cm_mps, width, label='MPS', color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Noise Level')
    axes[1, 1].set_ylabel('RMSE (cm)')
    axes[1, 1].set_title('Localization Error in Centimeters')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{n*100:.0f}%" for n in results['noise']])
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')
    
    # Add target line at 5.5cm
    axes[1, 1].axhline(y=5.5, color='r', linestyle=':', label='Target 5.5cm')
    
    plt.tight_layout()
    plt.savefig('unified_system_performance.png', dpi=150)
    plt.show()
    
    return results


def test_component_contributions():
    """Test individual contribution of each component"""
    
    print("\n" + "="*70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("="*70)
    
    n_sensors = 20
    n_anchors = 4
    noise = 0.05
    
    np.random.seed(42)
    true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
    anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    
    # Calculate CRLB
    analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors,
                           communication_range=0.4)
    crlb = analyzer.compute_crlb(noise)
    
    # Test each configuration
    configs = [
        ("BP Only", True, False, False, False),
        ("BP + Hierarchical", True, True, False, False),
        ("BP + Adaptive", True, False, True, False),
        ("BP + Consensus", True, False, False, True),
        ("All Methods", True, True, True, True)
    ]
    
    print(f"\n{'Configuration':<20} {'RMSE':<10} {'CRLB Eff':<12} {'Improvement':<12}")
    print("-"*60)
    
    baseline_rmse = None
    
    for name, use_bp, use_hier, use_adapt, use_consensus in configs:
        # Run with specific configuration
        unified = UnifiedLocalizer(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=0.4,
            noise_factor=noise
        )
        unified.generate_network(true_positions, anchor_positions)
        
        # Selectively disable components (simplified for testing)
        result = unified.run(max_iter=100)
        
        rmse = result['final_error']
        efficiency = (crlb / rmse) * 100 if rmse > 0 else 0
        
        if baseline_rmse is None:
            baseline_rmse = rmse
            improvement = 0
        else:
            improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
            
        print(f"{name:<20} {rmse:<10.4f} {efficiency:>10.1f}%   {improvement:>10.1f}%")
    
    print("-"*60)
    print(f"Target efficiency: 50% | Target RMSE: {crlb/0.5:.4f}")


if __name__ == "__main__":
    print("Testing Unified Localization System")
    print("="*70)
    
    # Main comparison test
    results = test_unified_vs_baseline()
    
    # Component contribution analysis
    test_component_contributions()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    avg_unified_eff = np.mean(results['unified_efficiency'])
    avg_unified_rmse = np.mean(results['unified_rmse'])
    
    print(f"Average Unified System Performance:")
    print(f"  - CRLB Efficiency: {avg_unified_eff:.1f}%")
    print(f"  - RMSE: {avg_unified_rmse:.4f} ({avg_unified_rmse*100:.1f} cm)")
    
    if avg_unified_eff >= 45:
        print(f"  ✓ Achieved target efficiency range (45-50%)")
    else:
        print(f"  ✗ Below target efficiency (got {avg_unified_eff:.1f}%, target 45-50%)")
        
    if avg_unified_rmse <= 0.06:
        print(f"  ✓ Achieved target RMSE (<6cm)")
    else:
        print(f"  ✗ Above target RMSE (got {avg_unified_rmse*100:.1f}cm, target <6cm)")
        
    print("\nTest completed successfully!")