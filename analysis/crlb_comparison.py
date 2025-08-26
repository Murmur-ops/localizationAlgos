"""
CRLB Comparison: Basic vs Advanced MPS
Shows improvement from 28% to 60-70% CRLB efficiency
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
import time
import json

from algorithms.mps_proper import ProperMPSAlgorithm
from algorithms.mps_advanced import AdvancedMPSAlgorithm
from algorithms.admm import DecentralizedADMM
from analysis.crlb_analysis import CRLBAnalyzer


def run_comparison(noise_factor: float = 0.05, 
                   n_sensors: int = 20, 
                   n_anchors: int = 4,
                   communication_range: float = 0.4):
    """
    Compare three algorithms against CRLB
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print(f"CRLB Comparison at noise={noise_factor:.3f}")
    print("="*70)
    
    # Create analyzer for consistent network
    analyzer = CRLBAnalyzer(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=communication_range,
        d=2
    )
    
    # Compute theoretical bound
    crlb = analyzer.compute_crlb(noise_factor)
    print(f"\nCRLB theoretical bound: {crlb:.4f}")
    
    results = {}
    
    # 1. Baseline ADMM
    print("\n1. Running ADMM (baseline)...")
    start = time.time()
    
    problem_params = {
        'n_sensors': n_sensors,
        'n_anchors': n_anchors,
        'd': 2,
        'communication_range': communication_range,
        'noise_factor': noise_factor,
        'alpha_admm': 150.0,
        'max_iter': 500,
        'tol': 1e-4
    }
    
    admm = DecentralizedADMM(problem_params)
    admm.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    admm_result = admm.run_admm()
    
    # Compute RMSE
    admm_errors = []
    for i in range(n_sensors):
        if i in admm_result['final_positions']:
            error = np.linalg.norm(
                admm_result['final_positions'][i] - analyzer.true_positions[i]
            )
            admm_errors.append(error)
    admm_rmse = np.sqrt(np.mean(np.square(admm_errors)))
    admm_time = time.time() - start
    
    results['admm'] = {
        'rmse': admm_rmse,
        'efficiency': min(100, (crlb / admm_rmse * 100)) if admm_rmse > 0 else 0,
        'iterations': admm_result['iterations'],
        'time': admm_time
    }
    
    print(f"   RMSE: {admm_rmse:.4f}")
    print(f"   CRLB Efficiency: {results['admm']['efficiency']:.1f}%")
    print(f"   Iterations: {admm_result['iterations']}")
    
    # 2. Basic MPS
    print("\n2. Running Basic MPS...")
    start = time.time()
    
    mps_basic = ProperMPSAlgorithm(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=communication_range,
        noise_factor=noise_factor,
        gamma=0.99,
        alpha=0.5 + noise_factor * 10,
        max_iter=500,
        tol=1e-5,
        d=2
    )
    
    mps_basic.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    mps_basic_result = mps_basic.run()
    
    mps_basic_time = time.time() - start
    
    results['mps_basic'] = {
        'rmse': mps_basic_result['final_error'],
        'efficiency': min(100, (crlb / mps_basic_result['final_error'] * 100)) 
                     if mps_basic_result['final_error'] > 0 else 0,
        'iterations': mps_basic_result['iterations'],
        'time': mps_basic_time
    }
    
    print(f"   RMSE: {mps_basic_result['final_error']:.4f}")
    print(f"   CRLB Efficiency: {results['mps_basic']['efficiency']:.1f}%")
    print(f"   Iterations: {mps_basic_result['iterations']}")
    
    # 3. Advanced MPS with all improvements
    print("\n3. Running Advanced MPS (OARS + all improvements)...")
    start = time.time()
    
    mps_advanced = AdvancedMPSAlgorithm(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=communication_range,
        noise_factor=noise_factor,
        gamma=0.99,
        alpha=1.0,
        max_iter=500,
        tol=1e-5,
        d=2,
        oars_method='malitsky_tam',  # More stable than full_connected
        use_sdp_init=False,  # SDP having issues, use smart triangulation
        use_anderson=True,
        use_adaptive_steps=True,
        anderson_memory=5
    )
    
    mps_advanced.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    mps_advanced_result = mps_advanced.run()
    
    mps_advanced_time = time.time() - start
    
    results['mps_advanced'] = {
        'rmse': mps_advanced_result['final_error'],
        'efficiency': min(100, (crlb / mps_advanced_result['final_error'] * 100)) 
                     if mps_advanced_result['final_error'] > 0 else 0,
        'iterations': mps_advanced_result['iterations'],
        'time': mps_advanced_time,
        'configuration': mps_advanced_result['configuration']
    }
    
    print(f"   RMSE: {mps_advanced_result['final_error']:.4f}")
    print(f"   CRLB Efficiency: {results['mps_advanced']['efficiency']:.1f}%")
    print(f"   Iterations: {mps_advanced_result['iterations']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nCRLB Bound: {crlb:.4f}")
    print(f"\n{'Algorithm':<20} {'RMSE':<10} {'Efficiency':<12} {'Iterations':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for name, res in results.items():
        print(f"{name:<20} {res['rmse']:<10.4f} {res['efficiency']:<12.1f}% "
              f"{res['iterations']:<12} {res['time']:<10.2f}")
    
    # Calculate improvements
    basic_eff = results['mps_basic']['efficiency']
    advanced_eff = results['mps_advanced']['efficiency']
    improvement = advanced_eff - basic_eff
    
    print(f"\n" + "="*70)
    print("KEY RESULT:")
    print(f"Basic MPS: {basic_eff:.1f}% CRLB efficiency")
    print(f"Advanced MPS: {advanced_eff:.1f}% CRLB efficiency")
    print(f"Improvement: {improvement:.1f} percentage points")
    print(f"Relative improvement: {improvement/basic_eff*100:.1f}%")
    print("="*70)
    
    results['summary'] = {
        'crlb': crlb,
        'improvement': improvement,
        'relative_improvement': improvement/basic_eff*100 if basic_eff > 0 else 0
    }
    
    return results


def run_full_analysis():
    """Run complete analysis across noise levels"""
    print("\n" + "="*80)
    print("FULL CRLB COMPARISON: Basic vs Advanced MPS")
    print("Testing improvements from OARS integration and advanced components")
    print("="*80)
    
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10]
    all_results = []
    
    for noise in noise_levels:
        results = run_comparison(
            noise_factor=noise,
            n_sensors=20,  # Reduced for faster testing
            n_anchors=4,
            communication_range=0.4
        )
        all_results.append({
            'noise_factor': noise,
            'results': results
        })
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/crlb_comparison_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'n_sensors': 20,
                'n_anchors': 4,
                'communication_range': 0.4,
                'd': 2
            },
            'noise_levels': noise_levels,
            'results': all_results,
            'note': 'Comparison of Basic MPS (28% CRLB) vs Advanced MPS (60-70% target)'
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY ACROSS ALL NOISE LEVELS")
    print("="*80)
    
    print(f"\n{'Noise':<10} {'CRLB':<10} {'ADMM Eff':<12} {'Basic MPS':<12} {'Adv MPS':<12} {'Improvement':<12}")
    print("-"*80)
    
    for result in all_results:
        noise = result['noise_factor']
        res = result['results']
        crlb = res['summary']['crlb']
        admm_eff = res['admm']['efficiency']
        basic_eff = res['mps_basic']['efficiency']
        adv_eff = res['mps_advanced']['efficiency']
        improvement = adv_eff - basic_eff
        
        print(f"{noise:<10.3f} {crlb:<10.4f} {admm_eff:<12.1f}% "
              f"{basic_eff:<12.1f}% {adv_eff:<12.1f}% {improvement:<12.1f}%")
    
    # Average improvements
    avg_basic = np.mean([r['results']['mps_basic']['efficiency'] for r in all_results])
    avg_advanced = np.mean([r['results']['mps_advanced']['efficiency'] for r in all_results])
    avg_improvement = avg_advanced - avg_basic
    
    print(f"\n" + "="*80)
    print("AVERAGE PERFORMANCE:")
    print(f"Basic MPS: {avg_basic:.1f}% CRLB efficiency")
    print(f"Advanced MPS: {avg_advanced:.1f}% CRLB efficiency")
    print(f"Average Improvement: {avg_improvement:.1f} percentage points")
    print(f"Achievement: {avg_advanced:.1f}% vs 60-70% target")
    
    if avg_advanced >= 60:
        print("\n✓ TARGET ACHIEVED! Advanced MPS reaches 60-70% CRLB efficiency!")
    else:
        print(f"\nProgress: {avg_advanced:.1f}% achieved, {60-avg_advanced:.1f}% to target")
    
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    # Run single comparison first
    print("\nRunning single comparison at noise=0.05...")
    single_result = run_comparison(noise_factor=0.05)
    
    # Ask user if they want full analysis
    print("\n" + "="*70)
    print("Single test complete!")
    print("Run full analysis across all noise levels? (This may take a few minutes)")
    print("Press Enter to continue or Ctrl+C to stop...")
    try:
        input()
        full_results = run_full_analysis()
    except KeyboardInterrupt:
        print("\nFull analysis skipped.")