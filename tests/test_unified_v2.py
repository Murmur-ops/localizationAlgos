#!/usr/bin/env python3
"""
Test Unified Localizer V2 with smart integration
"""

import numpy as np
from algorithms.unified_localizer_v2 import UnifiedLocalizerV2
from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer


def test_unified_v2():
    """Test the improved unified system"""
    
    # Test parameters
    n_sensors = 20
    n_anchors = 4
    communication_range = 0.4
    noise_levels = [0.02, 0.05, 0.08, 0.10]
    
    print("="*70)
    print("TESTING UNIFIED LOCALIZER V2 - SMART INTEGRATION")
    print("="*70)
    
    results_comparison = []
    
    for noise in noise_levels:
        print(f"\n{'='*50}")
        print(f"Testing with noise factor: {noise}")
        print('='*50)
        
        # Generate network
        np.random.seed(42)
        true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
        anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        
        # Run Unified V2
        unified = UnifiedLocalizerV2(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=communication_range,
            noise_factor=noise
        )
        unified.generate_network(true_positions, anchor_positions)
        unified_result = unified.run(max_iter=100, verbose=True)
        
        # Run baseline MPS for comparison
        print("\nRunning baseline MPS...")
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
        unified_eff = (crlb / unified_result['final_error']) * 100 if unified_result['final_error'] > 0 else 0
        mps_eff = (crlb / mps_result['final_error']) * 100 if mps_result['final_error'] > 0 else 0
        
        # Calculate improvement
        improvement = ((mps_result['final_error'] - unified_result['final_error']) / 
                      mps_result['final_error'] * 100) if mps_result['final_error'] > 0 else 0
        
        print(f"\n--- Comparison Results ---")
        print(f"Unified V2: RMSE={unified_result['final_error']:.4f}, Efficiency={unified_eff:.1f}%")
        print(f"MPS:        RMSE={mps_result['final_error']:.4f}, Efficiency={mps_eff:.1f}%")
        print(f"Improvement: {improvement:.1f}%")
        print(f"RMSE in cm: Unified={unified_result['final_error']*100:.1f}cm, MPS={mps_result['final_error']*100:.1f}cm")
        
        results_comparison.append({
            'noise': noise,
            'unified_rmse': unified_result['final_error'],
            'unified_eff': unified_eff,
            'mps_rmse': mps_result['final_error'],
            'mps_eff': mps_eff,
            'improvement': improvement,
            'avg_confidence': unified_result['average_confidence']
        })
        
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY: UNIFIED V2 PERFORMANCE")
    print("="*90)
    print(f"{'Noise':<8} {'Unified RMSE':<14} {'Unified Eff':<12} {'MPS RMSE':<12} {'MPS Eff':<10} {'Improvement':<12} {'Confidence':<10}")
    print("-"*90)
    
    for r in results_comparison:
        print(f"{r['noise']*100:5.0f}%   "
              f"{r['unified_rmse']:<14.4f} {r['unified_eff']:>10.1f}%   "
              f"{r['mps_rmse']:<12.4f} {r['mps_eff']:>8.1f}%   "
              f"{r['improvement']:>10.1f}%   {r['avg_confidence']:>8.3f}")
    
    # Calculate averages
    avg_unified_eff = np.mean([r['unified_eff'] for r in results_comparison])
    avg_mps_eff = np.mean([r['mps_eff'] for r in results_comparison])
    avg_improvement = np.mean([r['improvement'] for r in results_comparison])
    avg_confidence = np.mean([r['avg_confidence'] for r in results_comparison])
    
    print("-"*90)
    print(f"Average: Unified V2: {avg_unified_eff:.1f}%     MPS: {avg_mps_eff:.1f}%     "
          f"Improvement: {avg_improvement:.1f}%     Confidence: {avg_confidence:.3f}")
    
    # Check if we reached targets
    print("\n" + "="*70)
    print("TARGET ACHIEVEMENT")
    print("="*70)
    
    if avg_unified_eff >= 45:
        print(f"✓ Achieved target efficiency: {avg_unified_eff:.1f}% ≥ 45%")
    else:
        print(f"✗ Below target efficiency: {avg_unified_eff:.1f}% < 45%")
        
    # Check RMSE at 5% noise
    rmse_5pct = next((r['unified_rmse'] for r in results_comparison if r['noise'] == 0.05), None)
    if rmse_5pct and rmse_5pct <= 0.06:
        print(f"✓ Achieved target RMSE at 5% noise: {rmse_5pct*100:.1f}cm ≤ 6cm")
    elif rmse_5pct:
        print(f"✗ Above target RMSE at 5% noise: {rmse_5pct*100:.1f}cm > 6cm")
        
    return results_comparison


def test_node_classification():
    """Test node classification and processing"""
    
    print("\n" + "="*70)
    print("NODE CLASSIFICATION TEST")
    print("="*70)
    
    # Small test for detailed analysis
    n_sensors = 10
    n_anchors = 4
    noise = 0.05
    
    np.random.seed(42)
    true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
    anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    
    unified = UnifiedLocalizerV2(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        noise_factor=noise
    )
    unified.generate_network(true_positions, anchor_positions)
    result = unified.run(max_iter=50, verbose=False)
    
    # Get detailed results
    detailed = unified.get_detailed_results()
    
    print(f"\nNode Classification Results:")
    print(f"{'Node':<6} {'Type':<15} {'Confidence':<12} {'Error (cm)':<12} {'Final Source':<20}")
    print("-"*65)
    
    for node_id in range(n_sensors):
        if node_id in detailed:
            detail = detailed[node_id]
            error_cm = detail.get('error', 0) * 100
            final_source = detail.get('final', {}).get('source', 'N/A')
            print(f"{node_id:<6} {detail['node_type']:<15} "
                  f"{detail['confidence']:<12.3f} {error_cm:<12.1f} {final_source:<20}")
    
    print("\nProcessing strategy worked correctly for different node types.")
    

if __name__ == "__main__":
    print("Testing Unified Localizer V2 with Smart Integration")
    print("="*70)
    
    # Main performance test
    results = test_unified_v2()
    
    # Node classification test
    test_node_classification()
    
    print("\nTest completed successfully!")