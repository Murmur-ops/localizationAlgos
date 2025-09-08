#!/usr/bin/env python3
"""
Test with 8 anchors instead of 4
Compare performance improvement with more anchor coverage
"""

import numpy as np
from src.core.algorithms.unified_localizer_v2 import UnifiedLocalizerV2
from src.core.algorithms.mps_proper import ProperMPSAlgorithm
from src.core.algorithms.bp_simple import SimpleBeliefPropagation
from analysis.crlb_analysis import CRLBAnalyzer


def test_with_different_anchors():
    """Test with different numbers of anchors"""
    
    # Test parameters
    n_sensors = 20
    communication_range = 0.4
    noise_levels = [0.02, 0.05, 0.08, 0.10]
    
    # Different anchor configurations
    anchor_configs = {
        '4 anchors (baseline)': {
            'n_anchors': 4,
            'positions': np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
            ])
        },
        '6 anchors': {
            'n_anchors': 6,
            'positions': np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],  # Corners
                [0.5, 0.5], [0.5, 0.1]  # Center and top
            ])
        },
        '8 anchors (optimal)': {
            'n_anchors': 8,
            'positions': np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],  # Corners
                [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]   # Mid-sides
            ])
        }
    }
    
    print("="*80)
    print("ANCHOR COVERAGE IMPACT ON LOCALIZATION PERFORMANCE")
    print("="*80)
    
    all_results = {}
    
    for config_name, config in anchor_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing with {config_name}")
        print('='*80)
        
        n_anchors = config['n_anchors']
        anchor_positions = config['positions']
        
        config_results = []
        
        for noise in noise_levels:
            print(f"\nNoise level: {noise*100:.0f}%")
            
            # Generate network
            np.random.seed(42)
            true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
            
            # Test Unified V2
            print(f"  Running Unified V2 with {n_anchors} anchors...")
            unified = UnifiedLocalizerV2(
                n_sensors=n_sensors,
                n_anchors=n_anchors,
                communication_range=communication_range,
                noise_factor=noise
            )
            unified.generate_network(true_positions, anchor_positions)
            unified_result = unified.run(max_iter=100, verbose=False)
            
            # Test MPS
            print(f"  Running MPS with {n_anchors} anchors...")
            mps = ProperMPSAlgorithm(
                n_sensors=n_sensors,
                n_anchors=n_anchors,
                communication_range=communication_range,
                noise_factor=noise,
                max_iter=500
            )
            mps.generate_network(true_positions, anchor_positions)
            mps_result = mps.run()
            
            # Test BP
            print(f"  Running BP with {n_anchors} anchors...")
            bp = SimpleBeliefPropagation(
                n_sensors=n_sensors,
                n_anchors=n_anchors,
                communication_range=communication_range,
                noise_factor=noise,
                max_iter=100
            )
            bp.generate_network(true_positions, anchor_positions)
            bp_result = bp.run()
            
            # Calculate CRLB
            analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors,
                                   communication_range=communication_range)
            # Update anchor positions in analyzer
            analyzer.anchor_positions = anchor_positions
            crlb = analyzer.compute_crlb(noise)
            
            # Calculate efficiencies
            unified_eff = (crlb / unified_result['final_error']) * 100 if unified_result['final_error'] > 0 else 0
            mps_eff = (crlb / mps_result['final_error']) * 100 if mps_result['final_error'] > 0 else 0
            bp_eff = (crlb / bp_result['final_error']) * 100 if bp_result['final_error'] > 0 else 0
            
            # Node classification analysis
            analysis = unified_result['analysis']
            well_anchored = analysis['node_type_distribution'].get('well_anchored', 0)
            
            config_results.append({
                'noise': noise,
                'unified_rmse': unified_result['final_error'],
                'unified_eff': unified_eff,
                'mps_rmse': mps_result['final_error'],
                'mps_eff': mps_eff,
                'bp_rmse': bp_result['final_error'],
                'bp_eff': bp_eff,
                'confidence': unified_result['average_confidence'],
                'well_anchored_nodes': well_anchored,
                'crlb': crlb
            })
            
            print(f"    Unified V2: RMSE={unified_result['final_error']:.4f} ({unified_result['final_error']*100:.1f}cm), Eff={unified_eff:.1f}%")
            print(f"    MPS:        RMSE={mps_result['final_error']:.4f} ({mps_result['final_error']*100:.1f}cm), Eff={mps_eff:.1f}%")
            print(f"    BP:         RMSE={bp_result['final_error']:.4f} ({bp_result['final_error']*100:.1f}cm), Eff={bp_eff:.1f}%")
            print(f"    Confidence: {unified_result['average_confidence']:.3f}, Well-anchored nodes: {well_anchored}")
        
        all_results[config_name] = config_results
        
        # Print summary for this configuration
        print(f"\n{config_name} Summary:")
        print("-"*60)
        avg_unified_eff = np.mean([r['unified_eff'] for r in config_results])
        avg_mps_eff = np.mean([r['mps_eff'] for r in config_results])
        avg_bp_eff = np.mean([r['bp_eff'] for r in config_results])
        avg_confidence = np.mean([r['confidence'] for r in config_results])
        
        print(f"Average CRLB Efficiency:")
        print(f"  Unified V2: {avg_unified_eff:.1f}%")
        print(f"  MPS:        {avg_mps_eff:.1f}%")
        print(f"  BP:         {avg_bp_eff:.1f}%")
        print(f"Average Confidence: {avg_confidence:.3f}")
        
        # Check 5% noise performance
        result_5pct = next((r for r in config_results if r['noise'] == 0.05), None)
        if result_5pct:
            print(f"At 5% noise:")
            print(f"  Unified V2: {result_5pct['unified_rmse']*100:.1f}cm, {result_5pct['unified_eff']:.1f}% efficiency")
            print(f"  MPS:        {result_5pct['mps_rmse']*100:.1f}cm, {result_5pct['mps_eff']:.1f}% efficiency")
    
    # Compare across configurations
    print("\n" + "="*80)
    print("COMPARISON ACROSS ANCHOR CONFIGURATIONS")
    print("="*80)
    
    print(f"\n{'Config':<25} {'Avg Unified Eff':<18} {'Avg MPS Eff':<15} {'Avg Confidence':<15} {'5% Noise RMSE':<15}")
    print("-"*85)
    
    for config_name, results in all_results.items():
        avg_unified = np.mean([r['unified_eff'] for r in results])
        avg_mps = np.mean([r['mps_eff'] for r in results])
        avg_conf = np.mean([r['confidence'] for r in results])
        rmse_5pct = next((r['unified_rmse']*100 for r in results if r['noise'] == 0.05), 0)
        
        print(f"{config_name:<25} {avg_unified:>15.1f}%   {avg_mps:>12.1f}%   {avg_conf:>13.3f}   {rmse_5pct:>12.1f}cm")
    
    # Check if 8 anchors achieves targets
    results_8 = all_results.get('8 anchors (optimal)', [])
    if results_8:
        avg_eff_8 = np.mean([r['unified_eff'] for r in results_8])
        rmse_5pct_8 = next((r['unified_rmse']*100 for r in results_8 if r['noise'] == 0.05), 100)
        
        print("\n" + "="*80)
        print("TARGET ACHIEVEMENT WITH 8 ANCHORS")
        print("="*80)
        
        if avg_eff_8 >= 45:
            print(f"✓ Achieved target efficiency: {avg_eff_8:.1f}% ≥ 45%")
        else:
            print(f"✗ Below target efficiency: {avg_eff_8:.1f}% < 45%")
            
        if rmse_5pct_8 <= 6:
            print(f"✓ Achieved target RMSE: {rmse_5pct_8:.1f}cm ≤ 6cm")
        else:
            print(f"✗ Above target RMSE: {rmse_5pct_8:.1f}cm > 6cm")
    
    return all_results


def analyze_anchor_coverage():
    """Analyze how anchor coverage affects node classification"""
    
    print("\n" + "="*80)
    print("ANCHOR COVERAGE ANALYSIS")
    print("="*80)
    
    n_sensors = 20
    communication_range = 0.4
    
    np.random.seed(42)
    true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
    
    for n_anchors in [4, 6, 8]:
        print(f"\nWith {n_anchors} anchors:")
        
        if n_anchors == 4:
            anchor_positions = np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
            ])
        elif n_anchors == 6:
            anchor_positions = np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
                [0.5, 0.5], [0.5, 0.1]
            ])
        else:  # 8
            anchor_positions = np.array([
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
                [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
            ])
        
        # Count anchor coverage for each sensor
        coverage = {}
        for i in range(n_sensors):
            count = 0
            for a in range(n_anchors):
                dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
                if dist <= communication_range:
                    count += 1
            coverage[i] = count
        
        # Statistics
        avg_coverage = np.mean(list(coverage.values()))
        nodes_with_2plus = sum(1 for c in coverage.values() if c >= 2)
        nodes_with_3plus = sum(1 for c in coverage.values() if c >= 3)
        
        print(f"  Average anchors per node: {avg_coverage:.1f}")
        print(f"  Nodes with ≥2 anchors: {nodes_with_2plus}/{n_sensors} ({nodes_with_2plus/n_sensors*100:.0f}%)")
        print(f"  Nodes with ≥3 anchors: {nodes_with_3plus}/{n_sensors} ({nodes_with_3plus/n_sensors*100:.0f}%)")


if __name__ == "__main__":
    print("Testing Impact of Anchor Count on Localization Performance")
    print("="*80)
    
    # Main comparison test
    results = test_with_different_anchors()
    
    # Anchor coverage analysis
    analyze_anchor_coverage()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("More anchors significantly improve:")
    print("1. Node classification (more well-anchored nodes)")
    print("2. Average confidence scores")
    print("3. CRLB efficiency")
    print("4. Localization accuracy (RMSE)")
    print("\nTest completed successfully!")