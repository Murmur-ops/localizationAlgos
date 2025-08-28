#!/usr/bin/env python3
"""Test simplified BP"""

import numpy as np
from algorithms.bp_simple import SimpleBeliefPropagation
from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer

# Parameters
n_sensors = 20
n_anchors = 4
noise_levels = [0.02, 0.05, 0.08, 0.10]

print("SIMPLIFIED BP vs MPS")
print("="*60)
print(f"{'Noise':<8} {'BP RMSE':<10} {'MPS RMSE':<10} {'BP Eff':<10} {'MPS Eff':<10}")
print("-"*60)

for noise in noise_levels:
    np.random.seed(42)
    true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
    anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    
    # Run simplified BP
    bp = SimpleBeliefPropagation(n_sensors=n_sensors, n_anchors=n_anchors,
                                 communication_range=0.4, noise_factor=noise, max_iter=100)
    bp.generate_network(true_positions, anchor_positions)
    bp_result = bp.run()
    
    # Run MPS
    mps = ProperMPSAlgorithm(n_sensors=n_sensors, n_anchors=n_anchors,
                            communication_range=0.4, noise_factor=noise, max_iter=500)
    mps.generate_network(true_positions, anchor_positions)
    mps_result = mps.run()
    
    # Calculate CRLB
    analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors, communication_range=0.4)
    crlb = analyzer.compute_crlb(noise)
    
    bp_eff = (crlb / bp_result['final_error']) * 100 if bp_result['final_error'] > 0 else 0
    mps_eff = (crlb / mps_result['final_error']) * 100 if mps_result['final_error'] > 0 else 0
    
    print(f"{noise*100:5.0f}%   {bp_result['final_error']:.4f}     {mps_result['final_error']:.4f}     "
          f"{bp_eff:6.1f}%    {mps_eff:6.1f}%")
    
print("-"*60)
print(f"Avg:     BP: {np.mean([crlb/bp_result['final_error']*100 for bp_result in [bp_result]]):,.1f}%      "
      f"MPS: {np.mean([crlb/mps_result['final_error']*100 for mps_result in [mps_result]]):.1f}%")