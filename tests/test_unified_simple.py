#!/usr/bin/env python3
"""Simple test of unified system"""

import numpy as np
from algorithms.unified_localizer import UnifiedLocalizer
from analysis.crlb_analysis import CRLBAnalyzer

# Small test
n_sensors = 10
n_anchors = 4
noise = 0.05

print("Testing Unified Localizer...")
np.random.seed(42)
true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])

try:
    unified = UnifiedLocalizer(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        noise_factor=noise
    )
    print("Generating network...")
    unified.generate_network(true_positions, anchor_positions)
    
    print("Running unified localization...")
    result = unified.run(max_iter=50)
    
    print(f"\nResults:")
    print(f"  Final RMSE: {result['final_error']:.4f}")
    print(f"  BP RMSE: {result['bp_error']:.4f}")
    print(f"  Hierarchical RMSE: {result['hierarchical_error']:.4f}")
    
    # Calculate CRLB efficiency
    analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors,
                           communication_range=0.4)
    crlb = analyzer.compute_crlb(noise)
    efficiency = (crlb / result['final_error']) * 100
    
    print(f"  CRLB Efficiency: {efficiency:.1f}%")
    
    # RMSE in cm
    rmse_cm = result['final_error'] * 100
    print(f"  RMSE in cm: {rmse_cm:.1f} cm")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()