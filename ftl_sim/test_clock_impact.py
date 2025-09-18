#!/usr/bin/env python3
"""
Test impact of clock errors on positioning
"""

import numpy as np
from ftl.solver import FactorGraph

np.random.seed(42)

# Simple 4-anchor, 1-unknown setup
area_size = 20.0
anchors = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
unknown = (area_size/2, area_size/2)

# Test different clock bias levels
for clock_bias in [0, 1e-6, 1e-5, 1e-4]:
    print(f"\n{'='*50}")
    print(f"Clock bias = {clock_bias*1e6:.0f} µs")
    print('='*50)
    
    graph = FactorGraph()
    
    # Add anchors
    for i, (x, y) in enumerate(anchors):
        graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
    
    # Add unknown with clock bias
    true_state = np.array([unknown[0], unknown[1], clock_bias, 0, 0])
    initial = np.array([unknown[0] + 2, unknown[1] + 2, 0, 0, 0])  # Initial guess
    graph.add_node(4, initial, is_anchor=False)
    
    # Add measurements
    for i in range(4):
        anchor_pos = anchors[i]
        dist = np.sqrt((anchor_pos[0] - unknown[0])**2 + 
                      (anchor_pos[1] - unknown[1])**2)
        toa = dist / 3e8 + clock_bias  # Include clock bias
        variance = (10e-2 / 3e8)**2  # 10cm range accuracy
        graph.add_toa_factor(i, 4, toa, variance)
    
    # Optimize
    result = graph.optimize(max_iterations=100, verbose=False)
    
    # Check accuracy
    est_pos = result.estimates[4][:2]
    est_bias = result.estimates[4][2]
    pos_error = np.linalg.norm(est_pos - unknown)
    bias_error = abs(est_bias - clock_bias)
    
    print(f"Position error: {pos_error:.3f} m")
    print(f"Bias error: {bias_error*1e6:.1f} µs")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
