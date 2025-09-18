#!/usr/bin/env python3
"""
Debug why ideal simulation doesn't achieve CRLB
"""

import numpy as np
from ftl.solver import FactorGraph

def test_single_node_ideal():
    """Test simplest possible case"""
    
    print("Testing single node positioning (simplest case):")
    print("="*60)
    
    np.random.seed(42)
    
    area_size = 50.0
    anchors = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
    unknown_true = (area_size/2, area_size/2)
    
    # Very small variance to avoid numerical issues
    range_std = 0.1  # 10cm
    toa_var = (range_std / 3e8)**2
    
    print(f"Setup:")
    print(f"  Unknown at: {unknown_true}")
    print(f"  Range σ: {range_std*100:.1f} cm")
    print(f"  ToA variance: {toa_var:.2e} s²")
    
    # Check what variance will be used after floor
    from ftl.rx_frontend import cov_from_crlb
    effective_var = max(toa_var, 1e-12)
    print(f"  Effective variance (after floor): {effective_var:.2e} s²")
    print(f"  Weight = 1/var: {1/effective_var:.2e}")
    
    # Build graph
    graph = FactorGraph()
    
    # Add anchors
    for i, (x, y) in enumerate(anchors):
        graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
    
    # Add unknown - start close to true position
    initial = np.array([unknown_true[0] + 0.5, unknown_true[1] + 0.5, 0, 0, 0])
    graph.add_node(4, initial, is_anchor=False)
    print(f"\nInitial guess offset: {np.linalg.norm(initial[:2] - unknown_true):.3f} m")
    
    # Add perfect measurements (no noise for debugging)
    print("\nMeasurements:")
    for i, (ax, ay) in enumerate(anchors):
        anchor = np.array([ax, ay])
        unknown = np.array(unknown_true)
        true_dist = np.linalg.norm(anchor - unknown)
        true_toa = true_dist / 3e8
        
        # Use exact measurement for debugging
        measured_toa = true_toa
        
        print(f"  Anchor {i} at ({ax},{ay}): dist={true_dist:.2f}m, toa={true_toa*1e9:.3f}ns")
        
        graph.add_toa_factor(i, 4, measured_toa, effective_var)
    
    # Optimize with verbose output
    print("\nOptimizing...")
    result = graph.optimize(max_iterations=20, verbose=True, tolerance=1e-10)
    
    # Check result
    est_pos = result.estimates[4][:2]
    error = np.linalg.norm(est_pos - unknown_true)
    
    print(f"\nResults:")
    print(f"  True position: ({unknown_true[0]:.2f}, {unknown_true[1]:.2f})")
    print(f"  Estimated: ({est_pos[0]:.2f}, {est_pos[1]:.2f})")
    print(f"  Error: {error:.6f} m ({error*100:.4f} cm)")
    print(f"  Converged: {result.converged}")
    
    # Check residuals
    print(f"\nDiagnostics:")
    print(f"  Initial cost: {result.initial_cost:.2e}")
    print(f"  Final cost: {result.final_cost:.2e}")
    
    # Check if clock parameters moved
    est_clock = result.estimates[4][2:]
    print(f"  Estimated clock params: bias={est_clock[0]:.2e}, drift={est_clock[1]:.2e}, cfo={est_clock[2]:.2e}")

test_single_node_ideal()

# Now test with actual noise
print("\n" + "="*60)
print("Testing with measurement noise:")
print("="*60)

np.random.seed(42)
n_trials = 20
errors = []

for trial in range(n_trials):
    area_size = 50.0
    anchors = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
    unknown_true = (area_size/2, area_size/2)
    
    range_std = 0.1  # 10cm
    toa_var = (range_std / 3e8)**2
    effective_var = max(toa_var, 1e-12)
    
    graph = FactorGraph()
    
    for i, (x, y) in enumerate(anchors):
        graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
    
    initial = np.array([unknown_true[0] + np.random.randn(), 
                       unknown_true[1] + np.random.randn(), 0, 0, 0])
    graph.add_node(4, initial, is_anchor=False)
    
    # Add noisy measurements
    for i, (ax, ay) in enumerate(anchors):
        anchor = np.array([ax, ay])
        unknown = np.array(unknown_true)
        true_dist = np.linalg.norm(anchor - unknown)
        true_toa = true_dist / 3e8
        
        # Add Gaussian noise
        noise = np.random.normal(0, np.sqrt(toa_var))
        measured_toa = true_toa + noise
        
        graph.add_toa_factor(i, 4, measured_toa, effective_var)
    
    result = graph.optimize(max_iterations=50, verbose=False)
    
    est_pos = result.estimates[4][:2]
    error = np.linalg.norm(est_pos - unknown_true)
    errors.append(error)
    
    if trial < 3:
        print(f"Trial {trial+1}: error = {error*100:.2f} cm")

rmse = np.sqrt(np.mean(np.array(errors)**2))
print(f"\nEmpirical RMSE over {n_trials} trials: {rmse*100:.2f} cm")
print(f"Theoretical CRLB: {range_std*100:.2f} cm")
print(f"Efficiency: {range_std/rmse*100:.1f}%")
