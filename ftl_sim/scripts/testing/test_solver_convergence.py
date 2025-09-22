#!/usr/bin/env python3
"""
Test solver convergence properly
"""

import numpy as np
from ftl.solver import FactorGraph

def test_convergence_with_different_variances():
    """Test how variance affects convergence"""
    
    print("SOLVER CONVERGENCE ANALYSIS")
    print("="*70)
    
    np.random.seed(42)
    
    # Simple 4-anchor, 1-unknown setup
    area_size = 20.0
    anchors = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
    unknown_true = (area_size/2, area_size/2)
    
    # Test different variance levels
    test_cases = [
        (1e-18, "Realistic UWB (30cm)"),
        (1e-16, "Relaxed (3m)"),
        (1e-14, "Very relaxed (30m)"),
        (1e-12, "Our floor (300m)"),
        (1e-10, "Huge uncertainty (3km)")
    ]
    
    print(f"Testing convergence for different variance levels:")
    print("-"*70)
    
    for variance, description in test_cases:
        graph = FactorGraph()
        
        # Add anchors
        for i, (x, y) in enumerate(anchors):
            graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
        
        # Add unknown with poor initial guess
        initial = np.array([5.0, 5.0, 0, 0, 0])  # 5m offset from truth
        graph.add_node(4, initial, is_anchor=False)
        
        # Add measurements
        for i, (ax, ay) in enumerate(anchors):
            dist = np.linalg.norm(np.array([ax, ay]) - np.array(unknown_true))
            true_toa = dist / 3e8
            # Add small noise
            noise = np.random.normal(0, np.sqrt(variance))
            measured_toa = true_toa + noise
            
            # Use actual variance (not floor)
            graph.add_toa_factor(i, 4, measured_toa, variance)
        
        # Optimize
        result = graph.optimize(max_iterations=100, tolerance=1e-8, verbose=False)
        
        # Check result
        est_pos = result.estimates[4][:2]
        error = np.linalg.norm(est_pos - unknown_true)
        
        print(f"\nVariance = {variance:.0e} s² ({description})")
        print(f"  Initial error: 7.07 m")
        print(f"  Final error: {error:.3f} m")
        print(f"  Converged: {result.converged} in {result.iterations} iterations")
        print(f"  Final cost: {result.final_cost:.2e}")
        
        # Check if we actually moved
        initial_error = np.linalg.norm(initial[:2] - unknown_true)
        if error > initial_error * 0.9:
            print(f"  WARNING: Solver barely moved from initial guess!")

test_convergence_with_different_variances()

print("\n" + "="*70)
print("CONVERGENCE CRITERION CHECK")
print("="*70)

print("\nLooking at the convergence check in solver.py:")
print("  relative_decrease = cost_decrease / (prev_cost + 1e-20)")
print("  if relative_decrease < tolerance: converged = True")
print("")
print("With tiny variances, costs are tiny, so relative changes are huge")
print("This means the solver thinks it's not converging when it actually is")
print("")
print("The convergence criterion itself needs work!")
