#!/usr/bin/env python3
"""
Analyze the numerical issues in the solver
"""

import numpy as np

# Simulate the weight computation
variances = [1e-20, 1e-19, 1e-18, 1e-17, 1e-16]

print("Impact of tiny variances on weights:")
print("="*50)
for var in variances:
    weight = 1.0 / var
    print(f"variance = {var:.0e} s² → weight = {weight:.0e}")
    
print("\nImpact on Hessian matrix:")
print("="*50)
print("If J has values ~1 (typical Jacobian entries)")
print("Then J_weighted = W @ J has values ~weight")
print("And H = J_weighted.T @ J_weighted has values ~weight²")
print("")
for var in variances:
    weight = 1.0 / var
    H_scale = weight**2
    print(f"variance = {var:.0e} → H diagonal ~{H_scale:.0e}")

print("\nNumerical precision issues:")
print("="*50)
print(f"float64 max: ~{np.finfo(np.float64).max:.0e}")
print(f"float64 epsilon: ~{np.finfo(np.float64).eps:.0e}")

# Test what happens with extreme weights
print("\nTest matrix solve with extreme scaling:")
print("="*50)

# Simple 2x2 system
for var in [1e-10, 1e-15, 1e-18, 1e-20]:
    weight = 1.0 / var
    
    # Weighted system
    H = np.array([[weight**2, 0], 
                  [0, weight**2]])
    g = np.array([weight, weight])
    
    try:
        delta = np.linalg.solve(H, g)
        print(f"var={var:.0e}: delta = [{delta[0]:.2e}, {delta[1]:.2e}]")
    except:
        print(f"var={var:.0e}: FAILED to solve")

print("\nThe issue: weights are too large!")
print("When weight = 1e20, numerical precision is lost.")
