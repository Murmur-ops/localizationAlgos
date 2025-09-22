#!/usr/bin/env python3
"""
Verify the ranging mathematics are correct
"""

import numpy as np
from ftl.factors_scaled import ToAFactorMeters

print("VERIFYING RANGING MATHEMATICS")
print("=" * 50)

# True positions
anchor0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
anchor1 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
true_node = np.array([5.0, 4.0, 0.0, 0.0, 0.0])

# True distances
true_dist_0 = np.sqrt(5**2 + 4**2)
true_dist_1 = np.sqrt(5**2 + 4**2)

print(f"True distances: {true_dist_0:.4f}m to both anchors")

# Create factors
factor0 = ToAFactorMeters(0, 2, true_dist_0, 1e-6)
factor1 = ToAFactorMeters(1, 2, true_dist_1, 1e-6)

# Check residuals at true position
r0 = factor0.residual(anchor0, true_node)
r1 = factor1.residual(anchor1, true_node)

print(f"\nResiduals at true position:")
print(f"  Factor 0->2: {r0*1000:.4f}mm")
print(f"  Factor 1->2: {r1*1000:.4f}mm")

# Check at slightly wrong position
test_node = np.array([5.0, 4.05, 0.0, 0.0, 0.0])
r0_test = factor0.residual(anchor0, test_node)
r1_test = factor1.residual(anchor1, test_node)

print(f"\nResiduals at (5.0, 4.05):")
print(f"  Factor 0->2: {r0_test*1000:.4f}mm")
print(f"  Factor 1->2: {r1_test*1000:.4f}mm")

# Check Jacobians
r0_wh, J0i_wh, J0j_wh = factor0.whitened_residual_and_jacobian(anchor0, test_node)
r1_wh, J1i_wh, J1j_wh = factor1.whitened_residual_and_jacobian(anchor1, test_node)

print(f"\nJacobians at test position:")
print(f"  J0j (x,y): [{J0j_wh[0]:.4f}, {J0j_wh[1]:.4f}]")
print(f"  J1j (x,y): [{J1j_wh[0]:.4f}, {J1j_wh[1]:.4f}]")

# Build mini optimization problem
H = np.zeros((5, 5))
g = np.zeros(5)

# Add both measurements
H += np.outer(J0j_wh, J0j_wh)
g += J0j_wh * r0_wh

H += np.outer(J1j_wh, J1j_wh)
g += J1j_wh * r1_wh

print(f"\nOptimization at (5.0, 4.05):")
print(f"  H diagonal (x,y): [{H[0,0]:.4f}, {H[1,1]:.4f}]")
print(f"  Gradient (x,y): [{g[0]:.4f}, {g[1]:.4f}]")

# What step would this produce?
# Solve H * delta = g
H_reg = H + 1e-6 * np.eye(5)
delta = np.linalg.solve(H_reg, g)

print(f"  Delta (x,y): [{delta[0]:.6f}, {delta[1]:.6f}]")

new_pos = test_node - 1.0 * delta  # Full step
print(f"  New position: [{new_pos[0]:.6f}, {new_pos[1]:.6f}]")

# The issue might be that both anchors are on the x-axis
print("\n" + "=" * 50)
print("DIAGNOSIS:")
print("Both anchors are at y=0, creating poor vertical observability")
print("The gradient in y-direction might be very small")

# Let's verify with better anchor placement
print("\n" + "=" * 50)
print("TEST WITH BETTER ANCHOR GEOMETRY")

# One anchor not on x-axis
anchor0_good = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
anchor1_good = np.array([8.0, 6.0, 0.0, 0.0, 0.0])  # Not on x-axis!
true_node = np.array([5.0, 4.0, 0.0, 0.0, 0.0])

dist_0 = np.linalg.norm(true_node[:2] - anchor0_good[:2])
dist_1 = np.linalg.norm(true_node[:2] - anchor1_good[:2])

factor0 = ToAFactorMeters(0, 2, dist_0, 1e-6)
factor1 = ToAFactorMeters(1, 2, dist_1, 1e-6)

# Check at slightly wrong position
test_node = np.array([5.0, 4.05, 0.0, 0.0, 0.0])

H = np.zeros((5, 5))
g = np.zeros(5)

r0_wh, J0i_wh, J0j_wh = factor0.whitened_residual_and_jacobian(anchor0_good, test_node)
r1_wh, J1i_wh, J1j_wh = factor1.whitened_residual_and_jacobian(anchor1_good, test_node)

H += np.outer(J0j_wh, J0j_wh)
g += J0j_wh * r0_wh
H += np.outer(J1j_wh, J1j_wh)
g += J1j_wh * r1_wh

print(f"With better geometry:")
print(f"  H diagonal (x,y): [{H[0,0]:.4f}, {H[1,1]:.4f}]")
print(f"  Gradient (x,y): [{g[0]:.4f}, {g[1]:.4f}]")

print("\nConclusion: Collinear anchors create poor observability!")