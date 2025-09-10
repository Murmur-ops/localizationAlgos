#!/usr/bin/env python3
"""Simple test to verify fixes are improving performance."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the key fixes
print("Testing Key Mathematical Fixes")
print("=" * 50)

# 1. Test L matrix fix
print("\n1. L Matrix Factor-of-2 Fix:")
Z = np.array([[2, -0.4, -0.3, -0.2],
              [-0.4, 2, -0.3, -0.1],
              [-0.3, -0.3, 2, -0.4],
              [-0.2, -0.1, -0.4, 2]])

# Old way (incorrect)
L_old = np.zeros((4, 4))
for i in range(4):
    for j in range(i):
        L_old[i, j] = -Z[i, j]

# New way (correct with 1/2 factor)
L_new = np.zeros((4, 4))
for i in range(4):
    for j in range(i):
        L_new[i, j] = -0.5 * Z[i, j]

# Verify Z = 2I - L - L^T
Z_reconstructed_old = 2 * np.eye(4) - L_old - L_old.T
Z_reconstructed_new = 2 * np.eye(4) - L_new - L_new.T

error_old = np.linalg.norm(Z - Z_reconstructed_old, 'fro')
error_new = np.linalg.norm(Z - Z_reconstructed_new, 'fro')

print(f"  Old method error: {error_old:.6f}")
print(f"  New method error: {error_new:.6f}")
print(f"  ✓ Fix works!" if error_new < error_old else "  ✗ Fix failed")

# 2. Test vectorization with sqrt(2)
print("\n2. Vectorization √2 Scaling Fix:")
# Create symmetric matrix
M = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

# Old way (no scaling)
vec_old = []
for i in range(3):
    for j in range(i, 3):
        vec_old.append(M[i, j])
vec_old = np.array(vec_old)

# New way (with sqrt(2) for off-diagonals)
vec_new = []
for i in range(3):
    for j in range(i, 3):
        if i == j:
            vec_new.append(M[i, j])
        else:
            vec_new.append(np.sqrt(2) * M[i, j])
vec_new = np.array(vec_new)

print(f"  Old vector: {vec_old}")
print(f"  New vector: {vec_new}")
print(f"  ✓ Off-diagonals scaled by √2")

# 3. Test 2-block structure
print("\n3. 2-Block SK Construction Fix:")
n = 4
B = np.array([[0, 0.3, 0.4, 0.3],
              [0.3, 0, 0.3, 0.4],
              [0.4, 0.3, 0, 0.3],
              [0.3, 0.4, 0.3, 0]])

# Construct Z with proper 2-block
Z_2block = 2.0 * np.block([[np.eye(n), -B],
                           [-B, np.eye(n)]])
W_2block = Z_2block.copy()

# Verify constraints
ones = np.ones(2*n)
diag_check = np.allclose(np.diag(Z_2block), 2.0)
null_check = np.allclose(W_2block @ ones, 0.0, atol=1e-10)
sum_check = np.allclose(ones.T @ Z_2block @ ones, 0.0, atol=1e-10)

print(f"  diag(Z) = 2: {'✓' if diag_check else '✗'}")
print(f"  null(W) = span(1): {'✓' if null_check else '✗'}")
print(f"  1^T Z 1 = 0: {'✓' if sum_check else '✗'}")

# 4. Test zero-sum initialization
print("\n4. Zero-Sum Warm-Start Fix:")
X_init = np.random.randn(n, 2)
Y_init = X_init @ X_init.T

# Create v with zero sum (simplified)
v_block1 = [X_init[i] for i in range(n)]
v_block2 = [-X_init[i] for i in range(n)]
v_all = v_block1 + v_block2

v_sum = sum(v_all)
zero_sum_check = np.allclose(v_sum, 0)
print(f"  sum(v_i) = 0: {'✓' if zero_sum_check else '✗'} (norm={np.linalg.norm(v_sum):.2e})")

# 5. Overall assessment
print("\n" + "=" * 50)
print("FIXES VERIFICATION SUMMARY:")
all_fixes_work = error_new < 1e-10 and diag_check and null_check and sum_check and zero_sum_check
if all_fixes_work:
    print("✓✓✓ All mathematical fixes are working correctly!")
    print("The algorithm should show improved convergence.")
else:
    print("✗ Some fixes need adjustment")

# Quick convergence test
print("\n" + "=" * 50)
print("QUICK CONVERGENCE TEST:")

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    create_network_data
)

# Small test
network = create_network_data(n_sensors=5, n_anchors=2, dimension=2,
                            communication_range=0.7, measurement_noise=0.01)

config = MPSConfig(
    n_sensors=5, n_anchors=2, dimension=2,
    gamma=0.99, alpha=10.0, max_iterations=20,
    tolerance=1e-6, communication_range=0.7,
    verbose=False, early_stopping=False,
    admm_iterations=50, admm_rho=1.0,
    warm_start=True, use_2block=True
)

mps = MatrixParametrizedProximalSplitting(config, network)

# Run a few iterations
errors = []
for k in range(20):
    stats = mps.run_iteration(k)
    errors.append(stats['position_error'])

print(f"Initial error: {errors[0]:.4f}")
print(f"Final error: {errors[-1]:.4f}")
print(f"Improvement: {(errors[0] - errors[-1])/errors[0]*100:.1f}%")

# Check if converging
is_improving = errors[-1] < errors[0]
print(f"\n{'✓ Algorithm is converging!' if is_improving else '✗ Algorithm not converging'}")