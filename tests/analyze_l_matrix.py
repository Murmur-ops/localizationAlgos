#!/usr/bin/env python3
"""Analyze the L matrix decomposition issue."""

import numpy as np

print("L Matrix Decomposition Analysis")
print("=" * 50)

# Test symmetric Z matrix
Z = np.array([[2, -0.4, -0.3, -0.2],
              [-0.4, 2, -0.3, -0.1],
              [-0.3, -0.3, 2, -0.4],
              [-0.2, -0.1, -0.4, 2]])

print("Given Z matrix:")
print(Z)
print(f"\nZ is symmetric: {np.allclose(Z, Z.T)}")
print(f"diag(Z) = 2: {np.allclose(np.diag(Z), 2)}")

# For Z = 2I - L - L^T with strictly lower triangular L:
# - Diagonal: Z_ii = 2 - L_ii - L_ii = 2 (since L_ii = 0)
# - Off-diagonal (i≠j): Z_ij = -L_ij - L_ji

# Since L is strictly lower triangular:
# - L_ij = 0 for i < j (upper triangle is zero)
# - L_ij ≠ 0 for i > j (lower triangle)

# For symmetric Z:
# Z_ij = Z_ji = -L_ij - L_ji

# Case 1: i > j
# L_ij can be non-zero, L_ji = 0 (since j < i means (j,i) is upper)
# So: Z_ij = -L_ij - 0 = -L_ij
# Therefore: L_ij = -Z_ij

print("\nMethod 1: L_ij = -Z_ij for i>j (ORIGINAL):")
L1 = np.zeros((4, 4))
for i in range(4):
    for j in range(i):
        L1[i, j] = -Z[i, j]

Z1_reconstructed = 2 * np.eye(4) - L1 - L1.T
error1 = np.linalg.norm(Z - Z1_reconstructed, 'fro')
print(f"L1 (lower triangle):\n{L1}")
print(f"Reconstruction error: {error1:.10f}")
print(f"Z reconstructed correctly: {error1 < 1e-10}")

# Wait, this actually works! Let me check the 1/2 factor claim...
print("\nMethod 2: L_ij = -0.5*Z_ij for i>j (PROPOSED FIX):")
L2 = np.zeros((4, 4))
for i in range(4):
    for j in range(i):
        L2[i, j] = -0.5 * Z[i, j]

Z2_reconstructed = 2 * np.eye(4) - L2 - L2.T
error2 = np.linalg.norm(Z - Z2_reconstructed, 'fro')
print(f"L2 (lower triangle):\n{L2}")
print(f"Reconstruction error: {error2:.10f}")
print(f"Z reconstructed correctly: {error2 < 1e-10}")

print("\n" + "=" * 50)
print("CONCLUSION:")
print("The ORIGINAL implementation L_ij = -Z_ij was CORRECT!")
print("The 1/2 factor is NOT needed for strictly lower triangular L.")
print("\nThe confusion arose from:")
print("- Paper uses strictly lower triangular L (diagonal = 0)")
print("- When L_ji = 0 for j>i, we get Z_ij = -L_ij directly")
print("- No 1/2 factor needed!")

# Verify with paper's 2-block structure
print("\n" + "=" * 50)
print("Verifying with 2-Block Structure:")

n = 3
B = np.array([[0, 0.4, 0.6],
              [0.4, 0, 0.6],
              [0.6, 0.6, 0]])

# Paper's 2-block
Z_paper = 2.0 * np.block([[np.eye(n), -B],
                          [-B, np.eye(n)]])

print(f"Z shape: {Z_paper.shape}")
print(f"diag(Z) = 2: {np.allclose(np.diag(Z_paper), 2)}")

# Compute L for this Z
L_paper = np.zeros_like(Z_paper)
for i in range(Z_paper.shape[0]):
    for j in range(i):
        L_paper[i, j] = -Z_paper[i, j]  # Original formula

Z_check = 2 * np.eye(2*n) - L_paper - L_paper.T
error_check = np.linalg.norm(Z_paper - Z_check, 'fro')
print(f"Reconstruction error with original formula: {error_check:.10f}")
print(f"✓ Original formula L_ij = -Z_ij is CORRECT!")