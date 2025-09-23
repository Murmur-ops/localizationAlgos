#!/usr/bin/env python3
"""
Test gradient formulation in FTL consensus
"""

import numpy as np

def test_measurement_gradient():
    """
    Test how measurement residuals contribute to gradient in Gauss-Newton

    For least squares problem: min 1/2 ||f(x)||^2
    - Residual: r = f(x)
    - Cost: C = 1/2 * r^T * r
    - Gradient: g = J^T * r (where J is Jacobian of f)
    - Hessian approx: H = J^T * J
    - Newton step: solve H * delta = -g, then x_new = x + delta
    - Or equivalently: solve H * delta = g, then x_new = x - delta
    """

    print("=== Standard Gauss-Newton Formulation ===")

    # Simple 1D example: find x such that f(x) = x - 2 = 0
    x_true = 2.0
    x_curr = 5.0  # Current estimate

    # Residual and Jacobian
    r = x_curr - x_true  # Residual: f(x) = x - 2
    J = 1.0  # Jacobian: df/dx = 1

    # Gauss-Newton
    g = J * r  # Gradient of 1/2 ||f||^2
    H = J * J  # Hessian approximation

    print(f"Current x: {x_curr}")
    print(f"Residual r = x - {x_true} = {r}")
    print(f"Gradient g = J^T * r = {g}")
    print(f"Hessian H = J^T * J = {H}")

    # Newton step (two equivalent formulations)
    print("\nMethod 1: Solve H*delta = -g, then x_new = x + delta")
    delta1 = -g / H
    x_new1 = x_curr + delta1
    print(f"  delta = {delta1}")
    print(f"  x_new = {x_new1}")

    print("\nMethod 2: Solve H*delta = g, then x_new = x - delta")
    delta2 = g / H
    x_new2 = x_curr - delta2
    print(f"  delta = {delta2}")
    print(f"  x_new = {x_new2}")

    print(f"\nBoth give correct result: x_new = {x_new1} = {x_new2}")

    print("\n=== FTL Implementation Check ===")

    # FTL does: g += Ji_wh * r_wh
    # Then: solves H*delta = g
    # Then: x_new = x - alpha*delta

    print("FTL builds gradient as: g += J^T * r")
    print("FTL solves: H * delta = g")
    print("FTL updates: x_new = x - alpha * delta")
    print("\nThis is CORRECT for Gauss-Newton (Method 2 above)")

def test_consensus_gradient_formulation():
    """
    Test consensus penalty gradient
    """

    print("\n=== Consensus Penalty Gradient ===")

    # Consensus penalty: C = μ/2 * Σ ||x_i - x_j||^2
    # Gradient: ∂C/∂x_i = μ * Σ (x_i - x_j)

    x_i = np.array([5.0])
    x_j = np.array([3.0])
    mu = 1.0

    # Correct gradient
    consensus_gradient = mu * (x_i - x_j)
    print(f"x_i = {x_i[0]}, x_j = {x_j[0]}")
    print(f"Consensus gradient = μ * (x_i - x_j) = {consensus_gradient[0]}")

    # FTL computes:
    consensus_term = mu * (x_j - x_i)  # Note: x_j - x_i
    print(f"FTL consensus_term = μ * (x_j - x_i) = {consensus_term[0]}")

    # FTL then does: g -= consensus_term
    print(f"FTL adds to gradient: g -= consensus_term")
    print(f"This means: g += {-consensus_term[0]}")
    print(f"Which equals: g += μ * (x_i - x_j) = {mu * (x_i - x_j)[0]}")

    print("\nFTL's consensus gradient computation is CORRECT!")
    print("The g -= consensus_term is equivalent to g += μ*(x_i - x_j)")

if __name__ == "__main__":
    test_measurement_gradient()
    test_consensus_gradient_formulation()