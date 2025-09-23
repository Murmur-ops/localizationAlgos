#!/usr/bin/env python3
"""
Debug residual computation in ToAFactorMeters
"""

import numpy as np
from ftl.factors_scaled import ToAFactorMeters

# Test residual computation
def test_residual():
    """Test if residuals are computed correctly"""

    # Two nodes at known positions
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Node i at origin
    xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])  # Node j at (10, 0)

    # True distance
    true_distance = 10.0

    # Create factor with perfect measurement
    factor = ToAFactorMeters(
        i=0,
        j=1,
        range_meas_m=true_distance,
        range_var_m2=1e-6
    )

    # Compute residual
    residual = factor.residual(xi, xj)
    print(f"Perfect measurement residual: {residual}")
    print(f"Expected: 0.0")

    # Test with wrong position
    xj_wrong = np.array([11.0, 0.0, 0.0, 0.0, 0.0])  # 1m error
    residual_wrong = factor.residual(xi, xj_wrong)
    print(f"\nWith 1m position error:")
    print(f"  Residual: {residual_wrong}")
    print(f"  Expected: -1.0 (measured - predicted = 10 - 11)")

    # Test Jacobian
    r, Ji, Jj = factor.whitened_residual_and_jacobian(xi, xj)
    print(f"\nJacobians at true position:")
    print(f"  J_i: {Ji}")
    print(f"  J_j: {Jj}")
    print(f"  Sum should be zero for x,y components: {Ji[:2] + Jj[:2]}")

    # Test gradient direction
    print("\n=== Gradient Descent Test ===")

    # Start with error
    xi_est = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    xj_est = np.array([12.0, 0.0, 0.0, 0.0, 0.0])  # 2m error

    print(f"Initial: xj at {xj_est[:2]}, true at {xj[:2]}")

    for iter in range(5):
        # Get gradient
        r, Ji, Jj = factor.whitened_residual_and_jacobian(xi_est, xj_est)

        # For node j, gradient is J_j * r
        gradient_j = Jj * r

        # Update with gradient descent
        step_size = 0.1
        xj_est = xj_est - step_size * gradient_j

        error = np.linalg.norm(xj_est[:2] - xj[:2])
        print(f"Iter {iter}: xj at [{xj_est[0]:.3f}, {xj_est[1]:.3f}], error={error:.3f}m")

    print("\nShould converge toward true position (10.0, 0.0)")

if __name__ == "__main__":
    test_residual()