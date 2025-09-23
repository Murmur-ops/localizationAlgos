#!/usr/bin/env python3
"""
Test whitening issue causing numerical instability
"""

import numpy as np
from ftl.factors_scaled import ToAFactorMeters

def test_whitening_scaling():
    """Test how whitening affects gradient scaling"""

    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

    print("=== Effect of Variance on Jacobian Scaling ===\n")

    variances = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    for var in variances:
        factor = ToAFactorMeters(
            i=0,
            j=1,
            range_meas_m=10.0,
            range_var_m2=var
        )

        # Get whitened Jacobian
        _, Ji_wh, Jj_wh = factor.whitened_residual_and_jacobian(xi, xj)

        # Get unwhitened for comparison
        Ji, Jj = factor.jacobian(xi, xj)

        std = np.sqrt(var)
        scaling = 1.0 / std

        print(f"Variance: {var:.1e} m²")
        print(f"  Std dev: {std:.1e} m")
        print(f"  Whitening scale: {scaling:.1e}")
        print(f"  Unwhitened J_i[0]: {Ji[0]:.3f}")
        print(f"  Whitened J_i[0]: {Ji_wh[0]:.3f}")
        print(f"  Ratio: {Ji_wh[0]/Ji[0] if Ji[0] != 0 else 0:.1e}\n")

    print("=== Gradient Descent with Different Variances ===\n")

    # Test convergence with different variances
    for var in [1e-6, 1e-2, 1.0]:
        print(f"Variance: {var:.1e} m²")

        factor = ToAFactorMeters(
            i=0,
            j=1,
            range_meas_m=10.0,
            range_var_m2=var
        )

        # Start with error
        xi_est = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj_est = np.array([11.0, 0.0, 0.0, 0.0, 0.0])  # 1m error

        # Adaptive step size based on variance
        step_size = 0.01 * var  # Scale step size with variance

        errors = []
        for iter in range(10):
            r, Ji, Jj = factor.whitened_residual_and_jacobian(xi_est, xj_est)
            gradient_j = Jj * r
            xj_est = xj_est - step_size * gradient_j
            error = np.linalg.norm(xj_est[:2] - xj[:2])
            errors.append(error)

        print(f"  Final error: {errors[-1]:.6f} m")
        print(f"  Converged: {errors[-1] < 0.01}\n")

    print("=== Recommendation ===")
    print("Use realistic measurement variance (e.g., 1e-2 to 1.0 m²)")
    print("OR scale step size inversely with whitening: step_size * std")

if __name__ == "__main__":
    test_whitening_scaling()