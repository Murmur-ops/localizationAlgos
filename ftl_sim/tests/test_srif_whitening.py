"""
Test suite for proper SRIF (Square Root Information Form) whitening
Verifies that whitening produces N(0,1) distributed residuals
"""

import numpy as np
import pytest
from ftl.factors_scaled import ScaledFactor, ToAFactorMeters, ClockPriorFactor


class TestSRIFWhitening:
    """Test proper Square Root Information Form implementation"""

    def test_scalar_srif_whitening(self):
        """Test SRIF whitening for scalar measurements"""
        variance = 4.0  # std = 2.0
        factor = ScaledFactor(variance)

        # Check sqrt information is correct
        expected_sqrt_info = 1.0 / 2.0  # 1/std
        assert abs(factor.sqrt_information - expected_sqrt_info) < 1e-10

        # Test whitening of residual
        residual = 3.0
        whitened = factor.whiten(residual)
        expected = residual / 2.0  # Should divide by std
        assert abs(whitened - expected) < 1e-10

        # Verify SRIF property: L^T L = Information
        L = factor.sqrt_information
        information = L * L  # For scalar case
        assert abs(information - 1.0/variance) < 1e-10

    def test_whitened_residuals_distribution(self):
        """Test that whitened residuals are N(0,1) distributed"""
        np.random.seed(42)

        # Generate many measurements with known noise
        n_samples = 1000
        true_distance = 50.0
        noise_std = 0.5  # meters
        variance = noise_std ** 2

        whitened_residuals = []

        for _ in range(n_samples):
            # Add Gaussian noise to measurement
            noise = np.random.normal(0, noise_std)
            measured = true_distance + noise

            # Create factor
            factor = ToAFactorMeters(0, 1, measured, range_var_m2=variance)

            # Compute whitened residual
            state_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            state_j = np.array([50.0, 0.0, 0.0, 0.0, 0.0])  # True distance = 50m

            r_whitened, _, _ = factor.whitened_residual_and_jacobian(state_i, state_j)
            whitened_residuals.append(r_whitened)

        whitened_residuals = np.array(whitened_residuals)

        # Check statistics of whitened residuals
        mean = np.mean(whitened_residuals)
        std = np.std(whitened_residuals)

        # Should be approximately N(0,1)
        assert abs(mean) < 0.1  # Mean near 0
        assert abs(std - 1.0) < 0.1  # Std near 1

    def test_jacobian_whitening(self):
        """Test that Jacobian is properly whitened"""
        variance = 0.04  # 0.2m std
        factor = ToAFactorMeters(0, 1, 10.0, range_var_m2=variance)

        state_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        # Get raw Jacobian
        J_i_raw, J_j_raw = factor.jacobian(state_i, state_j)

        # Whiten manually
        std = np.sqrt(variance)
        J_i_expected = J_i_raw / std
        J_j_expected = J_j_raw / std

        # Get whitened Jacobian from factor
        J_i_whitened = factor.whiten_jacobian(J_i_raw)
        J_j_whitened = factor.whiten_jacobian(J_j_raw)

        assert np.allclose(J_i_whitened, J_i_expected)
        assert np.allclose(J_j_whitened, J_j_expected)

    def test_information_matrix_property(self):
        """Test that whitening satisfies information matrix properties"""
        variance = 0.01  # 0.1m std
        factor = ScaledFactor(variance)

        # For proper SRIF: L^T L = Σ^(-1) (information matrix)
        L = factor.sqrt_information
        info_from_L = L * L  # Scalar case

        # Direct information
        info_direct = factor.information

        assert abs(info_from_L - info_direct) < 1e-10

        # Test with actual residual
        residual = 0.15  # 15 cm error
        r_whitened = factor.whiten(residual)

        # Weighted squared residual should equal r_whitened^2
        weighted_r2 = residual**2 * factor.information
        whitened_r2 = r_whitened**2

        assert abs(weighted_r2 - whitened_r2) < 1e-10


class TestMatrixSRIF:
    """Test SRIF for matrix covariances (ClockPriorFactor)"""

    def test_clock_prior_srif(self):
        """Test SRIF implementation in ClockPriorFactor"""
        bias_var = 100.0  # 10 ns std
        drift_var = 4.0   # 2 ppb std

        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=50.0,
            drift_ppb=1.0,
            bias_var_ns2=bias_var,
            drift_var_ppb2=drift_var
        )

        # Check Cholesky decomposition
        cov = np.diag([bias_var, drift_var])
        info = np.linalg.inv(cov)

        # Verify L^T L = Information
        L = factor.L_info
        info_from_L = L.T @ L

        assert np.allclose(info_from_L, info)

    def test_clock_prior_whitened_residual(self):
        """Test that clock prior whitening produces correct statistics"""
        np.random.seed(42)

        bias_std = 10.0  # ns
        drift_std = 2.0  # ppb

        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=0.0,  # Prior mean
            drift_ppb=0.0,
            bias_var_ns2=bias_std**2,
            drift_var_ppb2=drift_std**2
        )

        # Generate many samples
        n_samples = 1000
        whitened_samples = []

        for _ in range(n_samples):
            # Random state with known distribution
            bias = np.random.normal(0, bias_std)
            drift = np.random.normal(0, drift_std)
            state = np.array([0.0, 0.0, bias, drift, 0.0])

            # Get whitened residual
            r_whitened = factor.whitened_residual(state)
            whitened_samples.append(r_whitened)

        whitened_samples = np.array(whitened_samples)

        # Check each dimension is approximately N(0,1)
        for dim in range(2):
            dim_samples = whitened_samples[:, dim]
            mean = np.mean(dim_samples)
            std = np.std(dim_samples)

            assert abs(mean) < 0.1  # Mean near 0
            assert abs(std - 1.0) < 0.1  # Std near 1

    def test_clock_prior_jacobian_whitening(self):
        """Test Jacobian whitening for clock prior"""
        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=0.0,
            drift_ppb=0.0,
            bias_var_ns2=100.0,
            drift_var_ppb2=4.0
        )

        state = np.array([0.0, 0.0, 5.0, 1.0, 0.0])

        # Get raw Jacobian
        J_raw = factor.jacobian(state)

        # Get whitened Jacobian
        J_whitened = factor.whitened_jacobian(state)

        # Manual whitening
        J_expected = factor.L_info @ J_raw

        assert np.allclose(J_whitened, J_expected)


class TestNumericalStability:
    """Test numerical stability of SRIF implementation"""

    def test_small_variance(self):
        """Test SRIF with very small variance (high precision)"""
        variance = 1e-10  # Very small variance
        factor = ScaledFactor(variance)

        # Should not overflow or underflow
        assert np.isfinite(factor.sqrt_information)
        assert factor.sqrt_information > 0

        # Test whitening
        residual = 1e-5
        whitened = factor.whiten(residual)
        assert np.isfinite(whitened)

    def test_large_variance(self):
        """Test SRIF with very large variance (low precision)"""
        variance = 1e10  # Very large variance
        factor = ScaledFactor(variance)

        # Should not overflow or underflow
        assert np.isfinite(factor.sqrt_information)
        assert factor.sqrt_information > 0

        # Test whitening
        residual = 1000.0
        whitened = factor.whiten(residual)
        assert np.isfinite(whitened)

    def test_zero_variance_handling(self):
        """Test handling of zero variance (infinite precision)"""
        variance = 0.0
        factor = ScaledFactor(variance)

        # Should handle gracefully
        assert factor.information == 1e10  # Capped at large value
        assert np.isfinite(factor.sqrt_information)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])