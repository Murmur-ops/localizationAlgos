"""
Unit tests for scaled factor classes
Test that factors work correctly with proper units (meters, nanoseconds, ppb, ppm)
"""

import numpy as np
import pytest
from ftl.factors_scaled import (
    ScaledState, ToAFactorMeters, TDOAFactorMeters, ClockPriorFactor
)


class TestScaledState:
    """Test the ScaledState dataclass"""

    def test_to_from_array(self):
        """Test conversion to/from numpy array"""
        state = ScaledState(x_m=10.0, y_m=20.0, bias_ns=5.0, drift_ppb=2.0, cfo_ppm=0.1)
        arr = state.to_array()

        assert arr.shape == (5,)
        assert arr[0] == 10.0
        assert arr[1] == 20.0
        assert arr[2] == 5.0
        assert arr[3] == 2.0
        assert arr[4] == 0.1

        # Round trip
        state2 = ScaledState.from_array(arr)
        assert state2.x_m == state.x_m
        assert state2.y_m == state.y_m
        assert state2.bias_ns == state.bias_ns
        assert state2.drift_ppb == state.drift_ppb
        assert state2.cfo_ppm == state.cfo_ppm


class TestToAFactorMeters:
    """Test ToA factor with meters/nanoseconds units"""

    def test_residual_no_bias(self):
        """Test residual computation without clock bias"""
        # Two nodes 10m apart
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # at origin, no clock bias
        xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])  # 10m away, no clock bias

        # Measured range = true range = 10m
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=0.01)

        residual = factor.residual(xi, xj)
        assert abs(residual) < 1e-10  # Should be zero

    def test_residual_with_bias(self):
        """Test residual with clock bias"""
        # Two nodes 10m apart
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no clock bias
        xj = np.array([10.0, 0.0, 10.0, 0.0, 0.0])  # 10ns clock bias

        # Clock bias of 10ns = 10e-9 * 299792458 ≈ 2.998m equivalent range
        # So measured range would be 10m + 2.998m = 12.998m
        c = 299792458.0  # m/s
        clock_range = 10e-9 * c
        expected_range = 10.0 + clock_range

        factor = ToAFactorMeters(i=0, j=1, range_meas_m=expected_range, range_var_m2=0.01)

        residual = factor.residual(xi, xj)
        assert abs(residual) < 1e-10  # Should be near zero

    def test_jacobian_position(self):
        """Test Jacobian w.r.t. position"""
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj = np.array([3.0, 4.0, 0.0, 0.0, 0.0])  # Distance = 5m

        factor = ToAFactorMeters(i=0, j=1, range_meas_m=5.0, range_var_m2=0.01)

        J_xi, J_xj = factor.jacobian(xi, xj)

        # Unit vector from i to j is [3/5, 4/5]
        # J_xi should have [3/5, 4/5, ...] for position components
        assert abs(J_xi[0] - 0.6) < 1e-10
        assert abs(J_xi[1] - 0.8) < 1e-10

        # J_xj should have [-3/5, -4/5, ...] for position components
        assert abs(J_xj[0] + 0.6) < 1e-10
        assert abs(J_xj[1] + 0.8) < 1e-10

    def test_jacobian_clock_bias(self):
        """Test Jacobian w.r.t. clock bias and drift"""
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=0.01)

        # Test with delta_t = 0 (no drift contribution)
        J_xi, J_xj = factor.jacobian(xi, xj, delta_t=0.0)

        # ∂r/∂bi should be c * 1e-9 (positive for node i)
        c_ns = 299792458.0 * 1e-9  # meters per nanosecond
        assert abs(J_xi[2] - c_ns) < 1e-6

        # ∂r/∂bj should be -c * 1e-9 (negative for node j)
        assert abs(J_xj[2] + c_ns) < 1e-6

        # Drift should be zero with delta_t = 0
        assert J_xi[3] == 0.0
        assert J_xj[3] == 0.0

        # CFO should not affect ToA
        assert J_xi[4] == 0.0
        assert J_xj[4] == 0.0

    def test_whitening(self):
        """Test that whitening produces unit variance residuals"""
        # 10cm range std
        range_std_m = 0.1
        range_var_m2 = range_std_m ** 2

        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        # Add 10cm error to measurement
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.1, range_var_m2=range_var_m2)

        # Raw residual should be 0.1m
        raw_residual = factor.residual(xi, xj)
        assert abs(raw_residual - 0.1) < 1e-10

        # Whitened residual should be 0.1m / 0.1m = 1.0
        whitened = factor.whiten(raw_residual)
        assert abs(whitened - 1.0) < 1e-10

    def test_realistic_uwb_variance(self):
        """Test with realistic UWB range variance"""
        # UWB can achieve 5-30cm ranging accuracy
        range_std_m = 0.05  # 5cm
        range_var_m2 = range_std_m ** 2

        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=range_var_m2)

        # Check that information (weight) is reasonable
        # Weight = 1/variance = 1/0.0025 = 400
        assert abs(factor.information - 400.0) < 1e-10

        # This is MUCH better than 1e18 we were getting before!


class TestTDOAFactorMeters:
    """Test TDOA factor with proper units"""

    def test_tdoa_residual(self):
        """Test TDOA residual computation"""
        # Mobile at origin
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Anchor 1 at 10m distance
        xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        # Anchor 2 at 20m distance
        xk = np.array([20.0, 0.0, 0.0, 0.0, 0.0])

        # TDOA range = (10m - 20m) = -10m
        factor = TDOAFactorMeters(i=0, j=1, k=2, tdoa_range_m=-10.0, range_var_m2=0.01)

        residual = factor.residual(xi, xj, xk)
        assert abs(residual) < 1e-10

    def test_tdoa_jacobian(self):
        """Test TDOA Jacobian computation"""
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj = np.array([3.0, 4.0, 0.0, 0.0, 0.0])  # 5m away
        xk = np.array([0.0, 10.0, 0.0, 0.0, 0.0])  # 10m away

        factor = TDOAFactorMeters(i=0, j=1, k=2, tdoa_range_m=-5.0, range_var_m2=0.01)

        J_xi, J_xj, J_xk = factor.jacobian(xi, xj, xk)

        # Unit vector to j: [3/5, 4/5]
        # Unit vector to k: [0, 1]
        # J_xi position should be uij - uik = [3/5, 4/5] - [0, 1] = [3/5, -1/5]
        assert abs(J_xi[0] - 0.6) < 1e-10
        assert abs(J_xi[1] + 0.2) < 1e-10

        # Mobile bias should not affect TDOA
        assert J_xi[2] == 0.0


class TestClockPriorFactor:
    """Test clock prior factors"""

    def test_clock_prior_residual(self):
        """Test clock prior residual"""
        # Prior: bias = 10ns, drift = 2ppb
        # State has bias = 12ns, drift = 3ppb
        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=10.0,
            drift_ppb=2.0,
            bias_var_ns2=1.0,  # 1ns² variance
            drift_var_ppb2=0.01  # 0.01ppb² variance
        )

        x = np.array([0.0, 0.0, 12.0, 3.0, 0.0])
        residual = factor.residual(x)

        assert residual.shape == (2,)
        assert abs(residual[0] - 2.0) < 1e-10  # bias error = 12 - 10 = 2ns
        assert abs(residual[1] - 1.0) < 1e-10  # drift error = 3 - 2 = 1ppb

    def test_clock_prior_jacobian(self):
        """Test clock prior Jacobian"""
        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=10.0,
            drift_ppb=2.0,
            bias_var_ns2=1.0,
            drift_var_ppb2=0.01
        )

        x = np.array([0.0, 0.0, 12.0, 3.0, 0.0])
        J = factor.jacobian(x)

        assert J.shape == (2, 5)

        # Should only have non-zero entries for bias and drift
        assert J[0, 2] == 1.0  # ∂(bias_residual)/∂bias
        assert J[1, 3] == 1.0  # ∂(drift_residual)/∂drift

        # Position and CFO shouldn't affect clock prior
        assert J[0, 0] == 0.0
        assert J[0, 1] == 0.0
        assert J[0, 4] == 0.0

    def test_clock_prior_whitening(self):
        """Test that clock prior whitening works correctly"""
        # Large variance for bias, small for drift
        factor = ClockPriorFactor(
            node_id=0,
            bias_ns=0.0,
            drift_ppb=0.0,
            bias_var_ns2=100.0,  # 10ns std
            drift_var_ppb2=1.0   # 1ppb std
        )

        x = np.array([0.0, 0.0, 10.0, 1.0, 0.0])  # 10ns bias, 1ppb drift

        raw_residual = factor.residual(x)
        whitened_residual = factor.whitened_residual(x)

        # After whitening, both components should have similar magnitude
        # 10ns / sqrt(100) = 1.0
        # 1ppb / sqrt(1) = 1.0
        assert abs(whitened_residual[0] - 1.0) < 1e-10
        assert abs(whitened_residual[1] - 1.0) < 1e-10


class TestNumericalScale:
    """Test that numerical scale is reasonable with proper units"""

    def test_typical_weights(self):
        """Verify weights are in reasonable range"""
        # Test various realistic range accuracies
        test_cases = [
            (0.01, 10000),   # 1cm → weight = 10000
            (0.05, 400),     # 5cm → weight = 400
            (0.10, 100),     # 10cm → weight = 100
            (0.30, 11.11),   # 30cm → weight ≈ 11
            (1.00, 1.0),     # 1m → weight = 1
        ]

        for range_std_m, expected_weight in test_cases:
            range_var_m2 = range_std_m ** 2
            factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=range_var_m2)
            assert abs(factor.information - expected_weight) < 0.1

        # All weights are between 1 and 10000 - perfectly reasonable!
        # No more 1e18 to 1e21 weights that cause numerical problems

    def test_whitened_residuals_distribution(self):
        """Test that whitened residuals are properly normalized"""
        np.random.seed(42)

        # Simulate many measurements with known error distribution
        n_trials = 1000
        range_std_m = 0.1  # 10cm
        range_var_m2 = range_std_m ** 2

        whitened_residuals = []

        for _ in range(n_trials):
            # True range is 10m
            true_range = 10.0

            # Add Gaussian noise
            measured_range = true_range + np.random.normal(0, range_std_m)

            # Create factor
            factor = ToAFactorMeters(i=0, j=1, range_meas_m=measured_range, range_var_m2=range_var_m2)

            # Compute whitened residual
            xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

            raw_residual = factor.residual(xi, xj)
            whitened = factor.whiten(raw_residual)
            whitened_residuals.append(whitened)

        whitened_residuals = np.array(whitened_residuals)

        # Whitened residuals should be ~N(0,1)
        mean = np.mean(whitened_residuals)
        std = np.std(whitened_residuals)

        assert abs(mean) < 0.1  # Mean should be near 0
        assert abs(std - 1.0) < 0.1  # Std should be near 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])