"""
Test suite for clock drift term in ToA factors
Verifies that drift properly affects range measurements
"""

import numpy as np
import pytest
from ftl.factors_scaled import ToAFactorMeters, ScaledState


class TestClockDriftInToA:
    """Test that clock drift term is properly included in ToA factors"""

    def test_drift_affects_residual(self):
        """Test that drift term changes the residual"""
        # Create two states at known positions
        state_i = np.array([0.0, 0.0, 100.0, 5.0, 0.0])  # x, y, bias_ns, drift_ppb, cfo_ppm
        state_j = np.array([10.0, 0.0, 200.0, 15.0, 0.0])

        # True distance is 10 meters
        true_distance = 10.0

        # Clock contribution: (200-100) ns * c * 1e-9 = 100 * 0.3 = 30 meters
        clock_contrib = 100 * 299792458.0 * 1e-9  # ~30 meters

        # Create factor with measurement including clock
        measured_range = true_distance + clock_contrib
        factor = ToAFactorMeters(0, 1, measured_range, range_var_m2=0.01)

        # Test without drift (delta_t = 0)
        residual_no_drift = factor.residual(state_i, state_j, delta_t=0.0)
        expected_no_drift = 0.0  # Should match exactly
        assert abs(residual_no_drift - expected_no_drift) < 1e-10

        # Test with drift (delta_t = 1 second)
        delta_t = 1.0
        drift_contrib = (15.0 - 5.0) * delta_t * 299792458.0 * 1e-9  # 10 ppb * 1s * c
        # 10e-9 * 299792458 = ~3 meters
        residual_with_drift = factor.residual(state_i, state_j, delta_t=delta_t)
        expected_with_drift = -drift_contrib  # negative because drift adds to predicted
        assert abs(residual_with_drift - expected_with_drift) < 1e-6

    def test_drift_jacobian(self):
        """Test that Jacobian includes drift derivatives"""
        state_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        factor = ToAFactorMeters(0, 1, 10.0, range_var_m2=0.01)

        # Test Jacobian without drift
        J_i_no_drift, J_j_no_drift = factor.jacobian(state_i, state_j, delta_t=0.0)
        assert J_i_no_drift[3] == 0.0  # No drift contribution
        assert J_j_no_drift[3] == 0.0

        # Test Jacobian with drift
        delta_t = 2.0
        J_i_drift, J_j_drift = factor.jacobian(state_i, state_j, delta_t=delta_t)
        expected_drift_deriv = 299792458.0 * delta_t * 1e-9  # c * delta_t * 1e-9

        assert abs(J_i_drift[3] - expected_drift_deriv) < 1e-10
        assert abs(J_j_drift[3] + expected_drift_deriv) < 1e-10  # Negative for j

    def test_drift_over_time(self):
        """Test that drift accumulates properly over time"""
        state_i = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # 1 ppb drift on node i
        state_j = np.array([100.0, 0.0, 0.0, 0.0, 0.0])  # No drift on node j

        true_range = 100.0
        factor = ToAFactorMeters(0, 1, true_range, range_var_m2=0.01)

        # Test at different time points
        times = [0.0, 1.0, 10.0, 100.0]
        for t in times:
            residual = factor.residual(state_i, state_j, delta_t=t)
            # Drift contribution: (dj - di) * t * c * 1e-9
            # = (0 - 1) * t * c * 1e-9 = -1 * t * c * 1e-9
            # This gets ADDED to predicted range, so residual becomes more positive
            expected_drift_m = 1.0 * t * 299792458.0 * 1e-9  # Positive residual
            assert abs(residual - expected_drift_m) < 1e-6

    def test_numerical_stability_with_drift(self):
        """Test numerical stability with typical drift values"""
        # Typical values
        state_i = np.array([25.0, 25.0, 150.0, 2.5, 10.0])  # Typical indoor values
        state_j = np.array([30.0, 35.0, 175.0, 3.2, 12.0])

        distance = np.linalg.norm([30-25, 35-25])  # ~11.18 meters
        clock_m = (175-150) * 299792458.0 * 1e-9  # 25 ns = ~7.5 meters
        drift_m = (3.2-2.5) * 10.0 * 299792458.0 * 1e-9  # 0.7 ppb * 10s = ~2.1 meters

        measured = distance + clock_m + drift_m
        factor = ToAFactorMeters(0, 1, measured, range_var_m2=0.0001)

        residual = factor.residual(state_i, state_j, delta_t=10.0)
        assert abs(residual) < 1e-10  # Should be nearly zero

        # Check Jacobian is reasonable
        J_i, J_j = factor.jacobian(state_i, state_j, delta_t=10.0)
        assert np.all(np.isfinite(J_i))
        assert np.all(np.isfinite(J_j))
        assert np.linalg.norm(J_i[:2]) < 2.0  # Position derivatives bounded
        assert abs(J_i[3]) < 10.0  # Drift derivative reasonable

    def test_zero_drift_case(self):
        """Test that zero drift gives same result as before"""
        state_i = np.array([0.0, 0.0, 100.0, 0.0, 0.0])  # Zero drift
        state_j = np.array([10.0, 0.0, 200.0, 0.0, 0.0])  # Zero drift

        true_distance = 10.0
        clock_contrib = 100 * 299792458.0 * 1e-9
        measured = true_distance + clock_contrib

        factor = ToAFactorMeters(0, 1, measured, range_var_m2=0.01)

        # With zero drift, delta_t shouldn't matter
        for delta_t in [0.0, 1.0, 10.0, 100.0]:
            residual = factor.residual(state_i, state_j, delta_t=delta_t)
            assert abs(residual) < 1e-10  # Should always be zero


class TestWhitenedOperations:
    """Test that whitened operations work with drift"""

    def test_whitened_with_drift(self):
        """Test whitened residual and Jacobian with drift"""
        state_i = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 2.0, 0.0])

        variance = 0.04  # 20 cm std
        factor = ToAFactorMeters(0, 1, 10.0, range_var_m2=variance)

        r_wh, J_i_wh, J_j_wh = factor.whitened_residual_and_jacobian(
            state_i, state_j, delta_t=1.0
        )

        # Check whitening applied
        std = np.sqrt(variance)
        r_raw = factor.residual(state_i, state_j, delta_t=1.0)
        J_i_raw, J_j_raw = factor.jacobian(state_i, state_j, delta_t=1.0)

        assert abs(r_wh - r_raw / std) < 1e-10
        assert np.allclose(J_i_wh, J_i_raw / std)
        assert np.allclose(J_j_wh, J_j_raw / std)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])