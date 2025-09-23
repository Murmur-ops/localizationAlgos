"""
Unit tests for frequency synchronization factors
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.frequency_factors.frequency_factors import (
    RangeFrequencyFactor,
    FrequencyPrior,
    DopplerFactor,
    CarrierPhaseFactor,
    MultiEpochFactor,
    FrequencyConfig
)


class TestRangeFrequencyFactor:
    """Test range measurements with frequency offset"""

    def test_zero_frequency_offset(self):
        """With zero frequency offset, should match standard range factor"""
        # Setup
        factor = RangeFrequencyFactor(
            measured_range=10.0,
            timestamp=100.0,
            sigma=0.1
        )

        # States with zero frequency offset
        state_i = np.array([0.0, 0.0, 5.0, 0.0])  # [x, y, tau_ns, freq_ppb]
        state_j = np.array([10.0, 0.0, 5.0, 0.0])

        # Geometric range = 10m, clock difference = 0
        error = factor.error(state_i, state_j)
        assert abs(error) < 1e-10, f"Expected zero error, got {error}"

    def test_frequency_drift_accumulation(self):
        """Frequency offset should accumulate over time"""
        # 10 ppb frequency offset over 1000 seconds
        factor = RangeFrequencyFactor(
            measured_range=10.015,  # Include drift effect
            timestamp=1000.0,
            sigma=0.1
        )

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 10.0])  # 10 ppb offset

        # Expected: 10m geometric + 10ppb * 1000s = 10m + 10us * c
        # 10us * 299792458 m/s * 1e-6 = 2.998m
        # But 10ppb * 1000s = 10000ns, * c * 1e-9 = 2.998m
        # Total = 10 + 2.998 = 12.998m
        error = factor.error(state_i, state_j)

        c = 299792458.0
        expected_range = 10.0 + 10.0 * 1000.0 * c * 1e-9
        actual_residual = (expected_range - 10.015) / 0.1

        assert abs(error - actual_residual) < 1e-6, \
            f"Frequency drift not accumulated correctly: {error} vs {actual_residual}"

    def test_jacobian_dimensions(self):
        """Jacobian should be 4D for extended state"""
        factor = RangeFrequencyFactor(100.0, 10.0, 0.1)

        state_i = np.array([1.0, 2.0, 3.0, 4.0])
        state_j = np.array([5.0, 6.0, 7.0, 8.0])

        J_i, J_j = factor.jacobian(state_i, state_j)

        assert J_i.shape == (4,), f"J_i shape {J_i.shape} != (4,)"
        assert J_j.shape == (4,), f"J_j shape {J_j.shape} != (4,)"

    def test_jacobian_finite_difference(self):
        """Verify Jacobian with finite differences"""
        factor = RangeFrequencyFactor(50.0, 100.0, 0.1)

        state_i = np.array([10.0, 20.0, 30.0, 5.0])
        state_j = np.array([40.0, 50.0, 60.0, -5.0])

        # Analytic Jacobian
        J_i_analytic, J_j_analytic = factor.jacobian(state_i, state_j)

        # Finite difference
        eps = 1e-7
        J_i_fd = np.zeros(4)
        J_j_fd = np.zeros(4)

        for k in range(4):
            # Perturb state_i
            state_i_plus = state_i.copy()
            state_i_plus[k] += eps
            err_plus = factor.error(state_i_plus, state_j)

            state_i_minus = state_i.copy()
            state_i_minus[k] -= eps
            err_minus = factor.error(state_i_minus, state_j)

            J_i_fd[k] = (err_plus - err_minus) / (2 * eps)

            # Perturb state_j
            state_j_plus = state_j.copy()
            state_j_plus[k] += eps
            err_plus = factor.error(state_i, state_j_plus)

            state_j_minus = state_j.copy()
            state_j_minus[k] -= eps
            err_minus = factor.error(state_i, state_j_minus)

            J_j_fd[k] = (err_plus - err_minus) / (2 * eps)

        # Compare
        np.testing.assert_allclose(J_i_analytic, J_i_fd, rtol=1e-5, atol=1e-7,
                                   err_msg="J_i mismatch")
        np.testing.assert_allclose(J_j_analytic, J_j_fd, rtol=1e-5, atol=1e-7,
                                   err_msg="J_j mismatch")


class TestFrequencyPrior:
    """Test frequency prior constraints"""

    def test_nominal_frequency(self):
        """At nominal frequency, prior should give zero error"""
        prior = FrequencyPrior(nominal_freq_ppb=5.0, sigma_ppb=10.0)

        state = np.array([1.0, 2.0, 3.0, 5.0])  # Frequency = nominal
        error = prior.error(state)

        assert abs(error) < 1e-10, f"Expected zero error at nominal, got {error}"

    def test_frequency_deviation(self):
        """Prior should penalize frequency deviation"""
        prior = FrequencyPrior(nominal_freq_ppb=0.0, sigma_ppb=10.0)

        state = np.array([1.0, 2.0, 3.0, 20.0])  # 20 ppb offset
        error = prior.error(state)

        expected = 20.0 / 10.0  # 2 sigma deviation
        assert abs(error - expected) < 1e-10, \
            f"Prior error {error} != expected {expected}"

    def test_jacobian(self):
        """Prior Jacobian should only affect frequency component"""
        prior = FrequencyPrior(nominal_freq_ppb=0.0, sigma_ppb=5.0)

        state = np.array([1.0, 2.0, 3.0, 10.0])
        J = prior.jacobian(state)

        assert J.shape == (4,)
        assert J[0] == 0.0, "Position x gradient should be zero"
        assert J[1] == 0.0, "Position y gradient should be zero"
        assert J[2] == 0.0, "Time gradient should be zero"
        assert abs(J[3] - 1.0/5.0) < 1e-10, f"Frequency gradient {J[3]} != 0.2"

    def test_short_state_vector(self):
        """Should handle state vectors without frequency gracefully"""
        prior = FrequencyPrior()

        state = np.array([1.0, 2.0, 3.0])  # No frequency component
        error = prior.error(state)

        assert error == 0.0, "Should return zero for missing frequency"


class TestDopplerFactor:
    """Test Doppler shift measurements"""

    def test_zero_relative_motion(self):
        """Zero relative velocity should give only frequency offset contribution"""
        factor = DopplerFactor(
            doppler_shift=15.75,  # 10 ppb at 1.575 GHz
            carrier_freq=1.575e9,
            sigma_hz=1.0
        )

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 10.0])  # 10 ppb offset

        # No velocity provided
        error = factor.error(state_i, state_j)

        # Expected: 10 ppb * 1.575e9 Hz = 15.75 Hz
        expected_doppler = 10.0 * 1e-9 * 1.575e9
        assert abs(expected_doppler - 15.75) < 1e-6

        # Error should be near zero since measurement matches prediction
        assert abs(error) < 1e-6, f"Error {error} too large"

    def test_with_velocity(self):
        """Test Doppler with relative motion"""
        factor = DopplerFactor(
            doppler_shift=100.0,
            carrier_freq=1.575e9,
            sigma_hz=1.0
        )

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 0.0])

        # Relative velocity along line-of-sight
        vel_i = np.array([0.0, 0.0])
        vel_j = np.array([10.0, 0.0])  # 10 m/s towards i

        error = factor.error(state_i, state_j, vel_i, vel_j)

        # Doppler from velocity: 10 m/s * 1.575e9 Hz / c
        c = 299792458.0
        expected_doppler = 10.0 * 1.575e9 / c

        residual = (expected_doppler - 100.0) / 1.0
        assert abs(error - residual) < 1e-6, \
            f"Doppler calculation error: {error} vs {residual}"

    def test_perpendicular_motion(self):
        """Perpendicular motion should not cause Doppler"""
        factor = DopplerFactor(0.0, 1.575e9, 1.0)

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 0.0])  # Along x-axis

        # Perpendicular velocity
        vel_i = np.array([0.0, 0.0])
        vel_j = np.array([0.0, 10.0])  # Along y-axis

        error = factor.error(state_i, state_j, vel_i, vel_j)

        assert abs(error) < 1e-10, \
            f"Perpendicular motion caused Doppler: {error}"


class TestCarrierPhaseFactor:
    """Test carrier phase measurements"""

    def test_phase_wrapping(self):
        """Phase should wrap correctly to [-0.5, 0.5] cycles"""
        factor = CarrierPhaseFactor(
            phase_cycles=0.1,
            wavelength=0.19,
            timestamp=100.0,
            sigma_cycles=0.01
        )

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([0.19, 0.0, 0.0, 0.0])  # 1 wavelength = 1 cycle

        # Predicted phase = 1 cycle, measured = 0.1 cycles
        # Wrapped difference should be 0.1 cycles (not 0.9)
        error = factor.error(state_i, state_j)
        expected = 0.1 / 0.01  # 0.1 cycle residual / sigma

        assert abs(error - expected) < 1e-6, \
            f"Phase wrapping error: {error} vs {expected}"

    def test_ambiguity_resolution(self):
        """Integer ambiguity should shift phase correctly"""
        factor = CarrierPhaseFactor(
            phase_cycles=0.3,
            wavelength=0.19,
            timestamp=100.0,
            sigma_cycles=0.01
        )

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([0.57, 0.0, 0.0, 0.0])  # 3 wavelengths

        # Without ambiguity: 3 cycles predicted, 0.3 measured
        error_no_amb = factor.error(state_i, state_j, ambiguity=None)

        # With correct ambiguity: predicted = 3, measured = 0.3
        # We need to add 3 to measured to get true phase = 3.3
        # But ambiguity is added to predicted, so we need negative
        # Actually: predicted + ambiguity should equal measured (mod 1)
        # 3 + (-2.7) = 0.3 after wrapping
        error_with_amb = factor.error(state_i, state_j, ambiguity=-2.7)

        assert abs(error_with_amb) < 1e-6, \
            f"Ambiguity resolution failed: {error_with_amb}"

    def test_frequency_phase_accumulation(self):
        """Frequency offset should cause quadratic phase accumulation"""
        factor = CarrierPhaseFactor(
            phase_cycles=0.0,
            wavelength=0.19,
            timestamp=10.0,
            sigma_cycles=0.01
        )

        # Large frequency offset to see effect
        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([0.0, 0.0, 0.0, 100.0])  # 100 ppb

        error = factor.error(state_i, state_j)

        # Frequency phase: 0.5 * df * t^2 * c / wavelength
        c = 299792458.0
        freq_phase = 0.5 * 100e-9 * 100 * c / 0.19

        # This should be non-zero due to frequency accumulation
        assert abs(freq_phase) > 0.01, \
            f"Frequency phase accumulation too small: {freq_phase}"


class TestMultiEpochFactor:
    """Test multi-epoch batch processing"""

    def test_multiple_measurements(self):
        """Should process multiple epochs correctly"""
        measurements = [10.0, 10.01, 10.02, 10.03]
        timestamps = [0.0, 10.0, 20.0, 30.0]

        factor = MultiEpochFactor(measurements, timestamps, sigma=0.01)

        assert factor.n_epochs == 4
        assert len(factor.measurements) == 4
        assert len(factor.timestamps) == 4

    def test_error_vector(self):
        """Should return vector of errors for all epochs"""
        measurements = [10.0, 10.0, 10.0]
        timestamps = [0.0, 100.0, 200.0]

        factor = MultiEpochFactor(measurements, timestamps, sigma=0.1)

        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 0.0])

        errors = factor.error_vector(state_i, state_j)

        assert errors.shape == (3,)
        # All should be near zero (10m range matches measurement)
        assert np.all(np.abs(errors) < 1e-6)

    def test_frequency_observability(self):
        """Multiple epochs should improve frequency observability"""
        # Create measurements with frequency drift
        true_freq_ppb = 5.0
        c = 299792458.0

        timestamps = [0.0, 100.0, 200.0, 300.0]
        measurements = []

        for t in timestamps:
            # Range increases due to frequency offset
            drift_m = true_freq_ppb * t * c * 1e-9
            measurements.append(10.0 + drift_m)

        factor = MultiEpochFactor(measurements, timestamps, sigma=0.01)

        # States with correct frequency
        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, true_freq_ppb])

        # Should have low error with correct frequency
        total_error = factor.total_error(state_i, state_j)
        assert total_error < 0.01, f"Error too high with correct frequency: {total_error}"

        # Wrong frequency should have high error
        state_j_wrong = np.array([10.0, 0.0, 0.0, 0.0])
        total_error_wrong = factor.total_error(state_i, state_j_wrong)
        assert total_error_wrong > 100.0, \
            f"Error too low with wrong frequency: {total_error_wrong}"


class TestFrequencyConfig:
    """Test configuration dataclass"""

    def test_default_values(self):
        """Check default configuration values"""
        config = FrequencyConfig()

        assert config.c == 299792458.0
        assert config.frequency_prior_ppb == 10.0
        assert config.phase_wavelength == 0.19
        assert config.doppler_carrier_freq == 1.575e9

    def test_custom_values(self):
        """Should accept custom configuration"""
        config = FrequencyConfig(
            frequency_prior_ppb=5.0,
            phase_wavelength=0.24
        )

        assert config.frequency_prior_ppb == 5.0
        assert config.phase_wavelength == 0.24
        assert config.c == 299792458.0  # Default unchanged


class TestIntegration:
    """Integration tests combining multiple factors"""

    def test_range_and_prior_combination(self):
        """Combine range factor with frequency prior"""
        # Create factors
        range_factor = RangeFrequencyFactor(10.0, 100.0, 0.1)
        freq_prior = FrequencyPrior(0.0, 10.0)

        # States
        state_i = np.array([0.0, 0.0, 0.0, 0.0])
        state_j = np.array([10.0, 0.0, 0.0, 50.0])  # Large frequency offset

        # Range error
        range_error = range_factor.error(state_i, state_j)

        # Prior errors
        prior_i = freq_prior.error(state_i)
        prior_j = freq_prior.error(state_j)

        # Prior should penalize large frequency
        assert abs(prior_j - 5.0) < 1e-10, f"Prior not penalizing: {prior_j}"

    def test_full_measurement_suite(self):
        """Test all measurement types together"""
        # Configuration
        config = FrequencyConfig()

        # Create all factor types
        range_factor = RangeFrequencyFactor(10.0, 100.0, 0.01)
        doppler_factor = DopplerFactor(10.0, config.doppler_carrier_freq, 1.0)
        phase_factor = CarrierPhaseFactor(0.5, config.phase_wavelength, 100.0, 0.01)
        prior = FrequencyPrior(0.0, config.frequency_prior_ppb)

        # States
        state_i = np.array([0.0, 0.0, 5.0, 2.0])
        state_j = np.array([10.0, 0.0, 10.0, 3.0])

        # All factors should produce finite errors
        assert np.isfinite(range_factor.error(state_i, state_j))
        assert np.isfinite(doppler_factor.error(state_i, state_j))
        assert np.isfinite(phase_factor.error(state_i, state_j))
        assert np.isfinite(prior.error(state_i))
        assert np.isfinite(prior.error(state_j))

        # Jacobians should be correct dimension
        J_i, J_j = range_factor.jacobian(state_i, state_j)
        assert J_i.shape == (4,) and J_j.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])