#!/usr/bin/env python3
"""
Unit tests for Allan variance-based clock noise modeling
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.clocks import ClockModel, ClockState


class TestAllanVarianceNoise(unittest.TestCase):
    """Test Allan variance-based noise evolution"""

    def test_bias_random_walk(self):
        """Test clock bias follows random walk with correct Allan variance"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)

        # Collect bias samples over time
        dt = 0.1  # 100ms steps
        n_steps = 1000
        biases = []

        for _ in range(n_steps):
            state = model.propagate_state(state, dt, add_noise=True)
            biases.append(state.bias)

        biases = np.array(biases)

        # Calculate Allan variance at different tau values
        tau_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        allan_tau, allan_dev = model.compute_allan_variance(biases, 1/dt, tau_values)

        # For random walk, Allan deviation should scale as sqrt(tau)
        # Check if it roughly follows this relationship
        log_tau = np.log10(tau_values[~np.isnan(allan_dev)])
        log_dev = np.log10(allan_dev[~np.isnan(allan_dev)])

        if len(log_tau) >= 2:
            # Fit a line to log-log plot
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_dev)

            # For random walk, slope should be ~0.5
            self.assertAlmostEqual(slope, 0.5, delta=0.3)

    def test_drift_white_noise(self):
        """Test clock drift innovations have white noise characteristics"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)

        # Collect drift samples
        dt = 0.1
        n_steps = 1000
        drifts = []

        for _ in range(n_steps):
            state = model.propagate_state(state, dt, add_noise=True)
            drifts.append(state.drift)

        drifts = np.array(drifts)

        # The INNOVATIONS (changes) in drift should be white noise
        drift_innovations = np.diff(drifts)

        # For white noise innovations, autocorrelation should be near zero for lag > 0
        from scipy.stats import pearsonr

        # Check autocorrelation at lag 1
        if len(drift_innovations) > 1:
            corr, _ = pearsonr(drift_innovations[:-1], drift_innovations[1:])
            # Should be low correlation for white noise
            self.assertLess(abs(corr), 0.3)  # Allow some correlation due to finite samples

    def test_allan_deviation_scaling(self):
        """Test Allan deviation matches specified oscillator parameters"""
        oscillator_types = {
            "TCXO": 1e-10,
            "OCXO": 1e-11,
            "CSAC": 1e-12
        }

        for osc_type, expected_allan in oscillator_types.items():
            model = ClockModel(oscillator_type=osc_type)
            self.assertEqual(model.allan_deviation_1s, expected_allan)

            # Sample initial state
            state = model.sample_initial_state()

            # Check noise parameters scale appropriately
            # Bias noise should be allan_dev * c
            expected_bias_noise = expected_allan * 3e8
            self.assertAlmostEqual(state.bias_noise_std, expected_bias_noise, delta=expected_bias_noise*0.1)

    def test_noise_accumulation_over_time(self):
        """Test noise accumulates correctly over different time scales"""
        model = ClockModel(oscillator_type="TCXO", allan_deviation_1s=1e-10)
        state = model.sample_initial_state(seed=42)

        # Run for different total times with different dt
        test_cases = [
            (0.1, 100),   # 0.1s steps for 10s total
            (1.0, 10),    # 1s steps for 10s total
        ]

        variances = []
        for dt, n_steps in test_cases:
            # Run multiple trials
            bias_samples = []
            for trial in range(100):
                test_state = ClockState(
                    bias=0, drift=0, cfo=0, sco_ppm=0,
                    bias_noise_std=state.bias_noise_std,
                    drift_noise_std=state.drift_noise_std,
                    cfo_noise_std=state.cfo_noise_std,
                    sco_noise_std=state.sco_noise_std
                )

                for _ in range(n_steps):
                    test_state = model.propagate_state(test_state, dt, add_noise=True)

                bias_samples.append(test_state.bias)

            # Calculate variance
            variances.append(np.var(bias_samples))

        # Both should give similar total variance for same total time
        # (within statistical fluctuation)
        ratio = variances[0] / variances[1]
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_cfo_sco_coherence(self):
        """Test CFO and SCO remain coherent as they evolve"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)

        # Initial CFO and SCO should be coherent
        initial_ratio = state.cfo / (model.carrier_freq_hz * state.sco_ppm / 1e6)

        # Evolve the clock
        dt = 1.0
        for _ in range(10):
            state = model.propagate_state(state, dt, add_noise=True)

        # Check they remain proportional (same oscillator drives both)
        # Note: they won't be perfectly proportional due to independent noise
        # but should remain correlated
        if state.sco_ppm != 0:
            final_ratio = state.cfo / (model.carrier_freq_hz * state.sco_ppm / 1e6)
            # Should be roughly similar (within factor of 2-3)
            ratio_change = abs(final_ratio / initial_ratio - 1)
            self.assertLess(ratio_change, 2.0)


class TestAllanVarianceComputation(unittest.TestCase):
    """Test Allan variance computation function"""

    def test_constant_frequency(self):
        """Test Allan variance of constant frequency offset"""
        # Constant frequency offset should give zero Allan deviation
        n_samples = 1000
        constant_offset = np.ones(n_samples) * 1e-9  # 1 ppb constant

        model = ClockModel()
        tau_values = np.array([1, 2, 5, 10])
        tau, allan_dev = model.compute_allan_variance(constant_offset, 1.0, tau_values)

        # Should be near zero for all tau
        for dev in allan_dev:
            if not np.isnan(dev):
                self.assertLess(dev, 1e-15)

    def test_linear_drift(self):
        """Test Allan variance of linear frequency drift"""
        # Linear drift in frequency
        n_samples = 1000
        t = np.arange(n_samples)
        freq_drift = 1e-12 * t  # Linear drift

        model = ClockModel()
        tau_values = np.array([1, 2, 5, 10])
        tau, allan_dev = model.compute_allan_variance(freq_drift, 1.0, tau_values)

        # Allan deviation should increase with tau for drift
        valid_devs = allan_dev[~np.isnan(allan_dev)]
        if len(valid_devs) > 1:
            for i in range(1, len(valid_devs)):
                self.assertGreater(valid_devs[i], valid_devs[i-1] * 0.9)

    def test_white_noise_allan(self):
        """Test Allan variance of white noise"""
        # White noise frequency
        np.random.seed(42)
        n_samples = 10000
        white_noise = np.random.normal(0, 1e-10, n_samples)

        model = ClockModel()
        tau_values = np.logspace(0, 2, 20)  # 1 to 100 seconds
        tau, allan_dev = model.compute_allan_variance(white_noise, 1.0, tau_values)

        # For white frequency noise, Allan deviation ∝ 1/sqrt(tau)
        valid_idx = ~np.isnan(allan_dev)
        log_tau = np.log10(tau[valid_idx])
        log_dev = np.log10(allan_dev[valid_idx])

        if len(log_tau) > 5:
            # Fit slope
            from scipy import stats
            slope, _, _, _, _ = stats.linregress(log_tau[:10], log_dev[:10])
            # Should be close to -0.5 for white frequency noise
            self.assertAlmostEqual(slope, -0.5, delta=0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)