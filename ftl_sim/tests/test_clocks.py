"""
Unit tests for clock models
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.clocks import ClockState, ClockModel, ClockEnsemble


class TestClockState(unittest.TestCase):
    """Test ClockState class"""

    def test_state_creation(self):
        """Test basic clock state creation"""
        state = ClockState(bias=1e-6, drift=1e-9, cfo=100.0)
        self.assertEqual(state.bias, 1e-6)
        self.assertEqual(state.drift, 1e-9)
        self.assertEqual(state.cfo, 100.0)

    def test_to_array(self):
        """Test conversion to numpy array"""
        state = ClockState(bias=1e-6, drift=2e-9, cfo=50.0, sco_ppm=5.0)
        arr = state.to_array()
        np.testing.assert_array_equal(arr, np.array([1e-6, 2e-9, 50.0, 5.0]))

    def test_from_array(self):
        """Test update from numpy array"""
        state = ClockState()
        arr = np.array([1e-6, 2e-9, 50.0, 5.0])
        state.from_array(arr)
        self.assertEqual(state.bias, 1e-6)
        self.assertEqual(state.drift, 2e-9)
        self.assertEqual(state.cfo, 50.0)
        self.assertEqual(state.sco_ppm, 5.0)


class TestClockModel(unittest.TestCase):
    """Test ClockModel class"""

    def test_oscillator_defaults(self):
        """Test default parameters for different oscillator types"""
        # Test TCXO defaults
        tcxo = ClockModel(oscillator_type="TCXO")
        self.assertEqual(tcxo.frequency_accuracy_ppm, 2.0)
        self.assertEqual(tcxo.allan_deviation_1s, 1e-10)
        self.assertEqual(tcxo.temp_coefficient_ppm, 0.5)

        # Test OCXO defaults
        ocxo = ClockModel(oscillator_type="OCXO")
        self.assertEqual(ocxo.frequency_accuracy_ppm, 0.1)
        self.assertEqual(ocxo.allan_deviation_1s, 1e-11)
        self.assertEqual(ocxo.temp_coefficient_ppm, 0.01)

        # Test CSAC defaults
        csac = ClockModel(oscillator_type="CSAC")
        self.assertEqual(csac.frequency_accuracy_ppm, 0.005)
        self.assertEqual(csac.allan_deviation_1s, 1e-12)

        # Test Crystal defaults
        crystal = ClockModel(oscillator_type="CRYSTAL")
        self.assertEqual(crystal.frequency_accuracy_ppm, 20.0)
        self.assertEqual(crystal.allan_deviation_1s, 1e-9)

    def test_sample_initial_state(self):
        """Test initial state sampling"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)

        # Check state is properly initialized
        self.assertIsInstance(state, ClockState)

        # Check bias is reasonable (within ±10ms)
        self.assertLess(abs(state.bias), 0.01)

        # Check drift is within frequency accuracy
        drift_ppm = state.drift * 1e6
        self.assertLess(abs(drift_ppm), model.frequency_accuracy_ppm * 3)

        # Check CFO is consistent with drift
        expected_cfo_range = model.carrier_freq_hz * model.frequency_accuracy_ppm * 3 / 1e6
        self.assertLess(abs(state.cfo), expected_cfo_range)

    def test_sample_reproducibility(self):
        """Test that same seed gives same initial state"""
        model = ClockModel(oscillator_type="TCXO")
        state1 = model.sample_initial_state(seed=42)
        state2 = model.sample_initial_state(seed=42)

        self.assertEqual(state1.bias, state2.bias)
        self.assertEqual(state1.drift, state2.drift)
        self.assertEqual(state1.cfo, state2.cfo)

    def test_propagate_state_deterministic(self):
        """Test state propagation without noise"""
        model = ClockModel(oscillator_type="TCXO")
        initial_state = ClockState(bias=1e-6, drift=1e-9, cfo=100.0)

        # Propagate for 1 second without noise
        dt = 1.0
        new_state = model.propagate_state(initial_state, dt, add_noise=False)

        # Bias should increase by drift * dt
        expected_bias = initial_state.bias + initial_state.drift * dt
        self.assertAlmostEqual(new_state.bias, expected_bias)

        # Drift and CFO should remain unchanged without noise
        self.assertEqual(new_state.drift, initial_state.drift)
        self.assertEqual(new_state.cfo, initial_state.cfo)

    def test_propagate_state_with_noise(self):
        """Test state propagation with process noise"""
        model = ClockModel(oscillator_type="TCXO")
        initial_state = ClockState(bias=0, drift=0, cfo=0)

        # Propagate multiple times to check noise is added
        states = []
        for i in range(10):
            np.random.seed(i)  # Different seed each time
            new_state = model.propagate_state(initial_state, dt=1.0, add_noise=True)
            states.append(new_state)

        # Check that states are different (noise is being added)
        biases = [s.bias for s in states]
        self.assertGreater(np.std(biases), 0)

    def test_apply_to_timestamp(self):
        """Test applying clock error to timestamp"""
        model = ClockModel()
        clock_state = ClockState(bias=1e-6, drift=1e-9, cfo=0)

        true_time = 10.0
        observed_time = model.apply_to_timestamp(true_time, clock_state)

        # observed = true + bias + drift * true
        expected = true_time + clock_state.bias + clock_state.drift * true_time
        self.assertAlmostEqual(observed_time, expected)

    def test_apply_cfo_to_signal(self):
        """Test CFO application to signal"""
        model = ClockModel()

        # Create simple test signal
        fs = 1000.0  # 1 kHz sample rate
        duration = 1.0
        t = np.arange(0, duration, 1/fs)
        signal = np.ones(len(t), dtype=complex)

        # Apply 10 Hz CFO
        cfo_hz = 10.0
        signal_with_cfo = model.apply_cfo_to_signal(signal, cfo_hz, fs)

        # Check phase rotation
        expected_phase = 2 * np.pi * cfo_hz * t
        actual_phase = np.angle(signal_with_cfo)

        # Phase should match (modulo 2π)
        phase_diff = np.mod(actual_phase - expected_phase + np.pi, 2*np.pi) - np.pi
        self.assertLess(np.max(np.abs(phase_diff)), 0.01)

    def test_carrier_frequency_effect(self):
        """Test that carrier frequency affects CFO calculation"""
        model_low = ClockModel(carrier_freq_hz=2.4e9)  # 2.4 GHz
        model_high = ClockModel(carrier_freq_hz=6.5e9)  # 6.5 GHz

        # Same oscillator error should give different CFO
        state_low = model_low.sample_initial_state(seed=42)
        state_high = model_high.sample_initial_state(seed=42)

        # CFO ratio should match carrier frequency ratio
        cfo_ratio = abs(state_high.cfo / state_low.cfo) if state_low.cfo != 0 else 1
        freq_ratio = model_high.carrier_freq_hz / model_low.carrier_freq_hz

        # Allow some tolerance due to randomness
        if state_low.cfo != 0:
            self.assertAlmostEqual(cfo_ratio, freq_ratio, places=1)


class TestClockEnsemble(unittest.TestCase):
    """Test ClockEnsemble class"""

    def test_ensemble_creation(self):
        """Test creating ensemble of clocks"""
        model = ClockModel(oscillator_type="TCXO")
        ensemble = ClockEnsemble(n_nodes=10, model=model)

        self.assertEqual(ensemble.n_nodes, 10)
        self.assertEqual(len(ensemble.states), 10)

        # Check all states are initialized
        for i in range(10):
            self.assertIn(i, ensemble.states)
            self.assertIsInstance(ensemble.states[i], ClockState)

    def test_anchor_better_clocks(self):
        """Test that anchors get better oscillators"""
        model = ClockModel(oscillator_type="TCXO")
        anchor_indices = [0, 1, 2]
        ensemble = ClockEnsemble(
            n_nodes=10,
            model=model,
            anchor_indices=anchor_indices
        )

        # Anchors should have better Allan deviation
        for i in range(10):
            if i in anchor_indices:
                # Anchors use OCXO (better)
                self.assertLessEqual(
                    ensemble.states[i].drift_noise_std,
                    1e-11  # OCXO level
                )
            else:
                # Regular nodes use TCXO
                self.assertGreater(
                    ensemble.states[i].drift_noise_std,
                    1e-11
                )

    def test_propagate_all(self):
        """Test propagating all clock states"""
        model = ClockModel(oscillator_type="TCXO")
        ensemble = ClockEnsemble(n_nodes=5, model=model)

        # Store initial biases
        initial_biases = {i: ensemble.states[i].bias for i in range(5)}

        # Propagate for 1 second
        ensemble.propagate_all(dt=1.0)

        # Check that all states have changed
        for i in range(5):
            # Bias should have changed due to drift
            self.assertNotEqual(
                ensemble.states[i].bias,
                initial_biases[i]
            )

    def test_relative_clock_offset(self):
        """Test computing relative clock offset"""
        model = ClockModel()
        ensemble = ClockEnsemble(n_nodes=3, model=model)

        # Set known clock states
        ensemble.states[0].bias = 1e-6
        ensemble.states[1].bias = 3e-6

        relative = ensemble.get_relative_clock_offset(0, 1)
        self.assertAlmostEqual(relative, 2e-6)

    def test_relative_cfo(self):
        """Test computing relative CFO"""
        model = ClockModel()
        ensemble = ClockEnsemble(n_nodes=3, model=model)

        # Set known CFOs
        ensemble.states[0].cfo = 100.0
        ensemble.states[1].cfo = 150.0

        relative = ensemble.get_relative_cfo(0, 1)
        self.assertAlmostEqual(relative, 50.0)


class TestAllanVariance(unittest.TestCase):
    """Test Allan variance computation"""

    def test_allan_variance_white_noise(self):
        """Test Allan variance for white frequency noise"""
        model = ClockModel()

        # Generate white frequency noise
        fs = 1.0  # 1 Hz sample rate
        n_samples = 10000
        noise_std = 1e-10
        time_series = np.random.normal(0, noise_std, n_samples)

        # Compute Allan variance
        tau_values = np.array([1, 2, 5, 10, 20])
        tau, allan_dev = model.compute_allan_variance(time_series, fs, tau_values)

        # For white frequency noise, Allan deviation should decrease as 1/sqrt(tau)
        # Check the trend (not exact values due to finite samples)
        ratios = allan_dev[1:] / allan_dev[:-1]
        expected_ratios = np.sqrt(tau_values[:-1] / tau_values[1:])

        # Allow significant tolerance due to finite sample size
        for ratio, expected in zip(ratios[np.isfinite(ratios)],
                                   expected_ratios[np.isfinite(ratios)]):
            self.assertAlmostEqual(ratio, expected, delta=0.5)

    def test_allan_variance_tau_range(self):
        """Test Allan variance tau value handling"""
        model = ClockModel()

        # Short time series
        time_series = np.random.randn(100)
        fs = 1.0

        # Request tau values that are too large
        tau_values = np.array([1, 10, 100, 1000])
        tau, allan_dev = model.compute_allan_variance(time_series, fs, tau_values)

        # Large tau values should return NaN
        self.assertFalse(np.isnan(allan_dev[0]))  # tau=1 should work
        self.assertTrue(np.isnan(allan_dev[-1]))  # tau=1000 should be NaN


if __name__ == "__main__":
    unittest.main()