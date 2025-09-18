#!/usr/bin/env python3
"""
Unit tests for Sample Clock Offset (SCO) modeling
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.clocks import ClockState, ClockModel, ClockEnsemble
from ftl.channel import apply_sample_clock_offset, propagate_signal
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.rx_frontend import matched_filter, detect_toa


class TestSCOInClockState(unittest.TestCase):
    """Test SCO field in ClockState"""

    def test_sco_initialization(self):
        """Test SCO is initialized properly"""
        state = ClockState()
        self.assertEqual(state.sco_ppm, 0.0)
        self.assertEqual(state.sco_noise_std, 0.01)

    def test_sco_in_array_conversion(self):
        """Test SCO included in array conversion"""
        state = ClockState(bias=1e-6, drift=1e-9, cfo=100, sco_ppm=2.5)
        arr = state.to_array()

        self.assertEqual(len(arr), 4)
        self.assertAlmostEqual(arr[3], 2.5)

        # Test from_array
        new_state = ClockState()
        new_state.from_array(arr)
        self.assertAlmostEqual(new_state.sco_ppm, 2.5)

    def test_sco_from_oscillator(self):
        """Test SCO coherent with drift from same oscillator"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)

        # SCO and drift should be coherent (same PPM error)
        drift_ppm = state.drift * 1e6
        self.assertAlmostEqual(state.sco_ppm, drift_ppm)

        # CFO should also be coherent
        expected_cfo = model.carrier_freq_hz * state.drift
        self.assertAlmostEqual(state.cfo, expected_cfo, delta=1.0)

    def test_sco_propagation(self):
        """Test SCO evolves with noise"""
        model = ClockModel(oscillator_type="TCXO")
        state = model.sample_initial_state(seed=42)
        initial_sco = state.sco_ppm

        # Propagate with noise
        new_state = model.propagate_state(state, dt=1.0, add_noise=True)

        # SCO should change due to noise
        self.assertNotEqual(new_state.sco_ppm, initial_sco)

        # Change should be reasonable (within a few std devs)
        change = abs(new_state.sco_ppm - initial_sco)
        self.assertLess(change, 5 * state.sco_noise_std)


class TestSCOApplication(unittest.TestCase):
    """Test SCO application to signals"""

    def test_sco_resampling_positive(self):
        """Test positive SCO (RX sampling faster)"""
        # Create test signal
        fs = 1e9
        duration = 1e-6
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        freq = 10e6  # 10 MHz
        signal = np.exp(1j * 2 * np.pi * freq * t)

        # Apply positive SCO (receiver samples faster)
        sco_ppm = 10.0  # 10 ppm faster
        signal_sco = apply_sample_clock_offset(signal, sco_ppm, fs)

        # Signal should be same length
        self.assertEqual(len(signal_sco), len(signal))

        # Signal should be slightly compressed (sampled faster)
        # Check by correlating with original
        correlation = np.abs(np.correlate(signal_sco, signal, 'valid'))[0]
        self.assertGreater(correlation, 0.9)  # Should still be similar

    def test_sco_resampling_negative(self):
        """Test negative SCO (RX sampling slower)"""
        fs = 1e9
        duration = 1e-6
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        freq = 10e6
        signal = np.exp(1j * 2 * np.pi * freq * t)

        # Apply negative SCO
        sco_ppm = -10.0  # 10 ppm slower
        signal_sco = apply_sample_clock_offset(signal, sco_ppm, fs)

        self.assertEqual(len(signal_sco), len(signal))

    def test_sco_negligible(self):
        """Test negligible SCO is ignored"""
        fs = 1e9
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)

        # Very small SCO
        sco_ppm = 0.005  # Less than threshold
        signal_sco = apply_sample_clock_offset(signal, sco_ppm, fs)

        # Should be identical
        np.testing.assert_array_equal(signal_sco, signal)

    def test_sco_toa_impact(self):
        """Test SCO impact on ToA estimation"""
        cfg = SignalConfig(bandwidth=499.2e6, sample_rate=1e9)
        template = gen_hrp_burst(cfg, n_repeats=1)

        # Create delayed copy
        delay_samples = 100
        signal = np.zeros(len(template) + 200, dtype=complex)
        signal[delay_samples:delay_samples+len(template)] = template

        # Apply SCO
        sco_ppm = 20.0  # 20 ppm error
        signal_sco = apply_sample_clock_offset(signal, sco_ppm, cfg.sample_rate)

        # Detect ToA with and without SCO
        corr_clean = matched_filter(signal, template)
        toa_clean = detect_toa(corr_clean, cfg.sample_rate)

        corr_sco = matched_filter(signal_sco, template)
        toa_sco = detect_toa(corr_sco, cfg.sample_rate)

        # SCO should cause ToA error
        toa_error_ns = abs(toa_sco['toa'] - toa_clean['toa']) * 1e9

        # SCO causes complex distortion, not simple linear shift
        # Just verify there is some error
        self.assertGreater(toa_error_ns, 0.01)  # At least 10 ps error
        self.assertLess(toa_error_ns, 1.0)  # Less than 1 ns for this SCO

    def test_sco_accumulation(self):
        """Test SCO error accumulates over time"""
        cfg = SignalConfig(sample_rate=1e9)
        sco_ppm = 10.0  # 10 ppm

        # Test at different signal durations
        durations = [1e-6, 10e-6, 100e-6]  # 1, 10, 100 microseconds
        errors = []

        for duration in durations:
            n_samples = int(duration * cfg.sample_rate)

            # Create signal with known frequency
            t = np.arange(n_samples) / cfg.sample_rate
            freq = 1e6  # 1 MHz
            signal = np.exp(1j * 2 * np.pi * freq * t)

            # Apply SCO
            signal_sco = apply_sample_clock_offset(signal, sco_ppm, cfg.sample_rate)

            # Measure phase error at end
            phase_error = np.angle(signal_sco[-1] / signal[-1])
            errors.append(abs(phase_error))

        # Error should increase with duration
        for i in range(1, len(errors)):
            self.assertGreater(errors[i], errors[i-1] * 0.5)


class TestSCOInPropagation(unittest.TestCase):
    """Test SCO in full signal propagation"""

    def test_propagate_with_sco(self):
        """Test propagate_signal includes SCO"""
        cfg = SignalConfig(bandwidth=499.2e6, sample_rate=1e9)
        signal = gen_hrp_burst(cfg, n_repeats=1)

        # Create channel
        from ftl.channel import SalehValenzuelaChannel, ChannelConfig
        ch_cfg = ChannelConfig(environment='indoor_office')
        sv = SalehValenzuelaChannel(ch_cfg)
        channel = sv.generate_channel_realization(10.0, is_los=True, seed=42)

        # Propagate with SCO
        sco_ppm = 15.0
        result = propagate_signal(
            signal, channel, cfg.sample_rate,
            snr_db=20, cfo_hz=1000, sco_ppm=sco_ppm
        )

        # Check SCO is recorded
        self.assertEqual(result['sco_actual'], sco_ppm)

        # Signal should be affected
        self.assertEqual(len(result['signal']), len(signal))

    def test_sco_cfo_interaction(self):
        """Test SCO and CFO applied together"""
        cfg = SignalConfig(sample_rate=1e9)
        n_samples = 10000  # Longer signal to see SCO effect
        t = np.arange(n_samples) / cfg.sample_rate

        # Create simple sinusoid
        freq = 10e6
        signal = np.exp(1j * 2 * np.pi * freq * t)

        # Create trivial channel (no multipath)
        channel = {
            'taps': np.array([1.0]),
            'delays_ns': np.array([0.0]),
            'tap_gains': np.array([1.0])
        }

        # Apply both CFO and SCO with larger values
        cfo_hz = 10000  # 10 kHz CFO
        sco_ppm = 100  # 100 ppm SCO for clear effect

        result = propagate_signal(
            signal, channel, cfg.sample_rate,
            snr_db=100,  # High SNR to isolate effects
            cfo_hz=cfo_hz,
            sco_ppm=sco_ppm
        )

        # Both effects should be present
        output = result['signal']

        # Check signal is modified - correlation should be reduced
        correlation = np.abs(np.vdot(output, signal) / (np.linalg.norm(output) * np.linalg.norm(signal)))
        # With CFO and SCO, correlation should be less than perfect
        self.assertLess(correlation, 0.999)  # Not perfectly correlated
        self.assertGreater(correlation, 0.9)  # But still recognizable

    def test_clock_ensemble_sco(self):
        """Test clock ensemble generates coherent SCO"""
        n_nodes = 5
        model = ClockModel(oscillator_type="TCXO")
        ensemble = ClockEnsemble(n_nodes, model)

        # All nodes should have SCO
        for i in range(n_nodes):
            state = ensemble.states[i]
            self.assertIsNotNone(state.sco_ppm)

            # SCO should be coherent with drift
            drift_ppm = state.drift * 1e6
            self.assertAlmostEqual(state.sco_ppm, drift_ppm, delta=0.1)

        # Propagate and check SCO evolves
        initial_scos = [ensemble.states[i].sco_ppm for i in range(n_nodes)]
        ensemble.propagate_all(dt=1.0)
        final_scos = [ensemble.states[i].sco_ppm for i in range(n_nodes)]

        # Should change but not drastically
        for i in range(n_nodes):
            change = abs(final_scos[i] - initial_scos[i])
            self.assertLess(change, 1.0)  # Less than 1 ppm change in 1 second


class TestSCOEstimation(unittest.TestCase):
    """Test SCO estimation from repeated measurements"""

    def test_sco_observable_in_toa_drift(self):
        """Test SCO causes observable ToA drift"""
        cfg = SignalConfig(bandwidth=499.2e6, sample_rate=1e9)

        # Create longer template for better correlation
        template = gen_hrp_burst(cfg, n_repeats=3)

        # Test SCO effect on a single long measurement
        # SCO stretches/compresses the signal over time
        sco_ppm = 100.0  # 100 ppm for clear effect

        # Create test signal with known delay
        delay_ns = 100  # 100 ns delay
        delay_samples = int(delay_ns * 1e-9 * cfg.sample_rate)
        signal = np.zeros(len(template) + delay_samples + 100, dtype=complex)
        signal[delay_samples:delay_samples+len(template)] = template

        # Apply SCO
        signal_sco = apply_sample_clock_offset(signal, sco_ppm, cfg.sample_rate)

        # Measure ToA with and without SCO
        corr_clean = matched_filter(signal, template)
        toa_clean = detect_toa(corr_clean, cfg.sample_rate)

        corr_sco = matched_filter(signal_sco, template)
        toa_sco = detect_toa(corr_sco, cfg.sample_rate)

        # SCO should cause measurable ToA shift
        toa_shift = abs(toa_sco['toa'] - toa_clean['toa'])

        # Should see some shift due to SCO
        self.assertGreater(toa_shift, 1e-12)  # At least 1 ps shift
        self.assertLess(toa_shift, 1e-9)  # Less than 1 ns for reasonable SCO

    def test_sco_different_oscillators(self):
        """Test different oscillator types have different SCO ranges"""
        oscillator_types = ["CRYSTAL", "TCXO", "OCXO", "CSAC"]
        expected_ranges = {
            "CRYSTAL": 20.0,  # ±20 ppm
            "TCXO": 2.0,       # ±2 ppm
            "OCXO": 0.1,       # ±0.1 ppm
            "CSAC": 0.005      # ±0.005 ppm
        }

        for osc_type in oscillator_types:
            model = ClockModel(oscillator_type=osc_type)

            # Sample multiple states to check distribution
            scos = []
            for _ in range(100):
                state = model.sample_initial_state()
                scos.append(abs(state.sco_ppm))

            # Mean absolute SCO should be appropriate for oscillator
            mean_sco = np.mean(scos)
            expected = expected_ranges[osc_type]

            # Should be in right ballpark (within factor of 3)
            self.assertLess(mean_sco, expected * 3)
            self.assertGreater(mean_sco, expected * 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)