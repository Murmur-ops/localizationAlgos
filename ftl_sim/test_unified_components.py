#!/usr/bin/env python3
"""
Unit tests for unified FTL system components
Tests each component individually to ensure correct functionality
"""

import numpy as np
import unittest
import yaml
from pathlib import Path

# Import components to test
from ftl.clocks import ClockState
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter, detect_toa, toa_crlb
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

# Import unified functions
from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    simulate_rf_measurement,
    setup_consensus_from_measurements
)


class TestRFSignalGeneration(unittest.TestCase):
    """Test RF signal generation components"""

    def test_signal_config_creation(self):
        """Test that SignalConfig can be created with proper types"""
        config = SignalConfig(
            carrier_freq=6.5e9,
            bandwidth=499.2e6,
            sample_rate=1e9,
            burst_duration=1e-6,
            prf=124.8e6
        )

        self.assertEqual(config.carrier_freq, 6.5e9)
        self.assertEqual(config.bandwidth, 499.2e6)
        self.assertEqual(config.sample_rate, 1e9)

    def test_hrp_burst_generation(self):
        """Test HRP-UWB burst generation"""
        config = SignalConfig(
            carrier_freq=6.5e9,
            bandwidth=499.2e6,
            sample_rate=1e9,
            burst_duration=1e-6
        )

        signal = gen_hrp_burst(config)

        # Check signal properties
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.dtype, np.complex128)
        expected_samples = int(config.burst_duration * config.sample_rate)
        self.assertEqual(len(signal), expected_samples)

        # Signal should have non-zero energy
        energy = np.sum(np.abs(signal)**2)
        self.assertGreater(energy, 0)

    def test_channel_config_creation(self):
        """Test ChannelConfig creation with environment presets"""
        config = ChannelConfig(
            environment='indoor',
            path_loss_exponent=2.0,
            shadowing_std_db=1.0
        )

        self.assertEqual(config.environment, 'indoor')
        self.assertEqual(config.path_loss_exponent, 2.0)
        self.assertEqual(config.shadowing_std_db, 1.0)

    def test_channel_propagation(self):
        """Test signal propagation through channel"""
        # Create signal
        sig_config = SignalConfig(sample_rate=1e9, burst_duration=1e-6)
        tx_signal = gen_hrp_burst(sig_config)

        # Create channel
        channel_config = ChannelConfig(environment='indoor')
        channel = SalehValenzuelaChannel(channel_config)
        channel_realization = channel.generate_channel_realization(
            distance_m=10.0,
            is_los=True
        )

        # Propagate signal
        result = propagate_signal(
            tx_signal,
            channel_realization,
            sig_config.sample_rate,
            snr_db=30.0
        )

        # Check result structure
        self.assertIn('signal', result)
        self.assertIn('true_toa', result)
        self.assertIsInstance(result['signal'], np.ndarray)
        self.assertEqual(len(result['signal']), len(tx_signal))


class TestClockModels(unittest.TestCase):
    """Test clock state initialization and behavior"""

    def test_clock_state_creation(self):
        """Test ClockState dataclass creation"""
        clock = ClockState(
            bias=1e-9,  # 1 ns
            drift=1e-12,  # 1 ppb
            cfo=1.0  # 1 Hz
        )

        self.assertEqual(clock.bias, 1e-9)
        self.assertEqual(clock.drift, 1e-12)
        self.assertEqual(clock.cfo, 1.0)

    def test_initialize_clock_states(self):
        """Test clock state initialization from config"""
        config = {
            'clocks': {
                'unknown_nodes': {
                    'initial_bias_std': '1e-9',
                    'initial_drift_std': '1e-12',
                    'initial_cfo_std': '1.0'
                },
                'anchor_nodes': {
                    'initial_bias_std': '1e-10',
                    'initial_drift_std': '1e-13',
                    'initial_cfo_std': '0.1'
                }
            }
        }

        clock_states = initialize_clock_states(config, n_nodes=10, n_anchors=3)

        # Check we have correct number of clocks
        self.assertEqual(len(clock_states), 10)

        # Check all clocks are ClockState objects
        for clock in clock_states.values():
            self.assertIsInstance(clock, ClockState)


class TestRFMeasurement(unittest.TestCase):
    """Test RF measurement simulation"""

    def setUp(self):
        """Set up test configuration"""
        self.rf_config = {
            'signal': {
                'carrier_freq': '6.5e9',
                'bandwidth': '499.2e6',
                'sample_rate': '1e9',
                'burst_duration': '1e-6',
                'prf': '124.8e6',
                'snr_db': '30.0'
            },
            'channel': {
                'environment': 'indoor',
                'path_loss_exponent': '2.0',
                'shadowing_std_db': '1.0'
            },
            'simulation': {
                'max_range': '50.0',
                'los_probability': '1.0',
                'enable_multipath': False  # Disable for testing
            }
        }

        # Create simple network
        self.true_positions = np.array([
            [0, 0],
            [10, 0],
            [0, 10]
        ])

        # Create clock states
        self.clock_states = {
            0: ClockState(bias=1e-9, drift=0, cfo=0),
            1: ClockState(bias=2e-9, drift=0, cfo=0),
            2: ClockState(bias=1.5e-9, drift=0, cfo=0)
        }

    def test_matched_filter(self):
        """Test matched filtering operation"""
        # Create signal
        config = SignalConfig(sample_rate=1e9, burst_duration=1e-6)
        signal = gen_hrp_burst(config)

        # Add noise
        noisy_signal = signal + 0.01 * np.random.randn(len(signal))

        # Matched filter
        correlation = matched_filter(noisy_signal, signal)

        # Check output
        self.assertIsInstance(correlation, np.ndarray)
        self.assertGreater(len(correlation), 0)

    def test_toa_detection(self):
        """Test ToA detection from correlation"""
        # Create correlation peak
        sample_rate = 1e9
        n_samples = 1000
        correlation = np.zeros(n_samples, dtype=complex)
        peak_idx = 500
        correlation[peak_idx] = 1.0
        correlation[peak_idx-1:peak_idx+2] = [0.5, 1.0, 0.5]

        # Add noise
        correlation += 0.01 * np.random.randn(n_samples)

        # Detect ToA
        result = detect_toa(correlation, sample_rate)

        # Check result
        self.assertIn('toa', result)
        self.assertIn('peak_value', result)
        self.assertIsInstance(result['toa'], float)

        # ToA should be close to peak position
        expected_toa = peak_idx / sample_rate
        self.assertAlmostEqual(result['toa'], expected_toa, places=8)

    def test_simulate_rf_measurement(self):
        """Test full RF measurement simulation"""
        meas = simulate_rf_measurement(
            0, 1,  # Nodes 0 to 1
            self.true_positions,
            self.clock_states,
            self.rf_config
        )

        # Check measurement structure
        self.assertIsNotNone(meas)
        self.assertIn('range_m', meas)
        self.assertIn('variance_m2', meas)
        self.assertIn('true_range', meas)

        # Check range is reasonable
        true_dist = np.linalg.norm(self.true_positions[1] - self.true_positions[0])
        self.assertAlmostEqual(meas['true_range'], true_dist, places=10)

        # Measured range should be close to true range in ideal conditions
        # Account for clock bias difference: (2ns - 1ns) * c ≈ 0.3m
        clock_bias_diff = (self.clock_states[1].bias - self.clock_states[0].bias) * 299792458
        expected_range = true_dist + clock_bias_diff

        # Debug output
        print(f"\nDebug RF measurement:")
        print(f"  True distance: {true_dist:.3f} m")
        print(f"  Clock bias diff: {clock_bias_diff:.3f} m")
        print(f"  Expected range: {expected_range:.3f} m")
        print(f"  Measured range: {meas['range_m']:.3f} m")
        print(f"  Error: {meas['range_m'] - expected_range:.3f} m")

        # In ideal conditions, should be within 1m
        self.assertLess(abs(meas['range_m'] - expected_range), 1.0)

    def test_crlb_calculation(self):
        """Test CRLB variance calculation"""
        snr_db = 30.0
        bandwidth_hz = 500e6

        snr_linear = 10 ** (snr_db / 10)
        variance = toa_crlb(snr_linear, bandwidth_hz)

        # Check variance is positive and reasonable
        self.assertGreater(variance, 0)
        self.assertLess(variance, 1e-6)  # Should be small for high SNR

        # Convert to distance std
        c = 299792458
        distance_std = np.sqrt(variance) * c

        # For 30dB SNR, 500MHz BW, should be ~cm level
        self.assertLess(distance_std, 0.1)  # Less than 10cm


class TestConsensusIntegration(unittest.TestCase):
    """Test consensus solver integration"""

    def test_consensus_config_creation(self):
        """Test ConsensusGNConfig creation"""
        config = ConsensusGNConfig(
            max_iterations=100,
            consensus_gain=0.05,
            step_size=0.5,
            gradient_tol=1e-4,
            step_tol=1e-5
        )

        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.consensus_gain, 0.05)

    def test_toa_factor_meters(self):
        """Test ToA factor in meters"""
        # Measurement includes both geometric distance and clock bias
        # If true distance is 10m and clock bias diff is 1ns (0.3m),
        # then measured range is 10.3m
        factor = ToAFactorMeters(
            i=0,
            j=1,
            range_meas_m=10.3,  # 10m + 0.3m from clock bias
            range_var_m2=0.01
        )

        # Create states with 1ns bias difference
        state_i = np.array([0, 0, 1.0, 0, 0])  # [x, y, bias_ns, drift_ppb, cfo_ppm]
        state_j = np.array([10, 0, 2.0, 0, 0])  # 1ns higher bias

        # Evaluate residual
        residual = factor.residual(state_i, state_j)

        # Residual should be near zero for perfect measurement
        # residual = measured - predicted
        # measured = 10.3m
        # predicted = 10m (geometric) + 0.3m (clock bias) = 10.3m
        # So residual should be ~0 (within floating point precision)
        self.assertAlmostEqual(residual, 0.0, places=3)  # Allow 1mm error

    def test_setup_consensus_from_measurements(self):
        """Test setting up consensus solver from measurements"""
        # Simple 3-node network
        true_positions = np.array([
            [0, 0],   # Anchor
            [10, 0],  # Unknown
            [0, 10]   # Unknown
        ])

        clock_states = {
            0: ClockState(bias=0, drift=0, cfo=0),
            1: ClockState(bias=1e-9, drift=0, cfo=0),
            2: ClockState(bias=2e-9, drift=0, cfo=0)
        }

        # Create simple measurements
        measurements = {
            (0, 1): [{
                'range_m': 10.3,  # 10m + 0.3m clock bias
                'variance_m2': 0.01,
                'cfo_diff': 0,
                'is_los': True
            }],
            (0, 2): [{
                'range_m': 10.6,  # 10m + 0.6m clock bias
                'variance_m2': 0.01,
                'cfo_diff': 0,
                'is_los': True
            }],
            (1, 2): [{
                'range_m': 14.44,  # ~14.14m + 0.3m clock diff
                'variance_m2': 0.01,
                'cfo_diff': 0,
                'is_los': True
            }]
        }

        consensus_config = {
            'comm_range': '50.0',
            'parameters': {
                'max_iterations': '10',
                'consensus_gain': '0.05',
                'step_size': '0.5',
                'gradient_tol': '1e-3',
                'step_tol': '1e-4',
                'verbose': False
            },
            'initialization': {
                'position_noise_std': '0.1'
            }
        }

        cgn = setup_consensus_from_measurements(
            true_positions,
            measurements,
            clock_states,
            n_anchors=1,
            consensus_config=consensus_config
        )

        # Check solver was created
        self.assertIsInstance(cgn, ConsensusGaussNewton)

        # Check nodes were added
        self.assertEqual(len(cgn.nodes), 3)

        # Run a few iterations
        cgn.set_true_positions(true_positions)
        results = cgn.optimize()

        # Check we got results
        self.assertIn('converged', results)
        self.assertIn('iterations', results)


class TestEndToEnd(unittest.TestCase):
    """Test full pipeline with minimal example"""

    def test_minimal_network(self):
        """Test with minimal 3-node network"""
        # Create config
        config = {
            'geometry': {
                'n_nodes': 2,  # Unknown nodes
                'n_anchors': 1,  # Anchor nodes
                'area_size': 20.0,
                'jitter_std': 0.0,
                'anchor_placement': 'corners'
            },
            'rf_simulation': {
                'clocks': {
                    'unknown_nodes': {
                        'initial_bias_std': '1e-10',
                        'initial_drift_std': '0',
                        'initial_cfo_std': '0'
                    },
                    'anchor_nodes': {
                        'initial_bias_std': '0',
                        'initial_drift_std': '0',
                        'initial_cfo_std': '0'
                    }
                },
                'signal': {
                    'carrier_freq': '6.5e9',
                    'bandwidth': '499.2e6',
                    'sample_rate': '1e9',
                    'burst_duration': '1e-6',
                    'prf': '124.8e6',
                    'snr_db': '50.0'
                },
                'channel': {
                    'environment': 'indoor',
                    'path_loss_exponent': '2.0',
                    'shadowing_std_db': '0.1'
                },
                'simulation': {
                    'n_rounds': 1,
                    'max_range': '50.0',
                    'los_probability': '1.0',
                    'enable_clock_drift': False
                }
            }
        }

        # Generate network
        true_positions, n_anchors_ret, n_total = generate_network_topology(config)

        self.assertEqual(len(true_positions), 3)
        self.assertEqual(n_anchors_ret, 1)
        self.assertEqual(n_total, 3)

        # Initialize clocks
        clock_states = initialize_clock_states(
            config['rf_simulation'],
            n_nodes=3,
            n_anchors=1
        )

        self.assertEqual(len(clock_states), 3)

        print("\nMinimal network test:")
        print(f"  Positions shape: {true_positions.shape}")
        print(f"  Clock states: {len(clock_states)}")

        # Test single measurement
        meas = simulate_rf_measurement(
            0, 1,
            true_positions,
            clock_states,
            config['rf_simulation']
        )

        if meas:
            print(f"  Measurement 0->1: range={meas['range_m']:.3f}m")
            print(f"    True range: {meas['true_range']:.3f}m")
            print(f"    Variance: {meas['variance_m2']:.6f}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)