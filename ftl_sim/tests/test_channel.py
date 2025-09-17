"""
Unit tests for Saleh-Valenzuela channel model
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSalehValenzuelaChannel(unittest.TestCase):
    """Test Saleh-Valenzuela channel model"""

    def test_channel_generation_los(self):
        """Test LOS channel generation"""
        from ftl.channel import SalehValenzuelaChannel, ChannelConfig

        config = ChannelConfig()
        sv_channel = SalehValenzuelaChannel(config)

        # Generate LOS channel
        channel = sv_channel.generate_channel_realization(
            distance_m=10.0,
            is_los=True
        )

        # Check structure
        self.assertIn('taps', channel)
        self.assertIn('delays_ns', channel)
        self.assertIn('tap_gains', channel)
        self.assertIn('k_factor', channel)

        # Check LOS dominance
        self.assertGreater(channel['k_factor'], 0)  # Should have K-factor

        # First tap should be strongest in LOS
        tap_powers = np.abs(channel['tap_gains'])**2
        self.assertEqual(np.argmax(tap_powers), 0)

    def test_channel_generation_nlos(self):
        """Test NLOS channel generation"""
        from ftl.channel import SalehValenzuelaChannel, ChannelConfig

        config = ChannelConfig()
        sv_channel = SalehValenzuelaChannel(config)

        # Generate NLOS channel
        channel = sv_channel.generate_channel_realization(
            distance_m=20.0,
            is_los=False
        )

        # NLOS should have low/zero K-factor
        self.assertEqual(channel['k_factor'], 0)

        # Should have positive excess delay
        self.assertGreater(channel['excess_delay_ns'], 0)

    def test_cluster_structure(self):
        """Test that channel has cluster structure"""
        from ftl.channel import SalehValenzuelaChannel, ChannelConfig

        config = ChannelConfig()
        sv_channel = SalehValenzuelaChannel(config)

        channel = sv_channel.generate_channel_realization(
            distance_m=15.0,
            is_los=True
        )

        # Should have multiple taps
        self.assertGreater(len(channel['taps']), 1)

        # Delays should be increasing
        delays = channel['delays_ns']
        self.assertTrue(all(delays[i] <= delays[i+1]
                           for i in range(len(delays)-1)))

    def test_path_loss_calculation(self):
        """Test path loss models"""
        from ftl.channel import compute_path_loss

        # Free space at 1m should give reference
        loss_1m = compute_path_loss(
            distance_m=1.0,
            frequency_ghz=6.5,
            model='freespace'
        )

        # Loss should increase with distance
        loss_10m = compute_path_loss(
            distance_m=10.0,
            frequency_ghz=6.5,
            model='freespace'
        )

        self.assertGreater(loss_10m, loss_1m)

        # Free space: 20 dB per decade
        self.assertAlmostEqual(loss_10m - loss_1m, 20.0, delta=0.1)

    def test_signal_propagation(self):
        """Test signal propagation through channel"""
        from ftl.channel import propagate_signal, SalehValenzuelaChannel, ChannelConfig
        from ftl.signal import gen_zc_burst, SignalConfig

        # Generate test signal
        sig_config = SignalConfig(signal_type="ZADOFF_CHU", n_repeats=1)
        signal = gen_zc_burst(sig_config)

        # Create channel
        chan_config = ChannelConfig()
        sv_channel = SalehValenzuelaChannel(chan_config)

        channel = sv_channel.generate_channel_realization(
            distance_m=10.0,
            is_los=True
        )

        # Propagate signal
        output = propagate_signal(
            signal=signal,
            channel=channel,
            sample_rate=sig_config.sample_rate_hz,
            snr_db=20.0,
            cfo_hz=100.0,
            clock_bias_s=1e-6
        )

        # Check output structure
        self.assertIn('signal', output)
        self.assertIn('true_toa', output)
        self.assertIn('snr_actual', output)

        # Output should have same length as input
        self.assertEqual(len(output['signal']), len(signal))

        # Should have delay from clock bias
        self.assertGreater(output['true_toa'], 0)

    def test_awgn_noise_level(self):
        """Test AWGN noise addition"""
        from ftl.channel import add_awgn

        # Create test signal
        signal = np.ones(1000, dtype=complex)
        signal_power = np.mean(np.abs(signal)**2)

        # Add noise at specific SNR
        snr_db = 10.0
        noisy = add_awgn(signal, snr_db)

        # Measure noise power
        noise = noisy - signal
        noise_power = np.mean(np.abs(noise)**2)

        # Check SNR
        measured_snr_db = 10 * np.log10(signal_power / noise_power)
        self.assertAlmostEqual(measured_snr_db, snr_db, delta=1.0)


class TestChannelConfig(unittest.TestCase):
    """Test channel configuration"""

    def test_default_config(self):
        """Test default channel parameters"""
        from ftl.channel import ChannelConfig

        config = ChannelConfig()

        # UWB parameters
        self.assertEqual(config.carrier_freq_ghz, 6.5)
        self.assertEqual(config.bandwidth_mhz, 499.2)

        # S-V model parameters
        self.assertGreater(config.cluster_arrival_rate, 0)
        self.assertGreater(config.ray_arrival_rate, 0)

    def test_environment_presets(self):
        """Test environment-specific configurations"""
        from ftl.channel import ChannelConfig

        # Indoor office
        office = ChannelConfig(environment='indoor_office')
        self.assertLess(office.path_loss_exponent, 3.0)
        self.assertGreater(office.k_factor_db, 0)

        # Urban NLOS
        urban = ChannelConfig(environment='urban_nlos')
        self.assertGreater(urban.path_loss_exponent, 3.0)
        self.assertLess(urban.k_factor_db, 5.0)


if __name__ == "__main__":
    unittest.main()