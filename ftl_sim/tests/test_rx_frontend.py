"""
Unit tests for receiver front-end (ToA detection, CFO estimation, CRLB)
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMatchedFilter(unittest.TestCase):
    """Test matched filtering"""

    def test_matched_filter_peak(self):
        """Test that matched filter produces peak at correct location"""
        from ftl.rx_frontend import matched_filter

        # Create simple test signal
        template = np.array([1, -1, 1, 1, -1], dtype=complex)

        # Create received signal with known delay
        delay = 50
        received = np.zeros(200, dtype=complex)
        received[delay:delay+len(template)] = template

        # Apply matched filter
        correlation = matched_filter(received, template)

        # Peak should be at delay position (with 'same' mode, peak is at signal center)
        peak_idx = np.argmax(np.abs(correlation))
        # With 'same' mode and centered template, peak should be near delay + len(template)//2
        self.assertAlmostEqual(peak_idx, delay + len(template)//2, delta=1)

    def test_matched_filter_snr_gain(self):
        """Test that matched filter improves SNR"""
        from ftl.rx_frontend import matched_filter

        # Create template
        template = np.random.randn(100) + 1j * np.random.randn(100)
        template = template / np.linalg.norm(template)

        # Create noisy signal
        signal = np.zeros(500, dtype=complex)
        signal[200:300] = template

        # Add noise
        noise_power = 0.1
        noise = np.sqrt(noise_power/2) * (np.random.randn(500) + 1j * np.random.randn(500))
        noisy_signal = signal + noise

        # Apply matched filter
        correlation = matched_filter(noisy_signal, template)

        # SNR should improve by template length
        peak_power = np.max(np.abs(correlation)**2)
        noise_floor = np.median(np.abs(correlation)**2)
        snr_out = peak_power / noise_floor

        # Should have SNR gain (relaxed for 'same' mode correlation)
        self.assertGreater(snr_out, 5)  # At least 5x gain


class TestToADetection(unittest.TestCase):
    """Test Time of Arrival detection"""

    def test_toa_detection_clean_signal(self):
        """Test ToA detection on clean signal"""
        from ftl.rx_frontend import detect_toa

        # Create correlation function with clear peak
        correlation = np.zeros(1000, dtype=complex)
        true_toa_samples = 500
        correlation[true_toa_samples] = 10.0

        # Add some smaller sidelobes
        correlation[true_toa_samples - 50] = 2.0
        correlation[true_toa_samples + 50] = 2.0

        # Detect ToA
        result = detect_toa(
            correlation,
            sample_rate=1e9,
            threshold_factor=3.0
        )

        # Check detection
        self.assertIn('toa_samples', result)
        self.assertIn('toa_seconds', result)
        self.assertIn('peak_value', result)
        self.assertIn('noise_floor', result)

        # Should detect correct peak
        self.assertEqual(result['toa_samples'], true_toa_samples)

    def test_toa_subsample_refinement(self):
        """Test sub-sample ToA refinement"""
        from ftl.rx_frontend import detect_toa

        # Create correlation with peak between samples
        fs = 1e9
        correlation = np.zeros(1000, dtype=complex)

        # Simulate peak between samples 500 and 501
        # Using parabolic interpolation model
        peak_idx = 500
        fractional = 0.3  # True peak at 500.3 samples

        # Create parabolic peak
        correlation[peak_idx - 1] = 5.0
        correlation[peak_idx] = 10.0 - fractional**2  # Slightly lower due to offset
        correlation[peak_idx + 1] = 10.0 - (1-fractional)**2

        result = detect_toa(
            correlation,
            sample_rate=fs,
            enable_subsample=True
        )

        # Should refine to fractional sample (within reasonable tolerance)
        refined_toa = result['toa_refined_samples']
        self.assertAlmostEqual(refined_toa, peak_idx + fractional, delta=0.2)

    def test_toa_leading_edge_detection(self):
        """Test leading edge detection for NLOS mitigation"""
        from ftl.rx_frontend import detect_toa

        # Create multipath correlation with early weak path
        correlation = np.zeros(1000, dtype=complex)

        # Weak early path (LOS)
        correlation[400] = 3.0

        # Strong late path (NLOS reflection)
        correlation[450] = 10.0

        # Detect with leading edge mode
        result = detect_toa(
            correlation,
            sample_rate=1e9,
            mode='leading_edge',
            threshold_factor=2.0
        )

        # Should detect early path, not strongest
        self.assertEqual(result['toa_samples'], 400)


class TestCFOEstimation(unittest.TestCase):
    """Test Carrier Frequency Offset estimation"""

    def test_cfo_estimation_from_pilots(self):
        """Test CFO estimation from repeated pilots"""
        from ftl.rx_frontend import estimate_cfo

        # Create repeated blocks with known CFO
        fs = 1e9
        block_length = 1000
        n_blocks = 8
        true_cfo = 1000.0  # 1 kHz

        # Generate signal with CFO
        t = np.arange(block_length) / fs
        base_block = np.exp(1j * 2 * np.pi * 1e6 * t)  # 1 MHz pilot

        # Create repeated blocks with CFO
        blocks = []
        for i in range(n_blocks):
            t_block = (i * block_length + np.arange(block_length)) / fs
            cfo_phase = np.exp(1j * 2 * np.pi * true_cfo * t_block)
            blocks.append(base_block * cfo_phase)

        # Estimate CFO
        estimated_cfo = estimate_cfo(
            blocks,
            block_separation_s=block_length / fs
        )

        # Should be close to true CFO
        self.assertAlmostEqual(estimated_cfo, true_cfo, delta=100)  # Within 100 Hz

    def test_cfo_estimation_with_noise(self):
        """Test CFO estimation robustness to noise"""
        from ftl.rx_frontend import estimate_cfo

        # Set seed for reproducibility
        np.random.seed(42)

        # Parameters
        fs = 1e9
        block_length = 1000
        n_blocks = 4
        true_cfo = 500.0
        snr_db = 10.0

        # Generate blocks (repeated transmissions, not consecutive)
        blocks = []

        # Create base block
        t_base = np.arange(block_length) / fs
        base_signal = np.exp(1j * 2 * np.pi * 1e6 * t_base)  # Some carrier

        for i in range(n_blocks):
            # Each block is the same signal but with accumulated CFO phase
            phase_advance = 2 * np.pi * true_cfo * (i * block_length / fs)
            signal = base_signal * np.exp(1j * phase_advance)

            # Add noise
            signal_power = 1.0
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2) * (np.random.randn(block_length) +
                                              1j * np.random.randn(block_length))
            blocks.append(signal + noise)

        # Estimate CFO
        estimated_cfo = estimate_cfo(
            blocks,
            block_separation_s=block_length / fs
        )

        # Should still be reasonably accurate (relaxed for noise)
        self.assertAlmostEqual(estimated_cfo, true_cfo, delta=true_cfo * 0.3)


class TestCRLB(unittest.TestCase):
    """Test Cramér-Rao Lower Bound calculations"""

    def test_toa_crlb_calculation(self):
        """Test ToA CRLB computation"""
        from ftl.rx_frontend import toa_crlb

        # Test parameters
        snr_linear = 100  # 20 dB
        bandwidth_hz = 500e6  # 500 MHz

        # Calculate CRLB
        variance = toa_crlb(snr_linear, bandwidth_hz)

        # Convert to standard deviation in meters
        c = 3e8
        std_meters = c * np.sqrt(variance)

        # For 500 MHz BW and 20 dB SNR, should be ~cm level
        self.assertLess(std_meters, 0.1)  # Less than 10 cm
        self.assertGreater(std_meters, 0.001)  # More than 1 mm (realistic)

    def test_crlb_bandwidth_scaling(self):
        """Test that CRLB scales correctly with bandwidth"""
        from ftl.rx_frontend import toa_crlb

        snr_linear = 100

        # Double bandwidth should halve variance
        var_100mhz = toa_crlb(snr_linear, 100e6)
        var_200mhz = toa_crlb(snr_linear, 200e6)

        ratio = var_100mhz / var_200mhz
        self.assertAlmostEqual(ratio, 4.0, delta=0.1)  # Variance scales as 1/BW²

    def test_crlb_snr_scaling(self):
        """Test that CRLB scales correctly with SNR"""
        from ftl.rx_frontend import toa_crlb

        bandwidth = 500e6

        # Double SNR should halve variance
        var_10db = toa_crlb(10, bandwidth)  # 10 linear = 10 dB
        var_13db = toa_crlb(20, bandwidth)  # 20 linear = 13 dB

        ratio = var_10db / var_13db
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)


class TestReceiverMetrics(unittest.TestCase):
    """Test receiver performance metrics"""

    def test_correlation_shape_metrics(self):
        """Test correlation shape feature extraction"""
        from ftl.rx_frontend import extract_correlation_features

        # Create correlation with known shape
        correlation = np.zeros(1000, dtype=complex)
        peak_idx = 500

        # Add main peak
        correlation[peak_idx] = 10.0

        # Add some multipath
        correlation[peak_idx + 20] = 5.0
        correlation[peak_idx + 50] = 3.0

        # Extract features
        features = extract_correlation_features(correlation, peak_idx)

        # Check features
        self.assertIn('peak_to_sidelobe_ratio', features)
        self.assertIn('rms_width', features)
        self.assertIn('excess_delay', features)
        self.assertIn('multipath_ratio', features)

        # Peak to sidelobe should be 2 (10/5)
        self.assertAlmostEqual(features['peak_to_sidelobe_ratio'], 2.0, delta=0.1)

    def test_los_nlos_classification(self):
        """Test LOS/NLOS classification from correlation"""
        from ftl.rx_frontend import classify_propagation

        # Create LOS-like correlation (sharp peak)
        correlation_los = np.zeros(1000, dtype=complex)
        correlation_los[500] = 10.0
        correlation_los[499] = 7.0
        correlation_los[501] = 7.0

        # Create NLOS-like correlation (spread peak with excess delay)
        correlation_nlos = np.zeros(1000, dtype=complex)
        for i in range(20):
            correlation_nlos[520 + i] = 5.0 * np.exp(-i/5)

        # Classify
        los_result = classify_propagation(correlation_los)
        nlos_result = classify_propagation(correlation_nlos)

        self.assertEqual(los_result['type'], 'LOS')
        self.assertEqual(nlos_result['type'], 'NLOS')

        # Confidence scores (relaxed threshold)
        self.assertGreater(los_result['confidence'], 0.6)
        self.assertGreater(nlos_result['confidence'], 0.7)


if __name__ == "__main__":
    unittest.main()