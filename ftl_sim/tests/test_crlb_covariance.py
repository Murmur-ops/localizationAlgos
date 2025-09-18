#!/usr/bin/env python3
"""
Unit tests for CRLB-based covariance calculation
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.signal import (
    SignalConfig, gen_hrp_burst, gen_zc_burst,
    compute_rms_bandwidth, compute_signal_snr
)
from ftl.rx_frontend import (
    matched_filter, detect_toa, toa_crlb, cov_from_crlb,
    extract_correlation_features, classify_propagation,
    covariance_from_features
)


class TestRMSBandwidth(unittest.TestCase):
    """Test RMS bandwidth calculation for CRLB"""

    def test_flat_spectrum(self):
        """Test RMS bandwidth for flat spectrum signal"""
        # White noise has flat spectrum
        np.random.seed(42)
        signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        fs = 1e9

        beta_rms = compute_rms_bandwidth(signal, fs)

        # For flat spectrum: β_rms ≈ B/sqrt(3) where B = fs/2
        expected = fs / (2 * np.sqrt(3))

        # Should be within 20% for finite sample
        self.assertAlmostEqual(beta_rms / expected, 1.0, delta=0.2)

    def test_hrp_uwb_bandwidth(self):
        """Test RMS bandwidth for HRP-UWB signal"""
        cfg = SignalConfig(bandwidth=499.2e6, sample_rate=1e9)
        signal = gen_hrp_burst(cfg, n_repeats=3)

        beta_rms = compute_rms_bandwidth(signal, cfg.sample_rate)

        # RMS bandwidth should be related to nominal bandwidth
        # For UWB pulses, typically β_rms ≈ 0.2-0.5 * B
        ratio = beta_rms / cfg.bandwidth
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 0.6)

    def test_zc_bandwidth(self):
        """Test RMS bandwidth for Zadoff-Chu signal"""
        cfg = SignalConfig(sequence_length=127, sample_rate=1e9)
        signal = gen_zc_burst(cfg, n_repeats=3)

        beta_rms = compute_rms_bandwidth(signal, cfg.sample_rate)

        # ZC sequence has narrower bandwidth
        self.assertGreater(beta_rms, 1e6)  # At least 1 MHz
        self.assertLess(beta_rms, 100e6)   # Less than 100 MHz

    def test_zero_signal(self):
        """Test RMS bandwidth for zero signal"""
        signal = np.zeros(1000, dtype=complex)
        fs = 1e9

        beta_rms = compute_rms_bandwidth(signal, fs)

        # Should return fallback value
        expected = fs / (2 * np.sqrt(3))
        self.assertAlmostEqual(beta_rms, expected)


class TestCRLBCalculation(unittest.TestCase):
    """Test CRLB variance calculation"""

    def test_crlb_basic(self):
        """Test basic CRLB calculation"""
        snr_linear = 100  # 20 dB
        bandwidth = 500e6

        crlb_var = toa_crlb(snr_linear, bandwidth)

        # Expected variance
        beta_rms = bandwidth / np.sqrt(3)
        expected = 1.0 / (8 * np.pi**2 * beta_rms**2 * snr_linear)

        self.assertAlmostEqual(crlb_var, expected)

    def test_crlb_scaling_with_snr(self):
        """Test CRLB scales inversely with SNR"""
        bandwidth = 500e6

        crlb_10db = toa_crlb(10, bandwidth)
        crlb_20db = toa_crlb(100, bandwidth)

        # Should scale by factor of 10
        ratio = crlb_10db / crlb_20db
        self.assertAlmostEqual(ratio, 10.0)

    def test_crlb_scaling_with_bandwidth(self):
        """Test CRLB scales with bandwidth squared"""
        snr_linear = 100

        crlb_250mhz = toa_crlb(snr_linear, 250e6)
        crlb_500mhz = toa_crlb(snr_linear, 500e6)

        # Should scale by factor of 4 (2^2)
        ratio = crlb_250mhz / crlb_500mhz
        self.assertAlmostEqual(ratio, 4.0, delta=0.1)

    def test_crlb_range_accuracy(self):
        """Test CRLB gives reasonable range accuracy"""
        # IEEE 802.15.4z: 500 MHz, 20 dB SNR
        snr_linear = 100
        bandwidth = 499.2e6

        crlb_var = toa_crlb(snr_linear, bandwidth)
        range_std_cm = np.sqrt(crlb_var) * 3e8 * 100

        # Should be ~1-2 cm for these parameters
        self.assertGreater(range_std_cm, 0.5)
        self.assertLess(range_std_cm, 3.0)


class TestCovarianceFromCRLB(unittest.TestCase):
    """Test covariance calculation from CRLB"""

    def test_los_covariance(self):
        """Test LOS covariance calculation"""
        snr_linear = 100
        beta_rms = 500e6 / np.sqrt(3)

        cov = cov_from_crlb(snr_linear, beta_rms, is_los=True)

        # Should equal CRLB for LOS
        crlb = toa_crlb(snr_linear, beta_rms)
        self.assertAlmostEqual(cov, crlb)

    def test_nlos_inflation(self):
        """Test NLOS variance inflation"""
        snr_linear = 100
        beta_rms = 500e6 / np.sqrt(3)

        cov_los = cov_from_crlb(snr_linear, beta_rms, is_los=True)
        cov_nlos = cov_from_crlb(snr_linear, beta_rms, is_los=False, nlos_factor=2.0)

        # NLOS should be 2x LOS
        ratio = cov_nlos / cov_los
        self.assertAlmostEqual(ratio, 2.0)

    def test_minimum_variance(self):
        """Test minimum variance floor"""
        snr_linear = 1e10  # Very high SNR
        beta_rms = 1e9  # Very wide bandwidth
        min_var = 1e-18

        cov = cov_from_crlb(snr_linear, beta_rms, min_variance=min_var)

        # Should be limited by floor
        self.assertGreaterEqual(cov, min_var)

    def test_custom_nlos_factor(self):
        """Test custom NLOS inflation factor"""
        snr_linear = 100
        beta_rms = 500e6 / np.sqrt(3)
        nlos_factor = 3.5

        cov_los = cov_from_crlb(snr_linear, beta_rms, is_los=True)
        cov_nlos = cov_from_crlb(snr_linear, beta_rms, is_los=False,
                                nlos_factor=nlos_factor)

        ratio = cov_nlos / cov_los
        self.assertAlmostEqual(ratio, nlos_factor)


class TestFeatureExtraction(unittest.TestCase):
    """Test correlation feature extraction"""

    def test_los_features(self):
        """Test features for LOS-like correlation"""
        # Create sharp peak (LOS-like)
        correlation = np.zeros(1000, dtype=complex)
        correlation[500] = 10.0
        correlation[499] = 7.0
        correlation[501] = 7.0
        correlation[498] = 3.0
        correlation[502] = 3.0

        features = extract_correlation_features(correlation, 500)

        # LOS should have small RMS width
        self.assertLess(features['rms_width'], 5)
        # High peak-to-sidelobe
        self.assertGreater(features['peak_to_sidelobe_ratio'], 2)
        # Low multipath ratio
        self.assertLess(features['multipath_ratio'], 0.3)
        # Small lead width
        self.assertLess(features['lead_width'], 10)

    def test_nlos_features(self):
        """Test features for NLOS-like correlation"""
        # Create spread peak with excess delay (NLOS-like)
        correlation = np.zeros(1000, dtype=complex)
        for i in range(30):
            correlation[520 + i] = 5.0 * np.exp(-i/10)

        peak_idx = np.argmax(np.abs(correlation))
        features = extract_correlation_features(correlation, peak_idx)

        # NLOS should have large RMS width
        self.assertGreater(features['rms_width'], 5)
        # Higher multipath ratio
        self.assertGreater(features['multipath_ratio'], 0.1)
        # Larger excess delay
        self.assertGreater(features['excess_delay'], 3)

    def test_kurtosis_calculation(self):
        """Test kurtosis feature calculation"""
        # Gaussian-like peak (kurtosis ≈ 0)
        x = np.linspace(-5, 5, 1000)
        correlation = np.exp(-x**2/2) + 0j
        peak_idx = np.argmax(np.abs(correlation))

        features = extract_correlation_features(correlation, peak_idx)

        # Gaussian has excess kurtosis ≈ 0
        self.assertAlmostEqual(features['kurtosis'], 0, delta=0.5)

    def test_early_late_ratio(self):
        """Test early-late energy ratio"""
        # Symmetric peak
        correlation = np.zeros(1000, dtype=complex)
        for i in range(-10, 11):
            correlation[500 + i] = np.exp(-abs(i)/3)

        features = extract_correlation_features(correlation, 500)

        # Should be close to 1 for symmetric peak
        self.assertAlmostEqual(features['early_late_ratio'], 1.0, delta=0.3)


class TestFeatureBasedCovariance(unittest.TestCase):
    """Test feature-based covariance scaling"""

    def test_los_no_inflation(self):
        """Test LOS features don't inflate variance"""
        base_var = 1e-18

        # Good LOS features
        features = {
            'lead_width': 5,
            'kurtosis': 2.0,
            'multipath_ratio': 0.1,
            'peak_to_sidelobe_ratio': 5.0
        }

        scaled_var = covariance_from_features(base_var, features)

        # Should have minimal inflation
        inflation = scaled_var / base_var
        self.assertLess(inflation, 1.5)

    def test_nlos_inflation(self):
        """Test NLOS features inflate variance"""
        base_var = 1e-18

        # Bad NLOS features
        features = {
            'lead_width': 20,  # Fat leading edge
            'kurtosis': -1.0,  # Spread peak
            'multipath_ratio': 0.5,  # High multipath
            'peak_to_sidelobe_ratio': 1.5  # Low peak/sidelobe
        }

        scaled_var = covariance_from_features(base_var, features)

        # Should have significant inflation
        inflation = scaled_var / base_var
        self.assertGreater(inflation, 2.0)

    def test_max_inflation_cap(self):
        """Test maximum inflation is capped"""
        base_var = 1e-18
        max_inflation = 4.0

        # Very bad features
        features = {
            'lead_width': 100,
            'kurtosis': -5.0,
            'multipath_ratio': 1.0,
            'peak_to_sidelobe_ratio': 1.0
        }

        scaled_var = covariance_from_features(base_var, features,
                                             max_inflation=max_inflation)

        # Should be capped
        inflation = scaled_var / base_var
        self.assertLessEqual(inflation, max_inflation)

    def test_partial_inflation(self):
        """Test partial inflation for mixed features"""
        base_var = 1e-18

        # Mixed features
        features = {
            'lead_width': 12,  # Slightly fat
            'kurtosis': 0.5,   # OK
            'multipath_ratio': 0.25,  # Moderate
            'peak_to_sidelobe_ratio': 3.0  # Good
        }

        scaled_var = covariance_from_features(base_var, features)

        # Should have moderate inflation
        inflation = scaled_var / base_var
        self.assertGreater(inflation, 1.2)
        self.assertLess(inflation, 2.5)


class TestEndToEndCovariance(unittest.TestCase):
    """Test end-to-end covariance pipeline"""

    def test_los_signal_processing(self):
        """Test complete pipeline for LOS signal"""
        # Generate signal
        cfg = SignalConfig(bandwidth=499.2e6, sample_rate=1e9)
        template = gen_hrp_burst(cfg, n_repeats=1)

        # Add noise for 20 dB SNR
        snr_db = 20
        snr_linear = 10**(snr_db/10)
        signal_power = np.mean(np.abs(template)**2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(template)) +
                                         1j * np.random.randn(len(template)))
        received = template + noise

        # Process signal
        correlation = matched_filter(received, template)
        toa_result = detect_toa(correlation, cfg.sample_rate)

        # Compute covariance
        beta_rms = compute_rms_bandwidth(template, cfg.sample_rate)
        measured_snr = toa_result['snr']
        cov = cov_from_crlb(measured_snr, beta_rms, is_los=True)

        # Convert to range uncertainty
        range_std_cm = np.sqrt(cov) * 3e8 * 100

        # Should be reasonable for 20 dB SNR
        self.assertGreater(range_std_cm, 0.1)
        self.assertLess(range_std_cm, 10.0)

    def test_nlos_classification_and_scaling(self):
        """Test NLOS detection and covariance scaling"""
        # Create NLOS-like correlation
        correlation = np.zeros(1000, dtype=complex)
        # Spread energy over time (multipath)
        for i in range(50):
            correlation[500 + i] = 5.0 * np.exp(-i/15) * np.exp(1j * np.random.randn())

        # Classify
        classification = classify_propagation(correlation)

        # Extract features
        peak_idx = np.argmax(np.abs(correlation))
        features = extract_correlation_features(correlation, peak_idx)

        # Base covariance
        base_cov = 1e-18

        # Scale based on classification
        if classification['type'] == 'NLOS':
            scaled_cov = covariance_from_features(base_cov, features)
        else:
            scaled_cov = base_cov

        # For NLOS, should be inflated
        if classification['type'] == 'NLOS':
            self.assertGreater(scaled_cov, base_cov * 1.5)

    def test_snr_dependent_covariance(self):
        """Test covariance scales properly with SNR"""
        cfg = SignalConfig(bandwidth=499.2e6)
        template = gen_hrp_burst(cfg, n_repeats=1)
        beta_rms = compute_rms_bandwidth(template, cfg.sample_rate)

        # Test different SNRs
        snr_values_db = [10, 15, 20, 25, 30]
        covariances = []

        for snr_db in snr_values_db:
            snr_linear = 10**(snr_db/10)
            cov = cov_from_crlb(snr_linear, beta_rms, is_los=True)
            covariances.append(cov)

        # Covariance should decrease with increasing SNR
        for i in range(1, len(covariances)):
            self.assertLess(covariances[i], covariances[i-1])

        # Check scaling ratio (10 dB = 10x SNR = 10x less variance)
        ratio_10_to_20 = covariances[0] / covariances[2]
        self.assertAlmostEqual(ratio_10_to_20, 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)