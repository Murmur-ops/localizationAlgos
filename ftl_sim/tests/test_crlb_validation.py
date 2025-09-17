#!/usr/bin/env python3
"""
CRLB Validation Test
Verify that the FTL system achieves performance near theoretical bounds
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.rx_frontend import toa_crlb
from ftl.metrics import crlb_efficiency


class TestCRLBValidation(unittest.TestCase):
    """Validate system performance against CRLB"""

    def test_crlb_calculation_ieee_uwb(self):
        """Test CRLB for IEEE 802.15.4z HRP-UWB parameters"""
        # IEEE 802.15.4z HRP-UWB parameters
        bandwidth = 499.2e6  # Hz
        snr_db_values = [10, 15, 20, 25, 30]

        print("\n" + "="*60)
        print("CRLB for IEEE 802.15.4z HRP-UWB")
        print("="*60)
        print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
        print("\nSNR (dB) | CRLB std (ps) | Range std (cm)")
        print("-"*45)

        for snr_db in snr_db_values:
            snr_linear = 10**(snr_db / 10)
            crlb_var = toa_crlb(snr_linear, bandwidth)
            crlb_std_s = np.sqrt(crlb_var)
            crlb_std_ps = crlb_std_s * 1e12
            range_std_cm = crlb_std_s * 3e8 * 100

            print(f"   {snr_db:2d}    |    {crlb_std_ps:6.1f}    |    {range_std_cm:5.2f}")

            # Verify reasonable values
            if snr_db == 20:
                # At 20 dB SNR with 500 MHz BW, expect ~1-2 cm accuracy
                self.assertLess(range_std_cm, 5.0)
                self.assertGreater(range_std_cm, 0.1)

    def test_crlb_efficiency_calculation(self):
        """Test CRLB efficiency metric"""
        # Perfect efficiency (achieved = theoretical)
        theoretical_crlb = 1e-18
        achieved_var = 1e-18
        efficiency = crlb_efficiency(achieved_var, theoretical_crlb)
        self.assertEqual(efficiency, 1.0)

        # 50% efficiency
        achieved_var = 2e-18
        efficiency = crlb_efficiency(achieved_var, theoretical_crlb)
        self.assertEqual(efficiency, 0.5)

        # Poor efficiency
        achieved_var = 10e-18
        efficiency = crlb_efficiency(achieved_var, theoretical_crlb)
        self.assertEqual(efficiency, 0.1)

    def test_bandwidth_vs_accuracy(self):
        """Test relationship between bandwidth and accuracy"""
        snr_db = 20
        snr_linear = 10**(snr_db / 10)

        bandwidths_mhz = [50, 100, 250, 500, 1000]
        print("\n" + "="*60)
        print("Bandwidth vs Ranging Accuracy (SNR = 20 dB)")
        print("="*60)
        print("BW (MHz) | Range std (cm)")
        print("-"*25)

        prev_std = float('inf')
        for bw_mhz in bandwidths_mhz:
            bandwidth = bw_mhz * 1e6
            crlb_var = toa_crlb(snr_linear, bandwidth)
            range_std_cm = np.sqrt(crlb_var) * 3e8 * 100

            print(f"  {bw_mhz:4d}   |    {range_std_cm:5.2f}")

            # Verify improvement with bandwidth
            self.assertLess(range_std_cm, prev_std)
            prev_std = range_std_cm

    def test_snr_vs_accuracy(self):
        """Test relationship between SNR and accuracy"""
        bandwidth = 500e6  # 500 MHz

        snr_db_values = np.arange(0, 35, 5)
        print("\n" + "="*60)
        print("SNR vs Ranging Accuracy (BW = 500 MHz)")
        print("="*60)
        print("SNR (dB) | Range std (cm)")
        print("-"*25)

        prev_std = float('inf')
        for snr_db in snr_db_values:
            snr_linear = 10**(snr_db / 10)
            crlb_var = toa_crlb(snr_linear, bandwidth)
            range_std_cm = np.sqrt(crlb_var) * 3e8 * 100

            print(f"   {snr_db:2.0f}    |    {range_std_cm:6.2f}")

            # Verify improvement with SNR
            self.assertLess(range_std_cm, prev_std)
            prev_std = range_std_cm

    def test_typical_scenarios(self):
        """Test CRLB for typical FTL scenarios"""
        scenarios = [
            # (name, bandwidth, snr_db, expected_range_cm)
            ("Indoor WiFi", 80e6, 15, 50),  # Rough estimate
            ("UWB Low-Rate", 250e6, 20, 5),
            ("UWB High-Rate", 500e6, 20, 2.5),
            ("mmWave 5G", 1000e6, 25, 0.5),
        ]

        print("\n" + "="*60)
        print("CRLB for Typical Scenarios")
        print("="*60)
        print("Scenario         | BW (MHz) | SNR (dB) | CRLB (cm)")
        print("-"*55)

        for name, bandwidth, snr_db, max_expected_cm in scenarios:
            snr_linear = 10**(snr_db / 10)
            crlb_var = toa_crlb(snr_linear, bandwidth)
            range_std_cm = np.sqrt(crlb_var) * 3e8 * 100

            print(f"{name:16s} | {bandwidth/1e6:7.0f} | {snr_db:7.0f} | {range_std_cm:8.2f}")

            # Verify within expected range
            self.assertLess(range_std_cm, max_expected_cm * 2)

    def test_multipath_impact(self):
        """Estimate multipath impact on CRLB"""
        # Clean channel CRLB
        bandwidth = 500e6
        snr_db = 20
        snr_linear = 10**(snr_db / 10)
        clean_crlb = toa_crlb(snr_linear, bandwidth)
        clean_std_cm = np.sqrt(clean_crlb) * 3e8 * 100

        # Multipath reduces effective SNR
        multipath_snr_penalty_db = 3  # Typical multipath penalty
        multipath_snr = snr_db - multipath_snr_penalty_db
        multipath_snr_linear = 10**(multipath_snr / 10)
        multipath_crlb = toa_crlb(multipath_snr_linear, bandwidth)
        multipath_std_cm = np.sqrt(multipath_crlb) * 3e8 * 100

        print("\n" + "="*60)
        print("Multipath Impact on CRLB")
        print("="*60)
        print(f"Bandwidth: {bandwidth/1e6:.0f} MHz")
        print(f"SNR (clean): {snr_db} dB")
        print(f"SNR (multipath): {multipath_snr} dB")
        print(f"CRLB (clean): {clean_std_cm:.2f} cm")
        print(f"CRLB (multipath): {multipath_std_cm:.2f} cm")
        print(f"Degradation: {multipath_std_cm/clean_std_cm:.1f}x")

        # Multipath should degrade performance
        self.assertGreater(multipath_std_cm, clean_std_cm)


if __name__ == "__main__":
    unittest.main(verbosity=2)