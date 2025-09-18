#!/usr/bin/env python3
"""
Check what variances we actually get from CRLB with realistic UWB
"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.rx_frontend import toa_crlb

print("REALISTIC UWB VARIANCE ANALYSIS")
print("="*70)

# IEEE 802.15.4z HRP-UWB configuration
sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)
template = gen_hrp_burst(sig_config, n_repeats=3)
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)

print(f"\nUWB Signal Parameters:")
print(f"  Bandwidth: {sig_config.bandwidth/1e6:.1f} MHz")
print(f"  RMS Bandwidth: {beta_rms/1e6:.1f} MHz")
print(f"  Bandwidth efficiency: {beta_rms/sig_config.bandwidth:.3f}")

print(f"\n1. CRLB-BASED VARIANCES (what we computed):")
print("-"*50)
print("SNR (dB) | σ_ToA (s)    | σ_range (m)  | Variance (s²)")
print("-"*50)

for snr_db in [5, 10, 15, 20, 25, 30]:
    snr_linear = 10**(snr_db/10)
    var_toa = toa_crlb(snr_linear, beta_rms)
    std_toa = np.sqrt(var_toa)
    std_range = std_toa * 3e8
    
    print(f"  {snr_db:2d}    | {std_toa:.2e} | {std_range:8.4f} | {var_toa:.2e}")

print(f"\n2. ACTUAL UWB PERFORMANCE (from literature):")
print("-"*50)
print("Real-world UWB ranging accuracy:")
print("  - LOS conditions: 10-30 cm typical")
print("  - NLOS conditions: 50-200 cm typical")
print("  - Best case (high SNR, LOS): ~5 cm")
print("  - IEEE 802.15.4z spec: ~10 cm target")

print(f"\n3. VARIANCE FLOOR ANALYSIS:")
print("-"*50)

test_accuracies_m = [0.01, 0.05, 0.10, 0.30, 1.0]
print("Range accuracy | ToA std (s)   | Variance (s²)  | Weight")
print("-"*60)

for acc in test_accuracies_m:
    toa_std = acc / 3e8
    toa_var = toa_std**2
    weight = 1.0 / toa_var
    print(f"  {acc*100:5.1f} cm   | {toa_std:.2e} | {toa_var:.2e} | {weight:.2e}")

print(f"\n4. PROBLEM WITH 1e-12 FLOOR:")
print("-"*50)
floor_var = 1e-12
floor_std = np.sqrt(floor_var)
floor_range = floor_std * 3e8

print(f"Variance floor: {floor_var:.2e} s²")
print(f"Equivalent to:  {floor_std:.2e} s = {floor_std*1e6:.1f} µs")
print(f"Range accuracy: {floor_range:.1f} m")
print(f"\nThis means we're limiting precision to {floor_range:.1f}m!")
print(f"That's {floor_range*100:.0f}× worse than actual UWB capability!")

print(f"\n5. WHAT WE SHOULD DO:")
print("-"*50)
print("Option 1: Work in different units")
print("  - Use nanoseconds instead of seconds")
print("  - 10cm → 0.33 ns → variance = 0.11 ns²")
print("  - This avoids tiny numbers")
print("")
print("Option 2: Reformulate the problem")
print("  - Scale measurements and Jacobians")
print("  - Use relative weighting")
print("")
print("Option 3: Accept that our solver can't handle realistic precision")
print("  - Be honest about limitations")
print("  - Test with realistic noise levels (10-30cm)")

print("\n" + "="*70)
print("BOTTOM LINE:")
print("="*70)
print("1. Real UWB has 5-30cm accuracy (1.7e-10 to 1e-9 s)")
print("2. Variances are 2.8e-20 to 1e-18 s² (not 1e-12!)")
print("3. Our 1e-12 floor throws away 6-8 orders of magnitude of precision")
print("4. We're not testing realistic UWB performance")
