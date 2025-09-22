#!/usr/bin/env python3
"""
Check if CRLB variances are realistic
"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.rx_frontend import toa_crlb

# Signal config
sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)

# Generate signal and compute RMS bandwidth
template = gen_hrp_burst(sig_config, n_repeats=3)
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)

print(f"Signal bandwidth: {sig_config.bandwidth/1e6:.1f} MHz")
print(f"RMS bandwidth: {beta_rms/1e6:.1f} MHz")
print(f"Bandwidth efficiency: {beta_rms/sig_config.bandwidth:.3f}")

print("\nCRLB-based range accuracy:")
for snr_db in [5, 10, 15, 20, 25, 30]:
    snr_linear = 10**(snr_db/10)
    var_toa = toa_crlb(snr_linear, beta_rms)
    std_toa = np.sqrt(var_toa)
    range_std_m = std_toa * 3e8
    
    print(f"  SNR = {snr_db:2d} dB: σ_ToA = {std_toa*1e12:6.1f} ps, σ_range = {range_std_m*100:6.2f} cm")

print("\nWith NLOS inflation (2x):")
for snr_db in [5, 10, 15, 20]:
    snr_linear = 10**(snr_db/10)
    var_toa = toa_crlb(snr_linear, beta_rms) * 2  # NLOS inflation
    std_toa = np.sqrt(var_toa)
    range_std_m = std_toa * 3e8
    
    print(f"  SNR = {snr_db:2d} dB: σ_range = {range_std_m*100:6.2f} cm")
