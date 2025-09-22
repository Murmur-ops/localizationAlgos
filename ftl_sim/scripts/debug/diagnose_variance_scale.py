#!/usr/bin/env python3
"""
Diagnose the variance scaling issue
"""

import numpy as np
from ftl.measurement_covariance import compute_measurement_covariance
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter

np.random.seed(42)

sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)
ch_config = ChannelConfig(environment='indoor_office')
channel_model = SalehValenzuelaChannel(ch_config)

template = gen_hrp_burst(sig_config, n_repeats=3)

print("Checking actual variance values from measurement_covariance:")
print("="*60)

# Test case
distance = 20.0
snr_db = 15.0

# Generate channel
channel = channel_model.generate_channel_realization(distance, is_los=True)

# Propagate
result = propagate_signal(
    template, channel, sig_config.sample_rate,
    snr_db=snr_db, cfo_hz=0, sco_ppm=0, clock_bias_s=0
)

# Process
correlation = matched_filter(result['signal'], template)

# Compute covariance
meas_cov = compute_measurement_covariance(
    correlation, template, sig_config.sample_rate,
    use_feature_scaling=True
)

print(f"Test case: distance={distance}m, SNR={snr_db}dB")
print(f"  ToA variance: {meas_cov.toa_variance:.2e} s²")
print(f"  ToA std: {np.sqrt(meas_cov.toa_variance)*1e12:.1f} ps")
print(f"  Range std: {np.sqrt(meas_cov.toa_variance)*3e8*100:.2f} cm")
print(f"  Weight (1/var): {1/meas_cov.toa_variance:.2e}")
print(f"  Is LOS: {meas_cov.is_los}")

# Check the minimum variance setting
print("\nChecking minimum variance in cov_from_crlb:")
from ftl.rx_frontend import cov_from_crlb
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)

for snr_linear in [10, 100, 1000, 10000]:
    var = cov_from_crlb(snr_linear, beta_rms, is_los=True)
    print(f"  SNR={10*np.log10(snr_linear):.0f}dB: var={var:.2e} s², weight={1/var:.2e}")

print("\n" + "="*60)
print("PROBLEM: Variances are in s² causing weights ~1e18!")
print("This causes numerical instability in the solver.")
print("\nPossible solutions:")
print("1. Scale variances to more reasonable range (e.g., ns² or ps²)")
print("2. Use normalized/unitless measurements")
print("3. Add variance floor to prevent extreme weights")
print("4. Reformulate as weighted least squares with bounded weights")
