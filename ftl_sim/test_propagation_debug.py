#!/usr/bin/env python3
"""Debug signal propagation and ToA"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter, detect_toa

# Create signal config
sig_config = SignalConfig(
    carrier_freq=6.5e9,
    bandwidth=499.2e6,
    sample_rate=1e9,
    burst_duration=1e-6,
    prf=124.8e6
)

# Generate transmit signal
tx_signal = gen_hrp_burst(sig_config)
print(f"TX signal length: {len(tx_signal)} samples")

# Create channel for 10m distance
channel_config = ChannelConfig(environment='indoor')
channel = SalehValenzuelaChannel(channel_config)
channel_realization = channel.generate_channel_realization(
    distance_m=10.0,
    is_los=True
)

print(f"\nChannel realization:")
print(f"  Distance: 10.0 m")
print(f"  Number of taps: {channel_realization['n_taps']}")
print(f"  First tap delay: {channel_realization['delays_ns'][0]:.3f} ns")

# Expected propagation delay
c = 299792458.0
expected_delay_s = 10.0 / c
expected_delay_ns = expected_delay_s * 1e9
print(f"  Expected propagation delay: {expected_delay_ns:.3f} ns")

# Propagate signal
propagation_result = propagate_signal(
    tx_signal,
    channel_realization,
    sig_config.sample_rate,
    snr_db=50.0  # Very high SNR
)

print(f"\nPropagation result:")
print(f"  True ToA from propagate_signal: {propagation_result['true_toa']*1e9:.3f} ns")
print(f"  RX signal length: {len(propagation_result['signal'])} samples")

# Now do matched filtering
rx_signal = propagation_result['signal']
template = gen_hrp_burst(sig_config)
corr_output = matched_filter(rx_signal, template)

print(f"\nMatched filter:")
print(f"  Correlation length: {len(corr_output)} samples")
peak_idx = np.argmax(np.abs(corr_output))
print(f"  Peak index: {peak_idx}")
print(f"  Peak at time: {peak_idx/sig_config.sample_rate*1e9:.3f} ns")

# Detect ToA
toa_result = detect_toa(corr_output, sig_config.sample_rate)
print(f"\nToA detection:")
print(f"  Detected ToA: {toa_result['toa']*1e9:.3f} ns")
print(f"  Peak value: {toa_result['peak_value']:.3f}")
print(f"  SNR: {toa_result['snr_db']:.1f} dB")

# Compare
print(f"\nComparison:")
print(f"  Expected delay: {expected_delay_ns:.3f} ns")
print(f"  True ToA (from propagate_signal): {propagation_result['true_toa']*1e9:.3f} ns")
print(f"  Detected ToA: {toa_result['toa']*1e9:.3f} ns")

# The issue is likely that detect_toa is returning absolute time from correlation start,
# but we need the DELAY, not the absolute position
print(f"\nFor 'same' mode correlation:")
print(f"  Zero delay should be at index: {len(corr_output)//2}")
print(f"  Actual peak is at: {peak_idx}")
print(f"  Delay in samples: {peak_idx - len(corr_output)//2}")

# Calculate the actual delay
delay_samples = peak_idx - len(corr_output)//2
delay_ns = delay_samples / sig_config.sample_rate * 1e9
print(f"  Delay in ns: {delay_ns:.3f}")