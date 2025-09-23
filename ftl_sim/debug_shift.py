#!/usr/bin/env python3
"""Debug signal shifting issue"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig

sig_config = SignalConfig(
    sample_rate=1e9,
    burst_duration=1e-6
)

# Generate signal
signal = gen_hrp_burst(sig_config)
print(f"Original signal: length={len(signal)}, energy={np.sum(np.abs(signal)**2):.1f}")

# Plot where the signal energy is
energy_per_sample = np.abs(signal)**2
cumulative_energy = np.cumsum(energy_per_sample)
total_energy = cumulative_energy[-1]

# Find where 90% of energy is
idx_90 = np.where(cumulative_energy > 0.9 * total_energy)[0][0]
idx_50 = np.where(cumulative_energy > 0.5 * total_energy)[0][0]
idx_10 = np.where(cumulative_energy > 0.1 * total_energy)[0][0]

print(f"Energy distribution:")
print(f"  10% of energy by sample: {idx_10}")
print(f"  50% of energy by sample: {idx_50}")
print(f"  90% of energy by sample: {idx_90}")

# Now shift by 33 samples
delay = 33
shifted = np.zeros_like(signal)
shifted[delay:] = signal[:-delay]

print(f"\nShifted signal (delay={delay}):")
print(f"  Energy: {np.sum(np.abs(shifted)**2):.1f}")
print(f"  Energy ratio: {np.sum(np.abs(shifted)**2) / np.sum(np.abs(signal)**2):.3f}")

# Where is the remaining energy?
remaining_samples = len(signal) - delay
print(f"  Remaining samples: {remaining_samples} out of {len(signal)}")
print(f"  Lost first {delay} samples which had {np.sum(np.abs(signal[:delay])**2):.1f} energy")

# The problem is clear - we're losing the beginning of the signal!
# The HRP burst has most of its energy at the beginning

# Check correlation
corr = np.correlate(shifted, signal, mode='full')
peak_idx = np.argmax(np.abs(corr))
zero_idx = len(signal) - 1
detected_delay = peak_idx - zero_idx

print(f"\nCorrelation:")
print(f"  Peak at: {peak_idx}")
print(f"  Zero at: {zero_idx}")
print(f"  Detected delay: {detected_delay} (expected: {delay})")