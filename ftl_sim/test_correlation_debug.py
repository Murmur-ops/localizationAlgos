#!/usr/bin/env python3
"""Debug correlation-based ToA detection"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig

# Create signal config
sig_config = SignalConfig(
    carrier_freq=6.5e9,
    bandwidth=499.2e6,
    sample_rate=1e9,
    burst_duration=1e-6,
    prf=124.8e6
)

# Generate template signal
template = gen_hrp_burst(sig_config)
print(f"Template length: {len(template)} samples")

# Create a delayed version for 10m (33.356 ns)
c = 299792458.0
distance = 10.0
prop_time = distance / c
delay_samples = int(prop_time * sig_config.sample_rate)

print(f"\nFor {distance}m distance:")
print(f"  Propagation time: {prop_time*1e9:.3f} ns")
print(f"  Delay in samples: {delay_samples}")

# Method 1: Simple shift
rx_signal = np.zeros(len(template))
if delay_samples < len(template):
    rx_signal[delay_samples:] = template[:-delay_samples]

# Cross-correlation with 'full' mode
corr_full = np.correlate(rx_signal, template, mode='full')
peak_idx_full = np.argmax(np.abs(corr_full))

print(f"\nMethod 1 - Simple shift + correlate (full mode):")
print(f"  Correlation length: {len(corr_full)}")
print(f"  Peak index: {peak_idx_full}")
print(f"  Zero delay index: {len(template) - 1}")
print(f"  Detected delay: {peak_idx_full - (len(template) - 1)} samples")
detected_delay_1 = (peak_idx_full - (len(template) - 1)) / sig_config.sample_rate
print(f"  Detected time: {detected_delay_1*1e9:.3f} ns")
print(f"  Error: {(detected_delay_1 - prop_time)*1e9:.3f} ns")

# Method 2: Proper signal with padding
rx_signal_2 = np.zeros(len(template) + delay_samples)
rx_signal_2[delay_samples:delay_samples+len(template)] = template

# Need to truncate or pad template for correlation
if len(rx_signal_2) > len(template):
    # Pad template with zeros at the end
    template_padded = np.pad(template, (0, len(rx_signal_2) - len(template)))
else:
    template_padded = template

# Cross-correlation
corr_2 = np.correlate(rx_signal_2[:len(template)], template, mode='full')
peak_idx_2 = np.argmax(np.abs(corr_2))

print(f"\nMethod 2 - Padded signal:")
print(f"  RX signal length: {len(rx_signal_2)}")
print(f"  Correlation length: {len(corr_2)}")
print(f"  Peak index: {peak_idx_2}")
detected_delay_2 = (peak_idx_2 - (len(template) - 1)) / sig_config.sample_rate
print(f"  Detected delay: {detected_delay_2*1e9:.3f} ns")

# Method 3: What our current code does
# Shift within same length
rx_signal_3 = np.zeros_like(template)
if delay_samples > 0 and delay_samples < len(template):
    rx_signal_3[delay_samples:] = template[:-delay_samples]

# Correlate
corr_3 = np.correlate(rx_signal_3, template, mode='full')
peak_idx_3 = np.argmax(np.abs(corr_3))
zero_delay_idx = len(template) - 1
delay_in_samples = peak_idx_3 - zero_delay_idx

print(f"\nMethod 3 - Current implementation:")
print(f"  Peak index: {peak_idx_3}")
print(f"  Zero delay index: {zero_delay_idx}")
print(f"  Detected delay samples: {delay_in_samples}")
print(f"  Detected time: {delay_in_samples/sig_config.sample_rate*1e9:.3f} ns")
print(f"  Expected: {prop_time*1e9:.3f} ns")

# The issue is that we're losing signal energy when we shift!
# Let's check energy
print(f"\nSignal energies:")
print(f"  Original template: {np.sum(np.abs(template)**2):.3f}")
print(f"  Shifted signal: {np.sum(np.abs(rx_signal_3)**2):.3f}")
print(f"  Energy ratio: {np.sum(np.abs(rx_signal_3)**2) / np.sum(np.abs(template)**2):.3f}")