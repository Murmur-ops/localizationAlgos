#!/usr/bin/env python3
"""Simple test of RF measurement without multipath"""

import numpy as np
from ftl.clocks import ClockState
from ftl.signal import gen_hrp_burst, SignalConfig

def simple_rf_measurement(i, j, true_positions, clock_states):
    """Simplified RF measurement - no multipath, just delay"""

    # Get true distance
    pi = true_positions[i]
    pj = true_positions[j]
    true_distance = np.linalg.norm(pi - pj)

    # Signal config
    sig_config = SignalConfig(
        carrier_freq=6.5e9,
        bandwidth=499.2e6,
        sample_rate=1e9,
        burst_duration=1e-6,
        prf=124.8e6
    )

    # Generate transmit signal
    tx_signal = gen_hrp_burst(sig_config)

    # Calculate propagation delay
    c = 299792458.0
    prop_time = true_distance / c
    delay_samples = int(prop_time * sig_config.sample_rate)

    print(f"True distance: {true_distance:.3f} m")
    print(f"Propagation time: {prop_time*1e9:.3f} ns")
    print(f"Delay samples: {delay_samples}")

    # Simple delay - just shift the signal
    rx_signal = np.zeros(len(tx_signal), dtype=complex)
    if 0 < delay_samples < len(tx_signal):
        rx_signal[delay_samples:] = tx_signal[:-delay_samples]

    # Add small noise
    noise_std = 0.01
    rx_signal += noise_std * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))

    # Cross-correlation
    template = tx_signal  # Use original as template
    corr = np.correlate(rx_signal, template, mode='full')

    # Find peak
    peak_idx = np.argmax(np.abs(corr))
    zero_delay_idx = len(template) - 1
    detected_delay_samples = peak_idx - zero_delay_idx

    print(f"\nCorrelation results:")
    print(f"  Peak index: {peak_idx}")
    print(f"  Zero delay index: {zero_delay_idx}")
    print(f"  Detected delay: {detected_delay_samples} samples")

    # Convert to time
    detected_time = detected_delay_samples / sig_config.sample_rate

    # Add clock biases
    clock_bias_diff = clock_states[j].bias - clock_states[i].bias
    toa_with_clocks = detected_time + clock_bias_diff

    # Convert to distance
    range_m = toa_with_clocks * c

    print(f"\nToA calculation:")
    print(f"  Detected propagation time: {detected_time*1e9:.3f} ns")
    print(f"  Clock bias difference: {clock_bias_diff*1e9:.3f} ns")
    print(f"  Total ToA: {toa_with_clocks*1e9:.3f} ns")
    print(f"  Range: {range_m:.3f} m")

    return {
        'range_m': range_m,
        'true_range': true_distance,
        'detected_delay_samples': detected_delay_samples
    }


# Test
true_positions = np.array([
    [0, 0],
    [10, 0],
    [0, 10]
])

clock_states = {
    0: ClockState(bias=1e-9, drift=0, cfo=0),
    1: ClockState(bias=2e-9, drift=0, cfo=0),
    2: ClockState(bias=1.5e-9, drift=0, cfo=0)
}

print("=" * 60)
print("Simple RF Measurement Test")
print("=" * 60)

result = simple_rf_measurement(0, 1, true_positions, clock_states)

print(f"\nFinal result:")
print(f"  Measured range: {result['range_m']:.3f} m")
print(f"  True range: {result['true_range']:.3f} m")
print(f"  Error: {result['range_m'] - result['true_range'] - 0.3:.3f} m (should be ~0)")

# Expected: 10m + 0.3m (from 1ns clock bias diff) = 10.3m