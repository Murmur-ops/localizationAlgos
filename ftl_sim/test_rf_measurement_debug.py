#!/usr/bin/env python3
"""Debug RF measurement issue"""

import numpy as np
from ftl.clocks import ClockState
from run_unified_ftl import simulate_rf_measurement

# Test configuration
rf_config = {
    'signal': {
        'carrier_freq': '6.5e9',
        'bandwidth': '499.2e6',
        'sample_rate': '1e9',
        'burst_duration': '1e-6',
        'prf': '124.8e6',
        'snr_db': '30.0'
    },
    'channel': {
        'environment': 'indoor',
        'path_loss_exponent': '2.0',
        'shadowing_std_db': '1.0'
    },
    'simulation': {
        'max_range': '50.0',
        'los_probability': '1.0'
    }
}

# Create simple network
true_positions = np.array([
    [0, 0],
    [10, 0],
    [0, 10]
])

# Create clock states with known biases
clock_states = {
    0: ClockState(bias=1e-9, drift=0, cfo=0),  # 1 ns
    1: ClockState(bias=2e-9, drift=0, cfo=0),  # 2 ns
    2: ClockState(bias=1.5e-9, drift=0, cfo=0)  # 1.5 ns
}

print("Test RF Measurement Debug")
print("=" * 50)

# Test measurement from node 0 to node 1
meas = simulate_rf_measurement(
    0, 1,  # Nodes 0 to 1
    true_positions,
    clock_states,
    rf_config
)

print(f"\nNode 0 position: {true_positions[0]}")
print(f"Node 1 position: {true_positions[1]}")
print(f"True geometric distance: {np.linalg.norm(true_positions[1] - true_positions[0]):.3f} m")

print(f"\nNode 0 clock bias: {clock_states[0].bias * 1e9:.3f} ns")
print(f"Node 1 clock bias: {clock_states[1].bias * 1e9:.3f} ns")
print(f"Clock bias difference: {(clock_states[1].bias - clock_states[0].bias) * 1e9:.3f} ns")

c = 299792458.0  # speed of light
clock_bias_m = (clock_states[1].bias - clock_states[0].bias) * c
print(f"Clock bias in meters: {clock_bias_m:.3f} m")

if meas:
    print(f"\nMeasurement results:")
    print(f"  Measured range: {meas['range_m']:.3f} m")
    print(f"  True range: {meas['true_range']:.3f} m")
    print(f"  Variance: {meas['variance_m2']:.6f} m²")

    # Expected range
    expected = 10.0 + clock_bias_m
    print(f"\nExpected range (geometric + clock): {expected:.3f} m")
    print(f"Error: {meas['range_m'] - expected:.3f} m")
else:
    print("\nNo measurement returned!")

# Let's also check the ToA detection directly
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.rx_frontend import detect_toa

sig_config = SignalConfig(
    carrier_freq=6.5e9,
    bandwidth=499.2e6,
    sample_rate=1e9,
    burst_duration=1e-6,
    prf=124.8e6
)

# Generate a simple test signal
signal = gen_hrp_burst(sig_config)
print(f"\nSignal properties:")
print(f"  Length: {len(signal)} samples")
print(f"  Duration: {len(signal)/sig_config.sample_rate * 1e6:.1f} µs")

# Create a delayed version (10m = 33.3ns delay)
delay_samples = int(10.0 / c * sig_config.sample_rate)
print(f"  Delay for 10m: {delay_samples} samples ({10.0/c*1e9:.1f} ns)")

# Check correlation peak detection
correlation = np.correlate(signal, signal, mode='same')
peak_idx = np.argmax(np.abs(correlation))
print(f"  Autocorrelation peak at: {peak_idx} (expected: {len(signal)//2})")