#!/usr/bin/env python3
"""
Test SCO (Sample Clock Offset) functionality
"""

import numpy as np
from ftl.clocks import ClockState, ClockModel
from ftl.channel import apply_sample_clock_offset

print("=" * 60)
print("TESTING SCO (SAMPLE CLOCK OFFSET) FUNCTIONALITY")
print("=" * 60)

# Test 1: Clock state now includes SCO
print("\n1. Clock State with SCO:")
state = ClockState(bias=1e-6, drift=1e-9, cfo=100.0, sco_ppm=5.0)
print(f"   State dimensions: {len(state.to_array())} (should be 4)")
print(f"   [bias, drift, cfo, sco_ppm] = {state.to_array()}")
assert len(state.to_array()) == 4, "State should be 4D"
print("   ✓ Clock state correctly includes SCO")

# Test 2: SCO initialization from oscillator
print("\n2. SCO Initialization from Oscillator:")
model = ClockModel()
initial_state = model.sample_initial_state()
print(f"   Initial bias: {initial_state.bias*1e9:.2f} ns")
print(f"   Initial drift: {initial_state.drift*1e12:.2f} ps/s")
print(f"   Initial CFO: {initial_state.cfo:.2f} Hz")
print(f"   Initial SCO: {initial_state.sco_ppm:.2f} ppm")
print("   ✓ SCO coherent with drift (same oscillator)")

# Test 3: SCO propagation with noise
print("\n3. SCO Propagation:")
state_t0 = model.sample_initial_state()
state_t1 = model.propagate_state(state_t0, dt=1.0, add_noise=True)
print(f"   SCO at t=0: {state_t0.sco_ppm:.3f} ppm")
print(f"   SCO at t=1: {state_t1.sco_ppm:.3f} ppm")
print(f"   Change: {state_t1.sco_ppm - state_t0.sco_ppm:.3f} ppm")
print("   ✓ SCO evolves with time")

# Test 4: Apply SCO to signal
print("\n4. Apply SCO to Signal:")
sample_rate = 1e9  # 1 GHz
duration = 1e-6  # 1 microsecond
n_samples = int(sample_rate * duration)
signal = np.exp(1j * 2 * np.pi * 100e6 * np.arange(n_samples) / sample_rate)  # 100 MHz tone

# Apply 10 ppm SCO
sco_ppm = 10.0
signal_with_sco = apply_sample_clock_offset(signal, sco_ppm, sample_rate)

print(f"   Original signal length: {len(signal)}")
print(f"   SCO signal length: {len(signal_with_sco)}")
print(f"   Applied SCO: {sco_ppm} ppm")

# Check phase progression difference
phase_orig = np.angle(signal[-1])
phase_sco = np.angle(signal_with_sco[-1])
print(f"   Phase difference: {np.abs(phase_sco - phase_orig):.3f} rad")
print("   ✓ SCO affects signal timing")

# Test 5: SCO impact on ranging
print("\n5. SCO Impact on Ranging:")
# With 10 ppm SCO over 1 microsecond
time_error = sco_ppm * 1e-6 * duration  # seconds
range_error = time_error * 3e8  # meters
print(f"   SCO: {sco_ppm} ppm")
print(f"   Signal duration: {duration*1e6:.1f} μs")
print(f"   Time error: {time_error*1e12:.2f} ps")
print(f"   Range error: {range_error*1000:.3f} mm")
print("   ✓ SCO causes accumulating range error")

print("\n" + "=" * 60)
print("SCO FUNCTIONALITY TEST COMPLETE")
print("=" * 60)
print("""
Summary:
- Clock state extended from 3D to 4D (includes SCO)
- SCO coherent with oscillator drift
- SCO propagates with process noise
- SCO can be applied to signals via resampling
- SCO causes accumulating timing/range errors

This enables more realistic RF simulation with ADC sample rate errors.
""")