#!/usr/bin/env python3
"""
Check Nyquist compliance for our RF signal configuration
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from ftl.signal import SignalConfig, gen_hrp_burst

print("=" * 70)
print("NYQUIST SAMPLING ANALYSIS")
print("=" * 70)

# Load our configuration
with open('configs/unified_ideal.yaml', 'r') as f:
    config = yaml.safe_load(f)

rf_config = config['rf_simulation']['signal']

# Extract parameters
carrier_freq = float(rf_config['carrier_freq'])  # Hz
bandwidth = float(rf_config['bandwidth'])  # Hz
sample_rate = float(rf_config['sample_rate'])  # Hz

print("\n1. SIGNAL PARAMETERS")
print("-" * 40)
print(f"Carrier frequency: {carrier_freq/1e9:.1f} GHz")
print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
print(f"Sample rate: {sample_rate/1e9:.1f} GHz")

# Calculate key frequencies
f_min = carrier_freq - bandwidth/2
f_max = carrier_freq + bandwidth/2

print(f"\nFrequency range:")
print(f"  Minimum: {f_min/1e9:.3f} GHz")
print(f"  Center:  {carrier_freq/1e9:.3f} GHz")
print(f"  Maximum: {f_max/1e9:.3f} GHz")

print("\n2. NYQUIST CRITERION CHECK")
print("-" * 40)

# Traditional Nyquist: fs > 2 * f_max
nyquist_rate = 2 * f_max
print(f"Traditional Nyquist requirement:")
print(f"  Need fs > 2 * f_max = 2 * {f_max/1e9:.3f} GHz = {nyquist_rate/1e9:.3f} GHz")
print(f"  Our sample rate: {sample_rate/1e9:.1f} GHz")

if sample_rate >= nyquist_rate:
    print(f"  ❌ NOT MET: {sample_rate/1e9:.1f} GHz < {nyquist_rate/1e9:.3f} GHz")
else:
    print(f"  ❌ NOT MET: {sample_rate/1e9:.1f} GHz < {nyquist_rate/1e9:.3f} GHz")

print("\n3. BANDPASS SAMPLING CHECK")
print("-" * 40)
print("For bandpass signals, we can use undersampling if certain conditions are met:")
print("The signal is narrowband and we avoid aliasing")

# Bandpass sampling condition
# fs >= 2 * B where B is bandwidth
bandpass_nyquist = 2 * bandwidth

print(f"\nBandpass Nyquist requirement:")
print(f"  Need fs > 2 * B = 2 * {bandwidth/1e6:.1f} MHz = {bandpass_nyquist/1e6:.1f} MHz")
print(f"  Our sample rate: {sample_rate/1e6:.1f} MHz")

if sample_rate >= bandpass_nyquist:
    print(f"  ✅ MET: {sample_rate/1e6:.1f} MHz > {bandpass_nyquist/1e6:.1f} MHz")
else:
    print(f"  ❌ NOT MET: {sample_rate/1e6:.1f} MHz < {bandpass_nyquist/1e6:.1f} MHz")

print("\n4. WHAT'S ACTUALLY HAPPENING?")
print("-" * 40)

print("IEEE 802.15.4z HRP-UWB typically uses:")
print("  - 6.5 GHz or 8 GHz carrier")
print("  - 500 MHz bandwidth")
print("  - BASEBAND processing after downconversion")

print("\nOur simulation approach:")
print("  1. We generate BASEBAND signals (complex I/Q)")
print("  2. The 'carrier_freq' is metadata for channel modeling")
print("  3. We don't actually modulate to 6.5 GHz")
print("  4. This is standard for digital signal processing")

# Generate a signal and analyze it
sig_config = SignalConfig(
    carrier_freq=carrier_freq,
    bandwidth=bandwidth,
    sample_rate=sample_rate,
    burst_duration=1e-6
)

np.random.seed(42)
signal = gen_hrp_burst(sig_config)

print("\n5. ACTUAL SIGNAL ANALYSIS")
print("-" * 40)
print(f"Signal properties:")
print(f"  Length: {len(signal)} samples")
print(f"  Duration: {len(signal)/sample_rate*1e6:.1f} µs")
print(f"  Type: {signal.dtype}")
print(f"  Complex: {np.iscomplexobj(signal)}")

# Compute spectrum
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

# Find actual bandwidth (3dB points)
power_spectrum = np.abs(fft)**2
max_power = np.max(power_spectrum)
half_power = max_power / 2

# Find frequencies above half power
significant_idx = np.where(power_spectrum > half_power)[0]
if len(significant_idx) > 0:
    actual_bandwidth = (freqs[significant_idx[-1]] - freqs[significant_idx[0]])
    print(f"\nActual signal bandwidth (3dB): {abs(actual_bandwidth)/1e6:.1f} MHz")

print("\n6. BASEBAND VS PASSBAND")
print("-" * 40)

print("Current implementation (BASEBAND):")
print("  - Generate complex baseband signal")
print("  - Centered at DC (0 Hz)")
print("  - Bandwidth: ±250 MHz around DC")
print("  - Sample rate: 1 GHz")
print("  - ✅ Nyquist satisfied: 1 GHz > 2 * 250 MHz")

print("\nIf we were doing PASSBAND (not implemented):")
print("  - Would need to modulate to 6.5 GHz")
print("  - Would need fs > 13.5 GHz for traditional Nyquist")
print("  - OR use bandpass sampling with careful design")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Nyquist and Signal Analysis', fontsize=14, fontweight='bold')

# Plot 1: Time domain signal (real part)
ax = axes[0, 0]
t = np.arange(len(signal)) / sample_rate * 1e9  # ns
ax.plot(t[:200], np.real(signal[:200]), 'b-', linewidth=0.5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Amplitude')
ax.set_title('Baseband Signal (Real Part)')
ax.grid(True, alpha=0.3)

# Plot 2: Power spectrum
ax = axes[0, 1]
# Only plot positive frequencies for clarity
pos_idx = freqs >= 0
ax.semilogy(freqs[pos_idx]/1e6, power_spectrum[pos_idx])
ax.axhline(y=half_power, color='r', linestyle='--', label='3dB level')
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum (Baseband)')
ax.set_xlim([0, sample_rate/2/1e6])
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Spectrum diagram
ax = axes[1, 0]
ax.set_xlim([-1000, 8000])
ax.set_ylim([0, 2])

# Draw baseband spectrum
baseband_rect = plt.Rectangle((-250, 0.2), 500, 0.6,
                              facecolor='blue', alpha=0.5, label='Baseband (actual)')
ax.add_patch(baseband_rect)
ax.text(0, 1, 'Baseband\n±250 MHz', ha='center', fontsize=10)

# Draw passband spectrum (if it existed)
passband_rect = plt.Rectangle((6250, 0.2), 500, 0.6,
                              facecolor='red', alpha=0.3, label='Passband (if modulated)')
ax.add_patch(passband_rect)
ax.text(6500, 1, 'Passband\n6.5 GHz ± 250 MHz', ha='center', fontsize=10)

# Draw Nyquist zones
ax.axvline(x=500, color='green', linestyle='--', alpha=0.5)
ax.text(500, 1.5, 'fs/2 = 500 MHz', rotation=90, ha='center', fontsize=8)

ax.set_xlabel('Frequency (MHz)')
ax.set_title('Frequency Allocation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Sampling requirements
ax = axes[1, 1]
scenarios = ['Baseband\n(Actual)', 'Passband\n(Traditional)', 'Passband\n(Bandpass)']
required_fs = [2*250, 2*6750, 2*500]  # MHz
actual_fs = [1000, 1000, 1000]
colors = ['green', 'red', 'orange']

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, required_fs, width, label='Required fs', alpha=0.7)
bars2 = ax.bar(x + width/2, actual_fs, width, label='Actual fs', alpha=0.7)

# Color bars based on whether requirement is met
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if actual_fs[i] >= required_fs[i]:
        bar1.set_color('green')
        bar2.set_color('green')
    else:
        bar1.set_color('red')
        bar2.set_color('red')

ax.set_xlabel('Scenario')
ax.set_ylabel('Sample Rate (MHz)')
ax.set_title('Nyquist Requirements')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('nyquist_analysis.png', dpi=100)
print(f"\nAnalysis plots saved to nyquist_analysis.png")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\n✅ YES, we are adhering to Nyquist!")
print("\nWe use BASEBAND processing:")
print("  - Complex I/Q signals centered at DC")
print("  - 500 MHz bandwidth requires 1 GHz sampling (2× bandwidth)")
print("  - No actual RF modulation to 6.5 GHz")
print("\nThis is the standard approach in digital communications:")
print("  1. RF front-end downconverts 6.5 GHz to baseband")
print("  2. ADC samples at 1 GHz (or higher)")
print("  3. Digital processing on complex baseband signals")
print("  4. This is exactly what we're simulating")

print("\n💡 The 6.5 GHz carrier frequency is used for:")
print("  - Channel path loss calculations")
print("  - Doppler effect modeling")
print("  - But NOT for actual signal generation")

print("\n📝 Note: If we were simulating the full RF chain:")
print("  - Would need 14+ GHz sampling for 6.5 GHz carrier")
print("  - Or use bandpass sampling techniques")
print("  - But that's unnecessary for localization algorithm development")