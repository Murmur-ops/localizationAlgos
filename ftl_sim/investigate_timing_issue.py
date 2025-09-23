#!/usr/bin/env python3
"""
Investigate why timing errors are larger than expected
The measurement timing errors show -0.52 ns mean with 0.59 ns RMS
This is worse than just quantization (0.29 ns RMS)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("TIMING ERROR INVESTIGATION")
print("=" * 70)

# Key finding from the analysis:
# - Measurement timing errors: -0.52 ns mean, 0.59 ns RMS
# - Quantization alone: 0.29 ns RMS
# - Final timing errors: 0.226 ns RMS (actually better!)

print("\n1. UNDERSTANDING THE MEASUREMENT TIMING ERROR")
print("-" * 40)

# The measurement shows a NEGATIVE bias of -0.52 ns
# This means measured ranges are systematically SHORT by 0.52 ns * c = 15.6 cm

print("Observed measurement bias: -0.52 ns = -15.6 cm")
print("This is systematic, not random!")

# Let's understand why
print("\n2. SOURCES OF SYSTEMATIC ERROR")
print("-" * 40)

# Source 1: Integer sample quantization
print("a) Quantization bias:")
print("   When we do int(delay), we always round DOWN")
print("   Average loss: 0.5 samples = 0.5 ns = 15 cm")

# Demonstrate
true_delays_ns = np.linspace(10, 50, 1000)  # Various true delays
quantized_delays = np.floor(true_delays_ns).astype(int)
quant_errors = quantized_delays - true_delays_ns

print(f"   Mean quantization error: {np.mean(quant_errors):.3f} ns")
print(f"   This explains our -0.52 ns bias!")

# Source 2: Signal energy distribution
print("\nb) Signal peak shift:")
print("   HRP burst has energy spread over time")
print("   Correlation peak may not align with first arrival")

# Let's test our actual quantization
print("\n3. TESTING DIFFERENT QUANTIZATION METHODS")
print("-" * 40)

methods = {
    'floor (current)': lambda x: np.floor(x),
    'round (nearest)': lambda x: np.round(x),
    'ceil (round up)': lambda x: np.ceil(x),
}

for name, method in methods.items():
    quantized = method(true_delays_ns)
    errors = quantized - true_delays_ns
    print(f"{name:20s}: mean={np.mean(errors):+.3f} ns, RMS={np.sqrt(np.mean(errors**2)):.3f} ns")

print("\n4. PROPOSED FIX")
print("-" * 40)
print("Change from int() to round() for delay calculation:")
print("  Current: delay_samples = int(true_prop_time * sig_config.sample_rate)")
print("  Better:  delay_samples = round(true_prop_time * sig_config.sample_rate)")
print("\nThis would reduce systematic bias from -0.5 ns to ~0 ns")

# Calculate impact on position accuracy
print("\n5. IMPACT ON POSITION ACCURACY")
print("-" * 40)

current_timing_rmse = 0.597  # ns from measurement
improved_timing_rmse = 0.289  # ns with proper rounding
c = 299792458.0  # m/s

current_position_impact = current_timing_rmse * 1e-9 * c * 100  # cm
improved_position_impact = improved_timing_rmse * 1e-9 * c * 100  # cm

print(f"Current timing RMSE: {current_timing_rmse:.3f} ns → {current_position_impact:.2f} cm")
print(f"With rounding:       {improved_timing_rmse:.3f} ns → {improved_position_impact:.2f} cm")
print(f"Improvement:         {current_position_impact - improved_position_impact:.2f} cm")

# But wait, the consensus actually IMPROVES the timing!
print("\n6. CONSENSUS PERFORMANCE")
print("-" * 40)
print("Interesting finding:")
print("  Measurement timing RMSE: 0.597 ns")
print("  Final timing RMSE:       0.226 ns")
print("  Consensus IMPROVES timing by 62%!")
print("\nThe consensus algorithm is doing its job well for timing.")
print("The 8cm error is mostly from POSITION estimation, not timing.")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Timing Error Investigation', fontsize=14, fontweight='bold')

# Plot 1: Quantization methods comparison
ax = axes[0, 0]
x = np.linspace(0, 1, 100)
floor_err = np.floor(x) - x
round_err = np.round(x) - x

ax.plot(x, floor_err, 'r-', label='floor() - current', linewidth=2)
ax.plot(x, round_err, 'g-', label='round() - proposed', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axhline(y=-0.5, color='r', linestyle=':', alpha=0.5, label='floor() mean')
ax.axhline(y=0, color='g', linestyle=':', alpha=0.5, label='round() mean')
ax.set_xlabel('Fractional part of delay')
ax.set_ylabel('Quantization error (samples)')
ax.set_title('Quantization Error Pattern')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Error distribution comparison
ax = axes[0, 1]
floor_errors = np.floor(true_delays_ns) - true_delays_ns
round_errors = np.round(true_delays_ns) - true_delays_ns

ax.hist(floor_errors, bins=30, alpha=0.5, label='floor()', color='red', density=True)
ax.hist(round_errors, bins=30, alpha=0.5, label='round()', color='green', density=True)
ax.set_xlabel('Quantization Error (ns)')
ax.set_ylabel('Probability Density')
ax.set_title('Error Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Position impact
ax = axes[1, 0]
timing_errors = [0.597, 0.289, 0.226]
labels = ['Measurement\n(with floor)', 'Theoretical\n(with round)', 'After Consensus']
colors = ['red', 'green', 'blue']
position_impact = [t * 0.3 for t in timing_errors]  # ns to cm at speed of light

bars = ax.bar(labels, position_impact, color=colors)
ax.set_ylabel('Position Impact (cm)')
ax.set_title('Timing Error Impact on Position')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, position_impact):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f} cm', ha='center', va='bottom')

# Plot 4: Error breakdown
ax = axes[1, 1]
components = ['Timing\n(6.78 cm)', 'Geometric\n(4.70 cm)', 'Total\n(8.25 cm)']
values = [6.78, 4.70, 8.25]
colors = ['blue', 'orange', 'red']

bars = ax.bar(components, values, color=colors)
ax.set_ylabel('RMSE (cm)')
ax.set_title('Position Error Breakdown')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    percentage = val / 8.25 * 100 if bar != bars[-1] else 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}\n({percentage:.0f}%)', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('timing_investigation.png', dpi=100)
print(f"\nInvestigation plots saved to timing_investigation.png")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("\n1. The timing offset is actually quite good:")
print("   - Final timing RMSE: 0.226 ns (6.78 cm)")
print("   - This is BETTER than single-sample quantization (0.289 ns)")
print("\n2. The measurement bias (-0.52 ns) is from using floor() instead of round()")
print("   - Easy fix: change int() to round() in delay calculation")
print("\n3. The 8.25 cm position RMSE breaks down as:")
print("   - 82% from timing (6.78 cm)")
print("   - 57% from geometry (4.70 cm)")
print("   - Note: These don't add linearly due to correlation")
print("\n4. To improve from 8 cm to sub-cm:")
print("   a) Fix quantization: use round() instead of int() → save ~9 cm bias")
print("   b) More consensus iterations or better tuning")
print("   c) Sub-sample interpolation for finer timing")
print("   d) Better initialization of unknown node positions")

print("\n💡 The system is working correctly - 0.226 ns timing is actually very good!")
print("   The 8 cm is reasonable given 1 GHz sampling (30 cm resolution).")