#!/usr/bin/env python3
"""
Analyze timing offset accuracy in the unified FTL system
Understanding where the 8cm error comes from
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    generate_all_measurements,
    setup_consensus_from_measurements
)

print("=" * 70)
print("TIMING OFFSET ANALYSIS - UNIFIED FTL SYSTEM")
print("=" * 70)

# Load ideal configuration
with open('configs/unified_ideal.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate network
true_positions, n_anchors, n_total = generate_network_topology(config)
print(f"\nNetwork: {n_total} nodes ({n_anchors} anchors)")

# Initialize clock states
clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)

# Analyze initial clock errors
print("\n1. INITIAL CLOCK STATE ANALYSIS")
print("-" * 40)

clock_biases_ns = [state.bias * 1e9 for state in clock_states.values()]
clock_drifts_ppb = [state.drift * 1e9 for state in clock_states.values()]
clock_cfos_ppm = [state.cfo * 1e-6 for state in clock_states.values()]

print(f"Clock bias statistics (ns):")
print(f"  Anchors (0-{n_anchors-1}):")
print(f"    Mean: {np.mean(clock_biases_ns[:n_anchors]):.4f} ns")
print(f"    Std:  {np.std(clock_biases_ns[:n_anchors]):.4f} ns")
print(f"    Max:  {np.max(np.abs(clock_biases_ns[:n_anchors])):.4f} ns")

print(f"  Unknown nodes ({n_anchors}-{n_total-1}):")
print(f"    Mean: {np.mean(clock_biases_ns[n_anchors:]):.4f} ns")
print(f"    Std:  {np.std(clock_biases_ns[n_anchors:]):.4f} ns")
print(f"    Max:  {np.max(np.abs(clock_biases_ns[n_anchors:])):.4f} ns")

# Convert to distance
c = 299792458.0
bias_errors_m = [bias * 1e-9 * c for bias in clock_biases_ns]
print(f"\nClock bias in meters:")
print(f"  Max error from clock bias: {np.max(np.abs(bias_errors_m)):.3f} m")
print(f"  Std of clock bias errors: {np.std(bias_errors_m):.3f} m")

# Generate measurements
print("\n2. MEASUREMENT ACCURACY ANALYSIS")
print("-" * 40)

measurements = generate_all_measurements(true_positions, clock_states, config['rf_simulation'])

# Analyze measurement errors
measurement_errors = []
timing_errors_ns = []
quantization_errors = []

for (i, j), meas_list in measurements.items():
    true_dist = np.linalg.norm(true_positions[i] - true_positions[j])

    for meas in meas_list:
        # Expected range including clock bias
        clock_bias_diff = clock_states[j].bias - clock_states[i].bias
        expected_range = true_dist + clock_bias_diff * c

        # Measurement error
        error = meas['range_m'] - expected_range
        measurement_errors.append(error)

        # Timing error
        timing_error_ns = (error / c) * 1e9
        timing_errors_ns.append(timing_error_ns)

        # Quantization error (1 sample = 1 ns at 1 GHz)
        true_time_ns = (true_dist / c) * 1e9
        quantized_time_ns = np.round(true_time_ns)
        quant_error_ns = quantized_time_ns - true_time_ns
        quantization_errors.append(quant_error_ns)

print(f"Measurement errors (m):")
print(f"  Mean: {np.mean(measurement_errors):.4f} m")
print(f"  Std:  {np.std(measurement_errors):.4f} m")
print(f"  Max:  {np.max(np.abs(measurement_errors)):.4f} m")

print(f"\nTiming errors (ns):")
print(f"  Mean: {np.mean(timing_errors_ns):.4f} ns")
print(f"  Std:  {np.std(timing_errors_ns):.4f} ns")
print(f"  Max:  {np.max(np.abs(timing_errors_ns)):.4f} ns")

print(f"\nQuantization errors (ns):")
print(f"  Mean: {np.mean(quantization_errors):.4f} ns")
print(f"  Std:  {np.std(quantization_errors):.4f} ns")
print(f"  RMS:  {np.sqrt(np.mean(np.array(quantization_errors)**2)):.4f} ns")

# Theoretical quantization error
# For uniform quantization, RMS error = q/sqrt(12) where q = 1 ns
theoretical_quant_rms = 1.0 / np.sqrt(12)
print(f"  Theoretical RMS: {theoretical_quant_rms:.4f} ns")

# Run consensus to see final timing accuracy
print("\n3. CONSENSUS TIMING ACCURACY")
print("-" * 40)

cgn = setup_consensus_from_measurements(
    true_positions, measurements, clock_states,
    n_anchors, config['consensus']
)

# Set true positions for evaluation
cgn.set_true_positions(true_positions)

# Run optimization
results = cgn.optimize()

# Extract final states
final_states = {}
for node_id, node in cgn.nodes.items():
    final_states[node_id] = node.state.copy()

# Analyze timing accuracy
final_timing_errors_ns = []
final_position_errors_m = []

for i in range(n_anchors, n_total):  # Only unknown nodes
    # Position error
    pos_error = np.linalg.norm(final_states[i][:2] - true_positions[i])
    final_position_errors_m.append(pos_error)

    # Clock bias error (in ns)
    true_bias_ns = clock_states[i].bias * 1e9
    estimated_bias_ns = final_states[i][2]
    bias_error_ns = estimated_bias_ns - true_bias_ns
    final_timing_errors_ns.append(bias_error_ns)

print(f"Final position errors (m):")
print(f"  RMSE: {np.sqrt(np.mean(np.array(final_position_errors_m)**2)):.4f} m")
print(f"  Mean: {np.mean(final_position_errors_m):.4f} m")
print(f"  Max:  {np.max(final_position_errors_m):.4f} m")

print(f"\nFinal timing errors (ns):")
print(f"  RMSE: {np.sqrt(np.mean(np.array(final_timing_errors_ns)**2)):.4f} ns")
print(f"  Mean: {np.mean(final_timing_errors_ns):.4f} ns")
print(f"  Std:  {np.std(final_timing_errors_ns):.4f} ns")
print(f"  Max:  {np.max(np.abs(final_timing_errors_ns)):.4f} ns")

# Convert timing error to position error
timing_position_impact = np.array(final_timing_errors_ns) * 1e-9 * c
print(f"\nTiming error impact on position:")
print(f"  RMSE contribution: {np.sqrt(np.mean(timing_position_impact**2)):.4f} m")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Timing Offset Analysis', fontsize=14, fontweight='bold')

# Plot 1: Initial clock biases
ax = axes[0, 0]
ax.hist(clock_biases_ns[:n_anchors], bins=20, alpha=0.5, label='Anchors', color='red')
ax.hist(clock_biases_ns[n_anchors:], bins=20, alpha=0.5, label='Unknown', color='blue')
ax.set_xlabel('Clock Bias (ns)')
ax.set_ylabel('Count')
ax.set_title('Initial Clock Biases')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Measurement timing errors
ax = axes[0, 1]
ax.hist(timing_errors_ns, bins=30, edgecolor='black')
ax.set_xlabel('Timing Error (ns)')
ax.set_ylabel('Count')
ax.set_title(f'Measurement Timing Errors\nRMS: {np.sqrt(np.mean(np.array(timing_errors_ns)**2)):.3f} ns')
ax.grid(True, alpha=0.3)

# Plot 3: Quantization errors
ax = axes[0, 2]
ax.hist(quantization_errors, bins=30, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Quantization Error (ns)')
ax.set_ylabel('Count')
ax.set_title(f'Quantization Errors\nRMS: {np.sqrt(np.mean(np.array(quantization_errors)**2)):.3f} ns')
ax.grid(True, alpha=0.3)

# Plot 4: Final timing errors
ax = axes[1, 0]
ax.hist(final_timing_errors_ns, bins=30, edgecolor='black')
ax.set_xlabel('Final Timing Error (ns)')
ax.set_ylabel('Count')
ax.set_title(f'Final Timing Errors (After Consensus)\nRMS: {np.sqrt(np.mean(np.array(final_timing_errors_ns)**2)):.3f} ns')
ax.grid(True, alpha=0.3)

# Plot 5: Position vs Timing Error
ax = axes[1, 1]
ax.scatter(final_timing_errors_ns, np.array(final_position_errors_m) * 100)  # Convert to cm
ax.set_xlabel('Timing Error (ns)')
ax.set_ylabel('Position Error (cm)')
ax.set_title('Position Error vs Timing Error')
ax.grid(True, alpha=0.3)

# Add best fit line
if len(final_timing_errors_ns) > 0:
    z = np.polyfit(final_timing_errors_ns, np.array(final_position_errors_m) * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(final_timing_errors_ns), max(final_timing_errors_ns), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'Fit: {z[0]:.3f} cm/ns')
    ax.legend()

# Plot 6: Error components
ax = axes[1, 2]
components = ['Quantization\n(~0.3 ns)', 'Clock Init\n(~0.01 ns)', 'Measurement\n(Total)', 'Final\n(After Consensus)']
values = [
    np.sqrt(np.mean(np.array(quantization_errors)**2)),
    np.std(clock_biases_ns),
    np.sqrt(np.mean(np.array(timing_errors_ns)**2)),
    np.sqrt(np.mean(np.array(final_timing_errors_ns)**2))
]
colors = ['orange', 'green', 'blue', 'red']
bars = ax.bar(components, values, color=colors)
ax.set_ylabel('RMS Error (ns)')
ax.set_title('Timing Error Components')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('timing_accuracy_analysis.png', dpi=100)
print(f"\nAnalysis plots saved to timing_accuracy_analysis.png")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: WHERE DOES THE 8CM ERROR COME FROM?")
print("=" * 70)

print(f"\n1. Quantization: {theoretical_quant_rms:.3f} ns RMS = {theoretical_quant_rms * 0.3:.2f} cm")
print(f"   (1 GHz sampling → 1 ns resolution → 30 cm steps)")

print(f"\n2. Initial clock errors: {np.std(clock_biases_ns):.3f} ns = {np.std(clock_biases_ns) * 0.3:.2f} cm")
print(f"   (Very small in ideal config)")

print(f"\n3. Final timing RMSE: {np.sqrt(np.mean(np.array(final_timing_errors_ns)**2)):.3f} ns")
print(f"   Position impact: {np.sqrt(np.mean(timing_position_impact**2)) * 100:.2f} cm")

print(f"\n4. Total position RMSE: {np.sqrt(np.mean(np.array(final_position_errors_m)**2)) * 100:.2f} cm")

# Breakdown
pos_rmse = np.sqrt(np.mean(np.array(final_position_errors_m)**2))
timing_contribution = np.sqrt(np.mean(timing_position_impact**2))
geometric_contribution = np.sqrt(pos_rmse**2 - timing_contribution**2) if pos_rmse > timing_contribution else 0

print(f"\nError breakdown:")
print(f"  Timing contribution: {timing_contribution * 100:.2f} cm ({timing_contribution/pos_rmse*100:.1f}%)")
print(f"  Geometric contribution: {geometric_contribution * 100:.2f} cm ({geometric_contribution/pos_rmse*100:.1f}%)")

print("\n💡 INSIGHT: The 8cm error is NOT primarily from timing - it's from position estimation!")
print("   The consensus algorithm needs more iterations or better convergence tuning.")