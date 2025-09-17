#!/usr/bin/env python3
"""
Quick Position Estimation Plot
Generate a figure showing estimated vs actual positions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Import FTL modules
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType
from ftl.clocks import ClockState
from ftl.robust import RobustConfig
from ftl.solver import FactorGraph
from ftl.init import trilateration
from ftl.metrics import position_rmse, position_mae

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_NODES = 25  # 5x5 grid
AREA_SIZE = 100.0  # meters
N_ANCHORS = 4
SNR_DB = 20.0

print("Generating FTL Position Estimation Demo...")
print("=" * 60)

# Step 1: Generate node geometry
print("\n1. Setting up geometry...")
nodes = place_grid_nodes(int(np.sqrt(N_NODES)), AREA_SIZE, jitter_std=2.0)
anchors = place_anchors(nodes, N_ANCHORS, AREA_SIZE, PlacementType.CORNERS)
anchor_indices = [a.node_id for a in anchors]
print(f"   {N_NODES} nodes in {AREA_SIZE}x{AREA_SIZE}m area")
print(f"   {N_ANCHORS} anchors at corners")

# Extract true positions
true_positions = np.array([[n.x, n.y] for n in nodes])
# Use corner nodes as anchors (0, 4, 20, 24 for 5x5 grid)
anchor_indices = [0, 4, 20, 24]
anchor_positions = true_positions[anchor_indices]

# Step 2: Initialize clock states
print("\n2. Initializing clocks...")
true_states = {}
for i in range(N_NODES):
    if i in anchor_indices:
        # Anchors have good clocks
        true_states[i] = np.array([
            true_positions[i, 0],
            true_positions[i, 1],
            np.random.randn() * 1e-9,  # 1 ns bias
            np.random.randn() * 1e-12,  # 1 ps/s drift
            np.random.randn() * 0.1     # 0.1 Hz CFO
        ])
    else:
        # Unknown nodes have worse clocks
        true_states[i] = np.array([
            true_positions[i, 0],
            true_positions[i, 1],
            np.random.randn() * 1e-6,  # 1 μs bias
            np.random.randn() * 1e-9,   # 1 ns/s drift
            np.random.randn() * 10.0    # 10 Hz CFO
        ])

# Step 3: Generate simulated measurements
print("\n3. Simulating measurements...")
measurements = []
measurement_pairs = []

# Generate measurements between all nodes within range
MAX_RANGE = 150.0
for i in range(N_NODES):
    for j in range(i+1, N_NODES):
        distance = np.linalg.norm(true_positions[i] - true_positions[j])
        if distance < MAX_RANGE:
            # Simulate ToA measurement with noise
            true_toa = distance / 3e8
            bias_diff = true_states[j][2] - true_states[i][2]

            # Add measurement noise (based on CRLB)
            # For 500 MHz BW at 20 dB SNR: ~40 ps std
            toa_noise = np.random.randn() * 40e-12
            measured_toa = true_toa + bias_diff + toa_noise

            # Also create TWR measurement (bias-free)
            twr_noise = np.random.randn() * 0.01  # 1 cm std
            measured_twr = distance + twr_noise

            measurements.append({
                'i': i, 'j': j,
                'toa': measured_toa,
                'twr': measured_twr,
                'cfo': true_states[j][4] - true_states[i][4] + np.random.randn()
            })
            measurement_pairs.append((i, j))

print(f"   Generated {len(measurements)} measurements")

# Step 4: Initialize positions using trilateration
print("\n4. Initializing positions...")
initial_positions = np.copy(true_positions)
for i in range(N_NODES):
    if i not in anchor_indices:
        # Use trilateration from anchors
        distances = []
        for anc_idx in anchor_indices:
            true_dist = np.linalg.norm(true_positions[i] - true_positions[anc_idx])
            noisy_dist = true_dist + np.random.randn() * 0.5  # 50 cm initial error
            distances.append(noisy_dist)

        if len(distances) >= 3:
            try:
                initial_positions[i] = trilateration(anchor_positions[:3], np.array(distances[:3]))
            except:
                # Fallback to random near center
                initial_positions[i] = np.array([AREA_SIZE/2, AREA_SIZE/2]) + np.random.randn(2) * 10

# Step 5: Build and optimize factor graph
print("\n5. Running factor graph optimization...")
robust_config = RobustConfig(use_huber=True, huber_delta=1.0)
graph = FactorGraph(robust_config)

# Add nodes
for i in range(N_NODES):
    initial_state = np.zeros(5)
    initial_state[:2] = initial_positions[i]
    initial_state[2:] = true_states[i][2:]  # Use true clock params for simplicity
    graph.add_node(i, initial_state, is_anchor=(i in anchor_indices))

# Add factors
for meas in measurements:
    graph.add_toa_factor(meas['i'], meas['j'], meas['toa'], 1e-18)
    graph.add_twr_factor(meas['i'], meas['j'], meas['twr'], 0.01)
    graph.add_cfo_factor(meas['i'], meas['j'], meas['cfo'], 1.0)

# Optimize
result = graph.optimize(max_iterations=50, verbose=False)
print(f"   Converged: {result.converged} in {result.iterations} iterations")

# Extract estimated positions
estimated_positions = np.array([result.estimates[i][:2] for i in range(N_NODES)])

# Step 6: Calculate metrics
print("\n6. Performance metrics:")
rmse = position_rmse(estimated_positions, true_positions, True, anchor_indices)
mae = position_mae(estimated_positions, true_positions, True, anchor_indices)
print(f"   Position RMSE: {rmse:.3f} m")
print(f"   Position MAE: {mae:.3f} m")

# Step 7: Create visualization
print("\n7. Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Full view
ax1.set_title('FTL Position Estimation Results', fontsize=14, fontweight='bold')
ax1.set_xlabel('X Position (m)', fontsize=12)
ax1.set_ylabel('Y Position (m)', fontsize=12)

# Plot measurement connections (faint)
for i, j in measurement_pairs[:50]:  # Limit to 50 for clarity
    ax1.plot([true_positions[i, 0], true_positions[j, 0]],
            [true_positions[i, 1], true_positions[j, 1]],
            'gray', alpha=0.1, linewidth=0.5, zorder=1)

# Plot error vectors
for i in range(N_NODES):
    if i not in anchor_indices:
        ax1.arrow(true_positions[i, 0], true_positions[i, 1],
                 estimated_positions[i, 0] - true_positions[i, 0],
                 estimated_positions[i, 1] - true_positions[i, 1],
                 head_width=0.8, head_length=0.5, fc='red', ec='red',
                 alpha=0.6, zorder=3)

# Plot nodes
# True positions
ax1.scatter(true_positions[:, 0], true_positions[:, 1],
           c='blue', marker='o', s=100, label='True Position',
           edgecolors='darkblue', linewidth=2, zorder=4)

# Estimated positions
unknown_mask = np.ones(N_NODES, dtype=bool)
unknown_mask[anchor_indices] = False
ax1.scatter(estimated_positions[unknown_mask, 0], estimated_positions[unknown_mask, 1],
           c='red', marker='x', s=150, label='Estimated',
           linewidth=3, zorder=5)

# Anchors (special markers)
ax1.scatter(true_positions[anchor_indices, 0], true_positions[anchor_indices, 1],
           c='green', marker='^', s=200, label='Anchors',
           edgecolors='darkgreen', linewidth=2, zorder=6)

ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(-5, AREA_SIZE + 5)
ax1.set_ylim(-5, AREA_SIZE + 5)

# Right plot: Error distribution
ax2.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Position Error (m)', fontsize=12)
ax2.set_ylabel('Number of Nodes', fontsize=12)

# Calculate errors for non-anchor nodes
errors = []
for i in range(N_NODES):
    if i not in anchor_indices:
        error = np.linalg.norm(estimated_positions[i] - true_positions[i])
        errors.append(error)

# Create histogram
n, bins, patches = ax2.hist(errors, bins=15, color='skyblue', edgecolor='navy', alpha=0.7)

# Add statistics lines
ax2.axvline(rmse, color='red', linestyle='--', linewidth=2, label=f'RMSE: {rmse:.3f}m')
ax2.axvline(mae, color='orange', linestyle='--', linewidth=2, label=f'MAE: {mae:.3f}m')

# Add theoretical CRLB line
# For 500 MHz, 20 dB SNR: ~1.17 cm ranging accuracy
theoretical_ranging_std = 0.0117  # meters
# Position error is roughly sqrt(2) * ranging error for 2D
theoretical_position_std = theoretical_ranging_std * np.sqrt(2)
ax2.axvline(theoretical_position_std, color='green', linestyle=':', linewidth=2,
           label=f'CRLB: {theoretical_position_std:.3f}m')

ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

# Add text box with summary statistics
textstr = f'Nodes: {N_NODES}\n'
textstr += f'Anchors: {N_ANCHORS}\n'
textstr += f'Measurements: {len(measurements)}\n'
textstr += f'SNR: {SNR_DB} dB\n'
textstr += f'Max Error: {max(errors):.3f}m\n'
textstr += f'Min Error: {min(errors):.3f}m'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.65, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.suptitle('FTL (Frequency-Time-Localization) Joint Estimation',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_path = 'ftl_position_estimation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved to: {output_path}")

# Don't block on show
# plt.show()

print("\n" + "="*60)
print("Demo complete!")
print(f"Final RMSE: {rmse:.3f} m (unknown nodes only)")
print(f"Theoretical CRLB: {theoretical_position_std:.3f} m")
print(f"Efficiency: {(theoretical_position_std/rmse)*100:.1f}%")