#!/usr/bin/env python3
"""
Generate visualization of 30-node consensus performance
Shows actual vs estimated positions under ideal conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

# Set up the network
np.random.seed(42)
n_nodes = 30
n_anchors = 5
area_size = 50
comm_range = 25
range_noise_std = 0.01

# Anchors: corners + center
anchor_positions = np.array([
    [0, 0],
    [area_size, 0],
    [area_size, area_size],
    [0, area_size],
    [area_size/2, area_size/2]
])

# Unknowns in grid
n_unknowns = n_nodes - n_anchors
grid_size = int(np.ceil(np.sqrt(n_unknowns)))
x_pos = np.linspace(5, area_size-5, grid_size)
y_pos = np.linspace(5, area_size-5, grid_size)

unknown_positions = []
for x in x_pos:
    for y in y_pos:
        unknown_positions.append([x, y])
        if len(unknown_positions) >= n_unknowns:
            break
    if len(unknown_positions) >= n_unknowns:
        break

unknown_positions = np.array(unknown_positions[:n_unknowns])
true_positions = np.vstack([anchor_positions, unknown_positions])

# Run consensus with optimal parameters
config = ConsensusGNConfig(
    max_iterations=500,
    consensus_gain=0.05,  # Optimal for accuracy
    step_size=0.3,
    gradient_tol=1e-5,
    step_tol=1e-6,
    verbose=False
)

cgn = ConsensusGaussNewton(config)

# Add nodes
initial_positions = []
for i in range(n_nodes):
    state = np.zeros(5)
    if i < n_anchors:
        state[:2] = true_positions[i]
        initial_positions.append(true_positions[i].copy())
        cgn.add_node(i, state, is_anchor=True)
    else:
        # Good initial guess
        initial_guess = true_positions[i] + np.random.normal(0, 0.5, 2)
        state[:2] = initial_guess
        initial_positions.append(initial_guess.copy())
        cgn.add_node(i, state, is_anchor=False)

initial_positions = np.array(initial_positions)

# Add edges and measurements
edges = []
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        dist = np.linalg.norm(true_positions[i] - true_positions[j])
        if dist <= comm_range:
            edges.append((i, j))
            cgn.add_edge(i, j)
            meas_range = dist + np.random.normal(0, range_noise_std)
            cgn.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

# Optimize
results = cgn.optimize()

# Extract final positions
estimated_positions = np.zeros((n_nodes, 2))
for i in range(n_nodes):
    estimated_positions[i] = cgn.nodes[i].state[:2]

# Calculate errors
errors = []
for i in range(n_anchors, n_nodes):
    error = np.linalg.norm(estimated_positions[i] - true_positions[i])
    errors.append(error)

rmse = np.sqrt(np.mean(np.array(errors)**2))

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Network Topology
ax1 = axes[0]
ax1.set_title('Network Topology', fontsize=14, fontweight='bold')
ax1.set_xlabel('X Position (m)', fontsize=12)
ax1.set_ylabel('Y Position (m)', fontsize=12)
ax1.set_xlim(-5, area_size+5)
ax1.set_ylim(-5, area_size+5)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Draw edges
for i, j in edges:
    x_coords = [true_positions[i, 0], true_positions[j, 0]]
    y_coords = [true_positions[i, 1], true_positions[j, 1]]
    ax1.plot(x_coords, y_coords, 'gray', alpha=0.2, linewidth=0.5)

# Draw nodes
ax1.scatter(true_positions[n_anchors:, 0], true_positions[n_anchors:, 1],
           c='blue', s=50, label='Unknown Nodes', zorder=3)
ax1.scatter(anchor_positions[:4, 0], anchor_positions[:4, 1],
           c='red', s=200, marker='^', label='Corner Anchors', zorder=4)
ax1.scatter(anchor_positions[4, 0], anchor_positions[4, 1],
           c='darkred', s=200, marker='s', label='Center Anchor', zorder=4)

# Add communication range circle for one node
circle = plt.Circle((true_positions[15, 0], true_positions[15, 1]),
                    comm_range, fill=False, color='green',
                    linestyle='--', alpha=0.3, label=f'{comm_range}m range')
ax1.add_patch(circle)

ax1.legend(loc='upper right', fontsize=10)

# Plot 2: Estimated vs Actual Positions
ax2 = axes[1]
ax2.set_title('Estimated vs Actual Positions', fontsize=14, fontweight='bold')
ax2.set_xlabel('X Position (m)', fontsize=12)
ax2.set_ylabel('Y Position (m)', fontsize=12)
ax2.set_xlim(-5, area_size+5)
ax2.set_ylim(-5, area_size+5)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Draw error lines
for i in range(n_anchors, n_nodes):
    ax2.plot([true_positions[i, 0], estimated_positions[i, 0]],
            [true_positions[i, 1], estimated_positions[i, 1]],
            'r-', alpha=0.5, linewidth=1)

# True positions
ax2.scatter(true_positions[n_anchors:, 0], true_positions[n_anchors:, 1],
           c='blue', s=100, marker='o', label='True Position',
           edgecolors='black', linewidth=1, zorder=3)

# Estimated positions
ax2.scatter(estimated_positions[n_anchors:, 0], estimated_positions[n_anchors:, 1],
           c='green', s=50, marker='x', label='Estimated', zorder=4)

# Anchors
ax2.scatter(anchor_positions[:4, 0], anchor_positions[:4, 1],
           c='red', s=200, marker='^', label='Anchors', zorder=5)
ax2.scatter(anchor_positions[4, 0], anchor_positions[4, 1],
           c='darkred', s=200, marker='s', zorder=5)

ax2.legend(loc='upper right', fontsize=10)
ax2.text(2, 47, f'RMSE: {rmse*100:.1f} cm', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Plot 3: Error Distribution
ax3 = axes[2]
ax3.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Position Error (cm)', fontsize=12)
ax3.set_ylabel('Number of Nodes', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

errors_cm = [e*100 for e in errors]
bins = np.linspace(0, max(errors_cm)*1.1, 15)
n, bins, patches = ax3.hist(errors_cm, bins=bins, color='skyblue',
                            edgecolor='black', alpha=0.7)

# Color code the bars
for i, patch in enumerate(patches):
    if bins[i] < 1.0:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)
    elif bins[i] < 2.0:
        patch.set_facecolor('yellow')
        patch.set_alpha(0.7)
    else:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)

# Add statistics
mean_error = np.mean(errors_cm)
median_error = np.median(errors_cm)
max_error = max(errors_cm)

ax3.axvline(mean_error, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_error:.1f} cm')
ax3.axvline(median_error, color='blue', linestyle='--', linewidth=2,
           label=f'Median: {median_error:.1f} cm')

# Add text box with statistics
stats_text = f'RMSE: {rmse*100:.1f} cm\nMean: {mean_error:.1f} cm\nMedian: {median_error:.1f} cm\nMax: {max_error:.1f} cm\nNodes < 1cm: {sum(1 for e in errors_cm if e < 1)}/{n_unknowns}'
ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax3.legend(loc='upper right', fontsize=10)

# Main title
fig.suptitle(f'30-Node Consensus Localization Performance (50×50m Area)\n' +
             f'{n_anchors} Anchors, {comm_range}m Communication Range, {range_noise_std*100:.0f}cm Noise',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_file = 'consensus_30node_performance.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Figure saved as {output_file}")

# Also save a convergence plot
fig2, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Convergence Behavior', fontsize=14, fontweight='bold')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('RMSE (cm)', fontsize=12)
ax.grid(True, alpha=0.3)

# Track RMSE over iterations (run again to get history)
cgn2 = ConsensusGaussNewton(config)

# Setup network again
for i in range(n_nodes):
    state = np.zeros(5)
    if i < n_anchors:
        state[:2] = true_positions[i]
        cgn2.add_node(i, state, is_anchor=True)
    else:
        state[:2] = initial_positions[i]
        cgn2.add_node(i, state, is_anchor=False)

for i, j in edges:
    cgn2.add_edge(i, j)
    dist = np.linalg.norm(true_positions[i] - true_positions[j])
    meas_range = dist + np.random.normal(0, range_noise_std)
    cgn2.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

cgn2.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

# Manual iteration to track progress
rmse_history = []
for k in range(500):
    cgn2._exchange_states()
    for node_id in range(n_anchors, n_nodes):
        cgn2.nodes[node_id].update_state()

    errors = []
    for i in range(n_anchors, n_nodes):
        est_pos = cgn2.nodes[i].state[:2]
        true_pos = true_positions[i]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors)**2))
    rmse_history.append(rmse * 100)  # Convert to cm

    if len(rmse_history) > 10:
        if abs(rmse_history[-1] - rmse_history[-10]) < 0.01:  # 0.01cm change
            break

ax.plot(rmse_history, 'b-', linewidth=2, label='RMSE')
ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1 cm target')
ax.axhline(y=rmse*100, color='r', linestyle='--', alpha=0.5, label=f'Final: {rmse*100:.1f} cm')

ax.set_xlim(0, len(rmse_history))
ax.set_ylim(0, max(rmse_history)*1.1)
ax.legend(fontsize=12)

# Add annotations
convergence_iter = len(rmse_history)
ax.annotate(f'Converged at iteration {convergence_iter}',
            xy=(convergence_iter, rmse_history[-1]),
            xytext=(convergence_iter*0.7, rmse_history[-1]*1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11)

plt.tight_layout()
plt.savefig('consensus_convergence.png', dpi=150, bbox_inches='tight')
print(f"Convergence plot saved as consensus_convergence.png")

plt.show()