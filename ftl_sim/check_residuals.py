#!/usr/bin/env python3
"""
Check measurement residuals to see if nodes are actually fitting the measurements.
"""

import numpy as np
import yaml
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import generate_network_topology, initialize_clock_states
from test_perfect_measurements import generate_perfect_measurements
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters


# Setup
print("Setting up network with perfect measurements...")
with open('configs/unified_perturbed.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(42)
true_positions, n_anchors, n_total = generate_network_topology(config)
n_total = min(10, n_total)  # Use fewer nodes for analysis

clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)
measurements = generate_perfect_measurements(true_positions[:n_total], clock_states)

# Create consensus system
cgn_config = ConsensusGNConfig(step_size=0.9, consensus_gain=0.0, verbose=False)
cgn = ConsensusGaussNewton(cgn_config)

# Add nodes with large initial errors
for i in range(n_total):
    state = np.zeros(5)
    if i < n_anchors:
        state[:2] = true_positions[i]
        state[2] = clock_states[i].bias * 1e9
        cgn.add_node(i, state, is_anchor=True)
    else:
        state[:2] = true_positions[i] + np.random.normal(0, 5.0, 2)
        state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 10)
        cgn.add_node(i, state, is_anchor=False)

# Add measurements
factors = []
for (i, j), meas_list in measurements.items():
    if i < n_total and j < n_total:
        cgn.add_edge(i, j)
        for meas in meas_list:
            factor = ToAFactorMeters(i, j, meas['range_m'], (1e-9)**2)
            cgn.add_measurement(factor)
            factors.append(factor)

print(f"Network: {n_total} nodes ({n_anchors} anchors), {len(factors)} measurements")

# Run some iterations
print("\nRunning 50 iterations...")
for iteration in range(50):
    # Share states
    current_time = time.time()
    for node_id, node in cgn.nodes.items():
        for edge in cgn.edges:
            if edge[0] == node_id and edge[1] in cgn.nodes:
                neighbor_id = edge[1]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 iteration, current_time)
                node.receive_state(msg)
            elif edge[1] == node_id and edge[0] in cgn.nodes:
                neighbor_id = edge[0]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 iteration, current_time)
                node.receive_state(msg)

    # Update
    for node_id, node in cgn.nodes.items():
        if not node.config.is_anchor:
            H, g = node.build_local_system()
            if np.sum(np.abs(H)) > 0:
                try:
                    delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                    node.state += cgn_config.step_size * delta
                except:
                    pass

# Check residuals
print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

residuals = []
for factor in factors[:20]:  # Check first 20
    xi = cgn.nodes[factor.i].state
    xj = cgn.nodes[factor.j].state
    residual = factor.residual(xi, xj)
    residuals.append(residual)

    if abs(residual) > 0.01:  # More than 1cm residual
        print(f"Factor ({factor.i},{factor.j}): residual = {residual:.4f} m")

residuals = np.array(residuals)
print(f"\nResidual statistics:")
print(f"  Mean:  {np.mean(np.abs(residuals)):.6f} m")
print(f"  RMS:   {np.sqrt(np.mean(residuals**2)):.6f} m")
print(f"  Max:   {np.max(np.abs(residuals)):.6f} m")

# Check position errors
print("\n" + "=" * 70)
print("POSITION ERRORS")
print("=" * 70)

for i in range(n_anchors, n_total):
    est_pos = cgn.nodes[i].state[:2]
    true_pos = true_positions[i]
    error = np.linalg.norm(est_pos - true_pos)
    print(f"Node {i}: {error*100:.2f} cm")

# Check if it's a global transformation issue
print("\n" + "=" * 70)
print("CHECK FOR GLOBAL TRANSFORMATION")
print("=" * 70)

# Calculate centroid shift
est_positions = np.array([cgn.nodes[i].state[:2] for i in range(n_anchors, n_total)])
true_positions_subset = np.array([true_positions[i] for i in range(n_anchors, n_total)])

est_centroid = np.mean(est_positions, axis=0)
true_centroid = np.mean(true_positions_subset, axis=0)
centroid_shift = est_centroid - true_centroid

print(f"Centroid shift: [{centroid_shift[0]:.3f}, {centroid_shift[1]:.3f}] m")

# Check if correcting for shift improves things
corrected_positions = est_positions - centroid_shift
corrected_errors = [np.linalg.norm(corrected_positions[i] - true_positions_subset[i])
                    for i in range(len(corrected_positions))]

print(f"\nAfter removing centroid shift:")
print(f"  Mean error:  {np.mean(corrected_errors)*100:.2f} cm")
print(f"  RMS error:   {np.sqrt(np.mean(np.array(corrected_errors)**2))*100:.2f} cm")