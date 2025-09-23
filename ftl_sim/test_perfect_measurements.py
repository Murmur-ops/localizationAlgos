#!/usr/bin/env python3
"""
Test convergence with truly perfect measurements (no noise at all).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import generate_network_topology, initialize_clock_states
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters


def generate_perfect_measurements(true_positions, clock_states):
    """Generate perfect noiseless measurements"""
    n_nodes = len(true_positions)
    measurements = {}

    # Create measurements for all pairs
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # True geometric distance
            true_distance = np.linalg.norm(true_positions[i] - true_positions[j])

            # Add clock bias effects (convert ns to meters)
            c = 299792458.0  # m/s
            clock_bias_m = (clock_states[j].bias - clock_states[i].bias) * c

            # Perfect measurement = true distance + clock effects
            measured_range = true_distance + clock_bias_m

            # Store measurement
            if (i, j) not in measurements:
                measurements[(i, j)] = []

            measurements[(i, j)].append({
                'range_m': measured_range,
                'variance_m2': (1e-9)**2  # Essentially zero variance
            })

    return measurements


def test_perfect_convergence():
    """Test convergence with perfect measurements"""

    print("=" * 70)
    print("TESTING CONVERGENCE WITH PERFECT MEASUREMENTS")
    print("=" * 70)

    # Load config
    with open('configs/unified_perturbed.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate network
    np.random.seed(42)
    true_positions, n_anchors, n_total = generate_network_topology(config)

    # Initialize clocks
    clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)

    # Generate PERFECT measurements
    print("\nGenerating perfect noiseless measurements...")
    measurements = generate_perfect_measurements(true_positions, clock_states)
    print(f"  Generated {len(measurements)} measurement pairs")

    # Setup consensus with larger step size and no consensus penalty
    cgn_config = ConsensusGNConfig(
        max_iterations=1,
        step_size=0.9,  # Increased from 0.5 to 0.9
        consensus_gain=0.0,  # No consensus penalty - pure measurement fitting
        verbose=False
    )

    cgn = ConsensusGaussNewton(cgn_config)
    cgn.set_true_positions(true_positions)

    # Add nodes with large initial errors
    print("\nInitializing nodes with 5m position errors...")
    for i in range(n_total):
        state = np.zeros(5)
        if i < n_anchors:
            # Perfect anchors
            state[:2] = true_positions[i]
            state[2] = clock_states[i].bias * 1e9
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Large initial errors
            state[:2] = true_positions[i] + np.random.normal(0, 5.0, 2)  # 5m error
            state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 10)  # 10ns error
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and perfect measurements
    for (i, j) in measurements.keys():
        cgn.add_edge(i, j)

    for (i, j), meas_list in measurements.items():
        for meas in meas_list:
            # Use tiny variance for numerical stability
            factor = ToAFactorMeters(i, j, meas['range_m'], (1e-9)**2)
            cgn.add_measurement(factor)

    # Track convergence
    n_iterations = 200
    position_errors = []
    timing_errors = []

    print(f"\nRunning {n_iterations} iterations...")
    print("-" * 40)

    for iteration in range(n_iterations):
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

        # Update nodes
        for node_id, node in cgn.nodes.items():
            if not node.config.is_anchor:
                H, g = node.build_local_system()
                if np.sum(np.abs(H)) > 0:
                    try:
                        delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                        node.state += cgn_config.step_size * delta
                    except:
                        pass

        # Calculate errors
        pos_errs = []
        time_errs = []
        for i in range(n_anchors, n_total):
            pos_err = np.linalg.norm(cgn.nodes[i].state[:2] - true_positions[i])
            pos_errs.append(pos_err)

            time_err = abs(cgn.nodes[i].state[2] - clock_states[i].bias * 1e9)
            time_errs.append(time_err)

        position_errors.append(pos_errs)
        timing_errors.append(time_errs)

        if iteration % 20 == 0:
            avg_pos = np.mean(pos_errs)
            avg_time = np.mean(time_errs)
            print(f"Iteration {iteration:3d}: Pos = {avg_pos*100:.4f} cm, Time = {avg_time:.4f} ns")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS WITH PERFECT MEASUREMENTS")
    print("=" * 70)

    final_pos_errors = position_errors[-1]
    final_time_errors = timing_errors[-1]

    print(f"\nPosition errors:")
    print(f"  Mean:  {np.mean(final_pos_errors)*100:.6f} cm")
    print(f"  RMS:   {np.sqrt(np.mean(np.array(final_pos_errors)**2))*100:.6f} cm")
    print(f"  Max:   {np.max(final_pos_errors)*100:.6f} cm")
    print(f"  Min:   {np.min(final_pos_errors)*100:.6f} cm")

    print(f"\nTiming errors:")
    print(f"  Mean:  {np.mean(final_time_errors):.6f} ns")
    print(f"  Max:   {np.max(final_time_errors):.6f} ns")

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    iterations = np.arange(n_iterations)
    position_errors = np.array(position_errors)
    timing_errors = np.array(timing_errors)

    # Position convergence
    for i in range(min(20, n_total - n_anchors)):
        ax1.semilogy(iterations, position_errors[:, i] * 100, alpha=0.3)

    # Add RMS line
    rms_pos = np.sqrt(np.mean(position_errors**2, axis=1)) * 100
    ax1.semilogy(iterations, rms_pos, 'k-', linewidth=2, label=f'RMS (final: {rms_pos[-1]:.4f} cm)')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position Error (cm)')
    ax1.set_title('Position Convergence with Perfect Measurements')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-4, 1e3])

    # Timing convergence
    for i in range(min(20, n_total - n_anchors)):
        ax2.semilogy(iterations, timing_errors[:, i], alpha=0.3)

    # Add RMS line
    rms_time = np.sqrt(np.mean(timing_errors**2, axis=1))
    ax2.semilogy(iterations, rms_time, 'k-', linewidth=2, label=f'RMS (final: {rms_time[-1]:.4f} ns)')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Timing Error (ns)')
    ax2.set_title('Timing Convergence with Perfect Measurements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1e-6, 100])

    plt.tight_layout()
    plt.savefig('perfect_convergence.png', dpi=100)
    print(f"\nPlot saved to perfect_convergence.png")
    plt.show()

    return position_errors, timing_errors


if __name__ == "__main__":
    pos_errors, time_errors = test_perfect_convergence()