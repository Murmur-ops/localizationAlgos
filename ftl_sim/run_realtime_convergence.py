#!/usr/bin/env python3
"""
Real-time per-node convergence tracking with proper iteration-by-iteration updates
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    generate_all_measurements
)
from ftl.consensus.consensus_gn import ConsensusGNConfig, ConsensusGaussNewton
from ftl.factors_scaled import ToAFactorMeters

def run_realtime_convergence():
    """Track real convergence by manually iterating the consensus algorithm"""

    print("=" * 70)
    print("REAL-TIME PER-NODE CONVERGENCE - UNIFIED FTL SYSTEM")
    print("=" * 70)

    # Load configuration
    with open('configs/unified_ideal.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate network
    np.random.seed(42)
    true_positions, n_anchors, n_total = generate_network_topology(config)

    # Initialize clocks
    clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)

    # Generate measurements
    print("\nGenerating RF measurements...")
    measurements = generate_all_measurements(true_positions, clock_states, config['rf_simulation'])

    # Setup consensus manually to have more control
    print("Setting up consensus algorithm...")

    cgn_config = ConsensusGNConfig(
        max_iterations=1,  # We'll iterate manually
        step_size=config['consensus'].get('damping_factor', 0.5),
        gradient_tol=1e-10,
        step_tol=1e-10,
        verbose=False
    )

    cgn = ConsensusGaussNewton(cgn_config)
    cgn.set_true_positions(true_positions)

    # Add nodes
    for i in range(n_total):
        if i < n_anchors:
            # Anchor node - perfect position, small clock errors
            state = np.zeros(5)
            state[:2] = true_positions[i]
            state[2] = clock_states[i].bias * 1e9
            state[3] = clock_states[i].drift * 1e9
            state[4] = clock_states[i].cfo
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Unknown node - noisy initial position
            state = np.zeros(5)
            state[:2] = true_positions[i] + np.random.normal(0,
                config['consensus'].get('initial_position_std', 0.1), 2)
            state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 0.1)
            state[3] = clock_states[i].drift * 1e9
            state[4] = clock_states[i].cfo + np.random.normal(0, 0.01)
            cgn.add_node(i, state, is_anchor=False)

    # Add edges for all measurement pairs
    for (i, j) in measurements.keys():
        cgn.add_edge(i, j)

    # Add measurements as factors
    measurement_variance_m2 = (0.01)**2  # 1 cm std dev
    for (i, j), meas_list in measurements.items():
        for meas in meas_list:
            # Convert to ToA factor
            factor = ToAFactorMeters(i, j, meas['range_m'], measurement_variance_m2)
            cgn.add_measurement(factor)

    # Track states over iterations
    n_iterations = 100
    state_history = {i: [] for i in range(n_anchors, n_total)}

    print(f"\nRunning {n_iterations} iterations manually...")
    print(f"Tracking {n_total - n_anchors} unknown nodes")

    for iteration in range(n_iterations):
        # Store current states
        for i in range(n_anchors, n_total):
            state_history[i].append(cgn.nodes[i].state.copy())

        # Manually perform one consensus iteration
        # Build and solve local systems for all non-anchor nodes
        for node_id, node in cgn.nodes.items():
            if not node.config.is_anchor:
                # Build local linearized system
                H, g = node.build_local_system()

                # Solve for state update with regularization
                try:
                    delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                    # Apply damped update
                    node.state += cgn_config.step_size * delta
                except np.linalg.LinAlgError:
                    pass  # Skip if singular

        # Update state sharing between neighbors
        for node_id, node in cgn.nodes.items():
            node.update_state()

        if iteration % 20 == 0:
            # Calculate average position error
            pos_errors = []
            for i in range(n_anchors, n_total):
                true_pos = true_positions[i]
                est_pos = cgn.nodes[i].state[:2]
                error = np.linalg.norm(est_pos - true_pos)
                pos_errors.append(error)
            avg_error = np.mean(pos_errors) * 100  # cm

            print(f"Iteration {iteration:3d}: Avg position error = {avg_error:.2f} cm")

    # Final state capture
    for i in range(n_anchors, n_total):
        state_history[i].append(cgn.nodes[i].state.copy())

    # Create convergence plots
    n_plot_iterations = len(state_history[n_anchors])
    iterations = np.arange(n_plot_iterations)
    n_unknown = n_total - n_anchors

    # Create colormap
    colors = cm.viridis(np.linspace(0.2, 0.9, n_unknown))

    # Figure 1: Timing Convergence (showing errors decreasing)
    plt.figure(figsize=(14, 8))

    # Plot individual node timing errors
    for idx, i in enumerate(range(n_anchors, min(n_anchors + 25, n_total))):  # Limit to 25 nodes for clarity
        true_bias_ns = clock_states[i].bias * 1e9
        timing_errors = [abs(state[2] - true_bias_ns) for state in state_history[i]]
        plt.semilogy(iterations, timing_errors, color=colors[idx % len(colors)],
                     alpha=0.5, linewidth=1)

    # Calculate and plot average
    avg_timing_errors = []
    for j in range(n_plot_iterations):
        errors = []
        for i in range(n_anchors, n_total):
            true_bias_ns = clock_states[i].bias * 1e9
            est_bias_ns = state_history[i][j][2]
            errors.append(abs(est_bias_ns - true_bias_ns))
        avg_timing_errors.append(np.sqrt(np.mean(np.array(errors)**2)))  # RMS

    plt.semilogy(iterations, avg_timing_errors, 'k-', linewidth=3,
                label=f'RMS (final: {avg_timing_errors[-1]:.3f} ns)')

    plt.axhline(y=0.037, color='r', linestyle='--', alpha=0.7, label='Target: 0.037 ns')
    plt.axhline(y=0.289, color='gray', linestyle=':', alpha=0.5,
                label='Single-sample limit: 0.289 ns')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Timing Error (ns)', fontsize=12)
    plt.title('Per-Node Timing Offset Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.01, 1])

    plt.tight_layout()
    plt.savefig('realtime_timing_convergence.png', dpi=100)
    print("\nSaved: realtime_timing_convergence.png")

    # Figure 2: Frequency Convergence
    plt.figure(figsize=(14, 8))

    for idx, i in enumerate(range(n_anchors, min(n_anchors + 25, n_total))):
        true_cfo_ppm = clock_states[i].cfo
        cfo_errors = [abs(state[4] - true_cfo_ppm) for state in state_history[i]]
        plt.semilogy(iterations, cfo_errors, color=colors[idx % len(colors)],
                     alpha=0.5, linewidth=1)

    # Calculate and plot average
    avg_cfo_errors = []
    for j in range(n_plot_iterations):
        errors = []
        for i in range(n_anchors, n_total):
            true_cfo = clock_states[i].cfo
            est_cfo = state_history[i][j][4]
            errors.append(abs(est_cfo - true_cfo))
        avg_cfo_errors.append(np.sqrt(np.mean(np.array(errors)**2)))  # RMS

    plt.semilogy(iterations, avg_cfo_errors, 'k-', linewidth=3,
                label=f'RMS (final: {avg_cfo_errors[-1]:.3f} ppm)')

    plt.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='OCXO: 1 ppm')
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='TCXO: 10 ppm')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Frequency Error (ppm)', fontsize=12)
    plt.title('Per-Node Frequency Offset Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.001, 1])

    plt.tight_layout()
    plt.savefig('realtime_frequency_convergence.png', dpi=100)
    print("Saved: realtime_frequency_convergence.png")

    # Figure 3: Position Convergence
    plt.figure(figsize=(14, 8))

    for idx, i in enumerate(range(n_anchors, min(n_anchors + 25, n_total))):
        true_pos = true_positions[i]
        pos_errors = []
        for state in state_history[i]:
            est_pos = state[:2]
            error = np.linalg.norm(est_pos - true_pos) * 100  # cm
            pos_errors.append(error)
        plt.semilogy(iterations, pos_errors, color=colors[idx % len(colors)],
                     alpha=0.5, linewidth=1)

    # Calculate and plot average
    avg_pos_errors = []
    rms_pos_errors = []
    for j in range(n_plot_iterations):
        errors = []
        for i in range(n_anchors, n_total):
            true_pos = true_positions[i]
            est_pos = state_history[i][j][:2]
            error = np.linalg.norm(est_pos - true_pos) * 100  # cm
            errors.append(error)
        avg_pos_errors.append(np.mean(errors))
        rms_pos_errors.append(np.sqrt(np.mean(np.array(errors)**2)))

    plt.semilogy(iterations, rms_pos_errors, 'k-', linewidth=3,
                label=f'RMS (final: {rms_pos_errors[-1]:.2f} cm)')

    plt.axhline(y=2.66, color='b', linestyle='--', alpha=0.7, label='Target: 2.66 cm')
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sub-cm target')
    plt.axhline(y=8.67, color='gray', linestyle=':', alpha=0.5,
                label='Quantization limit: 8.67 cm')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Position Error (cm)', fontsize=12)
    plt.title('Per-Node Position Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.1, 100])

    plt.tight_layout()
    plt.savefig('realtime_position_convergence.png', dpi=100)
    print("Saved: realtime_position_convergence.png")

    # Print final statistics
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE STATISTICS")
    print("=" * 70)

    print(f"\nTiming RMSE:    {avg_timing_errors[-1]:.3f} ns")
    print(f"Frequency RMSE: {avg_cfo_errors[-1]:.3f} ppm")
    print(f"Position RMSE:  {rms_pos_errors[-1]:.2f} cm")

    # Per-node statistics
    final_pos_errors = []
    for i in range(n_anchors, n_total):
        true_pos = true_positions[i]
        est_pos = state_history[i][-1][:2]
        error = np.linalg.norm(est_pos - true_pos) * 100
        final_pos_errors.append(error)

    print(f"\nPosition Error Distribution:")
    print(f"  Best node:  {np.min(final_pos_errors):.2f} cm")
    print(f"  Worst node: {np.max(final_pos_errors):.2f} cm")
    print(f"  Median:     {np.median(final_pos_errors):.2f} cm")
    print(f"  Std Dev:    {np.std(final_pos_errors):.2f} cm")

    return state_history

if __name__ == "__main__":
    history = run_realtime_convergence()