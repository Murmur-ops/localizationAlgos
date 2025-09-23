#!/usr/bin/env python3
"""
Track individual node convergence using the working consensus implementation.
Run the optimization properly and capture intermediate states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
from pathlib import Path
import sys
import copy

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    generate_all_measurements,
    setup_consensus_from_measurements
)

def run_individual_convergence():
    """Run proper consensus and track individual node convergence"""

    print("=" * 70)
    print("INDIVIDUAL NODE CONVERGENCE ANALYSIS")
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

    # Setup consensus
    print("Setting up consensus algorithm...")
    cgn = setup_consensus_from_measurements(
        true_positions, measurements, clock_states,
        n_anchors, config['consensus']
    )
    cgn.set_true_positions(true_positions)

    # Store initial states
    initial_states = {}
    for i in range(n_anchors, n_total):
        initial_states[i] = cgn.nodes[i].state.copy()

    # Track convergence by running multiple times with increasing iterations
    n_checkpoints = 50
    max_iterations = 100

    # Arrays to store state history
    state_history = {i: [] for i in range(n_anchors, n_total)}
    iteration_points = []

    print(f"\nTracking convergence at {n_checkpoints} checkpoints...")

    for checkpoint in range(n_checkpoints):
        # Calculate iterations for this checkpoint (exponential spacing for early detail)
        if checkpoint == 0:
            n_iter = 1
        else:
            # Exponential spacing: more detail early, less later
            n_iter = int(1 + (max_iterations - 1) * (checkpoint / (n_checkpoints - 1))**2)

        iteration_points.append(n_iter)

        # Reset to initial states
        cgn_temp = setup_consensus_from_measurements(
            true_positions, measurements, clock_states,
            n_anchors, config['consensus']
        )
        cgn_temp.set_true_positions(true_positions)

        # Run optimization for n_iter iterations
        cgn_temp.config.max_iterations = n_iter
        cgn_temp.config.verbose = False
        result = cgn_temp.optimize()

        # Store current states
        for i in range(n_anchors, n_total):
            state_history[i].append(cgn_temp.nodes[i].state.copy())

        if checkpoint % 10 == 0:
            # Calculate average error
            errors = []
            for i in range(n_anchors, n_total):
                true_pos = true_positions[i]
                est_pos = cgn_temp.nodes[i].state[:2]
                error = np.linalg.norm(est_pos - true_pos) * 100
                errors.append(error)
            print(f"  After {n_iter:3d} iterations: Avg error = {np.mean(errors):.2f} cm")

    # Create proper convergence plots
    n_unknown = n_total - n_anchors
    colors = cm.viridis(np.linspace(0.2, 0.9, min(n_unknown, 25)))  # Limit colors

    # Figure 1: Individual Timing Convergence
    plt.figure(figsize=(14, 8))

    # Plot a subset of nodes for clarity
    nodes_to_plot = list(range(n_anchors, min(n_anchors + 25, n_total)))

    for idx, i in enumerate(nodes_to_plot):
        true_bias_ns = clock_states[i].bias * 1e9
        timing_errors = []
        for state in state_history[i]:
            error = abs(state[2] - true_bias_ns)
            timing_errors.append(error)

        plt.semilogy(iteration_points, timing_errors,
                    color=colors[idx], alpha=0.6, linewidth=1.5,
                    label=f'Node {i}' if idx < 5 else '')  # Only label first 5

    # Calculate and plot RMS
    rms_timing = []
    for j in range(len(iteration_points)):
        errors = []
        for i in range(n_anchors, n_total):
            true_bias_ns = clock_states[i].bias * 1e9
            est_bias_ns = state_history[i][j][2]
            errors.append((est_bias_ns - true_bias_ns)**2)
        rms_timing.append(np.sqrt(np.mean(errors)))

    plt.semilogy(iteration_points, rms_timing, 'k-', linewidth=3,
                label=f'RMS (final: {rms_timing[-1]:.3f} ns)')

    plt.axhline(y=0.037, color='r', linestyle='--', alpha=0.7,
                label='Target: 0.037 ns')
    plt.axhline(y=0.289, color='gray', linestyle=':', alpha=0.5,
                label='Quantization limit: 0.289 ns')

    plt.xlabel('Consensus Iteration', fontsize=12)
    plt.ylabel('Timing Error (ns)', fontsize=12)
    plt.title('Individual Node Timing Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim([0, max_iterations])
    plt.ylim([0.01, 1])

    plt.tight_layout()
    plt.savefig('individual_timing_convergence.png', dpi=100)
    print("\nSaved: individual_timing_convergence.png")

    # Figure 2: Individual Frequency Convergence
    plt.figure(figsize=(14, 8))

    for idx, i in enumerate(nodes_to_plot):
        true_cfo_ppm = clock_states[i].cfo
        cfo_errors = []
        for state in state_history[i]:
            error = abs(state[4] - true_cfo_ppm)
            cfo_errors.append(error)

        plt.semilogy(iteration_points, cfo_errors,
                    color=colors[idx], alpha=0.6, linewidth=1.5,
                    label=f'Node {i}' if idx < 5 else '')

    # Calculate and plot RMS
    rms_cfo = []
    for j in range(len(iteration_points)):
        errors = []
        for i in range(n_anchors, n_total):
            true_cfo = clock_states[i].cfo
            est_cfo = state_history[i][j][4]
            errors.append((est_cfo - true_cfo)**2)
        rms_cfo.append(np.sqrt(np.mean(errors)))

    plt.semilogy(iteration_points, rms_cfo, 'k-', linewidth=3,
                label=f'RMS (final: {rms_cfo[-1]:.3f} ppm)')

    plt.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='OCXO: 1 ppm')
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='TCXO: 10 ppm')

    plt.xlabel('Consensus Iteration', fontsize=12)
    plt.ylabel('Frequency Error (ppm)', fontsize=12)
    plt.title('Individual Node Frequency Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim([0, max_iterations])
    plt.ylim([0.001, 1])

    plt.tight_layout()
    plt.savefig('individual_frequency_convergence.png', dpi=100)
    print("Saved: individual_frequency_convergence.png")

    # Figure 3: Individual Position Convergence
    plt.figure(figsize=(14, 8))

    for idx, i in enumerate(nodes_to_plot):
        true_pos = true_positions[i]
        pos_errors = []
        for state in state_history[i]:
            est_pos = state[:2]
            error = np.linalg.norm(est_pos - true_pos) * 100  # cm
            pos_errors.append(error)

        plt.semilogy(iteration_points, pos_errors,
                    color=colors[idx], alpha=0.6, linewidth=1.5,
                    label=f'Node {i}' if idx < 5 else '')

    # Calculate and plot RMS
    rms_pos = []
    for j in range(len(iteration_points)):
        errors = []
        for i in range(n_anchors, n_total):
            true_pos = true_positions[i]
            est_pos = state_history[i][j][:2]
            error = np.linalg.norm(est_pos - true_pos) * 100
            errors.append(error**2)
        rms_pos.append(np.sqrt(np.mean(errors)))

    plt.semilogy(iteration_points, rms_pos, 'k-', linewidth=3,
                label=f'RMS (final: {rms_pos[-1]:.2f} cm)')

    plt.axhline(y=2.66, color='b', linestyle='--', alpha=0.7,
                label='Target: 2.66 cm')
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5,
                label='Sub-cm target')
    plt.axhline(y=8.67, color='gray', linestyle=':', alpha=0.5,
                label='Quantization limit: 8.67 cm')

    plt.xlabel('Consensus Iteration', fontsize=12)
    plt.ylabel('Position Error (cm)', fontsize=12)
    plt.title('Individual Node Position Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlim([0, max_iterations])
    plt.ylim([0.1, 100])

    plt.tight_layout()
    plt.savefig('individual_position_convergence.png', dpi=100)
    print("Saved: individual_position_convergence.png")

    # Create combined summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Individual Node Convergence Summary', fontsize=16, fontweight='bold')

    # Timing subplot
    ax = axes[0]
    for idx, i in enumerate(nodes_to_plot[:10]):  # First 10 nodes only
        true_bias_ns = clock_states[i].bias * 1e9
        timing_errors = [abs(state[2] - true_bias_ns) for state in state_history[i]]
        ax.semilogy(iteration_points, timing_errors, alpha=0.5, linewidth=1)
    ax.semilogy(iteration_points, rms_timing, 'k-', linewidth=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Timing Error (ns)')
    ax.set_title(f'Timing: {rms_timing[-1]:.3f} ns RMS')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.01, 1])

    # Frequency subplot
    ax = axes[1]
    for idx, i in enumerate(nodes_to_plot[:10]):
        true_cfo = clock_states[i].cfo
        cfo_errors = [abs(state[4] - true_cfo) for state in state_history[i]]
        ax.semilogy(iteration_points, cfo_errors, alpha=0.5, linewidth=1)
    ax.semilogy(iteration_points, rms_cfo, 'k-', linewidth=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frequency Error (ppm)')
    ax.set_title(f'Frequency: {rms_cfo[-1]:.3f} ppm RMS')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.001, 1])

    # Position subplot
    ax = axes[2]
    for idx, i in enumerate(nodes_to_plot[:10]):
        true_pos = true_positions[i]
        pos_errors = [np.linalg.norm(state[:2] - true_pos) * 100 for state in state_history[i]]
        ax.semilogy(iteration_points, pos_errors, alpha=0.5, linewidth=1)
    ax.semilogy(iteration_points, rms_pos, 'k-', linewidth=3)
    ax.axhline(y=2.66, color='b', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position Error (cm)')
    ax.set_title(f'Position: {rms_pos[-1]:.2f} cm RMS')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.1, 100])

    plt.tight_layout()
    plt.savefig('individual_convergence_summary.png', dpi=100)
    print("Saved: individual_convergence_summary.png")

    # Print final statistics
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE STATISTICS")
    print("=" * 70)

    print(f"\nRMS Errors:")
    print(f"  Timing:    {rms_timing[-1]:.3f} ns")
    print(f"  Frequency: {rms_cfo[-1]:.3f} ppm")
    print(f"  Position:  {rms_pos[-1]:.2f} cm")

    # Individual node statistics
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

    # Identify best and worst performing nodes
    best_idx = n_anchors + np.argmin(final_pos_errors)
    worst_idx = n_anchors + np.argmax(final_pos_errors)
    print(f"\n  Best node ID:  {best_idx} ({final_pos_errors[best_idx-n_anchors]:.2f} cm)")
    print(f"  Worst node ID: {worst_idx} ({final_pos_errors[worst_idx-n_anchors]:.2f} cm)")

    return state_history, iteration_points

if __name__ == "__main__":
    history, iterations = run_individual_convergence()