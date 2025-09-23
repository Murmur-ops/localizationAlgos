#!/usr/bin/env python3
"""
Run per-node convergence analysis and generate three separate figures:
1. Timing offset convergence for each node
2. Frequency offset convergence for each node
3. Localization RMSE convergence for each node
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
    generate_all_measurements,
    setup_consensus_from_measurements
)

def run_pernode_convergence_analysis():
    """Run simulation and track per-node convergence of all three metrics"""

    print("=" * 70)
    print("PER-NODE CONVERGENCE ANALYSIS - UNIFIED FTL SYSTEM")
    print("=" * 70)

    # Load configuration
    with open('configs/unified_ideal.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate network
    np.random.seed(42)  # For reproducibility
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

    # Track metrics per node over iterations
    n_iterations = 100
    n_unknown = n_total - n_anchors

    # Initialize storage for per-node histories
    timing_histories = {i: [] for i in range(n_anchors, n_total)}
    frequency_histories = {i: [] for i in range(n_anchors, n_total)}
    position_histories = {i: [] for i in range(n_anchors, n_total)}

    print(f"\nRunning {n_iterations} consensus iterations...")
    print(f"Tracking {n_unknown} unknown nodes (IDs {n_anchors}-{n_total-1})")
    print("-" * 40)

    # Override max iterations to run one at a time
    cgn.config.max_iterations = 1

    for iteration in range(n_iterations):
        # Run one iteration using the optimize method
        cgn.optimize()

        # Reset for next iteration
        cgn.iteration = 0
        cgn.converged = False

        # Calculate metrics for each unknown node
        for i in range(n_anchors, n_total):
            # Timing error (bias in ns)
            true_bias_ns = clock_states[i].bias * 1e9
            est_bias_ns = cgn.nodes[i].state[2]
            timing_error = abs(est_bias_ns - true_bias_ns)
            timing_histories[i].append(timing_error)

            # Frequency error (CFO in ppm)
            true_cfo_ppm = clock_states[i].cfo
            est_cfo_ppm = cgn.nodes[i].state[4]
            frequency_error = abs(est_cfo_ppm - true_cfo_ppm)
            frequency_histories[i].append(frequency_error)

            # Position error (meters)
            true_pos = true_positions[i]
            est_pos = cgn.nodes[i].state[:2]
            position_error = np.linalg.norm(est_pos - true_pos)
            position_histories[i].append(position_error)

        # Print progress
        if iteration % 20 == 0 or iteration == n_iterations - 1:
            avg_timing = np.mean([timing_histories[i][-1] for i in range(n_anchors, n_total)])
            avg_freq = np.mean([frequency_histories[i][-1] for i in range(n_anchors, n_total)])
            avg_pos = np.mean([position_histories[i][-1] for i in range(n_anchors, n_total)])
            print(f"Iteration {iteration+1:3d}: "
                  f"Avg Timing={avg_timing:.3f} ns, "
                  f"Avg Freq={avg_freq:.3f} ppm, "
                  f"Avg Pos={avg_pos*100:.2f} cm")

    # Print final per-node results
    print("\n" + "=" * 70)
    print("FINAL PER-NODE RESULTS")
    print("=" * 70)
    print(f"{'Node':<6} {'Timing (ns)':<12} {'Freq (ppm)':<12} {'Position (cm)':<12}")
    print("-" * 45)

    for i in range(n_anchors, min(n_anchors + 10, n_total)):  # Show first 10 unknown nodes
        print(f"{i:<6} {timing_histories[i][-1]:<12.4f} "
              f"{frequency_histories[i][-1]:<12.4f} "
              f"{position_histories[i][-1]*100:<12.3f}")

    if n_unknown > 10:
        print(f"... ({n_unknown - 10} more nodes)")

    # Create three separate figures with per-node traces
    iterations = np.arange(1, n_iterations + 1)

    # Create colormap for nodes
    colors = cm.viridis(np.linspace(0.2, 0.9, n_unknown))

    # Figure 1: Per-Node Timing Offset Convergence
    plt.figure(figsize=(12, 7))
    for idx, i in enumerate(range(n_anchors, n_total)):
        plt.semilogy(iterations, timing_histories[i],
                    color=colors[idx], alpha=0.6, linewidth=1)

    # Add average line
    avg_timing = [np.mean([timing_histories[i][j] for i in range(n_anchors, n_total)])
                  for j in range(n_iterations)]
    plt.semilogy(iterations, avg_timing, 'k-', linewidth=3,
                label=f'Average (final: {avg_timing[-1]:.3f} ns)')

    plt.axhline(y=0.037, color='r', linestyle='--', alpha=0.5,
                label='Target: 0.037 ns')
    plt.axhline(y=0.289, color='gray', linestyle=':', alpha=0.5,
                label='Single-sample limit: 0.289 ns')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Timing Offset Error (ns)', fontsize=12)
    plt.title('Per-Node Timing Offset Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.01, 1])

    plt.tight_layout()
    plt.savefig('pernode_timing_convergence.png', dpi=100)
    print("\nSaved: pernode_timing_convergence.png")

    # Figure 2: Per-Node Frequency Offset Convergence
    plt.figure(figsize=(12, 7))
    for idx, i in enumerate(range(n_anchors, n_total)):
        plt.semilogy(iterations, frequency_histories[i],
                    color=colors[idx], alpha=0.6, linewidth=1)

    # Add average line
    avg_freq = [np.mean([frequency_histories[i][j] for i in range(n_anchors, n_total)])
                for j in range(n_iterations)]
    plt.semilogy(iterations, avg_freq, 'k-', linewidth=3,
                label=f'Average (final: {avg_freq[-1]:.3f} ppm)')

    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='TCXO: ~10 ppm')
    plt.axhline(y=1, color='red', linestyle=':', alpha=0.5, label='OCXO: ~1 ppm')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Frequency Offset Error (ppm)', fontsize=12)
    plt.title('Per-Node Frequency Offset (CFO) Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.001, 100])

    plt.tight_layout()
    plt.savefig('pernode_frequency_convergence.png', dpi=100)
    print("Saved: pernode_frequency_convergence.png")

    # Figure 3: Per-Node Localization RMSE Convergence
    plt.figure(figsize=(12, 7))
    for idx, i in enumerate(range(n_anchors, n_total)):
        position_cm = [x * 100 for x in position_histories[i]]
        plt.semilogy(iterations, position_cm,
                    color=colors[idx], alpha=0.6, linewidth=1)

    # Add average line
    avg_pos_cm = [np.mean([position_histories[i][j] * 100 for i in range(n_anchors, n_total)])
                  for j in range(n_iterations)]
    plt.semilogy(iterations, avg_pos_cm, 'k-', linewidth=3,
                label=f'Average (final: {avg_pos_cm[-1]:.2f} cm)')

    plt.axhline(y=2.66, color='b', linestyle='--', alpha=0.5,
                label='Target: 2.66 cm')
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5,
                label='Sub-cm target: 1 cm')
    plt.axhline(y=8.67, color='gray', linestyle=':', alpha=0.5,
                label='Quantization limit: 8.67 cm')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Position Error (cm)', fontsize=12)
    plt.title('Per-Node Localization Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim([0.1, 20])

    plt.tight_layout()
    plt.savefig('pernode_position_convergence.png', dpi=100)
    print("Saved: pernode_position_convergence.png")

    # Create combined figure showing all nodes' final performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Final Per-Node Performance (All Unknown Nodes)', fontsize=14, fontweight='bold')

    # Final timing distribution
    ax = axes[0]
    final_timing = [timing_histories[i][-1] for i in range(n_anchors, n_total)]
    ax.hist(final_timing, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(final_timing), color='r', linestyle='--',
               label=f'Mean: {np.mean(final_timing):.3f} ns')
    ax.set_xlabel('Timing Error (ns)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Timing Offset Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final frequency distribution
    ax = axes[1]
    final_freq = [frequency_histories[i][-1] for i in range(n_anchors, n_total)]
    ax.hist(final_freq, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(final_freq), color='r', linestyle='--',
               label=f'Mean: {np.mean(final_freq):.3f} ppm')
    ax.set_xlabel('Frequency Error (ppm)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Frequency Offset Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final position distribution
    ax = axes[2]
    final_pos_cm = [position_histories[i][-1] * 100 for i in range(n_anchors, n_total)]
    ax.hist(final_pos_cm, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(final_pos_cm), color='b', linestyle='--',
               label=f'Mean: {np.mean(final_pos_cm):.2f} cm')
    ax.set_xlabel('Position Error (cm)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Localization Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pernode_final_distributions.png', dpi=100)
    print("Saved: pernode_final_distributions.png")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated figures:")
    print("1. pernode_timing_convergence.png     - Individual node timing convergence")
    print("2. pernode_frequency_convergence.png  - Individual node frequency convergence")
    print("3. pernode_position_convergence.png   - Individual node position convergence")
    print("4. pernode_final_distributions.png    - Final error distributions across nodes")

    # Calculate and print statistics
    print("\n" + "=" * 70)
    print("STATISTICS ACROSS ALL UNKNOWN NODES")
    print("=" * 70)

    print(f"\nTiming Offset (ns):")
    print(f"  Mean:  {np.mean(final_timing):.4f}")
    print(f"  Std:   {np.std(final_timing):.4f}")
    print(f"  Min:   {np.min(final_timing):.4f}")
    print(f"  Max:   {np.max(final_timing):.4f}")

    print(f"\nFrequency Offset (ppm):")
    print(f"  Mean:  {np.mean(final_freq):.4f}")
    print(f"  Std:   {np.std(final_freq):.4f}")
    print(f"  Min:   {np.min(final_freq):.4f}")
    print(f"  Max:   {np.max(final_freq):.4f}")

    print(f"\nPosition Error (cm):")
    print(f"  Mean:  {np.mean(final_pos_cm):.3f}")
    print(f"  Std:   {np.std(final_pos_cm):.3f}")
    print(f"  Min:   {np.min(final_pos_cm):.3f}")
    print(f"  Max:   {np.max(final_pos_cm):.3f}")

    return timing_histories, frequency_histories, position_histories

if __name__ == "__main__":
    timing, frequency, position = run_pernode_convergence_analysis()