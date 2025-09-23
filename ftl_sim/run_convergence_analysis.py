#!/usr/bin/env python3
"""
Run convergence analysis and generate three separate figures:
1. Timing offset convergence
2. Frequency offset convergence
3. Localization RMSE convergence
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

def run_convergence_analysis():
    """Run simulation and track convergence of all three metrics"""

    print("=" * 70)
    print("CONVERGENCE ANALYSIS - UNIFIED FTL SYSTEM")
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

    # Track metrics over iterations - we'll run optimize() and capture intermediate states
    n_iterations = 100
    timing_rmse_history = []
    frequency_rmse_history = []
    position_rmse_history = []

    print(f"\nRunning {n_iterations} consensus iterations...")
    print("-" * 40)

    # Override the max iterations to run one at a time
    cgn.config.max_iterations = 1

    for iteration in range(n_iterations):
        # Store previous states to check convergence
        prev_states = {node_id: node.state.copy() for node_id, node in cgn.nodes.items()}

        # Run one iteration using the optimize method
        cgn.optimize()

        # Reset for next iteration
        cgn.iteration = 0
        cgn.converged = False

        # Calculate metrics for unknown nodes
        timing_errors = []
        frequency_errors = []
        position_errors = []

        for i in range(n_anchors, n_total):
            # Timing error (bias in ns)
            true_bias_ns = clock_states[i].bias * 1e9
            est_bias_ns = cgn.nodes[i].state[2]
            timing_errors.append(est_bias_ns - true_bias_ns)

            # Frequency error (CFO in ppm)
            true_cfo_ppm = clock_states[i].cfo
            est_cfo_ppm = cgn.nodes[i].state[4]
            frequency_errors.append(est_cfo_ppm - true_cfo_ppm)

            # Position error (meters)
            true_pos = true_positions[i]
            est_pos = cgn.nodes[i].state[:2]
            position_error = np.linalg.norm(est_pos - true_pos)
            position_errors.append(position_error)

        # Calculate RMS errors
        timing_rmse = np.sqrt(np.mean(np.array(timing_errors)**2))
        frequency_rmse = np.sqrt(np.mean(np.array(frequency_errors)**2))
        position_rmse = np.sqrt(np.mean(np.array(position_errors)**2))

        timing_rmse_history.append(timing_rmse)
        frequency_rmse_history.append(frequency_rmse)
        position_rmse_history.append(position_rmse)

        # Print progress
        if iteration % 10 == 0 or iteration == n_iterations - 1:
            print(f"Iteration {iteration+1:3d}: "
                  f"Timing={timing_rmse:.3f} ns, "
                  f"Freq={frequency_rmse:.3f} ppm, "
                  f"Pos={position_rmse*100:.2f} cm")

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Timing RMSE:    {timing_rmse_history[-1]:.3f} ns")
    print(f"Frequency RMSE: {frequency_rmse_history[-1]:.3f} ppm")
    print(f"Position RMSE:  {position_rmse_history[-1]*100:.2f} cm")

    # Create three separate figures
    iterations = np.arange(1, n_iterations + 1)

    # Figure 1: Timing Offset Convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, timing_rmse_history, 'b-', linewidth=2)
    plt.axhline(y=0.037, color='r', linestyle='--', alpha=0.5,
                label='Target: 0.037 ns')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Timing Offset RMSE (ns)', fontsize=12)
    plt.title('Timing Offset Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations
    plt.axhline(y=0.289, color='gray', linestyle=':', alpha=0.5,
                label='Single-sample limit: 0.289 ns')
    final_timing = timing_rmse_history[-1]
    plt.text(n_iterations * 0.7, final_timing * 1.5,
             f'Final: {final_timing:.3f} ns\n({0.289/final_timing:.1f}× better than limit)',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('timing_convergence.png', dpi=100)
    print("\nSaved: timing_convergence.png")

    # Figure 2: Frequency Offset Convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, frequency_rmse_history, 'g-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Frequency Offset RMSE (ppm)', fontsize=12)
    plt.title('Frequency Offset (CFO) Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add annotations
    final_freq = frequency_rmse_history[-1]
    initial_freq = frequency_rmse_history[0]
    plt.text(n_iterations * 0.7, final_freq * 10,
             f'Final: {final_freq:.3f} ppm\n'
             f'Improvement: {initial_freq/final_freq:.1f}×',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add reference lines for clock types
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='TCXO: ~10 ppm')
    plt.axhline(y=1, color='red', linestyle=':', alpha=0.5, label='OCXO: ~1 ppm')
    plt.legend()

    plt.tight_layout()
    plt.savefig('frequency_convergence.png', dpi=100)
    print("Saved: frequency_convergence.png")

    # Figure 3: Localization RMSE Convergence
    plt.figure(figsize=(10, 6))
    position_rmse_cm = [x * 100 for x in position_rmse_history]
    plt.semilogy(iterations, position_rmse_cm, 'r-', linewidth=2)
    plt.axhline(y=2.66, color='b', linestyle='--', alpha=0.5,
                label='Target: 2.66 cm')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Position RMSE (cm)', fontsize=12)
    plt.title('Localization RMSE Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations
    final_pos_cm = position_rmse_cm[-1]
    initial_pos_cm = position_rmse_cm[0]

    # Add reference lines
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5,
                label='Sub-cm target: 1 cm')
    plt.axhline(y=8.67, color='gray', linestyle=':', alpha=0.5,
                label='Quantization limit: 8.67 cm')

    plt.text(n_iterations * 0.7, final_pos_cm * 2,
             f'Final: {final_pos_cm:.2f} cm\n'
             f'Improvement: {initial_pos_cm/final_pos_cm:.1f}×\n'
             f'vs Quant limit: {8.67/final_pos_cm:.1f}× better',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.legend(loc='upper right')
    plt.ylim([1, max(position_rmse_cm) * 1.5])

    plt.tight_layout()
    plt.savefig('position_convergence.png', dpi=100)
    print("Saved: position_convergence.png")

    # Create combined figure for comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Unified FTL System - Convergence Analysis', fontsize=14, fontweight='bold')

    # Timing subplot
    ax = axes[0]
    ax.semilogy(iterations, timing_rmse_history, 'b-', linewidth=2)
    ax.axhline(y=0.037, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Timing RMSE (ns)')
    ax.set_title(f'Timing Offset\nFinal: {timing_rmse_history[-1]:.3f} ns')
    ax.grid(True, alpha=0.3)

    # Frequency subplot
    ax = axes[1]
    ax.semilogy(iterations, frequency_rmse_history, 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frequency RMSE (ppm)')
    ax.set_title(f'Frequency Offset\nFinal: {frequency_rmse_history[-1]:.3f} ppm')
    ax.grid(True, alpha=0.3)

    # Position subplot
    ax = axes[2]
    ax.semilogy(iterations, position_rmse_cm, 'r-', linewidth=2)
    ax.axhline(y=2.66, color='b', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position RMSE (cm)')
    ax.set_title(f'Localization\nFinal: {position_rmse_cm[-1]:.2f} cm')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_combined.png', dpi=100)
    print("Saved: convergence_combined.png")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated figures:")
    print("1. timing_convergence.png    - Timing offset RMSE over iterations")
    print("2. frequency_convergence.png - Frequency offset RMSE over iterations")
    print("3. position_convergence.png  - Localization RMSE over iterations")
    print("4. convergence_combined.png  - All three metrics in one figure")

    return timing_rmse_history, frequency_rmse_history, position_rmse_history

if __name__ == "__main__":
    timing, frequency, position = run_convergence_analysis()