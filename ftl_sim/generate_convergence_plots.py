#!/usr/bin/env python3
"""
Generate separate convergence plots for time offset, frequency offset, and position RMSE
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters, ClockPriorFactor

def run_consensus_with_history(config_path='configs/ideal_30node.yaml', seed=42):
    """Run consensus and collect full state history"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(seed)

    # Setup network
    n_nodes = 30
    n_anchors = 5
    n_unknowns = 25

    # Fixed anchor positions
    anchor_positions = np.array([
        [0.0, 0.0],
        [50.0, 0.0],
        [50.0, 50.0],
        [0.0, 50.0],
        [25.0, 25.0]
    ])

    # Generate grid positions for unknowns
    x_pos = np.linspace(5, 45, 5)
    y_pos = np.linspace(5, 45, 5)

    unknown_positions = []
    for x in x_pos:
        for y in y_pos:
            unknown_positions.append([x, y])
    unknown_positions = np.array(unknown_positions[:n_unknowns])

    true_positions = np.vstack([anchor_positions, unknown_positions])

    # Create consensus solver
    cgn_config = ConsensusGNConfig(
        max_iterations=300,  # Enough to see convergence
        consensus_gain=0.05,
        step_size=0.3,
        gradient_tol=1e-3,
        step_tol=1e-4,
        verbose=False
    )

    cgn = ConsensusGaussNewton(cgn_config)

    # Add nodes with initial states (following ideal_30node.yaml)
    initial_states = []
    for i in range(n_nodes):
        state = np.zeros(5)
        if i < n_anchors:
            # Anchors: perfect position, very small clock errors (ideal conditions)
            state[:2] = true_positions[i]
            state[2] = np.random.normal(0, 1e-9)  # 1 ns std (from yaml: initial_bias_std)
            state[3] = np.random.normal(0, 1e-3)  # 1 ppb std (drift_ppm=1ppm -> ppb scale)
            state[4] = np.random.normal(0, 1e-3)  # ~1 ppb for CFO
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Unknowns: position noise from yaml, small clock errors for ideal conditions
            initial_pos = true_positions[i] + np.random.normal(0, 0.5, 2)  # 50cm std from yaml
            state[:2] = initial_pos
            state[2] = np.random.normal(0, 1e-9)  # 1 ns std (ideal conditions)
            state[3] = np.random.normal(0, 1e-3)  # ~1 ppb
            state[4] = np.random.normal(0, 1e-3)  # ~1 ppb
            cgn.add_node(i, state, is_anchor=False)
        initial_states.append(state.copy())

    # Add measurements
    comm_range = 25.0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)
                meas_range = dist + np.random.normal(0, 0.01)  # 1cm noise
                cgn.add_measurement(ToAFactorMeters(i, j, meas_range, 0.01**2))

    # Set true positions for evaluation
    cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

    # Collect history during optimization
    history = {
        'positions': [],
        'clock_bias': [],
        'clock_drift': [],
        'cfo': [],
        'rmse': []
    }

    print("Running consensus optimization...")

    # Manual iteration loop to collect history
    for iteration in range(cgn_config.max_iterations):
        # Exchange states
        cgn._exchange_states()

        # Update each node
        for node in cgn.nodes.values():
            node.update_state()

        # Collect current states
        iter_positions = []
        iter_bias = []
        iter_drift = []
        iter_cfo = []

        for i in range(n_nodes):
            node_state = cgn.nodes[i].state
            iter_positions.append(node_state[:2].copy())
            iter_bias.append(node_state[2])
            iter_drift.append(node_state[3])
            iter_cfo.append(node_state[4])

        history['positions'].append(np.array(iter_positions))
        history['clock_bias'].append(np.array(iter_bias))
        history['clock_drift'].append(np.array(iter_drift))
        history['cfo'].append(np.array(iter_cfo))

        # Calculate RMSE for unknowns
        unknown_positions_est = np.array([cgn.nodes[i].state[:2] for i in range(n_anchors, n_nodes)])
        errors = np.linalg.norm(unknown_positions_est - true_positions[n_anchors:], axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        history['rmse'].append(rmse * 100)  # Convert to cm

        if iteration % 50 == 0:
            print(f"  Iteration {iteration}: RMSE = {rmse*100:.2f} cm")

        # Check for convergence
        converged, _ = cgn._check_global_convergence()
        if converged:
            print(f"Converged at iteration {iteration}")
            break

    print(f"Final RMSE: {history['rmse'][-1]:.2f} cm")

    return history, n_anchors, n_nodes, initial_states


def plot_time_offset_convergence(history, n_anchors, n_nodes, initial_states):
    """Plot clock bias convergence"""

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(len(history['clock_bias']))
    bias_history = np.array(history['clock_bias'])  # Shape: (iterations, n_nodes)

    # Plot unknown nodes
    for i in range(n_anchors, n_nodes):
        node_bias = bias_history[:, i]
        ax.plot(iterations, node_bias, alpha=0.5, linewidth=0.8,
                label=f'Node {i}' if i < n_anchors + 3 else None)

    # Add average line for unknowns
    avg_unknown_bias = np.mean(bias_history[:, n_anchors:], axis=1)
    ax.plot(iterations, avg_unknown_bias, 'k-', linewidth=2,
            label='Average (unknowns)')

    # Mark anchor biases
    for i in range(n_anchors):
        ax.axhline(y=initial_states[i][2], color='red', linestyle='--',
                   alpha=0.3, linewidth=0.5)

    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target (0)')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Clock Bias (nanoseconds)', fontsize=12)
    ax.set_title('Clock Bias Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add text box with final statistics
    final_bias_std = np.std(bias_history[-1, n_anchors:])
    final_bias_mean = np.mean(bias_history[-1, n_anchors:])
    textstr = f'Final Stats (unknowns):\nMean: {final_bias_mean:.2f} ns\nStd: {final_bias_std:.2f} ns'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_frequency_offset_convergence(history, n_anchors, n_nodes, initial_states):
    """Plot CFO convergence"""

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(len(history['cfo']))
    cfo_history = np.array(history['cfo'])  # Shape: (iterations, n_nodes)

    # Plot unknown nodes
    for i in range(n_anchors, n_nodes):
        node_cfo = cfo_history[:, i]
        ax.plot(iterations, node_cfo, alpha=0.5, linewidth=0.8,
                label=f'Node {i}' if i < n_anchors + 3 else None)

    # Add average line for unknowns
    avg_unknown_cfo = np.mean(cfo_history[:, n_anchors:], axis=1)
    ax.plot(iterations, avg_unknown_cfo, 'k-', linewidth=2,
            label='Average (unknowns)')

    # Mark anchor CFOs
    for i in range(n_anchors):
        ax.axhline(y=initial_states[i][4], color='red', linestyle='--',
                   alpha=0.3, linewidth=0.5)

    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target (0)')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Carrier Frequency Offset (ppm)', fontsize=12)
    ax.set_title('Frequency Offset Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add text box with final statistics
    final_cfo_std = np.std(cfo_history[-1, n_anchors:])
    final_cfo_mean = np.mean(cfo_history[-1, n_anchors:])
    textstr = f'Final Stats (unknowns):\nMean: {final_cfo_mean:.3f} ppm\nStd: {final_cfo_std:.3f} ppm'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_position_rmse_convergence(history):
    """Plot position RMSE convergence"""

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(len(history['rmse']))
    rmse = history['rmse']

    ax.plot(iterations, rmse, 'b-', linewidth=2, label='Position RMSE')

    # Mark key thresholds
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
               label='1 cm threshold')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5,
               label='2 cm threshold')

    # Find when RMSE first drops below 2cm
    below_2cm = None
    for i, r in enumerate(rmse):
        if r < 2.0:
            below_2cm = i
            break

    if below_2cm:
        ax.axvline(x=below_2cm, color='gray', linestyle=':', alpha=0.5)
        ax.text(below_2cm + 2, max(rmse) * 0.8,
                f'< 2cm at iter {below_2cm}', fontsize=10)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Position RMSE (cm)', fontsize=12)
    ax.set_title('Localization RMSE Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Use log scale if range is large
    if max(rmse) / min(rmse) > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Position RMSE (cm) - Log Scale', fontsize=12)

    # Add text box with final statistics
    final_rmse = rmse[-1]
    min_rmse = min(rmse)
    textstr = f'Final RMSE: {final_rmse:.2f} cm\nBest RMSE: {min_rmse:.2f} cm\nIterations: {len(rmse)}'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    """Generate all three convergence plots"""

    print("=" * 70)
    print("Generating Convergence Plots for FTL System")
    print("=" * 70)

    # Run consensus and collect history
    history, n_anchors, n_nodes, initial_states = run_consensus_with_history()

    print("\nGenerating plots...")

    # Generate time offset plot
    fig1 = plot_time_offset_convergence(history, n_anchors, n_nodes, initial_states)
    fig1.savefig('time_offset_convergence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved time_offset_convergence.png")

    # Generate frequency offset plot
    fig2 = plot_frequency_offset_convergence(history, n_anchors, n_nodes, initial_states)
    fig2.savefig('frequency_offset_convergence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved frequency_offset_convergence.png")

    # Generate position RMSE plot
    fig3 = plot_position_rmse_convergence(history)
    fig3.savefig('position_rmse_convergence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved position_rmse_convergence.png")

    # Show plots
    plt.show()

    print("\n" + "=" * 70)
    print("Complete! Three convergence plots have been generated:")
    print("  1. time_offset_convergence.png - Clock bias evolution")
    print("  2. frequency_offset_convergence.png - CFO evolution")
    print("  3. position_rmse_convergence.png - Localization accuracy")
    print("=" * 70)


if __name__ == "__main__":
    main()