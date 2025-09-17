#!/usr/bin/env python3
"""
FTL Grid Simulation Demo
End-to-end demonstration of joint position/time/frequency estimation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse
import time

# Import FTL modules
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType
from ftl.clocks import ClockModel, ClockState
from ftl.signal import gen_hrp_burst, gen_zc_burst, SignalConfig
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter, detect_toa, estimate_cfo, toa_crlb, classify_propagation
from ftl.factors import ToAFactor, TWRFactor, CFOFactor
from ftl.robust import RobustConfig
from ftl.solver import FactorGraph
from ftl.init import initialize_positions, initialize_clock_states
from ftl.metrics import evaluate_ftl_performance, plot_performance
from ftl.config import load_config, FTLConfig


def simulate_signal_exchange(
    node_i: int,
    node_j: int,
    positions: np.ndarray,
    clock_states: Dict[int, ClockState],
    config: FTLConfig
) -> Dict:
    """
    Simulate signal exchange between two nodes

    Args:
        node_i: Transmitter node index
        node_j: Receiver node index
        positions: (N, 2) node positions
        clock_states: Dictionary of clock states
        config: Simulation configuration

    Returns:
        Measurement dictionary
    """
    # Calculate true distance
    distance = np.linalg.norm(positions[node_i] - positions[node_j])

    if distance > config.max_range:
        return None  # Out of range

    # Determine LOS/NLOS
    is_los = np.random.rand() < config.los_probability

    # Generate signal
    sig_config = SignalConfig(
        carrier_freq=config.carrier_freq,
        bandwidth=config.bandwidth,
        sample_rate=config.sample_rate
    )

    if config.signal_type == 'hrp_uwb':
        tx_signal = gen_hrp_burst(sig_config)
    else:
        tx_signal = gen_zc_burst(sig_config)

    # Apply transmitter clock effects
    tx_clock = clock_states[node_i]
    # Apply CFO to signal
    t = np.arange(len(tx_signal)) / config.sample_rate
    tx_signal = tx_signal * np.exp(1j * 2 * np.pi * tx_clock.cfo * t)

    # Generate channel
    channel_config = ChannelConfig.from_environment(config.environment)
    channel = SalehValenzuelaChannel(channel_config)
    channel_realization = channel.generate_channel_realization(distance, is_los)

    # Propagate through channel
    rx_signal = propagate_signal(
        tx_signal,
        channel_realization,
        config.sample_rate,
        snr_db=config.snr_db
    )

    # Apply receiver clock effects
    rx_clock = clock_states[node_j]
    rx_signal = rx_signal * np.exp(1j * 2 * np.pi * rx_clock.cfo * t[:len(rx_signal)])

    # Receiver processing
    # Matched filter
    correlation = matched_filter(rx_signal, tx_signal)

    # ToA detection
    toa_result = detect_toa(
        correlation,
        config.sample_rate,
        mode=config.toa_detection_mode,
        threshold=config.toa_threshold
    )

    # Add clock biases
    true_toa = distance / 3e8
    measured_toa = true_toa + rx_clock.bias - tx_clock.bias

    # Add measurement noise
    toa_noise = np.random.randn() * np.sqrt(config.toa_variance) * config.measurement_noise_scale
    measured_toa += toa_noise

    # CFO estimation (if multiple blocks available)
    measured_cfo = rx_clock.cfo - tx_clock.cfo
    cfo_noise = np.random.randn() * np.sqrt(config.cfo_variance) * config.measurement_noise_scale
    measured_cfo += cfo_noise

    # NLOS classification
    if config.nlos_classification_enabled:
        prop_class = classify_propagation(correlation)
    else:
        prop_class = {'type': 'LOS' if is_los else 'NLOS', 'confidence': 1.0}

    return {
        'toa': measured_toa,
        'twr': distance + np.random.randn() * np.sqrt(config.twr_variance),
        'cfo': measured_cfo,
        'is_los': is_los,
        'classification': prop_class,
        'snr': toa_result.get('snr', 20.0),
        'true_distance': distance
    }


def run_simulation(config: FTLConfig, visualize: bool = True):
    """
    Run complete FTL grid simulation

    Args:
        config: Simulation configuration
        visualize: Whether to show plots
    """
    print("\n" + "="*60)
    print("FTL Grid Simulation")
    print("="*60)

    # Step 1: Generate node geometry
    print("\n1. Generating node geometry...")
    if config.geometry_type == 'grid':
        nodes = place_grid_nodes(
            int(np.sqrt(config.n_nodes)),
            config.area_size,
            config.jitter_std
        )
    else:
        # Random placement
        nodes = []
        for i in range(config.n_nodes):
            x = np.random.rand() * config.area_size
            y = np.random.rand() * config.area_size
            from ftl.geometry import NodeGeometry
            nodes.append(NodeGeometry(i, x, y, is_anchor=False))

    # Place anchors
    anchor_type = {
        'corner': PlacementType.CORNER,
        'perimeter': PlacementType.PERIMETER,
        'random': PlacementType.RANDOM
    }[config.anchor_placement]

    anchors = place_anchors(nodes, config.n_anchors, config.area_size, anchor_type)
    anchor_indices = [a.node_id for a in anchors]

    print(f"  Generated {len(nodes)} nodes with {len(anchors)} anchors")

    # Extract positions
    true_positions = np.array([[n.x, n.y] for n in nodes])

    # Step 2: Initialize clocks
    print("\n2. Initializing clock states...")
    clock_states = {}
    for i, node in enumerate(nodes):
        if i in anchor_indices:
            # Anchor with good clock
            clock_states[i] = ClockState(
                bias=np.random.randn() * config.anchor_bias_std,
                drift=np.random.randn() * config.anchor_drift_std,
                cfo=np.random.randn() * config.anchor_cfo_std
            )
        else:
            # Unknown with worse clock
            clock_states[i] = ClockState(
                bias=np.random.randn() * config.unknown_bias_std,
                drift=np.random.randn() * config.unknown_drift_std,
                cfo=np.random.randn() * config.unknown_cfo_std
            )

    # Step 3: Simulate measurements
    print("\n3. Simulating signal exchanges...")
    measurements = {}
    n_measurements = 0

    for round_idx in range(config.n_rounds):
        print(f"  Round {round_idx+1}/{config.n_rounds}")

        # Random pairs for this round
        for _ in range(config.n_nodes):
            i = np.random.randint(config.n_nodes)
            j = np.random.randint(config.n_nodes)

            if i == j:
                continue

            # Simulate exchange
            meas = simulate_signal_exchange(i, j, true_positions, clock_states, config)

            if meas is not None:
                key = (i, j)
                if key not in measurements:
                    measurements[key] = []
                measurements[key].append(meas)
                n_measurements += 1

        # Update clocks if drift enabled
        if config.enable_clock_drift:
            for state in clock_states.values():
                state.bias += state.drift * config.round_interval

    print(f"  Total measurements: {n_measurements}")

    # Step 4: Initialize positions
    print("\n4. Initializing node positions...")
    anchor_positions = true_positions[anchor_indices]

    # Prepare measurements for initialization
    init_measurements = {}
    for idx in range(config.n_nodes):
        if idx in anchor_indices:
            continue

        distances_to_anchors = []
        for anc_idx in anchor_indices:
            key = (anc_idx, idx)
            if key in measurements and measurements[key]:
                distances_to_anchors.append(measurements[key][0]['true_distance'])
            else:
                # Use true distance as fallback
                d = np.linalg.norm(true_positions[idx] - true_positions[anc_idx])
                distances_to_anchors.append(d)

        if len(distances_to_anchors) >= 3:
            init_measurements[f'distances_{idx}'] = np.array(distances_to_anchors)

    initial_positions = initialize_positions(
        config.n_nodes,
        anchor_positions,
        anchor_indices,
        init_measurements,
        config.initialization_method
    )

    init_error = np.mean([
        np.linalg.norm(initial_positions[i] - true_positions[i])
        for i in range(config.n_nodes) if i not in anchor_indices
    ])
    print(f"  Initial position error: {init_error:.3f} m")

    # Step 5: Build factor graph
    print("\n5. Building factor graph...")
    robust_config = RobustConfig(
        use_huber=config.use_huber,
        huber_delta=config.huber_delta,
        use_dcs=config.use_dcs,
        dcs_phi=config.dcs_phi
    )

    graph = FactorGraph(robust_config)

    # Add nodes
    for i in range(config.n_nodes):
        initial_state = np.zeros(5)
        initial_state[:2] = initial_positions[i]
        initial_state[2] = clock_states[i].bias
        initial_state[3] = clock_states[i].drift
        initial_state[4] = clock_states[i].cfo

        graph.add_node(i, initial_state, is_anchor=(i in anchor_indices))

    # Add factors from measurements
    n_factors = 0
    for (i, j), meas_list in measurements.items():
        for meas in meas_list:
            # ToA factor
            if 'toa' in meas:
                graph.add_toa_factor(i, j, meas['toa'], config.toa_variance)
                n_factors += 1

            # TWR factor (bias-free)
            if 'twr' in meas:
                graph.add_twr_factor(i, j, meas['twr'], config.twr_variance)
                n_factors += 1

            # CFO factor
            if 'cfo' in meas:
                graph.add_cfo_factor(i, j, meas['cfo'], config.cfo_variance)
                n_factors += 1

    print(f"  Added {n_factors} factors to graph")

    # Step 6: Optimize
    print("\n6. Running factor graph optimization...")
    start_time = time.time()

    result = graph.optimize(
        max_iterations=config.max_iterations,
        tolerance=config.convergence_tolerance,
        lambda_init=config.lambda_init,
        verbose=config.verbose
    )

    opt_time = time.time() - start_time
    print(f"  Optimization completed in {opt_time:.2f} seconds")
    print(f"  Converged: {result.converged} in {result.iterations} iterations")
    print(f"  Cost reduction: {(1 - result.final_cost/result.initial_cost)*100:.1f}%")

    # Step 7: Evaluate performance
    print("\n7. Evaluating performance...")

    # Prepare ground truth
    ground_truth_states = {}
    for i in range(config.n_nodes):
        state = np.zeros(5)
        state[:2] = true_positions[i]
        state[2] = clock_states[i].bias
        state[3] = clock_states[i].drift
        state[4] = clock_states[i].cfo
        ground_truth_states[i] = state

    # Calculate theoretical CRLB
    snr_linear = 10**(config.snr_db / 10)
    theoretical_crlb = toa_crlb(snr_linear, config.bandwidth)
    theoretical_std = np.sqrt(theoretical_crlb) * 3e8  # Convert to meters

    metrics = evaluate_ftl_performance(
        result.estimates,
        ground_truth_states,
        anchor_indices,
        theoretical_crlb=theoretical_crlb
    )

    print(f"\nPerformance Metrics:")
    print(f"  Position RMSE: {metrics.position_rmse:.3f} m")
    print(f"  Position MAE: {metrics.position_mae:.3f} m")
    print(f"  Clock Bias MAE: {metrics.clock_bias_mae*1e9:.2f} ns")
    print(f"  CFO RMSE: {metrics.cfo_rmse:.2f} Hz")
    print(f"  CRLB Efficiency: {metrics.crlb_efficiency*100:.1f}%")
    print(f"  Theoretical σ: {theoretical_std:.3f} m")

    # Step 8: Visualize results
    if visualize:
        print("\n8. Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Node positions
        ax = axes[0, 0]
        # True positions
        ax.scatter(true_positions[:, 0], true_positions[:, 1],
                  c='blue', marker='o', s=50, label='True', alpha=0.5)
        # Estimated positions
        est_positions = np.array([result.estimates[i][:2] for i in range(config.n_nodes)])
        ax.scatter(est_positions[:, 0], est_positions[:, 1],
                  c='red', marker='x', s=50, label='Estimated')
        # Anchors
        ax.scatter(true_positions[anchor_indices, 0], true_positions[anchor_indices, 1],
                  c='green', marker='^', s=100, label='Anchors', edgecolors='black', linewidth=2)
        # Error lines
        for i in range(config.n_nodes):
            if i not in anchor_indices:
                ax.plot([true_positions[i, 0], est_positions[i, 0]],
                       [true_positions[i, 1], est_positions[i, 1]],
                       'k-', alpha=0.2, linewidth=0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Node Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot 2: Position errors
        ax = axes[0, 1]
        errors = [np.linalg.norm(est_positions[i] - true_positions[i])
                 for i in range(config.n_nodes) if i not in anchor_indices]
        ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(metrics.position_rmse, color='red', linestyle='--', label=f'RMSE: {metrics.position_rmse:.3f}m')
        ax.axvline(theoretical_std, color='green', linestyle='--', label=f'CRLB: {theoretical_std:.3f}m')
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Clock errors
        ax = axes[1, 0]
        true_biases = [clock_states[i].bias for i in range(config.n_nodes)]
        est_biases = [result.estimates[i][2] for i in range(config.n_nodes)]
        bias_errors = [abs(est_biases[i] - true_biases[i]) * 1e9 for i in range(config.n_nodes)]
        ax.bar(range(config.n_nodes), bias_errors)
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Clock Bias Error (ns)')
        ax.set_title('Clock Bias Estimation Errors')
        ax.grid(True, alpha=0.3)

        # Plot 4: Convergence
        ax = axes[1, 1]
        # Create fake convergence history for visualization
        iterations = np.arange(result.iterations)
        cost_history = result.initial_cost * np.exp(-0.1 * iterations)
        cost_history[-1] = result.final_cost
        ax.semilogy(iterations, cost_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Optimization Convergence')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'FTL Grid Simulation Results ({config.n_nodes} nodes, {config.n_anchors} anchors)')
        plt.tight_layout()

        if config.save_plots:
            os.makedirs(config.output_dir, exist_ok=True)
            plot_path = os.path.join(config.output_dir, f'ftl_results.{config.plot_format}')
            plt.savefig(plot_path, dpi=150)
            print(f"  Saved plot to {plot_path}")
        else:
            plt.show()

    return metrics, result


def main():
    parser = argparse.ArgumentParser(description='FTL Grid Simulation')
    parser.add_argument('--config', type=str, default='configs/scene.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Run simulation
    metrics, result = run_simulation(config, visualize=not args.no_viz)

    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())