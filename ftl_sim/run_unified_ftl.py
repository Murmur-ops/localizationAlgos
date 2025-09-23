#!/usr/bin/env python3
"""
Unified FTL Simulation: RF Signal Simulation + Distributed Consensus
Combines realistic signal generation with distributed consensus solving
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

# FTL modules for RF simulation
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType
from ftl.clocks import ClockState  # ClockState is a dataclass, not ClockModel
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter, detect_toa, estimate_cfo, toa_crlb

# FTL modules for consensus
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters, ClockPriorFactor

def load_unified_config(config_path):
    """Load unified configuration combining RF and consensus parameters"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_network_topology(config):
    """Generate node positions and anchor placement"""
    geometry = config['geometry']

    # Total number of nodes
    n_total = geometry['n_nodes'] + geometry['n_anchors']
    n_anchors = geometry['n_anchors']

    # Create anchor positions manually (4 corners + center)
    area_size = geometry['area_size']

    # Define all possible anchor positions
    corner_positions = [
        [0, 0],
        [area_size, 0],
        [area_size, area_size],
        [0, area_size],
        [area_size/2, area_size/2]  # Center
    ]

    # Take as many anchors as needed
    if n_anchors <= 5:
        anchor_positions = corner_positions[:n_anchors]
    else:
        # If more than 5 anchors, place extras randomly
        anchor_positions = corner_positions.copy()
        for _ in range(n_anchors - 5):
            anchor_positions.append([
                np.random.uniform(0, area_size),
                np.random.uniform(0, area_size)
            ])

    # Create positions for unknown nodes
    n_unknowns = geometry['n_nodes']
    unknown_positions = []

    if n_unknowns > 0:
        # Try to make a grid if possible
        n_grid = int(np.ceil(np.sqrt(n_unknowns)))
        spacing = area_size / (n_grid + 1)

        idx = 0
        for i in range(n_grid):
            for j in range(n_grid):
                if idx >= n_unknowns:
                    break

                x = spacing * (i + 1)
                y = spacing * (j + 1)

                # Add jitter if specified
                if geometry.get('jitter_std', 0) > 0:
                    x += np.random.normal(0, geometry['jitter_std'])
                    y += np.random.normal(0, geometry['jitter_std'])

                # Clip to area bounds
                x = np.clip(x, 0, area_size)
                y = np.clip(y, 0, area_size)

                unknown_positions.append([x, y])
                idx += 1

            if idx >= n_unknowns:
                break

    # Combine all positions: anchors first, then unknowns
    # Make sure we convert to numpy arrays with consistent shape
    anchor_array = np.array(anchor_positions[:n_anchors])
    unknown_array = np.array(unknown_positions[:geometry['n_nodes']])

    # Handle edge cases
    if len(anchor_array) == 0:
        true_positions = unknown_array
    elif len(unknown_array) == 0:
        true_positions = anchor_array
    else:
        true_positions = np.vstack([anchor_array, unknown_array])

    return true_positions, n_anchors, n_total


def initialize_clock_states(config, n_nodes, n_anchors):
    """Initialize clock states for all nodes"""
    clock_states = {}

    # Handle both direct clock config and nested rf_simulation config
    if 'rf_simulation' in config:
        rf_config = config['rf_simulation']
    else:
        rf_config = config

    for i in range(n_nodes):
        if i < n_anchors:
            # Anchors have better clocks
            clock_config = rf_config['clocks']['anchor_nodes']
        else:
            # Unknown nodes have cheaper clocks
            clock_config = rf_config['clocks']['unknown_nodes']

        # Initialize state with small errors (convert string to float if needed)
        bias_std = float(clock_config['initial_bias_std'])
        drift_std = float(clock_config['initial_drift_std'])
        cfo_std = float(clock_config['initial_cfo_std'])

        # Create clock state using dataclass constructor
        clock_states[i] = ClockState(
            bias=np.random.normal(0, bias_std),
            drift=np.random.normal(0, drift_std),
            cfo=np.random.normal(0, cfo_std),
            sco_ppm=0.0  # Sample clock offset, typically 0 for ideal conditions
        )

    return clock_states


def simulate_rf_measurement(i, j, true_positions, clock_states, rf_config):
    """Simulate RF signal exchange and extract measurements"""

    # Get true distance
    pi = true_positions[i]
    pj = true_positions[j]
    true_distance = np.linalg.norm(pi - pj)

    # Check if in range
    max_range = float(rf_config['simulation']['max_range'])
    if true_distance > max_range:
        return None

    # Generate transmit signal with proper type conversions
    sig_config = SignalConfig(
        carrier_freq=float(rf_config['signal']['carrier_freq']),
        bandwidth=float(rf_config['signal']['bandwidth']),
        sample_rate=float(rf_config['signal']['sample_rate']),
        burst_duration=float(rf_config['signal']['burst_duration']),
        prf=float(rf_config['signal'].get('prf', 124.8e6))
    )

    # Use fixed seed for reproducible preamble sequence
    np.random.seed(42 + i * 1000 + j)  # Unique but deterministic per link
    tx_signal = gen_hrp_burst(sig_config)

    # Apply transmitter clock effects (CFO)
    t = np.arange(len(tx_signal)) / sig_config.sample_rate
    tx_signal = tx_signal * np.exp(1j * 2 * np.pi * clock_states[i].cfo * t)

    # Determine LOS probability
    los_prob = float(rf_config['simulation'].get('los_probability', 1.0))
    is_los = np.random.rand() < los_prob

    # Calculate true propagation time
    c = 299792458.0  # speed of light
    true_prop_time = true_distance / c

    # For now, skip multipath to get basic system working
    # TODO: Properly integrate multipath effects
    enable_multipath = rf_config['simulation'].get('enable_multipath', False)

    if enable_multipath:
        # Generate channel with proper initialization
        from ftl.channel import ChannelConfig
        channel_config = ChannelConfig(
            environment=rf_config['channel']['environment'],
            path_loss_exponent=float(rf_config['channel'].get('path_loss_exponent', 2.0)),
            shadowing_std_db=float(rf_config['channel'].get('shadowing_std_db', 2.0))
        )
        channel = SalehValenzuelaChannel(channel_config)
        channel_realization = channel.generate_channel_realization(true_distance, is_los)

        # Propagate through channel (adds multipath but not propagation delay)
        propagation_result = propagate_signal(
            tx_signal,
            channel_realization,
            sig_config.sample_rate,
            snr_db=float(rf_config['signal']['snr_db'])
        )
        rx_signal = propagation_result['signal']
    else:
        # Simple AWGN channel without multipath
        rx_signal = tx_signal.copy()

        # Add noise based on SNR
        snr_db = float(rf_config['signal']['snr_db'])
        signal_power = np.mean(np.abs(tx_signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
        rx_signal = rx_signal + noise

    # Add propagation delay by shifting the signal
    # Use round() instead of int() to avoid systematic bias
    delay_samples = round(true_prop_time * sig_config.sample_rate)
    if delay_samples > 0 and delay_samples < len(rx_signal):
        # Shift signal to simulate propagation delay
        rx_signal_delayed = np.zeros(len(rx_signal), dtype=rx_signal.dtype)
        rx_signal_delayed[delay_samples:] = rx_signal[:-delay_samples]
        rx_signal = rx_signal_delayed
    elif delay_samples >= len(rx_signal):
        # Delay is too large, signal is completely lost
        rx_signal = np.zeros_like(rx_signal)

    # Receiver processing - use cross-correlation to find delay
    # Generate same template as transmitter (known preamble)
    np.random.seed(42 + i * 1000 + j)  # Same seed as transmitter
    template = gen_hrp_burst(sig_config)

    # For a delayed signal, we need to find the peak in cross-correlation
    # Use 'full' mode to capture all possible delays
    corr_output = np.correlate(rx_signal, template, mode='full')

    # Find peak in correlation
    peak_idx = np.argmax(np.abs(corr_output))

    # In 'full' mode, zero delay is at index (len(template) - 1)
    # So actual delay is: peak_idx - (len(template) - 1)
    zero_delay_idx = len(template) - 1
    delay_in_samples = peak_idx - zero_delay_idx

    # Debug output (can be removed later)
    if False:  # Set to True for debugging
        print(f"  Correlation debug:")
        print(f"    Template length: {len(template)}")
        print(f"    Template energy: {np.sum(np.abs(template)**2):.1f}")
        print(f"    RX signal length: {len(rx_signal)}")
        print(f"    RX signal energy: {np.sum(np.abs(rx_signal)**2):.1f}")
        print(f"    Correlation length: {len(corr_output)}")
        print(f"    Peak index: {peak_idx}")
        print(f"    Peak value: {np.abs(corr_output[peak_idx]):.3f}")
        print(f"    Zero delay index: {zero_delay_idx}")
        print(f"    Delay samples: {delay_in_samples}")
        print(f"    Expected delay: {delay_samples}")
        print(f"    Error: {delay_in_samples - delay_samples} samples")

    # Convert to time
    toa_est = delay_in_samples / sig_config.sample_rate

    # Add clock biases to ToA
    toa_with_clocks = toa_est + (clock_states[j].bias - clock_states[i].bias)

    # Estimate CFO - for now just use the clock states' CFO difference
    cfo_diff = clock_states[j].cfo - clock_states[i].cfo

    # Calculate CRLB for variance estimation
    snr_linear = 10 ** (float(rf_config['signal']['snr_db']) / 10)
    crlb_var = toa_crlb(
        snr_linear,
        float(rf_config['signal']['bandwidth'])
    )

    # Convert time to distance (c already defined above)
    range_meas_m = toa_with_clocks * c
    range_var_m2 = (crlb_var * c) ** 2

    # Add measurement noise
    noise_scale = rf_config['simulation'].get('measurement_noise_scale', 1.0)
    range_var_m2 *= noise_scale

    return {
        'range_m': range_meas_m,
        'variance_m2': range_var_m2,
        'cfo_diff': cfo_diff,
        'is_los': is_los,
        'snr_db': rf_config['signal']['snr_db'],
        'true_range': true_distance
    }


def generate_all_measurements(true_positions, clock_states, rf_config):
    """Generate measurements for all node pairs using RF simulation"""

    n_nodes = len(true_positions)
    measurements = {}

    print("Generating RF measurements...")

    # Multiple rounds of measurements
    n_rounds = rf_config['simulation']['n_rounds']

    for round_idx in range(n_rounds):
        print(f"  Round {round_idx + 1}/{n_rounds}")

        # For each potential pair
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Simulate measurement
                meas = simulate_rf_measurement(i, j, true_positions, clock_states, rf_config)

                if meas is not None:
                    key = (i, j)
                    if key not in measurements:
                        measurements[key] = []
                    measurements[key].append(meas)

        # Update clocks between rounds
        if rf_config['simulation']['enable_clock_drift']:
            round_interval = rf_config['simulation']['round_interval']
            for state in clock_states.values():
                state.bias += state.drift * round_interval

    print(f"  Generated {len(measurements)} measurement pairs")

    return measurements


def setup_consensus_from_measurements(true_positions, measurements, clock_states,
                                      n_anchors, consensus_config):
    """Setup consensus solver with RF-generated measurements"""

    n_nodes = len(true_positions)

    # Create consensus solver with proper type conversions
    params = consensus_config['parameters']
    cgn_config = ConsensusGNConfig(
        max_iterations=int(params['max_iterations']),
        consensus_gain=float(params['consensus_gain']),
        step_size=float(params['step_size']),
        gradient_tol=float(params['gradient_tol']),
        step_tol=float(params['step_tol']),
        verbose=params.get('verbose', False)
    )

    cgn = ConsensusGaussNewton(cgn_config)

    # Add nodes with initial states
    for i in range(n_nodes):
        # Initial state estimate
        state = np.zeros(5)

        if i < n_anchors:
            # Anchors know their position
            state[:2] = true_positions[i]
            state[2] = clock_states[i].bias * 1e9  # Convert to nanoseconds
            state[3] = clock_states[i].drift * 1e9  # Convert to ppb
            state[4] = clock_states[i].cfo * 1e-6  # Convert to ppm
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Unknown nodes: add position noise
            noise_std = float(consensus_config['initialization']['position_noise_std'])
            state[:2] = true_positions[i] + np.random.normal(0, noise_std, 2)
            state[2] = clock_states[i].bias * 1e9
            state[3] = clock_states[i].drift * 1e9
            state[4] = clock_states[i].cfo * 1e-6
            cgn.add_node(i, state, is_anchor=False)

    # Add measurements as factors
    comm_range = float(consensus_config['comm_range'])
    n_factors = 0

    for (i, j), meas_list in measurements.items():
        # Check if nodes are in communication range
        dist = np.linalg.norm(true_positions[i] - true_positions[j])
        if dist <= comm_range:
            cgn.add_edge(i, j)

            # Average measurements from multiple rounds
            avg_range = np.mean([m['range_m'] for m in meas_list])
            avg_variance = np.mean([m['variance_m2'] for m in meas_list])

            # Scale variance based on LOS/NLOS
            los_count = sum(1 for m in meas_list if m['is_los'])
            if los_count < len(meas_list) / 2:  # Mostly NLOS
                avg_variance *= 4.0  # Increase uncertainty

            # Add ToA factor
            cgn.add_measurement(ToAFactorMeters(i, j, avg_range, avg_variance))
            n_factors += 1

    print(f"  Added {n_factors} factors to consensus system")

    # Set true positions for evaluation
    cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

    return cgn


def visualize_unified_results(true_positions, cgn, n_anchors, title="Unified FTL Results"):
    """Visualize the unified simulation results"""

    n_nodes = len(true_positions)

    # Get final estimated positions
    estimated_positions = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        estimated_positions[i] = cgn.nodes[i].state[:2]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Network topology
    ax = axes[0]
    ax.set_title('Network Topology')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')

    # Draw edges
    for (i, j) in cgn.edges:
        ax.plot([true_positions[i][0], true_positions[j][0]],
               [true_positions[i][1], true_positions[j][1]],
               'gray', alpha=0.3, linewidth=0.5)

    # Plot nodes
    ax.scatter(true_positions[:n_anchors, 0], true_positions[:n_anchors, 1],
              s=200, c='red', marker='s', label='Anchors', zorder=5)
    ax.scatter(true_positions[n_anchors:, 0], true_positions[n_anchors:, 1],
              s=50, c='blue', marker='o', label='Unknown nodes', zorder=4)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Position estimates
    ax = axes[1]
    ax.set_title('Position Estimates (RF + Consensus)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')

    # True positions
    ax.scatter(true_positions[:n_anchors, 0], true_positions[:n_anchors, 1],
              s=200, c='red', marker='s', label='Anchors', zorder=5)
    ax.scatter(true_positions[n_anchors:, 0], true_positions[n_anchors:, 1],
              s=100, c='blue', marker='o', label='True', zorder=4)

    # Estimated positions
    ax.scatter(estimated_positions[n_anchors:, 0], estimated_positions[n_anchors:, 1],
              s=50, c='green', marker='x', label='Estimated', zorder=6)

    # Error vectors
    for i in range(n_anchors, n_nodes):
        ax.arrow(true_positions[i][0], true_positions[i][1],
                estimated_positions[i][0] - true_positions[i][0],
                estimated_positions[i][1] - true_positions[i][1],
                head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.5)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error distribution
    ax = axes[2]
    errors = np.linalg.norm(true_positions[n_anchors:] - estimated_positions[n_anchors:], axis=1)
    errors_cm = errors * 100

    ax.hist(errors_cm, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title('Error Distribution')
    ax.set_xlabel('Position Error (cm)')
    ax.set_ylabel('Number of Nodes')

    # Statistics
    rmse = np.sqrt(np.mean(errors**2)) * 100
    mean_error = np.mean(errors_cm)
    max_error = np.max(errors_cm)

    stats_text = f'RMSE: {rmse:.1f} cm\n'
    stats_text += f'Mean: {mean_error:.1f} cm\n'
    stats_text += f'Max: {max_error:.1f} cm'
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, rmse


def main():
    """Main unified FTL simulation"""

    parser = argparse.ArgumentParser(description='Unified FTL: RF + Consensus')
    parser.add_argument('--config', type=str, default='configs/unified_30node.yaml',
                       help='Unified configuration file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 70)
    print("Unified FTL Simulation: RF Signal + Distributed Consensus")
    print("=" * 70)

    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    config = load_unified_config(args.config)

    # Phase 1: Network setup
    print("\n--- Phase 1: Network Setup ---")
    true_positions, n_anchors, n_nodes = generate_network_topology(config)
    print(f"  Nodes: {n_nodes} ({n_anchors} anchors)")
    print(f"  Area: {config['geometry']['area_size']}×{config['geometry']['area_size']}m")

    # Phase 2: Initialize clocks
    print("\n--- Phase 2: Clock Initialization ---")
    clock_states = initialize_clock_states(config, n_nodes, n_anchors)
    print(f"  Initialized clock states for {n_nodes} nodes")

    # Phase 3: RF signal simulation
    print("\n--- Phase 3: RF Signal Simulation ---")
    measurements = generate_all_measurements(true_positions, clock_states, config['rf_simulation'])

    # Phase 4: Distributed consensus
    print("\n--- Phase 4: Distributed Consensus ---")
    cgn = setup_consensus_from_measurements(
        true_positions, measurements, clock_states,
        n_anchors, config['consensus']
    )

    print("Running consensus optimization...")
    start_time = time.time()
    results = cgn.optimize()
    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Time: {elapsed:.2f}s")

    if 'position_errors' in results:
        errors = results['position_errors']
        print(f"\nPosition Accuracy:")
        print(f"  RMSE: {errors['rmse']*100:.2f} cm")
        print(f"  Mean: {errors['mean']*100:.2f} cm")
        print(f"  Max: {errors['max']*100:.2f} cm")

    # Visualization
    if not args.no_viz:
        print("\nGenerating visualization...")
        fig, rmse = visualize_unified_results(true_positions, cgn, n_anchors)

        # Save figure
        output_path = Path('unified_ftl_results.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved results to {output_path}")

        plt.show()

    print("\n" + "=" * 70)
    print("Unified FTL simulation complete!")
    print("RF measurements + Distributed consensus = Realistic & Scalable")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()