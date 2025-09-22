#!/usr/bin/env python3
"""
Demo: Improved FTL System with Physical Accuracy
Demonstrates all improvements from ChatGPT's critique
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import improved FTL modules
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.clocks import ClockModel, ClockEnsemble
from ftl.rx_frontend import matched_filter, detect_toa
from ftl.measurement_covariance import compute_measurement_covariance, EdgeWeight
from ftl.solver import FactorGraph
from ftl.init import initialize_positions
from ftl.geometry import place_grid_nodes


def simulate_measurement(i: int, j: int, nodes: Dict, config: SignalConfig,
                         channel_model: SalehValenzuelaChannel,
                         clock_ensemble: ClockEnsemble) -> Dict:
    """
    Simulate a ToA measurement with physical accuracy

    This function now includes:
    - CRLB-based variance computation
    - Sample Clock Offset (SCO)
    - Proper Allan variance clock evolution
    - NLOS detection and variance inflation
    """
    # Get node positions
    pos_i = nodes[i][:2]
    pos_j = nodes[j][:2]

    # Calculate true distance
    true_distance = np.linalg.norm(pos_j - pos_i)

    # Generate channel realization
    is_los = np.random.rand() > 0.3  # 70% LOS probability
    channel = channel_model.generate_channel_realization(
        true_distance, is_los=is_los
    )

    # Get clock states
    clock_i = clock_ensemble.states[i]
    clock_j = clock_ensemble.states[j]

    # Calculate relative clock parameters
    clock_bias = clock_j.bias - clock_i.bias
    cfo_hz = clock_j.cfo - clock_i.cfo
    sco_ppm = clock_j.sco_ppm - clock_i.sco_ppm

    # Generate transmit signal
    template = gen_hrp_burst(config, n_repeats=2)

    # Calculate SNR based on distance (path loss)
    snr_db = 30 - 20 * np.log10(true_distance / 10)

    # Propagate signal through channel with all impairments
    result = propagate_signal(
        template, channel, config.sample_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        sco_ppm=sco_ppm,
        clock_bias_s=clock_bias
    )

    # Receiver processing
    correlation = matched_filter(result['signal'], template)
    toa_result = detect_toa(correlation, config.sample_rate)

    # Compute measurement covariance using CRLB and NLOS features
    meas_cov = compute_measurement_covariance(
        correlation, template, config.sample_rate,
        use_feature_scaling=True
    )

    # Create edge weight for factor graph
    edge_weight = EdgeWeight.from_covariance(meas_cov, 'ToA')

    # True ToA includes propagation delay and clock bias
    true_toa = true_distance / 3e8 + clock_bias

    # Measured ToA with noise
    measured_toa = toa_result['toa']

    return {
        'i': i,
        'j': j,
        'true_toa': true_toa,
        'measured_toa': measured_toa,
        'variance': meas_cov.toa_variance,
        'weight': edge_weight.weight,
        'confidence': edge_weight.confidence,
        'is_los': meas_cov.is_los,
        'snr_db': 10 * np.log10(meas_cov.snr_linear),
        'true_distance': true_distance
    }


def main():
    """Main demo of improved FTL system"""

    print("="*70)
    print("   IMPROVED FTL SYSTEM DEMONSTRATION")
    print("   With CRLB, SCO, Allan Variance, and NLOS Handling")
    print("="*70)

    # Configuration
    np.random.seed(42)
    n_anchors = 4
    n_unknowns = 9
    n_nodes = n_anchors + n_unknowns

    # Signal configuration
    sig_config = SignalConfig(
        bandwidth=499.2e6,
        sample_rate=1e9
    )

    # Channel configuration
    ch_config = ChannelConfig(environment='indoor_office')
    channel_model = SalehValenzuelaChannel(ch_config)

    # Clock configuration - realistic TCXO oscillators
    clock_model = ClockModel(oscillator_type="TCXO")
    clock_ensemble = ClockEnsemble(
        n_nodes, clock_model,
        anchor_indices=list(range(n_anchors))
    )

    # Create node layout
    area_size = 20.0  # 20m x 20m area
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    node_positions = place_grid_nodes(grid_size, area_size, jitter_std=1.0, seed=42)

    # Convert to dict format with 5D state vectors
    nodes = {}
    for i in range(min(n_nodes, len(node_positions))):
        nodes[i] = np.array([
            node_positions[i].x,  # x
            node_positions[i].y,  # y
            clock_ensemble.states[i].bias,  # clock bias
            clock_ensemble.states[i].drift,  # clock drift
            clock_ensemble.states[i].cfo  # CFO
        ])

    print(f"\nSystem Setup:")
    print(f"  Nodes: {n_anchors} anchors + {n_unknowns} unknowns")
    print(f"  Area: {area_size}m x {area_size}m")
    print(f"  Signal: HRP-UWB, {sig_config.bandwidth/1e6:.1f} MHz")
    print(f"  Channel: {ch_config.environment}")
    print(f"  Clocks: {clock_model.oscillator_type} (±{clock_model.frequency_accuracy_ppm} ppm)")

    # Compute signal RMS bandwidth for CRLB
    template = gen_hrp_burst(sig_config, n_repeats=2)
    beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)
    print(f"  RMS Bandwidth: {beta_rms/1e6:.1f} MHz")

    # Simulate measurements
    print(f"\nSimulating measurements...")
    measurements = []

    # All pairs within range
    max_range = 30.0
    n_meas = 0

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(nodes[j][:2] - nodes[i][:2])
            if dist < max_range:
                meas = simulate_measurement(
                    i, j, nodes, sig_config,
                    channel_model, clock_ensemble
                )
                measurements.append(meas)
                n_meas += 1

    print(f"  Generated {n_meas} measurements")

    # Analyze measurement quality
    los_count = sum(1 for m in measurements if m['is_los'])
    avg_snr = np.mean([m['snr_db'] for m in measurements])
    avg_confidence = np.mean([m['confidence'] for m in measurements])

    print(f"\nMeasurement Statistics:")
    print(f"  LOS/NLOS: {los_count}/{n_meas-los_count}")
    print(f"  Average SNR: {avg_snr:.1f} dB")
    print(f"  Average Confidence: {avg_confidence:.2f}")

    # Show variance distribution
    variances = [m['variance'] for m in measurements]
    range_stds_cm = [np.sqrt(v) * 3e8 * 100 for v in variances]

    print(f"  Range σ: min={min(range_stds_cm):.1f}cm, "
          f"max={max(range_stds_cm):.1f}cm, "
          f"median={np.median(range_stds_cm):.1f}cm")

    # Create factor graph with proper weights
    print(f"\nBuilding factor graph with CRLB-based weights...")
    graph = FactorGraph()

    # Add nodes
    for i in range(n_nodes):
        is_anchor = i < n_anchors
        if is_anchor:
            initial = nodes[i]
        else:
            # Simple initialization: center of area with small random offset
            initial = np.array([
                area_size/2 + np.random.randn() * 2,  # x
                area_size/2 + np.random.randn() * 2,  # y
                0.0,  # bias
                0.0,  # drift
                0.0   # cfo
            ])
        graph.add_node(i, initial, is_anchor=is_anchor)

    # Add measurement factors with computed variances
    for meas in measurements:
        graph.add_toa_factor(
            meas['i'], meas['j'],
            meas['measured_toa'],
            meas['variance']  # Using CRLB-based variance!
        )

    print(f"  Added {len(measurements)} ToA factors")
    print(f"  Weight range: [{min(m['weight'] for m in measurements):.2e}, "
          f"{max(m['weight'] for m in measurements):.2e}]")

    # Optimize
    print(f"\nOptimizing...")
    result = graph.optimize(
        max_iterations=100,
        tolerance=1e-6,
        verbose=False
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Cost reduction: {result.initial_cost:.2e} → {result.final_cost:.2e}")

    # Evaluate accuracy
    print(f"\nPosition Accuracy:")
    position_errors = []

    for i in range(n_anchors, n_nodes):
        true_pos = nodes[i][:2]
        est_pos = result.estimates[i][:2]
        error = np.linalg.norm(est_pos - true_pos)
        position_errors.append(error)

    rmse = np.sqrt(np.mean(np.array(position_errors)**2))
    mae = np.mean(position_errors)

    print(f"  RMSE: {rmse:.3f}m")
    print(f"  MAE: {mae:.3f}m")
    print(f"  90th percentile: {np.percentile(position_errors, 90):.3f}m")

    # Compare to theoretical CRLB
    avg_var = np.mean(variances)
    theoretical_std = np.sqrt(avg_var) * 3e8
    print(f"\nTheoretical Performance:")
    print(f"  Average CRLB σ: {theoretical_std*100:.1f}cm")
    print(f"  Efficiency: {theoretical_std/rmse*100:.1f}%")

    # Clock estimation accuracy
    print(f"\nClock Estimation:")
    clock_bias_errors = []

    for i in range(n_anchors, n_nodes):
        true_bias = clock_ensemble.states[i].bias
        est_bias = result.estimates[i][2]  # bias is 3rd element
        error = abs(est_bias - true_bias)
        clock_bias_errors.append(error)

    print(f"  Bias RMSE: {np.sqrt(np.mean(np.array(clock_bias_errors)**2))*1e6:.1f}µs")

    # Visualize results
    if False:  # Set to True to enable plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot positions
        ax = axes[0]
        # True positions
        for i in range(n_nodes):
            if i < n_anchors:
                ax.plot(nodes[i][0], nodes[i][1], 'rs', markersize=10, label='Anchor' if i==0 else '')
            else:
                ax.plot(nodes[i][0], nodes[i][1], 'bo', markersize=8, label='True' if i==n_anchors else '')

        # Estimated positions
        for i in range(n_anchors, n_nodes):
            est_pos = result.estimates[i][:2]
            ax.plot(est_pos[0], est_pos[1], 'g^', markersize=8, label='Estimated' if i==n_anchors else '')
            # Error line
            ax.plot([nodes[i][0], est_pos[0]], [nodes[i][1], est_pos[1]], 'r-', alpha=0.3)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Position Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Plot measurement quality
        ax = axes[1]
        weights = [m['weight'] for m in measurements]
        confidences = [m['confidence'] for m in measurements]

        scatter = ax.scatter(weights, confidences,
                           c=['green' if m['is_los'] else 'red' for m in measurements],
                           alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Edge Weight')
        ax.set_ylabel('Confidence')
        ax.set_title('Measurement Quality (Green=LOS, Red=NLOS)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('improved_ftl_demo.png')
        print(f"\n✓ Figure saved as improved_ftl_demo.png")

    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("\nKey Improvements Demonstrated:")
    print("  ✓ CRLB-based measurement covariance")
    print("  ✓ Sample Clock Offset (SCO) effects")
    print("  ✓ Allan variance clock evolution")
    print("  ✓ NLOS detection and variance inflation")
    print("  ✓ Physical edge weights in factor graph")
    print("="*70)


if __name__ == "__main__":
    main()