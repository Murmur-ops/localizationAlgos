#!/usr/bin/env python3
"""
Test FTL performance in 50x50m area with realistic conditions
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
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType


def simulate_realistic_measurement(i: int, j: int, nodes: Dict, config: SignalConfig,
                                  channel_model: SalehValenzuelaChannel,
                                  clock_ensemble: ClockEnsemble) -> Dict:
    """Simulate measurement with all physical impairments"""

    # Get node positions
    pos_i = nodes[i][:2]
    pos_j = nodes[j][:2]

    # Calculate true distance
    true_distance = np.linalg.norm(pos_j - pos_i)

    # Determine LOS probability based on distance and environment
    # More realistic: LOS probability decreases with distance
    los_prob = np.exp(-true_distance / 15.0)  # 15m characteristic distance
    is_los = np.random.rand() < los_prob

    # Generate channel
    channel = channel_model.generate_channel_realization(
        true_distance, is_los=is_los
    )

    # Clock parameters
    clock_i = clock_ensemble.states[i]
    clock_j = clock_ensemble.states[j]
    clock_bias = clock_j.bias - clock_i.bias
    cfo_hz = clock_j.cfo - clock_i.cfo
    sco_ppm = clock_j.sco_ppm - clock_i.sco_ppm

    # Generate signal
    template = gen_hrp_burst(config, n_repeats=3)

    # Calculate SNR with realistic path loss
    # Free space path loss at 6.5 GHz
    path_loss_db = 20 * np.log10(true_distance) + 20 * np.log10(6.5e9) - 147.55
    tx_power_dbm = 0  # 0 dBm transmit power
    noise_figure_db = 6  # Receiver noise figure
    thermal_noise_dbm = -174 + 10 * np.log10(config.bandwidth) + noise_figure_db

    snr_db = tx_power_dbm - path_loss_db - thermal_noise_dbm
    snr_db = max(snr_db, 5)  # Minimum 5 dB SNR for detection

    # Propagate signal
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

    # Compute measurement covariance
    meas_cov = compute_measurement_covariance(
        correlation, template, config.sample_rate,
        use_feature_scaling=True
    )

    # Create edge weight
    edge_weight = EdgeWeight.from_covariance(meas_cov, 'ToA')

    # True and measured ToA
    true_toa = true_distance / 3e8 + clock_bias
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


def run_50x50_simulation(n_anchors: int = 4, n_unknowns: int = 21, seed: int = 42):
    """Run complete 50x50m simulation"""

    print("="*70)
    print("   50x50m FTL PERFORMANCE TEST")
    print("   With All Physical Improvements")
    print("="*70)

    np.random.seed(seed)

    # Configuration
    area_size = 50.0  # 50x50 meters
    n_nodes = n_anchors + n_unknowns

    # Signal configuration
    sig_config = SignalConfig(
        bandwidth=499.2e6,  # IEEE 802.15.4z HRP-UWB
        sample_rate=2e9      # 2 GHz sampling
    )

    # Channel configuration - mixed indoor/outdoor
    ch_config = ChannelConfig(environment='indoor_industrial')
    channel_model = SalehValenzuelaChannel(ch_config)

    # Clock configuration - realistic TCXO
    clock_model = ClockModel(oscillator_type="TCXO")
    clock_ensemble = ClockEnsemble(
        n_nodes, clock_model,
        anchor_indices=list(range(n_anchors))
    )

    print(f"\nConfiguration:")
    print(f"  Area: {area_size} × {area_size} meters")
    print(f"  Nodes: {n_anchors} anchors + {n_unknowns} unknowns = {n_nodes} total")
    print(f"  Signal: HRP-UWB, {sig_config.bandwidth/1e6:.1f} MHz BW")
    print(f"  Sampling: {sig_config.sample_rate/1e9:.1f} GHz")
    print(f"  Channel: {ch_config.environment}")
    print(f"  Clocks: {clock_model.oscillator_type} (±{clock_model.frequency_accuracy_ppm} ppm)")

    # Compute RMS bandwidth
    template = gen_hrp_burst(sig_config, n_repeats=3)
    beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)
    print(f"  RMS Bandwidth: {beta_rms/1e6:.1f} MHz (factor {beta_rms/sig_config.bandwidth:.3f})")

    # Place anchors at corners
    anchor_positions = place_anchors([], n_anchors, area_size, PlacementType.CORNERS)

    # Place unknown nodes on grid with jitter
    grid_size = int(np.ceil(np.sqrt(n_unknowns)))
    unknown_positions = place_grid_nodes(grid_size, area_size * 0.8, jitter_std=2.0, seed=seed+1)

    # Offset unknowns to center
    offset = area_size * 0.1
    for pos in unknown_positions:
        pos.x += offset
        pos.y += offset

    # Combine into node dictionary
    nodes = {}
    for i in range(n_anchors):
        nodes[i] = np.array([
            anchor_positions[i].x,
            anchor_positions[i].y,
            clock_ensemble.states[i].bias,
            clock_ensemble.states[i].drift,
            clock_ensemble.states[i].cfo
        ])

    for i in range(n_unknowns):
        if i < len(unknown_positions):
            nodes[n_anchors + i] = np.array([
                unknown_positions[i].x,
                unknown_positions[i].y,
                clock_ensemble.states[n_anchors + i].bias,
                clock_ensemble.states[n_anchors + i].drift,
                clock_ensemble.states[n_anchors + i].cfo
            ])

    print(f"\nNode Placement:")
    print(f"  Anchors: corners at (0,0), (0,{area_size}), ({area_size},0), ({area_size},{area_size})")
    print(f"  Unknowns: {grid_size}×{grid_size} grid with 2m jitter")

    # Simulate measurements
    print(f"\nGenerating measurements...")
    measurements = []
    max_range = 70.0  # Maximum measurement range (diagonal of 50x50)

    measurement_count = 0
    los_count = 0

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if i in nodes and j in nodes:
                dist = np.linalg.norm(nodes[j][:2] - nodes[i][:2])
                if dist < max_range:
                    # Probability of successful measurement decreases with distance
                    meas_prob = np.exp(-dist / 30.0)
                    if np.random.rand() < meas_prob:
                        meas = simulate_realistic_measurement(
                            i, j, nodes, sig_config,
                            channel_model, clock_ensemble
                        )
                        measurements.append(meas)
                        measurement_count += 1
                        if meas['is_los']:
                            los_count += 1

    print(f"  Total measurements: {measurement_count}")
    print(f"  LOS/NLOS: {los_count}/{measurement_count - los_count}")
    print(f"  LOS percentage: {100*los_count/measurement_count:.1f}%")

    # Measurement quality statistics
    snr_values = [m['snr_db'] for m in measurements]
    distances = [m['true_distance'] for m in measurements]
    variances = [m['variance'] for m in measurements]

    print(f"\nMeasurement Quality:")
    print(f"  SNR range: {min(snr_values):.1f} - {max(snr_values):.1f} dB")
    print(f"  Mean SNR: {np.mean(snr_values):.1f} dB")
    print(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} m")

    # Range accuracy from CRLB
    range_stds_cm = [np.sqrt(v) * 3e8 * 100 for v in variances]
    print(f"  Range σ (CRLB):")
    print(f"    Min: {min(range_stds_cm):.1f} cm")
    print(f"    Max: {max(range_stds_cm):.1f} cm")
    print(f"    Median: {np.median(range_stds_cm):.1f} cm")
    print(f"    Mean: {np.mean(range_stds_cm):.1f} cm")

    # Build factor graph
    print(f"\nBuilding factor graph...")
    graph = FactorGraph()

    # Add nodes
    for i in range(n_nodes):
        is_anchor = i < n_anchors
        if is_anchor:
            initial = nodes[i]
        else:
            # Initialize unknowns near center with random offset
            initial = np.array([
                area_size/2 + np.random.randn() * 5,
                area_size/2 + np.random.randn() * 5,
                0.0,  # bias
                0.0,  # drift
                0.0   # cfo
            ])
        graph.add_node(i, initial, is_anchor=is_anchor)

    # Add measurement factors with CRLB-based variances
    for meas in measurements:
        graph.add_toa_factor(
            meas['i'], meas['j'],
            meas['measured_toa'],
            meas['variance']  # Physical variance!
        )

    print(f"  Nodes: {n_nodes} ({n_anchors} fixed)")
    print(f"  Factors: {len(measurements)}")

    # Optimize
    print(f"\nOptimizing factor graph...")
    result = graph.optimize(
        max_iterations=200,
        tolerance=1e-7,
        verbose=False
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Cost: {result.initial_cost:.2e} → {result.final_cost:.2e}")

    # Calculate position errors
    print(f"\n" + "="*70)
    print("POSITION ACCURACY RESULTS:")
    print("="*70)

    position_errors = []
    for i in range(n_anchors, n_nodes):
        if i in nodes:
            true_pos = nodes[i][:2]
            est_pos = result.estimates[i][:2]
            error = np.linalg.norm(est_pos - true_pos)
            position_errors.append(error)

    if position_errors:
        rmse = np.sqrt(np.mean(np.array(position_errors)**2))
        mae = np.mean(position_errors)
        median_error = np.median(position_errors)
        p90 = np.percentile(position_errors, 90)
        p95 = np.percentile(position_errors, 95)
        max_error = max(position_errors)

        print(f"\nPosition Errors ({len(position_errors)} nodes):")
        print(f"  RMSE:    {rmse:.3f} m")
        print(f"  MAE:     {mae:.3f} m")
        print(f"  Median:  {median_error:.3f} m")
        print(f"  90%ile:  {p90:.3f} m")
        print(f"  95%ile:  {p95:.3f} m")
        print(f"  Max:     {max_error:.3f} m")

        # Theoretical bound
        avg_var = np.mean(variances)
        theoretical_std = np.sqrt(avg_var) * 3e8
        gdop = np.sqrt(len(position_errors) / n_anchors)  # Rough GDOP estimate

        print(f"\nTheoretical Performance:")
        print(f"  Avg CRLB σ: {theoretical_std:.3f} m")
        print(f"  Est. GDOP: {gdop:.1f}")
        print(f"  Expected RMSE: ~{theoretical_std * gdop:.3f} m")
        print(f"  Efficiency: {(theoretical_std * gdop / rmse * 100):.1f}%")

        # Clock estimation accuracy
        clock_errors = []
        for i in range(n_anchors, n_nodes):
            if i in nodes:
                true_bias = nodes[i][2]
                est_bias = result.estimates[i][2]
                error = abs(est_bias - true_bias)
                clock_errors.append(error)

        if clock_errors:
            clock_rmse = np.sqrt(np.mean(np.array(clock_errors)**2))
            print(f"\nClock Synchronization:")
            print(f"  Bias RMSE: {clock_rmse*1e6:.1f} µs")
            print(f"  Bias MAE: {np.mean(clock_errors)*1e6:.1f} µs")

    return {
        'rmse': rmse,
        'mae': mae,
        'measurements': measurements,
        'nodes': nodes,
        'result': result,
        'position_errors': position_errors
    }


if __name__ == "__main__":
    # Run main test
    results = run_50x50_simulation(n_anchors=4, n_unknowns=21, seed=42)

    print("\n" + "="*70)
    print(f"✅ 50×50m TEST COMPLETE")
    print(f"   Final RMSE: {results['rmse']:.3f} m")
    print("="*70)