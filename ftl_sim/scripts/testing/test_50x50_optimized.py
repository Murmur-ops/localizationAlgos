#!/usr/bin/env python3
"""
Test FTL performance in 50x50m area with optimized conditions
"""

import numpy as np
from typing import Dict

# Import improved FTL modules
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.clocks import ClockModel, ClockEnsemble
from ftl.rx_frontend import matched_filter, detect_toa
from ftl.measurement_covariance import compute_measurement_covariance, EdgeWeight
from ftl.solver import FactorGraph
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType


def simulate_measurement_optimized(i: int, j: int, nodes: Dict, config: SignalConfig,
                                  channel_model: SalehValenzuelaChannel,
                                  clock_ensemble: ClockEnsemble) -> Dict:
    """Simulate measurement with better LOS conditions"""

    # Get positions
    pos_i = nodes[i][:2]
    pos_j = nodes[j][:2]
    true_distance = np.linalg.norm(pos_j - pos_i)

    # Better LOS probability for shorter distances
    if true_distance < 20:
        los_prob = 0.8  # 80% LOS for short range
    elif true_distance < 35:
        los_prob = 0.5  # 50% LOS for medium range
    else:
        los_prob = 0.2  # 20% LOS for long range

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

    # Better SNR model
    path_loss_db = 40 + 20 * np.log10(true_distance/10)  # Simplified path loss
    snr_db = 35 - path_loss_db  # Start with 35 dB at reference
    snr_db = max(snr_db, 10)  # Minimum 10 dB SNR

    # Propagate signal
    result = propagate_signal(
        template, channel, config.sample_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        sco_ppm=sco_ppm,
        clock_bias_s=clock_bias
    )

    # Process
    correlation = matched_filter(result['signal'], template)
    toa_result = detect_toa(correlation, config.sample_rate)

    # Compute covariance
    meas_cov = compute_measurement_covariance(
        correlation, template, config.sample_rate,
        use_feature_scaling=True
    )

    edge_weight = EdgeWeight.from_covariance(meas_cov, 'ToA')

    return {
        'i': i,
        'j': j,
        'true_toa': true_distance / 3e8 + clock_bias,
        'measured_toa': toa_result['toa'],
        'variance': meas_cov.toa_variance,
        'weight': edge_weight.weight,
        'confidence': edge_weight.confidence,
        'is_los': is_los,
        'snr_db': snr_db,
        'true_distance': true_distance
    }


def run_optimized_50x50(n_anchors: int = 8, n_unknowns: int = 16):
    """Run 50x50m test with better anchor configuration"""

    print("="*70)
    print("   OPTIMIZED 50×50m FTL TEST")
    print("="*70)

    np.random.seed(123)  # Different seed for variety

    area_size = 50.0
    n_nodes = n_anchors + n_unknowns

    # Configuration
    sig_config = SignalConfig(
        bandwidth=499.2e6,
        sample_rate=2e9
    )

    # Use better channel model
    ch_config = ChannelConfig(environment='indoor_office')  # Better than industrial
    channel_model = SalehValenzuelaChannel(ch_config)

    # Better clocks for anchors
    clock_model = ClockModel(oscillator_type="TCXO")
    clock_ensemble = ClockEnsemble(
        n_nodes, clock_model,
        anchor_indices=list(range(n_anchors))
    )

    print(f"\nConfiguration:")
    print(f"  Area: {area_size} × {area_size} m")
    print(f"  Nodes: {n_anchors} anchors + {n_unknowns} unknowns")
    print(f"  Environment: {ch_config.environment}")

    # Compute RMS bandwidth
    template = gen_hrp_burst(sig_config, n_repeats=3)
    beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)
    print(f"  RMS Bandwidth: {beta_rms/1e6:.1f} MHz")

    # Better anchor placement - mix of corners and edges
    if n_anchors == 8:
        # Custom placement: 4 corners + 4 edge midpoints
        anchor_positions = [
            (0, 0), (area_size, 0), (area_size, area_size), (0, area_size),  # Corners
            (area_size/2, 0), (area_size, area_size/2),  # Edge midpoints
            (area_size/2, area_size), (0, area_size/2)
        ]
    else:
        # Use perimeter placement
        anchor_nodes = place_anchors([], n_anchors, area_size, PlacementType.PERIMETER)
        anchor_positions = [(node.x, node.y) for node in anchor_nodes]

    # Place unknowns
    grid_size = int(np.ceil(np.sqrt(n_unknowns)))
    unknown_positions = place_grid_nodes(grid_size, area_size * 0.7, jitter_std=1.0, seed=124)

    # Center the unknowns
    offset = area_size * 0.15
    for pos in unknown_positions:
        pos.x += offset
        pos.y += offset

    # Build node dictionary
    nodes = {}
    for i in range(n_anchors):
        if i < len(anchor_positions):
            x, y = anchor_positions[i]
            nodes[i] = np.array([x, y, 0, 0, 0])  # Anchors have perfect clocks

    for i in range(n_unknowns):
        if i < len(unknown_positions):
            nodes[n_anchors + i] = np.array([
                unknown_positions[i].x,
                unknown_positions[i].y,
                clock_ensemble.states[n_anchors + i].bias,
                clock_ensemble.states[n_anchors + i].drift,
                clock_ensemble.states[n_anchors + i].cfo
            ])

    print(f"  Anchors: {n_anchors} (corners + edges)")
    print(f"  Unknowns: {grid_size}×{grid_size} grid")

    # Generate ALL possible measurements (better connectivity)
    print(f"\nGenerating measurements...")
    measurements = []
    los_count = 0

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if i in nodes and j in nodes:
                dist = np.linalg.norm(nodes[j][:2] - nodes[i][:2])
                if dist < 75:  # Measure everything within range
                    meas = simulate_measurement_optimized(
                        i, j, nodes, sig_config,
                        channel_model, clock_ensemble
                    )
                    measurements.append(meas)
                    if meas['is_los']:
                        los_count += 1

    print(f"  Total: {len(measurements)}")
    print(f"  LOS: {los_count} ({100*los_count/len(measurements):.1f}%)")

    # Statistics
    snr_values = [m['snr_db'] for m in measurements]
    variances = [m['variance'] for m in measurements]
    range_stds_cm = [np.sqrt(v) * 3e8 * 100 for v in variances]

    print(f"  Avg SNR: {np.mean(snr_values):.1f} dB")
    print(f"  Median range σ: {np.median(range_stds_cm):.1f} cm")

    # Build graph
    print(f"\nOptimizing...")
    graph = FactorGraph()

    # Add nodes with better initialization
    for i in range(n_nodes):
        is_anchor = i < n_anchors
        if is_anchor:
            initial = nodes[i]
        else:
            # Better initialization using trilateration hint
            if len(measurements) > 0:
                # Use average of anchor positions as initial guess
                anchor_positions_array = [nodes[j][:2] for j in range(n_anchors)]
                initial_pos = np.mean(anchor_positions_array, axis=0)
                initial = np.array([
                    initial_pos[0] + np.random.randn() * 2,
                    initial_pos[1] + np.random.randn() * 2,
                    0.0, 0.0, 0.0
                ])
            else:
                initial = np.array([area_size/2, area_size/2, 0, 0, 0])

        graph.add_node(i, initial, is_anchor=is_anchor)

    # Add factors
    for meas in measurements:
        graph.add_toa_factor(
            meas['i'], meas['j'],
            meas['measured_toa'],
            meas['variance']
        )

    # Optimize with more iterations
    result = graph.optimize(
        max_iterations=500,
        tolerance=1e-8,
        verbose=False
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")

    # Calculate errors
    position_errors = []
    for i in range(n_anchors, n_nodes):
        if i in nodes:
            true_pos = nodes[i][:2]
            est_pos = result.estimates[i][:2]
            error = np.linalg.norm(est_pos - true_pos)
            position_errors.append(error)

    rmse = np.sqrt(np.mean(np.array(position_errors)**2))
    mae = np.mean(position_errors)
    median = np.median(position_errors)
    p90 = np.percentile(position_errors, 90)

    print(f"\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"  RMSE:   {rmse:.3f} m")
    print(f"  MAE:    {mae:.3f} m")
    print(f"  Median: {median:.3f} m")
    print(f"  90%:    {p90:.3f} m")

    # Theoretical
    avg_var = np.mean(variances)
    theoretical = np.sqrt(avg_var) * 3e8
    print(f"\n  Theory: {theoretical:.3f} m")
    print(f"  Efficiency: {theoretical/rmse*100:.1f}%")

    return rmse


# Run multiple configurations
if __name__ == "__main__":
    print("\nTesting different configurations...\n")

    configs = [
        (4, 16, "4 anchors (corners)"),
        (8, 16, "8 anchors (corners+edges)"),
        (6, 16, "6 anchors (perimeter)"),
    ]

    results = []
    for n_anchors, n_unknowns, desc in configs:
        print(f"\nConfiguration: {desc}")
        rmse = run_optimized_50x50(n_anchors, n_unknowns)
        results.append((desc, rmse))

    print("\n" + "="*70)
    print("SUMMARY - 50×50m Area Performance:")
    print("="*70)
    for desc, rmse in results:
        print(f"  {desc:30s}: RMSE = {rmse:.3f} m")
    print("="*70)