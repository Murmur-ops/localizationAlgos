#!/usr/bin/env python3
"""
Test FTL in 50x50m with proper clock initialization
"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.clocks import ClockModel, ClockEnsemble
from ftl.rx_frontend import matched_filter, detect_toa
from ftl.measurement_covariance import compute_measurement_covariance, EdgeWeight
from ftl.solver import FactorGraph
from ftl.geometry import place_grid_nodes, place_anchors, PlacementType

def simulate_measurement_fixed(i, j, nodes, config, channel_model):
    """Simulate with proper clock handling"""
    
    pos_i = nodes[i][:2]
    pos_j = nodes[j][:2]
    true_distance = np.linalg.norm(pos_j - pos_i)
    
    # LOS probability
    los_prob = np.exp(-true_distance / 20.0)
    is_los = np.random.rand() < los_prob
    
    # Generate channel
    channel = channel_model.generate_channel_realization(true_distance, is_los=is_los)
    
    # Clock parameters (already in nodes array)
    bias_i = nodes[i][2]
    bias_j = nodes[j][2]
    clock_bias = bias_j - bias_i
    
    # Generate signal
    template = gen_hrp_burst(config, n_repeats=3)
    
    # SNR based on distance
    path_loss_db = 40 + 20 * np.log10(true_distance/10)
    snr_db = 25 - path_loss_db
    snr_db = max(snr_db, 5)
    
    # Propagate (no CFO/SCO for simplicity)
    result = propagate_signal(
        template, channel, config.sample_rate,
        snr_db=snr_db, cfo_hz=0, sco_ppm=0,
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
        'i': i, 'j': j,
        'true_toa': true_distance / 3e8 + clock_bias,
        'measured_toa': toa_result['toa'],
        'variance': meas_cov.toa_variance,
        'is_los': meas_cov.is_los,
        'snr_db': 10 * np.log10(meas_cov.snr_linear)
    }

# Main test
np.random.seed(42)
area_size = 50.0
n_anchors = 4
n_unknowns = 9
n_nodes = n_anchors + n_unknowns

print("="*70)
print("50×50m Test with Fixed Clock Initialization")
print("="*70)

# Signal config
sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)

# Channel
ch_config = ChannelConfig(environment='indoor_office')
channel_model = SalehValenzuelaChannel(ch_config)

# Compute RMS bandwidth
template = gen_hrp_burst(sig_config, n_repeats=3)
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)
print(f"\nRMS Bandwidth: {beta_rms/1e6:.1f} MHz")

# Place nodes
anchor_positions = [(0,0), (area_size,0), (area_size,area_size), (0,area_size)]
grid_size = int(np.ceil(np.sqrt(n_unknowns)))
unknown_positions = place_grid_nodes(grid_size, area_size*0.6, jitter_std=2.0)

# Center unknowns
offset = area_size * 0.2
for pos in unknown_positions:
    pos.x += offset
    pos.y += offset

# Build nodes with PROPER clock initialization
nodes = {}

# Anchors: zero clock parameters (reference time)
for i in range(n_anchors):
    x, y = anchor_positions[i]
    nodes[i] = np.array([x, y, 0.0, 0.0, 0.0])  # Zero clock params

# Unknowns: small random clock bias (microseconds, not milliseconds)
for i in range(n_unknowns):
    if i < len(unknown_positions):
        # Small initial bias: ~10 microseconds std
        bias = np.random.normal(0, 10e-6)
        nodes[n_anchors + i] = np.array([
            unknown_positions[i].x,
            unknown_positions[i].y,
            bias,  # Microsecond scale, not millisecond!
            0.0, 0.0
        ])

print(f"\nNodes: {n_anchors} anchors + {n_unknowns} unknowns")

# Generate measurements
print("\nGenerating measurements...")
measurements = []
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        if i in nodes and j in nodes:
            dist = np.linalg.norm(nodes[j][:2] - nodes[i][:2])
            if dist < 70:
                meas = simulate_measurement_fixed(i, j, nodes, sig_config, channel_model)
                measurements.append(meas)

los_count = sum(1 for m in measurements if m['is_los'])
print(f"  Total: {len(measurements)} ({los_count} LOS)")

# Build graph
print("\nOptimizing...")
graph = FactorGraph()

# Add nodes
for i in range(n_nodes):
    is_anchor = i < n_anchors
    if is_anchor:
        initial = nodes[i]
    else:
        # Initial guess near center
        initial = np.array([
            area_size/2 + np.random.randn()*5,
            area_size/2 + np.random.randn()*5,
            0.0, 0.0, 0.0
        ])
    graph.add_node(i, initial, is_anchor=is_anchor)

# Add factors
for meas in measurements:
    graph.add_toa_factor(
        meas['i'], meas['j'],
        meas['measured_toa'],
        meas['variance']
    )

# Optimize
result = graph.optimize(max_iterations=100, tolerance=1e-6, verbose=False)
print(f"  Converged: {result.converged} in {result.iterations} iterations")

# Calculate errors
position_errors = []
for i in range(n_anchors, n_nodes):
    true_pos = nodes[i][:2]
    est_pos = result.estimates[i][:2]
    error = np.linalg.norm(est_pos - true_pos)
    position_errors.append(error)

rmse = np.sqrt(np.mean(np.array(position_errors)**2))
print(f"\n{'='*70}")
print(f"RESULTS:")
print(f"  RMSE: {rmse:.3f} m")
print(f"  MAE: {np.mean(position_errors):.3f} m")
print(f"  Max: {max(position_errors):.3f} m")
print("="*70)
