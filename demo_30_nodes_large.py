#!/usr/bin/env python3
"""
30-Node Large Area Localization Demo
4 anchors, 26 unknowns over 50x50m area
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge


def load_config(config_path: str = "configs/30_node_large.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_measurements(config: dict) -> Tuple[list, dict, dict, list]:
    """Generate ranging measurements between all node pairs"""

    print("\n" + "="*70)
    print("GENERATING MEASUREMENTS")
    print("="*70)

    # Set seed for reproducibility
    np.random.seed(config['system']['seed'])

    # Extract nodes
    nodes = config['nodes']
    positions = {node['id']: np.array(node['position']) for node in nodes}
    anchors = {node['id']: np.array(node['position']) for node in nodes if node['is_anchor']}
    unknowns = {node['id']: np.array(node['position']) for node in nodes if not node['is_anchor']}

    # Configure channel
    channel_config = ChannelConfig()
    channel_config.path_loss_exponent = config['channel']['path_loss_exponent']
    channel_config.nlos_bias_mean_m = config['channel']['nlos_bias_mean_m']
    channel_config.nlos_bias_std_m = config['channel']['nlos_bias_std_m']
    channel_config.delay_spread_ns = config['channel']['multipath_delay_spread_ns']

    channel = RangingChannel(channel_config)

    measurements = []
    measurement_details = []

    # Generate measurements between all pairs
    node_ids = list(positions.keys())
    n_meas = 0
    n_los = 0
    n_nlos = 0

    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            id_i = node_ids[i]
            id_j = node_ids[j]

            pos_i = positions[id_i]
            pos_j = positions[id_j]

            # Calculate true distance
            true_dist = np.linalg.norm(pos_i - pos_j)

            # Skip if distance > 40m (limited range)
            if true_dist > 40.0:
                continue

            # Determine propagation type
            if np.random.rand() < config['channel']['nlos_probability']:
                prop_type = PropagationType.NLOS
                n_nlos += 1
            else:
                prop_type = PropagationType.LOS
                n_los += 1

            # Generate measurement
            meas = channel.generate_measurement(
                true_dist,
                prop_type,
                config['channel']['environment']
            )

            # Create measurement edge
            edge = MeasurementEdge(
                node_i=id_i,
                node_j=id_j,
                distance=meas['measured_distance_m'],
                quality=meas['quality_score'],
                variance=meas['measurement_std_m']**2
            )
            measurements.append(edge)
            n_meas += 1

            # Store details for analysis
            detail = {
                'pair': (id_i, id_j),
                'true_distance': true_dist,
                'measured_distance': meas['measured_distance_m'],
                'error': meas['measured_distance_m'] - true_dist,
                'propagation': prop_type.value,
                'snr_db': meas['snr_db'],
                'quality_score': meas['quality_score']
            }
            measurement_details.append(detail)

    print(f"\nGenerated {n_meas} measurements")
    print(f"  LOS: {n_los} ({100*n_los/n_meas:.1f}%)")
    print(f"  NLOS: {n_nlos} ({100*n_nlos/n_meas:.1f}%)")
    print(f"  Anchors: {len(anchors)}")
    print(f"  Unknown nodes: {len(unknowns)}")

    # Print some statistics
    errors = [d['error'] for d in measurement_details]
    print(f"\nMeasurement errors:")
    print(f"  Mean: {np.mean(errors):.3f}m")
    print(f"  Std:  {np.std(errors):.3f}m")
    print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.3f}m")

    return measurements, anchors, unknowns, measurement_details


def smart_initialization(unknowns: dict, measurements: list, anchors: dict) -> np.ndarray:
    """Initialize unknown positions using trilateration from anchors"""
    n_unknowns = len(unknowns)
    initial_positions = np.zeros(n_unknowns * 2)
    unknown_ids = sorted(unknowns.keys())

    # For each unknown node, use trilateration from anchors if possible
    for idx, uid in enumerate(unknown_ids):
        # Find measurements from this unknown to anchors
        anchor_dists = {}
        for m in measurements:
            if m.node_i == uid and m.node_j in anchors:
                anchor_dists[m.node_j] = m.distance
            elif m.node_j == uid and m.node_i in anchors:
                anchor_dists[m.node_i] = m.distance

        if len(anchor_dists) >= 2:
            # Use least squares trilateration
            anchor_ids = list(anchor_dists.keys())
            n_anchors = len(anchor_ids)

            # Build linear system for trilateration
            # Using first anchor as reference
            ref_id = anchor_ids[0]
            ref_pos = anchors[ref_id]
            ref_dist = anchor_dists[ref_id]

            A = []
            b = []
            for i in range(1, n_anchors):
                aid = anchor_ids[i]
                apos = anchors[aid]
                adist = anchor_dists[aid]

                # Linear equation from difference of squared distances
                A.append([2*(apos[0] - ref_pos[0]), 2*(apos[1] - ref_pos[1])])
                b.append([ref_dist**2 - adist**2 + np.sum(apos**2) - np.sum(ref_pos**2)])

            if len(A) > 0:
                A = np.array(A)
                b = np.array(b).flatten()

                # Solve using least squares
                try:
                    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    initial_positions[idx*2:(idx+1)*2] = pos
                except:
                    # Fallback to center if trilateration fails
                    initial_positions[idx*2:(idx+1)*2] = [25.0, 25.0]
            else:
                initial_positions[idx*2:(idx+1)*2] = [25.0, 25.0]
        else:
            # Not enough anchor measurements, use center
            initial_positions[idx*2:(idx+1)*2] = [25.0, 25.0]

    return initial_positions


def run_localization(config: dict, measurements: list, anchors: dict, unknowns: dict) -> dict:
    """Run the robust localization solver"""

    print("\n" + "="*70)
    print("RUNNING LOCALIZATION")
    print("="*70)

    # Initialize solver
    solver = RobustLocalizer(
        dimension=2,
        huber_delta=config['solver']['huber_delta']
    )
    solver.max_iterations = config['system']['max_iterations']
    solver.convergence_threshold = config['system']['convergence_threshold']

    # Smart initialization using trilateration
    n_unknowns = len(unknowns)
    initial_positions = smart_initialization(unknowns, measurements, anchors)

    print(f"\nInitializing {n_unknowns} unknown nodes with trilateration...")

    # Create filtered measurement list (only those involving unknowns)
    filtered_measurements = []
    for edge in measurements:
        # Only include measurements that involve at least one unknown node
        if edge.node_i in unknowns or edge.node_j in unknowns:
            filtered_measurements.append(edge)

    print(f"Filtered {len(measurements)} to {len(filtered_measurements)} measurements (removed anchor-only)")

    # Debug: count measurement types
    n_anchor_unknown = sum(1 for m in filtered_measurements
                           if (m.node_i in anchors) != (m.node_j in anchors))
    n_unknown_unknown = sum(1 for m in filtered_measurements
                           if m.node_i in unknowns and m.node_j in unknowns)
    print(f"  Anchor-Unknown: {n_anchor_unknown}")
    print(f"  Unknown-Unknown: {n_unknown_unknown}")

    # Remap node IDs for solver (expects sequential IDs)
    unknown_ids = sorted(unknowns.keys())
    id_mapping = {}

    # Anchors keep their IDs
    for aid in anchors:
        id_mapping[aid] = aid

    # Unknowns get remapped to sequential IDs starting from len(anchors)
    for i, uid in enumerate(unknown_ids):
        id_mapping[uid] = len(anchors) + i

    # Remap measurements
    remapped_measurements = []
    for m in filtered_measurements:
        remapped = MeasurementEdge(
            node_i=id_mapping[m.node_i],
            node_j=id_mapping[m.node_j],
            distance=m.distance,
            quality=m.quality,
            variance=m.variance
        )
        remapped_measurements.append(remapped)

    # Remap anchors
    remapped_anchors = {id_mapping[aid]: anchors[aid] for aid in anchors}

    # Solve with remapped IDs
    optimized_positions, info = solver.solve(
        initial_positions,
        remapped_measurements,
        remapped_anchors
    )

    # Extract results
    results = {}
    unknown_ids = sorted(unknowns.keys())

    print("\nLocalization Results:")
    print("-" * 50)

    errors = []
    for i, uid in enumerate(unknown_ids):
        if i * 2 + 1 < len(optimized_positions):
            est_pos = optimized_positions[i*2:(i+1)*2]
        else:
            est_pos = optimized_positions

        true_pos = unknowns[uid]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)

        results[uid] = {
            'true': true_pos,
            'estimated': est_pos,
            'error': error
        }

        # Only print first few and worst cases
        if i < 3 or error > 5.0:
            print(f"Node {uid:2d}: True=({true_pos[0]:5.1f},{true_pos[1]:5.1f}) "
                  f"Est=({est_pos[0]:5.1f},{est_pos[1]:5.1f}) Err={error:.2f}m")

    # Calculate overall metrics
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    median_error = np.median(errors)

    print(f"\n" + "="*70)
    print(f"OVERALL STATISTICS")
    print(f"="*70)
    print(f"  RMSE:         {rmse:.2f}m")
    print(f"  Median Error: {median_error:.2f}m")
    print(f"  Max Error:    {max_error:.2f}m")
    print(f"  Mean Error:   {np.mean(errors):.2f}m")
    print(f"  Iterations:   {info['iterations']}")
    print(f"  Converged:    {info['converged']}")

    # Error distribution
    print(f"\nError Distribution:")
    print(f"  < 1m:  {sum(e < 1.0 for e in errors)} nodes ({100*sum(e < 1.0 for e in errors)/len(errors):.1f}%)")
    print(f"  < 2m:  {sum(e < 2.0 for e in errors)} nodes ({100*sum(e < 2.0 for e in errors)/len(errors):.1f}%)")
    print(f"  < 5m:  {sum(e < 5.0 for e in errors)} nodes ({100*sum(e < 5.0 for e in errors)/len(errors):.1f}%)")
    print(f"  > 5m:  {sum(e >= 5.0 for e in errors)} nodes ({100*sum(e >= 5.0 for e in errors)/len(errors):.1f}%)")

    return results, info


def plot_results(config: dict, anchors: dict, results: dict):
    """Visualize the localization results"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Node positions
    ax1 = axes[0]

    # Plot anchors
    anchor_pos = np.array(list(anchors.values()))
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
               s=300, c='red', marker='^', label='Anchors', zorder=5, edgecolors='black', linewidth=2)

    # Plot true and estimated positions
    for uid, result in results.items():
        true = result['true']
        est = result['estimated']
        error = result['error']

        # Color based on error magnitude
        if error < 1.0:
            color = 'green'
        elif error < 3.0:
            color = 'orange'
        else:
            color = 'red'

        # True position
        ax1.scatter(true[0], true[1], s=100, c='lightblue', marker='o', alpha=0.5)
        # Estimated position
        ax1.scatter(est[0], est[1], s=50, c=color, marker='x')
        # Error line
        ax1.plot([true[0], est[0]], [true[1], est[1]], 'k--', alpha=0.2, linewidth=0.5)

    ax1.set_xlim(-5, 55)
    ax1.set_ylim(-5, 55)
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('30-Node Localization Results (50×50m)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')

    # Right plot: Error histogram
    ax2 = axes[1]
    errors = [r['error'] for r in results.values()]

    ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
    ax2.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}m')
    ax2.set_xlabel('Localization Error (m)', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'30-Node Large Area Localization\nRMSE: {np.sqrt(np.mean(np.array(errors)**2)):.2f}m',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('30_node_localization_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nFigure saved to: 30_node_localization_results.png")


def main():
    """Run the 30-node large area demo"""

    print("="*70)
    print("30-NODE LARGE AREA LOCALIZATION DEMO")
    print("="*70)
    print("Configuration: configs/30_node_large.yaml")

    # Load configuration
    config = load_config()

    # Print key parameters
    print(f"\nSystem Parameters:")
    print(f"  Area: 50×50 meters")
    print(f"  Nodes: 30 (4 anchors, 26 unknowns)")
    print(f"  Environment: {config['channel']['environment']}")
    print(f"  NLOS probability: {config['channel']['nlos_probability']*100:.0f}%")
    print(f"  Path loss exponent: {config['channel']['path_loss_exponent']}")

    # Generate measurements
    measurements, anchors, unknowns, measurement_details = generate_measurements(config)

    # Run localization
    results, info = run_localization(config, measurements, anchors, unknowns)

    # Plot results
    plot_results(config, anchors, results)

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()