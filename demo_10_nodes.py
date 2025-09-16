#!/usr/bin/env python3
"""
10-Node Localization Demo with YAML Configuration
Demonstrates the complete system on a realistic indoor scenario
"""

import numpy as np
import yaml
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType
from src.rf.spread_spectrum import SpreadSpectrumGenerator, RangingCorrelator, WaveformConfig
import matplotlib.pyplot as plt

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_hardware_model(config: dict) -> dict:
    """Extract hardware assumptions from config"""
    hw = config['hardware']
    return {
        'bandwidth_hz': hw['bandwidth_mhz'] * 1e6,
        'sample_rate_hz': hw['sample_rate_msps'] * 1e6,
        'carrier_freq_hz': hw['carrier_freq_ghz'] * 1e9,
        'tx_power_dbm': hw['tx_power_dbm'],
        'noise_figure_db': hw['noise_figure_db'],
        'timestamp_resolution_ns': hw['timestamp_resolution_ns'],
        'clock_stability_ppm': hw['clock_stability_ppm'],
        'allan_deviation': hw['allan_deviation']
    }

def generate_measurements(config: dict) -> tuple:
    """Generate all pairwise measurements between nodes"""
    
    # Setup channel
    ch_config = ChannelConfig(
        carrier_freq_hz=config['hardware']['carrier_freq_ghz'] * 1e9,
        bandwidth_hz=config['hardware']['bandwidth_mhz'] * 1e6,
        path_loss_exponent=config['channel']['path_loss_exponent'],
        nlos_bias_mean_m=config['channel']['nlos_bias_mean_m'],
        nlos_bias_std_m=config['channel']['nlos_bias_std_m'],
        delay_spread_ns=config['channel']['multipath_delay_spread_ns'],
        noise_figure_db=config['hardware']['noise_figure_db']
    )
    channel = RangingChannel(ch_config)
    
    # Extract node positions
    nodes = config['nodes']
    positions = {node['id']: np.array(node['position']) for node in nodes}
    anchors = {node['id']: np.array(node['position']) for node in nodes if node['is_anchor']}
    unknowns = {node['id']: np.array(node['position']) for node in nodes if not node['is_anchor']}
    
    measurements = []
    measurement_details = []
    
    print("\n" + "="*70)
    print("GENERATING MEASUREMENTS")
    print("="*70)
    
    # Generate measurements between all node pairs
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i >= j:  # Skip self and duplicate pairs
                continue
                
            id_i = node_i['id']
            id_j = node_j['id']
            pos_i = positions[id_i]
            pos_j = positions[id_j]
            
            # Calculate true distance
            true_dist = np.linalg.norm(pos_i - pos_j)
            
            # Determine propagation type
            if np.random.rand() < config['channel']['nlos_probability']:
                prop_type = PropagationType.NLOS
            else:
                prop_type = PropagationType.LOS
            
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
            
            # Store details for analysis
            detail = {
                'pair': (id_i, id_j),
                'true_distance': true_dist,
                'measured_distance': meas['measured_distance_m'],
                'error': meas['measured_distance_m'] - true_dist,
                'propagation': prop_type.value,
                'snr_db': meas['snr_db'],
                'quality': meas['quality_score'],
                'std_m': meas['measurement_std_m']
            }
            measurement_details.append(detail)
            
            # Print measurement info
            if id_i in anchors or id_j in anchors:
                print(f"  {id_i}↔{id_j}: True={true_dist:5.2f}m, "
                      f"Meas={meas['measured_distance_m']:5.2f}m, "
                      f"Err={detail['error']:+5.2f}m, "
                      f"{prop_type.value:15s}, Q={meas['quality_score']:.2f}")
    
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
                    initial_positions[idx*2:(idx+1)*2] = [5.0, 5.0]
            else:
                initial_positions[idx*2:(idx+1)*2] = [5.0, 5.0]
        else:
            # Not enough anchor measurements, use center
            initial_positions[idx*2:(idx+1)*2] = [5.0, 5.0]

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
        # Remap node IDs to match solver expectations
        # Anchors: 0-3, Unknowns: remapped to sequential
        if edge.node_i in unknowns or edge.node_j in unknowns:
            filtered_measurements.append(edge)
    
    # Solve
    optimized_positions, info = solver.solve(
        initial_positions, 
        filtered_measurements, 
        anchors
    )
    
    # Extract results
    results = {}
    unknown_ids = sorted(unknowns.keys())
    for i, uid in enumerate(unknown_ids):
        if i * 2 + 1 < len(optimized_positions):
            est_pos = optimized_positions[i*2:(i+1)*2]
        else:
            # Fallback for single unknown case
            est_pos = optimized_positions
            
        true_pos = unknowns[uid]
        error = np.linalg.norm(est_pos - true_pos)
        
        results[uid] = {
            'true': true_pos,
            'estimated': est_pos,
            'error': error
        }
        
        print(f"\nNode {uid}:")
        print(f"  True:      ({true_pos[0]:5.2f}, {true_pos[1]:5.2f})")
        print(f"  Estimated: ({est_pos[0]:5.2f}, {est_pos[1]:5.2f})")
        print(f"  Error:     {error:.2f}m")
    
    # Calculate overall metrics
    errors = [r['error'] for r in results.values()]
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    
    print(f"\n" + "-"*40)
    print(f"Overall Statistics:")
    print(f"  RMSE:       {rmse:.2f}m")
    print(f"  Max Error:  {max_error:.2f}m")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Converged:  {info['converged']}")
    
    return results, info

def plot_results(config: dict, anchors: dict, results: dict, measurements_details: list):
    """Visualize the localization results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Node positions
    ax1 = axes[0]
    
    # Plot anchors
    anchor_pos = np.array(list(anchors.values()))
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               s=200, c='red', marker='^', label='Anchors', zorder=5)
    
    # Plot true and estimated positions
    for uid, result in results.items():
        true = result['true']
        est = result['estimated']
        
        # True position
        ax1.scatter(true[0], true[1], s=100, c='green', marker='o', alpha=0.5)
        # Estimated position
        ax1.scatter(est[0], est[1], s=100, c='blue', marker='x')
        # Error line
        ax1.plot([true[0], est[0]], [true[1], est[1]], 'k--', alpha=0.3)
        
        # Label with error
        ax1.annotate(f"{uid}\n{result['error']:.1f}m", 
                    xy=est, xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Node Localization Results')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Right plot: Measurement quality
    ax2 = axes[1]
    
    # Extract data for plotting
    errors = [m['error'] for m in measurements_details]
    qualities = [m['quality'] for m in measurements_details]
    
    # Color by propagation type
    colors = ['green' if m['propagation'] == 'line_of_sight' else 'red' 
             for m in measurements_details]
    
    ax2.scatter(qualities, errors, c=colors, alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Measurement Quality Score')
    ax2.set_ylabel('Measurement Error (m)')
    ax2.set_title('Measurement Quality vs Error')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='g', markersize=8, label='LOS'),
                      Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='r', markersize=8, label='NLOS')]
    ax2.legend(handles=legend_elements)
    
    plt.suptitle('10-Node Indoor Localization Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if configured
    if config['output']['save_trajectory']:
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'localization_results.png', dpi=150)
    
    if config['output']['plot_results']:
        plt.show()

def print_hardware_assumptions():
    """Print the hardware assumptions being simulated"""
    print("\n" + "="*70)
    print("HARDWARE ASSUMPTIONS")
    print("="*70)
    print("""
    RF Front-end:
    - Carrier Frequency: 2.4 GHz (ISM band)
    - Bandwidth: 100 MHz → 1.5m theoretical range resolution
    - TX Power: 20 dBm (100mW)
    - Noise Figure: 6 dB (typical commercial RF)
    
    Spread Spectrum:
    - Gold Codes: 1023 chips
    - Chip Rate: 100 Mcps (matches bandwidth)
    - Pilot Tones: 7 frequencies for frequency lock
    - Pulse Shaping: Root-Raised Cosine (β=0.35)
    
    Timing Hardware:
    - MAC/PHY Timestamp: 10ns resolution
    - Clock Stability: 10 ppm drift
    - Allan Deviation: 1e-10 (OCXO-grade)
    
    Signal Processing:
    - Sample Rate: 200 Msps (2x oversampling)
    - ADC: 12-bit resolution
    - Correlation: Parabolic sub-sample interpolation
    
    Expected Performance:
    - Range Accuracy: ~0.3-1m (LOS conditions)
    - Update Rate: 10-100 Hz (depends on TDMA)
    - Max Range: ~100m indoor
    """)

def main():
    """Run the complete 10-node demo"""
    
    # Load configuration
    config_path = "configs/10_node_demo.yaml"
    config = load_config(config_path)
    
    # Set random seed
    np.random.seed(config['system']['seed'])
    
    print("\n" + "="*70)
    print("10-NODE INDOOR LOCALIZATION DEMO")
    print("="*70)
    print(f"Configuration: {config_path}")
    
    # Print hardware assumptions
    print_hardware_assumptions()
    
    # Generate measurements
    measurements, anchors, unknowns, measurement_details = generate_measurements(config)
    
    print(f"\nGenerated {len(measurements)} measurements")
    print(f"  Anchors: {len(anchors)}")
    print(f"  Unknown nodes: {len(unknowns)}")
    
    # Run localization
    results, solver_info = run_localization(config, measurements, anchors, unknowns)
    
    # Plot results
    plot_results(config, anchors, results, measurement_details)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()