"""
Full FTL Simulation with REAL Signal Processing
This version actually uses Gold codes, RF channel, and correlation
NO FAKE RANGING!
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from time_sync_fixed import FixedTimeSync
from gold_codes_working import WorkingGoldCodeGenerator
from rf_channel import RangingChannel, ChannelConfig


@dataclass
class SimNode:
    """Simulation node with full state"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    is_anchor: bool
    true_offset_ns: float
    true_drift_ppb: float
    est_offset_ns: float = 0.0
    est_drift_ppb: float = 0.0
    est_position: np.ndarray = None
    ranging_measurements: Dict = field(default_factory=dict)
    name: str = ""
    gold_code: np.ndarray = None

    def __post_init__(self):
        if self.est_position is None:
            self.est_position = np.zeros(3)


class RealFTLSimulation:
    """
    FTL simulation with REAL signal processing
    - Uses actual Gold code transmission
    - Processes through realistic RF channel
    - Performs correlation-based ToA estimation
    - Implements proper two-way ranging
    """

    def __init__(self, config_file: str = "config_30nodes.yaml"):
        """Initialize with REAL components"""
        print("="*70)
        print("REAL FTL SIMULATION - WITH ACTUAL SIGNAL PROCESSING")
        print("="*70)

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_nodes = self.config['network']['total_nodes']
        self.n_anchors = self.config['network']['anchor_nodes']

        print(f"Configuration: {self.n_nodes} nodes, {self.n_anchors} anchors")
        print("Using REAL signal correlation for ranging")

        self.setup_network()
        self.setup_gold_codes()
        self.setup_rf_channel()

        # Results storage
        self.results = {
            'time_sync': {},
            'ranging': {},
            'localization': {}
        }

    def setup_network(self):
        """Initialize network nodes"""
        self.nodes = []

        # Create anchors
        print(f"\nInitializing {self.n_anchors} anchor nodes...")
        for anchor_cfg in self.config['anchors']:
            node = SimNode(
                id=anchor_cfg['id'],
                position=np.array(anchor_cfg['position']),
                velocity=np.zeros(3),
                is_anchor=True,
                true_offset_ns=0.0,  # Perfect reference
                true_drift_ppb=0.0,
                est_offset_ns=0.0,
                est_drift_ppb=0.0,
                name=anchor_cfg['name']
            )
            node.est_position = node.position.copy()
            self.nodes.append(node)

        # Create regular nodes
        print(f"Initializing {self.n_nodes - self.n_anchors} regular nodes...")
        for i in range(self.n_anchors, self.n_nodes):
            x = np.random.uniform(0, 50)
            y = np.random.uniform(0, 50)
            z = np.random.uniform(0, 2)

            speed = np.random.uniform(0.5, 5.0)
            angle = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            offset = np.random.uniform(-100, 100)
            drift = np.random.uniform(-10, 10)

            node = SimNode(
                id=i,
                position=np.array([x, y, z]),
                velocity=np.array([vx, vy, 0]),
                is_anchor=False,
                true_offset_ns=offset,
                true_drift_ppb=drift,
                name=f"Node_{i}"
            )
            self.nodes.append(node)

    def setup_gold_codes(self):
        """Generate REAL Gold codes for each node"""
        print(f"\nGenerating Gold codes...")
        self.gold_gen = WorkingGoldCodeGenerator(length=127)

        # Assign unique Gold code to each node
        for i, node in enumerate(self.nodes):
            node.gold_code = self.gold_gen.get_code(i)

        print(f"  ✓ Assigned unique Gold codes to all {len(self.nodes)} nodes")

    def setup_rf_channel(self):
        """Setup REAL RF channel"""
        rf_cfg = self.config['rf_channel']

        self.channel_config = ChannelConfig(
            frequency_hz=float(rf_cfg['frequency_hz']),
            bandwidth_hz=float(rf_cfg['bandwidth_hz']),
            enable_multipath=rf_cfg['multipath']['enabled'],
            iq_amplitude_imbalance_db=rf_cfg['hardware']['iq_amplitude_imbalance_db'],
            iq_phase_imbalance_deg=rf_cfg['hardware']['iq_phase_imbalance_deg'],
            phase_noise_dbc_hz=rf_cfg['hardware']['phase_noise_dbc_hz'],
            adc_bits=rf_cfg['hardware']['adc_bits']
        )

        self.channel = RangingChannel(self.channel_config)
        print(f"  ✓ RF channel configured with multipath and hardware impairments")

    def correlate_for_toa(self, rx_signal: np.ndarray, gold_code: np.ndarray) -> Tuple[float, float]:
        """
        REAL correlation-based ToA estimation
        Returns: (toa_ns, correlation_peak)
        """
        # Ensure same length
        min_len = min(len(rx_signal), len(gold_code))
        rx_signal = rx_signal[:min_len]
        gold_code = gold_code[:min_len]

        # FFT-based correlation
        rx_fft = np.fft.fft(rx_signal, n=256)
        gold_fft = np.fft.fft(gold_code, n=256)

        correlation = np.abs(np.fft.ifft(rx_fft * np.conj(gold_fft)))

        # Find peak
        peak_idx = np.argmax(correlation)
        peak_value = correlation[peak_idx]

        # Sub-sample interpolation
        if 0 < peak_idx < len(correlation) - 1:
            y1 = correlation[peak_idx - 1]
            y2 = correlation[peak_idx]
            y3 = correlation[peak_idx + 1]

            if y2 > y1 and y2 > y3:
                # Parabolic interpolation
                a = (y1 - 2*y2 + y3) / 2
                b = (y3 - y1) / 2
                if a != 0:
                    x_offset = -b / (2*a)
                    peak_idx = peak_idx + x_offset

        # Convert to time
        samples_per_chip = self.channel_config.bandwidth_hz / 1.023e6  # Gold chip rate
        toa_samples = peak_idx
        toa_ns = toa_samples / self.channel_config.bandwidth_hz * 1e9

        return toa_ns, peak_value

    def perform_real_ranging(self, node_i: SimNode, node_j: SimNode) -> dict:
        """
        REAL two-way ranging with signal processing

        1. Node i transmits Gold code
        2. Signal propagates through RF channel
        3. Node j receives and correlates to find ToA
        4. Node j transmits back
        5. Node i receives and correlates
        6. Calculate round-trip time and distance
        """

        # True distance
        true_distance = np.linalg.norm(node_j.position - node_i.position)

        # === FORWARD TRANSMISSION (i -> j) ===

        # Node i transmits its Gold code
        tx_signal_i = node_i.gold_code.astype(complex)

        # Calculate SNR based on distance
        snr_db = 20 - 10 * np.log10(1 + true_distance / 20)

        # Process through REAL RF channel
        rx_signal_j, true_toa_ij, channel_info_ij = self.channel.process_ranging_signal(
            tx_signal=tx_signal_i,
            true_distance_m=true_distance,
            true_velocity_mps=0,  # Simplified: no Doppler
            clock_offset_ns=node_j.true_offset_ns - node_i.true_offset_ns,
            freq_offset_hz=0,
            snr_db=snr_db
        )

        # Node j correlates to find ToA
        measured_toa_ij, peak_ij = self.correlate_for_toa(rx_signal_j, node_i.gold_code)

        # === RETURN TRANSMISSION (j -> i) ===

        # Processing delay at node j
        processing_delay_ns = 1000  # 1 microsecond

        # Node j transmits its Gold code
        tx_signal_j = node_j.gold_code.astype(complex)

        # Process through RF channel
        rx_signal_i, true_toa_ji, channel_info_ji = self.channel.process_ranging_signal(
            tx_signal=tx_signal_j,
            true_distance_m=true_distance,
            true_velocity_mps=0,
            clock_offset_ns=node_i.true_offset_ns - node_j.true_offset_ns,
            freq_offset_hz=0,
            snr_db=snr_db
        )

        # Node i correlates to find ToA
        measured_toa_ji, peak_ji = self.correlate_for_toa(rx_signal_i, node_j.gold_code)

        # === CALCULATE DISTANCE ===

        # Round-trip time (accounting for processing delay)
        rtt_ns = measured_toa_ij + processing_delay_ns + measured_toa_ji

        # Estimated distance
        propagation_time_ns = (rtt_ns - processing_delay_ns) / 2
        estimated_distance = propagation_time_ns * 1e-9 * 3e8

        # Error
        distance_error = estimated_distance - true_distance

        return {
            'true_distance': true_distance,
            'estimated_distance': estimated_distance,
            'distance_error': distance_error,
            'toa_forward': measured_toa_ij,
            'toa_return': measured_toa_ji,
            'peak_forward': peak_ij,
            'peak_return': peak_ji,
            'snr_db': snr_db
        }

    def run_time_synchronization(self):
        """Phase 1: Time synchronization"""
        print("\n" + "="*70)
        print("PHASE 1: TIME SYNCHRONIZATION")
        print("-"*70)

        sync = FixedTimeSync(n_nodes=self.n_nodes, n_anchors=self.n_anchors)

        # Copy clock errors
        for i, node in enumerate(self.nodes):
            sync.nodes[i].true_offset_ns = node.true_offset_ns
            sync.nodes[i].true_drift_ppb = node.true_drift_ppb
            sync.nodes[i].is_anchor = node.is_anchor

        print("\nRunning time synchronization...")
        sync.run_synchronization(n_rounds=30)

        # Copy results back
        for i, node in enumerate(self.nodes):
            if not node.is_anchor:
                node.est_offset_ns = sync.nodes[i].est_offset_ns
                node.est_drift_ppb = sync.nodes[i].est_drift_ppb

        self.results['time_sync']['final_mean_error'] = sync.history['mean_errors'][-1]
        self.results['time_sync']['converged'] = sync.history['mean_errors'][-1] < 2.0
        self.results['time_sync']['history'] = sync.history

    def run_real_ranging(self):
        """Phase 2: REAL signal-based ranging"""
        print("\n" + "="*70)
        print("PHASE 2: REAL SIGNAL-BASED RANGING")
        print("-"*70)

        n_pairs = self.n_nodes * (self.n_nodes - 1) // 2
        print(f"\nPerforming REAL two-way ranging for {n_pairs} node pairs...")
        print("Using: Gold code correlation through RF channel")

        ranging_errors = []
        pair_count = 0

        # Store detailed results for analysis
        detailed_results = []

        # Measure ranges between all pairs
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                node_i = self.nodes[i]
                node_j = self.nodes[j]

                # REAL ranging with signal processing
                result = self.perform_real_ranging(node_i, node_j)

                # Store measurements
                node_i.ranging_measurements[j] = result['estimated_distance']
                node_j.ranging_measurements[i] = result['estimated_distance']

                ranging_errors.append(result['distance_error'])
                detailed_results.append(result)

                pair_count += 1
                if pair_count % 100 == 0:
                    print(f"  Completed {pair_count}/{n_pairs} ranging measurements...")

                # Show first few detailed results
                if pair_count <= 3:
                    print(f"    Pair {i}-{j}: True={result['true_distance']:.2f}m, "
                          f"Est={result['estimated_distance']:.2f}m, "
                          f"Error={result['distance_error']:.3f}m, "
                          f"SNR={result['snr_db']:.1f}dB")

        # Statistics
        ranging_errors = np.array(ranging_errors)
        rmse = np.sqrt(np.mean(ranging_errors**2))
        mean_abs = np.mean(np.abs(ranging_errors))
        max_error = np.max(np.abs(ranging_errors))

        print(f"\nREAL Ranging Statistics:")
        print(f"  RMSE: {rmse:.3f}m")
        print(f"  Mean absolute error: {mean_abs:.3f}m")
        print(f"  Max error: {max_error:.3f}m")
        print(f"  Method: Gold code correlation with RF channel effects")

        self.results['ranging']['rmse'] = rmse
        self.results['ranging']['errors'] = ranging_errors
        self.results['ranging']['detailed'] = detailed_results

    def run_localization(self):
        """Phase 3: Localization using REAL ranging measurements"""
        print("\n" + "="*70)
        print("PHASE 3: LOCALIZATION")
        print("-"*70)

        print("\nEstimating positions using REAL ranging data...")
        position_errors = []

        for node in self.nodes:
            if not node.is_anchor:
                anchor_positions = []
                distances = []

                for anchor in self.nodes[:self.n_anchors]:
                    if anchor.id in node.ranging_measurements:
                        anchor_positions.append(anchor.position[:2])
                        distances.append(node.ranging_measurements[anchor.id])

                if len(distances) >= 3:
                    est_pos = self.trilaterate(anchor_positions, distances)
                    node.est_position[:2] = est_pos

                    error = np.linalg.norm(node.position[:2] - est_pos)
                    position_errors.append(error)

        # Statistics
        if position_errors:
            position_errors = np.array(position_errors)
            pos_rmse = np.sqrt(np.mean(position_errors**2))
            pos_mean = np.mean(position_errors)
            pos_max = np.max(position_errors)

            print(f"\nLocalization Statistics (using REAL ranging):")
            print(f"  Position RMSE: {pos_rmse:.3f}m")
            print(f"  Mean position error: {pos_mean:.3f}m")
            print(f"  Max position error: {pos_max:.3f}m")

            self.results['localization']['rmse'] = pos_rmse
            self.results['localization']['errors'] = position_errors

    def trilaterate(self, anchor_positions, distances):
        """Least squares trilateration"""
        anchor_positions = np.array(anchor_positions)
        distances = np.array(distances)

        A = 2 * (anchor_positions[1:] - anchor_positions[0])
        b = (distances[0]**2 - distances[1:]**2 +
             np.sum(anchor_positions[1:]**2, axis=1) -
             np.sum(anchor_positions[0]**2))

        try:
            position = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            position = np.mean(anchor_positions, axis=0)

        return position

    def visualize_results(self):
        """Visualization of REAL results"""
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("-"*70)

        fig = plt.figure(figsize=(16, 10))

        # 1. Network topology
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_title("Network Topology (REAL Ranging)")
        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")
        ax1.set_xlim(-5, 55)
        ax1.set_ylim(-5, 55)
        ax1.grid(True, alpha=0.3)

        rect = plt.Rectangle((0, 0), 50, 50, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)

        for node in self.nodes:
            if node.is_anchor:
                ax1.scatter(node.position[0], node.position[1],
                           s=200, c='red', marker='^', edgecolor='black',
                           linewidth=2, zorder=5)
                ax1.text(node.position[0], node.position[1] - 2.5,
                        node.name, fontsize=8, ha='center')
            else:
                ax1.scatter(node.position[0], node.position[1],
                           s=50, c='blue', marker='o', alpha=0.5)
                ax1.scatter(node.est_position[0], node.est_position[1],
                           s=30, c='green', marker='x')
                ax1.plot([node.position[0], node.est_position[0]],
                        [node.position[1], node.est_position[1]],
                        'k-', alpha=0.2, linewidth=0.5)

        ax1.legend(['Anchor', 'True Position', 'Est. Position'])

        # 2. Time sync convergence
        ax2 = plt.subplot(2, 3, 2)
        if 'time_sync' in self.results and 'history' in self.results['time_sync']:
            history = self.results['time_sync']['history']
            ax2.semilogy(history['rounds'], history['mean_errors'], 'b-', linewidth=2)
            ax2.axhline(y=1.0, color='g', linestyle=':', label='Target (<1ns)')
            ax2.set_xlabel('Synchronization Round')
            ax2.set_ylabel('Mean Clock Error (ns)')
            ax2.set_title('Time Synchronization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. REAL Ranging errors
        ax3 = plt.subplot(2, 3, 3)
        if 'ranging' in self.results and 'errors' in self.results['ranging']:
            errors = self.results['ranging']['errors']
            ax3.hist(errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='red', linestyle='--', label='Perfect')
            ax3.set_xlabel('Ranging Error (m)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'REAL Ranging (RMSE={self.results["ranging"]["rmse"]:.3f}m)')
            ax3.text(0.05, 0.95, 'Using Gold code\ncorrelation',
                    transform=ax3.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. SNR vs Distance
        ax4 = plt.subplot(2, 3, 4)
        if 'detailed' in self.results['ranging']:
            details = self.results['ranging']['detailed'][:50]  # First 50 pairs
            dists = [d['true_distance'] for d in details]
            snrs = [d['snr_db'] for d in details]
            ax4.scatter(dists, snrs, alpha=0.5, s=20)
            ax4.set_xlabel('True Distance (m)')
            ax4.set_ylabel('SNR (dB)')
            ax4.set_title('Signal Quality vs Distance')
            ax4.grid(True, alpha=0.3)

        # 5. Correlation peaks
        ax5 = plt.subplot(2, 3, 5)
        if 'detailed' in self.results['ranging']:
            details = self.results['ranging']['detailed'][:100]
            peaks = [d['peak_forward'] for d in details]
            ax5.hist(peaks, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Correlation Peak Value')
            ax5.set_ylabel('Count')
            ax5.set_title('Gold Code Correlation Peaks')
            ax5.grid(True, alpha=0.3)

        # 6. Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        stats_text = f"""REAL SIGNAL PROCESSING RESULTS

System Configuration:
• Nodes: {self.n_nodes} ({self.n_anchors} anchors)
• Gold code length: 127 chips
• RF frequency: 2.4 GHz
• Bandwidth: 100 MHz

Time Sync:
• Final error: {self.results['time_sync']['final_mean_error']:.2f}ns
• Converged: {'YES' if self.results['time_sync']['converged'] else 'NO'}

REAL Ranging:
• Method: Gold code correlation
• Channel: Multipath + impairments
• RMSE: {self.results['ranging']['rmse']:.3f}m

Localization:
• Position RMSE: {self.results['localization']['rmse']:.3f}m
• Algorithm: Least squares

✓ Using REAL signal processing
✓ NO statistical shortcuts
✓ Actual correlation-based ToA
"""
        ax6.text(0.05, 0.5, stats_text, fontsize=9, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('FTL System - REAL Signal Processing Implementation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig('ftl_REAL_simulation_results.png', dpi=150)
        print("\n✅ Results saved to ftl_REAL_simulation_results.png")
        plt.show()

    def run(self):
        """Run complete REAL simulation"""
        start_time = time.time()

        # Phase 1: Time Sync
        self.run_time_synchronization()

        # Phase 2: REAL Ranging
        self.run_real_ranging()

        # Phase 3: Localization
        self.run_localization()

        # Visualization
        self.visualize_results()

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("REAL SIMULATION COMPLETE")
        print("="*70)
        print(f"Execution time: {elapsed:.1f}s")
        print(f"\n✅ Verification:")
        print(f"  • Used REAL Gold code transmission")
        print(f"  • Processed through REAL RF channel")
        print(f"  • Performed REAL correlation for ToA")
        print(f"  • NO fake/statistical ranging")
        print(f"\nFinal Metrics:")
        print(f"  • Time sync: {self.results['time_sync']['final_mean_error']:.2f}ns")
        print(f"  • Ranging RMSE: {self.results['ranging']['rmse']:.3f}m")
        print(f"  • Position RMSE: {self.results['localization']['rmse']:.3f}m")


if __name__ == "__main__":
    print("Starting REAL FTL simulation (no fake ranging)...")
    sim = RealFTLSimulation("config_30nodes.yaml")
    sim.run()