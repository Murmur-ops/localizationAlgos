"""
Realistic FTL System with Physics-Based RF Channel
Replaces fake propagation with real RF physics
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys
import os

# Import our realistic channel
from rf_channel import RangingChannel, ChannelConfig

# Import Gold codes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gold_codes_working import WorkingGoldCodeGenerator as ProperGoldCodeGenerator


@dataclass
class FTLNode:
    """Enhanced node with realistic parameters"""
    id: int
    position: np.ndarray
    is_anchor: bool

    # True hardware parameters (unknown)
    clock_offset_ns: float = 0.0
    clock_drift_ppb: float = 0.0
    freq_offset_hz: float = 0.0
    velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Estimated parameters
    est_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    est_clock_offset_ns: float = 0.0
    est_freq_offset_hz: float = 0.0

    # Kalman filter state for time sync
    time_sync_state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    time_sync_covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 100)


class RealisticFTLSystem:
    """FTL system with realistic RF channel and iterative synchronization"""

    def __init__(self, area_size: float = 100, n_nodes: int = 10, n_anchors: int = 4):
        self.area_size = area_size
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors

        # RF channel configuration
        self.channel_config = ChannelConfig(
            frequency_hz=2.4e9,
            bandwidth_hz=100e6,
            enable_multipath=True,
            iq_amplitude_imbalance_db=0.5,
            iq_phase_imbalance_deg=3.0,
            phase_noise_dbc_hz=-80,
            adc_bits=12
        )
        self.channel = RangingChannel(self.channel_config)

        # Gold code configuration
        self.gold_length = 127
        self.gold_generator = ProperGoldCodeGenerator(self.gold_length)

        # Protocol parameters
        self.sync_rounds = 10
        self.sync_interval_ms = 100

        # Initialize network
        self.nodes = {}
        self.initialize_network()

        # Tracking for convergence
        self.sync_history = []
        self.ranging_measurements = []

    def initialize_network(self):
        """Create network with realistic impairments"""

        # Anchors at corners with good clocks
        anchor_positions = [
            [0, 0], [self.area_size, 0],
            [0, self.area_size], [self.area_size, self.area_size]
        ]

        for i in range(self.n_anchors):
            self.nodes[i] = FTLNode(
                id=i,
                position=np.array(anchor_positions[i]),
                is_anchor=True,
                clock_offset_ns=np.random.normal(0, 10),      # ±10ns for anchors
                clock_drift_ppb=np.random.normal(0, 1),       # ±1 ppb drift
                freq_offset_hz=np.random.normal(0, 100)       # ±100Hz
            )

        # Unknown nodes with worse clocks
        for i in range(self.n_anchors, self.n_nodes):
            position = np.random.uniform(0, self.area_size, 2)
            velocity = np.random.uniform(-5, 5, 2)  # ±5 m/s movement

            self.nodes[i] = FTLNode(
                id=i,
                position=position,
                is_anchor=False,
                clock_offset_ns=np.random.uniform(-100, 100),  # ±100ns
                clock_drift_ppb=np.random.uniform(-10, 10),    # ±10 ppb
                freq_offset_hz=np.random.uniform(-1000, 1000), # ±1kHz
                velocity_mps=velocity,
                est_position=np.array([self.area_size/2, self.area_size/2])
            )

    def generate_ranging_signal(self, node_id: int) -> np.ndarray:
        """Generate Gold code ranging signal for a node"""
        # Get unique Gold code for this node
        gold_code = self.gold_generator.get_code(node_id % self.gold_length)

        # Upsample to match channel bandwidth
        samples_per_chip = 10
        signal = np.repeat(gold_code, samples_per_chip).astype(float)

        # Apply pulse shaping (simple raised cosine)
        # In real system would use RRC matched filter
        window = np.hamming(samples_per_chip)
        for i in range(0, len(signal), samples_per_chip):
            signal[i:i+samples_per_chip] *= window

        return signal.astype(complex)

    def correlate_ranging_signal(self, received: np.ndarray, template: np.ndarray) -> Tuple[float, float]:
        """
        Correlate received signal with template to find ToA
        Uses FFT-based correlation for efficiency
        """
        # FFT-based correlation (much faster than time-domain)
        correlation = np.abs(np.fft.ifft(
            np.fft.fft(received) * np.conj(np.fft.fft(template, len(received)))
        ))

        # Find peak
        peak_idx = np.argmax(correlation)
        peak_value = correlation[peak_idx]

        # Sub-sample interpolation (parabolic)
        if 0 < peak_idx < len(correlation) - 1:
            y1 = correlation[peak_idx - 1]
            y2 = correlation[peak_idx]
            y3 = correlation[peak_idx + 1]

            a = (y1 - 2*y2 + y3) / 2
            b = (y3 - y1) / 2

            if a != 0:
                x_offset = -b / (2*a)
                fine_peak_idx = peak_idx + x_offset
            else:
                fine_peak_idx = peak_idx
        else:
            fine_peak_idx = peak_idx

        # Convert to time
        samples_to_ns = 1e9 / self.channel_config.bandwidth_hz
        toa_ns = fine_peak_idx * samples_to_ns

        # Calculate SNR from correlation peak
        noise_floor = np.median(correlation)
        snr_linear = peak_value / noise_floor if noise_floor > 0 else 1
        snr_db = 10 * np.log10(snr_linear)

        return toa_ns, snr_db

    def perform_ranging_exchange(self, node_i: FTLNode, node_j: FTLNode,
                                round_num: int) -> dict:
        """
        Perform two-way ranging with realistic channel
        Includes iterative time synchronization
        """
        # Calculate true distance
        true_distance = np.linalg.norm(node_i.position - node_j.position)

        # Calculate relative velocity (for Doppler)
        relative_velocity = np.dot(
            node_j.velocity_mps - node_i.velocity_mps,
            (node_j.position - node_i.position) / true_distance
        )

        # Update clocks with drift
        dt_s = round_num * self.sync_interval_ms / 1000
        node_i.clock_offset_ns += node_i.clock_drift_ppb * dt_s
        node_j.clock_offset_ns += node_j.clock_drift_ppb * dt_s

        # Generate ranging signal from node i
        tx_signal_i = self.generate_ranging_signal(node_i.id)

        # Node i -> Node j transmission
        rx_signal_j, toa_ij_ns, channel_info_ij = self.channel.process_ranging_signal(
            tx_signal=tx_signal_i,
            true_distance_m=true_distance,
            true_velocity_mps=relative_velocity,
            clock_offset_ns=node_j.clock_offset_ns - node_i.clock_offset_ns,
            freq_offset_hz=node_j.freq_offset_hz - node_i.freq_offset_hz,
            snr_db=20 - 10*np.log10(1 + true_distance/50)  # Distance-dependent SNR
        )

        # Node j processes and replies
        processing_delay_ns = np.random.normal(1000, 100)  # 1µs ± 100ns

        # Generate ranging signal from node j
        tx_signal_j = self.generate_ranging_signal(node_j.id)

        # Node j -> Node i transmission
        rx_signal_i, toa_ji_ns, channel_info_ji = self.channel.process_ranging_signal(
            tx_signal=tx_signal_j,
            true_distance_m=true_distance,
            true_velocity_mps=-relative_velocity,
            clock_offset_ns=node_i.clock_offset_ns - node_j.clock_offset_ns,
            freq_offset_hz=node_i.freq_offset_hz - node_j.freq_offset_hz,
            snr_db=20 - 10*np.log10(1 + true_distance/50)
        )

        # Correlation-based ToA estimation
        template_i = self.generate_ranging_signal(node_i.id)
        template_j = self.generate_ranging_signal(node_j.id)

        measured_toa_ij, snr_ij = self.correlate_ranging_signal(rx_signal_j, template_i)
        measured_toa_ji, snr_ji = self.correlate_ranging_signal(rx_signal_i, template_j)

        # Calculate round-trip time and estimated distance
        rtt_ns = measured_toa_ij + measured_toa_ji + processing_delay_ns
        estimated_distance = float((rtt_ns * 1e-9 * self.channel.c) / 2)

        # Time synchronization update (NTP-like)
        clock_offset_estimate = (measured_toa_ij - measured_toa_ji) / 2

        return {
            'true_distance': true_distance,
            'estimated_distance': estimated_distance,
            'distance_error': estimated_distance - true_distance,
            'clock_offset_estimate': clock_offset_estimate,
            'snr_forward': snr_ij,
            'snr_return': snr_ji,
            'channel_info_forward': channel_info_ij,
            'channel_info_return': channel_info_ji
        }

    def kalman_time_sync_update(self, node: FTLNode, measurement: float, dt: float):
        """
        Kalman filter update for time synchronization
        State: [clock_offset, clock_drift]
        """
        # State transition matrix
        F = np.array([[1, dt], [0, 1]])

        # Process noise
        Q = np.diag([0.1**2, 0.01**2])

        # Measurement matrix (observe offset only)
        H = np.array([[1, 0]])

        # Measurement noise
        R = 5.0**2  # 5ns measurement uncertainty

        # Predict
        x_pred = F @ node.time_sync_state
        P_pred = F @ node.time_sync_covariance @ F.T + Q

        # Update
        y = measurement - (H @ x_pred)[0]  # Scalar innovation
        S = (H @ P_pred @ H.T)[0, 0] + R   # Scalar
        K = (P_pred @ H.T).flatten() / S    # 2x1 Kalman gain

        # New estimates
        node.time_sync_state = x_pred + K * y
        node.time_sync_covariance = (np.eye(2) - np.outer(K, H)) @ P_pred

        # Update estimated clock offset
        node.est_clock_offset_ns = node.time_sync_state[0]

    def run_iterative_synchronization(self):
        """
        Run multi-round synchronization protocol
        This is what real systems do - NOT single-pass!
        """
        print("="*60)
        print("ITERATIVE TIME SYNCHRONIZATION")
        print("="*60)

        for round_num in range(self.sync_rounds):
            print(f"\nRound {round_num + 1}/{self.sync_rounds}")

            round_errors = []

            # Each unknown node syncs with anchors
            for node_id in range(self.n_anchors, self.n_nodes):
                node = self.nodes[node_id]

                clock_measurements = []

                # Perform ranging with each anchor
                for anchor_id in range(self.n_anchors):
                    anchor = self.nodes[anchor_id]

                    # Two-way ranging exchange
                    result = self.perform_ranging_exchange(node, anchor, round_num)

                    # Store measurement
                    clock_measurements.append(result['clock_offset_estimate'])
                    self.ranging_measurements.append(result)

                # Kalman filter update with average measurement
                if clock_measurements:
                    avg_measurement = np.mean(clock_measurements)
                    dt = self.sync_interval_ms / 1000
                    self.kalman_time_sync_update(node, avg_measurement, dt)

                    # Track convergence
                    error = abs(node.clock_offset_ns - node.est_clock_offset_ns)
                    round_errors.append(error)

            # Report convergence
            if round_errors:
                mean_error = np.mean(round_errors)
                max_error = np.max(round_errors)
                print(f"  Clock sync error: mean={mean_error:.1f}ns, max={max_error:.1f}ns")

                self.sync_history.append({
                    'round': round_num + 1,
                    'mean_error_ns': mean_error,
                    'max_error_ns': max_error,
                    'converged': mean_error < 10
                })

    def perform_localization(self):
        """
        Localize nodes using synchronized measurements
        """
        print("\n" + "="*60)
        print("LOCALIZATION WITH SYNCHRONIZED CLOCKS")
        print("="*60)

        position_errors = []

        for node_id in range(self.n_anchors, self.n_nodes):
            node = self.nodes[node_id]

            # Collect corrected measurements to anchors
            anchor_positions = []
            corrected_distances = []

            for anchor_id in range(self.n_anchors):
                anchor = self.nodes[anchor_id]

                # Perform final ranging
                result = self.perform_ranging_exchange(node, anchor, self.sync_rounds)

                # Correct for estimated clock offset
                time_correction_m = float(node.est_clock_offset_ns) * 1e-9 * self.channel.c
                corrected_distance = result['estimated_distance'] - time_correction_m

                anchor_positions.append(anchor.position)
                corrected_distances.append(corrected_distance)

            # Trilateration
            if len(anchor_positions) >= 3:
                # Linear least squares
                n = len(anchor_positions)
                A = np.zeros((n-1, 2))
                b = np.zeros(n-1)

                ref_anchor = np.array(anchor_positions[0])
                ref_dist = corrected_distances[0]

                for i in range(1, n):
                    anchor_pos = np.array(anchor_positions[i])
                    A[i-1] = 2 * (anchor_pos - ref_anchor)
                    b[i-1] = (ref_dist**2 - corrected_distances[i]**2 +
                             np.sum(anchor_pos**2) - np.sum(ref_anchor**2))

                try:
                    est_position, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    node.est_position = est_position

                    error = np.linalg.norm(node.position - est_position)
                    position_errors.append(error)

                    print(f"Node {node_id}:")
                    print(f"  True position: ({node.position[0]:.1f}, {node.position[1]:.1f})")
                    print(f"  Est. position: ({est_position[0]:.1f}, {est_position[1]:.1f})")
                    print(f"  Error: {error:.2f}m")
                    print(f"  Clock error: {abs(node.clock_offset_ns - node.est_clock_offset_ns):.1f}ns")
                except:
                    print(f"Node {node_id}: Localization failed")

        if position_errors:
            rmse = np.sqrt(np.mean(np.array(position_errors)**2))
            print(f"\nFinal RMSE: {rmse:.2f}m")
            return rmse
        return None

    def plot_results(self):
        """Visualize synchronization and localization results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Time sync convergence
        ax = axes[0, 0]
        if self.sync_history:
            rounds = [r['round'] for r in self.sync_history]
            mean_errors = [r['mean_error_ns'] for r in self.sync_history]
            max_errors = [r['max_error_ns'] for r in self.sync_history]

            ax.semilogy(rounds, mean_errors, 'b-', label='Mean error', linewidth=2)
            ax.semilogy(rounds, max_errors, 'r--', label='Max error', linewidth=2)
            ax.axhline(y=10, color='g', linestyle=':', label='Target (<10ns)')
            ax.set_xlabel('Sync Round')
            ax.set_ylabel('Clock Offset Error (ns)')
            ax.set_title('Iterative Time Sync Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Channel effects
        ax = axes[0, 1]
        if self.ranging_measurements:
            distances = [r['true_distance'] for r in self.ranging_measurements[-20:]]
            errors = [r['distance_error'] for r in self.ranging_measurements[-20:]]

            ax.scatter(distances, errors, alpha=0.6)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('True Distance (m)')
            ax.set_ylabel('Distance Error (m)')
            ax.set_title('Ranging Errors with Realistic Channel')
            ax.grid(True, alpha=0.3)

        # 3. Localization results
        ax = axes[1, 0]
        for node in self.nodes.values():
            if node.is_anchor:
                ax.plot(node.position[0], node.position[1], 'b^', markersize=12, label='Anchor' if node.id == 0 else '')
            else:
                ax.plot(node.position[0], node.position[1], 'go', markersize=8, label='True' if node.id == self.n_anchors else '')
                ax.plot(node.est_position[0], node.est_position[1], 'rx', markersize=8, label='Estimated' if node.id == self.n_anchors else '')
                ax.plot([node.position[0], node.est_position[0]],
                       [node.position[1], node.est_position[1]], 'k-', alpha=0.3)

        ax.set_xlim(-10, self.area_size + 10)
        ax.set_ylim(-10, self.area_size + 10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Localization with Realistic RF Channel')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. SNR distribution
        ax = axes[1, 1]
        if self.ranging_measurements:
            snr_forward = [r['snr_forward'] for r in self.ranging_measurements]
            snr_return = [r['snr_return'] for r in self.ranging_measurements]

            ax.hist(snr_forward, bins=20, alpha=0.5, label='Forward', color='blue')
            ax.hist(snr_return, bins=20, alpha=0.5, label='Return', color='red')
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('Count')
            ax.set_title('Signal Quality Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ftl_realistic_results.png', dpi=150)
        print("\nResults saved to ftl_realistic_results.png")
        plt.show()


def main():
    """Run realistic FTL demonstration"""
    print("="*60)
    print("REALISTIC FTL SYSTEM DEMONSTRATION")
    print("With Physics-Based RF Channel & Iterative Sync")
    print("="*60)

    # Create system
    system = RealisticFTLSystem(area_size=100, n_nodes=10, n_anchors=4)

    print("\nSystem Configuration:")
    print(f"  Frequency: {system.channel_config.frequency_hz/1e9:.1f} GHz")
    print(f"  Bandwidth: {system.channel_config.bandwidth_hz/1e6:.0f} MHz")
    print(f"  Multipath: Enabled")
    print(f"  I/Q imbalance: {system.channel_config.iq_amplitude_imbalance_db:.1f} dB")
    print(f"  Phase noise: {system.channel_config.phase_noise_dbc_hz:.0f} dBc/Hz")
    print(f"  ADC bits: {system.channel_config.adc_bits}")

    # Run iterative synchronization (NOT single-pass!)
    system.run_iterative_synchronization()

    # Check convergence
    if system.sync_history:
        final = system.sync_history[-1]
        if final['converged']:
            print(f"\n✓ Time sync CONVERGED to <10ns in {len(system.sync_history)} rounds")
        else:
            print(f"\n⚠ Time sync did not fully converge (current: {final['mean_error_ns']:.1f}ns)")

    # Perform localization
    rmse = system.perform_localization()

    # Plot results
    system.plot_results()

    print("\n" + "="*60)
    print("KEY IMPROVEMENTS OVER FAKE SYSTEM:")
    print("="*60)
    print("✓ Real RF propagation with path loss, multipath, atmospheric effects")
    print("✓ Hardware impairments: I/Q imbalance, phase noise, ADC quantization")
    print("✓ Iterative time synchronization with Kalman filtering")
    print("✓ FFT-based correlation for efficiency")
    print("✓ Distance-dependent SNR modeling")
    print("✓ Two-way ranging protocol with processing delays")
    print(f"✓ Achieved {rmse:.2f}m RMSE with realistic channel effects")


if __name__ == "__main__":
    main()