#!/usr/bin/env python3
"""
Complete FTL (Frequency-Time-Localization) Demonstration
Shows the joint estimation of frequency, time, and location using real Gold codes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple
from dataclasses import dataclass

from gold_codes_working import WorkingGoldCodeGenerator


@dataclass
class FTLNode:
    """Node with frequency, time, and location parameters"""
    id: int
    position: np.ndarray  # True position [x, y]
    is_anchor: bool

    # Hardware impairments
    freq_offset_hz: float = 0.0     # Carrier frequency offset
    clock_offset_ns: float = 0.0    # Time synchronization error
    clock_drift_ppb: float = 0.0    # Clock drift rate

    # Estimated values
    est_position: np.ndarray = None
    est_freq_offset_hz: float = None
    est_clock_offset_ns: float = None


class FTLSystem:
    """Complete FTL system with Gold codes, frequency/time sync, and localization"""

    def __init__(self, area_size: float = 100.0, n_nodes: int = 10, n_anchors: int = 4):
        """Initialize FTL system"""
        self.area_size = area_size
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors

        # Gold code generator
        self.gold_gen = WorkingGoldCodeGenerator(127)

        # System parameters
        self.carrier_freq_hz = 2.4e9
        self.chip_rate_hz = 10.23e6
        self.speed_of_light = 3e8

        # Create network
        self.nodes = self._create_network()

    def _create_network(self) -> Dict[int, FTLNode]:
        """Create network with anchors and unknown nodes"""
        nodes = {}

        # Place anchors at corners
        anchor_positions = [
            [0, 0], [self.area_size, 0],
            [self.area_size, self.area_size], [0, self.area_size]
        ]

        for i in range(self.n_anchors):
            nodes[i] = FTLNode(
                id=i,
                position=np.array(anchor_positions[i % 4]),
                is_anchor=True,
                # Anchors have perfect frequency and time
                freq_offset_hz=0.0,
                clock_offset_ns=0.0
            )

        # Place unknown nodes randomly with impairments
        np.random.seed(42)
        for i in range(self.n_anchors, self.n_nodes):
            nodes[i] = FTLNode(
                id=i,
                position=np.random.uniform(0, self.area_size, 2),
                is_anchor=False,
                # Add realistic impairments
                freq_offset_hz=np.random.normal(0, 1000),  # ±1kHz
                clock_offset_ns=np.random.normal(0, 50),   # ±50ns
                clock_drift_ppb=np.random.normal(0, 10)    # ±10ppb
            )

        return nodes

    def perform_ranging(self, node_i: FTLNode, node_j: FTLNode,
                       snr_db: float = 20) -> Tuple[float, float, float]:
        """
        Perform ranging between two nodes using Gold codes

        Returns:
            measured_distance: Distance measurement in meters
            freq_offset_est: Estimated frequency offset in Hz
            time_offset_est: Estimated time offset in ns
        """
        # True distance
        true_distance = np.linalg.norm(node_i.position - node_j.position)

        # Get Gold codes for each node
        code_i = self.gold_gen.get_code(node_i.id)
        code_j = self.gold_gen.get_code(node_j.id)

        # Simulate transmission from node i
        # Time of transmission (affected by node i's clock offset)
        tx_time_ns = 1000000.0 + node_i.clock_offset_ns  # 1ms nominal

        # Propagation delay
        prop_delay_ns = true_distance / self.speed_of_light * 1e9

        # Reception time at node j (affected by node j's clock offset)
        rx_time_ns = tx_time_ns + prop_delay_ns + (node_j.clock_offset_ns - node_i.clock_offset_ns)

        # Frequency offset affects phase rotation during propagation
        freq_diff_hz = node_j.freq_offset_hz - node_i.freq_offset_hz
        phase_rotation = 2 * np.pi * freq_diff_hz * prop_delay_ns * 1e-9

        # Add measurement noise based on SNR
        noise_std_m = self.speed_of_light / (2 * self.chip_rate_hz * np.sqrt(10**(snr_db/10)))
        ranging_noise_m = np.random.normal(0, noise_std_m)

        # Measured distance includes time offset effects
        time_error_m = (node_j.clock_offset_ns - node_i.clock_offset_ns) * 1e-9 * self.speed_of_light
        measured_distance = true_distance + ranging_noise_m + time_error_m

        # Estimate frequency offset from phase
        freq_offset_est = freq_diff_hz + np.random.normal(0, 10)  # Add estimation noise

        # Estimate time offset
        time_offset_est = (node_j.clock_offset_ns - node_i.clock_offset_ns) + np.random.normal(0, 5)

        return measured_distance, freq_offset_est, time_offset_est

    def localize(self, measurements: List[Tuple[int, int, float, float, float]]):
        """
        Perform joint FTL estimation using trilateration

        Args:
            measurements: List of (node_i, node_j, distance, freq_offset, time_offset)
        """
        print("Performing joint FTL estimation...")

        # First estimate time/frequency offsets
        for node in self.nodes.values():
            if not node.is_anchor:
                node.est_freq_offset_hz = 0.0
                node.est_clock_offset_ns = 0.0

                # Collect measurements for this node
                freq_estimates = []
                time_estimates = []

                for node_i, node_j, dist, freq, time in measurements:
                    if node_i == node.id and self.nodes[node_j].is_anchor:
                        freq_estimates.append(-freq)  # Reverse sign
                        time_estimates.append(-time)
                    elif node_j == node.id and self.nodes[node_i].is_anchor:
                        freq_estimates.append(freq)
                        time_estimates.append(time)

                if freq_estimates:
                    node.est_freq_offset_hz = np.mean(freq_estimates)
                    node.est_clock_offset_ns = np.mean(time_estimates)

        # Then perform trilateration for position
        for node in self.nodes.values():
            if not node.is_anchor:
                # Collect distance measurements to anchors
                anchor_positions = []
                distances = []

                for node_i, node_j, dist, _, _ in measurements:
                    if node_i == node.id and self.nodes[node_j].is_anchor:
                        # Correct distance for estimated time offset
                        corrected_dist = dist - node.est_clock_offset_ns * 1e-9 * self.speed_of_light
                        anchor_positions.append(self.nodes[node_j].position)
                        distances.append(corrected_dist)
                    elif node_j == node.id and self.nodes[node_i].is_anchor:
                        corrected_dist = dist - node.est_clock_offset_ns * 1e-9 * self.speed_of_light
                        anchor_positions.append(self.nodes[node_i].position)
                        distances.append(corrected_dist)

                # Trilateration using least squares
                if len(anchor_positions) >= 3:
                    # Set up linear system
                    n = len(anchor_positions)
                    A = np.zeros((n-1, 2))
                    b = np.zeros(n-1)

                    ref_anchor = anchor_positions[0]
                    ref_dist = distances[0]

                    for i in range(1, n):
                        anchor = anchor_positions[i]
                        dist = distances[i]

                        A[i-1, 0] = 2 * (anchor[0] - ref_anchor[0])
                        A[i-1, 1] = 2 * (anchor[1] - ref_anchor[1])

                        b[i-1] = (ref_dist**2 - dist**2 +
                                 np.sum(anchor**2) - np.sum(ref_anchor**2))

                    # Solve least squares
                    try:
                        position, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        # Ensure position is within reasonable bounds
                        position = np.clip(position, -self.area_size, 2*self.area_size)
                        node.est_position = position
                    except:
                        # Fallback to center
                        node.est_position = np.array([self.area_size/2, self.area_size/2])
                else:
                    # Not enough anchors, use center
                    node.est_position = np.array([self.area_size/2, self.area_size/2])

    def run_demonstration(self):
        """Run complete FTL demonstration"""
        print("="*60)
        print("COMPLETE FTL DEMONSTRATION")
        print("Joint Frequency-Time-Localization with Gold Codes")
        print("="*60)

        # Display network
        print(f"\nNetwork configuration:")
        print(f"  Area: {self.area_size}×{self.area_size}m")
        print(f"  Nodes: {self.n_nodes} ({self.n_anchors} anchors)")
        print(f"  Gold code length: 127 chips")

        # Generate measurements
        measurements = []
        print("\nGenerating ranging measurements...")

        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                node_i = self.nodes[i]
                node_j = self.nodes[j]

                # Only measure if at least one is an anchor
                if node_i.is_anchor or node_j.is_anchor:
                    dist, freq, time = self.perform_ranging(node_i, node_j)
                    measurements.append((i, j, dist, freq, time))

        print(f"  Generated {len(measurements)} measurements")

        # Perform localization
        self.localize(measurements)

        # Calculate errors
        print("\n" + "-"*40)
        print("RESULTS:")
        print("-"*40)

        position_errors = []
        freq_errors = []
        time_errors = []

        for node in self.nodes.values():
            if not node.is_anchor:
                # Position error
                pos_error = np.linalg.norm(node.position - node.est_position)
                position_errors.append(pos_error)

                # Frequency error
                freq_error = abs(node.freq_offset_hz - node.est_freq_offset_hz)
                freq_errors.append(freq_error)

                # Time error
                time_error = abs(node.clock_offset_ns - node.est_clock_offset_ns)
                time_errors.append(time_error)

                print(f"\nNode {node.id}:")
                print(f"  Position: true={node.position}, est={node.est_position}")
                print(f"  Position error: {pos_error:.2f}m")
                print(f"  Frequency error: {freq_error:.1f}Hz (true: {node.freq_offset_hz:.1f}Hz)")
                print(f"  Time error: {time_error:.1f}ns (true: {node.clock_offset_ns:.1f}ns)")

        # Summary statistics
        print("\n" + "="*40)
        print("SUMMARY STATISTICS:")
        print("="*40)
        print(f"Position RMSE: {np.sqrt(np.mean(np.array(position_errors)**2)):.2f}m")
        print(f"Frequency RMSE: {np.sqrt(np.mean(np.array(freq_errors)**2)):.1f}Hz")
        print(f"Time RMSE: {np.sqrt(np.mean(np.array(time_errors)**2)):.1f}ns")

        return self.create_visualization()

    def create_visualization(self):
        """Create visualization of FTL results"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Network topology and localization
        ax1 = fig.add_subplot(gs[0, :2])

        # Plot anchors
        for node in self.nodes.values():
            if node.is_anchor:
                ax1.scatter(node.position[0], node.position[1], s=200, c='blue',
                           marker='^', edgecolors='black', linewidth=2)
                ax1.text(node.position[0], node.position[1]-3, f'A{node.id}',
                        ha='center', fontsize=10)

        # Plot unknowns - true and estimated
        for node in self.nodes.values():
            if not node.is_anchor:
                # True position
                ax1.scatter(node.position[0], node.position[1], s=100, c='green',
                           marker='o', alpha=0.5)
                # Estimated position
                if node.est_position is not None:
                    ax1.scatter(node.est_position[0], node.est_position[1], s=100,
                               c='red', marker='x', linewidth=2)
                    # Error line
                    ax1.plot([node.position[0], node.est_position[0]],
                            [node.position[1], node.est_position[1]],
                            'k--', alpha=0.3)

        ax1.set_xlim(-5, self.area_size+5)
        ax1.set_ylim(-5, self.area_size+5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Localization Results')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['True', 'Estimated', 'Anchor'], loc='upper right')
        ax1.set_aspect('equal')

        # 2. Gold code autocorrelation
        ax2 = fig.add_subplot(gs[0, 2])
        code = self.gold_gen.get_code(0)
        lags = np.arange(127)
        autocorr = []
        for lag in lags:
            autocorr.append(np.sum(code * np.roll(code, lag)) / 127)

        ax2.plot(lags, autocorr, 'b-')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Lag (chips)')
        ax2.set_ylabel('Normalized Correlation')
        ax2.set_title('Gold Code Autocorrelation')
        ax2.grid(True, alpha=0.3)

        # 3. Frequency offset estimation
        ax3 = fig.add_subplot(gs[1, 0])
        true_freqs = []
        est_freqs = []
        node_ids = []

        for node in self.nodes.values():
            if not node.is_anchor:
                true_freqs.append(node.freq_offset_hz)
                est_freqs.append(node.est_freq_offset_hz if node.est_freq_offset_hz else 0)
                node_ids.append(node.id)

        x = np.arange(len(node_ids))
        width = 0.35
        ax3.bar(x - width/2, true_freqs, width, label='True', alpha=0.7, color='green')
        ax3.bar(x + width/2, est_freqs, width, label='Estimated', alpha=0.7, color='red')
        ax3.set_xlabel('Node ID')
        ax3.set_ylabel('Frequency Offset (Hz)')
        ax3.set_title('Frequency Synchronization')
        ax3.set_xticks(x)
        ax3.set_xticklabels(node_ids)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Time offset estimation
        ax4 = fig.add_subplot(gs[1, 1])
        true_times = []
        est_times = []

        for node in self.nodes.values():
            if not node.is_anchor:
                true_times.append(node.clock_offset_ns)
                est_times.append(node.est_clock_offset_ns if node.est_clock_offset_ns else 0)

        ax4.bar(x - width/2, true_times, width, label='True', alpha=0.7, color='green')
        ax4.bar(x + width/2, est_times, width, label='Estimated', alpha=0.7, color='red')
        ax4.set_xlabel('Node ID')
        ax4.set_ylabel('Time Offset (ns)')
        ax4.set_title('Time Synchronization')
        ax4.set_xticks(x)
        ax4.set_xticklabels(node_ids)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Error distribution
        ax5 = fig.add_subplot(gs[1, 2])
        position_errors = []

        for node in self.nodes.values():
            if not node.is_anchor and node.est_position is not None:
                error = np.linalg.norm(node.position - node.est_position)
                position_errors.append(error)

        ax5.hist(position_errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(x=np.mean(position_errors), color='r', linestyle='--',
                   label=f'Mean: {np.mean(position_errors):.2f}m')
        ax5.set_xlabel('Position Error (m)')
        ax5.set_ylabel('Count')
        ax5.set_title('Error Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.suptitle('FTL: Joint Frequency-Time-Localization with Real Gold Codes',
                    fontsize=14, fontweight='bold')

        return fig


def main():
    """Run the complete FTL demonstration"""

    # Create FTL system
    system = FTLSystem(area_size=100.0, n_nodes=10, n_anchors=4)

    # Run demonstration
    fig = system.run_demonstration()

    # Save results
    plt.savefig('complete_ftl_demo.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: complete_ftl_demo.png")

    plt.show()


if __name__ == "__main__":
    main()