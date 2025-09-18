"""
Realistic Iterative FTL Synchronization
Implements proper multi-round time sync like NTP/PTP
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class SyncExchange:
    """Single time sync exchange between two nodes"""
    t1_send: float      # Node i sends at its time t1
    t2_receive: float   # Node j receives at its time t2
    t3_reply: float     # Node j replies at its time t3
    t4_receive: float   # Node i receives at its time t4
    measured_toa: float # Ranging measurement from signal
    snr_db: float      # Signal quality

    @property
    def round_trip_time(self) -> float:
        """Total round trip time from initiator's perspective"""
        return (self.t4_receive - self.t1_send) - (self.t3_reply - self.t2_receive)

    @property
    def offset_estimate(self) -> float:
        """Estimated clock offset (like NTP)"""
        return ((self.t2_receive - self.t1_send) + (self.t3_reply - self.t4_receive)) / 2

    @property
    def delay_estimate(self) -> float:
        """Estimated one-way delay"""
        return ((self.t4_receive - self.t1_send) + (self.t3_reply - self.t2_receive)) / 2


@dataclass
class Node:
    """Network node with time sync state"""
    id: int
    position: np.ndarray
    is_anchor: bool

    # True clock parameters (unknown)
    clock_offset_ns: float = 0.0    # True offset from reference
    clock_drift_ppb: float = 0.0    # Parts per billion drift rate
    freq_offset_hz: float = 0.0     # Carrier frequency offset

    # Estimated parameters
    est_clock_offset_ns: float = 0.0
    est_clock_drift_ppb: float = 0.0
    est_freq_offset_hz: float = 0.0
    est_position: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Sync history for filtering
    sync_history: List[SyncExchange] = field(default_factory=list)
    offset_history: List[float] = field(default_factory=list)
    drift_history: List[float] = field(default_factory=list)

    def update_clock(self, dt_seconds: float):
        """Update clock with drift over time interval"""
        drift_ns = self.clock_drift_ppb * dt_seconds  # ppb * seconds = nanoseconds
        self.clock_offset_ns += drift_ns


class IterativeFTLSync:
    """Realistic iterative time synchronization for FTL"""

    def __init__(self, area_size: float = 100, n_nodes: int = 10, n_anchors: int = 4):
        self.area_size = area_size
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors
        self.speed_of_light = 3e8

        # Protocol parameters (similar to PTP/NTP)
        self.sync_interval_ms = 100      # Sync exchange interval
        self.sync_rounds = 10             # Number of sync rounds
        self.min_exchanges = 4           # Minimum exchanges before estimating
        self.filter_window = 8            # Moving average window

        # Initialize nodes
        self.nodes = {}
        self.initialize_network()

    def initialize_network(self):
        """Create network with realistic clock errors"""
        # Place anchors at corners
        anchor_positions = [
            [0, 0], [self.area_size, 0],
            [0, self.area_size], [self.area_size, self.area_size]
        ]

        # Create anchor nodes (with good clocks)
        for i in range(self.n_anchors):
            self.nodes[i] = Node(
                id=i,
                position=np.array(anchor_positions[i]),
                is_anchor=True,
                clock_offset_ns=np.random.normal(0, 10),      # ±10ns for anchors
                clock_drift_ppb=np.random.normal(0, 1),       # ±1 ppb drift
                freq_offset_hz=np.random.normal(0, 100)       # ±100Hz
            )

        # Create unknown nodes (with worse clocks)
        for i in range(self.n_anchors, self.n_nodes):
            position = np.random.uniform(0, self.area_size, 2)
            self.nodes[i] = Node(
                id=i,
                position=position,
                is_anchor=False,
                clock_offset_ns=np.random.uniform(-100, 100),  # ±100ns initial offset
                clock_drift_ppb=np.random.uniform(-10, 10),    # ±10 ppb drift
                freq_offset_hz=np.random.uniform(-1000, 1000), # ±1kHz offset
                est_position=np.array([self.area_size/2, self.area_size/2])  # Initial guess
            )

    def perform_sync_exchange(self, node_i: Node, node_j: Node,
                            current_time_ms: float) -> SyncExchange:
        """Perform two-way time transfer between nodes"""
        # Convert to seconds
        current_time = current_time_ms * 1e-3

        # True propagation delay
        true_distance = np.linalg.norm(node_i.position - node_j.position)
        true_delay_ns = true_distance / self.speed_of_light * 1e9

        # Node i initiates at its local time
        t1_send = current_time * 1e9 + node_i.clock_offset_ns

        # Node j receives after propagation delay
        t2_receive = current_time * 1e9 + true_delay_ns + node_j.clock_offset_ns

        # Node j processes and replies (add processing delay)
        processing_delay_ns = np.random.normal(1000, 100)  # 1µs ± 100ns processing
        t3_reply = t2_receive + processing_delay_ns

        # Node i receives reply
        t4_receive = current_time * 1e9 + true_delay_ns * 2 + processing_delay_ns + node_i.clock_offset_ns

        # Ranging measurement (with noise based on SNR)
        snr_db = 20 - 10 * np.log10(1 + true_distance/50)  # Distance-dependent SNR
        ranging_noise_m = 0.3 / np.sqrt(10**(snr_db/10))    # Noise based on SNR
        measured_distance = true_distance + np.random.normal(0, ranging_noise_m)
        measured_toa = measured_distance / self.speed_of_light * 1e9

        return SyncExchange(
            t1_send=t1_send,
            t2_receive=t2_receive,
            t3_reply=t3_reply,
            t4_receive=t4_receive,
            measured_toa=measured_toa,
            snr_db=snr_db
        )

    def estimate_clock_parameters(self, node: Node):
        """Estimate clock offset and drift from sync history"""
        if len(node.sync_history) < self.min_exchanges:
            return

        # Use recent exchanges (sliding window)
        recent = node.sync_history[-self.filter_window:]

        # Calculate offset estimates (NTP-style)
        offsets = [ex.offset_estimate for ex in recent]
        delays = [ex.delay_estimate for ex in recent]

        # Weight by SNR (higher SNR = more trust)
        weights = [10**(ex.snr_db/10) for ex in recent]
        weights = np.array(weights) / np.sum(weights)

        # Weighted average for offset
        node.est_clock_offset_ns = np.average(offsets, weights=weights)

        # Estimate drift from offset changes over time
        if len(node.offset_history) >= 2:
            # Linear regression for drift
            times = np.arange(len(node.offset_history[-10:])) * self.sync_interval_ms
            offsets = node.offset_history[-10:]

            if len(times) >= 2:
                # Fit line to estimate drift rate
                A = np.vstack([times, np.ones(len(times))]).T
                drift_rate, offset = np.linalg.lstsq(A, offsets, rcond=None)[0][:2]
                node.est_clock_drift_ppb = drift_rate  # ns/ms = ppb

        # Store in history
        node.offset_history.append(node.est_clock_offset_ns)
        node.drift_history.append(node.est_clock_drift_ppb)

    def run_sync_protocol(self):
        """Run iterative synchronization protocol"""
        print("="*60)
        print("ITERATIVE TIME SYNCHRONIZATION PROTOCOL")
        print("="*60)

        sync_results = []

        for round_num in range(self.sync_rounds):
            current_time_ms = round_num * self.sync_interval_ms

            print(f"\nSync Round {round_num + 1}/{self.sync_rounds} (t={current_time_ms}ms)")

            # Update all clocks with drift
            for node in self.nodes.values():
                node.update_clock(self.sync_interval_ms * 1e-3)

            # Perform sync exchanges (unknown nodes sync to anchors)
            round_offsets = []

            for node_id in range(self.n_anchors, self.n_nodes):
                node = self.nodes[node_id]

                # Sync with each anchor
                for anchor_id in range(self.n_anchors):
                    anchor = self.nodes[anchor_id]

                    # Perform two-way exchange
                    exchange = self.perform_sync_exchange(node, anchor, current_time_ms)
                    node.sync_history.append(exchange)

                    # Update estimates
                    self.estimate_clock_parameters(node)

                # Track convergence
                if node.est_clock_offset_ns != 0:
                    error = abs(node.clock_offset_ns - node.est_clock_offset_ns)
                    round_offsets.append(error)

            # Report round statistics
            if round_offsets:
                mean_error = np.mean(round_offsets)
                max_error = np.max(round_offsets)
                print(f"  Clock sync error: mean={mean_error:.1f}ns, max={max_error:.1f}ns")

                sync_results.append({
                    'round': round_num + 1,
                    'mean_error_ns': mean_error,
                    'max_error_ns': max_error,
                    'converged': mean_error < 10  # Converged if < 10ns error
                })

        return sync_results

    def perform_localization(self):
        """Localize using synchronized clocks"""
        print("\nLOCALIZATION AFTER TIME SYNC:")

        position_errors = []

        for node_id in range(self.n_anchors, self.n_nodes):
            node = self.nodes[node_id]

            # Collect corrected distance measurements
            anchor_positions = []
            corrected_distances = []

            for anchor_id in range(self.n_anchors):
                anchor = self.nodes[anchor_id]

                # Measure distance
                true_dist = np.linalg.norm(node.position - anchor.position)

                # Add measurement noise
                snr_db = 20 - 10 * np.log10(1 + true_dist/50)
                noise_m = 0.3 / np.sqrt(10**(snr_db/10))
                measured_dist = true_dist + np.random.normal(0, noise_m)

                # Correct for estimated time offset
                time_error_m = (node.clock_offset_ns - node.est_clock_offset_ns) * 1e-9 * self.speed_of_light
                corrected_dist = measured_dist - time_error_m

                anchor_positions.append(anchor.position)
                corrected_distances.append(corrected_dist)

            # Trilateration
            if len(anchor_positions) >= 3:
                # Linear least squares
                n = len(anchor_positions)
                A = np.zeros((n-1, 2))
                b = np.zeros(n-1)

                ref_anchor = anchor_positions[0]
                ref_dist = corrected_distances[0]

                for i in range(1, n):
                    A[i-1] = 2 * (anchor_positions[i] - ref_anchor)
                    b[i-1] = (ref_dist**2 - corrected_distances[i]**2 +
                             np.sum(anchor_positions[i]**2) - np.sum(ref_anchor**2))

                try:
                    est_position = np.linalg.lstsq(A, b, rcond=None)[0]
                    node.est_position = est_position

                    error = np.linalg.norm(node.position - est_position)
                    position_errors.append(error)

                    print(f"  Node {node_id}: error={error:.2f}m, "
                          f"time_error={abs(node.clock_offset_ns - node.est_clock_offset_ns):.1f}ns")
                except:
                    pass

        if position_errors:
            print(f"\nFinal RMSE: {np.sqrt(np.mean(np.array(position_errors)**2)):.2f}m")

        return position_errors

    def plot_convergence(self, sync_results):
        """Plot synchronization convergence over rounds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        rounds = [r['round'] for r in sync_results]
        mean_errors = [r['mean_error_ns'] for r in sync_results]
        max_errors = [r['max_error_ns'] for r in sync_results]

        # Clock sync convergence
        ax1.plot(rounds, mean_errors, 'b-', label='Mean error', linewidth=2)
        ax1.plot(rounds, max_errors, 'r--', label='Max error', linewidth=2)
        ax1.axhline(y=10, color='g', linestyle=':', label='Target (<10ns)')
        ax1.set_xlabel('Sync Round')
        ax1.set_ylabel('Clock Offset Error (ns)')
        ax1.set_title('Iterative Time Sync Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Node offset history
        ax2.set_xlabel('Sync Round')
        ax2.set_ylabel('Estimated Clock Offset (ns)')
        ax2.set_title('Individual Node Clock Estimates')

        for node_id in range(self.n_anchors, min(self.n_nodes, self.n_anchors + 3)):
            node = self.nodes[node_id]
            if node.offset_history:
                ax2.plot(range(1, len(node.offset_history)+1),
                        node.offset_history,
                        label=f'Node {node_id}', alpha=0.7)

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('iterative_sync_convergence.png', dpi=150)
        print(f"\nConvergence plot saved to iterative_sync_convergence.png")

        plt.show()


def main():
    """Demonstrate iterative time synchronization"""

    # Create system
    system = IterativeFTLSync(area_size=100, n_nodes=10, n_anchors=4)

    print("\nInitial clock errors:")
    for i in range(system.n_anchors, system.n_nodes):
        node = system.nodes[i]
        print(f"  Node {i}: offset={node.clock_offset_ns:.1f}ns, "
              f"drift={node.clock_drift_ppb:.1f}ppb, "
              f"freq={node.freq_offset_hz:.1f}Hz")

    # Run iterative synchronization
    sync_results = system.run_sync_protocol()

    # Check convergence
    final_round = sync_results[-1]
    if final_round['converged']:
        print(f"\n✓ CONVERGED after {len(sync_results)} rounds")
        print(f"  Final sync error: {final_round['mean_error_ns']:.1f}ns")
    else:
        print(f"\n✗ Not fully converged after {len(sync_results)} rounds")
        print(f"  Current sync error: {final_round['mean_error_ns']:.1f}ns")

    # Perform localization with synchronized clocks
    position_errors = system.perform_localization()

    # Plot results
    system.plot_convergence(sync_results)

    # Compare with single-pass
    print("\n" + "="*60)
    print("COMPARISON WITH SINGLE-PASS:")

    # Reset estimates and do single-pass
    for node in system.nodes.values():
        node.est_clock_offset_ns = 0
        node.est_clock_drift_ppb = 0
        node.sync_history = []
        node.offset_history = []

    # Single sync exchange
    sync_results_single = system.run_sync_protocol()[:1]  # Just one round
    position_errors_single = system.perform_localization()

    print(f"\nSingle-pass RMSE: {np.sqrt(np.mean(np.array(position_errors_single)**2)):.2f}m")
    print(f"Iterative RMSE: {np.sqrt(np.mean(np.array(position_errors)**2)):.2f}m")
    print(f"Improvement: {(1 - np.sqrt(np.mean(np.array(position_errors)**2)) / np.sqrt(np.mean(np.array(position_errors_single)**2))) * 100:.1f}%")


if __name__ == "__main__":
    main()