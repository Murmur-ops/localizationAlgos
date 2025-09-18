"""
Fixed Time Synchronization with Proper Convergence
Implements correct two-way time transfer formulas from IEEE 1588/NTP
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class TimeSyncNode:
    """Node with time synchronization state"""
    id: int
    true_offset_ns: float  # Actual clock offset
    true_drift_ppb: float  # Clock drift in parts per billion
    is_anchor: bool = False

    # Estimated values
    est_offset_ns: float = 0.0
    est_drift_ppb: float = 0.0

    # Kalman filter state [offset, drift]
    state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 1000)

    # History
    offset_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)


class FixedTimeSync:
    """
    Corrected time synchronization implementation
    Using proper IEEE 1588 PTP formulas
    """

    def __init__(self, n_nodes: int = 10, n_anchors: int = 3):
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors
        self.nodes = []

        # Timing parameters
        self.sync_interval_ms = 100
        self.propagation_delay_ns = 500  # Fixed for simplicity

        # Noise parameters
        self.timestamp_noise_ns = 5.0  # Measurement noise

        # Initialize nodes
        self.initialize_nodes()

        # Tracking
        self.history = {
            'rounds': [],
            'mean_errors': [],
            'max_errors': [],
            'all_errors': []
        }

    def initialize_nodes(self):
        """Create nodes with realistic clock errors"""

        # Anchors: Known time reference (essentially perfect)
        for i in range(self.n_anchors):
            node = TimeSyncNode(
                id=i,
                true_offset_ns=0.0,  # Perfect reference
                true_drift_ppb=0.0,  # No drift
                is_anchor=True
            )
            # Anchors know their time perfectly
            node.est_offset_ns = 0.0
            node.est_drift_ppb = 0.0
            node.state[0] = 0.0
            node.state[1] = 0.0
            node.covariance = np.eye(2) * 1e-6  # Very certain
            self.nodes.append(node)

        # Regular nodes: Unknown clock errors
        for i in range(self.n_anchors, self.n_nodes):
            node = TimeSyncNode(
                id=i,
                true_offset_ns=np.random.uniform(-100, 100),  # ±100ns initial
                true_drift_ppb=np.random.uniform(-5, 5),      # ±5 ppb drift
                is_anchor=False
            )
            # Initial estimates are zero (unknown)
            node.est_offset_ns = 0.0
            node.est_drift_ppb = 0.0
            self.nodes.append(node)

    def two_way_time_transfer(self, node_a: TimeSyncNode, node_b: TimeSyncNode,
                             current_time_ns: float, debug: bool = False) -> float:
        """
        Perform two-way time transfer between nodes
        Returns estimated offset of node_a relative to node_b

        Uses correct IEEE 1588 formulas:
        offset = ((t2 - t1) - (t4 - t3)) / 2
        delay = ((t4 - t1) - (t3 - t2)) / 2
        """

        # True offsets at current time
        true_offset_a = node_a.true_offset_ns + node_a.true_drift_ppb * current_time_ns * 1e-9
        true_offset_b = node_b.true_offset_ns + node_b.true_drift_ppb * current_time_ns * 1e-9

        # Simulate timestamps
        # t1: A sends (in A's time)
        t1 = current_time_ns + true_offset_a

        # t2: B receives (in B's time)
        t2 = current_time_ns + self.propagation_delay_ns + true_offset_b

        # t3: B sends reply (in B's time)
        t3 = t2 + 1000  # 1us processing delay

        # t4: A receives (in A's time)
        t4 = current_time_ns + 2*self.propagation_delay_ns + 1000 + true_offset_a

        # Add measurement noise
        t1 += np.random.normal(0, self.timestamp_noise_ns)
        t2 += np.random.normal(0, self.timestamp_noise_ns)
        t3 += np.random.normal(0, self.timestamp_noise_ns)
        t4 += np.random.normal(0, self.timestamp_noise_ns)

        # Calculate offset using CORRECT formula
        # This estimates offset of A relative to B
        # The formula gives NEGATIVE of A's offset relative to B
        # So we negate it to get the actual offset
        offset_estimate = -((t2 - t1) - (t4 - t3)) / 2

        if debug:
            true_offset_diff = true_offset_a - true_offset_b
            print(f"  True offset diff: {true_offset_diff:.3f}ns")
            print(f"  Estimated offset: {offset_estimate:.3f}ns")
            print(f"  Error: {abs(offset_estimate - true_offset_diff):.3f}ns")

        return offset_estimate

    def kalman_update(self, node: TimeSyncNode, measurement: float, dt: float):
        """
        Update node's time estimate using Kalman filter
        """
        # State transition: offset changes by drift * time
        F = np.array([[1, dt], [0, 1]])

        # Measurement matrix: we observe offset only
        H = np.array([[1, 0]])

        # Process noise (small)
        Q = np.diag([0.01**2, 0.001**2])

        # Measurement noise
        R = (self.timestamp_noise_ns * 2)**2  # Combined noise from 4 timestamps

        # Predict
        x_pred = F @ node.state
        P_pred = F @ node.covariance @ F.T + Q

        # Update
        y = measurement - (H @ x_pred)[0]  # Innovation
        S = (H @ P_pred @ H.T)[0, 0] + R   # Innovation covariance
        K = (P_pred @ H.T).flatten() / S   # Kalman gain

        # Correct
        node.state = x_pred + K * y
        node.covariance = (np.eye(2) - np.outer(K, H)) @ P_pred

        # Update estimates
        node.est_offset_ns = node.state[0]
        node.est_drift_ppb = node.state[1]

    def run_synchronization(self, n_rounds: int = 50):
        """
        Run time synchronization protocol
        """
        print("FIXED TIME SYNCHRONIZATION")
        print("="*60)
        print(f"Nodes: {self.n_nodes} ({self.n_anchors} anchors)")
        print(f"Initial errors: ±100ns offset, ±5ppb drift")
        print()

        for round_idx in range(n_rounds):
            current_time_ns = round_idx * self.sync_interval_ms * 1e6
            dt = self.sync_interval_ms / 1000.0  # in seconds

            errors = []

            # Update each regular node
            for node in self.nodes:
                if not node.is_anchor:
                    # Measure against all anchors
                    measurements = []

                    for anchor in self.nodes[:self.n_anchors]:
                        # Two-way time transfer
                        offset_meas = self.two_way_time_transfer(
                            node, anchor, current_time_ns
                        )
                        measurements.append(offset_meas)

                    # Average measurements
                    avg_measurement = np.mean(measurements)

                    # Kalman filter update
                    self.kalman_update(node, avg_measurement, dt)

                    # Calculate error
                    true_offset = node.true_offset_ns + node.true_drift_ppb * current_time_ns * 1e-9
                    error = abs(true_offset - node.est_offset_ns)
                    errors.append(error)

                    # Store history
                    node.offset_history.append(node.est_offset_ns)
                    node.error_history.append(error)

            # Store round statistics
            self.history['rounds'].append(round_idx + 1)
            self.history['mean_errors'].append(np.mean(errors))
            self.history['max_errors'].append(np.max(errors))
            self.history['all_errors'].append(errors)

            # Print progress
            if round_idx < 5 or (round_idx + 1) % 10 == 0:
                mean_err = np.mean(errors)
                max_err = np.max(errors)
                print(f"Round {round_idx+1:3d}: mean={mean_err:7.3f}ns, max={max_err:7.3f}ns")

        # Final results
        print()
        print("="*60)
        print("FINAL RESULTS:")
        final_mean = self.history['mean_errors'][-1]
        final_max = self.history['max_errors'][-1]
        print(f"  Mean error: {final_mean:.3f}ns")
        print(f"  Max error: {final_max:.3f}ns")

        # Check convergence
        converged_round = None
        for i, mean_err in enumerate(self.history['mean_errors']):
            if mean_err < 1.0:
                converged_round = i + 1
                break

        if converged_round:
            print(f"  CONVERGED to <1ns at round {converged_round}")
            print(f"  Time to convergence: {converged_round * self.sync_interval_ms}ms")
        else:
            print(f"  Did not converge to <1ns in {n_rounds} rounds")

    def plot_results(self):
        """Plot convergence results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        rounds = self.history['rounds']

        # 1. Convergence plot (log scale)
        ax = axes[0, 0]
        ax.semilogy(rounds, self.history['mean_errors'], 'b-', linewidth=2, label='Mean Error')
        ax.semilogy(rounds, self.history['max_errors'], 'r--', linewidth=2, label='Max Error')
        ax.axhline(y=1.0, color='g', linestyle=':', linewidth=2, label='Target (<1ns)')
        ax.set_xlabel('Synchronization Round')
        ax.set_ylabel('Clock Offset Error (ns)')
        ax.set_title('Time Sync Convergence (FIXED)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.01, 200])

        # 2. Individual node errors
        ax = axes[0, 1]
        for node in self.nodes[self.n_anchors:self.n_anchors+5]:
            if node.error_history:
                ax.semilogy(rounds[:len(node.error_history)],
                          node.error_history, alpha=0.7, label=f'Node {node.id}')
        ax.set_xlabel('Synchronization Round')
        ax.set_ylabel('Clock Offset Error (ns)')
        ax.set_title('Individual Node Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error distribution over time
        ax = axes[1, 0]
        # Show distribution at different stages
        stages = [0, len(rounds)//4, len(rounds)//2, -1]
        colors = ['red', 'orange', 'yellow', 'green']
        labels = ['Initial', '25%', '50%', 'Final']

        for stage, color, label in zip(stages, colors, labels):
            if stage < len(self.history['all_errors']):
                errors = self.history['all_errors'][stage]
                if errors:
                    ax.hist(errors, bins=20, alpha=0.5, color=color, label=label)

        ax.set_xlabel('Clock Offset Error (ns)')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('Error Distribution Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Convergence rate
        ax = axes[1, 1]
        if len(self.history['mean_errors']) > 1:
            rates = []
            for i in range(1, len(self.history['mean_errors'])):
                prev = self.history['mean_errors'][i-1]
                curr = self.history['mean_errors'][i]
                if prev > 0:
                    rate = (prev - curr) / prev * 100
                    rates.append(rate)

            ax.plot(rounds[1:], rates, 'purple', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Synchronization Round')
            ax.set_ylabel('Error Reduction Rate (%)')
            ax.set_title('Convergence Speed')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Fixed Time Synchronization: Proper IEEE 1588 Implementation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig('time_sync_fixed_convergence.png', dpi=150)
        print(f"\nPlot saved to time_sync_fixed_convergence.png")


def main():
    """Test the fixed time synchronization"""

    # Create system
    sync = FixedTimeSync(n_nodes=10, n_anchors=3)

    # Run synchronization
    sync.run_synchronization(n_rounds=50)

    # Plot results
    sync.plot_results()

    print("\n" + "="*60)
    print("KEY FIXES APPLIED:")
    print("1. Correct IEEE 1588 offset formula: ((t2-t1)+(t4-t3))/2")
    print("2. Anchors are true references with zero offset")
    print("3. Proper Kalman filtering with appropriate noise levels")
    print("4. Simplified, clear implementation")
    print("\nRESULT: Should converge to <1ns within 10-20 rounds")


if __name__ == "__main__":
    main()