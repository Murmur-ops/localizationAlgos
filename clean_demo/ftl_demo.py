#!/usr/bin/env python3
"""
True FTL (Frequency-Time-Localization) Demo
Demonstrates joint estimation of frequency, time, and location using spread spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.signal as signal


@dataclass
class FTLNode:
    """Node with frequency, time, and location state"""
    id: int
    true_position: np.ndarray
    is_anchor: bool

    # Frequency state
    carrier_freq_offset_hz: float = 0.0  # Frequency offset from nominal
    phase_offset_rad: float = 0.0

    # Time state
    clock_offset_ns: float = 0.0  # Time offset from global reference
    clock_drift_ppb: float = 0.0  # Clock drift in parts per billion

    # Estimated states
    est_position: Optional[np.ndarray] = None
    est_freq_offset_hz: Optional[float] = None
    est_clock_offset_ns: Optional[float] = None


@dataclass
class SpreadSpectrumSignal:
    """Spread spectrum ranging signal"""
    gold_code: np.ndarray
    chip_rate_hz: float
    carrier_freq_hz: float
    bandwidth_hz: float
    timestamp_tx_ns: float


class GoldCodeGenerator:
    """Generate Gold codes for spread spectrum ranging"""

    def __init__(self, length: int = 127):
        """Initialize with shorter Gold code for faster demo"""
        self.length = length
        self.codes = self._generate_gold_codes()

    def _generate_gold_codes(self) -> Dict[int, np.ndarray]:
        """Generate set of Gold codes using preferred pairs"""
        codes = {}

        # Simple PN sequence generation (simplified for demo)
        # In practice, use proper preferred pairs
        np.random.seed(42)
        for i in range(32):  # Generate 32 codes
            # Create pseudo-random sequence
            code = np.random.choice([-1, 1], size=self.length)
            codes[i] = code

        return codes

    def get_code(self, node_id: int) -> np.ndarray:
        """Get unique Gold code for node"""
        return self.codes[node_id % len(self.codes)]


class FTLTransceiver:
    """Transceiver for FTL signals with frequency and time synchronization"""

    def __init__(self, node: FTLNode, config: dict):
        self.node = node
        self.config = config
        self.gold_gen = GoldCodeGenerator()

        # Get node's unique Gold code
        self.gold_code = self.gold_gen.get_code(node.id)

        # Signal parameters
        self.chip_rate = float(config['signal']['chip_rate_hz'])
        self.carrier_freq = float(config['signal']['carrier_freq_hz'])
        self.bandwidth = float(config['signal']['bandwidth_hz'])
        self.sample_rate = float(config['signal']['sample_rate_hz'])

    def transmit_ranging_signal(self, current_time_ns: float) -> SpreadSpectrumSignal:
        """Transmit spread spectrum ranging signal"""

        # Apply node's time offset
        timestamp_tx = current_time_ns + self.node.clock_offset_ns

        return SpreadSpectrumSignal(
            gold_code=self.gold_code,
            chip_rate_hz=self.chip_rate,
            carrier_freq_hz=self.carrier_freq + self.node.carrier_freq_offset_hz,
            bandwidth_hz=self.bandwidth,
            timestamp_tx_ns=timestamp_tx
        )

    def correlate_signal(self, received_signal: SpreadSpectrumSignal,
                        local_time_ns: float) -> Tuple[float, float, float]:
        """
        Correlate received signal to estimate TOA, frequency offset, and SNR

        Returns:
            toa_ns: Time of arrival in nanoseconds
            freq_offset_hz: Estimated frequency offset
            snr_db: Signal-to-noise ratio
        """

        # Simplified correlation (in practice use FFT-based)
        correlation = np.correlate(received_signal.gold_code, self.gold_code, mode='same')
        peak_idx = np.argmax(np.abs(correlation))

        # Sub-sample interpolation for better accuracy
        if peak_idx > 0 and peak_idx < len(correlation) - 1:
            # Parabolic interpolation around peak
            y1 = np.abs(correlation[peak_idx - 1])
            y2 = np.abs(correlation[peak_idx])
            y3 = np.abs(correlation[peak_idx + 1])

            delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3 + 1e-10)
            peak_idx_refined = peak_idx + delta
        else:
            peak_idx_refined = peak_idx

        # Convert correlation peak to time
        chip_duration_ns = 1e9 / self.chip_rate
        toa_offset_ns = (peak_idx_refined - len(correlation)//2) * chip_duration_ns

        # Add receive timestamp and clock offset
        toa_ns = local_time_ns + self.node.clock_offset_ns + toa_offset_ns

        # Estimate frequency offset from phase rotation
        freq_offset_hz = received_signal.carrier_freq_hz - self.carrier_freq

        # Estimate SNR
        signal_power = np.abs(correlation[peak_idx])**2
        noise_power = np.mean(np.abs(correlation)**2)
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear)

        return toa_ns, freq_offset_hz, snr_db


class FTLSynchronizer:
    """Joint frequency and time synchronization"""

    def __init__(self, node: FTLNode):
        self.node = node

        # Kalman filter states for time sync
        self.time_offset_est_ns = 0.0
        self.time_drift_est_ppb = 0.0
        self.time_variance = 100.0  # ns^2

        # PLL states for frequency sync
        self.freq_offset_est_hz = 0.0
        self.phase_error_rad = 0.0
        self.loop_bandwidth_hz = 10.0

    def update_frequency_pll(self, measured_freq_offset_hz: float, dt_s: float):
        """Update frequency estimate using PLL"""

        # Phase detector
        phase_error = measured_freq_offset_hz - self.freq_offset_est_hz

        # Loop filter (PI controller)
        kp = 2 * self.loop_bandwidth_hz
        ki = self.loop_bandwidth_hz**2

        self.freq_offset_est_hz += kp * phase_error * dt_s
        self.phase_error_rad += ki * phase_error * dt_s

        # Update node's frequency offset estimate
        self.node.est_freq_offset_hz = self.freq_offset_est_hz

    def update_time_kalman(self, measured_toa_ns: float, expected_toa_ns: float, dt_s: float):
        """Update time offset estimate using Kalman filter"""

        # Innovation
        innovation = measured_toa_ns - expected_toa_ns - self.time_offset_est_ns

        # Kalman gain
        measurement_variance = 10.0  # ns^2
        kalman_gain = self.time_variance / (self.time_variance + measurement_variance)

        # Update estimates
        self.time_offset_est_ns += kalman_gain * innovation
        self.time_drift_est_ppb += kalman_gain * innovation / dt_s * 0.001

        # Update variance
        self.time_variance *= (1 - kalman_gain)
        self.time_variance += dt_s**2 * 1.0  # Process noise

        # Update node's time offset estimate
        self.node.est_clock_offset_ns = self.time_offset_est_ns


class FTLLocalizer:
    """Joint Frequency-Time-Localization solver"""

    def __init__(self, config: dict):
        self.config = config
        self.max_iterations = config['solver']['max_iterations']
        self.convergence_threshold = config['solver']['convergence_threshold']

    def solve_ftl(self, nodes: Dict[int, FTLNode],
                  measurements: List[Tuple[int, int, float, float, float]]) -> Dict:
        """
        Solve joint FTL problem

        Args:
            nodes: Dictionary of FTL nodes
            measurements: List of (node_i, node_j, toa_ns, freq_offset_hz, snr_db)

        Returns:
            Solution dictionary with positions, frequencies, and times
        """

        # Separate anchors and unknowns
        anchors = {nid: n for nid, n in nodes.items() if n.is_anchor}
        unknowns = {nid: n for nid, n in nodes.items() if not n.is_anchor}

        # Initialize state vector [x1,y1,f1,t1, x2,y2,f2,t2, ...]
        # Position (m), frequency offset (Hz), time offset (ns)
        state_dim = 4  # x, y, freq, time per node
        state = np.zeros(len(unknowns) * state_dim)

        # Initial guess: center position, zero frequency/time offsets
        area_size = self.config['area']['size_m']
        for i, (nid, node) in enumerate(unknowns.items()):
            state[i*state_dim] = area_size / 2      # x
            state[i*state_dim + 1] = area_size / 2  # y
            state[i*state_dim + 2] = 0              # freq offset
            state[i*state_dim + 3] = 0              # time offset

        # Iterative optimization
        print(f"Starting optimization with {len(measurements)} measurements...")
        for iteration in range(self.max_iterations):
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}...")
            old_state = state.copy()

            # Build Jacobian and residual for joint problem
            J = []
            residuals = []
            weights = []

            for node_i, node_j, toa_ns, freq_offset_hz, snr_db in measurements:
                # Get node positions and states
                if node_i in unknowns:
                    i_idx = list(unknowns.keys()).index(node_i)
                    pos_i = state[i_idx*state_dim:(i_idx*state_dim + 2)]
                    freq_i = state[i_idx*state_dim + 2]
                    time_i = state[i_idx*state_dim + 3]
                else:
                    pos_i = anchors[node_i].true_position
                    freq_i = 0  # Anchors have perfect frequency
                    time_i = 0  # Anchors have perfect time
                    i_idx = None

                if node_j in unknowns:
                    j_idx = list(unknowns.keys()).index(node_j)
                    pos_j = state[j_idx*state_dim:(j_idx*state_dim + 2)]
                    freq_j = state[j_idx*state_dim + 2]
                    time_j = state[j_idx*state_dim + 3]
                else:
                    pos_j = anchors[node_j].true_position
                    freq_j = 0
                    time_j = 0
                    j_idx = None

                # Compute expected measurements
                distance = np.linalg.norm(pos_i - pos_j)
                expected_toa_ns = distance / 3e8 * 1e9 + (time_j - time_i)
                expected_freq_hz = freq_j - freq_i

                # Residuals
                r_toa = toa_ns - expected_toa_ns
                r_freq = freq_offset_hz - expected_freq_hz

                # Combined residual weighted by measurement type
                residuals.append(r_toa / 10.0)  # Scale time residual to meters
                residuals.append(r_freq / 1000.0)  # Scale frequency residual

                # Weight by SNR
                weight = min(1.0, snr_db / 20.0) if snr_db > 0 else 0.1
                weights.extend([weight, weight])

                # Jacobian rows
                j_row_toa = np.zeros(len(state))
                j_row_freq = np.zeros(len(state))

                if distance > 1e-6:
                    # Position derivatives for TOA
                    if i_idx is not None:
                        grad_pos_i = -(pos_i - pos_j) / distance / 3e8 * 1e9
                        j_row_toa[i_idx*state_dim:(i_idx*state_dim + 2)] = grad_pos_i / 10.0
                        j_row_toa[i_idx*state_dim + 3] = -1.0 / 10.0  # Time derivative
                        j_row_freq[i_idx*state_dim + 2] = -1.0 / 1000.0  # Frequency derivative

                    if j_idx is not None:
                        grad_pos_j = (pos_i - pos_j) / distance / 3e8 * 1e9
                        j_row_toa[j_idx*state_dim:(j_idx*state_dim + 2)] = grad_pos_j / 10.0
                        j_row_toa[j_idx*state_dim + 3] = 1.0 / 10.0
                        j_row_freq[j_idx*state_dim + 2] = 1.0 / 1000.0

                J.append(j_row_toa)
                J.append(j_row_freq)

            if not J:
                break

            # Solve weighted least squares
            J = np.array(J)
            r = np.array(residuals)
            W = np.diag(weights)

            # Levenberg-Marquardt with regularization
            lambda_lm = 0.1
            H = J.T @ W @ J + lambda_lm * np.eye(len(state))
            g = -J.T @ W @ r

            try:
                delta = np.linalg.solve(H, g)
                state += delta * 0.5  # Damping factor
            except np.linalg.LinAlgError:
                break

            # Check convergence
            if np.linalg.norm(delta) < self.convergence_threshold:
                break

        # Extract results
        results = {}
        for i, (nid, node) in enumerate(unknowns.items()):
            node.est_position = state[i*state_dim:(i*state_dim + 2)]
            node.est_freq_offset_hz = state[i*state_dim + 2]
            node.est_clock_offset_ns = state[i*state_dim + 3]

            results[nid] = {
                'position': node.est_position,
                'freq_offset_hz': node.est_freq_offset_hz,
                'time_offset_ns': node.est_clock_offset_ns,
                'position_error_m': np.linalg.norm(node.est_position - node.true_position)
            }

        results['iterations'] = iteration + 1
        results['final_residual'] = np.linalg.norm(residuals)

        return results


def simulate_ftl_network(config: dict) -> Tuple[Dict[int, FTLNode], List]:
    """Simulate FTL network and generate measurements"""

    np.random.seed(config['seed'])

    # Create nodes
    nodes = {}
    area_size = config['area']['size_m']
    n_anchors = config['nodes']['anchors']
    n_total = config['nodes']['total']

    # Place anchors at optimal positions
    if n_anchors >= 4:
        anchor_positions = [
            [0, 0], [area_size, 0],
            [area_size, area_size], [0, area_size]
        ]
        for i in range(min(4, n_anchors)):
            nodes[i] = FTLNode(
                id=i,
                true_position=np.array(anchor_positions[i]),
                is_anchor=True
            )

    # Place unknown nodes
    for i in range(n_anchors, n_total):
        nodes[i] = FTLNode(
            id=i,
            true_position=np.random.uniform(0, area_size, 2),
            is_anchor=False,
            # Add realistic frequency and time offsets
            carrier_freq_offset_hz=np.random.normal(0, float(config['hardware']['freq_offset_std_hz'])),
            clock_offset_ns=np.random.normal(0, float(config['hardware']['time_offset_std_ns'])),
            clock_drift_ppb=np.random.normal(0, float(config['hardware']['drift_std_ppb']))
        )

    # Create transceivers
    transceivers = {nid: FTLTransceiver(node, config) for nid, node in nodes.items()}

    # Generate measurements
    measurements = []
    current_time_ns = 0

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            # Check if in range
            distance = np.linalg.norm(nodes[i].true_position - nodes[j].true_position)
            if distance > config['channel']['max_range_m']:
                continue

            # Node i transmits
            tx_signal = transceivers[i].transmit_ranging_signal(current_time_ns)

            # Propagation delay
            prop_delay_ns = distance / 3e8 * 1e9

            # Node j receives and correlates
            rx_time_ns = current_time_ns + prop_delay_ns
            toa_ns, freq_offset_hz, snr_db = transceivers[j].correlate_signal(tx_signal, rx_time_ns)

            # Add measurement noise
            bandwidth_hz = float(config['signal']['bandwidth_hz'])
            resolution_ns = 1e9 / (2 * bandwidth_hz)
            toa_noise_ns = np.random.normal(0, resolution_ns / np.sqrt(10**(snr_db/10)))
            toa_ns += toa_noise_ns

            measurements.append((i, j, toa_ns, freq_offset_hz, snr_db))

    return nodes, measurements


def visualize_ftl_results(nodes: Dict[int, FTLNode], results: Dict, config: dict):
    """Visualize FTL results showing position, frequency, and time estimates"""

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Position estimation
    ax1 = fig.add_subplot(gs[0, :2])

    # Plot anchors
    for nid, node in nodes.items():
        if node.is_anchor:
            ax1.scatter(node.true_position[0], node.true_position[1],
                       s=200, c='blue', marker='^', edgecolors='black',
                       linewidth=2, label='Anchor' if nid == 0 else '')

    # Plot unknowns - true and estimated
    for nid, node in nodes.items():
        if not node.is_anchor:
            # True position
            ax1.scatter(node.true_position[0], node.true_position[1],
                       s=100, c='green', marker='o', alpha=0.5,
                       label='True' if nid == len([n for n in nodes.values() if n.is_anchor]) else '')

            # Estimated position
            if node.est_position is not None:
                ax1.scatter(node.est_position[0], node.est_position[1],
                           s=100, c='red', marker='x', linewidth=2,
                           label='Estimated' if nid == len([n for n in nodes.values() if n.is_anchor]) else '')

                # Error line
                ax1.plot([node.true_position[0], node.est_position[0]],
                        [node.true_position[1], node.est_position[1]],
                        'k--', alpha=0.3)

    ax1.set_xlim(-2, config['area']['size_m']+2)
    ax1.set_ylim(-2, config['area']['size_m']+2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Position Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Frequency offset estimation
    ax2 = fig.add_subplot(gs[0, 2])

    true_freqs = []
    est_freqs = []
    node_ids = []

    for nid, node in nodes.items():
        if not node.is_anchor:
            true_freqs.append(node.carrier_freq_offset_hz)
            est_freqs.append(node.est_freq_offset_hz if node.est_freq_offset_hz else 0)
            node_ids.append(nid)

    x = np.arange(len(node_ids))
    width = 0.35

    ax2.bar(x - width/2, true_freqs, width, label='True', alpha=0.7, color='green')
    ax2.bar(x + width/2, est_freqs, width, label='Estimated', alpha=0.7, color='red')
    ax2.set_xlabel('Node ID')
    ax2.set_ylabel('Frequency Offset (Hz)')
    ax2.set_title('Frequency Synchronization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(node_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Time offset estimation
    ax3 = fig.add_subplot(gs[1, 0])

    true_times = []
    est_times = []

    for nid, node in nodes.items():
        if not node.is_anchor:
            true_times.append(node.clock_offset_ns)
            est_times.append(node.est_clock_offset_ns if node.est_clock_offset_ns else 0)

    ax3.bar(x - width/2, true_times, width, label='True', alpha=0.7, color='green')
    ax3.bar(x + width/2, est_times, width, label='Estimated', alpha=0.7, color='red')
    ax3.set_xlabel('Node ID')
    ax3.set_ylabel('Time Offset (ns)')
    ax3.set_title('Time Synchronization')
    ax3.set_xticks(x)
    ax3.set_xticklabels(node_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error summary
    ax4 = fig.add_subplot(gs[1, 1:])

    position_errors = []
    freq_errors = []
    time_errors = []

    for nid, node in nodes.items():
        if not node.is_anchor and nid in results:
            position_errors.append(results[nid]['position_error_m'])
            freq_errors.append(abs(node.carrier_freq_offset_hz - node.est_freq_offset_hz))
            time_errors.append(abs(node.clock_offset_ns - node.est_clock_offset_ns))

    metrics = ['Position\n(m)', 'Frequency\n(Hz)', 'Time\n(ns)']
    rmse_values = [
        np.sqrt(np.mean(np.array(position_errors)**2)),
        np.sqrt(np.mean(np.array(freq_errors)**2)),
        np.sqrt(np.mean(np.array(time_errors)**2))
    ]

    bars = ax4.bar(metrics, rmse_values, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_ylabel('RMSE')
    ax4.set_title('Joint FTL Performance')
    ax4.grid(True, alpha=0.3)

    # Add values on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')

    plt.suptitle('FTL: Joint Frequency-Time-Localization Results', fontsize=14, fontweight='bold')

    return fig


def run_ftl_demo(config_path: str):
    """Run the complete FTL demonstration"""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("FTL: FREQUENCY-TIME-LOCALIZATION DEMO")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Area: {config['area']['size_m']}×{config['area']['size_m']}m")
    print(f"Nodes: {config['nodes']['total']} ({config['nodes']['anchors']} anchors)")
    print(f"Signal: {float(config['signal']['bandwidth_hz'])/1e6:.0f}MHz bandwidth, ")
    print(f"        {float(config['signal']['chip_rate_hz'])/1e6:.1f}MHz chip rate")
    print(f"Hardware: ±{float(config['hardware']['freq_offset_std_hz'])}Hz frequency offset")
    print(f"          ±{float(config['hardware']['time_offset_std_ns'])}ns time offset")
    print()

    # Simulate network
    print("Simulating FTL network...")
    nodes, measurements = simulate_ftl_network(config)
    print(f"Generated {len(measurements)} ranging measurements")

    # Run joint FTL solver
    print("\nRunning joint FTL optimization...")
    localizer = FTLLocalizer(config)
    results = localizer.solve_ftl(nodes, measurements)

    print(f"Converged in {results['iterations']} iterations")
    print(f"Final residual: {results['final_residual']:.3f}")

    # Calculate performance metrics
    position_errors = []
    freq_errors = []
    time_errors = []

    print("\nPer-node results:")
    for nid, node in nodes.items():
        if not node.is_anchor and nid in results:
            pos_err = results[nid]['position_error_m']
            freq_err = abs(node.carrier_freq_offset_hz - node.est_freq_offset_hz)
            time_err = abs(node.clock_offset_ns - node.est_clock_offset_ns)

            position_errors.append(pos_err)
            freq_errors.append(freq_err)
            time_errors.append(time_err)

            print(f"Node {nid}:")
            print(f"  Position error: {pos_err:.3f}m")
            print(f"  Frequency error: {freq_err:.1f}Hz")
            print(f"  Time error: {time_err:.1f}ns")

    # Overall performance
    print("\n" + "="*60)
    print("OVERALL FTL PERFORMANCE")
    print("="*60)
    print(f"Position RMSE: {np.sqrt(np.mean(np.array(position_errors)**2)):.3f}m")
    print(f"Frequency RMSE: {np.sqrt(np.mean(np.array(freq_errors)**2)):.1f}Hz")
    print(f"Time RMSE: {np.sqrt(np.mean(np.array(time_errors)**2)):.1f}ns")

    # Visualize
    fig = visualize_ftl_results(nodes, results, config)
    plt.savefig('ftl_demo_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to ftl_demo_results.png")
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "configs/ftl.yaml"

    run_ftl_demo(config_file)