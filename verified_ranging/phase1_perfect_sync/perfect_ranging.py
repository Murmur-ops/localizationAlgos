"""
Perfect World Ranging System
Phase 1: Perfect synchronization, no noise, no channel effects
This should give EXACT distances
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from gold_codes import VerifiedGoldCodes


class PerfectRanging:
    """
    Ranging with perfect assumptions:
    - Perfect time synchronization (all clocks aligned)
    - Perfect frequency (no offset)
    - No noise
    - No channel effects
    - Just pure signal delay
    """

    def __init__(self, n_nodes: int = 8, area_size: float = 100.0):
        """Initialize perfect ranging system"""
        self.n_nodes = n_nodes
        self.area_size = area_size
        self.c = 3e8  # Speed of light

        # Gold codes for ranging
        print("\n" + "="*60)
        print("PERFECT RANGING SYSTEM INITIALIZATION")
        print("="*60)
        print(f"Nodes: {n_nodes}")
        print(f"Area: {area_size}×{area_size} m")

        # Generate verified Gold codes
        self.gold_gen = VerifiedGoldCodes(m=7)  # Length 127
        self.sample_rate = 1e9  # 1 GHz sampling
        self.chip_rate = 1.023e6  # GPS-like chip rate
        self.samples_per_chip = int(self.sample_rate / self.chip_rate)

        # Create nodes with positions
        self._create_nodes()

        # Verification results
        self.verification_results = {}

    def _create_nodes(self):
        """Create nodes with known positions"""
        self.nodes = []

        # First 4 nodes are anchors at corners
        anchor_positions = [
            [10, 10],
            [self.area_size - 10, 10],
            [self.area_size - 10, self.area_size - 10],
            [10, self.area_size - 10]
        ]

        for i in range(self.n_nodes):
            if i < 4:
                # Anchor
                pos = np.array(anchor_positions[i])
                is_anchor = True
            else:
                # Random position
                pos = np.random.uniform(20, self.area_size - 20, 2)
                is_anchor = False

            self.nodes.append({
                'id': i,
                'position': pos,
                'is_anchor': is_anchor,
                'gold_code': self.gold_gen.get_code(i),
                'ranging_results': {}
            })

            print(f"Node {i}: pos=[{pos[0]:.1f}, {pos[1]:.1f}]m, "
                  f"{'anchor' if is_anchor else 'mobile'}")

    def generate_transmitted_signal(self, code: np.ndarray) -> np.ndarray:
        """
        Generate transmitted signal from Gold code
        Upsamples to simulation sample rate
        """
        # Upsample Gold code to sample rate
        signal = np.zeros(len(code) * self.samples_per_chip, dtype=float)

        for i, chip in enumerate(code):
            signal[i * self.samples_per_chip:(i + 1) * self.samples_per_chip] = chip

        return signal

    def simulate_propagation(self, tx_signal: np.ndarray, distance: float) -> np.ndarray:
        """
        Simulate signal propagation in perfect world
        Only effect: pure delay based on distance
        """
        # Calculate propagation delay
        propagation_time = distance / self.c
        delay_samples = int(propagation_time * self.sample_rate)

        # Apply delay (pad with zeros)
        if delay_samples > 0:
            rx_signal = np.concatenate([np.zeros(delay_samples), tx_signal])
        else:
            rx_signal = tx_signal.copy()

        return rx_signal

    def correlate_for_ranging(self, rx_signal: np.ndarray,
                            ref_code: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Correlate received signal with reference code to find delay
        Returns: (delay_ns, peak_value, correlation)
        """
        # Generate reference signal
        ref_signal = self.generate_transmitted_signal(ref_code)

        # Ensure same length for correlation
        max_len = max(len(rx_signal), len(ref_signal))
        if len(rx_signal) < max_len:
            rx_signal = np.pad(rx_signal, (0, max_len - len(rx_signal)))
        if len(ref_signal) < max_len:
            ref_signal = np.pad(ref_signal, (0, max_len - len(ref_signal)))

        # Correlate
        correlation = np.correlate(rx_signal, ref_signal, mode='full')

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = correlation[peak_idx]

        # Convert to time delay
        # Peak location gives delay in samples
        delay_samples = peak_idx - (len(ref_signal) - 1)
        delay_ns = delay_samples / self.sample_rate * 1e9

        return delay_ns, peak_value, correlation

    def perform_ranging(self, node_i: Dict, node_j: Dict) -> Dict:
        """
        Perform ranging between two nodes in perfect world
        Should give EXACT distance
        """
        # True distance
        true_distance = np.linalg.norm(node_i['position'] - node_j['position'])

        # Node i transmits
        tx_signal = self.generate_transmitted_signal(node_i['gold_code'])

        # Propagate to node j
        rx_signal = self.simulate_propagation(tx_signal, true_distance)

        # Node j correlates to find delay
        delay_ns, peak, correlation = self.correlate_for_ranging(
            rx_signal, node_i['gold_code']
        )

        # Convert delay to distance
        measured_distance = delay_ns * 1e-9 * self.c

        # Calculate error (should be ~0 in perfect world)
        error = measured_distance - true_distance

        result = {
            'true_distance': true_distance,
            'measured_distance': measured_distance,
            'error': error,
            'delay_ns': delay_ns,
            'peak': peak,
            'correlation_length': len(correlation)
        }

        # Store result
        node_i['ranging_results'][node_j['id']] = measured_distance
        node_j['ranging_results'][node_i['id']] = measured_distance

        return result

    def run_all_ranging(self) -> List[Dict]:
        """Perform ranging between all node pairs"""
        print("\n" + "="*60)
        print("RANGING MEASUREMENTS (PERFECT WORLD)")
        print("="*60)

        results = []
        pair_count = 0

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                result = self.perform_ranging(self.nodes[i], self.nodes[j])
                results.append(result)

                pair_count += 1
                if pair_count <= 5:  # Show first few
                    print(f"Pair {i}-{j}: "
                          f"True={result['true_distance']:.3f}m, "
                          f"Measured={result['measured_distance']:.3f}m, "
                          f"Error={result['error']:.6f}m")

        return results

    def verify_perfect_ranging(self, results: List[Dict]) -> bool:
        """Verify that ranging is perfect (zero error)"""
        print("\n" + "="*60)
        print("VERIFICATION: PERFECT RANGING")
        print("="*60)

        errors = [r['error'] for r in results]
        max_error = max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))
        std_error = np.std(errors)

        print(f"Number of measurements: {len(results)}")
        print(f"Max absolute error: {max_error:.9f} m")
        print(f"Mean absolute error: {mean_error:.9f} m")
        print(f"Standard deviation: {std_error:.9f} m")

        # In perfect world, error should be due to sampling quantization only
        # At 1 GHz sampling, resolution is ~0.3m
        expected_max_error = self.c / self.sample_rate  # One sample uncertainty

        print(f"\nExpected quantization error: {expected_max_error:.3f} m")

        if max_error <= expected_max_error:
            print("✓ PASS: Ranging errors within quantization limit")
            success = True
        else:
            print("✗ FAIL: Ranging errors exceed quantization limit")
            success = False

        self.verification_results['ranging'] = {
            'max_error': max_error,
            'mean_error': mean_error,
            'quantization_limit': expected_max_error,
            'passed': success
        }

        return success

    def visualize_results(self, results: List[Dict]):
        """Visualize ranging results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Network topology
        ax = axes[0, 0]
        for node in self.nodes:
            if node['is_anchor']:
                ax.scatter(node['position'][0], node['position'][1],
                          s=200, c='red', marker='^', edgecolor='black',
                          linewidth=2, label='Anchor' if node['id'] == 0 else '')
            else:
                ax.scatter(node['position'][0], node['position'][1],
                          s=100, c='blue', marker='o',
                          label='Node' if node['id'] == 4 else '')

        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Network Topology')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Ranging errors (should be ~zero)
        ax = axes[0, 1]
        errors = [r['error'] * 1000 for r in results]  # Convert to mm
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Ranging Error (mm)')
        ax.set_ylabel('Count')
        ax.set_title(f'Perfect World Errors (max={max(np.abs(errors)):.3f}mm)')
        ax.grid(True, alpha=0.3)

        # 3. True vs Measured distances
        ax = axes[1, 0]
        true_dists = [r['true_distance'] for r in results]
        meas_dists = [r['measured_distance'] for r in results]
        ax.scatter(true_dists, meas_dists, alpha=0.6, s=20)
        ax.plot([0, max(true_dists)], [0, max(true_dists)], 'r--', label='Perfect')
        ax.set_xlabel('True Distance (m)')
        ax.set_ylabel('Measured Distance (m)')
        ax.set_title('Measurement Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Sample correlation
        ax = axes[1, 1]
        # Show one correlation example
        if results:
            # Redo one measurement to get correlation
            tx_signal = self.generate_transmitted_signal(self.nodes[0]['gold_code'])
            rx_signal = self.simulate_propagation(tx_signal, 50)  # 50m distance
            delay, peak, corr = self.correlate_for_ranging(rx_signal, self.nodes[0]['gold_code'])

            # Show around peak
            peak_idx = np.argmax(np.abs(corr))
            window = 500
            start = max(0, peak_idx - window)
            end = min(len(corr), peak_idx + window)

            ax.plot(corr[start:end])
            ax.axvline(x=peak_idx - start, color='r', linestyle='--', label='Peak')
            ax.set_xlabel('Correlation Lag')
            ax.set_ylabel('Correlation Value')
            ax.set_title(f'Example Correlation (50m range)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Perfect World Ranging - Phase 1 Verification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/maxburnett/Documents/DecentralizedLocale/verified_ranging/phase1_perfect_sync/perfect_ranging_results.png', dpi=150)
        print(f"\nVisualization saved to perfect_ranging_results.png")
        plt.show()

        return fig


def test_perfect_ranging():
    """Test perfect ranging system"""
    print("\n" + "🔬 STARTING PERFECT WORLD RANGING TEST")

    # Create system
    system = PerfectRanging(n_nodes=8, area_size=100)

    # Run ranging
    results = system.run_all_ranging()

    # Verify results
    passed = system.verify_perfect_ranging(results)

    # Visualize
    system.visualize_results(results)

    return system, results, passed


if __name__ == "__main__":
    system, results, passed = test_perfect_ranging()

    if passed:
        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETE: Perfect ranging verified")
        print("Ready to proceed to Phase 2 (add noise)")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ PHASE 1 FAILED: Perfect ranging has errors")
        print("Must fix before proceeding")
        print("="*60)