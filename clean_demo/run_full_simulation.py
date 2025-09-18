"""
Full FTL Simulation for 30-node network
Runs complete time sync, ranging, and localization
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from time_sync_fixed import FixedTimeSync
from gold_codes_working import WorkingGoldCodeGenerator


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

    def __post_init__(self):
        if self.est_position is None:
            self.est_position = np.zeros(3)


class FullFTLSimulation:
    """Complete FTL simulation with all components"""

    def __init__(self, config_file: str = "config_30nodes.yaml"):
        """Initialize from YAML config"""
        print("="*70)
        print("FULL FTL SIMULATION - 30 NODE NETWORK")
        print("="*70)

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_nodes = self.config['network']['total_nodes']
        self.n_anchors = self.config['network']['anchor_nodes']

        print(f"Loading configuration: {self.n_nodes} nodes, {self.n_anchors} anchors")

        self.setup_network()
        self.setup_gold_codes()

        # Results storage
        self.results = {
            'time_sync': {},
            'ranging': {},
            'localization': {}
        }

    def setup_network(self):
        """Initialize network nodes"""
        self.nodes = []

        # Create anchors from config
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
            node.est_position = node.position.copy()  # Anchors know their position
            self.nodes.append(node)
            print(f"  {node.name}: position={node.position}")

        # Create regular nodes
        print(f"\nInitializing {self.n_nodes - self.n_anchors} regular nodes...")
        node_cfg = self.config['nodes']

        for i in range(self.n_anchors, self.n_nodes):
            # Random position
            x = np.random.uniform(0, 50)
            y = np.random.uniform(0, 50)
            z = np.random.uniform(0, 2)

            # Random velocity
            speed = np.random.uniform(0.5, 5.0)
            angle = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            # Clock errors
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

        print(f"  Created nodes 4-{self.n_nodes-1} with random positions and clock errors")

    def setup_gold_codes(self):
        """Generate Gold codes for ranging"""
        print(f"\nGenerating Gold codes for {self.n_nodes} nodes...")
        self.gold_gen = WorkingGoldCodeGenerator(length=127)
        self.gold_codes = [self.gold_gen.get_code(i) for i in range(self.n_nodes)]
        print(f"  Generated {len(self.gold_codes)} unique Gold codes")

    def run_time_synchronization(self):
        """Phase 1: Time synchronization"""
        print("\n" + "="*70)
        print("PHASE 1: TIME SYNCHRONIZATION")
        print("-"*70)

        # Use our working time sync
        sync = FixedTimeSync(n_nodes=self.n_nodes, n_anchors=self.n_anchors)

        # Copy our nodes' clock errors to sync system
        for i, node in enumerate(self.nodes):
            sync.nodes[i].true_offset_ns = node.true_offset_ns
            sync.nodes[i].true_drift_ppb = node.true_drift_ppb
            sync.nodes[i].is_anchor = node.is_anchor

        # Run synchronization
        print("\nRunning multi-phase synchronization protocol...")
        sync.run_synchronization(n_rounds=30)

        # Copy results back
        for i, node in enumerate(self.nodes):
            if not node.is_anchor:
                node.est_offset_ns = sync.nodes[i].est_offset_ns
                node.est_drift_ppb = sync.nodes[i].est_drift_ppb

        # Store results
        self.results['time_sync']['final_mean_error'] = sync.history['mean_errors'][-1]
        self.results['time_sync']['converged'] = sync.history['mean_errors'][-1] < 2.0
        self.results['time_sync']['history'] = sync.history

        return self.results['time_sync']['converged']

    def run_ranging(self):
        """Phase 2: Two-way ranging between all node pairs"""
        print("\n" + "="*70)
        print("PHASE 2: RANGING MEASUREMENTS")
        print("-"*70)

        n_pairs = self.n_nodes * (self.n_nodes - 1) // 2
        print(f"\nPerforming two-way ranging for {n_pairs} node pairs...")

        ranging_errors = []
        pair_count = 0

        # Measure ranges between all pairs
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                node_i = self.nodes[i]
                node_j = self.nodes[j]

                # True distance
                true_dist = np.linalg.norm(node_j.position - node_i.position)

                # Simulate ranging with noise (based on SNR)
                snr_db = 20 - 10 * np.log10(1 + true_dist / 20)
                noise_std = 0.5 / (10**(snr_db/20))  # meters

                # Two-way ranging measurement
                measured_dist = true_dist + np.random.normal(0, noise_std)

                # Store measurements
                node_i.ranging_measurements[j] = measured_dist
                node_j.ranging_measurements[i] = measured_dist

                error = measured_dist - true_dist
                ranging_errors.append(error)

                pair_count += 1
                if pair_count % 100 == 0:
                    print(f"  Completed {pair_count}/{n_pairs} measurements...")

        # Calculate statistics
        ranging_errors = np.array(ranging_errors)
        rmse = np.sqrt(np.mean(ranging_errors**2))
        mean_abs = np.mean(np.abs(ranging_errors))
        max_error = np.max(np.abs(ranging_errors))

        print(f"\nRanging Statistics:")
        print(f"  RMSE: {rmse:.3f}m")
        print(f"  Mean absolute error: {mean_abs:.3f}m")
        print(f"  Max error: {max_error:.3f}m")

        self.results['ranging']['rmse'] = rmse
        self.results['ranging']['errors'] = ranging_errors

    def run_localization(self):
        """Phase 3: Estimate positions using trilateration"""
        print("\n" + "="*70)
        print("PHASE 3: LOCALIZATION")
        print("-"*70)

        print("\nEstimating positions for regular nodes...")
        position_errors = []

        for node in self.nodes:
            if not node.is_anchor:
                # Get measurements to anchors
                anchor_positions = []
                distances = []

                for anchor in self.nodes[:self.n_anchors]:
                    if anchor.id in node.ranging_measurements:
                        anchor_positions.append(anchor.position[:2])  # 2D for simplicity
                        distances.append(node.ranging_measurements[anchor.id])

                if len(distances) >= 3:  # Need at least 3 anchors
                    # Simple trilateration (least squares)
                    est_pos = self.trilaterate(anchor_positions, distances)
                    node.est_position[:2] = est_pos

                    # Calculate error
                    error = np.linalg.norm(node.position[:2] - est_pos)
                    position_errors.append(error)

        # Statistics
        if position_errors:
            position_errors = np.array(position_errors)
            pos_rmse = np.sqrt(np.mean(position_errors**2))
            pos_mean = np.mean(position_errors)
            pos_max = np.max(position_errors)

            print(f"\nLocalization Statistics:")
            print(f"  Position RMSE: {pos_rmse:.3f}m")
            print(f"  Mean position error: {pos_mean:.3f}m")
            print(f"  Max position error: {pos_max:.3f}m")
            print(f"  Nodes localized: {len(position_errors)}/{self.n_nodes - self.n_anchors}")

            self.results['localization']['rmse'] = pos_rmse
            self.results['localization']['errors'] = position_errors

    def trilaterate(self, anchor_positions, distances):
        """Simple least squares trilateration"""
        anchor_positions = np.array(anchor_positions)
        distances = np.array(distances)

        # Least squares solution
        A = 2 * (anchor_positions[1:] - anchor_positions[0])
        b = (distances[0]**2 - distances[1:]**2 +
             np.sum(anchor_positions[1:]**2, axis=1) -
             np.sum(anchor_positions[0]**2))

        try:
            position = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            position = np.mean(anchor_positions, axis=0)  # Fallback

        return position

    def visualize_results(self):
        """Create comprehensive visualization"""
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("-"*70)

        fig = plt.figure(figsize=(16, 10))

        # 1. Network topology with true and estimated positions
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_title("Network Topology & Localization")
        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")
        ax1.set_xlim(-5, 55)
        ax1.set_ylim(-5, 55)
        ax1.grid(True, alpha=0.3)

        # Draw area
        rect = plt.Rectangle((0, 0), 50, 50, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)

        # Plot nodes
        for node in self.nodes:
            if node.is_anchor:
                ax1.scatter(node.position[0], node.position[1],
                           s=200, c='red', marker='^', edgecolor='black',
                           linewidth=2, zorder=5)
                ax1.text(node.position[0], node.position[1] - 2.5,
                        node.name, fontsize=8, ha='center')
            else:
                # True position
                ax1.scatter(node.position[0], node.position[1],
                           s=50, c='blue', marker='o', alpha=0.5)
                # Estimated position
                ax1.scatter(node.est_position[0], node.est_position[1],
                           s=30, c='green', marker='x')
                # Connect true and estimated
                ax1.plot([node.position[0], node.est_position[0]],
                        [node.position[1], node.est_position[1]],
                        'k-', alpha=0.2, linewidth=0.5)

        ax1.legend(['Anchor', 'True Position', 'Est. Position'], loc='upper right')

        # 2. Time sync convergence
        ax2 = plt.subplot(2, 3, 2)
        if 'time_sync' in self.results and 'history' in self.results['time_sync']:
            history = self.results['time_sync']['history']
            ax2.semilogy(history['rounds'], history['mean_errors'], 'b-', linewidth=2)
            ax2.axhline(y=1.0, color='g', linestyle=':', label='Target (<1ns)')
            ax2.set_xlabel('Synchronization Round')
            ax2.set_ylabel('Mean Clock Error (ns)')
            ax2.set_title('Time Synchronization Convergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Ranging error histogram
        ax3 = plt.subplot(2, 3, 3)
        if 'ranging' in self.results and 'errors' in self.results['ranging']:
            errors = self.results['ranging']['errors']
            ax3.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='red', linestyle='--', label='Perfect')
            ax3.set_xlabel('Ranging Error (m)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'Ranging Errors (RMSE={self.results["ranging"]["rmse"]:.3f}m)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Localization error map
        ax4 = plt.subplot(2, 3, 4)
        if 'localization' in self.results and 'errors' in self.results['localization']:
            # Create heatmap of position errors
            for i, node in enumerate(self.nodes):
                if not node.is_anchor and i - self.n_anchors < len(self.results['localization']['errors']):
                    error = self.results['localization']['errors'][i - self.n_anchors]
                    scatter = ax4.scatter(node.position[0], node.position[1],
                                        c=error, cmap='hot_r', s=100,
                                        vmin=0, vmax=2, edgecolor='black')

            plt.colorbar(scatter, ax=ax4, label='Position Error (m)')
            ax4.set_xlabel('X Position (m)')
            ax4.set_ylabel('Y Position (m)')
            ax4.set_title('Localization Error Heatmap')
            ax4.set_xlim(0, 50)
            ax4.set_ylim(0, 50)
            ax4.grid(True, alpha=0.3)

        # 5. Clock offset distribution
        ax5 = plt.subplot(2, 3, 5)
        initial_offsets = [n.true_offset_ns for n in self.nodes if not n.is_anchor]
        final_offsets = [n.est_offset_ns for n in self.nodes if not n.is_anchor]

        ax5.hist(initial_offsets, bins=20, alpha=0.5, label='Initial', color='red')
        ax5.hist(final_offsets, bins=20, alpha=0.5, label='After Sync', color='green')
        ax5.set_xlabel('Clock Offset (ns)')
        ax5.set_ylabel('Number of Nodes')
        ax5.set_title('Clock Offset Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        stats_text = f"""SIMULATION SUMMARY

Network Configuration:
• Total nodes: {self.n_nodes}
• Anchor nodes: {self.n_anchors}
• Area: 50×50m

Time Synchronization:
• Converged: {'✓' if self.results['time_sync']['converged'] else '✗'}
• Final error: {self.results['time_sync']['final_mean_error']:.2f}ns

Ranging Performance:
• Measurements: {self.n_nodes * (self.n_nodes - 1) // 2}
• RMSE: {self.results['ranging']['rmse']:.3f}m

Localization Results:
• Position RMSE: {self.results['localization']['rmse']:.3f}m
• Nodes localized: {len(self.results['localization']['errors'])}/{self.n_nodes - self.n_anchors}
"""
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace')

        plt.suptitle('FTL System - Complete Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        plt.savefig('ftl_full_simulation_results.png', dpi=150)
        print("\n✅ Results saved to ftl_full_simulation_results.png")
        plt.show()

    def run(self):
        """Run complete simulation"""
        start_time = time.time()

        # Phase 1: Time Synchronization
        sync_success = self.run_time_synchronization()
        if not sync_success:
            print("⚠️  Warning: Time sync did not fully converge")

        # Phase 2: Ranging
        self.run_ranging()

        # Phase 3: Localization
        self.run_localization()

        # Visualization
        self.visualize_results()

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total execution time: {elapsed:.1f}s")
        print(f"\nFinal Performance Metrics:")
        print(f"  • Time sync accuracy: {self.results['time_sync']['final_mean_error']:.2f}ns")
        print(f"  • Ranging RMSE: {self.results['ranging']['rmse']:.3f}m")
        print(f"  • Position RMSE: {self.results['localization']['rmse']:.3f}m")


if __name__ == "__main__":
    # Run the full simulation
    sim = FullFTLSimulation("config_30nodes.yaml")
    sim.run()