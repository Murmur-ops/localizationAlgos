#!/usr/bin/env python3
"""
Fixed FTL implementation with proper numerical scaling
5-node system: 3 anchors, 2 unknowns
Shows convergence for both position and time synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FixedFTLConfig:
    """Configuration with proper scaling fixes"""
    # Network
    n_nodes: int = 8  # 3 anchors + 5 unknowns
    n_anchors: int = 3

    # Measurements
    measurement_std: float = 0.01  # 1 cm (realistic, avoids extreme scaling)

    # Base optimization parameters (before whitening adjustment)
    base_step_size: float = 0.5
    base_damping: float = 1e-4

    # Consensus
    consensus_gain: float = 0.1
    use_metropolis: bool = True

    # Convergence
    max_iterations: int = 100
    gradient_tol: float = 1e-12

    @property
    def whitening_scale(self):
        """Whitening scales by 1/σ"""
        return 1.0 / self.measurement_std

    @property
    def step_size(self):
        """Step size must be scaled by whitening factor squared"""
        return self.base_step_size / (self.whitening_scale ** 2)

    @property
    def damping(self):
        """LM damping must be scaled by whitening factor squared"""
        return self.base_damping * (self.whitening_scale ** 2)


class FixedFTLNode:
    """Node with corrected gradient computation"""

    def __init__(self, node_id: int, true_pos: np.ndarray, is_anchor: bool, config: FixedFTLConfig):
        self.id = node_id
        self.true_pos = true_pos
        self.is_anchor = is_anchor
        self.config = config

        # State: [x, y, clock_bias_ns, clock_drift_ppb, cfo_ppm]
        self.state = np.zeros(5)
        if is_anchor:
            # Anchors have perfect position and zero clock errors
            self.state[:2] = true_pos
            self.state[2:] = 0
        else:
            # Initialize unknowns with error
            self.state[:2] = true_pos + np.random.normal(0, 2.0, 2)  # 2m position error
            self.state[2] = np.random.normal(0, 10)  # 10 ns clock bias error
            self.state[3] = np.random.normal(0, 1)   # 1 ppb drift error

        self.measurements = []
        self.neighbors = {}

        # History tracking
        self.state_history = [self.state.copy()]
        self.error_history = []

    def compute_gradient_hessian(self, all_states: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient and Hessian from measurements"""
        n = 5  # State dimension
        H = np.zeros((n, n))
        g = np.zeros(n)

        c = 299792458.0  # Speed of light in m/s

        for meas in self.measurements:
            i, j = meas['i'], meas['j']

            # Get states
            if i == self.id:
                xi = self.state
                xj = all_states.get(j)
            else:
                xi = all_states.get(i)
                xj = self.state

            if xi is None or xj is None:
                continue

            # Extract positions and clock parameters
            pi, pj = xi[:2], xj[:2]
            bi_ns, bj_ns = xi[2], xj[2]  # Clock biases in nanoseconds

            # Geometric distance
            delta_p = pj - pi
            dist = np.linalg.norm(delta_p)

            if dist < 1e-10:
                continue

            # Predicted range including clock bias
            # Range = geometric_distance + c * (clock_j - clock_i) / 1e9
            clock_contribution = (bj_ns - bi_ns) * c * 1e-9  # Convert ns to meters
            predicted_range = dist + clock_contribution

            # Residual (measured - predicted)
            measured_range = meas['range']
            residual = measured_range - predicted_range

            # Jacobian components
            u = delta_p / dist  # Unit vector from i to j

            if i == self.id:
                # Jacobian w.r.t. node i
                J = np.zeros(n)
                J[0:2] = -u  # Position components
                J[2] = -c * 1e-9  # Clock bias component (m/ns)
            else:
                # Jacobian w.r.t. node j
                J = np.zeros(n)
                J[0:2] = u  # Position components
                J[2] = c * 1e-9  # Clock bias component (m/ns)

            # Apply whitening
            std = self.config.measurement_std
            r_wh = residual / std
            J_wh = J / std

            # Add to normal equations (Gauss-Newton approximation)
            H += np.outer(J_wh, J_wh)
            g += J_wh * r_wh

        return H, g

    def add_consensus(self, H: np.ndarray, g: np.ndarray, neighbor_states: Dict[int, np.ndarray]):
        """Add consensus penalty with Metropolis weights"""
        if self.is_anchor or len(neighbor_states) == 0:
            return H, g

        mu = self.config.consensus_gain

        if self.config.use_metropolis:
            # Metropolis weights for doubly-stochastic averaging
            degree_i = len(self.neighbors)

            for neighbor_id, neighbor_state in neighbor_states.items():
                # Simplified: assume similar degrees
                weight = 1.0 / (1 + max(degree_i, degree_i))

                # Consensus gradient: w * μ * (x_i - x_j)
                diff = self.state - neighbor_state
                g += mu * weight * diff
                H += mu * weight * np.eye(5)
        else:
            # Simple uniform averaging
            n_neighbors = len(neighbor_states)
            for neighbor_state in neighbor_states.values():
                diff = self.state - neighbor_state
                g += mu * diff / n_neighbors
            H += mu * np.eye(5)

        return H, g

    def update(self, H: np.ndarray, g: np.ndarray):
        """Gauss-Newton update with proper scaling"""
        if self.is_anchor:
            # Record history but don't update
            self.state_history.append(self.state.copy())
            pos_error = 0
            time_error = 0
        else:
            # Add Levenberg-Marquardt damping
            lambda_lm = self.config.damping
            diag_H = np.maximum(np.diag(H), 1e-6)
            H_damped = H + lambda_lm * np.diag(diag_H)

            try:
                # Solve H * delta = g
                delta = np.linalg.solve(H_damped, g)

                # Update with scaled step size
                self.state = self.state - self.config.step_size * delta
            except np.linalg.LinAlgError:
                # Gradient descent fallback
                norm_g = np.linalg.norm(g)
                if norm_g > 1e-10:
                    self.state = self.state - self.config.step_size * g / norm_g

            # Record history
            self.state_history.append(self.state.copy())

            # Compute errors
            pos_error = np.linalg.norm(self.state[:2] - self.true_pos)
            time_error = abs(self.state[2])  # Clock bias error in ns

        self.error_history.append({'pos': pos_error, 'time': time_error})
        return pos_error, time_error


def run_fixed_ftl():
    """Run FTL with numerical fixes"""

    config = FixedFTLConfig()

    print("=" * 60)
    print(f"Fixed FTL System - {config.n_nodes} Nodes ({config.n_anchors} Anchors, {config.n_nodes - config.n_anchors} Unknowns)")
    print("=" * 60)
    print(f"Measurement std: {config.measurement_std*100:.1f} cm")
    print(f"Whitening scale: {config.whitening_scale:.0f}x")
    print(f"Base step size: {config.base_step_size}")
    print(f"Adjusted step size: {config.step_size:.2e}")
    print(f"Base damping: {config.base_damping:.1e}")
    print(f"Adjusted damping: {config.damping:.1e}")
    print()

    # Create network geometry
    area_size = 20.0

    # Positions: 3 anchors in triangle, 5 unknowns distributed
    true_positions = np.array([
        [0, 0],                        # Anchor 0
        [area_size, 0],                # Anchor 1
        [area_size/2, area_size],      # Anchor 2
        [area_size/4, area_size/4],    # Unknown 3
        [3*area_size/4, area_size/4],  # Unknown 4
        [area_size/2, area_size/2],    # Unknown 5 (center)
        [area_size/4, 3*area_size/4],  # Unknown 6
        [3*area_size/4, 3*area_size/4], # Unknown 7
    ])

    # Create nodes
    nodes = []
    for i in range(config.n_nodes):
        is_anchor = i < config.n_anchors
        node = FixedFTLNode(i, true_positions[i], is_anchor, config)
        nodes.append(node)

    print("Initial states:")
    for node in nodes:
        if node.is_anchor:
            print(f"  Node {node.id} (Anchor): pos=[{node.state[0]:.2f}, {node.state[1]:.2f}], clock=0.00 ns")
        else:
            error = np.linalg.norm(node.state[:2] - node.true_pos)
            print(f"  Node {node.id} (Unknown): pos=[{node.state[0]:.2f}, {node.state[1]:.2f}], "
                  f"error={error:.2f}m, clock={node.state[2]:.2f} ns")
    print()

    # Create measurements (perfect distances)
    measurements = []
    for i in range(config.n_nodes):
        for j in range(i+1, config.n_nodes):
            # True geometric distance
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])

            # For zero-noise test, use perfect measurements
            measured_range = true_dist

            meas = {'i': i, 'j': j, 'range': measured_range}
            measurements.append(meas)

            # Add to both nodes
            nodes[i].measurements.append(meas)
            nodes[j].measurements.append(meas)

            # Mark as neighbors
            nodes[i].neighbors[j] = True
            nodes[j].neighbors[i] = True

    print(f"Created {len(measurements)} perfect range measurements")
    print()

    # Run optimization
    print("Running optimization...")
    for iteration in range(config.max_iterations):
        # Get all current states
        all_states = {node.id: node.state.copy() for node in nodes}

        # Each node computes its update
        max_pos_error = 0
        max_time_error = 0

        for node in nodes:
            # Compute local gradient and Hessian
            H, g = node.compute_gradient_hessian(all_states)

            # Add consensus if not an anchor
            if not node.is_anchor:
                neighbor_states = {nid: all_states[nid] for nid in node.neighbors}
                H, g = node.add_consensus(H, g, neighbor_states)

            # Update state
            pos_err, time_err = node.update(H, g)

            if not node.is_anchor:
                max_pos_error = max(max_pos_error, pos_err)
                max_time_error = max(max_time_error, time_err)

        # Print progress
        if iteration % 10 == 0 or max_pos_error < 1e-9:
            print(f"  Iter {iteration:3d}: max pos error = {max_pos_error:.3e} m, "
                  f"max time error = {max_time_error:.3e} ns")

        # Check convergence
        if max_pos_error < 1e-12 and max_time_error < 1e-12:
            print(f"\nConverged to machine precision at iteration {iteration}!")
            break

    print("\nFinal states:")
    for node in nodes:
        if node.is_anchor:
            print(f"  Node {node.id} (Anchor): pos=[{node.state[0]:.6f}, {node.state[1]:.6f}]")
        else:
            error = np.linalg.norm(node.state[:2] - node.true_pos)
            print(f"  Node {node.id} (Unknown): pos=[{node.state[0]:.6f}, {node.state[1]:.6f}], "
                  f"error={error:.3e}m, clock={node.state[2]:.3e} ns")

    return nodes, true_positions


def plot_results(nodes, true_positions):
    """Generate convergence and position plots"""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Extract unknown nodes
    unknown_nodes = [n for n in nodes if not n.is_anchor]
    iterations = range(len(unknown_nodes[0].error_history))

    # Plot 1: Position convergence for each unknown node
    ax1 = plt.subplot(2, 3, 1)
    for node in unknown_nodes:
        pos_errors = [e['pos'] for e in node.error_history]
        ax1.semilogy(iterations, pos_errors, linewidth=2, label=f'Node {node.id}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='1mm')
    ax1.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='1μm')
    ax1.axhline(y=1e-9, color='m', linestyle='--', alpha=0.5, label='1nm')
    ax1.legend()

    # Plot 2: Time synchronization convergence
    ax2 = plt.subplot(2, 3, 2)
    for node in unknown_nodes:
        time_errors = [e['time'] for e in node.error_history]
        # Handle zero values for log plot
        time_errors = [max(e, 1e-15) for e in time_errors]
        ax2.semilogy(iterations, time_errors, linewidth=2, label=f'Node {node.id}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Clock Bias Error (ns)')
    ax2.set_title('Time Synchronization Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='1ps')
    ax2.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='1fs')
    ax2.legend()

    # Plot 3: Combined RMSE
    ax3 = plt.subplot(2, 3, 3)
    pos_rmse = []
    time_rmse = []
    for i in iterations:
        pos_errors = [n.error_history[i]['pos'] for n in unknown_nodes]
        time_errors = [n.error_history[i]['time'] for n in unknown_nodes]
        pos_rmse.append(np.sqrt(np.mean(np.array(pos_errors)**2)))
        time_rmse.append(np.sqrt(np.mean(np.array(time_errors)**2)))

    ax3.semilogy(iterations, pos_rmse, 'b-', linewidth=2, label='Position RMSE')
    ax3_twin = ax3.twinx()
    ax3_twin.semilogy(iterations, time_rmse, 'g-', linewidth=2, label='Time RMSE')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Position RMSE (m)', color='b')
    ax3_twin.set_ylabel('Time RMSE (ns)', color='g')
    ax3.set_title('Combined RMSE Convergence')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='g')

    # Plot 4: X-coordinate convergence
    ax4 = plt.subplot(2, 3, 4)
    for node in unknown_nodes:
        x_history = [s[0] for s in node.state_history]
        ax4.plot(x_history, linewidth=2, label=f'Node {node.id}')
        ax4.axhline(y=node.true_pos[0], color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('X Position (m)')
    ax4.set_title('X-Coordinate Convergence')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Y-coordinate convergence
    ax5 = plt.subplot(2, 3, 5)
    for node in unknown_nodes:
        y_history = [s[1] for s in node.state_history]
        ax5.plot(y_history, linewidth=2, label=f'Node {node.id}')
        ax5.axhline(y=node.true_pos[1], color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('Y-Coordinate Convergence')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Clock bias convergence
    ax6 = plt.subplot(2, 3, 6)
    for node in unknown_nodes:
        clock_history = [s[2] for s in node.state_history]
        ax6.plot(clock_history, linewidth=2, label=f'Node {node.id}')
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='True (0)')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Clock Bias (ns)')
    ax6.set_title('Clock Bias Convergence')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.suptitle('FTL Convergence Analysis - Per Node', fontsize=14)
    plt.tight_layout()
    plt.savefig('ftl_convergence_per_node.png', dpi=150)
    plt.show()

    # Second figure: Estimated vs Actual positions
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 7: Initial positions
    ax7.scatter(true_positions[:3, 0], true_positions[:3, 1],
                s=200, c='red', marker='^', label='Anchors', zorder=5)
    ax7.scatter(true_positions[3:, 0], true_positions[3:, 1],
                s=150, c='green', marker='*', label='True Unknown', zorder=5)

    for node in nodes:
        if not node.is_anchor:
            initial_pos = node.state_history[0][:2]
            ax7.scatter(initial_pos[0], initial_pos[1],
                       s=100, c='blue', marker='o', alpha=0.6)
            ax7.plot([node.true_pos[0], initial_pos[0]],
                    [node.true_pos[1], initial_pos[1]],
                    'b--', alpha=0.3)
            ax7.annotate(f'{node.id}', xy=initial_pos, xytext=(3, 3),
                        textcoords='offset points', fontsize=9)

    ax7.set_xlabel('X Position (m)')
    ax7.set_ylabel('Y Position (m)')
    ax7.set_title('Initial Position Estimates')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.axis('equal')
    ax7.set_xlim([-2, 22])
    ax7.set_ylim([-2, 22])

    # Plot 8: Final positions
    ax8.scatter(true_positions[:3, 0], true_positions[:3, 1],
                s=200, c='red', marker='^', label='Anchors', zorder=5)
    ax8.scatter(true_positions[3:, 0], true_positions[3:, 1],
                s=150, c='green', marker='*', label='True Unknown', zorder=5)

    for node in nodes:
        if not node.is_anchor:
            final_pos = node.state[:2]
            ax8.scatter(final_pos[0], final_pos[1],
                       s=100, c='blue', marker='o', alpha=0.6, label=f'Est Node {node.id}')
            error = np.linalg.norm(final_pos - node.true_pos)
            ax8.annotate(f'{node.id}\n({error:.1e}m)',
                        xy=final_pos, xytext=(3, 3),
                        textcoords='offset points', fontsize=9)

    # Draw the trajectory
    for node in unknown_nodes:
        trajectory = np.array([s[:2] for s in node.state_history])
        ax8.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3, linewidth=1)

    ax8.set_xlabel('X Position (m)')
    ax8.set_ylabel('Y Position (m)')
    ax8.set_title('Final Position Estimates')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.axis('equal')
    ax8.set_xlim([-2, 22])
    ax8.set_ylim([-2, 22])

    plt.suptitle('Estimated vs Actual Positions', fontsize=14)
    plt.tight_layout()
    plt.savefig('ftl_positions.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Run the fixed FTL system
    nodes, true_positions = run_fixed_ftl()

    # Generate plots
    plot_results(nodes, true_positions)

    print("\n" + "="*60)
    print("Plots saved:")
    print("  - ftl_convergence_per_node.png")
    print("  - ftl_positions.png")
    print("="*60)