#!/usr/bin/env python3
"""
Working FTL implementation with correct gradient computation
8 nodes: 3 anchors + 5 unknowns
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleFTL:
    """Simple centralized FTL to verify algorithm works"""

    def __init__(self, n_nodes=8, n_anchors=3):
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors
        self.n_unknowns = n_nodes - n_anchors

        # Create positions
        area_size = 20.0
        self.true_positions = np.array([
            [0, 0],                        # Anchor 0
            [area_size, 0],                # Anchor 1
            [area_size/2, area_size],      # Anchor 2
            [area_size/4, area_size/4],    # Unknown 3
            [3*area_size/4, area_size/4],  # Unknown 4
            [area_size/2, area_size/2],    # Unknown 5 (center)
            [area_size/4, 3*area_size/4],  # Unknown 6
            [3*area_size/4, 3*area_size/4], # Unknown 7
        ])

        # Initialize states (position + clock bias for unknowns)
        self.states = np.zeros((n_nodes, 3))  # [x, y, clock_bias_ns]

        # Anchors: perfect position, zero clock
        for i in range(n_anchors):
            self.states[i, :2] = self.true_positions[i]
            self.states[i, 2] = 0

        # Unknowns: initial error
        np.random.seed(42)
        for i in range(n_anchors, n_nodes):
            self.states[i, :2] = self.true_positions[i] + np.random.normal(0, 2, 2)
            self.states[i, 2] = np.random.normal(0, 10)  # 10ns clock error

        # Create measurements
        self.measurements = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                self.measurements.append({
                    'i': i,
                    'j': j,
                    'dist': true_dist
                })

        # History
        self.position_errors = {i: [] for i in range(n_anchors, n_nodes)}
        self.clock_errors = {i: [] for i in range(n_anchors, n_nodes)}

    def compute_gradient_hessian(self):
        """Compute gradient and Hessian for all unknowns"""
        n_vars = self.n_unknowns * 3  # x, y, clock for each unknown
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        c = 299792458.0  # m/s

        for meas in self.measurements:
            i, j = meas['i'], meas['j']
            true_dist = meas['dist']

            # Skip if both are anchors
            if i < self.n_anchors and j < self.n_anchors:
                continue

            # Get positions and clock biases
            pi = self.states[i, :2]
            pj = self.states[j, :2]
            bi_ns = self.states[i, 2]
            bj_ns = self.states[j, 2]

            # Predicted measurement
            geom_dist = np.linalg.norm(pi - pj)
            if geom_dist < 1e-10:
                continue

            # Range including clock: r = ||p_i - p_j|| + c*(b_j - b_i)/1e9
            clock_term = (bj_ns - bi_ns) * c * 1e-9
            predicted_range = geom_dist + clock_term

            # Residual
            residual = true_dist - predicted_range

            # Unit vector
            u = (pj - pi) / geom_dist

            # Build Jacobians
            if i >= self.n_anchors:
                idx = (i - self.n_anchors) * 3
                Ji = np.zeros(n_vars)
                Ji[idx] = -u[0]      # ∂r/∂x_i
                Ji[idx+1] = -u[1]    # ∂r/∂y_i
                Ji[idx+2] = -c*1e-9  # ∂r/∂b_i (meters per nanosecond)

                H += np.outer(Ji, Ji)
                g += Ji * residual

            if j >= self.n_anchors:
                idx = (j - self.n_anchors) * 3
                Jj = np.zeros(n_vars)
                Jj[idx] = u[0]       # ∂r/∂x_j
                Jj[idx+1] = u[1]     # ∂r/∂y_j
                Jj[idx+2] = c*1e-9   # ∂r/∂b_j

                H += np.outer(Jj, Jj)
                g += Jj * residual

        return H, g

    def step(self, step_size=1.0, damping=1e-9):
        """One optimization step"""
        H, g = self.compute_gradient_hessian()

        # Add damping
        H += damping * np.eye(len(H))

        # Solve
        try:
            delta = np.linalg.solve(H, g)
        except:
            return False

        # Update states
        for i in range(self.n_unknowns):
            idx = i * 3
            node_id = i + self.n_anchors
            self.states[node_id] += step_size * delta[idx:idx+3]

        # Record errors
        for i in range(self.n_anchors, self.n_nodes):
            pos_error = np.linalg.norm(self.states[i, :2] - self.true_positions[i])
            clock_error = abs(self.states[i, 2])
            self.position_errors[i].append(pos_error)
            self.clock_errors[i].append(clock_error)

        return True

    def run(self, max_iters=200):
        """Run optimization"""
        print("Initial states:")
        for i in range(self.n_nodes):
            if i < self.n_anchors:
                print(f"  Node {i} (Anchor): pos={self.states[i,:2]}")
            else:
                err = np.linalg.norm(self.states[i,:2] - self.true_positions[i])
                print(f"  Node {i} (Unknown): pos={self.states[i,:2]}, error={err:.2f}m, clock={self.states[i,2]:.1f}ns")

        for iter in range(max_iters):
            if not self.step():
                break

            # Check convergence
            max_pos_err = max(self.position_errors[i][-1] for i in range(self.n_anchors, self.n_nodes))
            max_clock_err = max(self.clock_errors[i][-1] for i in range(self.n_anchors, self.n_nodes))

            if iter % 20 == 0 or (iter < 20 and iter % 5 == 0):
                print(f"Iter {iter:3d}: max pos error = {max_pos_err:.3e}m, max clock error = {max_clock_err:.3e}ns")

            if max_pos_err < 1e-12 and max_clock_err < 1e-12:
                print(f"Converged to machine precision at iteration {iter}")
                break

        print("\nFinal states:")
        for i in range(self.n_nodes):
            if i < self.n_anchors:
                print(f"  Node {i} (Anchor): pos={self.states[i,:2]}")
            else:
                err = np.linalg.norm(self.states[i,:2] - self.true_positions[i])
                print(f"  Node {i} (Unknown): pos={self.states[i,:2]}, error={err:.3e}m, clock={self.states[i,2]:.3e}ns")

    def plot_results(self):
        """Plot convergence and positions"""

        # Figure 1: Convergence plots
        fig1, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Position convergence per node
        ax = axes[0, 0]
        for i in range(self.n_anchors, self.n_nodes):
            ax.semilogy(self.position_errors[i], label=f'Node {i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Convergence per Node')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Clock convergence per node
        ax = axes[0, 1]
        for i in range(self.n_anchors, self.n_nodes):
            errors = [max(e, 1e-15) for e in self.clock_errors[i]]  # Avoid log(0)
            ax.semilogy(errors, label=f'Node {i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Clock Bias Error (ns)')
        ax.set_title('Clock Bias Convergence per Node')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Combined RMSE
        ax = axes[0, 2]
        iterations = range(len(self.position_errors[self.n_anchors]))
        pos_rmse = []
        clock_rmse = []
        for iter in iterations:
            pos_errs = [self.position_errors[i][iter] for i in range(self.n_anchors, self.n_nodes)]
            clock_errs = [self.clock_errors[i][iter] for i in range(self.n_anchors, self.n_nodes)]
            pos_rmse.append(np.sqrt(np.mean(np.array(pos_errs)**2)))
            clock_rmse.append(np.sqrt(np.mean(np.array(clock_errs)**2)))

        ax.semilogy(pos_rmse, 'b-', label='Position RMSE')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('Combined RMSE')
        ax2 = ax.twinx()
        ax2.semilogy(clock_rmse, 'g-', label='Clock RMSE')
        ax2.set_ylabel('Clock RMSE (ns)', color='g')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # X convergence
        ax = axes[1, 0]
        for i in range(self.n_anchors, self.n_nodes):
            x_history = [self.states[i, 0]] * (len(self.position_errors[i]) + 1)
            ax.plot(x_history[:len(self.position_errors[i])], label=f'Node {i}')
            ax.axhline(y=self.true_positions[i, 0], color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('X Position (m)')
        ax.set_title('X Coordinate Convergence')
        ax.grid(True, alpha=0.3)

        # Y convergence
        ax = axes[1, 1]
        for i in range(self.n_anchors, self.n_nodes):
            y_history = [self.states[i, 1]] * (len(self.position_errors[i]) + 1)
            ax.plot(y_history[:len(self.position_errors[i])], label=f'Node {i}')
            ax.axhline(y=self.true_positions[i, 1], color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Y Coordinate Convergence')
        ax.grid(True, alpha=0.3)

        # Clock bias final values
        ax = axes[1, 2]
        nodes = list(range(self.n_anchors, self.n_nodes))
        final_clocks = [self.states[i, 2] for i in nodes]
        ax.bar(nodes, final_clocks)
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Clock Bias (ns)')
        ax.set_title('Final Clock Biases')
        ax.grid(True, alpha=0.3)

        plt.suptitle('FTL Convergence Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('ftl_convergence_working.png', dpi=150)
        plt.show()

        # Figure 2: Position estimates
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Initial positions
        ax1.scatter(self.true_positions[:self.n_anchors, 0],
                   self.true_positions[:self.n_anchors, 1],
                   s=200, c='red', marker='^', label='Anchors', zorder=5)
        ax1.scatter(self.true_positions[self.n_anchors:, 0],
                   self.true_positions[self.n_anchors:, 1],
                   s=150, c='green', marker='*', label='True Unknown', zorder=4)

        # Show initial estimates with error lines
        for i in range(self.n_anchors, self.n_nodes):
            # Approximate initial position (before first update)
            init_error = self.position_errors[i][0] if self.position_errors[i] else 0
            direction = (self.states[i, :2] - self.true_positions[i])
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            init_pos = self.true_positions[i] + direction * init_error

            ax1.scatter(init_pos[0], init_pos[1], s=100, c='blue', marker='o', alpha=0.6)
            ax1.plot([self.true_positions[i, 0], init_pos[0]],
                    [self.true_positions[i, 1], init_pos[1]],
                    'b--', alpha=0.3)
            ax1.annotate(f'{i}', xy=init_pos, xytext=(3,3), textcoords='offset points')

        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Initial Position Estimates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.set_xlim([-2, 22])
        ax1.set_ylim([-2, 22])

        # Final positions
        ax2.scatter(self.true_positions[:self.n_anchors, 0],
                   self.true_positions[:self.n_anchors, 1],
                   s=200, c='red', marker='^', label='Anchors', zorder=5)
        ax2.scatter(self.true_positions[self.n_anchors:, 0],
                   self.true_positions[self.n_anchors:, 1],
                   s=150, c='green', marker='*', label='True Unknown', zorder=4)

        for i in range(self.n_anchors, self.n_nodes):
            final_pos = self.states[i, :2]
            ax2.scatter(final_pos[0], final_pos[1], s=100, c='blue', marker='o', alpha=0.6)
            error = np.linalg.norm(final_pos - self.true_positions[i])
            ax2.annotate(f'{i}\n({error:.1e}m)', xy=final_pos, xytext=(3,3),
                        textcoords='offset points', fontsize=9)

        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Final Position Estimates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.set_xlim([-2, 22])
        ax2.set_ylim([-2, 22])

        plt.suptitle('Estimated vs Actual Positions', fontsize=14)
        plt.tight_layout()
        plt.savefig('ftl_positions_working.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Working FTL Implementation")
    print("8 Nodes: 3 Anchors + 5 Unknowns")
    print("="*60)
    print()

    ftl = SimpleFTL()
    ftl.run(max_iters=50)
    ftl.plot_results()

    print("\n" + "="*60)
    print("Results:")
    print("  - Convergence plot: ftl_convergence_working.png")
    print("  - Position plot: ftl_positions_working.png")
    print("="*60)