#!/usr/bin/env python3
"""
Plot relative time offset convergence
Shows initial clock biases normally distributed and converging to consensus
"""

import numpy as np
import matplotlib.pyplot as plt

class FTLWithRelativeTime:
    """FTL focusing on relative time offset visualization"""

    def __init__(self, n_nodes=8, n_anchors=3, initial_clock_std=20.0):
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

        # Initialize states
        self.states = np.zeros((n_nodes, 3))  # [x, y, clock_bias_ns]

        # Set random seed for reproducibility
        np.random.seed(42)

        # Anchors: perfect position, but can have initial clock bias
        # In a real system, anchors might be synchronized to a reference
        # For this demo, we'll give anchors small initial clock biases
        for i in range(n_anchors):
            self.states[i, :2] = self.true_positions[i]
            self.states[i, 2] = np.random.normal(0, initial_clock_std/10)  # Anchors have smaller bias

        # Unknowns: position error and larger clock bias
        for i in range(n_anchors, n_nodes):
            self.states[i, :2] = self.true_positions[i] + np.random.normal(0, 2, 2)
            self.states[i, 2] = np.random.normal(0, initial_clock_std)  # Larger initial clock bias

        # Store initial clock biases for plotting
        self.initial_clocks = self.states[:, 2].copy()

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

        # History for all nodes (including anchors)
        self.clock_history = {i: [self.states[i, 2]] for i in range(n_nodes)}
        self.relative_clock_history = []

    def compute_relative_offsets(self):
        """Compute relative clock offsets to reference (mean of anchors)"""
        # Use mean of anchor clocks as reference
        anchor_mean = np.mean([self.states[i, 2] for i in range(self.n_anchors)])

        relative_offsets = {}
        for i in range(self.n_nodes):
            relative_offsets[i] = self.states[i, 2] - anchor_mean

        return relative_offsets

    def compute_gradient_hessian(self):
        """Compute gradient and Hessian"""
        n_vars = self.n_nodes * 3  # x, y, clock for ALL nodes (including anchors)
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        c = 299792458.0  # m/s

        for meas in self.measurements:
            i, j = meas['i'], meas['j']
            true_dist = meas['dist']

            # Get positions and clock biases
            pi = self.states[i, :2]
            pj = self.states[j, :2]
            bi_ns = self.states[i, 2]
            bj_ns = self.states[j, 2]

            # Predicted measurement
            geom_dist = np.linalg.norm(pi - pj)
            if geom_dist < 1e-10:
                continue

            clock_term = (bj_ns - bi_ns) * c * 1e-9
            predicted_range = geom_dist + clock_term

            # Residual
            residual = true_dist - predicted_range

            # Unit vector
            u = (pj - pi) / geom_dist

            # Build Jacobians for both nodes
            # Node i
            idx_i = i * 3
            Ji = np.zeros(n_vars)
            if i >= self.n_anchors:  # Only update position for unknowns
                Ji[idx_i] = -u[0]
                Ji[idx_i+1] = -u[1]
            Ji[idx_i+2] = -c*1e-9  # Always update clock

            # Node j
            idx_j = j * 3
            Jj = np.zeros(n_vars)
            if j >= self.n_anchors:  # Only update position for unknowns
                Jj[idx_j] = u[0]
                Jj[idx_j+1] = u[1]
            Jj[idx_j+2] = c*1e-9  # Always update clock

            H += np.outer(Ji, Ji) + np.outer(Jj, Jj)
            g += Ji * residual + Jj * residual

        return H, g

    def step(self, step_size=1.0, damping=1e-9):
        """One optimization step"""
        H, g = self.compute_gradient_hessian()

        # Add damping
        H += damping * np.eye(len(H))

        # Fix gauge freedom: pin one anchor's clock
        # This prevents the entire clock network from drifting
        reference_idx = 0  # Use first anchor as reference
        H[reference_idx*3 + 2, :] = 0
        H[:, reference_idx*3 + 2] = 0
        H[reference_idx*3 + 2, reference_idx*3 + 2] = 1
        g[reference_idx*3 + 2] = 0

        # Solve
        try:
            delta = np.linalg.solve(H, g)
        except:
            return False

        # Update states
        for i in range(self.n_nodes):
            idx = i * 3
            if i >= self.n_anchors:
                # Update position for unknowns
                self.states[i, :2] += step_size * delta[idx:idx+2]
            # Update clock for all nodes (except reference)
            if i != 0:
                self.states[i, 2] += step_size * delta[idx+2]

        # Record clock history
        for i in range(self.n_nodes):
            self.clock_history[i].append(self.states[i, 2])

        # Record relative offsets
        rel_offsets = self.compute_relative_offsets()
        self.relative_clock_history.append(rel_offsets)

        return True

    def run(self, max_iters=100):
        """Run optimization"""
        print("Initial clock biases (ns):")
        print(f"  Anchors: {[f'{self.initial_clocks[i]:.1f}' for i in range(self.n_anchors)]}")
        print(f"  Unknowns: {[f'{self.initial_clocks[i]:.1f}' for i in range(self.n_anchors, self.n_nodes)]}")
        print(f"  Range: [{self.initial_clocks.min():.1f}, {self.initial_clocks.max():.1f}] ns")
        print()

        # Record initial relative offsets
        rel_offsets = self.compute_relative_offsets()
        self.relative_clock_history.append(rel_offsets)

        for iter in range(max_iters):
            if not self.step():
                break

            if iter % 10 == 0:
                max_clock_spread = max(self.states[:, 2]) - min(self.states[:, 2])
                print(f"Iter {iter:3d}: clock spread = {max_clock_spread:.3e} ns")

            if max_clock_spread < 1e-10:
                print(f"Converged at iteration {iter}")
                break

        print(f"\nFinal clock biases (ns):")
        print(f"  All nodes: {[f'{self.states[i, 2]:.3e}' for i in range(self.n_nodes)]}")
        print(f"  Range: [{self.states[:, 2].min():.3e}, {self.states[:, 2].max():.3e}] ns")

    def plot_relative_time_convergence(self):
        """Plot relative time offset convergence"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = range(len(self.relative_clock_history))
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_nodes))

        # Plot 1: Relative time offset convergence (all nodes)
        ax = axes[0, 0]
        for i in range(self.n_nodes):
            offsets = [self.relative_clock_history[t][i] for t in iterations]
            label = f"Anchor {i}" if i < self.n_anchors else f"Node {i}"
            linestyle = '-' if i < self.n_anchors else '--'
            ax.plot(offsets, label=label, linestyle=linestyle, color=colors[i], linewidth=2)

        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Clock Offset (ns)')
        ax.set_title('Relative Time Offset Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=2)

        # Plot 2: Log-scale convergence
        ax = axes[0, 1]
        for i in range(self.n_nodes):
            offsets = [abs(self.relative_clock_history[t][i]) for t in iterations]
            # Avoid log(0)
            offsets = [max(o, 1e-15) for o in offsets]
            label = f"Anchor {i}" if i < self.n_anchors else f"Node {i}"
            linestyle = '-' if i < self.n_anchors else '--'
            ax.semilogy(offsets, label=label, linestyle=linestyle, color=colors[i], linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('|Relative Clock Offset| (ns)')
        ax.set_title('Convergence Rate (Log Scale)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1e-9, color='r', linestyle='--', alpha=0.5, label='1 ps')
        ax.axhline(y=1e-12, color='g', linestyle='--', alpha=0.5, label='1 fs')
        ax.legend(loc='best', ncol=2)

        # Plot 3: Initial vs Final distribution
        ax = axes[1, 0]

        # Initial distribution
        initial_offsets = [self.relative_clock_history[0][i] for i in range(self.n_nodes)]
        final_offsets = [self.relative_clock_history[-1][i] for i in range(self.n_nodes)]

        x = np.arange(self.n_nodes)
        width = 0.35

        bars1 = ax.bar(x - width/2, initial_offsets, width, label='Initial', alpha=0.7, color='red')
        bars2 = ax.bar(x + width/2, final_offsets, width, label='Final', alpha=0.7, color='green')

        ax.set_xlabel('Node ID')
        ax.set_ylabel('Relative Clock Offset (ns)')
        ax.set_title('Initial vs Final Clock Offsets')
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{i}" if i < self.n_anchors else f"N{i}" for i in range(self.n_nodes)])
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Clock spread over time
        ax = axes[1, 1]

        clock_spreads = []
        clock_stds = []
        for t in iterations:
            clocks_t = [self.clock_history[i][t] if t < len(self.clock_history[i]) else self.clock_history[i][-1]
                       for i in range(self.n_nodes)]
            spread = max(clocks_t) - min(clocks_t)
            std = np.std(clocks_t)
            clock_spreads.append(spread)
            clock_stds.append(std)

        ax.semilogy(clock_spreads, 'b-', linewidth=2, label='Range (max-min)')
        ax.semilogy(clock_stds, 'g--', linewidth=2, label='Std Dev')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Clock Spread (ns)')
        ax.set_title('Clock Synchronization Metrics')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle('Relative Time Offset Analysis - FTL Consensus', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('relative_time_offset_convergence.png', dpi=150)
        plt.show()

        # Create a focused plot just showing the classic convergence view
        fig2, ax = plt.subplots(figsize=(10, 6))

        for i in range(self.n_nodes):
            offsets = [self.relative_clock_history[t][i] for t in iterations]
            if i < self.n_anchors:
                ax.plot(offsets, label=f"Anchor {i}", linestyle='-', color=colors[i],
                       linewidth=2.5, alpha=0.8)
            else:
                ax.plot(offsets, label=f"Node {i}", linestyle='--', color=colors[i],
                       linewidth=2, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        ax.fill_between(iterations, -1, 1, color='green', alpha=0.1, label='±1 ns zone')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Relative Clock Offset (ns)', fontsize=12)
        ax.set_title('Distributed Clock Synchronization - Relative Time Offsets', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=2, fontsize=10)
        ax.set_xlim([0, len(iterations)-1])

        # Add annotation
        ax.annotate(f'Initial spread: {max(initial_offsets)-min(initial_offsets):.1f} ns',
                   xy=(1, max(initial_offsets)), xytext=(5, max(initial_offsets)+5),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                   fontsize=10, color='red')

        final_spread = max(final_offsets) - min(final_offsets)
        ax.annotate(f'Final spread: {final_spread:.2e} ns',
                   xy=(len(iterations)-1, 0), xytext=(len(iterations)-20, -10),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                   fontsize=10, color='green')

        plt.tight_layout()
        plt.savefig('relative_time_offset_classic.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("FTL Relative Time Offset Convergence")
    print("8 Nodes: 3 Anchors + 5 Unknowns")
    print("Initial clock biases: N(0, ±20ns) for unknowns")
    print("="*60)
    print()

    # Run with larger initial clock spread
    ftl = FTLWithRelativeTime(n_nodes=8, n_anchors=3, initial_clock_std=20.0)
    ftl.run(max_iters=80)

    print("\n" + "="*60)
    print("Generating plots...")
    ftl.plot_relative_time_convergence()

    print("\nPlots saved:")
    print("  - relative_time_offset_convergence.png (4-panel analysis)")
    print("  - relative_time_offset_classic.png (classic view)")
    print("="*60)