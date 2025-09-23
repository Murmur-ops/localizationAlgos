"""
Stable 30-node frequency synchronization demonstration
Using proper scaling to avoid numerical issues
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def run_stable_30node_demo():
    """Run stable 30-node demo with proper numerical scaling"""

    # Configuration
    n_nodes = 35  # 5 anchors + 30 unknowns
    n_anchors = 5
    area_size = 50.0
    n_iterations = 200

    print("="*70)
    print("30-NODE FREQUENCY SYNCHRONIZATION (STABLE VERSION)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Nodes: {n_nodes} (5 anchors + 30 unknowns)")
    print(f"  Area: {area_size}×{area_size} m")
    print(f"  Iterations: {n_iterations}")

    # Set random seed
    np.random.seed(42)

    # Generate true positions
    true_pos = np.zeros((n_nodes, 2))

    # Place anchors
    true_pos[0] = [5, 5]
    true_pos[1] = [45, 5]
    true_pos[2] = [45, 45]
    true_pos[3] = [5, 45]
    true_pos[4] = [25, 25]

    # Place unknowns in a grid pattern
    idx = n_anchors
    for i in range(6):
        for j in range(5):
            if idx < n_nodes:
                true_pos[idx] = [
                    7 + i * 7 + np.random.randn() * 1,
                    7 + j * 9 + np.random.randn() * 1
                ]
                idx += 1

    # True clock parameters (scaled to reasonable values)
    true_time = np.random.randn(n_nodes) * 10  # ±10 units (representing ns)
    true_freq = np.random.randn(n_nodes) * 2   # ±2 units (representing ppb)

    # Anchors are reference (zero offsets)
    true_time[:n_anchors] = 0
    true_freq[:n_anchors] = 0

    # Initialize estimates
    est_pos = true_pos.copy()
    est_time = np.zeros(n_nodes)
    est_freq = np.zeros(n_nodes)

    # Add initial position error to unknowns
    for i in range(n_anchors, n_nodes):
        est_pos[i] += np.random.randn(2) * 3  # 3m initial error

    # History storage
    pos_rmse_hist = []
    time_std_hist = []
    freq_std_hist = []

    # Individual node tracking (first 5 unknowns)
    track_nodes = list(range(n_anchors, min(n_anchors+5, n_nodes)))
    individual_time = {i: [] for i in track_nodes}
    individual_freq = {i: [] for i in track_nodes}

    print("\nRunning optimization...")

    # Main optimization loop
    for iter in range(n_iterations):
        # Current time
        t = iter * 0.1  # Scaled time

        # Compute current RMSE
        errors = []
        for i in range(n_anchors, n_nodes):
            err = np.linalg.norm(est_pos[i] - true_pos[i])
            errors.append(err)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        pos_rmse_hist.append(rmse)

        # Time and frequency statistics
        time_std_hist.append(np.std(est_time[n_anchors:]))
        freq_std_hist.append(np.std(est_freq[n_anchors:]))

        # Store individual histories
        for i in track_nodes:
            individual_time[i].append(est_time[i])
            individual_freq[i].append(est_freq[i])

        # Update each unknown node
        for i in range(n_anchors, n_nodes):
            # Gradient and Hessian for node i
            grad_pos = np.zeros(2)
            grad_time = 0
            grad_freq = 0

            H_pos = np.zeros((2, 2))
            H_time = 0
            H_freq = 0
            H_cross = np.zeros(2)  # Cross terms

            # Measurements to neighbors
            n_measurements = 0
            for j in range(n_nodes):
                if i == j:
                    continue

                # True distance
                true_dist = np.linalg.norm(true_pos[j] - true_pos[i])

                # Skip if too far
                if true_dist > 40:
                    continue

                # Simulated measurement (with frequency drift)
                drift = (true_freq[j] - true_freq[i]) * t
                meas = true_dist + (true_time[j] - true_time[i] + drift) * 0.3  # Scaled c
                meas += np.random.randn() * 0.01  # Noise

                # Predicted measurement
                est_dist = np.linalg.norm(est_pos[j] - est_pos[i])
                est_drift = (est_freq[j] - est_freq[i]) * t
                pred = est_dist + (est_time[j] - est_time[i] + est_drift) * 0.3

                # Residual
                res = pred - meas

                # Direction vector
                if est_dist > 1e-6:
                    dir = (est_pos[j] - est_pos[i]) / est_dist
                else:
                    dir = np.zeros(2)

                # Accumulate gradients
                grad_pos += -dir * res
                grad_time += -0.3 * res  # Scaled gradient
                grad_freq += -0.3 * t * res  # Scaled gradient

                # Accumulate Hessian (approximation)
                H_pos += np.outer(dir, dir)
                H_time += 0.3 * 0.3
                H_freq += 0.3 * 0.3 * t * t
                H_cross += -dir * 0.3

                n_measurements += 1

            if n_measurements == 0:
                continue

            # Average by number of measurements
            weight = 1.0 / max(1, n_measurements)
            grad_pos *= weight
            grad_time *= weight
            grad_freq *= weight
            H_pos *= weight
            H_time *= weight
            H_freq *= weight

            # Add regularization
            reg = 0.01
            H_pos += np.eye(2) * reg
            H_time += reg
            H_freq += reg * 0.1  # Less regularization for frequency

            # Compute updates (simple gradient descent)
            step_size = 0.1 / (1 + iter * 0.005)

            # Position update
            if np.linalg.det(H_pos) > 1e-10:
                delta_pos = -np.linalg.solve(H_pos, grad_pos)
                est_pos[i] += step_size * delta_pos

            # Time update
            if abs(H_time) > 1e-10:
                delta_time = -grad_time / H_time
                est_time[i] += step_size * delta_time

            # Frequency update
            if abs(H_freq) > 1e-10:
                delta_freq = -grad_freq / H_freq
                delta_freq = np.clip(delta_freq, -0.5, 0.5)  # Limit update
                est_freq[i] += step_size * 0.1 * delta_freq  # Smaller step for frequency

        # Print progress
        if iter % 40 == 0:
            print(f"  Iter {iter:3d}: RMSE={rmse:.4f}m, "
                  f"Time STD={time_std_hist[-1]:.2f}, "
                  f"Freq STD={freq_std_hist[-1]:.3f}")

    print("\nOptimization complete!")

    # Final statistics
    final_rmse = pos_rmse_hist[-1]
    final_time_std = time_std_hist[-1]
    final_freq_std = freq_std_hist[-1]

    print(f"\nFinal Results:")
    print(f"  Position RMSE: {final_rmse:.4f} m")
    print(f"  Time sync STD: {final_time_std:.3f} units")
    print(f"  Frequency STD: {final_freq_std:.3f} units")

    # Create visualization
    fig = plt.figure(figsize=(16, 10))

    # 1. Position RMSE convergence
    ax1 = plt.subplot(2, 4, 1)
    ax1.semilogy(pos_rmse_hist, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position RMSE (m)')
    ax1.set_title('Position Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.5)

    # 2. Time offset convergence
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(time_std_hist, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Time Offset STD')
    ax2.set_title('Time Synchronization')
    ax2.grid(True, alpha=0.3)

    # 3. Frequency convergence
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(freq_std_hist, 'purple', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Frequency STD')
    ax3.set_title('Frequency Synchronization')
    ax3.grid(True, alpha=0.3)

    # 4. Individual time offsets
    ax4 = plt.subplot(2, 4, 4)
    for node_id in track_nodes:
        ax4.plot(individual_time[node_id], label=f'Node {node_id}', alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time Offset')
    ax4.set_title('Individual Node Time Offsets')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Individual frequency offsets
    ax5 = plt.subplot(2, 4, 5)
    for node_id in track_nodes:
        ax5.plot(individual_freq[node_id], label=f'Node {node_id}', alpha=0.7)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Frequency Offset')
    ax5.set_title('Individual Node Frequencies')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Frequency histogram
    ax6 = plt.subplot(2, 4, 6)
    final_freqs = est_freq[n_anchors:]
    if not np.any(np.isnan(final_freqs)):
        ax6.hist(final_freqs, bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax6.set_xlabel('Frequency Offset')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Final Frequency Distribution')
    ax6.grid(True, alpha=0.3)

    # 7-8. Position comparison (large subplot)
    ax7 = plt.subplot(2, 4, (7, 8))

    # Plot anchors
    ax7.scatter(true_pos[:n_anchors, 0], true_pos[:n_anchors, 1],
               marker='^', s=200, c='red', label='Anchors',
               edgecolors='black', linewidth=2, zorder=5)

    # Plot true positions
    ax7.scatter(true_pos[n_anchors:, 0], true_pos[n_anchors:, 1],
               marker='o', s=80, c='blue', label='True Position',
               alpha=0.6, zorder=3)

    # Plot estimates
    ax7.scatter(est_pos[n_anchors:, 0], est_pos[n_anchors:, 1],
               marker='x', s=60, c='green', label='Estimate',
               linewidth=2, zorder=4)

    # Draw error lines
    for i in range(n_anchors, n_nodes):
        ax7.plot([true_pos[i, 0], est_pos[i, 0]],
                [true_pos[i, 1], est_pos[i, 1]],
                'gray', alpha=0.5, linewidth=0.5)

    ax7.set_xlabel('X Position (m)')
    ax7.set_ylabel('Y Position (m)')
    ax7.set_title(f'Position Estimates (RMSE={final_rmse:.3f}m)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, area_size])
    ax7.set_ylim([0, area_size])
    ax7.set_aspect('equal')

    # Add text box with statistics
    stats_text = f"Network: 30 unknown + 5 anchor nodes\n"
    stats_text += f"Area: {area_size}×{area_size} m\n"
    stats_text += f"Final RMSE: {final_rmse:.3f} m\n"
    stats_text += f"Max error: {max(errors):.3f} m\n"
    stats_text += f"Min error: {min(errors):.3f} m"

    ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('30-Node Frequency Synchronization: Complete Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig('30node_frequency_stable.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlot saved to '30node_frequency_stable.png'")

    # Print detailed error analysis
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    print("\nPosition errors for first 10 unknown nodes:")
    for i in range(n_anchors, min(n_anchors+10, n_nodes)):
        err = np.linalg.norm(est_pos[i] - true_pos[i])
        print(f"  Node {i:2d}: {err:.4f} m")

    print(f"\nTime offset estimates (first 5 unknowns):")
    for i in range(n_anchors, min(n_anchors+5, n_nodes)):
        print(f"  Node {i}: True={true_time[i]:.2f}, Est={est_time[i]:.2f}, "
              f"Error={abs(true_time[i]-est_time[i]):.2f}")

    print(f"\nFrequency offset estimates (first 5 unknowns):")
    for i in range(n_anchors, min(n_anchors+5, n_nodes)):
        print(f"  Node {i}: True={true_freq[i]:.3f}, Est={est_freq[i]:.3f}, "
              f"Error={abs(true_freq[i]-est_freq[i]):.3f}")

    return {
        'pos_rmse': pos_rmse_hist,
        'time_std': time_std_hist,
        'freq_std': freq_std_hist,
        'final_rmse': final_rmse
    }


if __name__ == "__main__":
    results = run_stable_30node_demo()