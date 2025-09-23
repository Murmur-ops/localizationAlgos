#!/usr/bin/env python3
"""
Test simple localization without consensus to isolate the issue
"""

import numpy as np
import matplotlib.pyplot as plt

def simple_trilateration():
    """
    Simple trilateration with gradient descent
    No consensus, no fancy features - just pure localization
    """

    # Anchors in square formation
    anchors = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])

    # True unknown position
    true_pos = np.array([5, 5])

    # Perfect distance measurements
    true_distances = np.linalg.norm(anchors - true_pos, axis=1)
    print(f"True position: {true_pos}")
    print(f"True distances: {true_distances}")

    # Start with initial guess
    est_pos = np.array([7.0, 3.0])

    errors = []
    positions = [est_pos.copy()]

    # Gradient descent
    learning_rate = 0.1

    for iter in range(100):
        # Compute gradient
        gradient = np.zeros(2)

        for i, anchor in enumerate(anchors):
            # Current estimated distance
            est_dist = np.linalg.norm(est_pos - anchor)

            if est_dist > 1e-10:
                # Error in distance
                dist_error = est_dist - true_distances[i]

                # Gradient contribution
                gradient += 2 * dist_error * (est_pos - anchor) / est_dist

        # Update position
        est_pos -= learning_rate * gradient

        # Track error
        pos_error = np.linalg.norm(est_pos - true_pos)
        errors.append(pos_error)
        positions.append(est_pos.copy())

        if pos_error < 1e-10:
            print(f"Converged at iteration {iter}")
            break

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convergence
    ax1.semilogy(errors, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Simple Trilateration Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-3, color='r', linestyle='--', label='1mm')
    ax1.axhline(y=1e-6, color='g', linestyle='--', label='1μm')
    ax1.legend()

    # Trajectory
    positions = np.array(positions)
    ax2.plot(positions[:, 0], positions[:, 1], 'b.-', alpha=0.5, label='Trajectory')
    ax2.plot(anchors[:, 0], anchors[:, 1], 'r^', markersize=10, label='Anchors')
    ax2.plot(true_pos[0], true_pos[1], 'g*', markersize=15, label='True position')
    ax2.plot(positions[0, 0], positions[0, 1], 'bo', markersize=8, label='Start')
    ax2.plot(positions[-1, 0], positions[-1, 1], 'mo', markersize=8, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Localization Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('simple_trilateration.png')
    plt.show()

    print(f"\nFinal error: {errors[-1]:.2e} m")
    print(f"Final position: {est_pos}")
    print(f"Iterations to 1mm: {np.argmax(np.array(errors) < 1e-3)}")
    print(f"Iterations to 1μm: {np.argmax(np.array(errors) < 1e-6)}")

    return errors

if __name__ == "__main__":
    errors = simple_trilateration()

    print("\n=== Analysis ===")
    print("Simple gradient descent with perfect measurements converges exponentially")
    print("to machine precision. The FTL consensus implementation should achieve")
    print("similar performance with zero noise.")

    # Theoretical limit
    print(f"\nWith zero noise, the theoretical limits are:")
    print(f"  Position RMSE: ~1e-15 m (machine precision)")
    print(f"  Time sync RMSE: ~1e-15 s (machine precision)")