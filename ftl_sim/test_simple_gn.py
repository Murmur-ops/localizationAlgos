#!/usr/bin/env python3
"""
Simple, clean Gauss-Newton implementation for verification
"""

import numpy as np
import matplotlib.pyplot as plt

def simple_gauss_newton():
    """
    Minimal Gauss-Newton for 1 unknown node with 3 anchors
    """
    # Three anchors in triangle
    anchors = np.array([
        [0, 0],
        [10, 0],
        [5, 8.66]
    ])

    # True unknown position
    true_pos = np.array([5, 3])

    # Perfect distance measurements
    true_dists = np.linalg.norm(anchors - true_pos, axis=1)
    print(f"True position: {true_pos}")
    print(f"True distances: {true_dists}")

    # Initial guess
    x = np.array([7.0, 4.0])
    print(f"Initial guess: {x}")

    errors = []

    for iter in range(50):
        # Build Jacobian and residual
        J = []
        r = []

        for i, anchor in enumerate(anchors):
            # Current estimated distance
            diff = x - anchor
            dist = np.linalg.norm(diff)

            if dist > 0:
                # Residual: predicted - measured
                # (we minimize this, so when it's zero, predicted = measured)
                residual = dist - true_dists[i]
                r.append(residual)

                # Jacobian: derivative of distance w.r.t. position
                # d(||x - a||)/dx = (x - a) / ||x - a||
                jacobian = diff / dist
                J.append(jacobian)

        J = np.array(J)
        r = np.array(r)

        # Gauss-Newton update
        # Normal equations: J^T J delta = -J^T r
        JTJ = J.T @ J
        JTr = J.T @ r

        # Add tiny damping for numerical stability
        JTJ += 1e-12 * np.eye(2)

        # Solve for step
        delta = np.linalg.solve(JTJ, -JTr)

        # Update position
        x = x + delta

        # Track error
        error = np.linalg.norm(x - true_pos)
        errors.append(error)

        print(f"Iter {iter:2d}: pos=[{x[0]:.6f}, {x[1]:.6f}], error={error:.3e}")

        if error < 1e-12:
            print(f"Converged to machine precision at iteration {iter}")
            break

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(errors, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Position Error (m)')
    plt.title('Gauss-Newton Convergence')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1e-3, color='r', linestyle='--', label='1mm')
    plt.axhline(y=1e-6, color='g', linestyle='--', label='1μm')
    plt.axhline(y=1e-9, color='m', linestyle='--', label='1nm')
    plt.axhline(y=1e-12, color='c', linestyle='--', label='1pm')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Show the geometry
    plt.plot(anchors[:, 0], anchors[:, 1], 'r^', markersize=10, label='Anchors')
    plt.plot(true_pos[0], true_pos[1], 'g*', markersize=15, label='True position')
    plt.plot(x[0], x[1], 'bo', markersize=8, label='Final estimate')

    # Draw circles
    theta = np.linspace(0, 2*np.pi, 100)
    for i, anchor in enumerate(anchors):
        circle_x = anchor[0] + true_dists[i] * np.cos(theta)
        circle_y = anchor[1] + true_dists[i] * np.sin(theta)
        plt.plot(circle_x, circle_y, 'k-', alpha=0.2, linewidth=0.5)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trilateration Geometry')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('simple_gn_convergence.png')
    # plt.show()  # Comment out for non-interactive

    return errors


if __name__ == "__main__":
    errors = simple_gauss_newton()

    print("\n=== Summary ===")
    if len(errors) > 0:
        print(f"Final error: {errors[-1]:.3e} m")
        if min(errors) < 1e-3:
            print(f"Iterations to 1mm: {np.argmax(np.array(errors) < 1e-3)}")
        if min(errors) < 1e-6:
            print(f"Iterations to 1μm: {np.argmax(np.array(errors) < 1e-6)}")
        if min(errors) < 1e-9:
            print(f"Iterations to 1nm: {np.argmax(np.array(errors) < 1e-9)}")

        # Check convergence rate
        if len(errors) > 5:
            # Linear convergence rate in log space
            log_errors = np.log10(errors[1:11])  # Use iterations 1-10
            iterations = np.arange(len(log_errors))
            if len(log_errors) > 1:
                rate = np.polyfit(iterations, log_errors, 1)[0]
                print(f"Log10 convergence rate: {rate:.3f} per iteration")
                print("(Should be ≈ -0.7 to -1.0 for quadratic convergence)")