#!/usr/bin/env python3
"""
Centralized baseline to establish what zero-noise convergence should look like
"""

import numpy as np
import matplotlib.pyplot as plt

def centralized_gn_localization():
    """
    Centralized Gauss-Newton for localization with perfect measurements
    This establishes the performance baseline
    """

    # Setup: 4 anchors, 6 unknowns
    n_anchors = 4
    n_unknowns = 6
    area_size = 20.0

    # Anchor positions (corners)
    anchors = np.array([
        [0, 0],
        [area_size, 0],
        [area_size, area_size],
        [0, area_size]
    ])

    # True unknown positions (grid)
    true_unknowns = []
    spacing = area_size / 3
    for i in range(2):
        for j in range(3):
            true_unknowns.append([spacing * (i+1), spacing * (j+1)])
    true_unknowns = np.array(true_unknowns[:n_unknowns])

    # All positions
    all_true = np.vstack([anchors, true_unknowns])

    # Perfect distance measurements
    measurements = []
    for i in range(n_anchors + n_unknowns):
        for j in range(i+1, n_anchors + n_unknowns):
            dist = np.linalg.norm(all_true[i] - all_true[j])
            measurements.append({'i': i, 'j': j, 'dist': dist})

    print(f"Setup: {n_anchors} anchors, {n_unknowns} unknowns")
    print(f"Measurements: {len(measurements)}")

    # Initialize unknowns with error
    est_unknowns = true_unknowns + np.random.normal(0, 2.0, true_unknowns.shape)

    # Track convergence
    rmse_history = []

    # Gauss-Newton iterations
    for iter in range(100):
        # Build system
        n_vars = 2 * n_unknowns  # x,y for each unknown
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        for meas in measurements:
            i, j = meas['i'], meas['j']
            true_dist = meas['dist']

            # Skip if both are anchors
            if i < n_anchors and j < n_anchors:
                continue

            # Get positions
            if i < n_anchors:
                pi = anchors[i]
                pj = est_unknowns[j - n_anchors]
                idx_j = 2 * (j - n_anchors)
            elif j < n_anchors:
                pi = est_unknowns[i - n_anchors]
                pj = anchors[j]
                idx_i = 2 * (i - n_anchors)
            else:
                pi = est_unknowns[i - n_anchors]
                pj = est_unknowns[j - n_anchors]
                idx_i = 2 * (i - n_anchors)
                idx_j = 2 * (j - n_anchors)

            # Compute residual
            est_dist = np.linalg.norm(pi - pj)
            if est_dist < 1e-10:
                continue

            residual = true_dist - est_dist
            u = (pj - pi) / est_dist

            # Add to normal equations
            # For Gauss-Newton: min 1/2 ||f(x)||^2
            # gradient = J^T * r
            # Hessian ≈ J^T * J

            if i >= n_anchors:
                # Jacobian for node i: ∂(dist)/∂pi = -u
                Ji = np.zeros(n_vars)
                Ji[idx_i] = -u[0]  # Note: negative because dist = ||pj - pi||
                Ji[idx_i + 1] = -u[1]
                H += np.outer(Ji, Ji)
                g -= Ji * residual  # g = -J^T * r for minimization

            if j >= n_anchors:
                # Jacobian for node j: ∂(dist)/∂pj = u
                Jj = np.zeros(n_vars)
                Jj[idx_j] = u[0]
                Jj[idx_j + 1] = u[1]
                H += np.outer(Jj, Jj)
                g -= Jj * residual  # g = -J^T * r for minimization

        # Add small damping for stability
        H += 1e-9 * np.eye(n_vars)

        # Solve H * delta = -g
        try:
            delta = np.linalg.solve(H, -g)  # Note: solve for -g
        except:
            break

        # Update with line search
        best_alpha = 1.0
        current_error = np.linalg.norm(est_unknowns - true_unknowns)

        for alpha in [1.0, 0.5, 0.1, 0.01]:
            test_unknowns = est_unknowns.copy()
            for i in range(n_unknowns):
                test_unknowns[i] += alpha * delta[2*i:2*i+2]

            test_error = np.linalg.norm(test_unknowns - true_unknowns)
            if test_error < current_error:
                best_alpha = alpha
                current_error = test_error
                break

        # Apply update
        for i in range(n_unknowns):
            est_unknowns[i] += best_alpha * delta[2*i:2*i+2]

        # Compute RMSE
        errors = np.linalg.norm(est_unknowns - true_unknowns, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        rmse_history.append(rmse)

        if rmse < 1e-12:
            print(f"Converged at iteration {iter}")
            break

    return rmse_history


# Run centralized baseline
np.random.seed(42)
centralized_rmse = centralized_gn_localization()

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(centralized_rmse, 'b-', linewidth=2, label='Centralized GN')
plt.xlabel('Iteration')
plt.ylabel('Position RMSE (m)')
plt.title('Centralized Baseline - Zero Noise Convergence')
plt.grid(True, alpha=0.3)
plt.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='1mm')
plt.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='1μm')
plt.axhline(y=1e-9, color='m', linestyle='--', alpha=0.5, label='1nm')
plt.axhline(y=1e-12, color='c', linestyle='--', alpha=0.5, label='1pm')
plt.legend()
plt.savefig('centralized_baseline.png')
plt.show()

print(f"\n=== Centralized Baseline Results ===")
print(f"Final RMSE: {centralized_rmse[-1]:.3e} m")

if len(centralized_rmse) > 1:
    print(f"Iterations to 1mm: {np.argmax(np.array(centralized_rmse) < 1e-3) if min(centralized_rmse) < 1e-3 else 'N/A'}")
    print(f"Iterations to 1μm: {np.argmax(np.array(centralized_rmse) < 1e-6) if min(centralized_rmse) < 1e-6 else 'N/A'}")
    print(f"Iterations to 1nm: {np.argmax(np.array(centralized_rmse) < 1e-9) if min(centralized_rmse) < 1e-9 else 'N/A'}")

    # Check convergence rate
    if len(centralized_rmse) > 20:
        log_errors = np.log10(centralized_rmse[:20])
        rate = (log_errors[-1] - log_errors[0]) / len(log_errors)
        print(f"Log10 convergence rate: {rate:.3f} per iteration")
        print("(Should be strongly negative for exponential convergence)")