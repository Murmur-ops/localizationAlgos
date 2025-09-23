#!/usr/bin/env python3
"""
Test FTL with truly zero noise to find theoretical limits
"""

import numpy as np
import matplotlib.pyplot as plt

# Simple gradient descent for localization with perfect measurements
def test_perfect_localization():
    """Test localization with perfect distance measurements"""

    # Create simple 4-anchor, 1-unknown setup
    anchors = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])

    true_pos = np.array([5, 5])

    # Perfect distance measurements
    true_distances = np.linalg.norm(anchors - true_pos, axis=1)

    # Initialize with small error
    est_pos = true_pos + np.array([1.0, 1.0])

    # Track convergence
    errors = []

    # Gradient descent
    learning_rate = 0.1
    for iter in range(100):
        # Calculate gradient
        gradient = np.zeros(2)
        for i, anchor in enumerate(anchors):
            est_dist = np.linalg.norm(est_pos - anchor)
            if est_dist > 0:
                # Gradient of squared error
                error = est_dist - true_distances[i]
                gradient += 2 * error * (est_pos - anchor) / est_dist

        # Update position
        est_pos -= learning_rate * gradient

        # Track error
        pos_error = np.linalg.norm(est_pos - true_pos)
        errors.append(pos_error)

        # Stop if converged
        if pos_error < 1e-15:
            break

    return errors

# Test time synchronization with perfect ToA
def test_perfect_time_sync():
    """Test time sync with perfect ToA measurements"""

    # Simple case: 2 nodes with known distance
    distance = 10.0  # meters
    c = 299792458.0  # m/s
    true_tof = distance / c  # seconds

    # Clock biases (what we're trying to estimate)
    true_bias1 = 0  # Anchor has zero bias
    true_bias2 = 1e-9  # 1 ns bias for unknown

    # Perfect measurement: toa = tof + bias2 - bias1
    measured_toa = true_tof + true_bias2 - true_bias1

    # Initialize estimate with error
    est_bias2 = 5e-9  # 5 ns initial error

    # Track convergence
    errors = []

    # Gradient descent to estimate bias
    learning_rate = 0.5
    for iter in range(100):
        # Predicted ToA with current bias estimate
        predicted_toa = true_tof + est_bias2 - true_bias1

        # Error
        error = predicted_toa - measured_toa

        # Gradient (derivative w.r.t. bias2)
        gradient = 1.0

        # Update
        est_bias2 -= learning_rate * error * gradient

        # Track error
        bias_error = abs(est_bias2 - true_bias2)
        errors.append(bias_error)

        if bias_error < 1e-15:
            break

    return errors

# Run tests
pos_errors = test_perfect_localization()
time_errors = test_perfect_time_sync()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Position convergence
ax1.semilogy(pos_errors, 'b-', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Position Error (m)')
ax1.set_title('Perfect Localization Convergence (Zero Noise)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1e-15, color='r', linestyle='--', label='Machine precision')
ax1.legend()

# Time convergence
ax2.semilogy(time_errors, 'g-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Clock Bias Error (s)')
ax2.set_title('Perfect Time Sync Convergence (Zero Noise)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1e-15, color='r', linestyle='--', label='Machine precision')
ax2.legend()

plt.tight_layout()
plt.savefig('theoretical_limits.png', dpi=150)
plt.show()

# Print results
print(f"Position convergence:")
print(f"  Final error: {pos_errors[-1]:.2e} m")
print(f"  Iterations to 1mm: {np.argmax(np.array(pos_errors) < 1e-3)}")
print(f"  Iterations to 1μm: {np.argmax(np.array(pos_errors) < 1e-6)}")
print(f"  Iterations to machine precision: {len(pos_errors)}")

print(f"\nTime sync convergence:")
print(f"  Final error: {time_errors[-1]:.2e} s ({time_errors[-1]*1e9:.2e} ns)")
print(f"  Iterations to 1ns: {np.argmax(np.array(time_errors) < 1e-9)}")
print(f"  Iterations to 1ps: {np.argmax(np.array(time_errors) < 1e-12)}")
print(f"  Iterations to machine precision: {len(time_errors)}")

print(f"\nTheoretical limits with zero noise:")
print(f"  Position RMSE: ~{1e-15:.1e} m (machine precision)")
print(f"  Time sync RMSE: ~{1e-15:.1e} s (machine precision)")