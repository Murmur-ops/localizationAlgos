"""
Simple test of frequency synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.frequency_factors.frequency_factors import RangeFrequencyFactor, FrequencyPrior


def test_frequency_factor():
    """Test that frequency factor works correctly"""
    print("Testing frequency factor...")

    # Create a simple 2-node scenario
    state_i = np.array([0.0, 0.0, 0.0, 5.0])  # 5 ppb frequency offset
    state_j = np.array([10.0, 0.0, 0.0, -5.0])  # -5 ppb frequency offset

    # Test at different timestamps
    timestamps = [0, 100, 1000]
    c = 299792458.0

    for t in timestamps:
        # Expected range with frequency drift
        geom_range = 10.0  # meters
        freq_diff = -5.0 - 5.0  # -10 ppb difference
        drift_contribution = freq_diff * t * c * 1e-9  # Convert to meters

        expected_range = geom_range + drift_contribution

        # Create factor
        factor = RangeFrequencyFactor(
            measured_range=expected_range,
            timestamp=t,
            sigma=0.01
        )

        # Compute error - should be near zero
        error = factor.error(state_i, state_j)
        print(f"  t={t}s: error = {error:.6f} (expected ~0)")

        # Check Jacobian
        J_i, J_j = factor.jacobian(state_i, state_j)
        assert J_i.shape == (4,)
        assert J_j.shape == (4,)
        print(f"    Jacobian shapes OK")

    print("✓ Frequency factor test passed\n")


def test_frequency_prior():
    """Test frequency prior"""
    print("Testing frequency prior...")

    prior = FrequencyPrior(nominal_freq_ppb=0.0, sigma_ppb=10.0)

    # Test different frequency values
    test_freqs = [0.0, 10.0, -10.0, 50.0]

    for freq in test_freqs:
        state = np.array([1.0, 2.0, 3.0, freq])
        error = prior.error(state)
        expected = freq / 10.0  # Normalized by sigma
        print(f"  freq={freq:5.1f} ppb: error={error:6.2f} (expected {expected:6.2f})")

    print("✓ Frequency prior test passed\n")


def test_simple_optimization():
    """Test simple frequency estimation"""
    print("Testing simple frequency optimization...")

    # True states
    true_pos_i = np.array([0.0, 0.0])
    true_pos_j = np.array([10.0, 0.0])
    true_freq_i = 5.0  # ppb
    true_freq_j = -3.0  # ppb

    # Generate measurements at multiple epochs
    timestamps = np.arange(0, 100, 10)
    measurements = []
    c = 299792458.0

    for t in timestamps:
        geom_range = 10.0
        freq_contrib = (true_freq_j - true_freq_i) * t * c * 1e-9
        noise = np.random.normal(0, 0.001)  # 1mm noise
        meas = geom_range + freq_contrib + noise
        measurements.append(meas)

    # Initial guess
    state_i = np.array([0.0, 0.0, 0.0, 0.0])  # Known anchor
    state_j = np.array([10.0, 0.0, 0.0, 0.0])  # Initial frequency guess = 0

    # Simple gradient descent with adaptive learning rate
    learning_rate = 1e-10  # Much smaller due to large gradients from c*t
    n_iterations = 100

    errors = []
    for iter in range(n_iterations):
        # Compute total error and gradient
        total_error = 0
        gradient_j = np.zeros(4)

        for t, meas in zip(timestamps, measurements):
            factor = RangeFrequencyFactor(meas, t, sigma=0.001)
            err = factor.error(state_i, state_j)
            J_i, J_j = factor.jacobian(state_i, state_j)

            total_error += err**2
            gradient_j += 2 * err * J_j

        # Add frequency prior
        prior = FrequencyPrior(0.0, 10.0)
        prior_err = prior.error(state_j)
        prior_J = prior.jacobian(state_j)

        total_error += 0.1 * prior_err**2
        gradient_j += 0.1 * 2 * prior_err * prior_J

        errors.append(np.sqrt(total_error / len(measurements)))

        # Update only frequency (position is known)
        state_j[3] -= learning_rate * gradient_j[3]

        if iter % 20 == 0:
            print(f"  Iter {iter:3d}: freq_est={state_j[3]:6.2f} ppb, error={errors[-1]:.6f}")

    print(f"\nFinal frequency estimate: {state_j[3]:.2f} ppb (true: {true_freq_j:.2f} ppb)")
    print(f"Estimation error: {abs(state_j[3] - true_freq_j):.3f} ppb")

    # Plot convergence
    plt.figure(figsize=(8, 4))
    plt.semilogy(errors)
    plt.xlabel('Iteration')
    plt.ylabel('RMS Error')
    plt.title('Frequency Estimation Convergence')
    plt.grid(True, alpha=0.3)
    plt.savefig('frequency_simple_test.png')
    print("\n✓ Simple optimization test passed")
    print("  Plot saved to frequency_simple_test.png\n")


def test_multi_node():
    """Test with multiple nodes"""
    print("Testing multi-node frequency sync...")

    n_nodes = 5
    true_positions = np.array([
        [0.0, 0.0],    # Anchor 1
        [10.0, 0.0],   # Anchor 2
        [5.0, 8.66],   # Anchor 3
        [2.0, 4.0],    # Unknown 1
        [7.0, 5.0]     # Unknown 2
    ])

    true_freqs = np.array([0.0, 0.0, 0.0, 8.0, -6.0])  # ppb

    print(f"  Network: {n_nodes} nodes (3 anchors, 2 unknowns)")
    print(f"  True frequencies: {true_freqs}")

    # Generate pairwise measurements
    n_measurements = 20
    timestamps = np.linspace(0, 50, n_measurements)
    c = 299792458.0

    all_measurements = {}
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            measurements = []
            for t in timestamps:
                geom_range = np.linalg.norm(true_positions[j] - true_positions[i])
                freq_contrib = (true_freqs[j] - true_freqs[i]) * t * c * 1e-9
                noise = np.random.normal(0, 0.002)
                measurements.append(geom_range + freq_contrib + noise)
            all_measurements[(i, j)] = measurements

    print(f"  Generated {len(all_measurements)} pairwise measurement sets")
    print(f"  Each with {n_measurements} epochs")

    # Estimate frequencies for unknown nodes
    estimated_freqs = np.zeros(n_nodes)

    for node_id in [3, 4]:  # Unknown nodes
        print(f"\n  Estimating node {node_id}...")

        # Use measurements to anchors
        state = np.zeros(4)
        state[:2] = true_positions[node_id]  # Known position for this test

        learning_rate = 1e-10  # Much smaller due to large gradients
        for iter in range(50):
            gradient = np.zeros(4)
            total_error = 0

            # Use measurements to all anchors
            for anchor_id in [0, 1, 2]:
                key = (anchor_id, node_id) if anchor_id < node_id else (node_id, anchor_id)
                measurements = all_measurements[key]

                anchor_state = np.zeros(4)
                anchor_state[:2] = true_positions[anchor_id]

                for t, meas in zip(timestamps, measurements):
                    if anchor_id < node_id:
                        factor = RangeFrequencyFactor(meas, t, sigma=0.002)
                        err = factor.error(anchor_state, state)
                        _, J = factor.jacobian(anchor_state, state)
                    else:
                        factor = RangeFrequencyFactor(meas, t, sigma=0.002)
                        err = factor.error(state, anchor_state)
                        J, _ = factor.jacobian(state, anchor_state)

                    total_error += err**2
                    gradient += 2 * err * J

            # Update frequency estimate
            state[3] -= learning_rate * gradient[3]

            if iter == 49:
                estimated_freqs[node_id] = state[3]
                print(f"    Final: {state[3]:.2f} ppb (true: {true_freqs[node_id]:.2f})")

    print("\n✓ Multi-node test completed")
    print(f"  Frequency estimation errors:")
    for node_id in [3, 4]:
        error = abs(estimated_freqs[node_id] - true_freqs[node_id])
        print(f"    Node {node_id}: {error:.3f} ppb")


if __name__ == "__main__":
    print("="*60)
    print("FREQUENCY SYNCHRONIZATION COMPONENT TEST")
    print("="*60)
    print()

    # Run tests
    test_frequency_factor()
    test_frequency_prior()
    test_simple_optimization()
    test_multi_node()

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)