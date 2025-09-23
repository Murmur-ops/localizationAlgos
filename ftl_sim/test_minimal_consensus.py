#!/usr/bin/env python3
"""
Minimal test of consensus algorithm with perfect measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

def test_minimal_consensus():
    """Test consensus with 3 nodes: 1 anchor, 2 unknowns"""

    # Simple triangular setup
    positions = np.array([
        [0.0, 0.0],   # Anchor at origin
        [10.0, 0.0],  # Unknown 1
        [5.0, 8.66]   # Unknown 2 (equilateral triangle)
    ])

    n_nodes = 3
    n_anchors = 1

    # Perfect distance measurements
    distances = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i,j] = np.linalg.norm(positions[i] - positions[j])

    print(f"True distances:")
    print(distances)

    # Create consensus system
    config = ConsensusGNConfig(
        max_iterations=100,
        gradient_tol=1e-10,
        step_size=0.5,
        consensus_gain=0.01,  # Small consensus gain
        verbose=True
    )

    consensus = ConsensusGaussNewton(config)

    # Initialize nodes
    for i in range(n_nodes):
        is_anchor = i < n_anchors
        if is_anchor:
            # Anchor with perfect state
            state = np.array([positions[i, 0], positions[i, 1], 0, 0, 0])
        else:
            # Unknown with initial error
            state = np.zeros(5)
            state[:2] = positions[i] + np.random.normal(0, 2.0, 2)  # 2m error
            state[2] = np.random.normal(0, 1e-9)  # Small clock bias

        consensus.add_node(i, state, is_anchor=is_anchor)
        print(f"Node {i}: anchor={is_anchor}, initial pos=[{state[0]:.2f}, {state[1]:.2f}]")

    # Add edges (full connectivity)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            consensus.add_edge(i, j)

    # Add perfect measurements
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Perfect range measurement
            range_m = distances[i, j]
            range_var = 1e-6  # Very small variance (1mm std)

            factor = ToAFactorMeters(
                i=i,
                j=j,
                range_meas_m=range_m,
                range_var_m2=range_var
            )
            consensus.add_measurement(factor)

    # Set true positions for error tracking
    true_pos_dict = {i: positions[i] for i in range(n_nodes)}
    consensus.set_true_positions(true_pos_dict)

    # Run optimization
    results = consensus.optimize()

    # Print results
    print(f"\n=== RESULTS ===")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")

    if 'position_errors' in results and results['position_errors']:
        print(f"Position RMSE: {results['position_errors']['rmse']*100:.3f} cm")
        print(f"Max error: {results['position_errors']['max']*100:.3f} cm")

    # Get final states
    final_states = results['final_states']
    for i in range(n_nodes):
        state = final_states[i]
        true_pos = positions[i]
        error = np.linalg.norm(state[:2] - true_pos)
        print(f"Node {i}: pos=[{state[0]:.3f}, {state[1]:.3f}], error={error*100:.3f}cm")

    # Plot convergence history
    if results['convergence_history']:
        iterations = [h['iteration'] for h in results['convergence_history']]
        max_grads = [h['max_gradient_norm'] for h in results['convergence_history']]

        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, max_grads, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Max Gradient Norm')
        plt.title('Convergence History')
        plt.grid(True, alpha=0.3)
        plt.savefig('minimal_consensus_convergence.png')
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    test_minimal_consensus()