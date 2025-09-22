#!/usr/bin/env python3
"""Test consensus on progressively harder problems"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

def test_network(n_nodes, comm_range, area_size):
    """Test consensus on a network"""
    np.random.seed(42)

    # Place anchors at corners
    n_anchors = 4
    anchor_positions = np.array([
        [0, 0],
        [area_size, 0],
        [area_size, area_size],
        [0, area_size]
    ])

    # Random unknowns
    unknown_positions = np.random.uniform(0, area_size, (n_nodes - n_anchors, 2))
    true_positions = np.vstack([anchor_positions, unknown_positions])

    # Create network
    config = ConsensusGNConfig(
        max_iterations=200,
        consensus_gain=0.05,  # Very low gain
        step_size=0.1,  # Very small steps
        gradient_tol=1e-5,
        verbose=False
    )
    cgn = ConsensusGaussNewton(config)

    # Add nodes
    for i in range(n_nodes):
        if i < n_anchors:
            state = np.zeros(5)
            state[:2] = true_positions[i]
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Start at center
            state = np.zeros(5)
            state[:2] = np.array([area_size/2, area_size/2])
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    n_edges = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)
                cgn.add_measurement(ToAFactorMeters(i, j, dist + np.random.normal(0, 0.1), 0.01))
                n_edges += 1

    # Check connectivity
    direct_anchor_connections = 0
    for i in range(n_anchors, n_nodes):
        for j in range(n_anchors):
            if (i, j) in cgn.edges or (j, i) in cgn.edges:
                direct_anchor_connections += 1
                break

    # Set true positions
    true_pos_dict = {}
    for i in range(n_anchors, n_nodes):
        true_pos_dict[i] = true_positions[i]
    cgn.set_true_positions(true_pos_dict)

    # Optimize
    results = cgn.optimize()

    rmse = results.get('position_errors', {}).get('rmse', float('inf'))

    print(f"Nodes: {n_nodes}, Range: {comm_range}m, Area: {area_size}x{area_size}m")
    print(f"  Edges: {n_edges}, Direct anchor: {direct_anchor_connections}/{n_nodes-n_anchors}")
    print(f"  Converged: {results['converged']}, Iterations: {results['iterations']}")
    print(f"  RMSE: {rmse*100:.1f}cm")
    print()

    return rmse

# Test progressively harder networks
print("CONSENSUS CONVERGENCE TESTING")
print("=" * 50)

# Easy: Small network, good connectivity
test_network(10, 25, 30)

# Medium: More nodes
test_network(15, 20, 40)

# Hard: Original 30 nodes
test_network(30, 15, 50)

# Try with better connectivity
print("With better connectivity (20m range):")
test_network(30, 20, 50)