#!/usr/bin/env python3
"""
Analyze why some nodes have flat convergence (not updating).
"""

import numpy as np
import time
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    generate_all_measurements
)
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters


def analyze_node_updates():
    """Analyze which nodes update and which don't"""

    # Load config
    with open('configs/unified_ideal.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate small network for analysis
    np.random.seed(42)
    true_positions, n_anchors, n_total = generate_network_topology(config)

    # Only use first 10 nodes for detailed analysis
    n_total = min(10, n_total)

    clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)
    # clock_states is a dict, not a list
    measurements = generate_all_measurements(true_positions[:n_total], clock_states,
                                           config['rf_simulation'])

    # Setup consensus
    cgn_config = ConsensusGNConfig(step_size=0.5, verbose=False)
    cgn = ConsensusGaussNewton(cgn_config)

    # Add nodes
    for i in range(n_total):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = true_positions[i]
            state[2] = clock_states[i].bias * 1e9
            cgn.add_node(i, state, is_anchor=True)
        else:
            state[:2] = true_positions[i] + np.random.normal(0, 0.1, 2)
            state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 0.1)
            state[4] = 0.5  # Initial CFO error
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    for (i, j) in measurements.keys():
        if i < n_total and j < n_total:
            cgn.add_edge(i, j)

    for (i, j), meas_list in measurements.items():
        if i < n_total and j < n_total:
            for meas in meas_list:
                factor = ToAFactorMeters(i, j, meas['range_m'], 0.01**2)
                cgn.add_measurement(factor)

    print("=" * 70)
    print("NODE UPDATE ANALYSIS")
    print("=" * 70)

    # Track initial states
    initial_states = {}
    for i in range(n_anchors, n_total):
        initial_states[i] = cgn.nodes[i].state.copy()

    # Run 10 iterations
    for iteration in range(10):
        # Share states
        current_time = time.time()
        for node_id, node in cgn.nodes.items():
            for edge in cgn.edges:
                if edge[0] == node_id and edge[1] in cgn.nodes:
                    neighbor_id = edge[1]
                    msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                     iteration, current_time)
                    node.receive_state(msg)
                elif edge[1] == node_id and edge[0] in cgn.nodes:
                    neighbor_id = edge[0]
                    msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                     iteration, current_time)
                    node.receive_state(msg)

        # Update each unknown node
        for node_id in range(n_anchors, n_total):
            node = cgn.nodes[node_id]
            H, g = node.build_local_system()

            if np.sum(np.abs(H)) > 0:
                try:
                    delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                    node.state += cgn_config.step_size * delta
                except:
                    pass

    # Analyze final states
    print(f"\n{'Node':<6} {'Neighbors':<10} {'Factors':<8} {'Pos Change':<12} {'Bias Change':<12} {'CFO Change':<12}")
    print("-" * 70)

    flat_nodes = []
    updating_nodes = []

    for i in range(n_anchors, n_total):
        node = cgn.nodes[i]

        # Count neighbors
        n_neighbors = len(node.neighbors)

        # Count factors
        n_factors = len(node.local_factors)

        # Calculate changes
        pos_change = np.linalg.norm(node.state[:2] - initial_states[i][:2])
        bias_change = abs(node.state[2] - initial_states[i][2])
        cfo_change = abs(node.state[4] - initial_states[i][4])

        print(f"{i:<6} {n_neighbors:<10} {n_factors:<8} {pos_change:<12.4f} {bias_change:<12.4f} {cfo_change:<12.6f}")

        # Categorize nodes
        if pos_change < 0.001:  # Less than 1mm movement
            flat_nodes.append(i)
        else:
            updating_nodes.append(i)

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\n✓ Updating nodes ({len(updating_nodes)}): {updating_nodes}")
    print(f"✗ Flat nodes ({len(flat_nodes)}): {flat_nodes}")

    # Analyze flat nodes
    if flat_nodes:
        print("\nInvestigating flat nodes:")
        for node_id in flat_nodes[:3]:  # Check first 3
            node = cgn.nodes[node_id]
            print(f"\nNode {node_id}:")
            print(f"  Neighbors: {list(node.neighbors.keys())}")
            print(f"  Neighbor states available: {list(node.neighbor_states.keys())}")

            # Check H matrix
            H, g = node.build_local_system()
            print(f"  H matrix sum: {np.sum(np.abs(H)):.1f}")

            if np.sum(np.abs(H)) == 0:
                print("  ❌ H matrix is zero - cannot update!")
                print("     Possible causes:")
                print("     - No measurements involving this node")
                print("     - Neighbor states not available")
                print("     - Isolated node")

                # Check factors
                print(f"  Local factors: {len(node.local_factors)}")
                if len(node.local_factors) > 0:
                    for factor in node.local_factors[:2]:
                        if isinstance(factor, ToAFactorMeters):
                            print(f"    Factor: ToA({factor.i}, {factor.j})")
            else:
                eigvals = np.linalg.eigvalsh(H)
                print(f"  H eigenvalues: min={np.min(eigvals):.1e}, max={np.max(eigvals):.1e}")
                if np.min(eigvals) < 1e-10:
                    print("  ⚠ H matrix is singular - update may fail")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("\n1. CFO (frequency offset) NEVER updates for ANY node")
    print("   - This is correct: ToA measurements don't constrain frequency")
    print("   - Would need phase/Doppler measurements")

    print("\n2. Some nodes may not update due to:")
    print("   - Poor connectivity (few neighbors)")
    print("   - Singular H matrix")
    print("   - Numerical issues")

    # Check measurement distribution
    print("\n3. Measurement distribution:")
    node_measurement_count = {i: 0 for i in range(n_anchors, n_total)}
    for (i, j), meas_list in measurements.items():
        if i >= n_anchors and i < n_total:
            node_measurement_count[i] += len(meas_list)
        if j >= n_anchors and j < n_total:
            node_measurement_count[j] += len(meas_list)

    avg_measurements = np.mean(list(node_measurement_count.values()))
    print(f"   Average measurements per unknown node: {avg_measurements:.1f}")

    # Find nodes with few measurements
    for node_id, count in node_measurement_count.items():
        if count < avg_measurements * 0.5:
            print(f"   Node {node_id}: Only {count} measurements (below average)")


if __name__ == "__main__":
    analyze_node_updates()