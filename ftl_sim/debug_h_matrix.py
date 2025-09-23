#!/usr/bin/env python3
"""
Debug why H matrices might be singular or invalid, preventing position updates.
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


def debug_h_matrix():
    """Debug H matrix construction and updates"""

    # Load config
    with open('configs/unified_ideal.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Small network
    np.random.seed(42)
    true_positions, n_anchors, n_total = generate_network_topology(config)
    n_total = min(10, n_total)

    clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)
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
            state[:2] = true_positions[i] + np.random.normal(0, 0.5, 2)  # 50cm error
            state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 1.0)
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
    print("H MATRIX ANALYSIS BEFORE STATE SHARING")
    print("=" * 70)

    # Check H matrices WITHOUT sharing states first
    for i in range(n_anchors, min(n_total, n_anchors+3)):
        node = cgn.nodes[i]
        H, g = node.build_local_system()

        print(f"\nNode {i} (BEFORE state sharing):")
        print(f"  Neighbors in memory: {list(node.neighbor_states.keys())}")
        print(f"  Local factors: {len(node.local_factors)}")
        print(f"  H matrix properties:")
        print(f"    Sum of |H|: {np.sum(np.abs(H)):.1f}")

        if np.sum(np.abs(H)) > 0:
            eigvals = np.linalg.eigvalsh(H)
            print(f"    Eigenvalues: min={eigvals.min():.1e}, max={eigvals.max():.1e}")
            print(f"    Condition number: {eigvals.max()/max(eigvals.min(), 1e-15):.1e}")
            print(f"    H diagonal: {np.diag(H)}")
        else:
            print("    ❌ H is all zeros! Cannot update.")

    print("\n" + "=" * 70)
    print("SHARING STATES WITH PROPER TIMESTAMPS")
    print("=" * 70)

    # Share states properly
    current_time = time.time()
    for node_id, node in cgn.nodes.items():
        for edge in cgn.edges:
            if edge[0] == node_id and edge[1] in cgn.nodes:
                neighbor_id = edge[1]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 0, current_time)
                node.receive_state(msg)
            elif edge[1] == node_id and edge[0] in cgn.nodes:
                neighbor_id = edge[0]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 0, current_time)
                node.receive_state(msg)

    print("\n" + "=" * 70)
    print("H MATRIX ANALYSIS AFTER STATE SHARING")
    print("=" * 70)

    # Check H matrices AFTER sharing states
    for i in range(n_anchors, min(n_total, n_anchors+3)):
        node = cgn.nodes[i]
        H, g = node.build_local_system()

        print(f"\nNode {i} (AFTER state sharing):")
        print(f"  Neighbors with states: {list(node.neighbor_states.keys())}")
        print(f"  Local factors: {len(node.local_factors)}")

        # Count how many factors can be evaluated
        evaluable = 0
        for factor in node.local_factors:
            if isinstance(factor, ToAFactorMeters):
                if factor.i == i:
                    xj = node._get_node_state(factor.j)
                    if xj is not None:
                        evaluable += 1
                elif factor.j == i:
                    xi = node._get_node_state(factor.i)
                    if xi is not None:
                        evaluable += 1

        print(f"  Evaluable factors: {evaluable}/{len(node.local_factors)}")
        print(f"  H matrix properties:")
        print(f"    Sum of |H|: {np.sum(np.abs(H)):.1f}")

        if np.sum(np.abs(H)) > 0:
            eigvals = np.linalg.eigvalsh(H)
            print(f"    Eigenvalues: min={eigvals.min():.1e}, max={eigvals.max():.1e}")
            print(f"    Condition number: {eigvals.max()/max(eigvals.min(), 1e-15):.1e}")
            print(f"    H diagonal (first 3): {np.diag(H)[:3]}")

            # Try to solve
            try:
                delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                print(f"    ✓ Can solve! Delta norm: {np.linalg.norm(delta):.3f}")
                print(f"      Position update: {np.linalg.norm(delta[:2]):.3f} m")
            except:
                print(f"    ❌ Cannot solve - singular matrix")
        else:
            print("    ❌ H is all zeros! Cannot update.")

    # Try actual updates
    print("\n" + "=" * 70)
    print("ATTEMPTING UPDATES")
    print("=" * 70)

    initial_positions = {}
    for i in range(n_anchors, n_total):
        initial_positions[i] = cgn.nodes[i].state[:2].copy()

    # Run one update
    for node_id, node in cgn.nodes.items():
        if not node.config.is_anchor:
            H, g = node.build_local_system()
            if np.sum(np.abs(H)) > 0:
                try:
                    delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                    node.state += cgn_config.step_size * delta
                except:
                    pass

    # Check who moved
    print("\nPosition changes after one update:")
    for i in range(n_anchors, n_total):
        change = np.linalg.norm(cgn.nodes[i].state[:2] - initial_positions[i])
        if change > 0.001:
            print(f"  Node {i}: {change:.3f} m ✓")
        else:
            print(f"  Node {i}: {change:.6f} m ❌ (didn't move)")


if __name__ == "__main__":
    debug_h_matrix()