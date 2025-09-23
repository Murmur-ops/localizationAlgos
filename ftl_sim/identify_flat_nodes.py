#!/usr/bin/env python3
"""
Identify which specific nodes have flat convergence and why.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from run_unified_ftl import (
    generate_network_topology,
    initialize_clock_states,
    generate_all_measurements
)
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters


def identify_flat_nodes():
    """Identify and analyze nodes with flat convergence"""

    # Load configuration
    with open('configs/unified_ideal.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Generate network
    np.random.seed(42)
    true_positions, n_anchors, n_total = generate_network_topology(config)

    # Initialize clocks and measurements
    clock_states = initialize_clock_states(config['rf_simulation'], n_total, n_anchors)
    measurements = generate_all_measurements(true_positions, clock_states, config['rf_simulation'])

    # Setup consensus
    cgn_config = ConsensusGNConfig(max_iterations=1, step_size=0.5, verbose=False)
    cgn = ConsensusGaussNewton(cgn_config)
    cgn.set_true_positions(true_positions)

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
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    for (i, j) in measurements.keys():
        cgn.add_edge(i, j)

    for (i, j), meas_list in measurements.items():
        for meas in meas_list:
            factor = ToAFactorMeters(i, j, meas['range_m'], 0.01**2)
            cgn.add_measurement(factor)

    print("=" * 70)
    print("IDENTIFYING FLAT CONVERGENCE NODES")
    print("=" * 70)

    # Track position changes over iterations
    n_iterations = 20
    position_changes = {i: [] for i in range(n_anchors, n_total)}

    for iteration in range(n_iterations):
        # Store positions before update
        before_positions = {i: cgn.nodes[i].state[:2].copy() for i in range(n_anchors, n_total)}

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

        # Update nodes
        for node_id, node in cgn.nodes.items():
            if not node.config.is_anchor:
                H, g = node.build_local_system()
                if np.sum(np.abs(H)) > 0:
                    try:
                        delta = np.linalg.solve(H + 1e-3 * np.eye(5), -g)
                        node.state += cgn_config.step_size * delta
                    except:
                        pass

        # Track changes
        for i in range(n_anchors, n_total):
            change = np.linalg.norm(cgn.nodes[i].state[:2] - before_positions[i])
            position_changes[i].append(change)

    # Analyze which nodes are flat
    print(f"\nAnalyzed {n_total - n_anchors} unknown nodes over {n_iterations} iterations")
    print("-" * 70)

    flat_nodes = []
    converging_nodes = []
    stuck_nodes = []

    for i in range(n_anchors, n_total):
        changes = position_changes[i]

        # Check if node stopped updating
        recent_changes = changes[-5:]  # Last 5 iterations
        avg_recent_change = np.mean(recent_changes)

        if avg_recent_change < 1e-6:  # Less than 1 micrometer
            flat_nodes.append(i)
        elif avg_recent_change < 1e-3:  # Less than 1mm
            stuck_nodes.append(i)
        else:
            converging_nodes.append(i)

    print(f"\nNode Categories:")
    print(f"  ✓ Converging (>1mm/iter): {len(converging_nodes)} nodes")
    print(f"  ⚠ Stuck (<1mm/iter): {len(stuck_nodes)} nodes")
    print(f"  ❌ Flat (<1μm/iter): {len(flat_nodes)} nodes")

    # Detailed analysis of flat nodes
    if flat_nodes:
        print("\n" + "=" * 70)
        print("ANALYZING FLAT NODES")
        print("=" * 70)

        for node_id in flat_nodes[:5]:  # Analyze first 5
            node = cgn.nodes[node_id]

            # Count connections
            n_neighbors = len([e for e in cgn.edges if node_id in e])
            n_factors = len(node.local_factors)

            # Check H matrix
            H, g = node.build_local_system()

            print(f"\nNode {node_id}:")
            print(f"  Connections: {n_neighbors} edges")
            print(f"  Factors: {n_factors}")
            print(f"  Neighbor states available: {len(node.neighbor_states)}")

            if np.sum(np.abs(H)) > 0:
                eigvals = np.linalg.eigvalsh(H)
                cond = eigvals.max() / max(eigvals.min(), 1e-15)
                print(f"  H matrix: sum={np.sum(np.abs(H)):.1f}, cond={cond:.1e}")
                print(f"  Gradient norm: {np.linalg.norm(g):.3e}")

                # Check if gradient is near zero
                if np.linalg.norm(g) < 1e-10:
                    print(f"  → Converged (gradient ≈ 0)")
                else:
                    print(f"  → Should be updating but isn't!")
            else:
                print(f"  H matrix: ALL ZEROS - cannot update")

    # Plot convergence behavior
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a few examples from each category
    iterations = np.arange(n_iterations)

    if converging_nodes:
        for i in converging_nodes[:3]:
            ax.semilogy(iterations, position_changes[i], 'g-', alpha=0.5, label='Converging' if i == converging_nodes[0] else '')

    if stuck_nodes:
        for i in stuck_nodes[:3]:
            ax.semilogy(iterations, position_changes[i], 'y-', alpha=0.5, label='Stuck' if i == stuck_nodes[0] else '')

    if flat_nodes:
        for i in flat_nodes[:3]:
            ax.semilogy(iterations, [max(c, 1e-10) for c in position_changes[i]], 'r-', alpha=0.5, label='Flat' if i == flat_nodes[0] else '')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position Change (m)')
    ax.set_title('Node Update Behavior Categories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('node_categories.png', dpi=100)
    print(f"\nPlot saved to node_categories.png")

    return flat_nodes, stuck_nodes, converging_nodes


if __name__ == "__main__":
    flat, stuck, converging = identify_flat_nodes()