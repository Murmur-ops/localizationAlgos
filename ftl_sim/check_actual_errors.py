#!/usr/bin/env python3
"""
Check the actual position errors to see if nodes are already converged.
"""

import numpy as np
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


def check_errors():
    """Check if nodes are stuck because they're already converged"""

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

    # Add nodes with initial errors
    initial_errors = {}
    for i in range(n_total):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = true_positions[i]
            state[2] = clock_states[i].bias * 1e9
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Add initial error
            error = np.random.normal(0, 0.1, 2)  # 10cm std dev
            state[:2] = true_positions[i] + error
            state[2] = clock_states[i].bias * 1e9 + np.random.normal(0, 0.1)
            cgn.add_node(i, state, is_anchor=False)
            initial_errors[i] = np.linalg.norm(error)

    # Add edges and measurements
    for (i, j) in measurements.keys():
        cgn.add_edge(i, j)

    for (i, j), meas_list in measurements.items():
        for meas in meas_list:
            factor = ToAFactorMeters(i, j, meas['range_m'], 0.01**2)
            cgn.add_measurement(factor)

    print("=" * 70)
    print("INITIAL VS CONVERGED ERRORS")
    print("=" * 70)

    # Print initial errors
    initial_rms = np.sqrt(np.mean([e**2 for e in initial_errors.values()]))
    print(f"\nInitial position errors:")
    print(f"  RMS: {initial_rms*100:.2f} cm")
    print(f"  Max: {max(initial_errors.values())*100:.2f} cm")
    print(f"  Min: {min(initial_errors.values())*100:.2f} cm")

    # Run 50 iterations
    for iteration in range(50):
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

        if iteration in [0, 9, 19, 49]:
            errors = []
            for i in range(n_anchors, n_total):
                err = np.linalg.norm(cgn.nodes[i].state[:2] - true_positions[i])
                errors.append(err)
            rms = np.sqrt(np.mean(np.array(errors)**2))
            print(f"\nAfter {iteration+1} iterations:")
            print(f"  RMS: {rms*100:.2f} cm")
            print(f"  Max: {max(errors)*100:.2f} cm")
            print(f"  Min: {min(errors)*100:.2f} cm")

    # Final detailed analysis
    print("\n" + "=" * 70)
    print("FINAL ERROR ANALYSIS")
    print("=" * 70)

    final_errors = []
    converged_well = []
    not_converged = []

    for i in range(n_anchors, n_total):
        err = np.linalg.norm(cgn.nodes[i].state[:2] - true_positions[i])
        final_errors.append(err)

        if err < 0.01:  # Less than 1 cm
            converged_well.append(i)
        elif err > 0.05:  # More than 5 cm
            not_converged.append(i)

    print(f"\nConvergence quality:")
    print(f"  Excellent (<1cm): {len(converged_well)} nodes")
    print(f"  Poor (>5cm): {len(not_converged)} nodes")

    # Check gradient magnitudes
    print("\nGradient analysis (should be near zero if converged):")
    large_gradient_nodes = []
    for i in range(n_anchors, min(n_anchors+5, n_total)):
        H, g = cgn.nodes[i].build_local_system()
        grad_norm = np.linalg.norm(g)
        err = np.linalg.norm(cgn.nodes[i].state[:2] - true_positions[i])
        print(f"  Node {i}: gradient={grad_norm:.3e}, error={err*100:.2f}cm")
        if grad_norm > 1e-3:
            large_gradient_nodes.append(i)

    if large_gradient_nodes:
        print(f"\n⚠ Nodes with large gradients but slow updates: {large_gradient_nodes}")
        print("  This suggests numerical issues or step size too small")

    return final_errors


if __name__ == "__main__":
    errors = check_errors()