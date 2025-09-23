#!/usr/bin/env python3
"""Diagnose why ideal 30-node case isn't converging"""

import numpy as np
import yaml
from pathlib import Path
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

def diagnose_convergence():
    # Load config
    with open('configs/ideal_30node.yaml', 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(42)

    # Setup from config
    n_nodes = 30
    n_anchors = 5
    n_unknowns = 25

    # Anchor positions (fixed in config)
    anchor_positions = np.array([
        [0.0, 0.0],
        [50.0, 0.0],
        [50.0, 50.0],
        [0.0, 50.0],
        [25.0, 25.0]  # Center anchor
    ])

    # Generate unknown positions on grid
    x_pos = np.linspace(5, 45, 5)
    y_pos = np.linspace(5, 45, 5)

    unknown_positions = []
    for x in x_pos:
        for y in y_pos:
            unknown_positions.append([x, y])
    unknown_positions = np.array(unknown_positions[:n_unknowns])

    true_positions = np.vstack([anchor_positions, unknown_positions])

    # Create consensus solver with verbose tracking
    cgn_config = ConsensusGNConfig(
        max_iterations=500,
        consensus_gain=0.05,
        step_size=0.3,
        gradient_tol=1e-5,
        step_tol=1e-6,
        verbose=True
    )

    cgn = ConsensusGaussNewton(cgn_config)

    # Add nodes
    for i in range(n_nodes):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = true_positions[i]
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Initialize with noise
            initial_pos = true_positions[i] + np.random.normal(0, 0.5, 2)
            state[:2] = initial_pos
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    comm_range = 25.0
    n_measurements = 0

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)

                # Add measurement with 1cm noise
                meas_range = dist + np.random.normal(0, 0.01)
                cgn.add_measurement(ToAFactorMeters(i, j, meas_range, 0.01**2))
                n_measurements += 1

    print(f"Network: {n_nodes} nodes, {n_measurements} measurements")
    print(f"Average degree: {n_measurements * 2 / n_nodes:.1f}")

    # Set true positions for evaluation
    cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

    # Track convergence metrics
    iteration_data = []

    # Custom optimization loop to track metrics
    print("\nIteration | RMSE (cm) | Max Grad | Max Step | Consensus")
    print("-" * 60)

    for iteration in range(50):  # Just 50 iterations for diagnosis
        # Store current positions
        positions_before = np.array([cgn.nodes[i].state[:2].copy()
                                     for i in range(n_anchors, n_nodes)])

        # One iteration
        cgn._iteration()

        # Get updated positions
        positions_after = np.array([cgn.nodes[i].state[:2]
                                    for i in range(n_anchors, n_nodes)])

        # Calculate metrics
        errors = np.linalg.norm(positions_after - true_positions[n_anchors:], axis=1)
        rmse = np.sqrt(np.mean(errors**2))

        # Get gradient and step norms
        max_grad_norm = 0
        max_step_norm = 0
        consensus_penalty = 0

        for i in range(n_anchors, n_nodes):
            node = cgn.nodes[i]
            if hasattr(node, 'last_gradient'):
                max_grad_norm = max(max_grad_norm, np.linalg.norm(node.last_gradient))

            step = positions_after[i-n_anchors] - positions_before[i-n_anchors]
            max_step_norm = max(max_step_norm, np.linalg.norm(step))

            # Calculate consensus disagreement
            for j in node.neighbor_states:
                if j >= n_anchors:
                    neighbor_pos = cgn.nodes[j].state[:2]
                    consensus_penalty += np.linalg.norm(positions_after[i-n_anchors] - neighbor_pos)

        print(f"{iteration:9d} | {rmse*100:9.2f} | {max_grad_norm:8.2e} | {max_step_norm:8.2e} | {consensus_penalty:8.2f}")

        iteration_data.append({
            'iteration': iteration,
            'rmse_cm': rmse * 100,
            'max_grad': max_grad_norm,
            'max_step': max_step_norm,
            'consensus': consensus_penalty
        })

        # Check if stuck
        if iteration > 10:
            recent_rmse = [d['rmse_cm'] for d in iteration_data[-10:]]
            if np.std(recent_rmse) < 0.01:  # RMSE not changing
                print(f"\n⚠️ Stuck at iteration {iteration}!")
                print(f"RMSE plateau: {np.mean(recent_rmse):.2f} cm")
                break

    # Analyze the issue
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)

    # Check final state
    final_rmse = iteration_data[-1]['rmse_cm']
    final_grad = iteration_data[-1]['max_grad']

    if final_rmse > 2.0:
        print("❌ Not converging to target accuracy")

        if final_grad < 1e-6:
            print("   → Gradient too small (local minimum or numerical issues)")
            print("   → Try: Increase step_size or decrease damping")
        elif final_grad > 0.01:
            print("   → Gradient still large (not at minimum)")
            print("   → Try: More iterations or better step size")

        if iteration_data[-1]['consensus'] > 10:
            print("   → High consensus penalty (nodes disagree)")
            print("   → Try: Adjust consensus_gain (μ)")

    # Check oscillation
    recent_rmse = [d['rmse_cm'] for d in iteration_data[-20:]]
    if np.std(recent_rmse) > 0.1:
        print("❌ Oscillating around solution")
        print("   → Try: Reduce step_size or add momentum")

    # Check if converged but tolerance too strict
    if final_rmse < 2.0 and final_grad > cgn_config.gradient_tol:
        print("⚠️ Good accuracy but gradient tolerance not met")
        print(f"   Current gradient: {final_grad:.2e}")
        print(f"   Target: {cgn_config.gradient_tol:.2e}")
        print("   → Solution: Relax gradient_tol or use different convergence criteria")

    return iteration_data

if __name__ == "__main__":
    diagnose_convergence()