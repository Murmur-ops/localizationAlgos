#!/usr/bin/env python3
"""
Debug the solver NaN issue
"""

import numpy as np
from ftl.solver import FactorGraph
from ftl.geometry import place_anchors, place_grid_nodes, PlacementType

def test_solver_numerical_stability():
    """Test what's causing the NaN in the solver"""

    print("Testing solver numerical stability...")
    print("="*60)

    np.random.seed(42)

    # Create a simple test case
    graph = FactorGraph()

    # 4 anchors at corners
    area_size = 50.0
    anchors = [
        (0, 0),
        (area_size, 0),
        (area_size, area_size),
        (0, area_size)
    ]

    for i, (x, y) in enumerate(anchors):
        graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)
        print(f"Anchor {i} at ({x}, {y})")

    # Add unknown nodes
    n_unknowns = 4
    for i in range(n_unknowns):
        x = (i % 2) * area_size/2 + area_size/4
        y = (i // 2) * area_size/2 + area_size/4
        initial = np.array([x + np.random.randn(), y + np.random.randn(), 0, 0, 0])
        graph.add_node(4+i, initial, is_anchor=False)
        print(f"Unknown {4+i} at ({x:.1f}, {y:.1f})")

    # Add measurements with different variances
    variances_to_test = [1e-20, 1e-19, 1e-18, 1e-17, 1e-16]

    for var_idx, variance in enumerate(variances_to_test):
        print(f"\n--- Testing variance = {variance:.1e} s² ---")

        # Reset graph factors
        graph.factors = []

        # Add measurements from each anchor to each unknown
        n_measurements = 0
        for anchor_id in range(4):
            for unknown_id in range(4, 8):
                anchor_pos = graph.nodes[anchor_id].state[:2]
                unknown_pos = graph.nodes[unknown_id].state[:2]

                # True distance and ToA
                dist = np.linalg.norm(anchor_pos - unknown_pos)
                toa = dist / 3e8

                # Add small noise
                noise = np.random.randn() * np.sqrt(variance)
                measured_toa = toa + noise

                graph.add_toa_factor(anchor_id, unknown_id, measured_toa, variance)
                n_measurements += 1

        print(f"Added {n_measurements} measurements")

        # Check Jacobian
        J, r, node_to_idx, idx_to_node = graph._build_jacobian()
        print(f"Jacobian shape: {J.shape}")
        print(f"Jacobian rank: {np.linalg.matrix_rank(J)}")
        print(f"Jacobian condition number: {np.linalg.cond(J):.2e}")

        # Check if any columns are zero
        zero_cols = np.where(np.all(J == 0, axis=0))[0]
        if len(zero_cols) > 0:
            print(f"WARNING: Zero columns in Jacobian: {zero_cols}")

        # Check Hessian
        residuals, stds = graph._compute_residuals()
        weights = np.ones(len(residuals))
        W = np.diag(weights)
        J_weighted = W @ J
        H = J_weighted.T @ J_weighted

        print(f"Hessian shape: {H.shape}")
        print(f"Hessian diagonal min: {np.min(np.diag(H)):.2e}")
        print(f"Hessian diagonal max: {np.max(np.diag(H)):.2e}")

        # Check for zero diagonal elements
        zero_diag = np.where(np.diag(H) == 0)[0]
        if len(zero_diag) > 0:
            print(f"WARNING: Zero diagonal elements in H: {zero_diag}")
            # Map back to variables
            for idx in zero_diag:
                node_idx = idx // 5
                var_idx = idx % 5
                var_names = ['x', 'y', 'bias', 'drift', 'cfo']
                print(f"  -> Node {idx_to_node.get(node_idx*5, '?')}, variable: {var_names[var_idx]}")

        # Try to optimize
        try:
            result = graph.optimize(max_iterations=10, verbose=False)
            print(f"Optimization result: converged={result.converged}, cost={result.final_cost:.2e}")

            # Check for NaN in estimates
            has_nan = False
            for node_id, estimate in result.estimates.items():
                if np.any(np.isnan(estimate)):
                    print(f"  WARNING: NaN in node {node_id} estimate")
                    has_nan = True

            if not has_nan:
                print("  ✓ No NaN values")

        except Exception as e:
            print(f"Optimization failed: {e}")

    print("\n" + "="*60)
    print("Analysis complete")

    # Now test with clock/drift/cfo factors
    print("\nTesting with clock parameters...")
    graph.factors = []

    # Add ToA factors
    for anchor_id in range(4):
        for unknown_id in range(4, 8):
            anchor_pos = graph.nodes[anchor_id].state[:2]
            unknown_pos = graph.nodes[unknown_id].state[:2]
            dist = np.linalg.norm(anchor_pos - unknown_pos)
            toa = dist / 3e8
            graph.add_toa_factor(anchor_id, unknown_id, toa, 1e-18)

    # Add a CFO measurement
    graph.add_cfo_factor(4, 5, 100.0, 1.0)  # 100 Hz CFO

    J, r, node_to_idx, idx_to_node = graph._build_jacobian()
    print(f"\nWith CFO factor:")
    print(f"  Jacobian rank: {np.linalg.matrix_rank(J)}")

    # Check which variables are constrained
    J_squared = J.T @ J
    eigenvalues = np.linalg.eigvals(J_squared)
    print(f"  Eigenvalues range: [{np.min(eigenvalues):.2e}, {np.max(eigenvalues):.2e}]")

    # Count zero eigenvalues
    zero_eigs = np.sum(np.abs(eigenvalues) < 1e-10)
    print(f"  Near-zero eigenvalues: {zero_eigs}")

    if zero_eigs > 0:
        print("  -> System is rank-deficient")
        print("  -> Need more constraints on clock parameters")


if __name__ == "__main__":
    test_solver_numerical_stability()