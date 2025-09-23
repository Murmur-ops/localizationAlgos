#!/usr/bin/env python3
"""
Test suite to diagnose consensus convergence issues.
Tests state sharing, convergence behavior, and identifies why nodes aren't converging properly.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.consensus_node import ConsensusNode, ConsensusNodeConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters, ClockPriorFactor


class TestConsensusConvergence(unittest.TestCase):
    """Test consensus algorithm convergence behavior"""

    def setUp(self):
        """Setup simple test network"""
        self.n_nodes = 5
        self.n_anchors = 2

        # Simple linear topology: 0--1--2--3--4
        # Anchors at 0 and 4
        self.true_positions = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0]
        ])

        # Simple clock states (all zero for simplicity)
        self.true_clock_bias = np.zeros(5)  # ns
        self.true_clock_drift = np.zeros(5)  # ppb
        self.true_clock_cfo = np.zeros(5)  # ppm

    def test_state_sharing_between_neighbors(self):
        """Test that nodes actually share states with neighbors"""
        print("\n" + "="*70)
        print("TEST: State Sharing Between Neighbors")
        print("="*70)

        # Create consensus system
        cgn_config = ConsensusGNConfig(
            max_iterations=10,
            step_size=0.5,
            verbose=False
        )
        cgn = ConsensusGaussNewton(cgn_config)

        # Add nodes
        for i in range(self.n_nodes):
            state = np.zeros(5)
            state[:2] = self.true_positions[i]
            if i < self.n_anchors or i == self.n_nodes - 1:  # Anchors at 0 and 4
                cgn.add_node(i, state, is_anchor=True)
            else:
                # Give unknown nodes initial error
                state[:2] += np.random.randn(2) * 0.1
                cgn.add_node(i, state, is_anchor=False)

        # Add edges (linear chain)
        for i in range(self.n_nodes - 1):
            cgn.add_edge(i, i+1)

        # Add perfect measurements
        for i in range(self.n_nodes - 1):
            true_dist = np.linalg.norm(self.true_positions[i+1] - self.true_positions[i])
            factor = ToAFactorMeters(i, i+1, true_dist, 0.01**2)
            cgn.add_measurement(factor)

        # Store initial states
        initial_states = {i: cgn.nodes[i].state.copy() for i in range(self.n_nodes)}

        # Initialize neighbor states by exchanging messages
        print("\nInitializing neighbor states...")
        import time
        current_time = time.time()
        for node_id, node in cgn.nodes.items():
            for edge in cgn.edges:
                if edge[0] == node_id:
                    neighbor_id = edge[1]
                    if neighbor_id in cgn.nodes:
                        msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(), 0, current_time)
                        node.receive_state(msg)
                elif edge[1] == node_id:
                    neighbor_id = edge[0]
                    if neighbor_id in cgn.nodes:
                        msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(), 0, current_time)
                        node.receive_state(msg)

        # Manually perform one consensus iteration
        print("\nInitial states:")
        for i in range(self.n_nodes):
            print(f"  Node {i}: pos=({cgn.nodes[i].state[0]:.3f}, {cgn.nodes[i].state[1]:.3f})")

        # Build and solve local systems
        for node_id, node in cgn.nodes.items():
            if not node.config.is_anchor:
                print(f"\nNode {node_id} debug:")
                print(f"  config.node_id: {node.config.node_id}")
                print(f"  Number of local factors: {len(node.local_factors)}")
                print(f"  Neighbor states: {list(node.neighbor_states.keys())}")

                if len(node.local_factors) > 0:
                    for idx, factor in enumerate(node.local_factors[:3]):  # Show first 3
                        if isinstance(factor, ToAFactorMeters):
                            print(f"    Factor {idx}: ToA between {factor.i} and {factor.j}")
                            # Test if we can get states
                            if factor.i == node.config.node_id:
                                xi = node.state
                                xj = node._get_node_state(factor.j)
                            else:
                                xi = node._get_node_state(factor.i)
                                xj = node.state
                            print(f"      xi is None: {xi is None}, xj is None: {xj is None}")

                H, g = node.build_local_system()
                print(f"  H shape: {H.shape}, g shape: {g.shape}")
                print(f"  H[0,0]: {H[0,0]:.3f}, g[0]: {g[0]:.3f}")

                # Check if H is singular
                eigvals = np.linalg.eigvalsh(H)
                print(f"  H eigenvalues: min={np.min(eigvals):.3e}, max={np.max(eigvals):.3e}")

                if np.min(eigvals) < 1e-10:
                    print(f"  WARNING: H is nearly singular!")

        # Test state message creation and processing
        print("\nTesting state message exchange:")
        messages = {}
        for node_id, node in cgn.nodes.items():
            msg = StateMessage(
                node_id=node_id,
                state=node.state.copy(),
                iteration=1,
                timestamp=current_time
            )
            messages[node_id] = msg
            print(f"  Created message from node {node_id}")

        # Process messages
        for node_id, node in cgn.nodes.items():
            neighbors = []
            for edge in cgn.edges:
                if edge[0] == node_id:
                    neighbors.append(edge[1])
                elif edge[1] == node_id:
                    neighbors.append(edge[0])

            print(f"\nNode {node_id} neighbors: {neighbors}")
            for neighbor_id in neighbors:
                if neighbor_id in messages:
                    # Use receive_state method
                    if hasattr(node, 'receive_state'):
                        node.receive_state(messages[neighbor_id])
                        print(f"  Node {node_id} received state from {neighbor_id}")
                    else:
                        print(f"  WARNING: Node {node_id} has no receive_state method!")

        # Check if states changed
        print("\nState changes after message exchange:")
        any_changed = False
        for i in range(self.n_nodes):
            if not cgn.nodes[i].config.is_anchor:
                change = np.linalg.norm(cgn.nodes[i].state - initial_states[i])
                print(f"  Node {i}: change = {change:.6f}")
                if change > 1e-6:
                    any_changed = True

        self.assertTrue(any_changed, "No states changed after message exchange - state sharing not working!")

    def test_convergence_over_iterations(self):
        """Test that errors decrease over iterations"""
        print("\n" + "="*70)
        print("TEST: Convergence Over Iterations")
        print("="*70)

        # Create consensus system
        cgn_config = ConsensusGNConfig(
            max_iterations=50,
            step_size=0.5,
            verbose=False
        )
        cgn = ConsensusGaussNewton(cgn_config)

        # Add nodes with larger initial errors
        for i in range(self.n_nodes):
            state = np.zeros(5)
            if i == 0 or i == self.n_nodes - 1:  # Anchors
                state[:2] = self.true_positions[i]
                cgn.add_node(i, state, is_anchor=True)
            else:
                # Large initial error for unknown nodes
                state[:2] = self.true_positions[i] + np.random.randn(2) * 0.5
                state[2] = np.random.randn() * 0.1  # Clock bias error
                cgn.add_node(i, state, is_anchor=False)

        # Add edges
        for i in range(self.n_nodes - 1):
            cgn.add_edge(i, i+1)

        # Add noisy measurements
        for i in range(self.n_nodes - 1):
            true_dist = np.linalg.norm(self.true_positions[i+1] - self.true_positions[i])
            noisy_dist = true_dist + np.random.randn() * 0.01
            factor = ToAFactorMeters(i, i+1, noisy_dist, 0.01**2)
            cgn.add_measurement(factor)

        # Track errors over iterations
        errors_over_time = []

        for iteration in range(50):
            # Calculate current error
            total_error = 0
            for i in range(1, self.n_nodes - 1):  # Unknown nodes only
                pos_error = np.linalg.norm(cgn.nodes[i].state[:2] - self.true_positions[i])
                total_error += pos_error
            avg_error = total_error / (self.n_nodes - 2)
            errors_over_time.append(avg_error)

            # Run one iteration
            cgn.config.max_iterations = 1
            cgn.optimize()

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: avg error = {avg_error:.4f} m")

        # Check that error decreases
        print(f"\nInitial error: {errors_over_time[0]:.4f} m")
        print(f"Final error: {errors_over_time[-1]:.4f} m")
        print(f"Improvement: {errors_over_time[0]/errors_over_time[-1]:.1f}x")

        self.assertLess(errors_over_time[-1], errors_over_time[0] * 0.1,
                        "Error did not decrease by at least 10x")

        # Check monotonic decrease (with some tolerance for noise)
        increasing_count = 0
        for i in range(1, len(errors_over_time)):
            if errors_over_time[i] > errors_over_time[i-1] * 1.1:  # Allow 10% increase
                increasing_count += 1

        print(f"Iterations with error increase: {increasing_count}/{len(errors_over_time)-1}")
        self.assertLess(increasing_count, 5, "Too many iterations with error increase")

    def test_frequency_offset_update(self):
        """Test that frequency offset (CFO) actually gets updated"""
        print("\n" + "="*70)
        print("TEST: Frequency Offset Updates")
        print("="*70)

        cgn_config = ConsensusGNConfig(max_iterations=20, step_size=0.5, verbose=False)
        cgn = ConsensusGaussNewton(cgn_config)

        # Add nodes with CFO errors
        for i in range(3):  # Simpler 3-node network
            state = np.zeros(5)
            state[:2] = [i, 0]  # Linear positions

            if i == 0:  # Anchor
                cgn.add_node(i, state, is_anchor=True)
            else:
                state[4] = 0.5  # Initial CFO error of 0.5 ppm
                cgn.add_node(i, state, is_anchor=False)

        # Add edges
        cgn.add_edge(0, 1)
        cgn.add_edge(1, 2)

        # Add measurements with CFO-dependent errors
        for i in range(2):
            true_dist = 1.0  # All nodes 1m apart
            # Add CFO-dependent measurement (simulating frequency error effect)
            factor = ToAFactorMeters(i, i+1, true_dist, 0.01**2)
            cgn.add_measurement(factor)

        # Track CFO over iterations
        initial_cfo_1 = cgn.nodes[1].state[4]
        initial_cfo_2 = cgn.nodes[2].state[4]

        print(f"Initial CFO - Node 1: {initial_cfo_1:.3f} ppm, Node 2: {initial_cfo_2:.3f} ppm")

        # Run optimization
        cgn.optimize()

        final_cfo_1 = cgn.nodes[1].state[4]
        final_cfo_2 = cgn.nodes[2].state[4]

        print(f"Final CFO - Node 1: {final_cfo_1:.3f} ppm, Node 2: {final_cfo_2:.3f} ppm")

        # Check that CFO changed
        cfo_change_1 = abs(final_cfo_1 - initial_cfo_1)
        cfo_change_2 = abs(final_cfo_2 - initial_cfo_2)

        print(f"CFO changes - Node 1: {cfo_change_1:.6f}, Node 2: {cfo_change_2:.6f}")

        # The issue is that CFO might not change if there are no CFO-specific measurements
        # This is actually revealing the problem!
        if cfo_change_1 < 1e-6 and cfo_change_2 < 1e-6:
            print("WARNING: CFO not updating! This explains the flat frequency convergence plots.")
            print("The consensus algorithm may not be properly handling frequency offset estimation.")

    def test_individual_node_behavior(self):
        """Test individual node update behavior"""
        print("\n" + "="*70)
        print("TEST: Individual Node Behavior")
        print("="*70)

        # Create a single unknown node with two anchor neighbors
        cgn_config = ConsensusGNConfig(max_iterations=10, step_size=0.5, verbose=False)
        cgn = ConsensusGaussNewton(cgn_config)

        # Triangle configuration
        positions = np.array([
            [0, 0],    # Anchor 0
            [1, 0],    # Unknown 1
            [0.5, 0.866]  # Anchor 2 (equilateral triangle)
        ])

        # Add nodes
        for i in range(3):
            state = np.zeros(5)
            if i != 1:  # Anchors at 0 and 2
                state[:2] = positions[i]
                cgn.add_node(i, state, is_anchor=True)
            else:  # Unknown at 1
                state[:2] = positions[i] + [0.2, 0.1]  # Initial error
                state[2] = 0.1  # Clock bias error
                cgn.add_node(i, state, is_anchor=False)
                initial_state = state.copy()

        # Add edges
        cgn.add_edge(0, 1)
        cgn.add_edge(1, 2)
        cgn.add_edge(0, 2)

        # Add perfect measurements
        for i in range(3):
            for j in range(i+1, 3):
                true_dist = np.linalg.norm(positions[j] - positions[i])
                factor = ToAFactorMeters(i, j, true_dist, 0.001**2)  # Very low noise
                cgn.add_measurement(factor)

        print(f"Initial position error: {np.linalg.norm(cgn.nodes[1].state[:2] - positions[1]):.4f} m")

        # Track node 1's state over iterations
        state_history = [cgn.nodes[1].state.copy()]

        for iter in range(10):
            # Build local system for node 1
            H, g = cgn.nodes[1].build_local_system()

            print(f"\nIteration {iter}:")
            print(f"  H diagonal: {np.diag(H)}")
            print(f"  g: {g}")

            # Check if update would improve position
            try:
                delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
                print(f"  Delta: {delta}")

                # Apply update
                cgn.nodes[1].state += cgn_config.step_size * delta
                state_history.append(cgn.nodes[1].state.copy())

                pos_error = np.linalg.norm(cgn.nodes[1].state[:2] - positions[1])
                print(f"  Position error: {pos_error:.4f} m")
            except np.linalg.LinAlgError:
                print("  Failed to solve system!")
                break

        # Check convergence
        final_error = np.linalg.norm(cgn.nodes[1].state[:2] - positions[1])
        self.assertLess(final_error, 0.01, f"Node did not converge to within 1cm (error: {final_error:.3f}m)")

    def test_measurement_whitening(self):
        """Test that measurement whitening is working correctly"""
        print("\n" + "="*70)
        print("TEST: Measurement Whitening")
        print("="*70)

        # Create simple 2-node system
        pos_i = np.array([0, 0])
        pos_j = np.array([1, 0])
        true_distance = 1.0

        # Create states
        state_i = np.zeros(5)
        state_i[:2] = pos_i

        state_j = np.zeros(5)
        state_j[:2] = pos_j + [0.1, 0]  # 10cm error

        # Create factor with known variance
        variance = 0.01**2  # 1cm std dev
        factor = ToAFactorMeters(0, 1, true_distance, variance)

        # Get whitened residual
        r_wh, Ji_wh, Jj_wh = factor.whitened_residual_and_jacobian(state_i, state_j)

        print(f"Measurement: {true_distance:.3f} m")
        print(f"Estimated distance: {np.linalg.norm(state_j[:2] - state_i[:2]):.3f} m")
        print(f"Residual (unwhitened): {true_distance - np.linalg.norm(state_j[:2] - state_i[:2]):.3f} m")
        print(f"Residual (whitened): {r_wh:.3f}")
        print(f"Expected whitened: {(true_distance - 1.1) / 0.01:.3f}")

        # Check whitening
        expected_whitened = (true_distance - np.linalg.norm(state_j[:2] - state_i[:2])) / np.sqrt(variance)
        self.assertAlmostEqual(r_wh, expected_whitened, places=5,
                              msg="Whitening not working correctly")

        # Check Jacobians
        print(f"\nJacobian i (whitened): {Ji_wh}")
        print(f"Jacobian j (whitened): {Jj_wh}")

        # Jacobians should be non-zero for position components
        self.assertGreater(np.linalg.norm(Ji_wh[:2]), 0.1, "Jacobian for position too small")
        self.assertGreater(np.linalg.norm(Jj_wh[:2]), 0.1, "Jacobian for position too small")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)