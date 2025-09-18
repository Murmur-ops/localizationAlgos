"""
Unit tests for ConsensusNode
Test local optimization, neighbor management, and consensus updates
"""

import numpy as np
import pytest
import time

from ftl.consensus.consensus_node import ConsensusNode, ConsensusNodeConfig
from ftl.consensus.message_types import StateMessage, ConvergenceStatus
from ftl.factors_scaled import ToAFactorMeters, ClockPriorFactor


class TestConsensusNodeConfig:
    """Test node configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ConsensusNodeConfig(node_id=1)

        assert config.node_id == 1
        assert not config.is_anchor
        assert config.consensus_gain == 1.0
        assert config.step_size == 0.5
        assert config.gradient_tol == 1e-6
        assert config.step_tol == 1e-8
        assert np.allclose(config.state_scale, [1.0, 1.0, 1.0, 0.1, 0.1])

    def test_anchor_config(self):
        """Test anchor node configuration"""
        config = ConsensusNodeConfig(node_id=0, is_anchor=True)
        assert config.is_anchor


class TestConsensusNode:
    """Test ConsensusNode implementation"""

    def test_node_creation(self):
        """Test creating a consensus node"""
        config = ConsensusNodeConfig(node_id=1)
        initial_state = np.array([10.0, 20.0, 5.0, 0.1, 0.01])
        node = ConsensusNode(config, initial_state)

        assert node.config.node_id == 1
        assert np.allclose(node.state, initial_state)
        assert node.iteration == 0
        assert not node.converged
        assert len(node.neighbors) == 0
        assert len(node.local_factors) == 0

    def test_neighbor_management(self):
        """Test adding and removing neighbors"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        # Add neighbors
        node.add_neighbor(2)
        node.add_neighbor(3)
        assert 2 in node.neighbors
        assert 3 in node.neighbors

        # Remove neighbor
        node.remove_neighbor(2)
        assert 2 not in node.neighbors
        assert 3 in node.neighbors

    def test_receive_state(self):
        """Test receiving neighbor state messages"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        # Add neighbor
        node.add_neighbor(2)

        # Create and receive state message
        neighbor_state = np.array([5.0, 10.0, 2.0, 0.05, 0.01])
        msg = StateMessage(
            node_id=2,
            state=neighbor_state,
            iteration=5,
            timestamp=time.time()
        )
        node.receive_state(msg)

        assert np.allclose(node.neighbor_states[2], neighbor_state)
        assert node.neighbor_timestamps[2] > 0

    def test_ignore_non_neighbor_messages(self):
        """Test that messages from non-neighbors are ignored"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        # Don't add node 2 as neighbor
        msg = StateMessage(
            node_id=2,
            state=np.ones(5),
            iteration=5,
            timestamp=time.time()
        )
        node.receive_state(msg)

        assert 2 not in node.neighbor_states

    def test_stale_message_rejection(self):
        """Test that stale messages are rejected"""
        config = ConsensusNodeConfig(node_id=1, max_stale_time=1.0)
        node = ConsensusNode(config, np.zeros(5))
        node.add_neighbor(2)

        # Create old message
        old_msg = StateMessage(
            node_id=2,
            state=np.ones(5),
            iteration=5,
            timestamp=time.time() - 2.0  # 2 seconds old
        )
        node.receive_state(old_msg)

        assert 2 not in node.neighbor_states or node.neighbor_states[2] is None

    def test_add_measurement(self):
        """Test adding measurement factors"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        # Add ToA factor
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=0.01)
        node.add_measurement(factor)

        assert len(node.local_factors) == 1
        assert node.local_factors[0] == factor

    def test_build_local_system(self):
        """Test building local linearized system"""
        config = ConsensusNodeConfig(node_id=1)
        initial_state = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        node = ConsensusNode(config, initial_state)

        # Add neighbor state (anchor at origin)
        node.add_neighbor(0)
        node.neighbor_states[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Add measurement from anchor
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=5.0, range_var_m2=0.01)
        node.add_measurement(factor)

        # Build system
        H, g = node.build_local_system()

        # Check dimensions
        assert H.shape == (5, 5)
        assert g.shape == (5,)

        # H should be non-zero (has information)
        assert np.linalg.norm(H) > 0

    def test_consensus_penalty(self):
        """Test consensus penalty addition"""
        config = ConsensusNodeConfig(node_id=1, consensus_gain=2.0)
        node = ConsensusNode(config, np.array([5.0, 5.0, 0.0, 0.0, 0.0]))

        # Add neighbors with states (asymmetric so gradient is non-zero)
        node.add_neighbor(2)
        node.add_neighbor(3)
        node.neighbor_states[2] = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
        node.neighbor_states[3] = np.array([3.0, 3.0, 0.0, 0.0, 0.0])
        node.neighbor_timestamps[2] = time.time()
        node.neighbor_timestamps[3] = time.time()

        # Initial system
        H = np.eye(5)
        g = np.zeros(5)

        # Add consensus
        H_new, g_new = node.add_consensus_penalty(H, g)

        # H should increase on diagonal (consensus regularization)
        # With gain=2.0 and 2 neighbors: H += 2.0 * 2 * I = 4*I
        # So diagonal should go from 1 to 5
        assert np.allclose(np.diag(H_new), 5.0)

        # g should be modified by neighbor states
        # Consensus term: -μ * Σ(x_j - x_i)
        # = -2.0 * ((2-5) + (3-5)) = -2.0 * (-3 + -2) = -2.0 * (-5) = 10
        expected_g_change = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
        assert np.allclose(g_new, expected_g_change)

    def test_compute_step(self):
        """Test step computation"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        # Simple system: gradient points toward [1,2,3,4,5]
        # In optimization, we move opposite to gradient: x -= step
        # So if gradient is negative, step should be negative (to increase x)
        H = np.eye(5)
        g = -np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        step = node.compute_step(H, g)

        # Step should be negative (same sign as gradient since we solve Hx=g)
        assert step.shape == (5,)
        # The step solves H*step = g, so step = g for identity H
        # With damping and scaling, should be close to g
        assert np.all(step < 0)  # Negative steps (same direction as negative gradient)

    def test_anchor_no_update(self):
        """Test that anchor nodes don't update their state"""
        config = ConsensusNodeConfig(node_id=0, is_anchor=True)
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        node = ConsensusNode(config, initial_state)

        # Add a measurement (shouldn't affect anchor)
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.0, range_var_m2=0.01)
        node.add_measurement(factor)

        # Update (should be no-op)
        node.update_state()

        assert np.allclose(node.state, initial_state)
        assert node.converged
        assert node.gradient_norm == 0.0
        assert node.step_norm == 0.0

    def test_state_update(self):
        """Test full state update iteration"""
        config = ConsensusNodeConfig(node_id=1, step_size=0.1)
        initial = np.array([3.0, 3.0, 0.0, 0.0, 0.0])
        node = ConsensusNode(config, initial)

        # Add anchor as neighbor
        node.add_neighbor(0)
        anchor_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        node.neighbor_states[0] = anchor_state

        # Add measurement placing node at (5, 0)
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=5.0, range_var_m2=0.01)
        node.add_measurement(factor)

        # Perform update
        old_state = node.state.copy()
        node.update_state()

        # State should change
        assert not np.allclose(node.state, old_state)

        # Should move toward correct position
        # Initial was (3,3), truth is (5,0), so x should increase, y decrease
        assert node.state[0] > old_state[0]  # x increases
        # Note: y behavior depends on measurement geometry

    def test_convergence_detection(self):
        """Test convergence detection"""
        config = ConsensusNodeConfig(
            node_id=1,
            gradient_tol=1e-4,
            step_tol=1e-6
        )
        node = ConsensusNode(config, np.zeros(5))

        # Simulate convergence
        node.gradient_norm = 1e-5  # Below threshold
        node.step_norm = 1e-7  # Below threshold

        # Need 3 consecutive iterations
        node._check_convergence()
        assert not node.converged  # First time

        node._check_convergence()
        assert not node.converged  # Second time

        node._check_convergence()
        assert node.converged  # Third time - converged!

        # Break convergence
        node.gradient_norm = 1e-3  # Above threshold
        node._check_convergence()
        assert not node.converged

    def test_get_state_message(self):
        """Test creating state message for broadcast"""
        config = ConsensusNodeConfig(node_id=1)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        node = ConsensusNode(config, state)
        node.iteration = 10

        msg = node.get_state_message()

        assert msg.node_id == 1
        assert np.allclose(msg.state, state)
        assert msg.iteration == 10
        assert msg.timestamp > 0

    def test_get_convergence_status(self):
        """Test getting convergence status"""
        config = ConsensusNodeConfig(node_id=1)
        node = ConsensusNode(config, np.zeros(5))

        node.iteration = 5
        node.converged = True
        node.gradient_norm = 1e-7
        node.step_norm = 1e-9
        node.cost = 10.0

        status = node.get_convergence_status()

        assert status.node_id == 1
        assert status.iteration == 5
        assert status.converged
        assert status.gradient_norm == 1e-7
        assert status.step_norm == 1e-9
        assert status.cost == 10.0

    def test_reset(self):
        """Test resetting node to initial state"""
        config = ConsensusNodeConfig(node_id=1)
        initial = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        node = ConsensusNode(config, initial)

        # Modify state
        node.state = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        node.iteration = 10
        node.converged = True
        node.add_neighbor(2)
        node.neighbor_states[2] = np.ones(5)

        # Reset
        node.reset()

        assert np.allclose(node.state, initial)
        assert node.iteration == 0
        assert not node.converged
        assert node.neighbor_states[2] is None

    def test_history_tracking(self):
        """Test that history is properly tracked"""
        config = ConsensusNodeConfig(node_id=1)
        initial = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        node = ConsensusNode(config, initial)

        # Add simple measurement
        node.add_neighbor(0)
        node.neighbor_states[0] = np.zeros(5)
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=5.0, range_var_m2=0.01)
        node.add_measurement(factor)

        # Perform updates
        for _ in range(3):
            node.update_state()

        assert len(node.state_history) == 4  # Initial + 3 updates
        assert len(node.cost_history) == 3
        assert len(node.gradient_history) == 3

        # History should show changes
        assert not np.allclose(node.state_history[0], node.state_history[-1])


class TestIntegration:
    """Integration tests for consensus node interactions"""

    def test_two_node_consensus(self):
        """Test consensus between two nodes"""
        # Create two nodes
        config1 = ConsensusNodeConfig(node_id=1, consensus_gain=0.5)
        config2 = ConsensusNodeConfig(node_id=2, consensus_gain=0.5)

        node1 = ConsensusNode(config1, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        node2 = ConsensusNode(config2, np.array([10.0, 10.0, 0.0, 0.0, 0.0]))

        # Make them neighbors
        node1.add_neighbor(2)
        node2.add_neighbor(1)

        # Exchange states
        msg1 = node1.get_state_message()
        msg2 = node2.get_state_message()

        node1.receive_state(msg2)
        node2.receive_state(msg1)

        # Update both
        old_distance = np.linalg.norm(node1.state[:2] - node2.state[:2])

        node1.update_state()
        node2.update_state()

        new_distance = np.linalg.norm(node1.state[:2] - node2.state[:2])

        # Nodes should move closer due to consensus
        assert new_distance < old_distance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])