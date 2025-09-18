"""
Unit tests for Consensus-Gauss-Newton algorithm
Test distributed optimization, network validation, and convergence
"""

import numpy as np
import pytest
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters, ClockPriorFactor


class TestConsensusGNConfig:
    """Test algorithm configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ConsensusGNConfig()

        assert config.max_iterations == 20
        assert config.consensus_gain == 1.0
        assert config.step_size == 0.5
        assert config.gradient_tol == 1e-6
        assert config.synchronous
        assert config.require_global_convergence

    def test_custom_config(self):
        """Test custom configuration"""
        config = ConsensusGNConfig(
            max_iterations=50,
            consensus_gain=2.0,
            synchronous=False
        )

        assert config.max_iterations == 50
        assert config.consensus_gain == 2.0
        assert not config.synchronous


class TestConsensusGaussNewton:
    """Test Consensus-GN algorithm"""

    def test_algorithm_creation(self):
        """Test creating algorithm instance"""
        cgn = ConsensusGaussNewton()

        assert len(cgn.nodes) == 0
        assert len(cgn.edges) == 0
        assert cgn.iteration == 0
        assert not cgn.converged

    def test_add_nodes(self):
        """Test adding nodes to network"""
        cgn = ConsensusGaussNewton()

        # Add anchor
        cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        assert 0 in cgn.nodes
        assert cgn.nodes[0].config.is_anchor

        # Add unknown
        cgn.add_node(1, np.array([5.0, 5.0, 0.0, 0.0, 0.0]), is_anchor=False)
        assert 1 in cgn.nodes
        assert not cgn.nodes[1].config.is_anchor

    def test_add_edges(self):
        """Test adding edges between nodes"""
        cgn = ConsensusGaussNewton()

        # Add nodes
        cgn.add_node(0, np.zeros(5))
        cgn.add_node(1, np.ones(5))

        # Add edge
        cgn.add_edge(0, 1)

        assert (0, 1) in cgn.edges
        assert 1 in cgn.nodes[0].neighbors
        assert 0 in cgn.nodes[1].neighbors

    def test_edge_requires_existing_nodes(self):
        """Test that edges require existing nodes"""
        cgn = ConsensusGaussNewton()
        cgn.add_node(0, np.zeros(5))

        with pytest.raises(ValueError, match="Both nodes must exist"):
            cgn.add_edge(0, 1)  # Node 1 doesn't exist

    def test_add_measurements(self):
        """Test adding measurements to network"""
        cgn = ConsensusGaussNewton()
        cgn.add_node(0, np.zeros(5))
        cgn.add_node(1, np.ones(5))

        # Add ToA measurement
        factor = ToAFactorMeters(0, 1, 5.0, 0.01)
        cgn.add_measurement(factor)

        assert len(cgn.measurements) == 1
        assert len(cgn.nodes[0].local_factors) == 1
        assert len(cgn.nodes[1].local_factors) == 1

    def test_network_validation(self):
        """Test network validation"""
        cgn = ConsensusGaussNewton()

        # Empty network
        is_valid, issues = cgn.validate_network()
        assert not is_valid
        assert "No anchor nodes" in str(issues)
        assert "No measurements" in str(issues)

        # Add disconnected nodes
        cgn.add_node(0, np.zeros(5), is_anchor=True)
        cgn.add_node(1, np.ones(5), is_anchor=False)

        is_valid, issues = cgn.validate_network()
        assert not is_valid
        assert "not fully connected" in str(issues)

        # Connect and add measurement
        cgn.add_edge(0, 1)
        cgn.add_measurement(ToAFactorMeters(0, 1, 5.0, 0.01))

        is_valid, issues = cgn.validate_network()
        assert is_valid
        assert len(issues) == 0

    def test_is_connected(self):
        """Test connectivity check"""
        cgn = ConsensusGaussNewton()

        # Single node is connected
        cgn.add_node(0, np.zeros(5))
        assert cgn._is_connected()

        # Two disconnected nodes
        cgn.add_node(1, np.ones(5))
        assert not cgn._is_connected()

        # Connect them
        cgn.add_edge(0, 1)
        assert cgn._is_connected()

        # Add third disconnected node
        cgn.add_node(2, np.ones(5) * 2)
        assert not cgn._is_connected()

        # Connect to network
        cgn.add_edge(1, 2)
        assert cgn._is_connected()

    def test_state_exchange(self):
        """Test state exchange between neighbors"""
        cgn = ConsensusGaussNewton()

        # Create simple network
        cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        cgn.add_node(1, np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
        cgn.add_node(2, np.array([2.5, 4.0, 0.0, 0.0, 0.0]))

        # Connect in triangle
        cgn.add_edge(0, 1)
        cgn.add_edge(1, 2)
        cgn.add_edge(2, 0)

        # Exchange states
        cgn._exchange_states()

        # Each node should have received neighbor states
        assert cgn.nodes[0].neighbor_states[1] is not None
        assert cgn.nodes[0].neighbor_states[2] is not None
        assert cgn.nodes[1].neighbor_states[0] is not None
        assert cgn.nodes[1].neighbor_states[2] is not None

        # Check message counting
        assert cgn.message_count == 3  # One per node
        assert cgn.total_bytes > 0

    def test_simple_convergence(self):
        """Test convergence on simple problem"""
        config = ConsensusGNConfig(max_iterations=50, verbose=False)
        cgn = ConsensusGaussNewton(config)

        # Two anchors
        cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)

        # One unknown (true position at (5, 0))
        cgn.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Connect all
        cgn.add_edge(0, 2)
        cgn.add_edge(1, 2)
        cgn.add_edge(0, 1)  # Anchors also connected

        # Add measurements
        cgn.add_measurement(ToAFactorMeters(0, 2, 5.0, 0.01))
        cgn.add_measurement(ToAFactorMeters(1, 2, 5.0, 0.01))

        # Set true positions for error computation
        cgn.set_true_positions({
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([5.0, 0.0])
        })

        # Optimize
        results = cgn.optimize()

        assert results['success']
        assert results['converged']
        assert results['iterations'] > 0
        assert results['iterations'] < 50

        # Check final position
        final_pos = results['final_states'][2][:2]
        error = np.linalg.norm(final_pos - np.array([5.0, 0.0]))
        assert error < 0.1  # Should be close to true position

        # Check error statistics
        assert 'position_errors' in results
        assert results['position_errors']['rmse'] < 0.1

    def test_multi_node_consensus(self):
        """Test consensus with multiple unknown nodes"""
        config = ConsensusGNConfig(max_iterations=50)
        cgn = ConsensusGaussNewton(config)

        # Create square network
        # Anchors at corners
        cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(2, np.array([10.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(3, np.array([0.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)

        # Unknown at center
        cgn.add_node(4, np.array([3.0, 7.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Connect in star topology
        for anchor in range(4):
            cgn.add_edge(anchor, 4)

        # Add measurements (true position at (5, 5))
        true_range = np.sqrt(50)  # Distance from corners to center
        for anchor in range(4):
            cgn.add_measurement(ToAFactorMeters(anchor, 4, true_range, 0.01))

        results = cgn.optimize()

        assert results['converged']
        final_pos = results['final_states'][4][:2]

        # Should converge near center
        error = np.linalg.norm(final_pos - np.array([5.0, 5.0]))
        assert error < 0.2

    def test_convergence_history(self):
        """Test that convergence history is tracked"""
        config = ConsensusGNConfig(max_iterations=20)
        cgn = ConsensusGaussNewton(config)

        # Simple network
        cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        cgn.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)

        cgn.add_edge(0, 2)
        cgn.add_edge(1, 2)

        cgn.add_measurement(ToAFactorMeters(0, 2, 5.0, 0.01))
        cgn.add_measurement(ToAFactorMeters(1, 2, 5.0, 0.01))

        results = cgn.optimize()

        # Check history
        assert len(cgn.convergence_history) > 0
        assert len(cgn.convergence_history) <= 20

        # History should show improvement
        first_cost = cgn.convergence_history[0]['total_cost']
        last_cost = cgn.convergence_history[-1]['total_cost']
        assert last_cost <= first_cost

        # Gradient should decrease
        first_grad = cgn.convergence_history[0]['max_gradient_norm']
        last_grad = cgn.convergence_history[-1]['max_gradient_norm']
        assert last_grad < first_grad

    def test_network_statistics(self):
        """Test network statistics computation"""
        cgn = ConsensusGaussNewton()

        # Build network
        cgn.add_node(0, np.zeros(5), is_anchor=True)
        cgn.add_node(1, np.ones(5), is_anchor=False)
        cgn.add_node(2, np.ones(5) * 2, is_anchor=False)

        cgn.add_edge(0, 1)
        cgn.add_edge(0, 2)

        cgn.add_measurement(ToAFactorMeters(0, 1, 5.0, 0.01))
        cgn.add_measurement(ToAFactorMeters(0, 2, 5.0, 0.01))

        stats = cgn.get_network_statistics()

        assert stats['n_nodes'] == 3
        assert stats['n_anchors'] == 1
        assert stats['n_unknowns'] == 2
        assert stats['n_edges'] == 2
        assert stats['n_measurements'] == 2
        assert stats['avg_degree'] == 4/3  # degrees: [2, 1, 1]
        assert stats['min_degree'] == 1
        assert stats['max_degree'] == 2
        assert stats['is_connected']

    def test_reset(self):
        """Test resetting algorithm"""
        cgn = ConsensusGaussNewton()

        # Build and run network
        cgn.add_node(0, np.zeros(5), is_anchor=True)
        cgn.add_node(1, np.ones(5), is_anchor=False)
        cgn.add_edge(0, 1)
        cgn.add_measurement(ToAFactorMeters(0, 1, 5.0, 0.01))

        results = cgn.optimize()
        assert cgn.iteration > 0
        assert cgn.message_count > 0

        # Reset
        cgn.reset()

        assert cgn.iteration == 0
        assert not cgn.converged
        assert len(cgn.convergence_history) == 0
        assert cgn.message_count == 0
        assert cgn.total_bytes == 0

        # Nodes should also be reset
        assert cgn.nodes[1].iteration == 0
        assert not cgn.nodes[1].converged

    def test_no_measurements_fails(self):
        """Test that optimization fails without measurements"""
        cgn = ConsensusGaussNewton()

        cgn.add_node(0, np.zeros(5), is_anchor=True)
        cgn.add_node(1, np.ones(5), is_anchor=False)
        cgn.add_edge(0, 1)

        # No measurements added

        results = cgn.optimize()

        assert not results['success']
        assert not results['converged']
        assert 'No measurements' in str(results['errors'])

    def test_disconnected_network_fails(self):
        """Test that optimization fails with disconnected network"""
        cgn = ConsensusGaussNewton()

        # Two disconnected components
        cgn.add_node(0, np.zeros(5), is_anchor=True)
        cgn.add_node(1, np.ones(5), is_anchor=False)
        cgn.add_edge(0, 1)

        cgn.add_node(2, np.array([10.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=False)
        # Node 2 not connected!

        cgn.add_measurement(ToAFactorMeters(0, 1, 5.0, 0.01))

        results = cgn.optimize()

        assert not results['success']
        assert 'not fully connected' in str(results['errors'])


class TestIntegration:
    """Integration tests for complete scenarios"""

    def test_consensus_vs_no_consensus(self):
        """Test that consensus improves estimates"""
        # Network with weak measurements
        config_no_consensus = ConsensusGNConfig(
            consensus_gain=0.0,  # No consensus
            max_iterations=30
        )
        cgn_no_consensus = ConsensusGaussNewton(config_no_consensus)

        config_with_consensus = ConsensusGNConfig(
            consensus_gain=1.0,  # With consensus
            max_iterations=30
        )
        cgn_with_consensus = ConsensusGaussNewton(config_with_consensus)

        # Build identical networks
        for cgn in [cgn_no_consensus, cgn_with_consensus]:
            # Anchors
            cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
            cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)

            # Unknowns in line
            cgn.add_node(2, np.array([2.0, 2.0, 0.0, 0.0, 0.0]))  # True: (3, 0)
            cgn.add_node(3, np.array([6.0, 2.0, 0.0, 0.0, 0.0]))  # True: (7, 0)

            # Connect in line: 0 -- 2 -- 3 -- 1
            cgn.add_edge(0, 2)
            cgn.add_edge(2, 3)
            cgn.add_edge(3, 1)

            # Weak measurements (large variance)
            cgn.add_measurement(ToAFactorMeters(0, 2, 3.0, 1.0))  # High variance
            cgn.add_measurement(ToAFactorMeters(1, 3, 3.0, 1.0))
            cgn.add_measurement(ToAFactorMeters(2, 3, 4.0, 1.0))

            cgn.set_true_positions({
                2: np.array([3.0, 0.0]),
                3: np.array([7.0, 0.0])
            })

        # Run both
        results_no_consensus = cgn_no_consensus.optimize()
        results_with_consensus = cgn_with_consensus.optimize()

        # Consensus should improve accuracy
        error_no_consensus = results_no_consensus.get('position_errors', {}).get('rmse', float('inf'))
        error_with_consensus = results_with_consensus.get('position_errors', {}).get('rmse', float('inf'))

        # With consensus should be better (or at least not worse)
        assert error_with_consensus <= error_no_consensus + 0.1  # Allow small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])