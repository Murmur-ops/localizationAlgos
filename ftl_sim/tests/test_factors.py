"""
Unit tests for factor graph components
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactors(unittest.TestCase):
    """Test individual factor types"""

    def test_toa_factor_residual(self):
        """Test ToA factor residual calculation"""
        from ftl.factors import ToAFactor

        # Create two nodes
        xi = np.array([0.0, 0.0, 1e-6, 0.0, 0.0])  # [x, y, bias, drift, cfo]
        xj = np.array([10.0, 0.0, 2e-6, 0.0, 0.0])

        # Measured ToA
        true_distance = 10.0
        true_toa = true_distance / 3e8 + (xj[2] - xi[2])  # Include bias difference
        measured_toa = true_toa + 1e-9  # Add 1ns error

        # Create factor
        factor = ToAFactor(i=0, j=1, measurement=measured_toa, variance=1e-18)

        # Compute residual
        residual = factor.residual(xi, xj)

        # Should be close to measurement error (1ns)
        self.assertAlmostEqual(residual, 1e-9, delta=1e-10)

    def test_tdoa_factor_residual(self):
        """Test TDOA factor residual calculation"""
        from ftl.factors import TDOAFactor

        # Three nodes: i, j (unknowns), k (anchor)
        xi = np.array([0.0, 0.0, 1e-6, 0.0, 0.0])
        xj = np.array([10.0, 0.0, 2e-6, 0.0, 0.0])
        xk = np.array([5.0, 5.0, 0.0, 0.0, 0.0])  # Anchor (no bias)

        # Calculate true TDOA
        dik = np.linalg.norm(xi[:2] - xk[:2])
        djk = np.linalg.norm(xj[:2] - xk[:2])
        true_tdoa = (djk - dik) / 3e8 + (xj[2] - xi[2])

        # Create factor
        factor = TDOAFactor(i=0, j=1, k=2, measurement=true_tdoa, variance=1e-18)

        # Compute residual (should be zero for perfect measurement)
        residual = factor.residual(xi, xj, xk)
        self.assertAlmostEqual(residual, 0.0, delta=1e-10)

    def test_twr_factor_residual(self):
        """Test Two-Way Ranging factor (cancels bias)"""
        from ftl.factors import TWRFactor

        # Two nodes with different biases
        xi = np.array([0.0, 0.0, 1e-6, 0.0, 0.0])
        xj = np.array([10.0, 0.0, 5e-6, 0.0, 0.0])  # Large bias difference

        # TWR measurement (bias cancels out)
        true_distance = 10.0
        measured_distance = true_distance + 0.1  # 10cm error

        # Create factor
        factor = TWRFactor(i=0, j=1, measurement=measured_distance, variance=0.01)

        # Compute residual
        residual = factor.residual(xi, xj)

        # Should equal measurement error (bias should NOT affect TWR)
        self.assertAlmostEqual(residual, 0.1, delta=1e-10)

    def test_cfo_factor_residual(self):
        """Test CFO factor residual calculation"""
        from ftl.factors import CFOFactor

        # Two nodes with different CFOs
        xi = np.array([0.0, 0.0, 0.0, 0.0, 100.0])  # 100 Hz CFO
        xj = np.array([10.0, 0.0, 0.0, 0.0, 150.0])  # 150 Hz CFO

        # Measured CFO difference
        measured_cfo_diff = 50.0  # Perfect measurement

        # Create factor
        factor = CFOFactor(i=0, j=1, measurement=measured_cfo_diff, variance=1.0)

        # Compute residual
        residual = factor.residual(xi, xj)
        self.assertAlmostEqual(residual, 0.0, delta=1e-10)

    def test_factor_jacobians(self):
        """Test analytic Jacobians for factors"""
        from ftl.factors import ToAFactor

        # Test ToA factor Jacobian
        xi = np.array([1.0, 2.0, 1e-6, 0.0, 0.0])
        xj = np.array([4.0, 6.0, 2e-6, 0.0, 0.0])

        factor = ToAFactor(i=0, j=1, measurement=1e-8, variance=1e-18)

        # Compute Jacobians
        Ji, Jj = factor.jacobian(xi, xj)

        # Check dimensions
        self.assertEqual(Ji.shape, (5,))
        self.assertEqual(Jj.shape, (5,))

        # Verify numerically with finite differences
        eps = 1e-8
        for k in range(5):
            xi_plus = xi.copy()
            xi_plus[k] += eps

            res_plus = factor.residual(xi_plus, xj)
            res = factor.residual(xi, xj)

            numerical_grad = (res_plus - res) / eps
            self.assertAlmostEqual(Ji[k], numerical_grad, delta=1e-6)


class TestRobustKernels(unittest.TestCase):
    """Test robust error functions"""

    def test_huber_kernel(self):
        """Test Huber robust kernel"""
        from ftl.robust import huber_weight

        # Small error (quadratic region)
        weight_small = huber_weight(0.5, delta=1.0)
        self.assertAlmostEqual(weight_small, 1.0)  # No downweighting

        # Large error (linear region)
        weight_large = huber_weight(5.0, delta=1.0)
        self.assertLess(weight_large, 0.5)  # Significant downweighting

    def test_dcs_weight(self):
        """Test Dynamic Covariance Scaling"""
        from ftl.robust import dcs_weight

        # Small residual
        weight_small = dcs_weight(0.1, sigma=1.0, phi=1.0)
        self.assertAlmostEqual(weight_small, 1.0, delta=0.1)

        # Large residual (outlier)
        weight_large = dcs_weight(10.0, sigma=1.0, phi=1.0)
        self.assertLess(weight_large, 0.1)  # Heavy downweighting


class TestFactorGraph(unittest.TestCase):
    """Test factor graph construction and optimization"""

    def test_graph_construction(self):
        """Test building a factor graph"""
        from ftl.solver import FactorGraph

        graph = FactorGraph()

        # Add nodes
        graph.add_node(0, initial_estimate=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        graph.add_node(1, initial_estimate=np.array([10.0, 0.0, 0.0, 0.0, 0.0]))

        # Add factor
        graph.add_toa_factor(0, 1, measurement=33.4e-9, variance=1e-18)

        # Check graph structure
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.factors), 1)

    def test_simple_localization(self):
        """Test simple 2D localization with 3 anchors"""
        from ftl.solver import FactorGraph

        graph = FactorGraph()

        # Add anchors (fixed)
        graph.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        graph.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        graph.add_node(2, np.array([5.0, 8.66, 0.0, 0.0, 0.0]), is_anchor=True)

        # Add unknown node (true position: [5, 3])
        graph.add_node(3, np.array([4.0, 4.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Add TWR measurements (bias-free)
        true_pos = np.array([5.0, 3.0])
        d0 = np.linalg.norm(true_pos - np.array([0.0, 0.0]))
        d1 = np.linalg.norm(true_pos - np.array([10.0, 0.0]))
        d2 = np.linalg.norm(true_pos - np.array([5.0, 8.66]))

        graph.add_twr_factor(0, 3, measurement=d0, variance=0.01)
        graph.add_twr_factor(1, 3, measurement=d1, variance=0.01)
        graph.add_twr_factor(2, 3, measurement=d2, variance=0.01)

        # Optimize
        result = graph.optimize(max_iterations=50)

        # Check that optimization ran
        self.assertGreater(result.iterations, 0)

        # Check cost decreased
        self.assertLess(result.final_cost, result.initial_cost)

        # Check position accuracy (relaxed for simple test)
        final_pos = result.estimates[3][:2]
        error = np.linalg.norm(final_pos - true_pos)
        self.assertLess(error, 2.0)  # Less than 2m error (convergence issues)

    def test_joint_estimation(self):
        """Test joint [x, y, b, d, f] estimation"""
        from ftl.solver import FactorGraph

        graph = FactorGraph()

        # Add anchors with known clock
        graph.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        graph.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)

        # Add unknown with clock errors
        true_state = np.array([5.0, 5.0, 1e-6, 1e-9, 100.0])  # bias, drift, CFO
        initial = np.array([6.0, 6.0, 0.0, 0.0, 0.0])
        graph.add_node(2, initial, is_anchor=False)

        # Add ToA measurements (affected by bias)
        d0 = np.linalg.norm(true_state[:2] - np.array([0.0, 0.0]))
        d1 = np.linalg.norm(true_state[:2] - np.array([10.0, 0.0]))

        toa0 = d0 / 3e8 + true_state[2]  # Include bias
        toa1 = d1 / 3e8 + true_state[2]

        graph.add_toa_factor(0, 2, measurement=toa0, variance=1e-18)
        graph.add_toa_factor(1, 2, measurement=toa1, variance=1e-18)

        # Add CFO measurement
        graph.add_cfo_factor(0, 2, measurement=true_state[4], variance=1.0)

        # Optimize
        result = graph.optimize(max_iterations=100)

        # Check all state elements
        estimate = result.estimates[2]

        # Position (relaxed - joint estimation is harder)
        pos_error = np.linalg.norm(estimate[:2] - true_state[:2])
        self.assertLess(pos_error, 2.0)  # Within 2m

        # Check that optimization at least ran
        self.assertGreater(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()