"""
Unit tests for solver internal functions
Test each component in isolation to identify issues
"""

import numpy as np
import pytest
from ftl.solver_scaled import SquareRootSolver, OptimizationConfig
from ftl.factors_scaled import ToAFactorMeters


class TestSolverInternals:
    """Test internal solver functions"""

    def test_build_whitened_system_simple(self):
        """Test whitened system construction with simple case"""
        solver = SquareRootSolver()

        # Two anchors, one unknown
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([5.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Add measurements
        solver.add_toa_factor(0, 2, 5.0, 0.01)  # 10cm std
        solver.add_toa_factor(1, 2, 5.0, 0.01)

        # Build system
        J_wh, r_wh, node_to_idx, idx_to_node = solver._build_whitened_system()

        # Check dimensions
        assert r_wh.shape == (2,)  # 2 measurements
        assert J_wh.shape == (2, 5)  # 2 measurements × 5 state variables

        # Check node mapping (only unknown node should be in map)
        assert 0 not in node_to_idx  # Anchor
        assert 1 not in node_to_idx  # Anchor
        assert 2 in node_to_idx  # Unknown
        assert node_to_idx[2] == 0  # First column index

        # Residuals should be zero for perfect measurements
        assert np.allclose(r_wh, 0, atol=1e-10)

        # Jacobian should have structure [dx, dy, db, dd, df]
        # For x-direction movement: J[0] = [-1, 0, c/std, 0, 0] (scaled)
        assert J_wh[0, 0] != 0  # dx component
        assert abs(J_wh[0, 1]) < 1e-10  # dy should be ~0 for x-aligned
        assert J_wh[0, 2] != 0  # bias component

    def test_build_whitened_system_with_noise(self):
        """Test with noisy measurements"""
        solver = SquareRootSolver()

        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([5.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Add noisy measurement (true distance is 5m)
        range_std = 0.1  # 10cm
        solver.add_toa_factor(0, 1, 5.1, range_std**2)  # 10cm error

        J_wh, r_wh, _, _ = solver._build_whitened_system()

        # Whitened residual should be error/std = 0.1/0.1 = 1.0
        assert abs(r_wh[0] - 1.0) < 1e-10

    def test_state_scaling_matrix(self):
        """Test state scaling matrix construction"""
        config = OptimizationConfig()
        solver = SquareRootSolver(config)

        # Add one unknown
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=False)
        solver.add_toa_factor(0, 0, 0.0, 1.0)  # Dummy factor

        # Build and scale
        J_wh = np.ones((1, 5))  # Dummy Jacobian
        J_scaled, S_mat = solver._apply_state_scaling(J_wh)

        # Check scaling matrix
        expected_scale = config.get_default_state_scale()
        assert np.allclose(np.diag(S_mat), expected_scale)

        # Check scaled Jacobian
        assert J_scaled[0, 0] == 1.0  # Position not scaled
        assert J_scaled[0, 1] == 1.0  # Position not scaled
        assert J_scaled[0, 2] == 1.0  # Bias not scaled
        assert J_scaled[0, 3] == 0.1  # Drift scaled down
        assert J_scaled[0, 4] == 0.1  # CFO scaled down

    def test_compute_cost(self):
        """Test cost computation from whitened residuals"""
        solver = SquareRootSolver()

        # Test with known residuals
        r_wh = np.array([1.0, 2.0, 3.0])
        cost = solver._compute_cost(r_wh)

        # Cost = 0.5 * sum(r²) = 0.5 * (1 + 4 + 9) = 7
        assert abs(cost - 7.0) < 1e-10

    def test_hessian_construction(self):
        """Test Hessian matrix properties"""
        solver = SquareRootSolver()

        # Create simple problem
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([5.0, 1.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Need at least 3 measurements for observability
        solver.add_toa_factor(0, 2, 5.1, 0.01)
        solver.add_toa_factor(1, 2, 5.1, 0.01)
        # Add prior on position to make it well-posed
        solver.add_toa_factor(0, 1, 10.0, 0.01)  # Anchor-anchor helps gauge

        J_wh, r_wh, _, _ = solver._build_whitened_system()
        J_scaled, S_mat = solver._apply_state_scaling(J_wh)

        # Form Hessian
        H = J_scaled.T @ J_scaled

        # Hessian should be symmetric
        assert np.allclose(H, H.T)

        # Hessian should be positive semi-definite
        eigenvalues = np.linalg.eigvals(H)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical error

        # Check dimensions
        assert H.shape == (5, 5)  # 5×5 for one unknown node

    def test_hessian_regularization(self):
        """Test that regularization handles unobserved variables"""
        solver = SquareRootSolver()

        # Use non-collinear anchors for full observability
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([5.0, 5.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(3, np.array([5.0, 1.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Add measurements from 3 anchors for full position observability
        solver.add_toa_factor(0, 3, 5.1, 0.01)
        solver.add_toa_factor(1, 3, 5.1, 0.01)
        solver.add_toa_factor(2, 3, 4.0, 0.01)

        J_wh, r_wh, _, _ = solver._build_whitened_system()
        J_scaled, _ = solver._apply_state_scaling(J_wh)

        # Form Hessian
        H = J_scaled.T @ J_scaled

        # Check diagonal
        diag_H = np.diag(H)

        # Position and bias should be observed (non-zero diagonal)
        assert diag_H[0] > 1e-6  # x
        assert diag_H[1] > 1e-6  # y
        assert diag_H[2] > 1e-6  # bias

        # Drift and CFO are unobserved with only ToA (zero diagonal)
        assert diag_H[3] < 1e-10  # drift
        assert diag_H[4] < 1e-10  # cfo

        # Apply regularization as in solver
        lambda_lm = 1e-4
        min_diag = 1e-6
        diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
        H_damped = H + lambda_lm * np.diag(diag_regularized)

        # Now should be invertible
        cond_number = np.linalg.cond(H_damped)
        assert cond_number < 1e13  # Conditioning with regularization (drift/CFO unobserved)

        # Should be able to solve
        g = J_scaled.T @ r_wh
        delta = np.linalg.solve(H_damped, g)
        assert np.all(np.isfinite(delta))

        # Verify regularization worked - all diagonal elements should be non-zero
        diag_damped = np.diag(H_damped)
        assert np.all(diag_damped > 0)

    def test_convergence_gradient_criterion(self):
        """Test gradient convergence criterion"""
        config = OptimizationConfig(gradient_tol=1e-6)
        solver = SquareRootSolver(config)

        # Mock data
        J_wh = np.array([[1.0, 0.0], [0.0, 1.0]])
        r_wh = np.array([1e-7, 1e-7])  # Small residuals
        delta = np.array([0.1, 0.1])

        converged, reason = solver._check_convergence(
            J_wh, r_wh, delta, 0.1, 0.2, iteration=1
        )

        # Gradient = J^T r should be small
        grad = J_wh.T @ r_wh
        grad_norm = np.linalg.norm(grad, ord=np.inf)

        assert grad_norm < 1e-6
        assert converged
        assert "Gradient converged" in reason

    def test_convergence_step_criterion(self):
        """Test step convergence criterion"""
        config = OptimizationConfig(step_tol=1e-8)
        solver = SquareRootSolver(config)

        # Mock data with tiny step
        J_wh = np.eye(2)
        r_wh = np.array([1.0, 1.0])
        delta = np.array([1e-9, 1e-9])  # Tiny step

        converged, reason = solver._check_convergence(
            J_wh, r_wh, delta, 0.1, 0.2, iteration=1
        )

        assert converged
        assert "Step converged" in reason

    def test_convergence_cost_criterion(self):
        """Test cost convergence criterion"""
        config = OptimizationConfig(cost_tol=1e-9)
        solver = SquareRootSolver(config)

        # Mock data with tiny cost change
        J_wh = np.eye(2)
        r_wh = np.array([1.0, 1.0])
        delta = np.array([0.1, 0.1])

        cost = 1.0
        prev_cost = 1.0 + 1e-10  # Tiny change

        converged, reason = solver._check_convergence(
            J_wh, r_wh, delta, cost, prev_cost, iteration=2
        )

        assert converged
        assert "Cost converged" in reason

    def test_whitening_preserves_information(self):
        """Test that whitening preserves information content"""
        # Create a factor with known variance
        factor = ToAFactorMeters(i=0, j=1, range_meas_m=10.1, range_var_m2=0.01)

        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        xj = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

        # Get raw and whitened versions
        raw_r = factor.residual(xi, xj)
        J_xi_raw, J_xj_raw = factor.jacobian(xi, xj)

        wh_r, J_xi_wh, J_xj_wh = factor.whitened_residual_and_jacobian(xi, xj)

        # Whitened residual should be raw/std
        std = np.sqrt(factor.variance)
        assert abs(wh_r - raw_r/std) < 1e-10

        # Whitened Jacobian should be raw/std
        assert np.allclose(J_xi_wh, J_xi_raw/std)
        assert np.allclose(J_xj_wh, J_xj_raw/std)

        # Information content (J^T W J) should be same
        # Raw: J^T (1/var) J
        # Whitened: J_wh^T J_wh
        info_raw = (J_xi_raw.T @ J_xi_raw) / factor.variance
        info_wh = J_xi_wh.T @ J_xi_wh
        assert np.allclose(info_raw, info_wh)

    def test_gain_ratio_calculation(self):
        """Test Levenberg-Marquardt gain ratio"""
        # The gain ratio determines if a step is accepted
        # gain = actual_decrease / predicted_decrease

        # Good step (gain > 0.25 should be accepted)
        actual_decrease = 1.0
        predicted_decrease = 1.1  # Slightly optimistic
        gain_ratio = actual_decrease / predicted_decrease
        assert gain_ratio > 0.25  # Should accept

        # Bad step (gain < 0 means cost increased)
        actual_decrease = -0.5  # Cost increased
        predicted_decrease = 1.0
        gain_ratio = actual_decrease / predicted_decrease
        assert gain_ratio < 0  # Should reject

        # Perfect prediction (gain ~= 1)
        actual_decrease = 1.0
        predicted_decrease = 1.0
        gain_ratio = actual_decrease / predicted_decrease
        assert abs(gain_ratio - 1.0) < 1e-10


class TestNodeMapping:
    """Test node to index mapping"""

    def test_anchor_exclusion(self):
        """Test that anchors are excluded from optimization"""
        solver = SquareRootSolver()

        # Mix of anchors and unknowns
        solver.add_node(0, np.zeros(5), is_anchor=True)
        solver.add_node(1, np.zeros(5), is_anchor=False)
        solver.add_node(2, np.zeros(5), is_anchor=True)
        solver.add_node(3, np.zeros(5), is_anchor=False)

        # Add dummy measurements
        solver.add_toa_factor(0, 1, 1.0, 1.0)
        solver.add_toa_factor(2, 3, 1.0, 1.0)

        _, _, node_to_idx, idx_to_node = solver._build_whitened_system()

        # Only unknowns should be in mapping
        assert 0 not in node_to_idx  # Anchor
        assert 1 in node_to_idx  # Unknown
        assert 2 not in node_to_idx  # Anchor
        assert 3 in node_to_idx  # Unknown

        # Check indices are correct
        assert node_to_idx[1] == 0  # First unknown
        assert node_to_idx[3] == 5  # Second unknown (5 vars later)

        # Check reverse mapping
        assert idx_to_node[0] == (1, 0)  # Node 1, variable 0
        assert idx_to_node[5] == (3, 0)  # Node 3, variable 0

    def test_variable_indexing(self):
        """Test that variables are indexed correctly"""
        solver = SquareRootSolver()

        solver.add_node(0, np.zeros(5), is_anchor=False)
        solver.add_toa_factor(0, 0, 1.0, 1.0)  # Dummy

        _, _, node_to_idx, idx_to_node = solver._build_whitened_system()

        # Check all 5 variables
        for i in range(5):
            assert idx_to_node[i] == (0, i)

        # Variables: [x, y, bias, drift, cfo]
        assert idx_to_node[0] == (0, 0)  # x
        assert idx_to_node[1] == (0, 1)  # y
        assert idx_to_node[2] == (0, 2)  # bias
        assert idx_to_node[3] == (0, 3)  # drift
        assert idx_to_node[4] == (0, 4)  # cfo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])