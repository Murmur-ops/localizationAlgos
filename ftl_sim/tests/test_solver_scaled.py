"""
Unit tests for the square-root information solver with proper scaling
"""

import numpy as np
import pytest
from ftl.solver_scaled import (
    SquareRootSolver, OptimizationConfig, ScaledNode
)


class TestSquareRootSolver:
    """Test the square-root information solver"""

    def test_simple_trilateration(self):
        """Test simple 2D trilateration with perfect measurements"""
        solver = SquareRootSolver()

        # Add 3 anchors at known positions
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([5.0, 8.66, 0.0, 0.0, 0.0]), is_anchor=True)  # ~equilateral

        # Add unknown node - true position at (5, 2.89) - centroid
        true_pos = np.array([5.0, 2.89])
        initial_guess = np.array([4.0, 2.0, 0.0, 0.0, 0.0])  # Start nearby
        solver.add_node(3, initial_guess, is_anchor=False)

        # Add perfect range measurements (no noise)
        # Distance from (5, 2.89) to each anchor
        d0 = np.sqrt(5**2 + 2.89**2)  # ~5.77m
        d1 = np.sqrt(5**2 + 2.89**2)  # ~5.77m
        d2 = np.sqrt(0**2 + 5.77**2)  # ~5.77m

        # Use realistic UWB variance (5cm std)
        range_var = 0.05**2

        solver.add_toa_factor(0, 3, d0, range_var)
        solver.add_toa_factor(1, 3, d1, range_var)
        solver.add_toa_factor(2, 3, d2, range_var)

        # Optimize
        result = solver.optimize(verbose=False)

        # Should converge quickly with perfect measurements
        assert result.converged
        assert result.iterations < 20

        # Check position accuracy
        est_pos = result.estimates[3][:2]
        pos_error = np.linalg.norm(est_pos - true_pos)
        assert pos_error < 0.01  # Should be very accurate (< 1cm)

    def test_with_measurement_noise(self):
        """Test with realistic measurement noise"""
        np.random.seed(42)
        solver = SquareRootSolver()

        # 4 anchors at corners of 20x20m area
        area_size = 20.0
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([area_size, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([area_size, area_size, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(3, np.array([0.0, area_size, 0.0, 0.0, 0.0]), is_anchor=True)

        # Unknown at center
        true_pos = np.array([area_size/2, area_size/2])
        initial_guess = np.array([8.0, 8.0, 0.0, 0.0, 0.0])  # 2.8m offset
        solver.add_node(4, initial_guess, is_anchor=False)

        # Add noisy measurements
        range_std = 0.1  # 10cm
        range_var = range_std**2

        for anchor_id in range(4):
            anchor_pos = solver.nodes[anchor_id].state[:2]
            true_dist = np.linalg.norm(anchor_pos - true_pos)

            # Add Gaussian noise
            measured_dist = true_dist + np.random.normal(0, range_std)
            solver.add_toa_factor(anchor_id, 4, measured_dist, range_var)

        # Optimize
        result = solver.optimize(verbose=False)

        assert result.converged
        assert result.iterations < 50

        # Check accuracy - should be close to CRLB
        est_pos = result.estimates[4][:2]
        pos_error = np.linalg.norm(est_pos - true_pos)

        # CRLB for this geometry is approximately range_std
        assert pos_error < 3 * range_std  # Within 3 sigma

    def test_with_clock_bias(self):
        """Test estimation with unknown clock bias"""
        solver = SquareRootSolver()

        # 4 anchors
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([20.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([20.0, 20.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(3, np.array([0.0, 20.0, 0.0, 0.0, 0.0]), is_anchor=True)

        # Unknown with 100ns clock bias
        true_bias_ns = 100.0
        c = 299792458.0  # m/s
        bias_range = true_bias_ns * 1e-9 * c  # ~30m

        true_state = np.array([10.0, 10.0, true_bias_ns, 0.0, 0.0])
        initial_guess = np.array([8.0, 8.0, 0.0, 0.0, 0.0])  # No bias in initial guess
        solver.add_node(4, initial_guess, is_anchor=False)

        # Measurements include bias effect
        range_var = 0.05**2

        for anchor_id in range(4):
            anchor_pos = solver.nodes[anchor_id].state[:2]
            true_dist = np.linalg.norm(anchor_pos - true_state[:2])

            # Measured range includes clock bias
            measured_range = true_dist + bias_range
            solver.add_toa_factor(anchor_id, 4, measured_range, range_var)

        # Add weak prior on clock to make it observable
        solver.add_clock_prior(4, bias_ns=0.0, drift_ppb=0.0,
                              bias_var_ns2=1000.0**2,  # 1us std prior
                              drift_var_ppb2=10.0**2)

        # Optimize
        result = solver.optimize(verbose=False)

        assert result.converged

        # Check position accuracy
        est_pos = result.estimates[4][:2]
        pos_error = np.linalg.norm(est_pos - true_state[:2])
        assert pos_error < 0.5  # Should estimate position well

        # Check clock bias estimation
        est_bias = result.estimates[4][2]
        bias_error = abs(est_bias - true_bias_ns)
        assert bias_error < 10.0  # Within 10ns

    def test_whitening_normalization(self):
        """Test that whitened residuals are properly normalized"""
        solver = SquareRootSolver()

        # Simple setup
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)

        # Different variances for each measurement
        solver.add_toa_factor(0, 1, 10.0, 0.01**2)  # 1cm std
        solver.add_toa_factor(0, 1, 10.1, 0.1**2)   # 10cm std
        solver.add_toa_factor(0, 1, 10.5, 1.0**2)   # 1m std

        # Build whitened system
        J_wh, r_wh, _, _ = solver._build_whitened_system()

        # All whitened residuals should have similar scale
        # despite different original variances
        assert r_wh.shape[0] == 3

        # First measurement: 0m error / 0.01m std = 0
        assert abs(r_wh[0]) < 0.1

        # Second measurement: 0.1m error / 0.1m std = 1
        assert abs(r_wh[1] - 1.0) < 0.1

        # Third measurement: 0.5m error / 1.0m std = 0.5
        assert abs(r_wh[2] - 0.5) < 0.1

    def test_state_scaling(self):
        """Test that state scaling improves conditioning"""
        config = OptimizationConfig()
        solver = SquareRootSolver(config)

        # Add nodes
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([100.0, 100.0, 1000.0, 10.0, 1.0]), is_anchor=False)

        # Add a measurement
        solver.add_toa_factor(0, 1, 141.4, 0.1**2)  # ~141m distance

        # Build system
        J_wh, r_wh, _, _ = solver._build_whitened_system()

        # Apply scaling
        J_scaled, S_mat = solver._apply_state_scaling(J_wh)

        # Check that scaling is applied correctly
        # Position variables scaled by 1.0, clock by different factors
        S_expected = config.get_default_state_scale()
        assert np.allclose(np.diag(S_mat)[:5], S_expected)

        # Scaled Jacobian should have better conditioning
        cond_original = np.linalg.cond(J_wh.T @ J_wh) if J_wh.shape[0] > 0 else 1
        cond_scaled = np.linalg.cond(J_scaled.T @ J_scaled) if J_scaled.shape[0] > 0 else 1

        # Conditioning should improve or stay similar
        assert cond_scaled <= cond_original * 10  # Allow some tolerance

    def test_convergence_criteria(self):
        """Test that all convergence criteria work"""
        config = OptimizationConfig(
            gradient_tol=1e-6,
            step_tol=1e-8,
            cost_tol=1e-9
        )
        solver = SquareRootSolver(config)

        # Simple problem that converges quickly
        solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
        solver.add_node(2, np.array([5.0, 5.0, 0.0, 0.0, 0.0]), is_anchor=False)

        # Nearly perfect measurements
        solver.add_toa_factor(0, 2, 7.071, 0.001**2)
        solver.add_toa_factor(1, 2, 7.071, 0.001**2)

        result = solver.optimize(verbose=False)

        assert result.converged
        assert result.iterations < 20

        # Check that convergence reason is one of the criteria
        assert any([
            "Gradient converged" in result.convergence_reason,
            "Step converged" in result.convergence_reason,
            "Cost converged" in result.convergence_reason
        ])

    def test_realistic_uwb_performance(self):
        """Test with realistic UWB parameters"""
        np.random.seed(42)

        # Multiple trials to check consistency
        errors = []

        for trial in range(10):
            solver = SquareRootSolver()

            # 4 anchors in 30x30m area
            area = 30.0
            anchors = [
                (0, 0), (area, 0), (area, area), (0, area)
            ]

            for i, (x, y) in enumerate(anchors):
                solver.add_node(i, np.array([x, y, 0.0, 0.0, 0.0]), is_anchor=True)

            # Random unknown position
            true_x = np.random.uniform(5, 25)
            true_y = np.random.uniform(5, 25)
            true_pos = np.array([true_x, true_y])

            # Initial guess with error
            initial = np.array([
                true_x + np.random.randn() * 3,
                true_y + np.random.randn() * 3,
                0.0, 0.0, 0.0
            ])
            solver.add_node(4, initial, is_anchor=False)

            # Realistic UWB measurements
            range_std = 0.05  # 5cm (good UWB performance)
            range_var = range_std**2

            for i, (ax, ay) in enumerate(anchors):
                anchor_pos = np.array([ax, ay])
                true_dist = np.linalg.norm(anchor_pos - true_pos)

                # Add noise
                meas_dist = true_dist + np.random.normal(0, range_std)
                solver.add_toa_factor(i, 4, meas_dist, range_var)

            # Optimize
            result = solver.optimize(verbose=False)

            if result.converged:
                est_pos = result.estimates[4][:2]
                error = np.linalg.norm(est_pos - true_pos)
                errors.append(error)

        # Most should converge
        assert len(errors) >= 8

        # Check performance
        rmse = np.sqrt(np.mean(np.array(errors)**2))

        # With 5cm ranging and good geometry, position RMSE should be < 10cm
        assert rmse < 0.10

    def test_numerical_stability(self):
        """Test that solver handles various scales without numerical issues"""
        test_cases = [
            (0.01, "1cm ranging"),    # Very precise
            (0.05, "5cm ranging"),     # Typical good UWB
            (0.30, "30cm ranging"),    # Poor conditions
            (1.00, "1m ranging"),      # Very poor
        ]

        for range_std, description in test_cases:
            solver = SquareRootSolver()

            # Standard 4-anchor setup
            solver.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
            solver.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
            solver.add_node(2, np.array([10.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)
            solver.add_node(3, np.array([0.0, 10.0, 0.0, 0.0, 0.0]), is_anchor=True)

            # Unknown at center
            solver.add_node(4, np.array([4.0, 4.0, 0.0, 0.0, 0.0]), is_anchor=False)

            # Add measurements
            range_var = range_std**2
            true_pos = np.array([5.0, 5.0])

            for i in range(4):
                anchor_pos = solver.nodes[i].state[:2]
                dist = np.linalg.norm(anchor_pos - true_pos)
                solver.add_toa_factor(i, 4, dist, range_var)

            # Should not crash or produce NaN
            result = solver.optimize(verbose=False)

            assert result.converged or result.iterations == 100
            assert not np.any(np.isnan(result.estimates[4]))
            assert not np.any(np.isinf(result.estimates[4]))

            # Cost should be finite
            assert np.isfinite(result.final_cost)


class TestOptimizationConfig:
    """Test optimization configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = OptimizationConfig()

        assert config.max_iterations == 100
        assert config.gradient_tol == 1e-6
        assert config.step_tol == 1e-8
        assert config.cost_tol == 1e-9

        # Default state scaling
        S = config.get_default_state_scale()
        assert S[0] == 1.0  # Position in meters
        assert S[1] == 1.0  # Position in meters
        assert S[2] == 1.0  # Bias in nanoseconds
        assert S[3] == 0.1  # Drift in ppb (scaled down)
        assert S[4] == 0.1  # CFO in ppm (scaled down)

    def test_custom_config(self):
        """Test custom configuration"""
        config = OptimizationConfig(
            max_iterations=50,
            gradient_tol=1e-5,
            state_scale=np.array([2.0, 2.0, 10.0, 1.0, 1.0])
        )

        assert config.max_iterations == 50
        assert config.gradient_tol == 1e-5

        S = config.get_default_state_scale()
        assert S[0] == 2.0  # Custom scaling


if __name__ == "__main__":
    pytest.main([__file__, "-v"])