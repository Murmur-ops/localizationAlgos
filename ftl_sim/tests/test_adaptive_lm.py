"""
Unit tests for adaptive Levenberg-Marquardt optimization
"""

import numpy as np
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.optimization.adaptive_lm import AdaptiveLM, AdaptiveLMConfig


class TestAdaptiveLM(unittest.TestCase):
    """Test suite for adaptive Levenberg-Marquardt optimizer"""

    def test_quadratic_function(self):
        """Test optimization of simple quadratic function"""
        # f(x) = 0.5 * x^T * A * x - b^T * x
        # Minimum at x = A^(-1) * b

        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        x_true = np.linalg.solve(A, b)

        def cost_fn(x):
            return 0.5 * x.T @ A @ x - b.T @ x

        def gradient_fn(x):
            return A @ x - b

        def hessian_fn(x):
            return A

        # Initialize far from solution
        x_init = np.array([10.0, 10.0])

        # Create optimizer
        config = AdaptiveLMConfig(verbose=False)
        optimizer = AdaptiveLM(config)

        # Optimize
        x_opt, info = optimizer.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        # Check solution
        self.assertLess(np.linalg.norm(x_opt - x_true), 1e-6,
                       f"Failed to find minimum. Got {x_opt}, expected {x_true}")

        # Check convergence
        self.assertLess(info['final_cost'], 1e-10,
                       f"Final cost too high: {info['final_cost']}")

        print(f"✓ Quadratic: converged in {info['iterations']} iterations")

    def test_rosenbrock_function(self):
        """Test on non-convex Rosenbrock function"""
        # f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        # Global minimum at (1, 1)

        def cost_fn(x):
            return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

        def gradient_fn(x):
            g = np.zeros(2)
            g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
            g[1] = 200*(x[1] - x[0]**2)
            return g

        def hessian_fn(x):
            H = np.zeros((2, 2))
            H[0, 0] = 2 + 800*x[0]**2 - 400*(x[1] - x[0]**2)
            H[0, 1] = -400*x[0]
            H[1, 0] = -400*x[0]
            H[1, 1] = 200
            return H

        # Start from difficult point
        x_init = np.array([-2.0, 2.0])

        config = AdaptiveLMConfig(
            initial_lambda=0.01,
            max_iterations=200,
            verbose=False
        )
        optimizer = AdaptiveLM(config)

        x_opt, info = optimizer.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        # Check if close to global minimum
        x_true = np.array([1.0, 1.0])
        error = np.linalg.norm(x_opt - x_true)
        self.assertLess(error, 0.01,
                       f"Rosenbrock: Failed to converge. Error = {error}")

        print(f"✓ Rosenbrock: converged to within {error:.2e} in {info['iterations']} iterations")

    def test_damping_adaptation(self):
        """Test that damping parameter adapts correctly"""
        # Test that lambda increases when steps fail
        def cost_with_barrier(x):
            # Function with a valley that requires careful navigation
            if x[0] > 5:
                return 1e10  # Barrier
            return (x[0] - 1)**2 + 100*(x[1] - x[0]**2)**2

        def grad_with_barrier(x):
            if x[0] > 5:
                return np.array([1e10, 0])
            g = np.zeros(2)
            g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
            g[1] = 200*(x[1] - x[0]**2)
            return g

        def hess_with_barrier(x):
            if x[0] > 5:
                return np.eye(2) * 1e10
            H = np.zeros((2, 2))
            H[0, 0] = 2 + 800*x[0]**2 - 400*(x[1] - x[0]**2)
            H[0, 1] = -400*x[0]
            H[1, 0] = -400*x[0]
            H[1, 1] = 200
            return H

        # Start near barrier
        x_init = np.array([4.5, 20.0])

        config = AdaptiveLMConfig(initial_lambda=0.001, verbose=False)
        optimizer = AdaptiveLM(config)

        # Track lambda values during optimization
        initial_lambda = optimizer.lambda_current
        x_opt, info = optimizer.optimize(x_init, grad_with_barrier,
                                        hess_with_barrier, cost_with_barrier)

        # Check that lambda adapted during optimization
        lambda_history = info['lambda_history']
        max_lambda = max(lambda_history)
        min_lambda = min(lambda_history)

        # Lambda should have varied significantly
        self.assertGreater(max_lambda / min_lambda, 10,
                          "Lambda should vary by at least 10x during optimization")

        print(f"✓ Lambda adaptation: varied from {min_lambda:.2e} to {max_lambda:.2e} "
              f"(ratio: {max_lambda/min_lambda:.1f}x)")

        # Test that lambda decreases for simple problem
        def simple_cost(x):
            return np.sum(x**2)

        def simple_grad(x):
            return 2*x

        def simple_hess(x):
            return 2*np.eye(len(x))

        config2 = AdaptiveLMConfig(initial_lambda=10.0, max_iterations=10, verbose=False)
        optimizer2 = AdaptiveLM(config2)

        x_init2 = np.array([1.0, 1.0])
        x_opt2, info2 = optimizer2.optimize(x_init2, simple_grad, simple_hess, simple_cost)

        # Lambda should decrease for simple quadratic (or converge immediately)
        # If it converged in 1-2 iterations, lambda might not have time to adapt
        if info2['iterations'] > 2:
            self.assertLess(info2['final_lambda'], config2.initial_lambda,
                           "Lambda should decrease for simple problem")
        else:
            # Converged too fast to adapt lambda
            pass

        print(f"✓ Simple problem: converged in {info2['iterations']} iters, "
              f"λ: {config2.initial_lambda} -> {info2['final_lambda']:.2e}")

    def test_step_acceptance_rejection(self):
        """Test step acceptance/rejection logic"""
        # Create a function with a barrier
        def cost_fn(x):
            if x[0] < 0:  # Barrier at x=0
                return 1e10
            return x[0]**2 + x[1]**2

        def gradient_fn(x):
            if x[0] < 0:
                return np.array([1e10, 2*x[1]])
            return 2*x

        def hessian_fn(x):
            return 2*np.eye(2)

        config = AdaptiveLMConfig(initial_lambda=0.01, verbose=False)
        optimizer = AdaptiveLM(config)

        # Start near barrier
        x_current = np.array([0.1, 1.0])
        H = hessian_fn(x_current)
        g = gradient_fn(x_current)

        # First step might hit barrier
        x_new, cost, converged = optimizer.step(x_current, H, g, cost_fn)

        # Check that optimizer handles barrier correctly
        self.assertGreaterEqual(x_new[0], 0, "Should not cross barrier")
        self.assertTrue(len(optimizer.success_history) > 0, "Should have recorded step outcome")

        print(f"✓ Barrier handling: stayed at x={x_new[0]:.3f} ≥ 0")

    def test_convergence_criteria(self):
        """Test different convergence criteria"""
        # Simple quadratic
        def cost_fn(x):
            return np.sum(x**2)

        def gradient_fn(x):
            return 2*x

        def hessian_fn(x):
            return 2*np.eye(len(x))

        # Test gradient tolerance
        config1 = AdaptiveLMConfig(gradient_tol=1e-4, step_tol=0, verbose=False)
        optimizer1 = AdaptiveLM(config1)
        x_init = np.array([1.0, 1.0])
        x_opt1, info1 = optimizer1.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        final_grad = gradient_fn(x_opt1)
        self.assertLess(np.linalg.norm(final_grad), 1e-3,
                       "Should converge when gradient is small")

        print(f"✓ Gradient convergence: ||g|| = {np.linalg.norm(final_grad):.2e}")

        # Test step tolerance
        config2 = AdaptiveLMConfig(gradient_tol=0, step_tol=1e-8, verbose=False)
        optimizer2 = AdaptiveLM(config2)
        x_opt2, info2 = optimizer2.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        # Should converge when steps become small
        self.assertLess(info2['final_cost'], 1e-6,
                       "Should converge to low cost")

        print(f"✓ Step convergence: final cost = {info2['final_cost']:.2e}")

    def test_cost_history(self):
        """Test that cost decreases monotonically with adaptive LM"""
        # Convex function
        def cost_fn(x):
            return np.sum(x**4) + 2*np.sum(x**2)

        def gradient_fn(x):
            return 4*x**3 + 4*x

        def hessian_fn(x):
            return np.diag(12*x**2 + 4)

        config = AdaptiveLMConfig(verbose=False)
        optimizer = AdaptiveLM(config)

        x_init = np.array([2.0, -3.0, 1.5])
        x_opt, info = optimizer.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        # Check monotonic decrease
        costs = info['cost_history']
        for i in range(1, len(costs)):
            self.assertLessEqual(costs[i], costs[i-1] + 1e-10,
                               f"Cost increased at iteration {i}: {costs[i-1]} -> {costs[i]}")

        print(f"✓ Monotonic decrease: {len(costs)} steps, all decreasing")
        print(f"  Initial cost: {costs[0]:.2e}, Final: {costs[-1]:.2e}")

    def test_singular_hessian(self):
        """Test handling of singular/indefinite Hessian"""
        # Function with singular Hessian at certain points
        def cost_fn(x):
            return x[0]**2  # No dependence on x[1]

        def gradient_fn(x):
            return np.array([2*x[0], 0])

        def hessian_fn(x):
            # Singular Hessian
            H = np.zeros((2, 2))
            H[0, 0] = 2
            return H

        config = AdaptiveLMConfig(initial_lambda=0.1, verbose=False)
        optimizer = AdaptiveLM(config)

        x_init = np.array([1.0, 5.0])  # x[1] is arbitrary

        # Should handle singular Hessian without crashing
        x_opt, info = optimizer.optimize(x_init, gradient_fn, hessian_fn, cost_fn)

        # Check x[0] converged to 0
        self.assertLess(abs(x_opt[0]), 1e-6,
                       f"x[0] should converge to 0, got {x_opt[0]}")

        print(f"✓ Singular Hessian: handled gracefully, x[0] = {x_opt[0]:.2e}")

    def test_comparison_with_fixed_damping(self):
        """Compare adaptive vs fixed damping performance"""
        # Test function with varying curvature
        def cost_fn(x):
            return np.exp(x[0]) + np.exp(-x[0]) + x[1]**2

        def gradient_fn(x):
            return np.array([np.exp(x[0]) - np.exp(-x[0]), 2*x[1]])

        def hessian_fn(x):
            H = np.zeros((2, 2))
            H[0, 0] = np.exp(x[0]) + np.exp(-x[0])
            H[1, 1] = 2
            return H

        x_init = np.array([2.0, 3.0])

        # Adaptive damping
        config_adaptive = AdaptiveLMConfig(initial_lambda=0.1, verbose=False)
        optimizer_adaptive = AdaptiveLM(config_adaptive)
        x_opt_adaptive, info_adaptive = optimizer_adaptive.optimize(
            x_init, gradient_fn, hessian_fn, cost_fn)

        # Fixed damping (simulated by disabling adaptation)
        config_fixed = AdaptiveLMConfig(
            initial_lambda=0.1,
            lambda_increase_factor=1.0,  # No increase
            lambda_decrease_factor=1.0,  # No decrease
            verbose=False
        )
        optimizer_fixed = AdaptiveLM(config_fixed)
        x_opt_fixed, info_fixed = optimizer_fixed.optimize(
            x_init, gradient_fn, hessian_fn, cost_fn)

        # Adaptive should converge faster or to better solution
        self.assertLessEqual(info_adaptive['final_cost'], info_fixed['final_cost'] + 1e-10,
                           "Adaptive should achieve at least as good cost as fixed")

        print(f"✓ Adaptive vs Fixed:")
        print(f"  Adaptive: {info_adaptive['iterations']} iters, cost = {info_adaptive['final_cost']:.2e}")
        print(f"  Fixed:    {info_fixed['iterations']} iters, cost = {info_fixed['final_cost']:.2e}")


def run_all_tests():
    """Run all adaptive LM tests"""
    print("="*60)
    print("Testing Adaptive Levenberg-Marquardt")
    print("="*60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdaptiveLM)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ All adaptive LM tests passed!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
        for test, trace in result.failures:
            print(f"\nFailed: {test}")
            print(trace)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)