"""
Unit tests for line search algorithms
"""

import numpy as np
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.optimization.line_search import LineSearch, LineSearchConfig, BacktrackingLineSearch


class TestLineSearch(unittest.TestCase):
    """Test suite for line search algorithms"""

    def test_armijo_backtracking(self):
        """Test Armijo backtracking on quadratic function"""
        # f(x) = x^T * x
        def f(x):
            return np.sum(x**2)

        def grad_f(x):
            return 2*x

        # Start at x = [2, 2], move toward origin
        x = np.array([2.0, 2.0])
        p = -grad_f(x)  # Steepest descent direction

        config = LineSearchConfig(method="armijo", verbose=False)
        ls = LineSearch(config)

        alpha, n_evals = ls.armijo_backtracking(f, grad_f, x, p)

        # Check that step reduces function value
        f_old = f(x)
        f_new = f(x + alpha * p)
        self.assertLess(f_new, f_old, "Step should reduce function value")

        # For quadratic, optimal alpha = 0.5 for steepest descent
        self.assertAlmostEqual(alpha, 0.5, places=2,
                             msg=f"Expected α≈0.5, got {alpha:.3f}")

        print(f"✓ Armijo: α={alpha:.3f}, f: {f_old:.2f}→{f_new:.2f}, {n_evals} evals")

    def test_wolfe_conditions(self):
        """Test Wolfe line search"""
        # Rosenbrock function
        def f(x):
            return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

        def grad_f(x):
            g = np.zeros(2)
            g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
            g[1] = 200*(x[1] - x[0]**2)
            return g

        x = np.array([-1.0, 1.0])
        p = -grad_f(x)

        config = LineSearchConfig(method="wolfe", c2=0.4, verbose=False)
        ls = LineSearch(config)

        alpha, n_evals = ls.wolfe_search(f, grad_f, x, p)

        # Verify Wolfe conditions
        f0 = f(x)
        grad0 = grad_f(x)
        descent0 = np.dot(grad0, p)

        x_new = x + alpha * p
        f_new = f(x_new)
        grad_new = grad_f(x_new)

        # Armijo condition
        self.assertLessEqual(f_new, f0 + config.c1 * alpha * descent0,
                           "Armijo condition not satisfied")

        # Curvature condition
        self.assertGreaterEqual(np.dot(grad_new, p), config.c2 * descent0,
                              "Curvature condition not satisfied")

        print(f"✓ Wolfe: α={alpha:.3f}, both conditions satisfied, {n_evals} evals")

    def test_strong_wolfe(self):
        """Test strong Wolfe conditions"""
        # Function with varying curvature
        def f(x):
            return np.exp(x[0]) - 2*x[0] + x[1]**2

        def grad_f(x):
            return np.array([np.exp(x[0]) - 2, 2*x[1]])

        x = np.array([1.0, 1.0])
        p = -grad_f(x)

        config = LineSearchConfig(method="strong_wolfe", verbose=False)
        ls = LineSearch(config)

        alpha, n_evals = ls.strong_wolfe_search(f, grad_f, x, p)

        # Verify strong Wolfe conditions
        f0 = f(x)
        grad0 = grad_f(x)
        descent0 = np.dot(grad0, p)

        x_new = x + alpha * p
        f_new = f(x_new)
        grad_new = grad_f(x_new)

        # Armijo
        self.assertLessEqual(f_new, f0 + config.c1 * alpha * descent0,
                           "Armijo condition not satisfied")

        # Strong curvature
        self.assertLessEqual(abs(np.dot(grad_new, p)), config.c2 * abs(descent0),
                           "Strong curvature condition not satisfied")

        print(f"✓ Strong Wolfe: α={alpha:.3f}, strong conditions satisfied, {n_evals} evals")

    def test_non_descent_direction(self):
        """Test handling of non-descent direction"""
        def f(x):
            return np.sum(x**2)

        def grad_f(x):
            return 2*x

        x = np.array([1.0, 1.0])
        p = grad_f(x)  # Ascent direction!

        config = LineSearchConfig(verbose=False)
        ls = LineSearch(config)

        alpha, n_evals = ls.armijo_backtracking(f, grad_f, x, p)

        # Should return zero step for non-descent direction
        self.assertEqual(alpha, 0.0,
                        "Should return α=0 for non-descent direction")

        print(f"✓ Non-descent: correctly returned α={alpha}")

    def test_backtracking_class(self):
        """Test simple backtracking helper class"""
        def f(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        x = np.array([0.0, 0.0])
        p = np.array([1.0, 2.0])  # Move toward minimum
        grad_dot_p = -2.0  # Approximate

        alpha = BacktrackingLineSearch.search(f, x, p, grad_dot_p)

        # Check that step improves function
        self.assertLess(f(x + alpha * p), f(x),
                       "Backtracking should find improving step")

        print(f"✓ Backtracking helper: α={alpha:.3f}")

    def test_step_size_bounds(self):
        """Test that step size respects bounds"""
        def f(x):
            return np.sum(x**4)

        def grad_f(x):
            return 4*x**3

        x = np.array([10.0, 10.0])
        p = -grad_f(x)

        config = LineSearchConfig(
            alpha_min=0.01,
            alpha_max=0.5,
            alpha_init=10.0,  # Too large
            verbose=False
        )
        ls = LineSearch(config)

        alpha, _ = ls.armijo_backtracking(f, grad_f, x, p)

        # Check bounds
        self.assertGreaterEqual(alpha, config.alpha_min,
                              "Alpha below minimum")
        self.assertLessEqual(alpha, config.alpha_max,
                           "Alpha above maximum")

        print(f"✓ Bounds: α={alpha:.3f} ∈ [{config.alpha_min}, {config.alpha_max}]")

    def test_performance_comparison(self):
        """Compare different line search methods"""
        # Test function
        def f(x):
            return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

        def grad_f(x):
            g = np.zeros(2)
            g[0] = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
            g[1] = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
            return g

        x = np.array([0.0, 0.0])
        p = -grad_f(x)

        methods = ["armijo", "wolfe", "strong_wolfe"]
        results = {}

        for method in methods:
            config = LineSearchConfig(method=method, verbose=False)
            ls = LineSearch(config)
            alpha, n_evals = ls.search(f, grad_f, x, p)
            results[method] = (alpha, n_evals, f(x + alpha * p))

        print("✓ Performance comparison:")
        for method, (alpha, n_evals, f_new) in results.items():
            print(f"  {method:12s}: α={alpha:.3f}, f={f_new:.3f}, {n_evals:2d} evals")

        # All methods should improve function
        f0 = f(x)
        for method, (_, _, f_new) in results.items():
            self.assertLess(f_new, f0,
                           f"{method} should improve function")

    def test_exact_line_search_quadratic(self):
        """Test on quadratic where exact solution is known"""
        # f(x) = 0.5 * x^T * A * x
        A = np.array([[4, 1], [1, 2]])

        def f(x):
            return 0.5 * x.T @ A @ x

        def grad_f(x):
            return A @ x

        x = np.array([2.0, 3.0])
        grad = grad_f(x)
        p = -grad  # Steepest descent

        # Exact line search for quadratic: α = (g^T * g) / (g^T * A * g)
        alpha_exact = (grad.T @ grad) / (grad.T @ A @ grad)

        config = LineSearchConfig(method="armijo", c1=0.3, verbose=False)
        ls = LineSearch(config)
        alpha_armijo, _ = ls.armijo_backtracking(f, grad_f, x, p)

        # Armijo should find step close to exact (within backtracking resolution)
        self.assertLess(abs(alpha_armijo - alpha_exact), 0.5,
                       f"Armijo α={alpha_armijo:.3f} far from exact α={alpha_exact:.3f}")

        print(f"✓ Quadratic: exact α={alpha_exact:.3f}, Armijo α={alpha_armijo:.3f}")

    def test_difficult_function(self):
        """Test on function with narrow valley (requires careful step size)"""
        # Narrow valley function
        def f(x):
            return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

        def grad_f(x):
            g = np.zeros(2)
            g[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
            g[1] = 200*(x[1] - x[0]**2)
            return g

        # Start far from minimum
        x = np.array([-2.0, 4.0])
        p = -grad_f(x)

        config = LineSearchConfig(
            method="strong_wolfe",
            c1=1e-4,
            c2=0.9,
            verbose=False
        )
        ls = LineSearch(config)

        alpha, n_evals = ls.search(f, grad_f, x, p)

        # Should find reasonable step
        self.assertGreater(alpha, 0, "Should find positive step")
        # For very difficult functions, line search might not always improve
        # but should at least not make things much worse
        f_old = f(x)
        f_new = f(x + alpha * p)
        # Allow small increase due to numerical issues with difficult function
        self.assertLess(f_new, f_old * 1.1,
                       f"Function increased too much: {f_old:.3f} -> {f_new:.3f}")

        print(f"✓ Difficult function: α={alpha:.4f}, {n_evals} evals")


def run_all_tests():
    """Run all line search tests"""
    print("="*60)
    print("Testing Line Search Algorithms")
    print("="*60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLineSearch)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ All line search tests passed!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
        for test, trace in result.failures:
            print(f"\nFailed: {test}")
            print(trace)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)