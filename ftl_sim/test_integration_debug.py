"""
Debug integration issues with enhanced FTL optimization
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


def test_gradient_computation():
    """Test gradient and Hessian computation"""
    print("Testing gradient computation...")

    config = EnhancedFTLConfig(
        n_nodes=6,  # 3 anchors + 3 unknowns
        n_anchors=3,
        verbose=False
    )

    ftl = EnhancedFTL(config)

    # Get initial states
    n_unknowns = config.n_nodes - config.n_anchors
    unknown_states = np.zeros((n_unknowns, 3))
    for i in range(n_unknowns):
        unknown_states[i] = ftl.states[config.n_anchors + i]

    states_flat = unknown_states.flatten()

    # Compute gradient and Hessian
    H, g = ftl.compute_gradient_hessian(states_flat)

    print(f"  State vector size: {len(states_flat)}")
    print(f"  Gradient norm: {np.linalg.norm(g):.2e}")
    print(f"  Gradient range: [{np.min(g):.2e}, {np.max(g):.2e}]")
    print(f"  Hessian condition number: {np.linalg.cond(H):.2e}")
    print(f"  Hessian diagonal range: [{np.min(np.diag(H)):.2e}, {np.max(np.diag(H)):.2e}]")

    # Test finite differences
    eps = 1e-8
    grad_fd = np.zeros_like(g)

    for i in range(len(states_flat)):
        states_plus = states_flat.copy()
        states_minus = states_flat.copy()
        states_plus[i] += eps
        states_minus[i] -= eps

        cost_plus = ftl.compute_cost(states_plus)
        cost_minus = ftl.compute_cost(states_minus)
        grad_fd[i] = (cost_plus - cost_minus) / (2 * eps)

    # Since g is -gradient, compare with -grad_fd
    grad_error = np.linalg.norm(g + grad_fd) / (np.linalg.norm(grad_fd) + 1e-10)
    print(f"  Gradient finite diff error: {grad_error:.2e}")

    if grad_error > 0.01:
        print("  WARNING: Large gradient error!")
        for i in range(len(g)):
            if abs(g[i] + grad_fd[i]) > 0.1 * abs(grad_fd[i]):
                print(f"    Component {i}: analytic={g[i]:.2e}, FD={-grad_fd[i]:.2e}")

    return H, g, grad_fd


def test_step_methods():
    """Test different step methods"""
    print("\nTesting step methods...")

    config = EnhancedFTLConfig(
        n_nodes=6,
        n_anchors=3,
        verbose=False,
        max_iterations=10
    )

    # Test basic method
    print("\n  Basic method:")
    ftl_basic = EnhancedFTL(config)
    initial_cost = ftl_basic.compute_cost(
        ftl_basic.states[config.n_anchors:].flatten()
    )

    for i in range(5):
        ftl_basic.step_basic()
        cost = ftl_basic.compute_cost(
            ftl_basic.states[config.n_anchors:].flatten()
        )
        pos_rmse, _ = ftl_basic.compute_errors()
        print(f"    Iter {i}: cost={cost:.2e}, pos_rmse={pos_rmse:.3f}m")

    # Test LM method
    print("\n  LM method:")
    config.use_adaptive_lm = True
    config.use_line_search = False
    ftl_lm = EnhancedFTL(config)

    # Debug initial gradient
    states_flat = ftl_lm.states[config.n_anchors:].flatten()
    H, g = ftl_lm.compute_gradient_hessian(states_flat)
    print(f"    Initial ||g|| = {np.linalg.norm(g):.2e}")

    for i in range(5):
        converged = ftl_lm.step_with_lm()
        cost = ftl_lm.compute_cost(
            ftl_lm.states[config.n_anchors:].flatten()
        )
        pos_rmse, _ = ftl_lm.compute_errors()
        lambda_val = ftl_lm.lm_optimizer.lambda_current
        print(f"    Iter {i}: cost={cost:.2e}, pos_rmse={pos_rmse:.3f}m, λ={lambda_val:.2e}, converged={converged}")

    # Test line search method
    print("\n  Line search method:")
    config.use_adaptive_lm = False
    config.use_line_search = True
    ftl_ls = EnhancedFTL(config)

    for i in range(5):
        ftl_ls.step_with_line_search()
        cost = ftl_ls.compute_cost(
            ftl_ls.states[config.n_anchors:].flatten()
        )
        pos_rmse, _ = ftl_ls.compute_errors()
        alpha = ftl_ls.alpha_history[-1] if ftl_ls.alpha_history else 0
        print(f"    Iter {i}: cost={cost:.2e}, pos_rmse={pos_rmse:.3f}m, α={alpha:.2e}")


def test_cost_function():
    """Test cost function behavior"""
    print("\nTesting cost function...")

    config = EnhancedFTLConfig(n_nodes=6, n_anchors=3, verbose=False)
    ftl = EnhancedFTL(config)

    # Get initial state
    n_unknowns = config.n_nodes - config.n_anchors
    unknown_states = np.zeros((n_unknowns, 3))
    for i in range(n_unknowns):
        unknown_states[i] = ftl.states[config.n_anchors + i]

    states_flat = unknown_states.flatten()
    initial_cost = ftl.compute_cost(states_flat)

    print(f"  Initial cost: {initial_cost:.2e}")

    # Move toward true solution
    true_states = np.zeros((n_unknowns, 3))
    for i in range(n_unknowns):
        true_states[i, :2] = ftl.true_positions[config.n_anchors + i]
        true_states[i, 2] = 0  # True clock bias is 0 (synchronized)

    true_flat = true_states.flatten()
    true_cost = ftl.compute_cost(true_flat)
    print(f"  True solution cost: {true_cost:.2e}")

    # Test along line to true solution
    alphas = [0.0, 0.1, 0.5, 0.9, 1.0]
    for alpha in alphas:
        test_state = (1-alpha) * states_flat + alpha * true_flat
        cost = ftl.compute_cost(test_state)
        print(f"    α={alpha:.1f}: cost={cost:.2e}")

    # Check if cost decreases toward true solution
    if true_cost > initial_cost * 0.5:
        print("  WARNING: True solution cost is not much lower than initial!")


def main():
    print("="*60)
    print("Integration Debugging")
    print("="*60)

    test_gradient_computation()
    test_cost_function()
    test_step_methods()

    print("\n" + "="*60)
    print("Debug Complete")
    print("="*60)


if __name__ == "__main__":
    main()