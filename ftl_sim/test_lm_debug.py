"""
Debug LM convergence issue
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from ftl.optimization.adaptive_lm import AdaptiveLM, AdaptiveLMConfig


def test_lm_step():
    """Test LM step in detail"""
    print("Testing LM step in detail...")

    config = EnhancedFTLConfig(
        n_nodes=6,
        n_anchors=3,
        use_adaptive_lm=True,
        use_line_search=False,
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

    print(f"Initial state norm: {np.linalg.norm(states_flat):.2e}")
    print(f"Gradient norm: {np.linalg.norm(g):.2e}")
    print(f"Hessian condition: {np.linalg.cond(H):.2e}")

    # Define cost function
    def cost_fn(x):
        return ftl.compute_cost(x)

    initial_cost = cost_fn(states_flat)
    print(f"Initial cost: {initial_cost:.2e}")

    # Create LM optimizer with verbose output
    lm_config = AdaptiveLMConfig(
        initial_lambda=1e-4,
        gradient_tol=1e-6,
        step_tol=1e-8,
        verbose=True  # Enable verbose
    )
    lm = AdaptiveLM(lm_config)

    print("\nPerforming LM step...")
    x_new, cost_new, converged = lm.step(states_flat, H, g, cost_fn)

    print(f"\nResult:")
    print(f"  New cost: {cost_new:.2e}")
    print(f"  Cost reduction: {initial_cost - cost_new:.2e}")
    print(f"  Step norm: {np.linalg.norm(x_new - states_flat):.2e}")
    print(f"  Converged: {converged}")
    print(f"  Lambda: {lm.lambda_current:.2e}")

    # Test with different lambda values
    print("\nTesting different lambda values:")
    lambdas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e8]

    for lambda_val in lambdas:
        lm2 = AdaptiveLM(AdaptiveLMConfig(initial_lambda=lambda_val, verbose=False))
        lm2.lambda_current = lambda_val
        delta = lm2.compute_damped_step(H, g, lambda_val)

        # Test step
        test_cost = cost_fn(states_flat + delta)
        print(f"  λ={lambda_val:.0e}: ||δ||={np.linalg.norm(delta):.2e}, "
              f"cost={test_cost:.2e}, reduction={initial_cost-test_cost:.2e}")


if __name__ == "__main__":
    test_lm_step()