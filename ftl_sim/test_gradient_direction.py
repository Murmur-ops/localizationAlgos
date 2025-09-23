"""
Test gradient direction
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


def test_gradient_direction():
    """Test if gradient points in right direction"""
    print("Testing gradient direction...")

    config = EnhancedFTLConfig(
        n_nodes=6,
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

    # Cost function
    def cost_fn(x):
        return ftl.compute_cost(x)

    initial_cost = cost_fn(states_flat)
    print(f"Initial cost: {initial_cost:.2e}")
    print(f"Gradient norm: {np.linalg.norm(g):.2e}")

    # Test small steps along gradient direction
    print("\nTesting steps along gradient g:")
    alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    for alpha in alphas:
        # Step along g (not -g)
        test_state = states_flat + alpha * g
        new_cost = cost_fn(test_state)
        print(f"  α={alpha:.0e}: cost={new_cost:.2e}, change={new_cost-initial_cost:.2e}")

    print("\nTesting steps along negative gradient -g:")
    for alpha in alphas:
        # Step along -g
        test_state = states_flat - alpha * g
        new_cost = cost_fn(test_state)
        print(f"  α={alpha:.0e}: cost={new_cost:.2e}, change={new_cost-initial_cost:.2e}")

    # Test Newton direction
    print("\nTesting Newton direction -H^{-1}*g:")
    try:
        # Add small damping for stability
        H_damped = H + 1e-9 * np.eye(len(H))
        newton_dir = -np.linalg.solve(H_damped, g)

        for alpha in [1e-3, 1e-2, 0.1, 0.5, 1.0]:
            test_state = states_flat + alpha * newton_dir
            new_cost = cost_fn(test_state)
            print(f"  α={alpha:.2e}: cost={new_cost:.2e}, change={new_cost-initial_cost:.2e}")
    except:
        print("  Failed to compute Newton direction")

    # Test Gauss-Newton direction (what we should be using)
    print("\nTesting Gauss-Newton direction H^{-1}*g:")
    try:
        # For least squares: H = J^T*J, g = J^T*r
        # Gauss-Newton step: delta = (J^T*J)^{-1} * J^T*r = H^{-1}*g
        H_damped = H + 1e-9 * np.eye(len(H))
        gn_dir = np.linalg.solve(H_damped, g)

        for alpha in [1e-3, 1e-2, 0.1, 0.5, 1.0]:
            test_state = states_flat + alpha * gn_dir
            new_cost = cost_fn(test_state)
            print(f"  α={alpha:.2e}: cost={new_cost:.2e}, change={new_cost-initial_cost:.2e}")
    except:
        print("  Failed to compute GN direction")


if __name__ == "__main__":
    test_gradient_direction()