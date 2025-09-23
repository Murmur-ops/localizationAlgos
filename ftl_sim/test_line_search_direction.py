"""
Test line search with Gauss-Newton direction
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


def test_ls_direction():
    """Test line search direction"""
    print("Testing line search direction...")

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

    # Add small damping for stability
    H += 1e-9 * np.eye(len(H))

    # Gauss-Newton direction
    p = np.linalg.solve(H, g)

    print(f"Gradient norm: {np.linalg.norm(g):.2e}")
    print(f"Direction norm: {np.linalg.norm(p):.2e}")
    print(f"grad·p = {np.dot(g, p):.2e}")
    print(f"Sign: {'positive' if np.dot(g, p) > 0 else 'negative'}")

    # Test if this is a descent direction
    def cost_fn(x):
        return ftl.compute_cost(x)

    initial_cost = cost_fn(states_flat)

    # Test small step
    alpha = 0.001
    new_state = states_flat + alpha * p
    new_cost = cost_fn(new_state)

    print(f"\nStep test with α={alpha}:")
    print(f"  Initial cost: {initial_cost:.2e}")
    print(f"  New cost: {new_cost:.2e}")
    print(f"  Cost reduction: {initial_cost - new_cost:.2e}")
    print(f"  Is descent? {new_cost < initial_cost}")


if __name__ == "__main__":
    test_ls_direction()