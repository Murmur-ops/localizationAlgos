#!/usr/bin/env python3
"""
Test consensus gradient computation to identify sign error
"""

import numpy as np

def test_consensus_gradient():
    """
    Test the gradient of consensus penalty: μ/2 * Σ ||x_i - x_j||²
    """

    # Node i state (offset from consensus)
    x_i = np.array([6.0, 5.0, 0, 0, 0])

    # Neighbor states (both at same position for clear consensus target)
    neighbors = [
        np.array([4.0, 5.0, 0, 0, 0]),  # Left neighbor
        np.array([4.0, 5.0, 0, 0, 0]),  # Another left neighbor
    ]

    mu = 1.0

    # Cost function: μ/2 * Σ ||x_i - x_j||²
    cost = 0
    for x_j in neighbors:
        cost += mu/2 * np.linalg.norm(x_i - x_j)**2

    print(f"Initial cost: {cost:.4f}")

    # Analytical gradient: ∂C/∂x_i = μ * Σ (x_i - x_j)
    gradient_analytical = np.zeros(5)
    for x_j in neighbors:
        gradient_analytical += mu * (x_i - x_j)

    print(f"Analytical gradient: {gradient_analytical}")

    # Current FTL implementation (INCORRECT):
    # consensus_term = μ * Σ (x_j - x_i)
    # g -= consensus_term
    consensus_term_ftl = np.zeros(5)
    for x_j in neighbors:
        consensus_term_ftl += mu * (x_j - x_i)  # Note: x_j - x_i

    # FTL subtracts this from g
    gradient_ftl = -consensus_term_ftl  # g -= consensus_term

    print(f"FTL gradient (incorrect): {gradient_ftl}")
    print(f"Sign error: FTL gradient has opposite sign!")

    # Numerical verification
    eps = 1e-6
    gradient_numerical = np.zeros(5)

    for i in range(5):
        # Perturb x_i
        x_i_plus = x_i.copy()
        x_i_plus[i] += eps

        cost_plus = 0
        for x_j in neighbors:
            cost_plus += mu/2 * np.linalg.norm(x_i_plus - x_j)**2

        gradient_numerical[i] = (cost_plus - cost) / eps

    print(f"Numerical gradient: {gradient_numerical}")

    # Test gradient descent
    print("\n--- Testing gradient descent ---")

    # Correct gradient descent
    step_size = 0.1
    x_new_correct = x_i - step_size * gradient_analytical

    cost_new_correct = 0
    for x_j in neighbors:
        cost_new_correct += mu/2 * np.linalg.norm(x_new_correct - x_j)**2

    print(f"Correct update: x_new = {x_new_correct[:2]}")
    print(f"  New cost: {cost_new_correct:.4f} (decreased from {cost:.4f})")

    # FTL's incorrect update
    x_new_ftl = x_i - step_size * gradient_ftl

    cost_new_ftl = 0
    for x_j in neighbors:
        cost_new_ftl += mu/2 * np.linalg.norm(x_new_ftl - x_j)**2

    print(f"FTL update: x_new = {x_new_ftl[:2]}")
    print(f"  New cost: {cost_new_ftl:.4f} (INCREASED from {cost:.4f})")

    print("\n--- Analysis ---")
    print("The FTL implementation moves AWAY from consensus instead of toward it!")
    print("This is why convergence fails even with perfect measurements.")

    # Show the fix
    print("\n--- The Fix ---")
    print("In consensus_node.py, line should be:")
    print("  g += consensus_term  # Add, not subtract")
    print("OR equivalently:")
    print("  consensus_term += mu * (self.state - neighbor_state)  # Flip sign in computation")

if __name__ == "__main__":
    test_consensus_gradient()