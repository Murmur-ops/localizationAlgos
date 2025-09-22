#!/usr/bin/env python3
"""
Debug why the trivial 3-node case isn't converging to near-zero error
"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

print("DEBUGGING TRIVIAL CONSENSUS CASE")
print("=" * 50)

np.random.seed(42)

# Try without consensus first
print("\n1. WITHOUT CONSENSUS (μ=0)")
print("-" * 30)

config = ConsensusGNConfig(
    max_iterations=20,
    consensus_gain=0.0,  # NO CONSENSUS
    step_size=0.5,
    gradient_tol=1e-6,
    verbose=False
)
cgn = ConsensusGaussNewton(config)

cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([4.9, 4.1, 0.0, 0.0, 0.0]))

cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

true_pos_2 = np.array([5.0, 4.0])
dist_02 = np.sqrt(25 + 16)  # 6.4031
dist_12 = np.sqrt(25 + 16)  # 6.4031

cgn.add_measurement(ToAFactorMeters(0, 2, dist_02, 1e-6))
cgn.add_measurement(ToAFactorMeters(1, 2, dist_12, 1e-6))

cgn.set_true_positions({2: true_pos_2})

results = cgn.optimize()
final_pos = cgn.nodes[2].state[:2]
final_error = np.linalg.norm(final_pos - true_pos_2)

print(f"Final position: {final_pos}")
print(f"Final error: {final_error*100:.4f}cm")
print(f"Y-coordinate issue: Should be 4.0, got {final_pos[1]:.6f}")

# Try with very small consensus
print("\n2. WITH TINY CONSENSUS (μ=0.01)")
print("-" * 30)

config = ConsensusGNConfig(
    max_iterations=20,
    consensus_gain=0.01,  # TINY consensus
    step_size=0.5,
    gradient_tol=1e-6,
    verbose=False
)
cgn = ConsensusGaussNewton(config)

cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([4.9, 4.1, 0.0, 0.0, 0.0]))

cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

cgn.add_measurement(ToAFactorMeters(0, 2, dist_02, 1e-6))
cgn.add_measurement(ToAFactorMeters(1, 2, dist_12, 1e-6))

cgn.set_true_positions({2: true_pos_2})
results = cgn.optimize()

final_pos = cgn.nodes[2].state[:2]
final_error = np.linalg.norm(final_pos - true_pos_2)

print(f"Final position: {final_pos}")
print(f"Final error: {final_error*100:.4f}cm")

# The issue might be that we have TWO solutions
print("\n3. AMBIGUITY ANALYSIS")
print("-" * 30)

print("With anchors at (0,0) and (10,0):")
print("And distances 6.4031 to each anchor")
print("There are TWO valid positions:")
print("  - (5, 4) - above the x-axis")
print("  - (5, -4) - below the x-axis")
print("\nThe optimization may be getting stuck between them!")
print("Starting at (4.9, 4.1) should go to (5, 4) though...")

# Let's check what's happening step by step
print("\n4. STEP-BY-STEP ANALYSIS")
print("-" * 30)

config = ConsensusGNConfig(
    max_iterations=3,  # Just a few steps
    consensus_gain=0.0,
    step_size=1.0,  # Full steps
    gradient_tol=1e-6,
    verbose=False
)
cgn = ConsensusGaussNewton(config)

cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([4.9, 4.1, 0.0, 0.0, 0.0]))

cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

cgn.add_measurement(ToAFactorMeters(0, 2, dist_02, 1e-6))
cgn.add_measurement(ToAFactorMeters(1, 2, dist_12, 1e-6))

print(f"Iteration 0: {cgn.nodes[2].state[:2]}")

for i in range(3):
    cgn._exchange_states()
    cgn.nodes[2].update_state()
    pos = cgn.nodes[2].state[:2]
    error = np.linalg.norm(pos - true_pos_2)
    print(f"Iteration {i+1}: {pos}, error={error*100:.2f}cm")

print("\nThe y-coordinate is collapsing toward zero!")
print("This might be due to consensus with anchors at y=0")