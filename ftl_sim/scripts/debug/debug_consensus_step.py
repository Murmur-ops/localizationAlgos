#!/usr/bin/env python3
"""Debug a single consensus optimization step"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

# Very simple 3-node problem
config = ConsensusGNConfig(
    max_iterations=1,  # Just one iteration
    consensus_gain=0.1,
    step_size=1.0,  # Full step
    verbose=True
)
cgn = ConsensusGaussNewton(config)

# Triangle: two anchors and one unknown
cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]))  # True: (5, 4)

# Connect all
cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

# Perfect measurements to true position (5, 4)
true_dist_0 = np.sqrt(5**2 + 4**2)  # 6.4031
true_dist_1 = np.sqrt(5**2 + 4**2)  # 6.4031

cgn.add_measurement(ToAFactorMeters(0, 2, true_dist_0, 0.01))
cgn.add_measurement(ToAFactorMeters(1, 2, true_dist_1, 0.01))

print("Initial position:", cgn.nodes[2].state[:2])
print("True position: [5, 4]")

# Manually step through
node = cgn.nodes[2]

# Exchange states
cgn._exchange_states()

# Build system
H, g = node.build_local_system()
print(f"\nLocal system:")
print(f"H diagonal: {np.diag(H)[:2]}")  # Just x,y
print(f"g (x,y): {g[:2]}")

# No consensus for now
delta = node.compute_step(H, g)
print(f"\nDelta (x,y): {delta[:2]}")

# What would new position be?
new_pos = node.state[:2] - node.config.step_size * delta[:2]
print(f"New position would be: {new_pos}")
print(f"Error before: {np.linalg.norm(node.state[:2] - [5, 4]):.2f}")
print(f"Error after: {np.linalg.norm(new_pos - [5, 4]):.2f}")

# Now try with consensus
H_cons, g_cons = node.add_consensus_penalty(H, g)
print(f"\nWith consensus:")
print(f"H diagonal: {np.diag(H_cons)[:2]}")
print(f"g (x,y): {g_cons[:2]}")

delta_cons = node.compute_step(H_cons, g_cons)
new_pos_cons = node.state[:2] - node.config.step_size * delta_cons[:2]
print(f"New position: {new_pos_cons}")
print(f"Error: {np.linalg.norm(new_pos_cons - [5, 4]):.2f}")