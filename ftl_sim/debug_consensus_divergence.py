#!/usr/bin/env python3
"""Debug why consensus optimization is diverging"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

# Simple test case
config = ConsensusGNConfig(max_iterations=5, verbose=True, step_size=0.1)
cgn = ConsensusGaussNewton(config)

# Two anchors
cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)

# One unknown (true position at (5, 0))
cgn.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]), is_anchor=False)

# Connect
cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

# Add measurements
cgn.add_measurement(ToAFactorMeters(0, 2, 5.0, 0.01))
cgn.add_measurement(ToAFactorMeters(1, 2, 5.0, 0.01))

print("Initial state:", cgn.nodes[2].state)

# Manually check one iteration
node = cgn.nodes[2]

# Exchange states first
cgn._exchange_states()

# Build local system
H, g = node.build_local_system()
print(f"\nH shape: {H.shape}")
print(f"H diagonal: {np.diag(H)}")
print(f"g: {g}")

# Add consensus
H_cons, g_cons = node.add_consensus_penalty(H, g)
print(f"\nAfter consensus:")
print(f"H diagonal: {np.diag(H_cons)}")
print(f"g: {g_cons}")

# Compute step
delta = node.compute_step(H_cons, g_cons)
print(f"\nDelta: {delta}")
print(f"Step size: {node.config.step_size}")

# What would new state be?
new_state = node.state - node.config.step_size * delta
print(f"\nNew state would be: {new_state}")

# Check if this is reasonable
print(f"\nMovement: {np.linalg.norm(new_state[:2] - node.state[:2]):.3f}m")