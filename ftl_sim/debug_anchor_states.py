#!/usr/bin/env python3
"""Check what states the anchors are broadcasting"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

config = ConsensusGNConfig(max_iterations=1)
cgn = ConsensusGaussNewton(config)

# Add anchors
cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([3.0, 3.0, 0.0, 0.0, 0.0]))

cgn.add_edge(0, 2)
cgn.add_edge(1, 2)

# Exchange states
cgn._exchange_states()

print("Anchor 0 state:", cgn.nodes[0].state)
print("Anchor 1 state:", cgn.nodes[1].state)
print("\nNode 2 sees:")
print("  Neighbor 0 state:", cgn.nodes[2].neighbor_states.get(0))
print("  Neighbor 1 state:", cgn.nodes[2].neighbor_states.get(1))