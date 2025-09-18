#!/usr/bin/env python3
"""Debug why 30-node consensus is performing poorly"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

# Simplified test
np.random.seed(42)

config = ConsensusGNConfig(
    max_iterations=100,  # More iterations
    consensus_gain=0.1,  # Lower gain for stability
    step_size=0.3,  # Smaller steps
    gradient_tol=1e-4,
    verbose=True
)
cgn = ConsensusGaussNewton(config)

# Simple line network: A--U1--U2--A
cgn.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(1, np.array([30.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
cgn.add_node(2, np.array([8.0, 2.0, 0.0, 0.0, 0.0]))  # True: (10, 0)
cgn.add_node(3, np.array([22.0, 2.0, 0.0, 0.0, 0.0]))  # True: (20, 0)

# Connect in line
cgn.add_edge(0, 2)
cgn.add_edge(2, 3)
cgn.add_edge(3, 1)

# Add measurements
cgn.add_measurement(ToAFactorMeters(0, 2, 10.0, 0.01))
cgn.add_measurement(ToAFactorMeters(2, 3, 10.0, 0.01))
cgn.add_measurement(ToAFactorMeters(3, 1, 10.0, 0.01))

cgn.set_true_positions({
    2: np.array([10.0, 0.0]),
    3: np.array([20.0, 0.0])
})

print("\nInitial positions:")
print(f"Node 2: {cgn.nodes[2].state[:2]} (true: [10, 0])")
print(f"Node 3: {cgn.nodes[3].state[:2]} (true: [20, 0])")

results = cgn.optimize()

print(f"\nFinal RMSE: {results['position_errors']['rmse']*100:.1f}cm")
print(f"Node 2: {cgn.nodes[2].state[:2]} (true: [10, 0])")
print(f"Node 3: {cgn.nodes[3].state[:2]} (true: [20, 0])")