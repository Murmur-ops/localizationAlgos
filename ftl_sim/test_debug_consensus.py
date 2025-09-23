#!/usr/bin/env python3
"""
Debug test to find why H matrix is zero.
"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters

# Create simple system
cgn_config = ConsensusGNConfig(verbose=False)
cgn = ConsensusGaussNewton(cgn_config)

# Add 3 nodes in a line
positions = np.array([[0, 0], [1, 0], [2, 0]])

# Add nodes
for i in range(3):
    state = np.zeros(5)
    state[:2] = positions[i]
    if i == 0 or i == 2:  # Anchors at ends
        cgn.add_node(i, state, is_anchor=True)
    else:  # Unknown in middle
        state[:2] += [0.1, 0.05]  # Add error
        cgn.add_node(i, state, is_anchor=False)

# Add edges
cgn.add_edge(0, 1)
cgn.add_edge(1, 2)

print(f"Node 1 neighbors after add_edge: {list(cgn.nodes[1].neighbors.keys())}")

# Add measurements
for i in range(2):
    true_dist = 1.0
    factor = ToAFactorMeters(i, i+1, true_dist, 0.01**2)
    cgn.add_measurement(factor)

print(f"Node 1 local factors: {len(cgn.nodes[1].local_factors)}")

# Exchange states
import time
current_time = time.time()
for node_id, node in cgn.nodes.items():
    for neighbor_id in node.neighbors:
        msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(), 0, current_time)
        print(f"Sending message from {neighbor_id} to {node_id}: age={msg.age()}, max_stale={node.config.max_stale_time}")

        # Debug the receive_state logic
        if msg.node_id not in node.neighbors:
            print(f"  BLOCKED: {msg.node_id} not in neighbors {list(node.neighbors.keys())}")
        elif msg.age() > node.config.max_stale_time:
            print(f"  BLOCKED: message too old ({msg.age()} > {node.config.max_stale_time})")
        else:
            print(f"  SHOULD WORK")

        node.receive_state(msg)
        print(f"  After receive: neighbor_states[{neighbor_id}] = {node.neighbor_states.get(neighbor_id, 'NONE')}")

print(f"\nNode 1 neighbor_states keys: {list(cgn.nodes[1].neighbor_states.keys())}")
print(f"Node 1 neighbor_states values: {[v is not None for v in cgn.nodes[1].neighbor_states.values()]}")
print(f"Node 1 neighbor_states[0]: {cgn.nodes[1].neighbor_states.get(0, 'NOT FOUND')}")
print(f"Node 1 neighbor_states[2]: {cgn.nodes[1].neighbor_states.get(2, 'NOT FOUND')}")

# Try to build system for node 1
node = cgn.nodes[1]
print(f"\nBuilding system for node 1:")
print(f"  config.node_id: {node.config.node_id}")
print(f"  state: {node.state}")

# Manually check each factor
for idx, factor in enumerate(node.local_factors):
    print(f"\nFactor {idx}: ToA between {factor.i} and {factor.j}")

    print(f"  Checking: factor.i={factor.i}, factor.j={factor.j}, node_id={node.config.node_id}")

    if factor.i == node.config.node_id:
        xi = node.state
        xj = node._get_node_state(factor.j)
        print(f"  Branch 1: Node {factor.i} is self, getting neighbor {factor.j}")
        print(f"  Calling _get_node_state({factor.j})")
    elif factor.j == node.config.node_id:
        xi = node._get_node_state(factor.i)
        xj = node.state
        print(f"  Branch 2: Node {factor.j} is self, getting neighbor {factor.i}")
        print(f"  Calling _get_node_state({factor.i})")
    else:
        xi = None
        xj = None
        print(f"  Branch 3: Factor doesn't involve this node")

    # Debug _get_node_state
    test_0 = node._get_node_state(0)
    test_1 = node._get_node_state(1)
    test_2 = node._get_node_state(2)
    print(f"  _get_node_state(0): {test_0 is not None}")
    print(f"  _get_node_state(1): {test_1 is not None} (should be self)")
    print(f"  _get_node_state(2): {test_2 is not None}")

    print(f"  xi: {xi}")
    print(f"  xj: {xj}")

    if xi is not None and xj is not None:
        r_wh, Ji_wh, Jj_wh = factor.whitened_residual_and_jacobian(xi, xj)
        print(f"  Residual (whitened): {r_wh:.3f}")
        print(f"  Ji_wh: {Ji_wh[:3]}")  # Just position and bias components
        print(f"  Jj_wh: {Jj_wh[:3]}")

# Now build the full system
H, g = node.build_local_system()
print(f"\nBuilt system:")
print(f"  H diagonal: {np.diag(H)}")
print(f"  g: {g}")
print(f"  H sum: {np.sum(np.abs(H)):.3f}")

# Try to solve and update
print(f"\nTrying to update node 1:")
initial_pos = node.state[:2].copy()
try:
    delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
    print(f"  Delta: {delta}")
    node.state += 0.5 * delta  # Damped update
    print(f"  Initial pos: {initial_pos}")
    print(f"  New pos: {node.state[:2]}")
    print(f"  Position change: {np.linalg.norm(node.state[:2] - initial_pos):.4f} m")
except np.linalg.LinAlgError as e:
    print(f"  Failed to solve: {e}")