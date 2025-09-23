#!/usr/bin/env python3
"""
Debug if nodes are actually updating their states with neighbor information.
"""

import numpy as np
import time
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.consensus.message_types import StateMessage
from ftl.factors_scaled import ToAFactorMeters

# Create simple 3-node system
cgn_config = ConsensusGNConfig(verbose=False, step_size=0.5)
cgn = ConsensusGaussNewton(cgn_config)

positions = np.array([[0, 0], [1, 0], [2, 0]])

# Add nodes
for i in range(3):
    state = np.zeros(5)
    state[:2] = positions[i]

    if i == 0 or i == 2:  # Anchors
        cgn.add_node(i, state, is_anchor=True)
    else:  # Unknown node 1
        state[:2] += [0.2, 0.1]  # Add error
        state[2] = 0.5  # Bias error (ns)
        state[4] = 0.5  # CFO error (ppm)
        cgn.add_node(i, state, is_anchor=False)
        print(f"Node 1 initial state: {state}")

# Add edges and measurements
cgn.add_edge(0, 1)
cgn.add_edge(1, 2)

for i in range(2):
    factor = ToAFactorMeters(i, i+1, 1.0, 0.01**2)
    cgn.add_measurement(factor)

print("\n" + "="*60)
print("ITERATION-BY-ITERATION STATE TRACKING")
print("="*60)

# Track node 1's state over iterations
for iteration in range(5):
    print(f"\n--- Iteration {iteration} ---")

    # Store state before update
    state_before = cgn.nodes[1].state.copy()
    print(f"State before: {state_before}")

    # Share states with proper timestamps
    current_time = time.time()
    for node_id, node in cgn.nodes.items():
        for edge in cgn.edges:
            if edge[0] == node_id:
                neighbor_id = edge[1]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 iteration, current_time)
                node.receive_state(msg)
            elif edge[1] == node_id:
                neighbor_id = edge[0]
                msg = StateMessage(neighbor_id, cgn.nodes[neighbor_id].state.copy(),
                                 iteration, current_time)
                node.receive_state(msg)

    # Check neighbor states
    print(f"Node 1 neighbor_states: {list(cgn.nodes[1].neighbor_states.keys())}")

    # Build local system for node 1
    node = cgn.nodes[1]
    H, g = node.build_local_system()

    print(f"H matrix sum: {np.sum(np.abs(H)):.1f}")
    print(f"H diagonal: {np.diag(H)[:3]}")  # Just show first 3
    print(f"g vector: {g[:3]}")  # Just show first 3

    # Try to solve and update
    if np.sum(np.abs(H)) > 0:
        try:
            delta = np.linalg.solve(H + 1e-6 * np.eye(5), -g)
            print(f"Delta: {delta}")

            # Apply update
            node.state += cgn_config.step_size * delta

            state_after = node.state.copy()
            print(f"State after: {state_after}")

            # Check what changed
            change = state_after - state_before
            print(f"State change: {change}")
            print(f"  Position change: {np.linalg.norm(change[:2]):.4f} m")
            print(f"  Bias change: {change[2]:.4f} ns")
            print(f"  Drift change: {change[3]:.6f} ppb")
            print(f"  CFO change: {change[4]:.6f} ppm")

        except np.linalg.LinAlgError as e:
            print(f"Failed to solve: {e}")
    else:
        print("H matrix is all zeros - no update possible")

# Final check
print("\n" + "="*60)
print("FINAL STATE")
print("="*60)
final_state = cgn.nodes[1].state
true_state = np.array([1, 0, 0, 0, 0])
print(f"Node 1 final state: {final_state}")
print(f"True state: {true_state}")
print(f"Final errors:")
print(f"  Position: {np.linalg.norm(final_state[:2] - true_state[:2]):.4f} m")
print(f"  Bias: {final_state[2] - true_state[2]:.4f} ns")
print(f"  CFO: {final_state[4] - true_state[4]:.4f} ppm")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

# Check if CFO ever changed
if abs(final_state[4] - 0.5) < 0.0001:
    print("❌ CFO NEVER CHANGED from initial value!")
    print("   This explains the flat frequency convergence plots")
else:
    print(f"✓ CFO changed by {abs(final_state[4] - 0.5):.4f} ppm")

# Check if position converged
if np.linalg.norm(final_state[:2] - true_state[:2]) > 0.1:
    print("❌ Position did not converge (error > 10cm)")
else:
    print("✓ Position converged reasonably")

# Check H matrix structure
print("\nH matrix structure check:")
H, g = cgn.nodes[1].build_local_system()
print(f"  H[4,4] (CFO component): {H[4,4]:.3f}")
if H[4,4] == 0:
    print("  ❌ CFO has zero weight in H matrix - no updates possible!")
    print("     This means measurements don't constrain frequency offset")
else:
    print("  ✓ CFO component exists in H matrix")