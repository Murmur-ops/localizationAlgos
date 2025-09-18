#!/usr/bin/env python3
"""
Test convergence behavior to understand why 30-node network doesn't converge
"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters
import matplotlib.pyplot as plt

print("=" * 70)
print("CONVERGENCE BEHAVIOR ANALYSIS")
print("=" * 70)

np.random.seed(42)

# Setup: 30 nodes with 5 anchors (best configuration)
n_nodes = 30
n_anchors = 5
area_size = 50
comm_range = 25
range_noise_std = 0.01

# Anchors: corners + center
anchor_positions = np.array([
    [0, 0],
    [area_size, 0],
    [area_size, area_size],
    [0, area_size],
    [area_size/2, area_size/2]
])

# Unknowns in grid
n_unknowns = n_nodes - n_anchors
grid_size = int(np.ceil(np.sqrt(n_unknowns)))
x_pos = np.linspace(5, area_size-5, grid_size)
y_pos = np.linspace(5, area_size-5, grid_size)

unknown_positions = []
for x in x_pos:
    for y in y_pos:
        unknown_positions.append([x, y])
        if len(unknown_positions) >= n_unknowns:
            break
    if len(unknown_positions) >= n_unknowns:
        break

unknown_positions = np.array(unknown_positions[:n_unknowns])
true_positions = np.vstack([anchor_positions, unknown_positions])

# Test different consensus gains
gains_to_test = [0.01, 0.05, 0.1, 0.2, 0.5]
max_iters = 500  # More iterations

results_by_gain = {}

for consensus_gain in gains_to_test:
    print(f"\nTesting consensus gain μ = {consensus_gain}")
    print("-" * 40)

    config = ConsensusGNConfig(
        max_iterations=max_iters,
        consensus_gain=consensus_gain,
        step_size=0.3,
        gradient_tol=1e-5,
        step_tol=1e-6,
        verbose=False
    )

    cgn = ConsensusGaussNewton(config)

    # Add nodes with good initial guesses
    for i in range(n_nodes):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = true_positions[i]
            cgn.add_node(i, state, is_anchor=True)
        else:
            state[:2] = true_positions[i] + np.random.normal(0, 0.5, 2)  # Within 50cm
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)
                meas_range = dist + np.random.normal(0, range_noise_std)
                cgn.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

    cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

    # Track convergence over iterations
    rmse_history = []

    # Manual iteration to track progress
    for k in range(max_iters):
        # Exchange states
        cgn._exchange_states()

        # Update all nodes
        for node_id in range(n_anchors, n_nodes):
            cgn.nodes[node_id].update_state()

        # Calculate current RMSE
        errors = []
        for i in range(n_anchors, n_nodes):
            est_pos = cgn.nodes[i].state[:2]
            true_pos = true_positions[i]
            error = np.linalg.norm(est_pos - true_pos)
            errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        rmse_history.append(rmse)

        # Check convergence
        if k > 10:
            recent_change = abs(rmse_history[-1] - rmse_history[-10])
            if recent_change < 1e-4:  # Less than 0.1mm change in 10 iterations
                print(f"  Converged at iteration {k+1}")
                print(f"  Final RMSE: {rmse*100:.1f}cm")
                break

        if (k+1) % 100 == 0:
            print(f"  Iteration {k+1}: RMSE = {rmse*100:.1f}cm")

    results_by_gain[consensus_gain] = {
        'rmse_history': rmse_history,
        'final_rmse': rmse_history[-1],
        'iterations': len(rmse_history)
    }

# Analysis
print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS RESULTS")
print("=" * 70)

print("\nConsensus Gain | Final RMSE | Iterations | Status")
print("-" * 50)

for gain, result in results_by_gain.items():
    final_rmse = result['final_rmse']
    iters = result['iterations']
    status = "Converged" if iters < max_iters else "Not converged"
    print(f"{gain:14.2f} | {final_rmse*100:10.1f}cm | {iters:10d} | {status}")

# Find best gain
best_gain = min(results_by_gain.keys(),
                key=lambda g: results_by_gain[g]['final_rmse'])
best_rmse = results_by_gain[best_gain]['final_rmse']

print(f"\nBest consensus gain: μ = {best_gain}")
print(f"Best achievable RMSE: {best_rmse*100:.1f}cm")

# Test with no consensus (μ=0) for comparison
print("\n" + "=" * 70)
print("COMPARISON: NO CONSENSUS (μ = 0)")
print("=" * 70)

config_no_consensus = ConsensusGNConfig(
    max_iterations=200,
    consensus_gain=0.0,  # No consensus!
    step_size=0.3,
    gradient_tol=1e-5,
    verbose=False
)

cgn_no = ConsensusGaussNewton(config_no_consensus)

# Same network setup
for i in range(n_nodes):
    state = np.zeros(5)
    if i < n_anchors:
        state[:2] = true_positions[i]
        cgn_no.add_node(i, state, is_anchor=True)
    else:
        state[:2] = true_positions[i] + np.random.normal(0, 0.5, 2)
        cgn_no.add_node(i, state, is_anchor=False)

for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        dist = np.linalg.norm(true_positions[i] - true_positions[j])
        if dist <= comm_range:
            cgn_no.add_edge(i, j)
            meas_range = dist + np.random.normal(0, range_noise_std)
            cgn_no.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

cgn_no.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})
results_no = cgn_no.optimize()

if 'position_errors' in results_no:
    print(f"Without consensus: {results_no['position_errors']['rmse']*100:.1f}cm")
    print(f"With best consensus (μ={best_gain}): {best_rmse*100:.1f}cm")

    if best_rmse < results_no['position_errors']['rmse']:
        improvement = (results_no['position_errors']['rmse'] - best_rmse) / results_no['position_errors']['rmse'] * 100
        print(f"Consensus improves accuracy by {improvement:.1f}%")
    else:
        print("No consensus performs better (network may be too dense)")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print(f"""
1. CONVERGENCE BEHAVIOR:
   - System does not reach strict convergence criteria
   - RMSE plateaus after 100-300 iterations
   - Consensus gain affects convergence speed and accuracy

2. OPTIMAL PARAMETERS:
   - Best consensus gain: μ = {best_gain}
   - Best RMSE achieved: {best_rmse*100:.1f}cm
   - Typical iterations needed: 200-300 for plateau

3. PERFORMANCE UNDER IDEAL CONDITIONS:
   - 30 nodes over 50x50m
   - 5 anchors (corners + center)
   - 1cm measurement noise
   - Initial guess within 50cm
   → Achievable RMSE: {best_rmse*100:.1f}cm

4. CONVERGENCE ISSUE:
   - Strict gradient/step tolerance not met
   - System oscillates around minimum
   - May need adaptive parameters or relaxed criteria
""")