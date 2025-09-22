#!/usr/bin/env python3
"""
Test 30-node consensus performance under IDEAL conditions
- Non-collinear anchors (corners + center)
- Good initial guesses
- Low measurement noise
- Sufficient communication range
"""

import numpy as np
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters
import time

print("=" * 70)
print("30-NODE CONSENSUS UNDER IDEAL CONDITIONS")
print("=" * 70)

# IDEAL CONDITIONS SETUP
np.random.seed(42)
n_nodes = 30
n_anchors = 4  # Will try with and without center anchor
area_size = 50  # meters
comm_range = 25  # Good connectivity (half the area)
range_noise_std = 0.01  # 1cm - very low noise
init_noise_std = 2.0  # Start within 2m of true position

print("\n1. NETWORK CONFIGURATION")
print("-" * 40)
print(f"Nodes: {n_nodes} ({n_anchors} anchors)")
print(f"Area: {area_size}x{area_size}m")
print(f"Communication range: {comm_range}m")
print(f"Measurement noise: {range_noise_std*100:.0f}cm")
print(f"Initial position noise: {init_noise_std:.1f}m")

# Test 1: 4 corner anchors (potentially problematic geometry)
print("\n" + "=" * 70)
print("TEST 1: 4 CORNER ANCHORS")
print("=" * 70)

# Anchors at corners
anchor_positions_4 = np.array([
    [0, 0],
    [area_size, 0],
    [area_size, area_size],
    [0, area_size]
])

# Distribute unknowns in a grid pattern for good coverage
n_unknowns = n_nodes - n_anchors
grid_size = int(np.ceil(np.sqrt(n_unknowns)))
x_positions = np.linspace(5, area_size-5, grid_size)
y_positions = np.linspace(5, area_size-5, grid_size)

unknown_positions = []
for x in x_positions:
    for y in y_positions:
        unknown_positions.append([x, y])
        if len(unknown_positions) >= n_unknowns:
            break
    if len(unknown_positions) >= n_unknowns:
        break

unknown_positions = np.array(unknown_positions[:n_unknowns])
true_positions_4 = np.vstack([anchor_positions_4, unknown_positions])

# Create consensus solver with tuned parameters
config_4 = ConsensusGNConfig(
    max_iterations=200,  # Allow more iterations
    consensus_gain=0.1,  # Low gain for stability
    step_size=0.3,  # Conservative step size
    gradient_tol=1e-5,
    step_tol=1e-6,
    verbose=False
)

cgn_4 = ConsensusGaussNewton(config_4)

# Add nodes
for i in range(n_nodes):
    state = np.zeros(5)
    if i < n_anchors:
        state[:2] = true_positions_4[i]
        cgn_4.add_node(i, state, is_anchor=True)
    else:
        # Good initial guess
        state[:2] = true_positions_4[i] + np.random.normal(0, init_noise_std, 2)
        cgn_4.add_node(i, state, is_anchor=False)

# Add edges and measurements
n_edges = 0
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        dist = np.linalg.norm(true_positions_4[i] - true_positions_4[j])
        if dist <= comm_range:
            cgn_4.add_edge(i, j)
            n_edges += 1
            # Low noise measurement
            meas_range = dist + np.random.normal(0, range_noise_std)
            cgn_4.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

print(f"Network edges: {n_edges}")
print(f"Average degree: {n_edges * 2 / n_nodes:.1f}")

# Check connectivity
direct_anchor = sum(1 for i in range(n_anchors, n_nodes)
                   if any((i, j) in cgn_4.edges or (j, i) in cgn_4.edges
                         for j in range(n_anchors)))
print(f"Direct anchor connections: {direct_anchor}/{n_unknowns}")

# Set true positions and optimize
cgn_4.set_true_positions({i: true_positions_4[i] for i in range(n_anchors, n_nodes)})

print("\nOptimizing...")
start_time = time.time()
results_4 = cgn_4.optimize()
elapsed = time.time() - start_time

print(f"\nResults with 4 corner anchors:")
print(f"  Iterations: {results_4['iterations']}")
print(f"  Converged: {results_4['converged']}")
print(f"  Time: {elapsed:.2f}s")

if 'position_errors' in results_4:
    print(f"  RMSE: {results_4['position_errors']['rmse']*100:.1f}cm")
    print(f"  Mean error: {results_4['position_errors']['mean']*100:.1f}cm")
    print(f"  Max error: {results_4['position_errors']['max']*100:.1f}cm")

# Test 2: 5 anchors (corners + center)
print("\n" + "=" * 70)
print("TEST 2: 5 ANCHORS (CORNERS + CENTER)")
print("=" * 70)

n_anchors_5 = 5
n_nodes_5 = 30  # Keep total nodes same

# 5 anchors: corners + center
anchor_positions_5 = np.array([
    [0, 0],
    [area_size, 0],
    [area_size, area_size],
    [0, area_size],
    [area_size/2, area_size/2]  # Center anchor for better geometry
])

# Fewer unknowns now
n_unknowns_5 = n_nodes_5 - n_anchors_5
grid_size_5 = int(np.sqrt(n_unknowns_5))
x_positions_5 = np.linspace(5, area_size-5, grid_size_5)
y_positions_5 = np.linspace(5, area_size-5, grid_size_5)

unknown_positions_5 = []
for x in x_positions_5:
    for y in y_positions_5:
        unknown_positions_5.append([x, y])
        if len(unknown_positions_5) >= n_unknowns_5:
            break
    if len(unknown_positions_5) >= n_unknowns_5:
        break

unknown_positions_5 = np.array(unknown_positions_5[:n_unknowns_5])
true_positions_5 = np.vstack([anchor_positions_5, unknown_positions_5])

config_5 = ConsensusGNConfig(
    max_iterations=200,
    consensus_gain=0.1,
    step_size=0.3,
    gradient_tol=1e-5,
    step_tol=1e-6,
    verbose=False
)

cgn_5 = ConsensusGaussNewton(config_5)

# Add nodes
for i in range(n_nodes_5):
    state = np.zeros(5)
    if i < n_anchors_5:
        state[:2] = true_positions_5[i]
        cgn_5.add_node(i, state, is_anchor=True)
    else:
        state[:2] = true_positions_5[i] + np.random.normal(0, init_noise_std, 2)
        cgn_5.add_node(i, state, is_anchor=False)

# Add edges and measurements
n_edges_5 = 0
for i in range(n_nodes_5):
    for j in range(i+1, n_nodes_5):
        dist = np.linalg.norm(true_positions_5[i] - true_positions_5[j])
        if dist <= comm_range:
            cgn_5.add_edge(i, j)
            n_edges_5 += 1
            meas_range = dist + np.random.normal(0, range_noise_std)
            cgn_5.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

print(f"Network edges: {n_edges_5}")
print(f"Average degree: {n_edges_5 * 2 / n_nodes_5:.1f}")

direct_anchor_5 = sum(1 for i in range(n_anchors_5, n_nodes_5)
                      if any((i, j) in cgn_5.edges or (j, i) in cgn_5.edges
                            for j in range(n_anchors_5)))
print(f"Direct anchor connections: {direct_anchor_5}/{n_unknowns_5}")

cgn_5.set_true_positions({i: true_positions_5[i] for i in range(n_anchors_5, n_nodes_5)})

print("\nOptimizing...")
start_time = time.time()
results_5 = cgn_5.optimize()
elapsed_5 = time.time() - start_time

print(f"\nResults with 5 anchors (corners + center):")
print(f"  Iterations: {results_5['iterations']}")
print(f"  Converged: {results_5['converged']}")
print(f"  Time: {elapsed_5:.2f}s")

if 'position_errors' in results_5:
    print(f"  RMSE: {results_5['position_errors']['rmse']*100:.1f}cm")
    print(f"  Mean error: {results_5['position_errors']['mean']*100:.1f}cm")
    print(f"  Max error: {results_5['position_errors']['max']*100:.1f}cm")

# Test 3: Even better initial guesses
print("\n" + "=" * 70)
print("TEST 3: EXCELLENT INITIAL GUESSES (within 50cm)")
print("=" * 70)

config_best = ConsensusGNConfig(
    max_iterations=200,
    consensus_gain=0.15,  # Can use slightly higher gain with better init
    step_size=0.4,
    gradient_tol=1e-5,
    step_tol=1e-6,
    verbose=False
)

cgn_best = ConsensusGaussNewton(config_best)

# Use 5-anchor setup with excellent initial guesses
for i in range(n_nodes_5):
    state = np.zeros(5)
    if i < n_anchors_5:
        state[:2] = true_positions_5[i]
        cgn_best.add_node(i, state, is_anchor=True)
    else:
        # Excellent initial guess - within 50cm
        state[:2] = true_positions_5[i] + np.random.normal(0, 0.5, 2)
        cgn_best.add_node(i, state, is_anchor=False)

# Same edges and measurements
for i in range(n_nodes_5):
    for j in range(i+1, n_nodes_5):
        dist = np.linalg.norm(true_positions_5[i] - true_positions_5[j])
        if dist <= comm_range:
            cgn_best.add_edge(i, j)
            meas_range = dist + np.random.normal(0, range_noise_std)
            cgn_best.add_measurement(ToAFactorMeters(i, j, meas_range, range_noise_std**2))

cgn_best.set_true_positions({i: true_positions_5[i] for i in range(n_anchors_5, n_nodes_5)})

print("Optimizing with excellent initial guesses...")
start_time = time.time()
results_best = cgn_best.optimize()
elapsed_best = time.time() - start_time

print(f"\nResults with excellent initialization:")
print(f"  Iterations: {results_best['iterations']}")
print(f"  Converged: {results_best['converged']}")
print(f"  Time: {elapsed_best:.2f}s")

if 'position_errors' in results_best:
    print(f"  RMSE: {results_best['position_errors']['rmse']*100:.1f}cm")
    print(f"  Mean error: {results_best['position_errors']['mean']*100:.1f}cm")
    print(f"  Max error: {results_best['position_errors']['max']*100:.1f}cm")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: 30 NODES OVER 50x50m UNDER IDEAL CONDITIONS")
print("=" * 70)

print("\nConfiguration          | Iterations | RMSE (cm) | Converged")
print("-" * 60)

rmse_4 = results_4['position_errors']['rmse']*100 if 'position_errors' in results_4 else float('inf')
rmse_5 = results_5['position_errors']['rmse']*100 if 'position_errors' in results_5 else float('inf')
rmse_best = results_best['position_errors']['rmse']*100 if 'position_errors' in results_best else float('inf')

print(f"4 corner anchors       | {results_4['iterations']:10d} | {rmse_4:9.1f} | {results_4['converged']}")
print(f"5 anchors (+ center)   | {results_5['iterations']:10d} | {rmse_5:9.1f} | {results_5['converged']}")
print(f"5 anchors + best init  | {results_best['iterations']:10d} | {rmse_best:9.1f} | {results_best['converged']}")

print(f"""
KEY FINDINGS:
1. Best achievable RMSE: {min(rmse_4, rmse_5, rmse_best):.1f}cm
2. Iterations to converge: {results_best['iterations'] if results_best['converged'] else '>200'}
3. Center anchor improves: {(rmse_4 - rmse_5)/rmse_4*100:.1f}% if rmse_4 > rmse_5 else 'No'
4. Better init helps: {(rmse_5 - rmse_best)/rmse_5*100:.1f}% if rmse_5 > rmse_best else 'No'

IDEAL CONDITIONS:
✓ Communication range: {comm_range}m (covers half the area)
✓ Measurement noise: {range_noise_std*100}cm (very low)
✓ Initial guess quality: 50cm-2m from truth
✓ Non-collinear anchors (with center anchor)
✓ Grid distribution of unknowns
""")