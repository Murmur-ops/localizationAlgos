#!/usr/bin/env python3
"""
Test FTL in 50x50m area with IDEAL conditions
Gradually introduce imperfections to understand their impact
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.solver import FactorGraph

def run_ideal_simulation(
    n_anchors=4,
    n_unknowns=9,
    area_size=50.0,
    range_std_m=0.01,  # 1cm ranging accuracy
    seed=42
):
    """Run simulation with ideal conditions"""

    np.random.seed(seed)
    n_nodes = n_anchors + n_unknowns

    print("="*70)
    print(f"IDEAL 50×50m Simulation")
    print("="*70)

    # Perfect anchor placement at corners
    if n_anchors == 4:
        anchor_positions = [
            (0, 0),
            (area_size, 0),
            (area_size, area_size),
            (0, area_size)
        ]
    elif n_anchors == 8:
        # Corners + edge midpoints
        anchor_positions = [
            (0, 0), (area_size, 0), (area_size, area_size), (0, area_size),
            (area_size/2, 0), (area_size, area_size/2),
            (area_size/2, area_size), (0, area_size/2)
        ]
    else:
        raise ValueError("Only 4 or 8 anchors supported")

    # Place unknowns on regular grid
    grid_size = int(np.ceil(np.sqrt(n_unknowns)))
    unknown_positions = []
    spacing = area_size * 0.6 / (grid_size - 1) if grid_size > 1 else 0
    offset = area_size * 0.2

    for i in range(grid_size):
        for j in range(grid_size):
            if len(unknown_positions) < n_unknowns:
                x = offset + i * spacing
                y = offset + j * spacing
                unknown_positions.append((x, y))

    print(f"\nConfiguration:")
    print(f"  Area: {area_size}×{area_size} m")
    print(f"  Anchors: {n_anchors} at corners")
    print(f"  Unknowns: {n_unknowns} on {grid_size}×{grid_size} grid")
    print(f"  Range σ: {range_std_m*100:.1f} cm")

    # Build true node positions (no clock errors in ideal case)
    true_positions = {}
    for i in range(n_anchors):
        true_positions[i] = np.array([anchor_positions[i][0], anchor_positions[i][1], 0, 0, 0])

    for i in range(n_unknowns):
        true_positions[n_anchors + i] = np.array([
            unknown_positions[i][0],
            unknown_positions[i][1],
            0, 0, 0  # No clock errors
        ])

    # Generate IDEAL measurements (all pairs, perfect LOS)
    measurements = []
    toa_variance = (range_std_m / 3e8)**2  # Convert range std to ToA variance

    print(f"\nGenerating ideal measurements...")
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            pos_i = true_positions[i][:2]
            pos_j = true_positions[j][:2]

            # True distance
            true_dist = np.linalg.norm(pos_j - pos_i)

            # Perfect ToA with small Gaussian noise
            true_toa = true_dist / 3e8
            noise = np.random.normal(0, np.sqrt(toa_variance))
            measured_toa = true_toa + noise

            measurements.append({
                'i': i,
                'j': j,
                'true_toa': true_toa,
                'measured_toa': measured_toa,
                'true_dist': true_dist
            })

    print(f"  Generated {len(measurements)} perfect measurements")

    # Build factor graph
    print(f"\nOptimizing with ideal measurements...")
    graph = FactorGraph()

    # Add nodes
    for i in range(n_nodes):
        is_anchor = i < n_anchors
        if is_anchor:
            initial = true_positions[i]
        else:
            # Start from slightly perturbed position
            true_pos = true_positions[i]
            initial = np.array([
                true_pos[0] + np.random.randn() * 1.0,
                true_pos[1] + np.random.randn() * 1.0,
                0, 0, 0
            ])
        graph.add_node(i, initial, is_anchor=is_anchor)

    # Add ToA factors
    for meas in measurements:
        graph.add_toa_factor(
            meas['i'], meas['j'],
            meas['measured_toa'],
            toa_variance
        )

    # Optimize
    result = graph.optimize(
        max_iterations=100,
        tolerance=1e-8,
        verbose=False
    )

    print(f"  Converged: {result.converged} in {result.iterations} iterations")
    print(f"  Final cost: {result.final_cost:.2e}")

    # Calculate position errors
    position_errors = []
    for i in range(n_anchors, n_nodes):
        true_pos = true_positions[i][:2]
        est_pos = result.estimates[i][:2]
        error = np.linalg.norm(est_pos - true_pos)
        position_errors.append(error)

    if position_errors:
        rmse = np.sqrt(np.mean(np.array(position_errors)**2))
        mae = np.mean(position_errors)
        max_error = max(position_errors)

        print(f"\n{'='*70}")
        print(f"RESULTS (Ideal Conditions):")
        print(f"  RMSE:  {rmse*100:.2f} cm")
        print(f"  MAE:   {mae*100:.2f} cm")
        print(f"  Max:   {max_error*100:.2f} cm")

        # Theoretical bound (simplified)
        gdop = np.sqrt(n_unknowns / n_anchors)
        theoretical_rmse = range_std_m * gdop
        print(f"\nTheoretical:")
        print(f"  GDOP: ~{gdop:.1f}")
        print(f"  Expected RMSE: ~{theoretical_rmse*100:.1f} cm")
        print(f"  Efficiency: {theoretical_rmse/rmse*100:.1f}%")
        print("="*70)

        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'converged': result.converged,
            'iterations': result.iterations,
            'measurements': measurements,
            'true_positions': true_positions,
            'estimates': result.estimates
        }

    return None


def add_los_nlos_effects(measurements, los_probability=0.7, nlos_bias_range=(0.5, 2.0)):
    """Add NLOS effects to measurements"""

    modified_measurements = []
    n_los = 0

    for meas in measurements:
        new_meas = meas.copy()

        # Determine if LOS or NLOS
        is_los = np.random.rand() < los_probability

        if is_los:
            n_los += 1
        else:
            # Add positive bias for NLOS
            bias = np.random.uniform(nlos_bias_range[0], nlos_bias_range[1])
            bias_time = bias / 3e8
            new_meas['measured_toa'] += bias_time
            new_meas['is_los'] = False

        new_meas['is_los'] = is_los
        modified_measurements.append(new_meas)

    print(f"  LOS/NLOS: {n_los}/{len(measurements)-n_los} ({100*n_los/len(measurements):.1f}% LOS)")
    return modified_measurements


def add_clock_errors(true_positions, clock_std_s=1e-6):
    """Add clock bias to unknown nodes"""

    modified_positions = true_positions.copy()

    for node_id in modified_positions:
        if node_id >= 4:  # Unknown nodes (assuming first 4 are anchors)
            # Add random clock bias
            clock_bias = np.random.normal(0, clock_std_s)
            modified_positions[node_id][2] = clock_bias

    print(f"  Clock bias σ: {clock_std_s*1e6:.1f} µs")
    return modified_positions


def run_comparison():
    """Compare ideal vs realistic conditions"""

    print("\n" + "="*70)
    print("COMPARING IDEAL VS REALISTIC CONDITIONS")
    print("="*70)

    # 1. IDEAL conditions
    print("\n1. IDEAL CONDITIONS (perfect LOS, no clock errors)")
    ideal_results = run_ideal_simulation(
        n_anchors=4,
        n_unknowns=16,
        range_std_m=0.01  # 1cm
    )

    # 2. Add measurement noise
    print("\n2. WITH MEASUREMENT NOISE (10cm ranging)")
    noisy_results = run_ideal_simulation(
        n_anchors=4,
        n_unknowns=16,
        range_std_m=0.10  # 10cm
    )

    # 3. Add more anchors
    print("\n3. WITH MORE ANCHORS (8 instead of 4)")
    more_anchors_results = run_ideal_simulation(
        n_anchors=8,
        n_unknowns=16,
        range_std_m=0.10
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS:")
    print("="*70)
    print(f"{'Scenario':<30} {'RMSE (cm)':<15} {'Iterations'}")
    print("-"*60)
    print(f"{'Ideal (1cm noise)':<30} {ideal_results['rmse']*100:>10.2f} cm  {ideal_results['iterations']:>10d}")
    print(f"{'With 10cm noise':<30} {noisy_results['rmse']*100:>10.2f} cm  {noisy_results['iterations']:>10d}")
    print(f"{'8 anchors + 10cm noise':<30} {more_anchors_results['rmse']*100:>10.2f} cm  {more_anchors_results['iterations']:>10d}")
    print("="*70)


if __name__ == "__main__":
    # Run comparison
    run_comparison()

    # You can also run individual tests
    print("\n\nTesting with different configurations...")

    # Test effect of number of unknowns
    print("\n" + "="*70)
    print("EFFECT OF NETWORK DENSITY")
    print("="*70)

    for n_unknowns in [4, 9, 16, 25]:
        print(f"\n{n_unknowns} unknown nodes:")
        results = run_ideal_simulation(
            n_anchors=4,
            n_unknowns=n_unknowns,
            range_std_m=0.05  # 5cm
        )