#!/usr/bin/env python3
"""
Analyze what goes into ideal scenario and compare to CRLB
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.solver import FactorGraph

def compute_crlb_position_bound(anchor_positions, unknown_position, range_variance):
    """
    Compute CRLB for position estimation

    The Fisher Information Matrix for 2D positioning:
    FIM = sum_i (1/σ²_i) * g_i * g_i^T
    where g_i is the unit vector from unknown to anchor i

    CRLB = inverse(FIM)
    """

    n_anchors = len(anchor_positions)
    FIM = np.zeros((2, 2))

    for anchor_pos in anchor_positions:
        # Direction vector from unknown to anchor
        dx = anchor_pos[0] - unknown_position[0]
        dy = anchor_pos[1] - unknown_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Unit direction vector
        g = np.array([dx/distance, dy/distance])

        # Add to Fisher Information Matrix
        FIM += np.outer(g, g) / range_variance

    # CRLB is inverse of FIM
    try:
        crlb_matrix = np.linalg.inv(FIM)
        # Position error bound (trace gives total variance)
        position_variance = np.trace(crlb_matrix)
        return np.sqrt(position_variance)
    except:
        return float('inf')


def analyze_ideal_scenario():
    """Analyze components of ideal scenario"""

    print("="*70)
    print("ANALYZING IDEAL SCENARIO COMPONENTS")
    print("="*70)

    area_size = 50.0

    # 1. WHAT GOES INTO "IDEAL"?
    print("\n1. COMPONENTS OF IDEAL SCENARIO:")
    print("-"*40)
    print("a) GEOMETRY:")
    print("   - Perfect anchor placement at corners")
    print("   - Regular grid of unknown nodes")
    print("   - All-to-all measurements (full connectivity)")

    print("\nb) CHANNEL:")
    print("   - Line-of-sight (LOS) only")
    print("   - No multipath")
    print("   - No NLOS bias")

    print("\nc) MEASUREMENTS:")
    print("   - Gaussian noise only (no bias)")
    print("   - Known, constant variance")
    print("   - Independent measurements")

    print("\nd) CLOCK:")
    print("   - No clock bias")
    print("   - No clock drift")
    print("   - No carrier frequency offset (CFO)")

    print("\ne) SOLVER:")
    print("   - Least squares optimization")
    print("   - Perfect linearization (close to true solution)")

    # 2. CRLB ANALYSIS
    print("\n2. CRAMÉR-RAO LOWER BOUND (CRLB) ANALYSIS:")
    print("-"*40)

    # Test different ranging accuracies
    range_stds = [0.01, 0.05, 0.10, 0.20, 0.50]  # meters

    # 4-anchor configuration
    anchors = [
        (0, 0),
        (area_size, 0),
        (area_size, area_size),
        (0, area_size)
    ]

    print("\nFor 4 anchors at corners of 50x50m area:")
    print(f"Anchor positions: {anchors}")

    # Test positions: center and off-center
    test_positions = [
        (area_size/2, area_size/2, "Center"),
        (area_size*0.25, area_size*0.25, "Near corner"),
        (area_size*0.1, area_size*0.5, "Near edge")
    ]

    for x, y, label in test_positions:
        print(f"\nPosition: {label} ({x:.1f}, {y:.1f})")
        print("Range σ (m) | CRLB σ_pos (m) | GDOP")
        print("-"*40)

        for range_std in range_stds:
            range_var = range_std**2
            crlb_bound = compute_crlb_position_bound(anchors, (x, y), range_var)
            gdop = crlb_bound / range_std if range_std > 0 else 0
            print(f"  {range_std:6.3f}    |    {crlb_bound:6.3f}      | {gdop:5.2f}")

    # 3. WHY ACTUAL PERFORMANCE DIFFERS FROM CRLB
    print("\n3. WHY ACTUAL DIFFERS FROM CRLB:")
    print("-"*40)
    print("a) CRLB assumes:")
    print("   - Unbiased estimator")
    print("   - Estimator achieves the bound (efficient)")
    print("   - Known measurement model")
    print("   - Linear or well-linearized problem")

    print("\nb) Real factors causing degradation:")
    print("   - Nonlinear optimization (may get stuck in local minima)")
    print("   - Linearization errors")
    print("   - Numerical precision issues")
    print("   - Incomplete convergence")

    # 4. THEORETICAL vs SIMULATED
    print("\n4. TESTING IDEAL SIMULATION VS CRLB:")
    print("-"*40)

    # Run a simple ideal test
    np.random.seed(42)
    n_anchors = 4
    n_unknowns = 1  # Single node for clear comparison
    range_std = 0.01  # 1cm

    # Single unknown at center
    unknown_pos = (area_size/2, area_size/2)

    # Theoretical CRLB
    theoretical_bound = compute_crlb_position_bound(
        anchors, unknown_pos, range_std**2
    )

    print(f"\nSingle node at center with {range_std*100:.1f}cm ranging:")
    print(f"  Theoretical CRLB: {theoretical_bound*100:.2f} cm")

    # Run simulation
    n_trials = 100
    errors = []

    for trial in range(n_trials):
        # Build factor graph
        graph = FactorGraph()

        # Add anchors
        for i, (x, y) in enumerate(anchors):
            graph.add_node(i, np.array([x, y, 0, 0, 0]), is_anchor=True)

        # Add unknown with small perturbation
        initial = np.array([
            unknown_pos[0] + np.random.randn() * 1.0,
            unknown_pos[1] + np.random.randn() * 1.0,
            0, 0, 0
        ])
        graph.add_node(4, initial, is_anchor=False)

        # Add measurements with Gaussian noise
        toa_var = (range_std / 3e8)**2
        for i in range(4):
            anchor = np.array(anchors[i])
            unknown = np.array(unknown_pos)
            true_dist = np.linalg.norm(anchor - unknown)
            true_toa = true_dist / 3e8

            # Add Gaussian noise
            noise = np.random.normal(0, np.sqrt(toa_var))
            measured_toa = true_toa + noise

            graph.add_toa_factor(i, 4, measured_toa, toa_var)

        # Optimize
        result = graph.optimize(max_iterations=50, verbose=False)

        # Calculate error
        est_pos = result.estimates[4][:2]
        error = np.linalg.norm(est_pos - unknown_pos)
        errors.append(error)

    empirical_rmse = np.sqrt(np.mean(np.array(errors)**2))
    empirical_std = np.std(errors)

    print(f"  Empirical RMSE: {empirical_rmse*100:.2f} cm")
    print(f"  Empirical STD: {empirical_std*100:.2f} cm")
    print(f"  Efficiency: {theoretical_bound/empirical_rmse*100:.1f}%")

    # 5. GDOP ANALYSIS
    print("\n5. GEOMETRIC DILUTION OF PRECISION (GDOP):")
    print("-"*40)

    # Create grid of test points
    n_grid = 11
    x_grid = np.linspace(5, 45, n_grid)
    y_grid = np.linspace(5, 45, n_grid)

    gdop_map = np.zeros((n_grid, n_grid))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            # Compute CRLB for unit variance
            crlb = compute_crlb_position_bound(anchors, (x, y), 1.0)
            gdop_map[j, i] = crlb  # This is GDOP when variance=1

    print(f"GDOP across 50x50m area (4 corner anchors):")
    print(f"  Minimum GDOP: {np.min(gdop_map):.2f} (at center)")
    print(f"  Maximum GDOP: {np.max(gdop_map):.2f} (at corners)")
    print(f"  Mean GDOP: {np.mean(gdop_map):.2f}")

    # Plot GDOP map
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.contourf(x_grid, y_grid, gdop_map, levels=20, cmap='RdYlBu_r')
    ax.contour(x_grid, y_grid, gdop_map, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Mark anchors
    for x, y in anchors:
        ax.plot(x, y, 'r^', markersize=15, label='Anchor' if x==0 and y==0 else '')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('GDOP Map for 4 Corner Anchors')
    ax.legend()
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('GDOP')

    plt.tight_layout()
    plt.savefig('gdop_map_50x50.png', dpi=150)
    print(f"\n✓ GDOP map saved as 'gdop_map_50x50.png'")

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. CRLB provides fundamental lower bound on position error")
    print("2. GDOP varies with position (best at center, worst at edges)")
    print("3. Actual performance typically 50-100% of CRLB in ideal conditions")
    print("4. More anchors reduce GDOP and improve accuracy")
    print("5. Position error ≈ GDOP × ranging_error")


if __name__ == "__main__":
    analyze_ideal_scenario()