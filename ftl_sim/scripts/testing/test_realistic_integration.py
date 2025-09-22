#!/usr/bin/env python3
"""
Integration test with realistic UWB parameters
Demonstrates the solver working without artificial variance floors or weight caps
"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver, OptimizationConfig

def test_realistic_uwb_network():
    """Test with realistic UWB network parameters"""

    # UWB parameters
    c = 299792458.0  # m/s
    range_std_m = 0.15  # 15cm standard deviation (realistic for UWB)
    clock_bias_std_ns = 10.0  # 10ns initial uncertainty
    clock_drift_std_ppb = 50.0  # 50ppb drift uncertainty

    # Network layout - 4 anchors, 2 unknown nodes
    anchors = {
        0: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([10.0, 10.0, 0.0, 0.0, 0.0]),
        3: np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    }

    # Unknown nodes with initial guesses
    unknowns = {
        4: np.array([3.0, 3.0, 5.0, 0.0, 0.0]),  # Initial guess with 5ns bias
        5: np.array([7.0, 7.0, -3.0, 0.0, 0.0])  # Initial guess with -3ns bias
    }

    # True positions for generating measurements
    true_positions = {
        4: np.array([5.0, 3.0, 2.0, 0.0, 0.0]),  # 2ns bias
        5: np.array([6.0, 8.0, -1.0, 0.0, 0.0])  # -1ns bias
    }

    # Create solver
    config = OptimizationConfig(
        max_iterations=50,
        gradient_tol=1e-8
    )
    solver = SquareRootSolver(config)

    # Add nodes
    for id, state in anchors.items():
        solver.add_node(id, state, is_anchor=True)
    for id, state in unknowns.items():
        solver.add_node(id, state, is_anchor=False)

    # Generate measurements with noise
    np.random.seed(42)
    measurements = []

    for anchor_id, anchor_pos in anchors.items():
        for unknown_id, true_pos in true_positions.items():
            # Compute true range
            geometric_range = np.linalg.norm(true_pos[:2] - anchor_pos[:2])
            clock_contrib_m = true_pos[2] * c * 1e-9  # bias in meters
            true_range = geometric_range + clock_contrib_m

            # Add measurement noise
            meas_range = true_range + np.random.normal(0, range_std_m)

            # Add to solver
            solver.add_toa_factor(anchor_id, unknown_id, meas_range, range_std_m**2)
            measurements.append({
                'from': anchor_id,
                'to': unknown_id,
                'range': meas_range,
                'true_range': true_range
            })

    # Add clock priors (weak priors to regularize)
    for unknown_id in unknowns:
        solver.add_clock_prior(unknown_id, 0.0, 0.0,
                             clock_bias_std_ns**2, clock_drift_std_ppb**2)

    print("=" * 60)
    print("REALISTIC UWB NETWORK TEST")
    print("=" * 60)
    print(f"Range std: {range_std_m*100:.1f} cm")
    print(f"Clock bias std: {clock_bias_std_ns:.1f} ns")
    print(f"Number of measurements: {len(measurements)}")

    # Print initial errors
    print("\nINITIAL ERRORS:")
    for id in unknowns:
        pos_error = np.linalg.norm(unknowns[id][:2] - true_positions[id][:2])
        bias_error = abs(unknowns[id][2] - true_positions[id][2])
        print(f"  Node {id}: position={pos_error:.3f}m, bias={bias_error:.3f}ns")

    # Optimize
    print("\nOPTIMIZING...")
    result = solver.optimize(verbose=False)

    print(f"\nOPTIMIZATION RESULT:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final cost: {result.final_cost:.6f}")
    print(f"  Gradient norm: {result.gradient_norm:.2e}")
    print(f"  Reason: {result.convergence_reason}")

    # Print final errors
    print("\nFINAL ERRORS:")
    for id in unknowns:
        est = result.estimates[id]
        pos_error = np.linalg.norm(est[:2] - true_positions[id][:2])
        bias_error = abs(est[2] - true_positions[id][2])
        print(f"  Node {id}: position={pos_error:.3f}m, bias={bias_error:.3f}ns")
        print(f"    Estimated: [{est[0]:.3f}, {est[1]:.3f}]m, bias={est[2]:.3f}ns")
        print(f"    True:      [{true_positions[id][0]:.3f}, {true_positions[id][1]:.3f}]m, "
              f"bias={true_positions[id][2]:.3f}ns")

    # Verify accuracy is within expected bounds (3-sigma)
    print("\nACCURACY VERIFICATION:")
    all_good = True
    for id in unknowns:
        est = result.estimates[id]
        pos_error = np.linalg.norm(est[:2] - true_positions[id][:2])

        # Expected position accuracy from CRLB (simplified)
        # With 4 anchors and 15cm ranging, expect ~10-20cm position accuracy
        expected_pos_accuracy = 0.3  # 30cm (conservative)

        if pos_error < expected_pos_accuracy:
            print(f"  ✓ Node {id} position error ({pos_error:.3f}m) < {expected_pos_accuracy}m")
        else:
            print(f"  ✗ Node {id} position error ({pos_error:.3f}m) >= {expected_pos_accuracy}m")
            all_good = False

    # Check weights are reasonable (not approaching float64 limits)
    print("\nWEIGHT VERIFICATION:")
    weights = []
    for factor in solver.factors:
        if hasattr(factor, 'variance'):
            weight = 1.0 / factor.variance
            weights.append(weight)

    max_weight = max(weights)
    min_weight = min(weights)
    print(f"  Weight range: [{min_weight:.2e}, {max_weight:.2e}]")

    # Weights should be reasonable (not 1e18!)
    if max_weight < 1e6:
        print(f"  ✓ Maximum weight ({max_weight:.2e}) is reasonable")
    else:
        print(f"  ✗ Maximum weight ({max_weight:.2e}) is too large!")
        all_good = False

    # Check Hessian conditioning
    J_wh, r_wh, _, _ = solver._build_whitened_system()
    J_scaled, _ = solver._apply_state_scaling(J_wh)
    H = J_scaled.T @ J_scaled
    cond = np.linalg.cond(H)
    print(f"\n  Hessian condition number: {cond:.2e}")

    if cond < 1e10:
        print(f"  ✓ Conditioning ({cond:.2e}) is good")
    else:
        print(f"  ⚠ Conditioning ({cond:.2e}) is marginal but solver handles it")

    print("\n" + "=" * 60)
    if all_good and result.converged:
        print("SUCCESS: Solver handles realistic UWB without numerical issues!")
        print("No artificial variance floors or weight caps needed!")
    else:
        print("ISSUES DETECTED - Check details above")
    print("=" * 60)

    return result


if __name__ == "__main__":
    test_realistic_uwb_network()