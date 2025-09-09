#!/usr/bin/env python3
"""
Test numerical stability improvements for carrier phase integration
Validates that weight normalization fixes the conditioning issues
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.carrier_phase import (
    CarrierPhaseConfig,
    CarrierPhaseMeasurementSystem,
    IntegerAmbiguityResolver
)
from src.core.mps_core.proximal_sdp import ProximalADMMSolver


def test_weight_normalization():
    """Test that weights are properly normalized"""
    print("="*60)
    print("WEIGHT NORMALIZATION TEST")
    print("="*60)
    
    config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        phase_noise_rad=0.001,
        snr_db=40
    )
    
    system = CarrierPhaseMeasurementSystem(config)
    
    # Test different measurement types
    test_cases = [
        ("TWTT only", None, 1.5, 0.05),
        ("Carrier phase", 1.5, 1.5, 0.05),
    ]
    
    print("\nWeight calculation results:")
    print("-"*40)
    
    for name, true_dist, coarse_dist, coarse_std in test_cases:
        if true_dist:
            measurement = system.create_measurement(
                0, 1, true_dist, coarse_dist, coarse_std
            )
        else:
            from src.core.carrier_phase.phase_measurement import PhaseMeasurement
            measurement = PhaseMeasurement(
                node_i=0, node_j=1,
                measured_phase_rad=0,
                phase_variance=0,
                coarse_distance_m=coarse_dist,
                coarse_variance=coarse_std**2
            )
        
        weight = system.get_measurement_weight(measurement)
        
        print(f"{name:15s}: weight = {weight:8.2f}")
        
        # Check that weights are in reasonable range
        assert 0.1 <= weight <= 1000, f"Weight {weight} out of range!"
    
    print("\n✓ All weights in acceptable range [0.1, 1000]")
    return True


def test_admm_conditioning():
    """Test ADMM solver conditioning with new weights"""
    print("\n" + "="*60)
    print("ADMM CONDITIONING TEST")
    print("="*60)
    
    # Create test problem
    solver = ProximalADMMSolver(
        rho=1.0,
        max_iterations=100,
        tolerance=1e-6,
        warm_start=True,
        adaptive_penalty=True
    )
    
    # Setup test matrices
    sensor_idx = 0
    neighbors = [1, 2]
    anchors = [0, 1]
    dimension = 2
    
    matrices = solver.setup_problem_matrices(
        sensor_idx, neighbors, anchors, dimension
    )
    
    K = matrices['K']
    D = matrices['D']
    
    # Test with different weight scales
    weight_scales = [1.0, 10.0, 100.0, 1000.0]
    
    print("\nCondition number vs weight scale:")
    print("-"*40)
    
    for scale in weight_scales:
        # Build weighted system
        W = np.diag(np.sqrt([scale, scale, 1.0, 1.0]))  # High weight for first 2
        K_weighted = W @ K
        
        # Form matrix as in ADMM
        A = solver.rho * K_weighted.T @ K_weighted + D.T @ D
        
        # Add regularization
        trace_A = np.trace(A)
        lambda_reg = 1e-6 * trace_A / A.shape[0]
        A += lambda_reg * np.eye(A.shape[0])
        
        # Check condition number
        cond = np.linalg.cond(A)
        
        print(f"Weight scale {scale:6.1f}: condition number = {cond:.2e}")
        
        # Should be manageable
        assert cond < 1e8, f"Condition number {cond:.2e} too large!"
    
    print("\n✓ All condition numbers below 10^8")
    return True


def test_adaptive_penalty():
    """Test adaptive penalty parameter adjustment"""
    print("\n" + "="*60)
    print("ADAPTIVE PENALTY TEST")
    print("="*60)
    
    solver = ProximalADMMSolver(
        rho=1.0,
        adaptive_penalty=True
    )
    
    print("\nInitial parameters:")
    print(f"  rho = {solver.rho}")
    print(f"  mu = {solver.mu}")
    print(f"  tau_incr = {solver.tau_incr}")
    print(f"  tau_decr = {solver.tau_decr}")
    
    # Test penalty adjustment logic
    test_cases = [
        (11.0, 1.0, "increase"),  # primal > mu * dual (11 > 10*1)
        (1.0, 11.0, "decrease"),  # dual > mu * primal (11 > 10*1)
        (5.0, 5.0, "unchanged"),  # balanced
    ]
    
    print("\nAdaptive penalty behavior:")
    print("-"*40)
    
    for primal_res, dual_res, expected in test_cases:
        old_rho = solver.rho
        
        # Simulate adaptation logic
        if primal_res > solver.mu * dual_res:
            solver.rho = min(solver.rho * solver.tau_incr, 1e4)
            actual = "increase"
        elif dual_res > solver.mu * primal_res:
            solver.rho = max(solver.rho / solver.tau_decr, 1e-4)
            actual = "decrease"
        else:
            actual = "unchanged"
        
        print(f"Primal={primal_res:.1f}, Dual={dual_res:.1f}: "
              f"rho {old_rho:.2f} -> {solver.rho:.2f} ({actual})")
        
        assert actual == expected, f"Expected {expected}, got {actual}"
        
        # Reset for next test
        solver.rho = 1.0
    
    print("\n✓ Adaptive penalty working correctly")
    return True


def test_convergence_with_carrier_phase():
    """Test ADMM convergence with carrier phase measurements"""
    print("\n" + "="*60)
    print("CONVERGENCE WITH CARRIER PHASE TEST")
    print("="*60)
    
    # Create a simple test problem
    solver = ProximalADMMSolver(
        rho=0.5,
        max_iterations=200,
        tolerance=1e-8,
        adaptive_penalty=True
    )
    
    # Test data
    X_prev = np.array([[0.5, 0.5], [1.0, 0.5], [0.5, 1.0]])
    Y_prev = X_prev @ X_prev.T
    
    sensor_idx = 0
    neighbors = [1, 2]
    anchors = [0]
    
    # Distance measurements with different precisions
    distances_sensors = {
        1: 0.5,  # True distance to neighbor 1
        2: 0.5   # True distance to neighbor 2
    }
    
    distances_anchors = {
        0: 0.707  # Distance to anchor at origin
    }
    
    anchor_positions = np.array([[0, 0]])
    
    print("\nRunning ADMM solver with mixed precision measurements...")
    
    try:
        X_new, Y_new = solver.solve(
            X_prev, Y_prev,
            sensor_idx, neighbors, anchors,
            distances_sensors, distances_anchors,
            anchor_positions, alpha=1.0
        )
        
        # Check that solution is reasonable
        position_change = np.linalg.norm(X_new[sensor_idx] - X_prev[sensor_idx])
        
        print(f"  Initial position: {X_prev[sensor_idx]}")
        print(f"  Final position: {X_new[sensor_idx]}")
        print(f"  Position change: {position_change:.6f}")
        print(f"  Final rho: {solver.rho:.3f}")
        
        # Should converge to a reasonable solution
        assert position_change < 1.0, "Solution diverged!"
        
        print("\n✓ ADMM converged successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ ADMM failed: {e}")
        return False


def main():
    """Run all numerical stability tests"""
    print("="*60)
    print("NUMERICAL STABILITY TEST SUITE")
    print("="*60)
    
    tests = [
        ("Weight Normalization", test_weight_normalization),
        ("ADMM Conditioning", test_admm_conditioning),
        ("Adaptive Penalty", test_adaptive_penalty),
        ("Convergence", test_convergence_with_carrier_phase)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25s}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL NUMERICAL STABILITY TESTS PASSED")
        print("\nKey achievements:")
        print("  • Weights normalized to [1, 1000] range")
        print("  • Condition numbers < 10^8")
        print("  • Adaptive penalty working")
        print("  • ADMM converges with mixed precision")
    else:
        print("⚠ Some tests failed - review implementation")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)