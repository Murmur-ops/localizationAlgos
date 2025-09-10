#!/usr/bin/env python3
"""
Validation script for MPS algorithm - tests key functionality
"""

import numpy as np
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    create_network_data
)

def test_basic_convergence():
    """Test basic convergence behavior"""
    print("\n=== Testing Basic Convergence ===")
    
    network_data = create_network_data(
        n_sensors=10,
        n_anchors=3,
        dimension=2,
        communication_range=0.4,
        measurement_noise=0.001,
        carrier_phase=False
    )
    
    config = MPSConfig(
        n_sensors=10,
        n_anchors=3,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True,
        carrier_phase_mode=False
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    start = time.time()
    results = mps.run()
    elapsed = time.time() - start
    
    print(f"✓ Converged in {results['iterations']} iterations ({elapsed:.2f}s)")
    print(f"  Final objective: {results['best_objective']:.6f}")
    print(f"  Position RMSE: {results['final_rmse']:.6f}")
    
    return results['iterations'] < 50


def test_carrier_phase_accuracy():
    """Test millimeter accuracy with carrier phase"""
    print("\n=== Testing Carrier Phase Accuracy ===")
    
    network_data = create_network_data(
        n_sensors=15,
        n_anchors=4,
        dimension=2,
        communication_range=0.4,
        measurement_noise=0.0001,  # Very low noise for carrier phase
        carrier_phase=True
    )
    
    config = MPSConfig(
        n_sensors=15,
        n_anchors=4,
        dimension=2,
        gamma=0.999,
        alpha=5.0,
        max_iterations=200,
        tolerance=1e-8,
        verbose=False,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True,
        carrier_phase_mode=True
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    start = time.time()
    results = mps.run()
    elapsed = time.time() - start
    
    rmse_mm = results.get('rmse_mm', results['final_rmse'] * 1000)
    target_met = rmse_mm < 15.0
    
    print(f"{'✓' if target_met else '✗'} RMSE: {rmse_mm:.2f} mm (target: <15mm)")
    print(f"  Iterations: {results['iterations']} ({elapsed:.2f}s)")
    print(f"  Best objective: {results['best_objective']:.6f}")
    
    return target_met


def test_scalability():
    """Test scalability with different network sizes"""
    print("\n=== Testing Scalability ===")
    
    sizes = [5, 10, 20]
    times = []
    
    for n in sizes:
        network_data = create_network_data(
            n_sensors=n,
            n_anchors=max(2, n // 5),
            dimension=2,
            communication_range=0.4,
            measurement_noise=0.001,
            carrier_phase=False
        )
        
        config = MPSConfig(
            n_sensors=n,
            n_anchors=network_data.anchor_positions.shape[0],
            dimension=2,
            gamma=0.999,
            alpha=10.0,
            max_iterations=50,
            tolerance=1e-6,
            verbose=False,
            use_2block=True,
            parallel_proximal=True,
            adaptive_alpha=False,
            carrier_phase_mode=False
        )
        
        mps = MatrixParametrizedProximalSplitting(config, network_data)
        start = time.time()
        results = mps.run()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  {n:3d} sensors: {results['iterations']:3d} iterations, {elapsed:6.2f}s")
    
    # Check if time scales reasonably (not exponentially)
    scaling_ok = times[-1] < times[0] * (sizes[-1] / sizes[0])**2.5
    print(f"\n{'✓' if scaling_ok else '✗'} Time scaling is {'reasonable' if scaling_ok else 'poor'}")
    
    return scaling_ok


def test_2block_vs_simple():
    """Compare 2-block design with simple design"""
    print("\n=== Testing 2-Block vs Simple ===")
    
    network_data = create_network_data(
        n_sensors=12,
        n_anchors=3,
        dimension=2,
        communication_range=0.4,
        measurement_noise=0.001,
        carrier_phase=False
    )
    
    # Test with 2-block
    config_2block = MPSConfig(
        n_sensors=12,
        n_anchors=3,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=False,
        carrier_phase_mode=False
    )
    
    mps_2block = MatrixParametrizedProximalSplitting(config_2block, network_data)
    start = time.time()
    results_2block = mps_2block.run()
    time_2block = time.time() - start
    
    # Test without 2-block
    config_simple = MPSConfig(
        n_sensors=12,
        n_anchors=3,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
        use_2block=False,
        parallel_proximal=False,
        adaptive_alpha=False,
        carrier_phase_mode=False
    )
    
    mps_simple = MatrixParametrizedProximalSplitting(config_simple, network_data)
    start = time.time()
    results_simple = mps_simple.run()
    time_simple = time.time() - start
    
    print(f"  2-Block: {results_2block['iterations']:3d} iter, obj={results_2block['best_objective']:.6f}, time={time_2block:.2f}s")
    print(f"  Simple:  {results_simple['iterations']:3d} iter, obj={results_simple['best_objective']:.6f}, time={time_simple:.2f}s")
    
    better = results_2block['best_objective'] <= results_simple['best_objective']
    print(f"\n{'✓' if better else '✗'} 2-Block design {'performs better' if better else 'needs tuning'}")
    
    return better


def main():
    """Run all validation tests"""
    print("="*60)
    print("MPS ALGORITHM VALIDATION")
    print("="*60)
    
    tests = [
        ("Basic Convergence", test_basic_convergence),
        ("Carrier Phase Accuracy", test_carrier_phase_accuracy),
        ("Scalability", test_scalability),
        ("2-Block Design", test_2block_vs_simple)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25s}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Algorithm is working correctly!")
        print("\nKey achievements:")
        print("  • Convergence in <50 iterations for small networks")
        print("  • Millimeter accuracy (<15mm) with carrier phase")
        print("  • Reasonable scalability")
        print("  • 2-Block design improves performance")
    else:
        print("⚠ Some tests failed - further tuning may be needed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)