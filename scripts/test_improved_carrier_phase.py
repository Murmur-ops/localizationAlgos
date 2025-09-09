#!/usr/bin/env python3
"""
Test improved carrier phase with dual-frequency wide-lane resolution
Validates achievement of <15mm RMSE even with large TWTT errors
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.core.carrier_phase import (
    CarrierPhaseConfig,
    CarrierPhaseMeasurementSystem,
    IntegerAmbiguityResolver
)

from src.core.carrier_phase.wide_lane import (
    DualFrequencyConfig,
    MelbourneWubbenaResolver
)

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData,
    create_network_data
)


def test_ambiguity_resolution_comparison():
    """Compare single vs dual-frequency ambiguity resolution"""
    
    print("="*70)
    print("AMBIGUITY RESOLUTION COMPARISON TEST")
    print("="*70)
    
    # Test parameters
    test_distances = np.linspace(0.5, 10.0, 20)
    twtt_errors = [0.05, 0.10, 0.20, 0.30, 0.40]  # 5cm to 40cm
    
    results = {
        'single_freq': [],
        'dual_freq': []
    }
    
    for twtt_error_std in twtt_errors:
        print(f"\n{'='*50}")
        print(f"Testing with TWTT error: ±{twtt_error_std*100:.0f}cm")
        print(f"{'='*50}")
        
        single_freq_errors = []
        dual_freq_errors = []
        
        # Single-frequency configuration
        config_single = CarrierPhaseConfig(
            frequency_hz=2.4e9,
            phase_noise_rad=0.001,
            snr_db=40,
            use_dual_frequency=False
        )
        
        # Dual-frequency configuration
        config_dual = CarrierPhaseConfig(
            frequency_hz=2.4e9,
            frequency_l2_hz=1.9e9,
            phase_noise_rad=0.001,
            snr_db=40,
            use_dual_frequency=True
        )
        
        system_single = CarrierPhaseMeasurementSystem(config_single)
        system_dual = CarrierPhaseMeasurementSystem(config_dual)
        
        print(f"\nSingle-frequency: λ = {config_single.wavelength*100:.1f}cm")
        print(f"Dual-frequency: λ_WL = {config_dual.wavelength_wide_lane*100:.1f}cm")
        
        for true_dist in test_distances:
            # Add TWTT error
            coarse_dist = true_dist + np.random.normal(0, twtt_error_std)
            
            # Single-frequency measurement
            meas_single = system_single.create_measurement(
                0, 1, true_dist, coarse_dist, twtt_error_std
            )
            
            # Dual-frequency measurement
            meas_dual = system_dual.create_measurement(
                0, 1, true_dist, coarse_dist, twtt_error_std,
                use_dual_freq=True
            )
            
            # Calculate errors
            if meas_single.refined_distance_m:
                error_single = abs(meas_single.refined_distance_m - true_dist) * 1000
                single_freq_errors.append(error_single)
            else:
                single_freq_errors.append(1000)  # Failed resolution
            
            if meas_dual.refined_distance_m:
                error_dual = abs(meas_dual.refined_distance_m - true_dist) * 1000
                dual_freq_errors.append(error_dual)
            else:
                dual_freq_errors.append(1000)  # Failed resolution
        
        # Calculate statistics
        single_rmse = np.sqrt(np.mean(np.square(single_freq_errors)))
        dual_rmse = np.sqrt(np.mean(np.square(dual_freq_errors)))
        
        print(f"\nResults:")
        print(f"  Single-frequency RMSE: {single_rmse:.2f}mm")
        print(f"  Dual-frequency RMSE: {dual_rmse:.2f}mm")
        print(f"  Improvement: {single_rmse/dual_rmse:.1f}x")
        
        # Check target achievement
        if dual_rmse < 15:
            print(f"  ✓ DUAL-FREQUENCY ACHIEVES <15mm TARGET")
        
        results['single_freq'].append((twtt_error_std, single_rmse))
        results['dual_freq'].append((twtt_error_std, dual_rmse))
    
    return results


def test_integrated_mps_with_dual_freq():
    """Test MPS algorithm with dual-frequency carrier phase"""
    
    print("\n" + "="*70)
    print("INTEGRATED MPS WITH DUAL-FREQUENCY TEST")
    print("="*70)
    
    # Network setup
    n_sensors = 9
    n_anchors = 4
    
    # Create test network
    positions = []
    grid_size = 3
    for i in range(grid_size):
        for j in range(grid_size):
            positions.append([i / (grid_size-1), j / (grid_size-1)])
    
    true_positions = np.array(positions)
    anchor_positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])[:n_anchors]
    
    # Dual-frequency carrier phase configuration
    carrier_config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        frequency_l2_hz=1.9e9,
        phase_noise_rad=0.0005,
        snr_db=45,
        use_dual_frequency=True
    )
    
    phase_system = CarrierPhaseMeasurementSystem(carrier_config)
    
    print(f"\nNetwork Configuration:")
    print(f"  Sensors: {n_sensors}")
    print(f"  Anchors: {n_anchors}")
    print(f"  Dual-frequency: L1={carrier_config.frequency_hz/1e9:.1f}GHz, L2={carrier_config.frequency_l2_hz/1e9:.1f}GHz")
    print(f"  Wide-lane λ: {carrier_config.wavelength_wide_lane*100:.1f}cm")
    
    # Generate measurements
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    carrier_phase_measurements = {}
    
    # Test with realistic TWTT error
    twtt_error_std = 0.15  # 15cm error
    
    print(f"\nTWTT error: ±{twtt_error_std*100:.0f}cm")
    print("Generating measurements...")
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            
            if true_dist <= 0.5:  # Within range
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                
                # TWTT measurement with significant error
                twtt_dist = true_dist + np.random.normal(0, twtt_error_std)
                distance_measurements[(i, j)] = twtt_dist
                
                # Dual-frequency carrier phase
                measurement = phase_system.create_measurement(
                    i, j, true_dist, twtt_dist, twtt_error_std,
                    use_dual_freq=True
                )
                
                carrier_phase_measurements[(i, j)] = {
                    'distance': measurement.refined_distance_m,
                    'phase': measurement.measured_phase_rad,
                    'precision_mm': carrier_config.phase_precision_mm,
                    'weight': phase_system.get_measurement_weight(measurement),
                    'method': 'wide_lane'
                }
    
    # Anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 0.7:
                connected.append(a)
                
                # Measurements to anchors
                twtt_dist = true_dist + np.random.normal(0, twtt_error_std)
                distance_measurements[(i, a)] = twtt_dist
                
                measurement = phase_system.create_measurement(
                    i, n_sensors + a, true_dist, twtt_dist, twtt_error_std,
                    use_dual_freq=True
                )
                
                carrier_phase_measurements[(i, a)] = {
                    'distance': measurement.refined_distance_m,
                    'phase': measurement.measured_phase_rad,
                    'precision_mm': carrier_config.phase_precision_mm,
                    'weight': phase_system.get_measurement_weight(measurement),
                    'method': 'wide_lane'
                }
        
        anchor_connections[i] = connected
    
    # Create network data
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        carrier_phase_measurements=carrier_phase_measurements,
        measurement_variance=1e-6
    )
    
    # MPS configuration
    mps_config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=1.0,
        max_iterations=300,
        tolerance=1e-8,
        verbose=True,
        early_stopping=True,
        early_stopping_window=50,
        admm_iterations=100,
        admm_tolerance=1e-8,
        admm_rho=0.5,
        warm_start=True,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True,
        carrier_phase_mode=True
    )
    
    # Run MPS
    print("\nRunning MPS with dual-frequency carrier phase...")
    mps = MatrixParametrizedProximalSplitting(mps_config, network_data)
    results = mps.run()
    
    # Calculate accuracy
    final_positions = results['final_positions']
    position_errors = np.linalg.norm(final_positions - true_positions, axis=1) * 1000  # mm
    rmse_mm = np.sqrt(np.mean(position_errors**2))
    
    print(f"\n{'='*50}")
    print("LOCALIZATION RESULTS")
    print(f"{'='*50}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Final objective: {results['best_objective']:.6f}")
    
    print(f"\nAccuracy Statistics:")
    print(f"  Mean error: {np.mean(position_errors):.3f} mm")
    print(f"  Std dev: {np.std(position_errors):.3f} mm")
    print(f"  Max error: {np.max(position_errors):.3f} mm")
    print(f"  RMSE: {rmse_mm:.3f} mm")
    
    if rmse_mm < 15:
        print(f"\n✓ TARGET ACHIEVED: {rmse_mm:.2f}mm < 15mm")
        print("  MILLIMETER ACCURACY WITH DUAL-FREQUENCY!")
    else:
        print(f"\n⚠ Target not met: {rmse_mm:.2f}mm > 15mm")
    
    print(f"\nPer-sensor errors:")
    for i, error in enumerate(position_errors):
        status = "✓" if error < 15 else "✗"
        print(f"  Sensor {i}: {error:.2f}mm {status}")
    
    return rmse_mm, results


def visualize_comparison(results_comparison):
    """Visualize single vs dual-frequency performance"""
    
    twtt_errors = [r[0] * 100 for r in results_comparison['single_freq']]
    single_rmse = [r[1] for r in results_comparison['single_freq']]
    dual_rmse = [r[1] for r in results_comparison['dual_freq']]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(twtt_errors, single_rmse, 'o-', label='Single-frequency', linewidth=2)
    plt.plot(twtt_errors, dual_rmse, 's-', label='Dual-frequency (Wide-lane)', linewidth=2)
    plt.axhline(y=15, color='r', linestyle='--', label='15mm target', linewidth=1)
    
    plt.xlabel('TWTT Error (cm)')
    plt.ylabel('RMSE (mm)')
    plt.title('Carrier Phase Ambiguity Resolution: Single vs Dual-Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations
    for i, (twtt, single, dual) in enumerate(zip(twtt_errors, single_rmse, dual_rmse)):
        if dual < 15:
            plt.annotate('✓', xy=(twtt, dual), xytext=(twtt, dual*0.8),
                        fontsize=12, color='green', ha='center')
    
    plt.tight_layout()
    plt.savefig('dual_frequency_comparison.png', dpi=150)
    print(f"\nComparison plot saved to dual_frequency_comparison.png")


def main():
    """Run complete test suite"""
    
    print("="*70)
    print("IMPROVED CARRIER PHASE TEST SUITE")
    print("WITH DUAL-FREQUENCY WIDE-LANE RESOLUTION")
    print("="*70)
    
    # Test 1: Compare single vs dual-frequency
    print("\n[TEST 1] Ambiguity Resolution Comparison")
    results_comparison = test_ambiguity_resolution_comparison()
    
    # Test 2: Integrated MPS with dual-frequency
    print("\n[TEST 2] Integrated MPS System")
    rmse_mm, mps_results = test_integrated_mps_with_dual_freq()
    
    # Visualize results
    visualize_comparison(results_comparison)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Check dual-frequency performance at different TWTT errors
    print("\nDual-frequency RMSE at different TWTT errors:")
    all_passed = True
    for twtt_error, rmse in results_comparison['dual_freq']:
        status = "✓ PASS" if rmse < 15 else "✗ FAIL"
        print(f"  ±{twtt_error*100:.0f}cm: {rmse:.2f}mm {status}")
        if rmse >= 15 and twtt_error <= 0.3:  # Should work up to 30cm
            all_passed = False
    
    print(f"\nIntegrated MPS RMSE: {rmse_mm:.2f}mm")
    
    if all_passed and rmse_mm < 15:
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS: MILLIMETER ACCURACY ACHIEVED! ✓✓✓")
        print("="*70)
        print("\nKey achievements:")
        print("  • Dual-frequency wide-lane resolution working")
        print("  • Robust to TWTT errors up to ±30cm")
        print("  • Integrated MPS achieves <15mm RMSE")
        print("  • GPS RTK-level accuracy demonstrated")
    else:
        print("\n⚠ Some tests did not meet target")
    
    return all_passed and rmse_mm < 15


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)