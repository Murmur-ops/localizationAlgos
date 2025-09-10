#!/usr/bin/env python3
"""
Integrated test for carrier phase with TWTT and MPS algorithm
Validates millimeter accuracy achievement
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.core.carrier_phase import (
    CarrierPhaseConfig,
    CarrierPhaseMeasurementSystem,
    IntegerAmbiguityResolver,
    PhaseUnwrapper
)

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData,
    create_network_data
)


def create_precise_test_network(n_sensors=12, n_anchors=4):
    """Create test network with carrier phase measurements"""
    
    # Create regular grid for controlled testing
    grid_size = int(np.sqrt(n_sensors))
    positions = []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            if len(positions) < n_sensors:
                positions.append([i / grid_size, j / grid_size])
    
    true_positions = np.array(positions)
    
    # Anchor positions at corners
    anchor_positions = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])[:n_anchors]
    
    return true_positions, anchor_positions


def test_carrier_phase_accuracy():
    """Test carrier phase measurement accuracy"""
    
    print("="*60)
    print("CARRIER PHASE ACCURACY TEST")
    print("="*60)
    
    # Setup
    config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        phase_noise_rad=0.001,  # 1 mrad
        snr_db=40  # High SNR
    )
    
    system = CarrierPhaseMeasurementSystem(config)
    resolver = IntegerAmbiguityResolver(config)
    
    print(f"\nConfiguration:")
    print(f"  Wavelength: {config.wavelength*100:.1f} cm")
    print(f"  Phase precision: {config.phase_precision_mm:.3f} mm")
    
    # Test at various distances
    test_distances = np.linspace(0.1, 5.0, 20)
    errors_mm = []
    
    for true_dist in test_distances:
        # Simulate TWTT measurement (±10cm accuracy)
        twtt_noise = np.random.normal(0, 0.1)
        coarse_dist = true_dist + twtt_noise
        
        # Measure carrier phase
        phase, phase_var = system.measure_carrier_phase(true_dist, add_noise=True)
        
        # Resolve ambiguity
        result = resolver.resolve_single_baseline(
            phase, coarse_dist, 0.1
        )
        
        # Calculate error
        error_mm = abs(result.refined_distance_m - true_dist) * 1000
        errors_mm.append(error_mm)
    
    # Statistics
    rmse_mm = np.sqrt(np.mean(np.square(errors_mm)))
    
    print(f"\nResults over {len(test_distances)} measurements:")
    print(f"  Mean error: {np.mean(errors_mm):.3f} mm")
    print(f"  Std dev: {np.std(errors_mm):.3f} mm")
    print(f"  Max error: {np.max(errors_mm):.3f} mm")
    print(f"  RMSE: {rmse_mm:.3f} mm")
    print(f"  Target (<15mm): {'✓' if rmse_mm < 15 else '✗'}")
    
    return rmse_mm < 15


def test_integrated_system():
    """Test complete integrated system with MPS"""
    
    print("\n" + "="*60)
    print("INTEGRATED SYSTEM TEST")
    print("="*60)
    
    # Create test network
    n_sensors = 9
    n_anchors = 4
    true_positions, anchor_positions = create_precise_test_network(n_sensors, n_anchors)
    
    # Initialize carrier phase system
    carrier_config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        phase_noise_rad=0.0005,  # 0.5 mrad for better accuracy
        snr_db=45
    )
    
    phase_system = CarrierPhaseMeasurementSystem(carrier_config)
    resolver = IntegerAmbiguityResolver(carrier_config)
    
    # Generate measurements
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    carrier_phase_measurements = {}
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            
            if true_dist <= 0.5:  # Within range
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                
                # TWTT measurement (coarse)
                twtt_dist = true_dist + np.random.normal(0, 0.05)  # ±5cm
                distance_measurements[(i, j)] = twtt_dist
                
                # Carrier phase measurement (fine)
                measurement = phase_system.create_measurement(
                    i, j, true_dist, twtt_dist, 0.05
                )
                
                # Store refined distance
                carrier_phase_measurements[(i, j)] = {
                    'distance': measurement.refined_distance_m,
                    'phase': measurement.measured_phase_rad,
                    'precision_mm': carrier_config.phase_precision_mm,
                    'weight': phase_system.get_measurement_weight(measurement)
                }
    
    # Anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 0.7:
                connected.append(a)
                
                # TWTT measurement
                twtt_dist = true_dist + np.random.normal(0, 0.05)
                distance_measurements[(i, a)] = twtt_dist
                
                # Carrier phase
                measurement = phase_system.create_measurement(
                    i, n_sensors + a, true_dist, twtt_dist, 0.05
                )
                
                carrier_phase_measurements[(i, a)] = {
                    'distance': measurement.refined_distance_m,
                    'phase': measurement.measured_phase_rad,
                    'precision_mm': carrier_config.phase_precision_mm,
                    'weight': phase_system.get_measurement_weight(measurement)
                }
        
        anchor_connections[i] = connected
    
    # Create network data for MPS
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        carrier_phase_measurements=carrier_phase_measurements,
        measurement_variance=1e-6  # Very small for carrier phase
    )
    
    # Configure MPS for millimeter accuracy
    mps_config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=1.0,  # Will be adaptively scaled
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
    
    # Run MPS algorithm
    print("\nRunning MPS with carrier phase measurements...")
    mps = MatrixParametrizedProximalSplitting(mps_config, network_data)
    results = mps.run()
    
    # Calculate accuracy
    final_positions = results['final_positions']
    position_errors = np.linalg.norm(final_positions - true_positions, axis=1) * 1000  # mm
    rmse_mm = np.sqrt(np.mean(position_errors**2))
    
    print(f"\nLocalization Results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Final objective: {results['best_objective']:.6f}")
    print(f"\nAccuracy:")
    print(f"  Mean error: {np.mean(position_errors):.3f} mm")
    print(f"  Std dev: {np.std(position_errors):.3f} mm")
    print(f"  Max error: {np.max(position_errors):.3f} mm")
    print(f"  RMSE: {rmse_mm:.3f} mm")
    print(f"  Target (<15mm): {'✓' if rmse_mm < 15 else '✗'}")
    
    # Per-sensor errors
    print(f"\nPer-sensor errors (mm):")
    for i, error in enumerate(position_errors):
        print(f"  Sensor {i}: {error:.2f} mm")
    
    return rmse_mm, results


def test_phase_unwrapping():
    """Test phase unwrapping for dynamic scenario"""
    
    print("\n" + "="*60)
    print("PHASE UNWRAPPING TEST")
    print("="*60)
    
    unwrapper = PhaseUnwrapper()
    
    # Simulate moving sensor
    dt = 0.1  # 100ms
    n_steps = 30
    velocity = 0.05  # m/s
    
    print("\nSimulating moving sensor...")
    
    cycle_slips = 0
    for i in range(n_steps):
        # True distance changes
        distance = 1.0 + velocity * i * dt
        
        # Convert to phase
        wavelength = 0.125  # 12.5cm at 2.4GHz
        true_phase = (distance / wavelength) * 2 * np.pi
        wrapped_phase = true_phase % (2 * np.pi)
        
        # Add noise
        wrapped_phase += np.random.normal(0, 0.01)
        wrapped_phase = wrapped_phase % (2 * np.pi)
        
        # Process
        timestamp = int(i * dt * 1e9)
        unwrapped, slip = unwrapper.process_measurement(0, 1, wrapped_phase, timestamp)
        
        if slip:
            cycle_slips += 1
    
    # Get quality
    quality = unwrapper.get_phase_quality((0, 1))
    
    print(f"  Measurements: {quality['num_measurements']}")
    print(f"  Cycle slips: {cycle_slips}")
    print(f"  Quality score: {quality['quality_score']:.3f}")
    
    return quality['quality_score'] > 0.5


def visualize_results(rmse_mm, results):
    """Visualize convergence and accuracy"""
    
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convergence
    iterations = range(len(history['objective']))
    
    axes[0, 0].semilogy(iterations, history['objective'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Objective')
    axes[0, 0].set_title('Objective Convergence')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position error
    axes[0, 1].plot(iterations, history['position_error'])
    axes[0, 1].axhline(y=15, color='r', linestyle='--', label='15mm target')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('RMSE (mm)')
    axes[0, 1].set_title('Position Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSD violation
    axes[1, 0].semilogy(iterations, np.maximum(history['psd_violation'], 1e-15))
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('PSD Violation')
    axes[1, 0].set_title('Constraint Violation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Consensus error
    axes[1, 1].semilogy(iterations, np.maximum(history['consensus_error'], 1e-15))
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Consensus Error')
    axes[1, 1].set_title('Consensus Convergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Carrier Phase Integration Results (RMSE: {rmse_mm:.2f}mm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('carrier_phase_integration.png', dpi=150)
    print(f"\nPlot saved to carrier_phase_integration.png")


def main():
    """Run all integration tests"""
    
    print("="*60)
    print("CARRIER PHASE INTEGRATION TEST SUITE")
    print("="*60)
    
    # Test 1: Carrier phase accuracy
    phase_test = test_carrier_phase_accuracy()
    
    # Test 2: Integrated system
    rmse_mm, results = test_integrated_system()
    
    # Test 3: Phase unwrapping
    unwrap_test = test_phase_unwrapping()
    
    # Visualize
    if results:
        visualize_results(rmse_mm, results)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"Carrier Phase Accuracy: {'✓ PASSED' if phase_test else '✗ FAILED'}")
    print(f"Integrated System: {'✓ PASSED' if rmse_mm < 15 else '✗ FAILED'} ({rmse_mm:.2f}mm)")
    print(f"Phase Unwrapping: {'✓ PASSED' if unwrap_test else '✗ FAILED'}")
    
    overall_success = phase_test and (rmse_mm < 15) and unwrap_test
    
    print("\n" + "="*60)
    if overall_success:
        print("✓ ALL TESTS PASSED - MILLIMETER ACCURACY ACHIEVED!")
    else:
        print("⚠ Some tests failed - further tuning needed")
    print("="*60)
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)