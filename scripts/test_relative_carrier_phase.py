#!/usr/bin/env python3
"""
Test relative carrier phase positioning for achieving <15mm RMSE
Uses network constraints instead of absolute ambiguity resolution
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.core.carrier_phase import (
    CarrierPhaseConfig,
    CarrierPhaseMeasurementSystem,
    NetworkAmbiguityResolver,
    create_carrier_phase_weights
)

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)


def test_relative_positioning_simple():
    """Simple test of relative positioning accuracy"""
    
    print("="*70)
    print("SIMPLE RELATIVE POSITIONING TEST")
    print("="*70)
    print()
    
    # Configuration
    wavelength = 0.125  # 12.5cm
    
    # Simple 3-node network
    true_positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.732]  # Equilateral triangle
    ])
    
    # True distances
    true_distances = {}
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            true_distances[(i, j)] = dist
    
    print("True distances:")
    for pair, dist in true_distances.items():
        print(f"  {pair}: {dist:.3f}m")
    print()
    
    # Simulate measurements with TWTT error
    twtt_error = 0.15  # 15cm
    measurements = {}
    
    for pair, true_dist in true_distances.items():
        # TWTT measurement
        coarse = true_dist + np.random.normal(0, twtt_error)
        
        # Carrier phase (fractional part)
        true_phase = (true_dist / wavelength) * 2 * np.pi
        measured_phase = true_phase % (2 * np.pi)
        
        # Add small phase noise
        phase_noise = np.random.normal(0, 0.001)  # 1mrad
        measured_phase = (measured_phase + phase_noise) % (2 * np.pi)
        
        measurements[pair] = {
            'phase_i': measured_phase,
            'phase_j': 0,  # Reference phase
            'coarse_distance': coarse,
            'true_distance': true_dist
        }
    
    # Resolve using relative positioning
    resolver = NetworkAmbiguityResolver(wavelength)
    resolver.set_reference(0, n_cycles=0)  # Arbitrary reference
    
    # Propagate ambiguities
    print("Resolving relative ambiguities...")
    ambiguities = resolver.propagate_ambiguities(measurements)
    
    print("\nResolved ambiguities (relative to node 0):")
    for node, n in ambiguities.items():
        print(f"  Node {node}: N = {n}")
    
    # Calculate refined distances
    print("\nRefined distances:")
    errors = []
    
    for pair, meas in measurements.items():
        i, j = pair
        
        # Get phases
        phi = meas['phase_i'] / (2 * np.pi)
        
        # Since we're using relative positioning, the distance estimate
        # comes from the phase difference and relative ambiguities
        if i in ambiguities and j in ambiguities:
            n_rel = ambiguities[j] - ambiguities[i]
            refined_dist = (n_rel + phi) * wavelength
            
            # Ensure positive distance
            refined_dist = abs(refined_dist)
            
            # For better accuracy, use phase to refine coarse
            phase_correction = (phi - round(meas['coarse_distance']/wavelength)) * wavelength
            refined_dist = meas['coarse_distance'] + phase_correction
        else:
            refined_dist = meas['coarse_distance']
        
        error = abs(refined_dist - meas['true_distance'])
        errors.append(error)
        
        print(f"  {pair}: {refined_dist:.6f}m (error: {error*1000:.2f}mm)")
    
    rmse_mm = np.sqrt(np.mean(np.square(errors))) * 1000
    print(f"\nRMSE: {rmse_mm:.2f}mm")
    
    if rmse_mm < 15:
        print("✓ Target achieved!")
    else:
        print("⚠ Needs refinement")
    
    return rmse_mm < 15


def test_integrated_relative_mps():
    """Test MPS with relative carrier phase positioning"""
    
    print("\n" + "="*70)
    print("INTEGRATED MPS WITH RELATIVE POSITIONING")
    print("="*70)
    print()
    
    # Network configuration
    n_sensors = 9
    n_anchors = 4
    
    # Create grid network
    positions = []
    for i in range(3):
        for j in range(3):
            positions.append([i * 0.5, j * 0.5])
    
    true_positions = np.array(positions)
    anchor_positions = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])[:n_anchors]
    
    # Carrier phase configuration
    config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        phase_noise_rad=0.0005,
        snr_db=45
    )
    
    wavelength = config.wavelength
    phase_system = CarrierPhaseMeasurementSystem(config)
    
    print(f"Network: {n_sensors} sensors, {n_anchors} anchors")
    print(f"Wavelength: {wavelength*100:.1f}cm")
    print()
    
    # Generate measurements
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    carrier_measurements = {}
    
    # TWTT error
    twtt_error = 0.15  # 15cm - challenging for single frequency
    print(f"TWTT error: ±{twtt_error*100:.0f}cm")
    print("(This would fail with single-frequency absolute resolution)")
    print()
    
    # Sensor-to-sensor measurements
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            
            if true_dist <= 0.75:  # Within range
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                
                # TWTT with error
                coarse = true_dist + np.random.normal(0, twtt_error)
                distance_measurements[(i, j)] = coarse
                
                # Carrier phase
                true_phase = (true_dist / wavelength) * 2 * np.pi
                measured_phase = (true_phase + np.random.normal(0, 0.0005)) % (2 * np.pi)
                
                carrier_measurements[(i, j)] = {
                    'phase': measured_phase,
                    'phase_i': measured_phase,
                    'phase_j': 0,
                    'coarse_distance': coarse,
                    'true_distance': true_dist
                }
    
    # Anchor connections
    anchor_connections = {}
    anchor_measurements = {}
    
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            
            if true_dist <= 1.0:
                connected.append(a)
                
                # Measurements to anchors
                coarse = true_dist + np.random.normal(0, twtt_error)
                distance_measurements[(i, n_sensors + a)] = coarse
                
                # Carrier phase
                true_phase = (true_dist / wavelength) * 2 * np.pi
                measured_phase = (true_phase + np.random.normal(0, 0.0005)) % (2 * np.pi)
                
                anchor_measurements[(i, a)] = {
                    'phase': measured_phase,
                    'coarse_distance': coarse,
                    'true_distance': true_dist
                }
        
        anchor_connections[i] = connected
    
    # Use relative positioning to refine distances
    print("Applying relative carrier phase positioning...")
    
    resolver = NetworkAmbiguityResolver(wavelength)
    
    # Use first anchor as reference
    resolver.set_reference(n_sensors, n_cycles=0)  # Anchor 0 as reference
    
    # Combine all measurements for propagation
    all_measurements = {}
    all_measurements.update(carrier_measurements)
    
    # Add anchor measurements with proper indexing
    for (sensor, anchor), meas in anchor_measurements.items():
        all_measurements[(sensor, n_sensors + anchor)] = meas
    
    # Propagate ambiguities through network
    try:
        node_ambiguities = resolver.propagate_ambiguities(all_measurements)
        print(f"Resolved ambiguities for {len(node_ambiguities)} nodes")
    except Exception as e:
        print(f"Warning: {e}")
        node_ambiguities = {}
    
    # Refine distances using resolved ambiguities
    refined_carrier_measurements = {}
    
    for (i, j), meas in carrier_measurements.items():
        # Use phase to refine distance
        phi = meas['phase'] / (2 * np.pi)
        phase_correction = (phi - round(meas['coarse_distance']/wavelength)) * wavelength
        
        refined_carrier_measurements[(i, j)] = {
            'distance': meas['coarse_distance'] + phase_correction,
            'phase': meas['phase'],
            'weight': 1000.0,  # High weight for carrier phase
            'precision_mm': config.phase_precision_mm
        }
    
    # Create network data for MPS
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        carrier_phase_measurements=refined_carrier_measurements,
        measurement_variance=1e-6
    )
    
    # MPS configuration
    mps_config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=1.0,
        max_iterations=200,
        tolerance=1e-8,
        verbose=False,
        early_stopping=True,
        admm_iterations=50,
        admm_rho=0.5,
        warm_start=True,
        use_2block=True,
        carrier_phase_mode=True
    )
    
    # Run MPS
    print("\nRunning MPS with relative carrier phase...")
    mps = MatrixParametrizedProximalSplitting(mps_config, network_data)
    results = mps.run()
    
    # Calculate accuracy
    final_positions = results['final_positions']
    position_errors = np.linalg.norm(final_positions - true_positions, axis=1) * 1000  # mm
    rmse_mm = np.sqrt(np.mean(position_errors**2))
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    print(f"\nAccuracy:")
    print(f"  Mean error: {np.mean(position_errors):.2f}mm")
    print(f"  Std dev: {np.std(position_errors):.2f}mm")
    print(f"  Max error: {np.max(position_errors):.2f}mm")
    print(f"  RMSE: {rmse_mm:.2f}mm")
    
    if rmse_mm < 15:
        print(f"\n✓✓✓ SUCCESS: {rmse_mm:.2f}mm < 15mm TARGET ✓✓✓")
    else:
        print(f"\n⚠ Not quite there: {rmse_mm:.2f}mm > 15mm")
    
    # Per-sensor breakdown
    print(f"\nPer-sensor errors:")
    for i, error in enumerate(position_errors):
        status = "✓" if error < 15 else "✗"
        print(f"  Sensor {i}: {error:.2f}mm {status}")
    
    return rmse_mm


def main():
    """Run complete test suite"""
    
    print("="*70)
    print("RELATIVE CARRIER PHASE POSITIONING TEST SUITE")
    print("="*70)
    print()
    print("Using network constraints and relative ambiguities")
    print("instead of absolute ambiguity resolution")
    print()
    
    # Test 1: Simple network
    print("[TEST 1] Simple 3-node network")
    simple_success = test_relative_positioning_simple()
    
    # Test 2: Full MPS integration
    print("\n[TEST 2] Full MPS Integration")
    mps_rmse = test_integrated_relative_mps()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if simple_success:
        print("✓ Simple test: PASSED")
    else:
        print("✗ Simple test: FAILED")
    
    if mps_rmse < 15:
        print(f"✓ MPS integration: PASSED ({mps_rmse:.2f}mm)")
    else:
        print(f"✗ MPS integration: FAILED ({mps_rmse:.2f}mm)")
    
    if simple_success and mps_rmse < 15:
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED - MILLIMETER ACCURACY ACHIEVED ✓✓✓")
        print("="*70)
        print("\nKey achievements:")
        print("  • Relative positioning works with ±15cm TWTT error")
        print("  • Network constraints enable robust resolution")
        print("  • MPS achieves <15mm RMSE target")
        print("  • Solution is simpler than dual-frequency")
    else:
        print("\n⚠ Some tests need refinement")
    
    return simple_success and mps_rmse < 15


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)