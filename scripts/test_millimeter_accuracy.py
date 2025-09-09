#!/usr/bin/env python3
"""
Focused test for achieving millimeter accuracy with MPS algorithm
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)

def create_precise_network(n_sensors=10, n_anchors=4):
    """Create a network with very precise measurements for testing"""
    
    # Create a regular grid for better geometry
    grid_size = int(np.sqrt(n_sensors))
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n_sensors:
                positions.append([i / (grid_size-1), j / (grid_size-1)])
    
    true_positions = np.array(positions)
    
    # Place anchors at strategic locations
    anchor_positions = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])[:n_anchors]
    
    # Build adjacency matrix with good connectivity
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= 0.5:  # Good connectivity
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                
                # Very precise measurement (sub-millimeter noise)
                noise = np.random.randn() * 0.00001  # 0.01mm noise
                distance_measurements[(i, j)] = true_dist + noise
    
    # Anchor connections with high precision
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= 0.8:
                connected.append(a)
                noise = np.random.randn() * 0.00001
                distance_measurements[(i, a)] = true_dist + noise
        anchor_connections[i] = connected
    
    # Generate carrier phase measurements
    carrier_phase_measurements = {}
    wavelength = 0.1905  # S-band wavelength in meters
    
    for key, dist in distance_measurements.items():
        phase_cycles = dist / wavelength
        integer_cycles = int(phase_cycles)
        fractional_phase = phase_cycles - integer_cycles
        
        # Ultra-precise phase measurement
        phase_noise = np.random.randn() * 0.00001 / wavelength
        measured_phase = fractional_phase + phase_noise
        
        carrier_phase_measurements[key] = {
            'distance': dist,
            'phase': measured_phase,
            'wavelength': wavelength,
            'precision_mm': 0.01  # 0.01mm precision
        }
    
    return NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        carrier_phase_measurements=carrier_phase_measurements,
        measurement_variance=1e-10
    )


def test_configurations():
    """Test different parameter configurations to achieve millimeter accuracy"""
    
    print("Testing parameter configurations for millimeter accuracy...")
    print("="*60)
    
    network_data = create_precise_network(n_sensors=9, n_anchors=4)
    
    # Different configurations to test
    configs = [
        {
            'name': 'Conservative',
            'gamma': 0.9999,
            'alpha': 1.0,
            'admm_iterations': 200,
            'admm_rho': 0.1,
            'max_iterations': 500
        },
        {
            'name': 'Aggressive',
            'gamma': 0.99,
            'alpha': 50.0,
            'admm_iterations': 100,
            'admm_rho': 1.0,
            'max_iterations': 300
        },
        {
            'name': 'Balanced',
            'gamma': 0.999,
            'alpha': 10.0,
            'admm_iterations': 150,
            'admm_rho': 0.5,
            'max_iterations': 400
        },
        {
            'name': 'Fine-tuned',
            'gamma': 0.9995,
            'alpha': 5.0,
            'admm_iterations': 100,
            'admm_rho': 0.3,
            'max_iterations': 600
        }
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\nTesting {cfg['name']} configuration...")
        
        config = MPSConfig(
            n_sensors=9,
            n_anchors=4,
            dimension=2,
            gamma=cfg['gamma'],
            alpha=cfg['alpha'],
            max_iterations=cfg['max_iterations'],
            tolerance=1e-10,
            verbose=False,
            early_stopping=True,
            early_stopping_window=100,
            admm_iterations=cfg['admm_iterations'],
            admm_tolerance=1e-10,
            admm_rho=cfg['admm_rho'],
            warm_start=True,
            use_2block=True,
            parallel_proximal=True,
            adaptive_alpha=True,
            carrier_phase_mode=True
        )
        
        mps = MatrixParametrizedProximalSplitting(config, network_data)
        result = mps.run()
        
        # Compute detailed accuracy metrics
        final_positions = result['final_positions']
        true_positions = network_data.true_positions
        
        errors = np.linalg.norm(final_positions - true_positions, axis=1) * 1000  # mm
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        results.append({
            'config': cfg['name'],
            'rmse': rmse,
            'max_error': max_error,
            'mean_error': mean_error,
            'iterations': result['iterations'],
            'objective': result['best_objective']
        })
        
        print(f"  RMSE: {rmse:.3f} mm")
        print(f"  Max error: {max_error:.3f} mm")
        print(f"  Mean error: {mean_error:.3f} mm")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final objective: {result['best_objective']:.6f}")
        
        # Check if target met
        if rmse < 15.0:
            print(f"  ✓ TARGET MET! ({rmse:.3f} mm < 15 mm)")
        else:
            print(f"  ✗ Target not met ({rmse:.3f} mm >= 15 mm)")
    
    # Find best configuration
    best = min(results, key=lambda x: x['rmse'])
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION:")
    print(f"  {best['config']}: RMSE = {best['rmse']:.3f} mm")
    
    if best['rmse'] < 15.0:
        print("\n✓ MILLIMETER ACCURACY ACHIEVED!")
        print(f"  Best RMSE: {best['rmse']:.3f} mm")
        print(f"  Configuration: {best['config']}")
    else:
        print("\n⚠ Millimeter accuracy not achieved with current configurations")
        print("  Further parameter tuning needed")
    
    return results


def visualize_convergence(network_data):
    """Visualize convergence behavior with best configuration"""
    
    print("\n" + "="*60)
    print("Running detailed convergence analysis...")
    
    config = MPSConfig(
        n_sensors=9,
        n_anchors=4,
        dimension=2,
        gamma=0.9995,
        alpha=5.0,
        max_iterations=500,
        tolerance=1e-12,
        verbose=True,
        early_stopping=False,
        admm_iterations=100,
        admm_tolerance=1e-10,
        admm_rho=0.3,
        warm_start=True,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True,
        carrier_phase_mode=True
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    result = mps.run()
    
    # Plot convergence
    history = result['history']
    iterations = range(len(history['position_error']))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position error
    axes[0, 0].plot(iterations, history['position_error'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=15, color='r', linestyle='--', label='15mm target')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('RMSE (mm)')
    axes[0, 0].set_title('Position Error Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Objective
    axes[0, 1].semilogy(iterations, history['objective'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Objective Value')
    axes[0, 1].set_title('Objective Function')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSD violation
    axes[1, 0].semilogy(iterations, np.maximum(history['psd_violation'], 1e-15), 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('PSD Violation')
    axes[1, 0].set_title('Constraint Violation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Consensus error
    axes[1, 1].semilogy(iterations, np.maximum(history['consensus_error'], 1e-15), 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Consensus Error')
    axes[1, 1].set_title('Consensus Convergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('MPS Algorithm Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mps_millimeter_convergence.png', dpi=150)
    print(f"Convergence plot saved to mps_millimeter_convergence.png")
    
    # Final accuracy analysis
    final_positions = result['final_positions']
    true_positions = network_data.true_positions
    errors = np.linalg.norm(final_positions - true_positions, axis=1) * 1000
    
    print("\nFinal Accuracy Statistics:")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.3f} mm")
    print(f"  Mean error: {np.mean(errors):.3f} mm")
    print(f"  Std dev: {np.std(errors):.3f} mm")
    print(f"  Max error: {np.max(errors):.3f} mm")
    print(f"  Min error: {np.min(errors):.3f} mm")
    print(f"  95th percentile: {np.percentile(errors, 95):.3f} mm")
    
    return result


def main():
    """Main test routine"""
    print("="*60)
    print("MILLIMETER ACCURACY TESTING FOR MPS ALGORITHM")
    print("="*60)
    
    # Create precise test network
    network_data = create_precise_network(n_sensors=9, n_anchors=4)
    
    # Test different configurations
    results = test_configurations()
    
    # Visualize best configuration
    final_result = visualize_convergence(network_data)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Check final achievement
    best = min(results, key=lambda x: x['rmse'])
    if best['rmse'] < 15.0:
        print("✓ SUCCESS: Millimeter accuracy target achieved!")
        print(f"  Best RMSE: {best['rmse']:.3f} mm < 15 mm")
        return True
    else:
        print("⚠ Millimeter accuracy not fully achieved")
        print(f"  Best RMSE: {best['rmse']:.3f} mm")
        print("  Recommendations:")
        print("  - Increase ADMM iterations")
        print("  - Tune alpha parameter based on measurement precision")
        print("  - Ensure sufficient anchor coverage")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)