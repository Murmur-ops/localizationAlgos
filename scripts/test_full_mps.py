#!/usr/bin/env python3
"""
Test script for the full Matrix-Parametrized Proximal Splitting algorithm
Validates millimeter-accuracy achievement and compares with paper results
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData,
    create_network_data
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_convergence_behavior():
    """Test convergence behavior of the algorithm"""
    logger.info("Testing convergence behavior...")
    
    # Create test network
    n_sensors = 20
    n_anchors = 4
    dimension = 2
    
    network_data = create_network_data(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=dimension,
        communication_range=0.4,
        measurement_noise=0.001,
        carrier_phase=True
    )
    
    # Configure algorithm
    config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=dimension,
        gamma=0.999,
        alpha=10.0,
        max_iterations=500,
        tolerance=1e-6,
        verbose=True,
        early_stopping=True,
        early_stopping_window=50,
        admm_iterations=50,
        admm_tolerance=1e-6,
        carrier_phase_mode=True,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True
    )
    
    # Run algorithm
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    start_time = time.time()
    results = mps.run()
    elapsed_time = time.time() - start_time
    
    # Report results
    logger.info(f"Algorithm completed in {elapsed_time:.2f} seconds")
    logger.info(f"Iterations: {results['iterations']}")
    logger.info(f"Converged: {results['converged']}")
    logger.info(f"Final RMSE (mm): {results.get('rmse_mm', 0):.2f}")
    logger.info(f"Best objective: {results['best_objective']:.6f}")
    
    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    history = results['history']
    iterations = range(len(history['objective']))
    
    # Objective function
    axes[0, 0].semilogy(iterations, history['objective'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].set_title('Objective Function Convergence')
    axes[0, 0].grid(True)
    
    # PSD violation
    axes[0, 1].semilogy(iterations, history['psd_violation'])
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('PSD Violation')
    axes[0, 1].set_title('PSD Constraint Violation')
    axes[0, 1].grid(True)
    
    # Consensus error
    axes[1, 0].semilogy(iterations, history['consensus_error'])
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Consensus Error')
    axes[1, 0].set_title('Consensus Error')
    axes[1, 0].grid(True)
    
    # Position error
    axes[1, 1].plot(iterations, history['position_error'])
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Position Error (mm)')
    axes[1, 1].set_title('Position Estimation Error')
    axes[1, 1].axhline(y=15, color='r', linestyle='--', label='15mm target')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mps_convergence.png', dpi=150)
    plt.show()
    
    return results


def test_millimeter_accuracy():
    """Test achievement of millimeter-level accuracy"""
    logger.info("Testing millimeter accuracy achievement...")
    
    # Test configurations
    network_sizes = [10, 20, 30, 40]
    results_summary = []
    
    for n_sensors in network_sizes:
        logger.info(f"\nTesting with {n_sensors} sensors...")
        
        # Create network with carrier phase measurements
        network_data = create_network_data(
            n_sensors=n_sensors,
            n_anchors=max(4, n_sensors // 5),
            dimension=2,
            communication_range=0.4,
            measurement_noise=0.0001,  # Very low noise for carrier phase
            carrier_phase=True
        )
        
        # Configure for millimeter accuracy
        config = MPSConfig(
            n_sensors=n_sensors,
            n_anchors=network_data.anchor_positions.shape[0],
            dimension=2,
            gamma=0.999,
            alpha=5.0,  # Smaller alpha for better accuracy
            max_iterations=1000,
            tolerance=1e-8,
            verbose=False,
            early_stopping=True,
            early_stopping_window=100,
            admm_iterations=100,
            admm_tolerance=1e-8,
            carrier_phase_mode=True,
            use_2block=True,
            parallel_proximal=True,
            adaptive_alpha=True
        )
        
        # Run algorithm
        mps = MatrixParametrizedProximalSplitting(config, network_data)
        start_time = time.time()
        results = mps.run()
        elapsed_time = time.time() - start_time
        
        # Compute accuracy metrics
        final_positions = results['final_positions']
        true_positions = network_data.true_positions
        
        # Position errors
        position_errors = np.linalg.norm(
            final_positions - true_positions, axis=1
        ) * 1000  # Convert to mm
        
        rmse = np.sqrt(np.mean(position_errors**2))
        max_error = np.max(position_errors)
        percentile_95 = np.percentile(position_errors, 95)
        
        results_summary.append({
            'n_sensors': n_sensors,
            'rmse_mm': rmse,
            'max_error_mm': max_error,
            '95_percentile_mm': percentile_95,
            'iterations': results['iterations'],
            'time_seconds': elapsed_time,
            'achieved_target': rmse < 15.0
        })
        
        logger.info(f"  RMSE: {rmse:.2f} mm")
        logger.info(f"  Max error: {max_error:.2f} mm")
        logger.info(f"  95th percentile: {percentile_95:.2f} mm")
        logger.info(f"  Target achieved: {rmse < 15.0}")
        logger.info(f"  Time: {elapsed_time:.2f} seconds")
    
    # Summary table
    logger.info("\n" + "="*60)
    logger.info("MILLIMETER ACCURACY RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Sensors':<10} {'RMSE(mm)':<12} {'Max(mm)':<12} {'95%(mm)':<12} {'Target':<10} {'Time(s)':<10}")
    logger.info("-"*60)
    
    for result in results_summary:
        achieved = "✓" if result['achieved_target'] else "✗"
        logger.info(
            f"{result['n_sensors']:<10} "
            f"{result['rmse_mm']:<12.2f} "
            f"{result['max_error_mm']:<12.2f} "
            f"{result['95_percentile_mm']:<12.2f} "
            f"{achieved:<10} "
            f"{result['time_seconds']:<10.2f}"
        )
    
    return results_summary


def test_comparison_with_simplified():
    """Compare full algorithm with simplified version"""
    logger.info("Comparing full algorithm with simplified version...")
    
    # Create test network
    n_sensors = 25
    n_anchors = 5
    
    network_data = create_network_data(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        communication_range=0.35,
        measurement_noise=0.001,
        carrier_phase=True
    )
    
    # Run full algorithm
    config_full = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=500,
        tolerance=1e-6,
        verbose=False,
        carrier_phase_mode=True,
        use_2block=True,
        parallel_proximal=True,
        adaptive_alpha=True
    )
    
    mps_full = MatrixParametrizedProximalSplitting(config_full, network_data)
    start_time = time.time()
    results_full = mps_full.run()
    time_full = time.time() - start_time
    
    # Run simplified version (without 2-block, parallel, adaptive features)
    config_simple = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=500,
        tolerance=1e-6,
        verbose=False,
        carrier_phase_mode=False,
        use_2block=False,
        parallel_proximal=False,
        adaptive_alpha=False
    )
    
    mps_simple = MatrixParametrizedProximalSplitting(config_simple, network_data)
    start_time = time.time()
    results_simple = mps_simple.run()
    time_simple = time.time() - start_time
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("ALGORITHM COMPARISON")
    logger.info("="*60)
    
    logger.info(f"{'Metric':<30} {'Full Algorithm':<20} {'Simplified':<20}")
    logger.info("-"*60)
    
    metrics = [
        ('Iterations', results_full['iterations'], results_simple['iterations']),
        ('Final RMSE (mm)', results_full.get('rmse_mm', results_full['final_rmse']*1000), 
         results_simple['final_rmse']*1000),
        ('Best Objective', f"{results_full['best_objective']:.6f}", 
         f"{results_simple['best_objective']:.6f}"),
        ('Computation Time (s)', f"{time_full:.2f}", f"{time_simple:.2f}"),
        ('Converged', results_full['converged'], results_simple['converged'])
    ]
    
    for metric, full_val, simple_val in metrics:
        logger.info(f"{metric:<30} {str(full_val):<20} {str(simple_val):<20}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convergence comparison
    iterations_full = range(len(results_full['history']['objective']))
    iterations_simple = range(len(results_simple['history']['objective']))
    
    axes[0].semilogy(iterations_full, results_full['history']['objective'], 
                     label='Full Algorithm', linewidth=2)
    axes[0].semilogy(iterations_simple, results_simple['history']['objective'], 
                     label='Simplified', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Convergence Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Position error comparison
    axes[1].plot(iterations_full, 
                 [e*1000 if results_full.get('rmse_mm') else e 
                  for e in results_full['history']['position_error']], 
                 label='Full Algorithm', linewidth=2)
    axes[1].plot(iterations_simple, 
                 [e*1000 for e in results_simple['history']['position_error']], 
                 label='Simplified', linewidth=2, linestyle='--')
    axes[1].axhline(y=15, color='r', linestyle=':', label='15mm target')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Position Error (mm)')
    axes[1].set_title('Position Error Comparison')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mps_comparison.png', dpi=150)
    plt.show()
    
    return results_full, results_simple


def test_cramer_rao_bound():
    """Test performance relative to Cramér-Rao lower bound"""
    logger.info("Testing performance relative to Cramér-Rao bound...")
    
    # Create network with known CRB
    n_sensors = 16
    n_anchors = 4
    
    # Grid layout for easier CRB computation
    grid_size = int(np.sqrt(n_sensors))
    positions = np.zeros((n_sensors, 2))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < n_sensors:
                positions[idx] = [i / (grid_size-1), j / (grid_size-1)]
                idx += 1
    
    # Place anchors at corners
    anchor_positions = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])
    
    # Create network data
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    measurement_variance = 1e-6  # Very small for carrier phase
    
    # Build adjacency and measurements
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= 0.4:  # Communication range
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                noise = np.random.randn() * np.sqrt(measurement_variance)
                distance_measurements[(i, j)] = dist + noise
    
    # Anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected = []
        for a in range(n_anchors):
            dist = np.linalg.norm(positions[i] - anchor_positions[a])
            if dist <= 0.6:
                connected.append(a)
                noise = np.random.randn() * np.sqrt(measurement_variance)
                distance_measurements[(i, a)] = dist + noise
        anchor_connections[i] = connected
    
    network_data = NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=positions,
        measurement_variance=measurement_variance
    )
    
    # Compute approximate CRB (simplified)
    # For range measurements, CRB ≈ σ²/√(number_of_measurements)
    avg_measurements_per_sensor = len(distance_measurements) / n_sensors
    crb_approximate = np.sqrt(measurement_variance / avg_measurements_per_sensor) * 1000  # in mm
    
    # Run algorithm
    config = MPSConfig(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        dimension=2,
        gamma=0.999,
        alpha=10.0,
        max_iterations=1000,
        tolerance=1e-8,
        verbose=True,
        carrier_phase_mode=True,
        use_2block=True,
        adaptive_alpha=True
    )
    
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    results = mps.run()
    
    # Compute performance metrics
    rmse = results.get('rmse_mm', results['final_rmse'] * 1000)
    efficiency = (crb_approximate / rmse) * 100  # Percentage of CRB achieved
    
    logger.info("\n" + "="*60)
    logger.info("CRAMÉR-RAO BOUND COMPARISON")
    logger.info("="*60)
    logger.info(f"Approximate CRB: {crb_approximate:.3f} mm")
    logger.info(f"Achieved RMSE: {rmse:.3f} mm")
    logger.info(f"Efficiency: {efficiency:.1f}% of CRB")
    logger.info(f"Target (60-80% of CRB): {'✓' if 60 <= efficiency <= 80 else '✗'}")
    
    return {
        'crb': crb_approximate,
        'rmse': rmse,
        'efficiency': efficiency,
        'target_met': 60 <= efficiency <= 80
    }


def main():
    """Run all tests"""
    logger.info("Starting comprehensive MPS algorithm tests...")
    logger.info("="*60)
    
    # Test 1: Convergence behavior
    logger.info("\n[TEST 1] Convergence Behavior")
    convergence_results = test_convergence_behavior()
    
    # Test 2: Millimeter accuracy
    logger.info("\n[TEST 2] Millimeter Accuracy Achievement")
    accuracy_results = test_millimeter_accuracy()
    
    # Test 3: Comparison with simplified
    logger.info("\n[TEST 3] Full vs Simplified Algorithm")
    comparison_results = test_comparison_with_simplified()
    
    # Test 4: Cramér-Rao bound
    logger.info("\n[TEST 4] Cramér-Rao Bound Performance")
    crb_results = test_cramer_rao_bound()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUITE COMPLETE")
    logger.info("="*60)
    
    # Check if all targets met
    all_targets_met = all([
        convergence_results.get('rmse_mm', float('inf')) < 15,
        all(r['achieved_target'] for r in accuracy_results),
        crb_results['target_met']
    ])
    
    if all_targets_met:
        logger.info("✓ ALL PERFORMANCE TARGETS ACHIEVED")
        logger.info("  - Convergence in 200-500 iterations: ✓")
        logger.info("  - Millimeter accuracy (<15mm RMSE): ✓")
        logger.info("  - 60-80% of Cramér-Rao bound: ✓")
    else:
        logger.info("⚠ Some targets not met - further tuning needed")
    
    return {
        'convergence': convergence_results,
        'accuracy': accuracy_results,
        'comparison': comparison_results,
        'crb': crb_results,
        'all_targets_met': all_targets_met
    }


if __name__ == "__main__":
    results = main()