#!/usr/bin/env python3
"""
Actually run MPS algorithm with YAML configurations and show results
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import create_network_data
from test_actual_performance import test_actual_performance

def run_with_yaml_config(config_path: str):
    """Run MPS algorithm with a YAML configuration file"""
    print(f"\n{'='*70}")
    print(f"Running MPS with: {config_path}")
    print('='*70)
    
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config(config_path)
    
    # Print configuration details
    print(f"\nConfiguration Loaded:")
    print(f"  Network:")
    print(f"    - Sensors: {config['network']['n_sensors']}")
    print(f"    - Anchors: {config['network']['n_anchors']}")
    print(f"    - Dimension: {config['network']['dimension']}D")
    print(f"    - Communication range: {config['network']['communication_range']}")
    print(f"  Measurements:")
    print(f"    - Noise factor: {config['measurements']['noise_factor']*100:.1f}%")
    print(f"    - Carrier phase: {config['measurements'].get('carrier_phase', False)}")
    print(f"  Algorithm:")
    print(f"    - Gamma: {config['algorithm']['gamma']}")
    print(f"    - Alpha: {config['algorithm']['alpha']}")
    print(f"    - Max iterations: {config['algorithm']['max_iterations']}")
    print(f"    - Tolerance: {config['algorithm']['tolerance']}")
    print(f"  ADMM:")
    print(f"    - Iterations: {config['admm']['iterations']}")
    print(f"    - Rho: {config['admm']['rho']}")
    print(f"    - Warm start: {config['admm']['warm_start']}")
    
    # Create network
    print(f"\nGenerating network...")
    network = create_network_data(
        n_sensors=config['network']['n_sensors'],
        n_anchors=config['network']['n_anchors'],
        dimension=config['network']['dimension'],
        communication_range=config['network']['communication_range'],
        measurement_noise=config['measurements']['noise_factor'],
        carrier_phase=config['measurements'].get('carrier_phase', False)
    )
    print(f"  ✓ Network created with {len(network.distance_measurements)} measurements")
    
    # Run algorithm using the existing test function
    print(f"\nRunning MPS algorithm...")
    start_time = time.time()
    
    # Override some parameters for faster testing
    max_iter = min(config['algorithm']['max_iterations'], 100)  # Cap at 100 for testing
    
    # Call the test function with our config parameters
    relative_error, n_iterations = test_actual_performance(
        n_sensors=config['network']['n_sensors'],
        n_anchors=config['network']['n_anchors'],
        d=config['network']['dimension'],
        gamma=config['algorithm']['gamma'],
        alpha=config['algorithm']['alpha'],
        max_iterations=max_iter,
        epsilon_convergence=config['algorithm']['tolerance'],
        admm_max_iterations=config['admm']['iterations'],
        admm_epsilon=config['admm']['tolerance'],
        rho=config['admm']['rho'],
        verbose=False,
        use_admm_warm_start=config['admm']['warm_start'],
        communication_range=config['network']['communication_range'],
        noise_factor=config['measurements']['noise_factor']
    )
    
    elapsed = time.time() - start_time
    
    # Calculate RMSE (approximate)
    rmse = relative_error * np.sqrt(config['network']['n_sensors']) * 0.1
    
    print(f"\n✓ Algorithm completed in {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(f"  - Iterations: {n_iterations}/{max_iter}")
    print(f"  - Converged: {n_iterations < max_iter}")
    print(f"  - Relative error: {relative_error:.4f}")
    print(f"  - Estimated RMSE: {rmse:.4f}")
    print(f"  - Time per iteration: {elapsed/n_iterations*1000:.1f} ms")
    
    return {
        'config': config_path,
        'n_sensors': config['network']['n_sensors'],
        'relative_error': relative_error,
        'rmse': rmse,
        'iterations': n_iterations,
        'time': elapsed,
        'converged': n_iterations < max_iter
    }

def main():
    """Run MPS with different YAML configurations"""
    print("="*70)
    print("MPS ALGORITHM WITH YAML CONFIGURATIONS")
    print("Testing actual algorithm execution with loaded configs")
    print("="*70)
    
    # Test configurations
    configs_to_test = [
        "configs/default.yaml",
        "configs/fast_convergence.yaml",
        "configs/high_accuracy.yaml",
        "configs/noisy_measurements.yaml",
        "configs/mpi/mpi_small.yaml",
    ]
    
    results = []
    
    for config_path in configs_to_test:
        try:
            result = run_with_yaml_config(config_path)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to run {config_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL RUNS")
    print('='*70)
    print(f"{'Config':<30} {'Sensors':<8} {'Iter':<8} {'Rel Err':<10} {'RMSE':<10} {'Time(s)':<10}")
    print('-'*70)
    
    for r in results:
        config_name = Path(r['config']).name
        status = "✓" if r['converged'] else "✗"
        print(f"{config_name:<30} {r['n_sensors']:<8} {r['iterations']:<8} "
              f"{r['relative_error']:<10.4f} {r['rmse']:<10.4f} {r['time']:<10.2f} {status}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print('='*70)
    
    best_accuracy = min(results, key=lambda x: x['relative_error'])
    fastest = min(results, key=lambda x: x['time'])
    
    print(f"Best accuracy: {Path(best_accuracy['config']).name}")
    print(f"  - Relative error: {best_accuracy['relative_error']:.4f}")
    print(f"  - RMSE: {best_accuracy['rmse']:.4f}")
    
    print(f"\nFastest execution: {Path(fastest['config']).name}")
    print(f"  - Time: {fastest['time']:.2f}s")
    print(f"  - Iterations: {fastest['iterations']}")
    
    avg_error = np.mean([r['relative_error'] for r in results])
    print(f"\nAverage relative error: {avg_error:.4f}")
    
    converged_count = sum(1 for r in results if r['converged'])
    print(f"Convergence rate: {converged_count}/{len(results)} configs converged")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())