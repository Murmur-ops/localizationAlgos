#!/usr/bin/env python3
"""
Performance testing script for different network configurations
Tests the 20 nodes with 8 anchors configuration and compares single vs MPI execution
"""

import time
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).parent))

from mps_core import MPSAlgorithm, MPSConfig


def test_single_process(config):
    """Test single process execution"""
    print("\n" + "="*60)
    print("SINGLE PROCESS EXECUTION")
    print("="*60)
    
    # Create algorithm instance
    mps = MPSAlgorithm(config)
    mps.generate_network()
    
    # Run multiple times for statistics
    times = []
    iterations = []
    rmse_values = []
    
    n_runs = 3
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        start = time.time()
        results = mps.run()
        elapsed = time.time() - start
        
        times.append(elapsed)
        iterations.append(results['iterations'])
        if results['final_rmse']:
            rmse_values.append(results['final_rmse'])
        
        print(f"  Time: {elapsed:.3f}s, Iterations: {results['iterations']}, "
              f"RMSE: {results['final_rmse']:.4f}")
    
    print(f"\nSingle Process Statistics (n={n_runs}):")
    print(f"  Avg Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"  Avg Iterations: {np.mean(iterations):.0f} ± {np.std(iterations):.1f}")
    print(f"  Avg RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
    print(f"  Time per iteration: {np.mean(times)/np.mean(iterations)*1000:.2f} ms")
    
    return {
        'mode': 'single',
        'avg_time': np.mean(times),
        'avg_iterations': np.mean(iterations),
        'avg_rmse': np.mean(rmse_values),
        'time_per_iter_ms': np.mean(times)/np.mean(iterations)*1000
    }


def test_mpi_execution(config_path, n_processes=4):
    """Test MPI distributed execution"""
    print("\n" + "="*60)
    print(f"MPI DISTRIBUTED EXECUTION ({n_processes} processes)")
    print("="*60)
    
    times = []
    iterations = []
    rmse_values = []
    
    n_runs = 3
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        
        # Run MPI process
        cmd = f"mpirun -n {n_processes} python3 run_distributed.py --config {config_path} --quiet"
        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - start
        
        # Parse output to get statistics
        output = result.stdout
        for line in output.split('\n'):
            if 'Iterations:' in line:
                iter_val = int(line.split(':')[1].strip())
                iterations.append(iter_val)
            elif 'Final RMSE:' in line:
                rmse_val = float(line.split(':')[1].strip())
                rmse_values.append(rmse_val)
        
        times.append(elapsed)
        
        if iterations and rmse_values:
            print(f"  Time: {elapsed:.3f}s, Iterations: {iterations[-1]}, "
                  f"RMSE: {rmse_values[-1]:.4f}")
    
    if times and iterations:
        print(f"\nMPI Statistics ({n_processes} processes, n={n_runs}):")
        print(f"  Avg Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Avg Iterations: {np.mean(iterations):.0f} ± {np.std(iterations):.1f}")
        if rmse_values:
            print(f"  Avg RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
        print(f"  Time per iteration: {np.mean(times)/np.mean(iterations)*1000:.2f} ms")
        
        return {
            'mode': f'mpi_{n_processes}',
            'avg_time': np.mean(times),
            'avg_iterations': np.mean(iterations),
            'avg_rmse': np.mean(rmse_values) if rmse_values else None,
            'time_per_iter_ms': np.mean(times)/np.mean(iterations)*1000
        }
    else:
        print("  MPI execution failed or no results parsed")
        return None


def main():
    """Run performance tests for 20 nodes with 8 anchors"""
    
    print("\n" + "="*70)
    print(" PERFORMANCE TEST: 20 NODES WITH 8 ANCHORS ")
    print("="*70)
    
    # Configuration
    config = MPSConfig(
        n_sensors=20,
        n_anchors=8,
        communication_range=0.35,
        noise_factor=0.05,
        gamma=0.98,
        alpha=1.2,
        max_iterations=400,
        tolerance=0.00005,
        dimension=2,
        seed=2024
    )
    
    config_path = "config/examples/20_nodes_8_anchors.yaml"
    
    print(f"\nConfiguration:")
    print(f"  Sensors: {config.n_sensors}")
    print(f"  Anchors: {config.n_anchors}")
    print(f"  Communication Range: {config.communication_range}")
    print(f"  Noise Factor: {config.noise_factor}")
    print(f"  Max Iterations: {config.max_iterations}")
    
    # Test single process
    single_results = test_single_process(config)
    
    # Test MPI with 2 processes
    mpi2_results = test_mpi_execution(config_path, n_processes=2)
    
    # Test MPI with 4 processes
    mpi4_results = test_mpi_execution(config_path, n_processes=4)
    
    # Summary comparison
    print("\n" + "="*70)
    print(" PERFORMANCE COMPARISON SUMMARY ")
    print("="*70)
    
    print(f"\n{'Mode':<15} {'Time (s)':<12} {'Iterations':<12} {'RMSE':<12} {'ms/iter':<12}")
    print("-"*70)
    
    if single_results:
        print(f"{'Single':<15} {single_results['avg_time']:<12.3f} "
              f"{single_results['avg_iterations']:<12.0f} "
              f"{single_results['avg_rmse']:<12.4f} "
              f"{single_results['time_per_iter_ms']:<12.2f}")
    
    if mpi2_results:
        print(f"{'MPI (2 proc)':<15} {mpi2_results['avg_time']:<12.3f} "
              f"{mpi2_results['avg_iterations']:<12.0f} "
              f"{mpi2_results['avg_rmse'] or 'N/A':<12} "
              f"{mpi2_results['time_per_iter_ms']:<12.2f}")
    
    if mpi4_results:
        print(f"{'MPI (4 proc)':<15} {mpi4_results['avg_time']:<12.3f} "
              f"{mpi4_results['avg_iterations']:<12.0f} "
              f"{mpi4_results['avg_rmse'] or 'N/A':<12} "
              f"{mpi4_results['time_per_iter_ms']:<12.2f}")
    
    # Speedup analysis
    if single_results and mpi4_results:
        speedup = single_results['avg_time'] / mpi4_results['avg_time']
        print(f"\nSpeedup (4 processes vs single): {speedup:.2f}x")
    
    print("\n" + "="*70)
    
    # Save results
    results_file = "results/20_nodes/performance_comparison.json"
    Path("results/20_nodes").mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'configuration': {
            'n_sensors': config.n_sensors,
            'n_anchors': config.n_anchors,
            'communication_range': config.communication_range,
            'noise_factor': config.noise_factor
        },
        'single_process': single_results,
        'mpi_2_processes': mpi2_results,
        'mpi_4_processes': mpi4_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()