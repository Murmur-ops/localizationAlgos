#!/usr/bin/env python3
"""
Test script for the fixed distributed MPS implementation
Compares single-process vs fixed distributed execution
"""

import sys
import time
from pathlib import Path
from mpi4py import MPI
import numpy as np

sys.path.append(str(Path(__file__).parent))

from mps_core import MPSAlgorithm, DistributedMPSFixed, MPSConfig


def test_single_vs_distributed():
    """Compare single process and fixed distributed implementations"""
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # Configuration for 20 nodes with 8 anchors
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
    
    if rank == 0:
        print("\n" + "="*70)
        print(" TESTING FIXED DISTRIBUTED MPS IMPLEMENTATION ")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Sensors: {config.n_sensors}")
        print(f"  Anchors: {config.n_anchors}")
        print(f"  MPI Processes: {size}")
        
        # First, run single-process baseline for comparison
        print("\n--- SINGLE PROCESS BASELINE ---")
        mps_single = MPSAlgorithm(config)
        mps_single.generate_network()
        
        start = time.time()
        single_results = mps_single.run()
        single_time = time.time() - start
        
        print(f"  Converged: {single_results['converged']}")
        print(f"  Iterations: {single_results['iterations']}")
        print(f"  Final RMSE: {single_results['final_rmse']:.4f}")
        print(f"  Runtime: {single_time:.3f}s")
    
    # Synchronize before distributed run
    comm.Barrier()
    
    # Run fixed distributed implementation
    if rank == 0:
        print(f"\n--- FIXED DISTRIBUTED ({size} processes) ---")
    
    distributed_mps = DistributedMPSFixed(config, comm)
    
    start = time.time()
    distributed_results = distributed_mps.run_distributed()
    distributed_time = time.time() - start
    
    # Results only on rank 0
    if rank == 0 and distributed_results:
        print(f"  Converged: {distributed_results['converged']}")
        print(f"  Iterations: {distributed_results['iterations']}")
        print(f"  Final RMSE: {distributed_results['final_rmse']:.4f}")
        print(f"  Runtime: {distributed_time:.3f}s")
        
        # Compare results
        print("\n" + "="*70)
        print(" COMPARISON RESULTS ")
        print("="*70)
        
        rmse_diff = abs(single_results['final_rmse'] - distributed_results['final_rmse'])
        rmse_ratio = distributed_results['final_rmse'] / single_results['final_rmse']
        
        print(f"\nRMSE Comparison:")
        print(f"  Single Process:    {single_results['final_rmse']:.4f}")
        print(f"  Distributed Fixed: {distributed_results['final_rmse']:.4f}")
        print(f"  Difference:        {rmse_diff:.4f}")
        print(f"  Ratio:             {rmse_ratio:.2f}x")
        
        print(f"\nPerformance:")
        print(f"  Single Process:    {single_time:.3f}s")
        print(f"  Distributed Fixed: {distributed_time:.3f}s")
        print(f"  Speedup:           {single_time/distributed_time:.2f}x")
        
        # Success criteria
        if rmse_ratio < 1.1:  # Within 10% is acceptable
            print("\n✅ SUCCESS: Distributed RMSE matches single-process!")
        else:
            print("\n⚠️  WARNING: Distributed RMSE still differs significantly")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    test_single_vs_distributed()