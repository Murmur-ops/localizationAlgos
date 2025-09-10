#!/usr/bin/env python3
"""
Quick MPI performance test
"""

import numpy as np
from mpi4py import MPI
import time
from mpi_performance_benchmark import MPIBenchmarkSuite

def main():
    """Run a quick performance test"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("="*60)
        print(f"MPI Performance Test - {size} processes")
        print("="*60)
    
    # Create benchmark suite
    benchmark = MPIBenchmarkSuite()
    
    # Test with small networks only
    sensor_counts = [10, 20, 30]
    
    results = []
    for n_sensors in sensor_counts:
        if n_sensors < size * 2:
            if rank == 0:
                print(f"Skipping {n_sensors} sensors (too few for {size} processes)")
            continue
            
        try:
            result = benchmark.run_single_benchmark(
                n_sensors=n_sensors,
                n_anchors=4,
                communication_range=0.5,
                noise_factor=0.05
            )
            results.append(result)
            
        except Exception as e:
            if rank == 0:
                print(f"Error with {n_sensors} sensors: {str(e)}")
    
    # Report results
    if rank == 0:
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        print(f"{'Sensors':>10} {'Time (s)':>10} {'Iter':>8} {'Error':>10}")
        print("-"*40)
        
        for r in results:
            print(f"{r.n_sensors:>10} {r.total_time:>10.3f} "
                  f"{r.iterations:>8} {r.final_error:>10.6f}")
        
        # Communication analysis
        comm_result = benchmark.analyze_communication_patterns(n_sensors=20)
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)

if __name__ == "__main__":
    main()