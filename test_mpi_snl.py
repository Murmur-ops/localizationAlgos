#!/usr/bin/env python3
"""
Test the optimized MPI SNL implementation
"""

import numpy as np
from mpi4py import MPI
import time
import sys

# Import our implementations
from snl_mpi_optimized import OptimizedMPISNL
from mpi_distributed_operations import DistributedMatrixOps, DistributedSinkhornKnopp

def test_small_network():
    """Test with a small sensor network"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("="*60)
        print("Testing Small Sensor Network")
        print(f"Processes: {size}")
        print("="*60)
    
    # Small network parameters
    problem_params = {
        'n_sensors': 20,
        'n_anchors': 4,
        'd': 2,
        'communication_range': 0.5,
        'noise_factor': 0.05,
        'gamma': 0.999,
        'alpha_mps': 10.0,
        'max_iter': 100,
        'tol': 1e-3
    }
    
    # Create solver
    solver = OptimizedMPISNL(problem_params)
    
    # Generate network
    if rank == 0:
        print("\n1. Generating network...")
    solver.generate_network()
    
    # Report local sensors
    print(f"Process {rank}: Managing sensors {solver.local_sensors}")
    comm.Barrier()
    
    # Compute matrix parameters
    if rank == 0:
        print("\n2. Computing matrix parameters...")
    start = time.time()
    solver.compute_matrix_parameters_optimized()
    matrix_time = time.time() - start
    
    if rank == 0:
        print(f"   Matrix generation time: {matrix_time:.3f}s")
    
    # Run MPS algorithm
    if rank == 0:
        print("\n3. Running MPS algorithm...")
    
    results = solver.run_mps_optimized(max_iter=50)
    
    # Report results
    if rank == 0:
        print(f"\nResults:")
        print(f"  Converged: {results['converged']}")
        print(f"  Iterations: {results['iterations']}")
        if results['objectives']:
            print(f"  Final objective: {results['objectives'][-1]:.6f}")
        if results['errors']:
            print(f"  Final error: {results['errors'][-1]:.6f}")
        print(f"\nTiming breakdown:")
        for category, time_spent in results['timing_stats'].items():
            print(f"  {category}: {time_spent:.3f}s")

def test_distributed_operations():
    """Test distributed matrix operations"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Testing Distributed Matrix Operations")
        print("="*60)
    
    # Test parameters
    n_sensors = 12
    sensors_per_proc = n_sensors // size
    remainder = n_sensors % size
    
    # Compute local sensors
    if rank < remainder:
        start = rank * (sensors_per_proc + 1)
        end = start + sensors_per_proc + 1
    else:
        start = rank * sensors_per_proc + remainder
        end = start + sensors_per_proc
    
    local_sensors = list(range(start, end))
    
    # Create matrix operations handler
    matrix_ops = DistributedMatrixOps(comm, n_sensors)
    
    # Test 1: Distributed Sinkhorn-Knopp
    if rank == 0:
        print("\n1. Testing Distributed Sinkhorn-Knopp...")
    
    # Create simple adjacency (ring topology)
    adjacency = {}
    for sensor in local_sensors:
        neighbors = {(sensor - 1) % n_sensors, (sensor + 1) % n_sensors}
        adjacency[sensor] = neighbors
    
    sk = DistributedSinkhornKnopp(matrix_ops)
    L_blocks = sk.compute(adjacency, max_iter=50)
    
    # Verify doubly stochastic property
    row_sums_local = {sid: sum(L_row.values()) for sid, L_row in L_blocks.items()}
    
    if rank == 0:
        print(f"   Generated doubly stochastic matrix blocks")
        print(f"   Example row sums: {list(row_sums_local.values())[:3]}")
    
    # Test 2: Distributed matrix-vector multiplication
    if rank == 0:
        print("\n2. Testing Distributed Matrix-Vector Multiplication...")
    
    # Create test vector
    v_local = {sensor: np.array([1.0, 0.5]) for sensor in local_sensors}
    
    # Perform multiplication
    result = matrix_ops.distributed_L_multiply(L_blocks, v_local)
    
    # Compute norm
    norm = matrix_ops.distributed_norm(result)
    
    if rank == 0:
        print(f"   Input vector norm: {matrix_ops.distributed_norm(v_local):.6f}")
        print(f"   Result vector norm: {norm:.6f}")
    
    # Test 3: Communication pattern analysis
    if rank == 0:
        print("\n3. Communication Pattern Analysis...")
    
    local_edges = sum(len(L_row) for L_row in L_blocks.values())
    total_edges = comm.allreduce(local_edges, op=MPI.SUM)
    
    if rank == 0:
        print(f"   Total edges: {total_edges}")
        print(f"   Average edges per process: {total_edges/size:.1f}")

def run_scaling_test():
    """Run a simple scaling test"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Scaling Test")
        print("="*60)
    
    # Test different problem sizes
    sensor_counts = [20, 50, 100]
    
    for n_sensors in sensor_counts:
        # Skip if too few sensors per process
        if n_sensors < size * 2:
            continue
        
        problem_params = {
            'n_sensors': n_sensors,
            'n_anchors': max(4, n_sensors // 10),
            'd': 2,
            'communication_range': 0.4,
            'noise_factor': 0.05,
            'gamma': 0.999,
            'alpha_mps': 10.0,
            'max_iter': 50,
            'tol': 1e-3
        }
        
        if rank == 0:
            print(f"\nTesting with {n_sensors} sensors...")
        
        solver = OptimizedMPISNL(problem_params)
        
        # Time the full process
        start = time.time()
        solver.generate_network()
        solver.compute_matrix_parameters_optimized()
        results = solver.run_mps_optimized(max_iter=30)
        total_time = time.time() - start
        
        if rank == 0:
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Iterations: {results['iterations']}")
            if results['errors']:
                print(f"  Final error: {results['errors'][-1]:.6f}")

def main():
    """Run all tests"""
    
    comm = MPI.COMM_WORLD
    
    try:
        # Test 1: Small network
        test_small_network()
        comm.Barrier()
        
        # Test 2: Distributed operations
        test_distributed_operations()
        comm.Barrier()
        
        # Test 3: Scaling
        run_scaling_test()
        comm.Barrier()
        
        if comm.rank == 0:
            print("\n" + "="*60)
            print("All MPI tests completed successfully!")
            print("="*60)
            
    except Exception as e:
        print(f"Error on rank {comm.rank}: {str(e)}")
        import traceback
        traceback.print_exc()
        comm.Abort(1)

if __name__ == "__main__":
    main()