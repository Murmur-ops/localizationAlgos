#!/usr/bin/env python3
"""
Basic test of MPI SNL components
"""

import numpy as np
from mpi4py import MPI
import time

def test_basic_distributed_computation():
    """Test basic distributed computation pattern"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    print(f"Process {rank}/{size} starting")
    
    # Test 1: Distributed matrix generation
    n_sensors = 10
    sensors_per_proc = n_sensors // size
    
    # Each process handles some sensors
    if rank < n_sensors % size:
        my_sensors = list(range(rank * (sensors_per_proc + 1), 
                               (rank + 1) * (sensors_per_proc + 1)))
    else:
        offset = (n_sensors % size)
        my_sensors = list(range(offset + rank * sensors_per_proc,
                               offset + (rank + 1) * sensors_per_proc))
    
    print(f"Rank {rank}: Managing sensors {my_sensors}")
    
    # Test 2: Simple collective operation
    local_data = np.array([rank + 1.0] * len(my_sensors))
    print(f"Rank {rank}: Local data = {local_data}")
    
    # Sum across all processes
    local_sum = np.sum(local_data)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    
    print(f"Rank {rank}: Local sum = {local_sum}, Global sum = {global_sum}")
    
    # Test 3: Point-to-point communication pattern
    if size > 1:
        # Send to next process in ring
        next_rank = (rank + 1) % size
        prev_rank = (rank - 1) % size
        
        send_data = f"Hello from {rank}"
        recv_data = comm.sendrecv(send_data, dest=next_rank, source=prev_rank)
        
        print(f"Rank {rank}: Sent '{send_data}' to {next_rank}, "
              f"Received '{recv_data}' from {prev_rank}")
    
    # Test 4: Barrier synchronization
    comm.Barrier()
    if rank == 0:
        print("\nAll processes synchronized!")
    
    # Test 5: Simple Sinkhorn-Knopp iteration
    if rank == 0:
        print("\nTesting distributed Sinkhorn-Knopp pattern...")
    
    # Initialize local matrix block
    local_matrix = np.ones((len(my_sensors), n_sensors)) / n_sensors
    
    # One iteration of row normalization
    row_sums = np.sum(local_matrix, axis=1)
    local_matrix = local_matrix / row_sums[:, np.newaxis]
    
    # Compute column sums (need global reduction)
    local_col_sums = np.sum(local_matrix, axis=0)
    global_col_sums = np.zeros(n_sensors)
    comm.Allreduce(local_col_sums, global_col_sums, op=MPI.SUM)
    
    if rank == 0:
        print(f"Global column sums: {global_col_sums[:5]}... (first 5)")
    
    comm.Barrier()
    print(f"Rank {rank}: Test completed successfully!")

def test_sensor_network_pattern():
    """Test sensor network communication pattern"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("\n" + "="*50)
        print("Testing Sensor Network Pattern")
        print("="*50)
    
    # Simulate sensor positions
    n_sensors = 8
    if rank == 0:
        # Generate positions
        np.random.seed(42)
        positions = np.random.uniform(0, 1, (n_sensors, 2))
    else:
        positions = None
    
    # Broadcast positions
    positions = comm.bcast(positions, root=0)
    
    # Each process computes distances for its sensors
    sensors_per_proc = n_sensors // size
    my_start = rank * sensors_per_proc
    my_end = my_start + sensors_per_proc
    if rank == size - 1:  # Last process takes remainder
        my_end = n_sensors
    
    my_sensors = list(range(my_start, my_end))
    print(f"Rank {rank}: Computing distances for sensors {my_sensors}")
    
    # Compute distances
    for i in my_sensors:
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbors = np.where((distances > 0) & (distances < 0.5))[0]
        print(f"  Sensor {i}: {len(neighbors)} neighbors")
    
    comm.Barrier()
    if rank == 0:
        print("\nSensor network pattern test completed!")

if __name__ == "__main__":
    # Run basic test
    test_basic_distributed_computation()
    
    # Run sensor network pattern test
    test_sensor_network_pattern()
    
    # Final message
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("\n" + "="*50)
        print("All basic MPI SNL tests passed!")
        print("="*50)