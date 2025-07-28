#!/usr/bin/env python3
"""
Simple test of MPI functionality
"""

from mpi4py import MPI
import numpy as np
import time

def test_basic_mpi():
    """Test basic MPI functionality"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    print(f"Process {rank} of {size} is running")
    
    # Test 1: Basic communication
    if rank == 0:
        data = {'numbers': [1, 2, 3, 4, 5], 'text': 'Hello from rank 0'}
        print(f"Rank 0: Broadcasting data: {data}")
    else:
        data = None
    
    data = comm.bcast(data, root=0)
    print(f"Rank {rank}: Received data: {data}")
    
    # Test 2: Scatter and gather
    if rank == 0:
        sendbuf = np.arange(size * 3, dtype='i').reshape(size, 3)
        print(f"Rank 0: Scattering array:\n{sendbuf}")
    else:
        sendbuf = None
    
    recvbuf = np.empty(3, dtype='i')
    comm.Scatter(sendbuf, recvbuf, root=0)
    print(f"Rank {rank}: Received scatter: {recvbuf}")
    
    # Test 3: Reduce
    local_sum = rank + 1
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Rank 0: Global sum = {global_sum}, expected = {sum(range(1, size+1))}")
    
    # Test 4: Barrier synchronization
    comm.Barrier()
    print(f"Rank {rank}: Passed barrier")
    
    # Test 5: Point-to-point communication
    if size > 1:
        if rank == 0:
            msg = "Hello from rank 0!"
            comm.send(msg, dest=1, tag=11)
            print(f"Rank 0: Sent message to rank 1")
        elif rank == 1:
            msg = comm.recv(source=0, tag=11)
            print(f"Rank 1: Received message: '{msg}'")

def test_distributed_computation():
    """Test a simple distributed computation"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # Distributed vector norm computation
    n = 100
    local_n = n // size
    
    # Each process computes part of a vector
    np.random.seed(rank)
    local_vector = np.random.randn(local_n)
    
    # Local computation
    local_sum = np.sum(local_vector ** 2)
    
    # Global reduction
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_norm = np.sqrt(global_sum)
    
    if rank == 0:
        print(f"\nDistributed computation test:")
        print(f"Vector size: {n}")
        print(f"Processes: {size}")
        print(f"Global norm: {global_norm:.6f}")

if __name__ == "__main__":
    print("="*50)
    print("Testing MPI Installation")
    print("="*50)
    
    test_basic_mpi()
    test_distributed_computation()
    
    # Final synchronization
    comm = MPI.COMM_WORLD
    comm.Barrier()
    
    if comm.rank == 0:
        print("\nAll tests completed successfully!")