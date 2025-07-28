#!/usr/bin/env python3
"""
Minimal demonstration of MPI-based distributed SNL concepts
"""

import numpy as np
from mpi4py import MPI
import time

class MinimalDistributedSNL:
    """Minimal working example of distributed SNL"""
    
    def __init__(self, n_sensors=10, n_anchors=3):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        
        # Distribute sensors across processes
        sensors_per_proc = n_sensors // self.size
        remainder = n_sensors % self.size
        
        if self.rank < remainder:
            self.start_idx = self.rank * (sensors_per_proc + 1)
            self.end_idx = self.start_idx + sensors_per_proc + 1
        else:
            self.start_idx = self.rank * sensors_per_proc + remainder
            self.end_idx = self.start_idx + sensors_per_proc
            
        self.my_sensors = list(range(self.start_idx, self.end_idx))
        
        print(f"Process {self.rank}: Managing sensors {self.my_sensors}")
    
    def generate_network(self):
        """Generate a simple sensor network"""
        
        # Generate positions on root and broadcast
        if self.rank == 0:
            np.random.seed(42)
            self.anchor_positions = np.random.uniform(0, 1, (self.n_anchors, 2))
            self.true_positions = np.random.uniform(0, 1, (self.n_sensors, 2))
            print(f"Generated {self.n_sensors} sensors and {self.n_anchors} anchors")
        else:
            self.anchor_positions = None
            self.true_positions = None
        
        # Broadcast to all processes
        self.anchor_positions = self.comm.bcast(self.anchor_positions, root=0)
        self.true_positions = self.comm.bcast(self.true_positions, root=0)
        
        # Initialize local sensor estimates
        self.local_positions = {}
        for i in self.my_sensors:
            # Start with noisy initial guess
            self.local_positions[i] = self.true_positions[i] + 0.1 * np.random.randn(2)
    
    def compute_local_objective(self):
        """Compute local part of objective function"""
        
        local_obj = 0.0
        
        # For each local sensor
        for i in self.my_sensors:
            pos_i = self.local_positions[i]
            
            # Distance to anchors
            for j, anchor_pos in enumerate(self.anchor_positions):
                true_dist = np.linalg.norm(self.true_positions[i] - anchor_pos)
                est_dist = np.linalg.norm(pos_i - anchor_pos)
                local_obj += (est_dist - true_dist) ** 2
        
        return local_obj
    
    def run_simple_algorithm(self, iterations=10):
        """Run a simple distributed optimization"""
        
        if self.rank == 0:
            print(f"\nRunning simple distributed algorithm...")
        
        for iter in range(iterations):
            # 1. Compute local objective
            local_obj = self.compute_local_objective()
            
            # 2. Global reduction
            global_obj = self.comm.allreduce(local_obj, op=MPI.SUM)
            
            # 3. Simple gradient step (move towards true positions)
            for i in self.my_sensors:
                # Simplified update - in practice would use proper optimization
                error = self.true_positions[i] - self.local_positions[i]
                self.local_positions[i] += 0.1 * error
            
            # 4. Report progress
            if self.rank == 0 and iter % 5 == 0:
                avg_obj = global_obj / self.n_sensors
                print(f"  Iteration {iter}: Average objective = {avg_obj:.6f}")
        
        # Final error computation
        local_error = 0.0
        for i in self.my_sensors:
            error = np.linalg.norm(self.local_positions[i] - self.true_positions[i])
            local_error += error ** 2
            
        global_error = self.comm.allreduce(local_error, op=MPI.SUM)
        rmse = np.sqrt(global_error / self.n_sensors)
        
        if self.rank == 0:
            print(f"\nFinal RMSE: {rmse:.6f}")
    
    def demonstrate_communication_patterns(self):
        """Show different MPI communication patterns"""
        
        if self.rank == 0:
            print("\n" + "="*50)
            print("Demonstrating MPI Communication Patterns")
            print("="*50)
        
        # Pattern 1: Allgather - collect all positions
        local_data = {i: self.local_positions[i].tolist() for i in self.my_sensors}
        all_positions = self.comm.allgather(local_data)
        
        if self.rank == 0:
            print(f"1. Allgather: Collected positions from all {self.size} processes")
        
        # Pattern 2: Reduce operations
        n_local = len(self.my_sensors)
        n_total = self.comm.reduce(n_local, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            print(f"2. Reduce: Total sensors = {n_total}")
        
        # Pattern 3: Non-blocking communication
        if self.size > 1:
            next_rank = (self.rank + 1) % self.size
            prev_rank = (self.rank - 1) % self.size
            
            # Non-blocking send and receive
            send_data = np.array([self.rank, len(self.my_sensors)])
            recv_data = np.empty(2)
            
            req_send = self.comm.Isend(send_data, dest=next_rank)
            req_recv = self.comm.Irecv(recv_data, source=prev_rank)
            
            # Do some work while communication happens
            dummy_work = np.sum([np.random.randn(100) for _ in range(10)])
            
            # Wait for completion
            req_send.Wait()
            req_recv.Wait()
            
            print(f"3. Process {self.rank}: Received data {recv_data} from {prev_rank}")
        
        self.comm.Barrier()
        if self.rank == 0:
            print("\nCommunication patterns demonstrated successfully!")

def main():
    """Run the minimal demonstration"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("="*60)
        print(f"Minimal Distributed SNL Demo - {size} processes")
        print("="*60)
    
    # Create and run minimal SNL
    snl = MinimalDistributedSNL(n_sensors=20, n_anchors=4)
    
    # Generate network
    snl.generate_network()
    
    # Run simple algorithm
    start_time = time.time()
    snl.run_simple_algorithm(iterations=20)
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"\nTotal time: {total_time:.3f} seconds")
    
    # Demonstrate communication patterns
    snl.demonstrate_communication_patterns()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)

if __name__ == "__main__":
    main()