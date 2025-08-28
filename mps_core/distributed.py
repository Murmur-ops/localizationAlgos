"""
Distributed MPS implementation using MPI
Each MPI process handles a subset of sensors
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Set, Optional, Tuple
import logging

from .algorithm import MPSConfig, MPSState
from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations


class DistributedMPS:
    """
    Distributed Matrix-Parametrized Proximal Splitting using MPI
    Each process owns a subset of sensors and performs local computations
    """
    
    def __init__(self, config: MPSConfig, comm: Optional[MPI.Comm] = None):
        """
        Initialize distributed MPS
        
        Args:
            config: Algorithm configuration
            comm: MPI communicator (defaults to COMM_WORLD)
        """
        self.config = config
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        # Logging setup
        logging.basicConfig(level=logging.INFO if self.rank == 0 else logging.WARNING)
        self.logger = logging.getLogger(f"MPS_Rank{self.rank}")
        
        # Local sensor assignment
        self.local_sensors = []
        self.sensor_to_rank = {}
        
        # Local data structures
        self.local_positions = {}
        self.local_measurements = {}
        self.local_anchor_distances = {}
        self.neighbor_ranks = set()  # Ranks we need to communicate with
        
        # Global network data (broadcast from rank 0)
        self.anchor_positions = None
        self.adjacency = None
        self.Z_matrix_local = None
        
    def distribute_sensors(self):
        """Distribute sensors across MPI processes"""
        n = self.config.n_sensors
        sensors_per_rank = n // self.size
        remainder = n % self.size
        
        # Assign sensors to ranks
        sensor_id = 0
        for rank in range(self.size):
            count = sensors_per_rank + (1 if rank < remainder else 0)
            for _ in range(count):
                if rank == self.rank:
                    self.local_sensors.append(sensor_id)
                self.sensor_to_rank[sensor_id] = rank
                sensor_id += 1
        
        # Broadcast sensor assignments
        self.sensor_to_rank = self.comm.bcast(self.sensor_to_rank, root=0)
        
        if self.rank == 0:
            self.logger.info(f"Distributed {n} sensors across {self.size} processes")
            self.logger.info(f"Rank 0 owns sensors: {self.local_sensors}")
    
    def generate_network_distributed(self):
        """Generate network data in distributed manner"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        if self.rank == 0:
            # Generate complete network on rank 0
            np.random.seed(self.config.seed)
            
            # True positions
            true_positions = {}
            for i in range(n):
                true_positions[i] = np.random.uniform(0, 1, d)
            
            # Anchors
            if self.config.n_anchors > 0 and d == 2:
                anchor_positions = np.array([
                    [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
                ])
                for i in range(4, self.config.n_anchors):
                    anchor_positions = np.vstack([
                        anchor_positions,
                        np.random.uniform(0.2, 0.8, d)
                    ])
                anchor_positions = anchor_positions[:self.config.n_anchors]
            else:
                anchor_positions = np.random.uniform(0, 1, (self.config.n_anchors, d))
            
            # Adjacency matrix
            adjacency = MatrixOperations.build_adjacency(
                true_positions,
                self.config.communication_range
            )
            
            # Distance measurements
            all_measurements = {}
            for i in range(n):
                for j in range(i+1, n):
                    if adjacency[i, j] > 0:
                        true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
                        noise = self.config.noise_factor * np.random.randn()
                        noisy_dist = true_dist * (1 + noise)
                        all_measurements[(i, j)] = max(0.01, noisy_dist)
                        all_measurements[(j, i)] = all_measurements[(i, j)]
            
            # Anchor distances
            all_anchor_distances = {}
            for i in range(n):
                all_anchor_distances[i] = {}
                for k in range(self.config.n_anchors):
                    true_dist = np.linalg.norm(true_positions[i] - anchor_positions[k])
                    if true_dist <= self.config.communication_range:
                        noise = self.config.noise_factor * np.random.randn()
                        noisy_dist = true_dist * (1 + noise)
                        all_anchor_distances[i][k] = max(0.01, noisy_dist)
        else:
            true_positions = None
            anchor_positions = None
            adjacency = None
            all_measurements = None
            all_anchor_distances = None
        
        # Broadcast global data
        self.anchor_positions = self.comm.bcast(anchor_positions, root=0)
        self.adjacency = self.comm.bcast(adjacency, root=0)
        all_measurements = self.comm.bcast(all_measurements, root=0)
        all_anchor_distances = self.comm.bcast(all_anchor_distances, root=0)
        
        # Extract local data
        for sensor in self.local_sensors:
            # Local measurements
            for (i, j), dist in all_measurements.items():
                if i == sensor or j == sensor:
                    self.local_measurements[(i, j)] = dist
            
            # Local anchor distances
            if sensor in all_anchor_distances:
                self.local_anchor_distances[sensor] = all_anchor_distances[sensor]
        
        # Identify neighbor ranks for communication
        for sensor in self.local_sensors:
            for j in range(n):
                if self.adjacency[sensor, j] > 0 and j not in self.local_sensors:
                    self.neighbor_ranks.add(self.sensor_to_rank[j])
        
        # Create local consensus matrix blocks
        self._create_local_consensus_matrix()
        
        # Store true positions for evaluation (only for testing)
        self.true_positions = self.comm.bcast(true_positions, root=0)
    
    def _create_local_consensus_matrix(self):
        """Create local blocks of consensus matrix"""
        # For simplicity, each rank stores full Z matrix
        # In production, would only store relevant blocks
        self.Z_matrix_local = MatrixOperations.create_consensus_matrix(
            self.adjacency,
            self.config.gamma
        )
    
    def initialize_state_distributed(self) -> MPSState:
        """Initialize local state variables"""
        n = self.config.n_sensors
        d = self.config.dimension
        n_local = len(self.local_sensors)
        
        # Initialize local positions
        for sensor in self.local_sensors:
            if sensor in self.local_anchor_distances and len(self.local_anchor_distances[sensor]) > 0:
                # Initialize near anchors
                anchor_ids = list(self.local_anchor_distances[sensor].keys())
                self.local_positions[sensor] = np.mean(
                    self.anchor_positions[anchor_ids], axis=0
                )
                self.local_positions[sensor] += 0.05 * np.random.randn(d)
            else:
                self.local_positions[sensor] = np.random.uniform(0, 1, d)
        
        # Initialize full state vectors (needed for consensus)
        X = np.zeros((2*n, d))
        Y = np.zeros((2*n, d))
        U = np.zeros((2*n, d))
        
        # Fill in local sensor data
        for sensor in self.local_sensors:
            X[sensor] = self.local_positions[sensor]
            X[sensor + n] = self.local_positions[sensor]
            Y[sensor] = self.local_positions[sensor]
            Y[sensor + n] = self.local_positions[sensor]
        
        return MPSState(
            positions=self.local_positions.copy(),
            X=X,
            Y=Y,
            U=U
        )
    
    def exchange_positions(self, state: MPSState):
        """Exchange position updates with neighboring processes"""
        n = self.config.n_sensors
        
        # Prepare send/receive buffers
        send_requests = []
        recv_requests = []
        
        # Send local positions to neighbors
        for rank in self.neighbor_ranks:
            # Pack local sensor positions
            send_data = {}
            for sensor in self.local_sensors:
                send_data[sensor] = state.X[sensor]
            
            req = self.comm.isend(send_data, dest=rank, tag=1)
            send_requests.append(req)
        
        # Receive positions from neighbors
        recv_buffers = {}
        for rank in self.neighbor_ranks:
            req = self.comm.irecv(source=rank, tag=1)
            recv_requests.append((rank, req))
            
        # Wait for all communications
        MPI.Request.Waitall(send_requests)
        
        for rank, req in recv_requests:
            recv_data = req.wait()
            # Update non-local sensor positions
            for sensor, position in recv_data.items():
                if sensor not in self.local_sensors:
                    state.X[sensor] = position
                    state.X[sensor + n] = position
    
    def local_prox_step(self, state: MPSState) -> np.ndarray:
        """Apply proximal operator locally for owned sensors"""
        n = self.config.n_sensors
        X_new = state.X.copy()
        
        for sensor in self.local_sensors:
            position = X_new[sensor]
            
            # Apply distance constraints with neighbors
            for j in range(n):
                if (sensor, j) in self.local_measurements:
                    measured_dist = self.local_measurements[(sensor, j)]
                    position = ProximalOperators.prox_distance(
                        position, X_new[j], measured_dist,
                        alpha=self.config.alpha / 10
                    )
            
            # Apply anchor constraints
            if sensor in self.local_anchor_distances:
                for k, measured_dist in self.local_anchor_distances[sensor].items():
                    position = ProximalOperators.prox_distance(
                        position, self.anchor_positions[k], measured_dist,
                        alpha=self.config.alpha / 5
                    )
            
            # Update both blocks
            X_new[sensor] = position
            X_new[sensor + n] = position
            
            # Box constraints
            X_new[sensor] = ProximalOperators.prox_box_constraint(
                X_new[sensor], -0.5, 1.5
            )
            X_new[sensor + n] = X_new[sensor]
        
        return X_new
    
    def distributed_consensus(self, state: MPSState):
        """Perform distributed consensus step"""
        # Apply consensus matrix multiplication
        # For simplicity, using full matrix (could optimize with sparse operations)
        state.Y = self.Z_matrix_local @ state.X
        
        # Ensure consistency across processes
        # Allreduce for consensus variables
        Y_global = np.zeros_like(state.Y)
        self.comm.Allreduce(state.Y, Y_global, op=MPI.SUM)
        
        # Average based on ownership
        n = self.config.n_sensors
        ownership_count = np.zeros(2 * n)
        for sensor in range(n):
            if sensor in self.local_sensors:
                ownership_count[sensor] = 1
                ownership_count[sensor + n] = 1
        
        ownership_global = np.zeros_like(ownership_count)
        self.comm.Allreduce(ownership_count, ownership_global, op=MPI.SUM)
        
        # Normalize by ownership count
        for i in range(2 * n):
            if ownership_global[i] > 0:
                state.Y[i] = Y_global[i] / ownership_global[i]
    
    def compute_local_objective(self, state: MPSState) -> float:
        """Compute local contribution to objective"""
        local_error = 0.0
        local_count = 0
        
        # Local sensor-to-sensor errors
        for (i, j), measured_dist in self.local_measurements.items():
            if i in self.local_sensors and i < j:
                actual_dist = np.linalg.norm(state.positions[i] - state.positions.get(j, state.X[j]))
                local_error += (actual_dist - measured_dist) ** 2
                local_count += 1
        
        # Local anchor errors
        for sensor in self.local_sensors:
            if sensor in self.local_anchor_distances:
                for k, measured_dist in self.local_anchor_distances[sensor].items():
                    actual_dist = np.linalg.norm(
                        state.positions[sensor] - self.anchor_positions[k]
                    )
                    local_error += (actual_dist - measured_dist) ** 2
                    local_count += 1
        
        return local_error, local_count
    
    def run_distributed(self) -> Dict:
        """
        Run distributed MPS algorithm
        
        Returns:
            Dictionary with results (on rank 0)
        """
        # Distribute sensors
        self.distribute_sensors()
        
        # Generate network
        self.generate_network_distributed()
        
        # Initialize state
        state = self.initialize_state_distributed()
        
        # Metrics tracking (rank 0 only)
        if self.rank == 0:
            objective_history = []
            rmse_history = []
        
        # Main iteration loop
        for iteration in range(self.config.max_iterations):
            X_old = state.X.copy()
            
            # Step 1: Exchange current positions with neighbors
            self.exchange_positions(state)
            
            # Step 2: Local proximal step
            state.X = self.local_prox_step(state)
            
            # Step 3: Distributed consensus
            self.distributed_consensus(state)
            
            # Step 4: Dual update (local)
            state.U = state.U + self.config.alpha * (state.X - state.Y)
            
            # Step 5: Extract local position estimates
            n = self.config.n_sensors
            for sensor in self.local_sensors:
                state.positions[sensor] = (state.Y[sensor] + state.Y[sensor + n]) / 2
            
            # Check convergence and compute metrics periodically
            if iteration % 10 == 0:
                # Compute local change
                local_change = np.linalg.norm(state.X - X_old)
                local_norm = np.linalg.norm(X_old) + 1e-10
                
                # Global convergence check
                global_change = self.comm.allreduce(local_change, op=MPI.SUM)
                global_norm = self.comm.allreduce(local_norm, op=MPI.SUM)
                relative_change = global_change / global_norm
                
                if self.rank == 0:
                    # Compute global objective
                    local_err, local_cnt = self.compute_local_objective(state)
                    total_error = self.comm.reduce(local_err, op=MPI.SUM, root=0)
                    total_count = self.comm.reduce(local_cnt, op=MPI.SUM, root=0)
                    
                    if total_count > 0:
                        objective = np.sqrt(total_error / total_count)
                        objective_history.append(objective)
                    
                    # Compute RMSE if true positions available
                    if self.true_positions is not None:
                        # Gather all positions
                        all_positions = self.comm.gather(state.positions, root=0)
                        merged_positions = {}
                        for pos_dict in all_positions:
                            merged_positions.update(pos_dict)
                        
                        errors = []
                        for i in range(self.config.n_sensors):
                            if i in merged_positions:
                                error = np.linalg.norm(
                                    merged_positions[i] - self.true_positions[i]
                                )
                                errors.append(error ** 2)
                        rmse = np.sqrt(np.mean(errors))
                        rmse_history.append(rmse)
                    
                    # Log progress
                    if iteration % 50 == 0:
                        self.logger.info(f"Iteration {iteration}: "
                                       f"Objective={objective:.4f}, "
                                       f"Change={relative_change:.6f}")
                else:
                    # Non-root ranks participate in reductions
                    local_err, local_cnt = self.compute_local_objective(state)
                    self.comm.reduce(local_err, op=MPI.SUM, root=0)
                    self.comm.reduce(local_cnt, op=MPI.SUM, root=0)
                    
                    if self.true_positions is not None:
                        self.comm.gather(state.positions, root=0)
                
                # Check convergence
                if relative_change < self.config.tolerance:
                    state.converged = True
                    state.iteration = iteration
                    break
        
        if not state.converged:
            state.iteration = self.config.max_iterations
        
        # Gather final results on rank 0
        if self.rank == 0:
            # Gather all final positions
            all_positions = self.comm.gather(state.positions, root=0)
            final_positions = {}
            for pos_dict in all_positions:
                final_positions.update(pos_dict)
            
            # Compute final metrics
            state.positions = final_positions
            local_err, local_cnt = self.compute_local_objective(state)
            final_objective = np.sqrt(local_err / max(local_cnt, 1))
            
            final_rmse = None
            if self.true_positions is not None:
                errors = []
                for i in range(self.config.n_sensors):
                    error = np.linalg.norm(
                        final_positions[i] - self.true_positions[i]
                    )
                    errors.append(error ** 2)
                final_rmse = np.sqrt(np.mean(errors))
            
            self.logger.info(f"Algorithm {'converged' if state.converged else 'terminated'} "
                           f"after {state.iteration} iterations")
            self.logger.info(f"Final objective: {final_objective:.4f}")
            if final_rmse is not None:
                self.logger.info(f"Final RMSE: {final_rmse:.4f}")
            
            return {
                'converged': state.converged,
                'iterations': state.iteration,
                'final_objective': final_objective,
                'final_rmse': final_rmse,
                'objective_history': objective_history,
                'rmse_history': rmse_history if self.true_positions else [],
                'final_positions': final_positions,
                'n_processes': self.size
            }
        else:
            # Non-root ranks
            self.comm.gather(state.positions, root=0)
            return None