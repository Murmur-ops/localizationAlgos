"""
Fixed Distributed MPS implementation using MPI
Corrects consensus and synchronization bugs from original implementation
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Set, Optional, Tuple
import logging

from .algorithm import MPSConfig, MPSState
from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations


class DistributedMPSFixed:
    """
    Fixed distributed Matrix-Parametrized Proximal Splitting using MPI
    Properly synchronizes state across all processes
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
        
        # Global network data (same on all processes)
        self.anchor_positions = None
        self.adjacency = None
        self.Z_matrix = None
        self.all_measurements = {}
        self.all_anchor_distances = {}
        
        # For evaluation
        self.true_positions = None
        
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
        """Generate network data and broadcast to all processes"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        if self.rank == 0:
            # Generate complete network on rank 0
            np.random.seed(self.config.seed)
            
            # True positions (with scale)
            scale = self.config.scale if hasattr(self.config, 'scale') else 50.0
            true_positions = {}
            for i in range(n):
                true_positions[i] = np.random.uniform(0, scale, d)
            
            # Anchors (with scale)
            if self.config.n_anchors > 0 and d == 2:
                # Place anchors at corners for 2D (scaled)
                anchor_positions = np.array([
                    [0.1*scale, 0.1*scale], [0.9*scale, 0.1*scale], 
                    [0.9*scale, 0.9*scale], [0.1*scale, 0.9*scale]
                ])
                for i in range(4, self.config.n_anchors):
                    anchor_positions = np.vstack([
                        anchor_positions,
                        np.random.uniform(0.2*scale, 0.8*scale, d)
                    ])
                anchor_positions = anchor_positions[:self.config.n_anchors]
            else:
                anchor_positions = np.random.uniform(0, scale, (self.config.n_anchors, d))
            
            # Adjacency matrix (with scaled communication range)
            comm_range = self.config.communication_range * scale
            adjacency = MatrixOperations.build_adjacency(
                true_positions,
                comm_range
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
                    if true_dist <= comm_range:
                        noise = self.config.noise_factor * np.random.randn()
                        noisy_dist = true_dist * (1 + noise)
                        all_anchor_distances[i][k] = max(0.01, noisy_dist)
        else:
            true_positions = None
            anchor_positions = None
            adjacency = None
            all_measurements = None
            all_anchor_distances = None
        
        # Broadcast ALL data to ensure consistency
        self.true_positions = self.comm.bcast(true_positions, root=0)
        self.anchor_positions = self.comm.bcast(anchor_positions, root=0)
        self.adjacency = self.comm.bcast(adjacency, root=0)
        self.all_measurements = self.comm.bcast(all_measurements, root=0)
        self.all_anchor_distances = self.comm.bcast(all_anchor_distances, root=0)
        
        # Extract local data
        for sensor in self.local_sensors:
            # Local measurements
            for (i, j), dist in self.all_measurements.items():
                if i == sensor or j == sensor:
                    self.local_measurements[(i, j)] = dist
            
            # Local anchor distances
            if sensor in self.all_anchor_distances:
                self.local_anchor_distances[sensor] = self.all_anchor_distances[sensor]
        
        # Create consensus matrix (same on all processes)
        self.Z_matrix = MatrixOperations.create_consensus_matrix(
            self.adjacency,
            self.config.gamma
        )
    
    def initialize_state_distributed(self) -> MPSState:
        """Initialize local state variables"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Initialize ALL positions (will be synchronized)
        all_positions = {}
        
        # Each process initializes its local sensors
        for sensor in self.local_sensors:
            if sensor in self.local_anchor_distances and len(self.local_anchor_distances[sensor]) > 0:
                # Initialize near anchors
                anchor_ids = list(self.local_anchor_distances[sensor].keys())
                all_positions[sensor] = np.mean(
                    self.anchor_positions[anchor_ids], axis=0
                )
                all_positions[sensor] += 0.05 * np.random.randn(d)
            else:
                scale = self.config.scale if hasattr(self.config, 'scale') else 50.0
                all_positions[sensor] = np.random.uniform(0, scale, d)
        
        # Gather all initial positions
        all_init_positions = self.comm.allgather(all_positions)
        
        # Merge all positions
        merged_positions = {}
        for pos_dict in all_init_positions:
            merged_positions.update(pos_dict)
        
        # Initialize full state vectors with all positions
        X = np.zeros((2*n, d))
        Y = np.zeros((2*n, d))
        U = np.zeros((2*n, d))
        
        for sensor in range(n):
            if sensor in merged_positions:
                X[sensor] = merged_positions[sensor]
                X[sensor + n] = merged_positions[sensor]
                Y[sensor] = merged_positions[sensor]
                Y[sensor + n] = merged_positions[sensor]
        
        return MPSState(
            positions=merged_positions.copy(),
            X=X,
            Y=Y,
            U=U
        )
    
    def synchronize_positions(self, state: MPSState):
        """Synchronize all positions across all processes"""
        n = self.config.n_sensors
        
        # Each process prepares its local sensor positions
        local_positions = {}
        for sensor in self.local_sensors:
            local_positions[sensor] = state.X[sensor].copy()
        
        # Gather all positions from all processes
        all_positions = self.comm.allgather(local_positions)
        
        # Merge and update state.X with latest positions
        for pos_dict in all_positions:
            for sensor, position in pos_dict.items():
                state.X[sensor] = position
                state.X[sensor + n] = position
                state.positions[sensor] = position
    
    def local_prox_step(self, state: MPSState) -> np.ndarray:
        """Apply proximal operator locally for owned sensors"""
        n = self.config.n_sensors
        X_new = state.X.copy()
        
        for sensor in self.local_sensors:
            position = X_new[sensor]
            
            # Apply distance constraints with ALL sensors (we have full state)
            for j in range(n):
                if (sensor, j) in self.all_measurements:
                    measured_dist = self.all_measurements[(sensor, j)]
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
            
            # Box constraints (scaled)
            scale = self.config.scale if hasattr(self.config, 'scale') else 50.0
            X_new[sensor] = ProximalOperators.prox_box_constraint(
                X_new[sensor], -0.1*scale, 1.1*scale
            )
            X_new[sensor + n] = X_new[sensor]
        
        return X_new
    
    def distributed_consensus_fixed(self, state: MPSState):
        """Fixed distributed consensus step"""
        n = self.config.n_sensors
        
        # Each process computes consensus for ALL sensors using full Z matrix
        # This ensures consistency
        Y_local = self.Z_matrix @ state.X
        
        # Since all processes have the same X and Z, Y_local should be identical
        # We can verify this with a small tolerance check
        state.Y = Y_local
    
    def compute_local_objective(self, state: MPSState) -> Tuple[float, int]:
        """Compute local contribution to objective"""
        local_error = 0.0
        local_count = 0
        
        # Only count errors for local sensors to avoid double-counting
        for sensor in self.local_sensors:
            # Sensor-to-sensor errors (count each pair once)
            for j in range(self.config.n_sensors):
                if (sensor, j) in self.all_measurements and sensor < j:
                    measured_dist = self.all_measurements[(sensor, j)]
                    actual_dist = np.linalg.norm(state.positions[sensor] - state.positions[j])
                    local_error += (actual_dist - measured_dist) ** 2
                    local_count += 1
            
            # Anchor errors
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
        Run fixed distributed MPS algorithm
        
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
            
            # Step 1: Ensure all processes have consistent state
            self.synchronize_positions(state)
            
            # Step 2: Local proximal step (each process updates its sensors)
            X_local = self.local_prox_step(state)
            
            # Step 3: Gather all local updates
            local_updates = {}
            for sensor in self.local_sensors:
                local_updates[sensor] = X_local[sensor]
            
            all_updates = self.comm.allgather(local_updates)
            
            # Merge updates into state.X
            for updates_dict in all_updates:
                for sensor, position in updates_dict.items():
                    state.X[sensor] = position
                    state.X[sensor + self.config.n_sensors] = position
            
            # Step 4: Consensus step (all processes compute the same result)
            self.distributed_consensus_fixed(state)
            
            # Step 5: Dual update
            state.U = state.U + self.config.alpha * (state.X - state.Y)
            
            # Step 6: Extract position estimates
            n = self.config.n_sensors
            for i in range(n):
                state.positions[i] = (state.Y[i] + state.Y[i + n]) / 2
            
            # Check convergence and compute metrics periodically
            if iteration % 10 == 0:
                # Compute change norm
                change = np.linalg.norm(state.X - X_old) / (np.linalg.norm(X_old) + 1e-10)
                
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
                        errors = []
                        for i in range(self.config.n_sensors):
                            error = np.linalg.norm(
                                state.positions[i] - self.true_positions[i]
                            )
                            errors.append(error ** 2)
                        rmse = np.sqrt(np.mean(errors))
                        rmse_history.append(rmse)
                    
                    # Log progress
                    if iteration % 50 == 0:
                        self.logger.info(f"Iteration {iteration}: "
                                       f"Objective={objective:.4f}, "
                                       f"Change={change:.6f}")
                else:
                    # Non-root ranks participate in reductions
                    local_err, local_cnt = self.compute_local_objective(state)
                    self.comm.reduce(local_err, op=MPI.SUM, root=0)
                    self.comm.reduce(local_cnt, op=MPI.SUM, root=0)
                
                # Check convergence (all processes)
                if change < self.config.tolerance:
                    state.converged = True
                    state.iteration = iteration
                    break
        
        if not state.converged:
            state.iteration = self.config.max_iterations
        
        # Return results on rank 0
        if self.rank == 0:
            # Compute final metrics
            local_err, local_cnt = self.compute_local_objective(state)
            total_error = self.comm.reduce(local_err, op=MPI.SUM, root=0)
            total_count = self.comm.reduce(local_cnt, op=MPI.SUM, root=0)
            final_objective = np.sqrt(total_error / max(total_count, 1))
            
            final_rmse = None
            if self.true_positions is not None:
                errors = []
                for i in range(self.config.n_sensors):
                    error = np.linalg.norm(
                        state.positions[i] - self.true_positions[i]
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
                'final_positions': dict(state.positions),
                'n_processes': self.size
            }
        else:
            # Non-root ranks
            local_err, local_cnt = self.compute_local_objective(state)
            self.comm.reduce(local_err, op=MPI.SUM, root=0)
            self.comm.reduce(local_cnt, op=MPI.SUM, root=0)
            return None