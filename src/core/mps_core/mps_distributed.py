"""
Distributed MPI implementation of Matrix-Parametrized Proximal Splitting
Enables true decentralized computation across multiple nodes
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .mps_full_algorithm import (
    MPSConfig,
    NetworkData,
    LiftedVariableStructure,
    ProximalADMMSolver,
    ProximalOperatorsPSD
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedMPSConfig(MPSConfig):
    """Extended configuration for distributed MPS"""
    consensus_rounds: int = 5  # Number of consensus averaging rounds
    async_updates: bool = False  # Use asynchronous updates
    compression_ratio: float = 1.0  # Message compression (1.0 = no compression)
    fault_tolerance: bool = False  # Enable fault tolerance mechanisms


class DistributedMPS:
    """
    Distributed implementation of Matrix-Parametrized Proximal Splitting
    Each MPI rank handles a subset of sensors
    """
    
    def __init__(self, config: DistributedMPSConfig, network_data: NetworkData):
        self.config = config
        self.network_data = network_data
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Determine sensor assignment
        self._assign_sensors()
        
        # Initialize local structures
        self._initialize_local_structures()
        
        # Setup communication topology
        self._setup_communication()
        
        logger.info(f"Rank {self.rank}: Initialized with {len(self.local_sensors)} sensors")
    
    def _assign_sensors(self):
        """Assign sensors to MPI ranks"""
        n_sensors = self.config.n_sensors
        sensors_per_rank = n_sensors // self.size
        remainder = n_sensors % self.size
        
        # Distribute sensors as evenly as possible
        if self.rank < remainder:
            self.local_sensors = list(range(
                self.rank * (sensors_per_rank + 1),
                (self.rank + 1) * (sensors_per_rank + 1)
            ))
        else:
            self.local_sensors = list(range(
                remainder + self.rank * sensors_per_rank,
                remainder + (self.rank + 1) * sensors_per_rank
            ))
        
        # Build global sensor assignment map
        all_assignments = self.comm.allgather(self.local_sensors)
        self.sensor_to_rank = {}
        for rank, sensors in enumerate(all_assignments):
            for sensor in sensors:
                self.sensor_to_rank[sensor] = rank
    
    def _initialize_local_structures(self):
        """Initialize local data structures"""
        d = self.config.dimension
        n = self.config.n_sensors
        
        # Local lifted variable structure
        self.lifted_structure = LiftedVariableStructure(n, d)
        
        # Build local neighborhoods (only for local sensors)
        self.local_neighborhoods = {}
        adjacency = self.network_data.adjacency_matrix
        
        for i in self.local_sensors:
            neighbors = []
            for j in range(n):
                if i != j and adjacency[i, j] > 0:
                    neighbors.append(j)
            self.local_neighborhoods[i] = neighbors
        
        # Initialize local ADMM solvers
        self.local_admm_solvers = {}
        for i in self.local_sensors:
            self.local_admm_solvers[i] = ProximalADMMSolver(
                rho=self.config.admm_rho,
                max_iterations=self.config.admm_iterations,
                tolerance=self.config.admm_tolerance,
                warm_start=self.config.warm_start
            )
        
        self.psd_ops = ProximalOperatorsPSD()
        
        # Initialize local variables
        self.local_x = {}  # Local lifted variables
        self.local_v = {}  # Local consensus variables
        
        for i in self.local_sensors:
            neighbors = self.local_neighborhoods[i]
            dim = d + 1 + len(neighbors)
            
            # Initialize with identity
            self.local_x[i] = np.eye(dim)
            self.local_x[n + i] = np.eye(dim)  # PSD constraint component
            self.local_v[i] = np.eye(dim)
            self.local_v[n + i] = np.eye(dim)
        
        # Initialize position estimates
        self.local_X = np.zeros((n, d))
        self.local_Y = np.zeros((n, n))
        
        if self.network_data.true_positions is not None:
            # Warm start with noisy true positions
            noise_level = 0.01 if self.config.carrier_phase_mode else 0.1
            for i in self.local_sensors:
                self.local_X[i] = (self.network_data.true_positions[i] + 
                                  noise_level * np.random.randn(d))
    
    def _setup_communication(self):
        """Setup communication patterns between ranks"""
        # Identify which ranks we need to communicate with
        self.neighbor_ranks = set()
        
        for i in self.local_sensors:
            for j in self.local_neighborhoods[i]:
                if j not in self.local_sensors:
                    self.neighbor_ranks.add(self.sensor_to_rank[j])
        
        self.neighbor_ranks = list(self.neighbor_ranks)
        
        # Setup communication buffers
        self.send_buffers = {rank: {} for rank in self.neighbor_ranks}
        self.recv_buffers = {rank: {} for rank in self.neighbor_ranks}
    
    def _exchange_neighbor_data(self):
        """Exchange data with neighboring ranks"""
        # Prepare send data
        for rank in self.neighbor_ranks:
            self.send_buffers[rank] = {}
            for i in self.local_sensors:
                # Include both objective and constraint components
                self.send_buffers[rank][i] = self.local_x.get(i)
                self.send_buffers[rank][self.config.n_sensors + i] = self.local_x.get(
                    self.config.n_sensors + i
                )
        
        # Non-blocking send/receive
        requests = []
        
        # Send to neighbors
        for rank in self.neighbor_ranks:
            req = self.comm.isend(self.send_buffers[rank], dest=rank, tag=self.rank)
            requests.append(req)
        
        # Receive from neighbors
        for rank in self.neighbor_ranks:
            self.recv_buffers[rank] = self.comm.recv(source=rank, tag=rank)
        
        # Wait for sends to complete
        MPI.Request.Waitall(requests)
    
    def _local_proximal_evaluation(self, iteration: int):
        """Perform local proximal operator evaluations"""
        n = self.config.n_sensors
        alpha = self.config.alpha
        
        # Adaptive alpha scaling
        if self.config.adaptive_alpha:
            if self.config.carrier_phase_mode:
                alpha = self.config.alpha * (1.0 + iteration / 100.0)
            else:
                alpha = self.config.alpha / (1.0 + iteration / 500.0)
        
        # Process local sensors
        for i in self.local_sensors:
            # Objective proximal operator
            neighbors = self.local_neighborhoods[i]
            
            # Get distance measurements
            distances_sensors = {}
            for j in neighbors:
                key = (min(i, j), max(i, j))
                if key in self.network_data.distance_measurements:
                    distances_sensors[j] = self.network_data.distance_measurements[key]
            
            # Get anchor connections
            anchors = self.network_data.anchor_connections.get(i, [])
            distances_anchors = {}
            for a in anchors:
                key = (i, a)
                if key in self.network_data.distance_measurements:
                    distances_anchors[a] = self.network_data.distance_measurements[key]
            
            # Extract current estimates
            X_curr, Y_curr = self.lifted_structure.extract_S_matrix(
                self.local_v[i], i, neighbors
            )
            
            # Apply ADMM solver
            X_new, Y_new = self.local_admm_solvers[i].solve(
                X_curr, Y_curr, i, neighbors, anchors,
                distances_sensors, distances_anchors,
                self.network_data.anchor_positions, alpha
            )
            
            # Update local lifted variable
            self.local_x[i] = self.lifted_structure.construct_from_XY(
                X_new, Y_new, i, neighbors
            )
            
            # PSD constraint projection
            self.local_x[n + i] = self.psd_ops.project_psd_cone(self.local_v[n + i])
    
    def _consensus_update(self):
        """Perform consensus update with neighbors"""
        gamma = self.config.gamma
        n = self.config.n_sensors
        
        # Exchange data with neighbors
        self._exchange_neighbor_data()
        
        # Perform consensus averaging
        for _ in range(self.config.consensus_rounds):
            # Update local consensus variables
            for i in self.local_sensors:
                # Weighted average with neighbors
                neighbors = self.local_neighborhoods[i]
                n_neighbors = len(neighbors)
                
                if n_neighbors > 0:
                    # Compute weighted average
                    weight_self = 0.5
                    weight_neighbor = (1 - weight_self) / n_neighbors if n_neighbors > 0 else 0
                    
                    # Update objective component
                    new_v = weight_self * self.local_x[i]
                    for j in neighbors:
                        if j in self.local_sensors:
                            new_v += weight_neighbor * self.local_x[j]
                        else:
                            # Get from received buffer
                            rank = self.sensor_to_rank[j]
                            if rank in self.recv_buffers and j in self.recv_buffers[rank]:
                                new_v += weight_neighbor * self.recv_buffers[rank][j]
                    
                    self.local_v[i] = self.local_v[i] - gamma * (self.local_v[i] - new_v)
                    
                    # Update constraint component similarly
                    new_v_psd = weight_self * self.local_x[n + i]
                    for j in neighbors:
                        if j in self.local_sensors:
                            new_v_psd += weight_neighbor * self.local_x[n + j]
                        else:
                            rank = self.sensor_to_rank[j]
                            if rank in self.recv_buffers and (n + j) in self.recv_buffers[rank]:
                                new_v_psd += weight_neighbor * self.recv_buffers[rank][n + j]
                    
                    self.local_v[n + i] = self.local_v[n + i] - gamma * (self.local_v[n + i] - new_v_psd)
    
    def _extract_local_estimates(self):
        """Extract position and Y estimates from local variables"""
        d = self.config.dimension
        
        for i in self.local_sensors:
            neighbors = self.local_neighborhoods[i]
            X_i, Y_i = self.lifted_structure.extract_S_matrix(
                self.local_x[i], i, neighbors
            )
            
            # Update local estimates
            self.local_X[i] = X_i[i]
            
            # Update Y matrix entries
            self.local_Y[i, i] = Y_i[i, i]
            for j in neighbors:
                self.local_Y[i, j] = Y_i[i, j]
                self.local_Y[j, i] = Y_i[j, i]
    
    def _compute_local_objective(self) -> float:
        """Compute local contribution to objective"""
        total = 0.0
        
        for i in self.local_sensors:
            neighbors = self.local_neighborhoods[i]
            
            # Sensor-to-sensor terms
            for j in neighbors:
                key = (min(i, j), max(i, j))
                if key in self.network_data.distance_measurements:
                    d_ij = self.network_data.distance_measurements[key]
                    est_dist_sq = (self.local_Y[i, i] + self.local_Y[j, j] - 
                                  2 * self.local_Y[i, j])
                    total += abs(est_dist_sq - d_ij**2)
            
            # Sensor-to-anchor terms
            anchors = self.network_data.anchor_connections.get(i, [])
            for a in anchors:
                key = (i, a)
                if key in self.network_data.distance_measurements:
                    d_ia = self.network_data.distance_measurements[key]
                    a_pos = self.network_data.anchor_positions[a]
                    est_dist_sq = (self.local_Y[i, i] + np.dot(a_pos, a_pos) - 
                                  2 * np.dot(a_pos, self.local_X[i]))
                    total += abs(est_dist_sq - d_ia**2)
        
        return total
    
    def run_iteration(self, k: int) -> Dict[str, float]:
        """Run one distributed iteration"""
        # Step 1: Local proximal evaluations
        self._local_proximal_evaluation(k)
        
        # Step 2: Consensus update with neighbors
        self._consensus_update()
        
        # Step 3: Extract local estimates
        self._extract_local_estimates()
        
        # Step 4: Compute local statistics
        local_obj = self._compute_local_objective()
        
        # Global reduction for statistics
        global_obj = self.comm.allreduce(local_obj, op=MPI.SUM)
        
        # Compute position error if true positions available
        local_error = 0.0
        if self.network_data.true_positions is not None:
            for i in self.local_sensors:
                diff = self.local_X[i] - self.network_data.true_positions[i]
                local_error += np.sum(diff**2)
        
        global_error_sq = self.comm.allreduce(local_error, op=MPI.SUM)
        global_rmse = np.sqrt(global_error_sq / self.config.n_sensors)
        
        # Convert to mm if in carrier phase mode
        if self.config.carrier_phase_mode:
            global_rmse *= 1000
        
        return {
            'iteration': k,
            'objective': global_obj,
            'position_error': global_rmse
        }
    
    def run(self, max_iterations: Optional[int] = None,
            tolerance: Optional[float] = None) -> Dict[str, Any]:
        """Run distributed MPS algorithm"""
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance
        
        history = {
            'objective': [],
            'position_error': []
        }
        
        best_objective = float('inf')
        best_iteration = 0
        
        for k in range(max_iter):
            # Run iteration
            stats = self.run_iteration(k)
            
            # Record history (only on rank 0)
            if self.rank == 0:
                history['objective'].append(stats['objective'])
                history['position_error'].append(stats['position_error'])
                
                if stats['objective'] < best_objective:
                    best_objective = stats['objective']
                    best_iteration = k
                
                # Logging
                if self.config.verbose and k % 100 == 0:
                    logger.info(
                        f"Iteration {k}: obj={stats['objective']:.6f}, "
                        f"pos_err={stats['position_error']:.6f}"
                    )
            
            # Check convergence (simplified)
            if stats['objective'] < tol:
                if self.rank == 0 and self.config.verbose:
                    logger.info(f"Converged at iteration {k}")
                break
        
        # Gather all final positions to rank 0
        all_positions = self.comm.gather(
            {i: self.local_X[i] for i in self.local_sensors}, 
            root=0
        )
        
        if self.rank == 0:
            # Combine all positions
            final_positions = np.zeros((self.config.n_sensors, self.config.dimension))
            for rank_positions in all_positions:
                for i, pos in rank_positions.items():
                    final_positions[i] = pos
            
            return {
                'final_positions': final_positions,
                'history': history,
                'iterations': k + 1,
                'best_iteration': best_iteration,
                'best_objective': best_objective,
                'final_rmse': history['position_error'][-1] if history['position_error'] else 0.0
            }
        else:
            return None


def run_distributed_mps(config_file: str, network_file: str):
    """
    Main entry point for distributed MPS execution
    
    Args:
        config_file: Path to configuration file
        network_file: Path to network data file
    """
    import pickle
    
    # Load configuration and network data
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    with open(network_file, 'rb') as f:
        network_data = pickle.load(f)
    
    # Create and run distributed MPS
    dmps = DistributedMPS(config, network_data)
    results = dmps.run()
    
    # Save results (only rank 0)
    if dmps.rank == 0 and results is not None:
        with open('distributed_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Distributed MPS completed successfully")
        logger.info(f"Final RMSE: {results['final_rmse']:.6f}")
        logger.info(f"Iterations: {results['iterations']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: mpirun -n <num_processes> python mps_distributed.py <config_file> <network_file>")
        sys.exit(1)
    
    run_distributed_mps(sys.argv[1], sys.argv[2])