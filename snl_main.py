"""
Main implementation of Decentralized Sensor Network Localization
using Matrix-Parametrized Proximal Splittings

Based on the paper by Barkley and Bassett (2025)
"""

import numpy as np
from mpi4py import MPI
import scipy.linalg as la
import scipy.sparse as sp
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SNLProblem:
    """Configuration parameters for the SNL problem"""
    n_sensors: int = 30
    n_anchors: int = 6
    d: int = 2  # dimension (2D or 3D)
    communication_range: float = 0.7
    max_neighbors: int = 7
    noise_factor: float = 0.05
    gamma: float = 0.999  # step size for MPS
    alpha_mps: float = 10.0  # scaling parameter for MPS
    alpha_admm: float = 150.0  # scaling parameter for ADMM
    max_iter: int = 500
    early_termination_window: int = 100
    tol: float = 1e-6
    seed: Optional[int] = None


@dataclass
class SensorData:
    """Data structure for each sensor"""
    id: int
    neighbors: List[int] = field(default_factory=list)
    anchor_neighbors: List[int] = field(default_factory=list)
    distance_measurements: Dict[int, float] = field(default_factory=dict)
    anchor_distances: Dict[int, float] = field(default_factory=dict)
    
    # Position and Gram matrix
    X: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    
    # Dual variables for MPS
    v_gi: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    v_delta: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    
    # ADMM variables
    U: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    V: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    R: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))


class DistributedSNL:
    """Main class for distributed sensor network localization"""
    
    def __init__(self, problem: SNLProblem):
        self.problem = problem
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        # Ensure we have enough processes
        if self.size < self.problem.n_sensors:
            if self.rank == 0:
                logger.warning(f"Running with {self.size} processes for {self.problem.n_sensors} sensors")
        
        # Map sensors to processes
        self.sensor_ids = self._map_sensors_to_processes()
        self.sensor_data = {}
        
        # Store anchor positions (same for all processes)
        self.anchor_positions = None
        
        # Store true positions for evaluation (only on rank 0)
        self.true_positions = None
        
        # Set random seed
        if problem.seed is not None:
            np.random.seed(problem.seed + self.rank)
    
    def _map_sensors_to_processes(self) -> List[int]:
        """Map sensors to MPI processes"""
        sensors_per_process = self.problem.n_sensors // self.size
        remainder = self.problem.n_sensors % self.size
        
        if self.rank < remainder:
            start = self.rank * (sensors_per_process + 1)
            end = start + sensors_per_process + 1
        else:
            start = self.rank * sensors_per_process + remainder
            end = start + sensors_per_process
        
        return list(range(start, end))
    
    def generate_network(self):
        """Generate random sensor and anchor positions"""
        if self.rank == 0:
            # Generate positions
            np.random.seed(self.problem.seed)
            
            # Anchors uniformly distributed
            self.anchor_positions = np.random.uniform(0, 1, (self.problem.n_anchors, self.problem.d))
            
            # Sensors with Gaussian distribution around center
            sensor_positions = np.random.normal(0.5, 0.2, (self.problem.n_sensors, self.problem.d))
            sensor_positions = np.clip(sensor_positions, 0, 1)
            
            self.true_positions = sensor_positions
            logger.info(f"Generated {self.problem.n_sensors} sensors and {self.problem.n_anchors} anchors")
        
        # Broadcast anchor positions to all processes
        self.anchor_positions = self.comm.bcast(self.anchor_positions, root=0)
        
        # Build adjacency and distance measurements
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build communication graph and add noisy distance measurements"""
        # First, rank 0 builds the full adjacency matrix
        if self.rank == 0:
            adjacency = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
            
            for i in range(self.problem.n_sensors):
                # Find sensors within communication range
                distances = np.linalg.norm(self.true_positions[i] - self.true_positions, axis=1)
                neighbors = np.where((distances < self.problem.communication_range) & (distances > 0))[0]
                
                # Limit to max_neighbors
                if len(neighbors) > self.problem.max_neighbors:
                    neighbors = np.random.choice(neighbors, self.problem.max_neighbors, replace=False)
                
                adjacency[i, neighbors] = 1
            
            # Make symmetric
            adjacency = np.maximum(adjacency, adjacency.T)
            
            # Generate noisy distances
            distance_matrix = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
            for i in range(self.problem.n_sensors):
                for j in range(i+1, self.problem.n_sensors):
                    if adjacency[i, j] > 0:
                        true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                        noise = 1 + self.problem.noise_factor * np.random.randn()
                        distance_matrix[i, j] = distance_matrix[j, i] = true_dist * noise
            
            # Anchor distances
            anchor_distances = {}
            for i in range(self.problem.n_sensors):
                anchor_distances[i] = {}
                distances_to_anchors = np.linalg.norm(
                    self.true_positions[i] - self.anchor_positions, axis=1
                )
                close_anchors = np.where(distances_to_anchors < self.problem.communication_range)[0]
                
                for k in close_anchors:
                    noise = 1 + self.problem.noise_factor * np.random.randn()
                    anchor_distances[i][k] = distances_to_anchors[k] * noise
        else:
            adjacency = None
            distance_matrix = None
            anchor_distances = None
        
        # Broadcast adjacency and distances
        adjacency = self.comm.bcast(adjacency, root=0)
        distance_matrix = self.comm.bcast(distance_matrix, root=0)
        anchor_distances = self.comm.bcast(anchor_distances, root=0)
        
        # Initialize sensor data for local sensors
        for sensor_id in self.sensor_ids:
            self.sensor_data[sensor_id] = SensorData(id=sensor_id)
            
            # Set neighbors
            neighbors = np.where(adjacency[sensor_id] > 0)[0].tolist()
            self.sensor_data[sensor_id].neighbors = neighbors
            
            # Set distance measurements
            for j in neighbors:
                self.sensor_data[sensor_id].distance_measurements[j] = distance_matrix[sensor_id, j]
            
            # Set anchor neighbors and distances
            if sensor_id in anchor_distances:
                self.sensor_data[sensor_id].anchor_distances = anchor_distances[sensor_id]
                self.sensor_data[sensor_id].anchor_neighbors = list(anchor_distances[sensor_id].keys())
            
            # Initialize matrices based on neighborhood size
            n_neighbors = len(neighbors)
            matrix_size = 1 + n_neighbors
            self.sensor_data[sensor_id].Y = np.zeros((matrix_size, matrix_size))
            self.sensor_data[sensor_id].v_gi = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
            self.sensor_data[sensor_id].v_delta = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
            self.sensor_data[sensor_id].U = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
            self.sensor_data[sensor_id].V = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
            self.sensor_data[sensor_id].R = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
    
    def distributed_sinkhorn_knopp(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, Dict[int, float]]:
        """
        Implement distributed Sinkhorn-Knopp algorithm
        Returns edge weights for each sensor
        """
        # Initialize edge weights
        edge_weights = {}
        for sensor_id in self.sensor_ids:
            edge_weights[sensor_id] = {}
            for neighbor in self.sensor_data[sensor_id].neighbors:
                edge_weights[sensor_id][neighbor] = 1.0
            # Add self-loop
            edge_weights[sensor_id][sensor_id] = 1.0
        
        converged = False
        iteration = 0
        
        while not converged and iteration < max_iter:
            # Row scaling (normalize outgoing weights)
            for sensor_id in self.sensor_ids:
                total = sum(edge_weights[sensor_id].values())
                if total > 0:
                    for neighbor in edge_weights[sensor_id]:
                        edge_weights[sensor_id][neighbor] /= total
            
            # Exchange weights with neighbors
            incoming_weights = {}
            for sensor_id in self.sensor_ids:
                incoming_weights[sensor_id] = {}
            
            # Send weights to neighbors
            requests = []
            for sensor_id in self.sensor_ids:
                for neighbor in self.sensor_data[sensor_id].neighbors:
                    # Find which process owns this neighbor
                    neighbor_rank = self._get_sensor_rank(neighbor)
                    if neighbor_rank != self.rank:
                        data = (neighbor, sensor_id, edge_weights[sensor_id][neighbor])
                        req = self.comm.isend(data, dest=neighbor_rank, tag=iteration)
                        requests.append(req)
                    else:
                        # Local communication
                        incoming_weights[neighbor][sensor_id] = edge_weights[sensor_id][neighbor]
            
            # Receive weights from neighbors
            status = MPI.Status()
            while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=iteration, status=status):
                data = self.comm.recv(source=status.Get_source(), tag=iteration)
                target, source, weight = data
                if target in self.sensor_ids:
                    incoming_weights[target][source] = weight
            
            # Wait for all sends to complete
            MPI.Request.Waitall(requests)
            
            # Column scaling (normalize incoming weights)
            max_change = 0
            for sensor_id in self.sensor_ids:
                # Add self-loop to incoming
                incoming_weights[sensor_id][sensor_id] = edge_weights[sensor_id][sensor_id]
                
                total = sum(incoming_weights[sensor_id].values())
                if total > 0:
                    for source in incoming_weights[sensor_id]:
                        old_weight = edge_weights[sensor_id].get(source, 0)
                        new_weight = incoming_weights[sensor_id][source] / total
                        
                        # Update edge weight
                        if source in edge_weights[sensor_id]:
                            edge_weights[sensor_id][source] = new_weight
                        
                        # Track convergence
                        max_change = max(max_change, abs(new_weight - old_weight))
            
            # Check global convergence
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            converged = global_max_change < tol
            iteration += 1
            
            if self.rank == 0 and iteration % 10 == 0:
                logger.debug(f"Sinkhorn-Knopp iteration {iteration}, max change: {global_max_change:.6f}")
        
        if self.rank == 0:
            logger.info(f"Sinkhorn-Knopp converged in {iteration} iterations")
        
        return edge_weights
    
    def _get_sensor_rank(self, sensor_id: int) -> int:
        """Get MPI rank that owns a given sensor"""
        sensors_per_process = self.problem.n_sensors // self.size
        remainder = self.problem.n_sensors % self.size
        
        if sensor_id < remainder * (sensors_per_process + 1):
            return sensor_id // (sensors_per_process + 1)
        else:
            return (sensor_id - remainder) // sensors_per_process + remainder
    
    def compute_matrix_parameters(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Build Z and W matrices using Sinkhorn-Knopp algorithm
        Returns local blocks of Z and W for each sensor
        """
        # Run distributed Sinkhorn-Knopp
        edge_weights = self.distributed_sinkhorn_knopp()
        
        # Build local blocks of Z and W matrices
        # For 2-Block design: Z = W = 2(I - SK(A+I))
        Z_blocks = {}
        W_blocks = {}
        
        for sensor_id in self.sensor_ids:
            n_neighbors = len(self.sensor_data[sensor_id].neighbors)
            
            # Create adjacency matrix with self-loops
            A_plus_I = np.zeros((n_neighbors + 1, n_neighbors + 1))
            
            # Map global indices to local indices
            global_to_local = {sensor_id: 0}
            for i, neighbor in enumerate(self.sensor_data[sensor_id].neighbors):
                global_to_local[neighbor] = i + 1
            
            # Fill adjacency matrix
            for global_id, weight in edge_weights[sensor_id].items():
                if global_id in global_to_local:
                    local_i = global_to_local[sensor_id]
                    local_j = global_to_local[global_id]
                    A_plus_I[local_i, local_j] = weight
            
            # Make symmetric (should already be from Sinkhorn-Knopp)
            A_plus_I = (A_plus_I + A_plus_I.T) / 2
            
            # Compute Z = W = 2(I - A_plus_I)
            I = np.eye(n_neighbors + 1)
            Z_blocks[sensor_id] = 2 * (I - A_plus_I)
            W_blocks[sensor_id] = Z_blocks[sensor_id].copy()
        
        return Z_blocks, W_blocks
    
    def matrix_parametrized_splitting(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Main MPS algorithm (Algorithm 1 from paper)
        Returns estimated positions and Gram matrices
        """
        # Import proximal operators
        from proximal_operators import ProximalOperators
        prox_ops = ProximalOperators(self.problem)
        
        # Compute matrix parameters
        Z_blocks, W_blocks = self.compute_matrix_parameters()
        
        # Initialize variables
        converged = False
        iteration = 0
        objective_history = []
        min_objective = float('inf')
        iterations_since_min = 0
        
        while not converged and iteration < self.problem.max_iter:
            # Store old values for convergence check
            old_X = {}
            for sensor_id in self.sensor_ids:
                old_X[sensor_id] = self.sensor_data[sensor_id].X.copy()
            
            # Compute x^k = J_alpha_F(v^k + Lx^k)
            # This involves solving the proximal operators in order
            
            # First block: prox of g_i functions
            x_gi = {}
            for sensor_id in self.sensor_ids:
                # Get current v values
                v_X, v_Y = self.sensor_data[sensor_id].v_gi
                
                # Compute prox_gi
                X_new, Y_new = prox_ops.prox_gi_admm(
                    sensor_id, 
                    self.sensor_data[sensor_id],
                    v_X, v_Y,
                    self.anchor_positions,
                    self.problem.alpha_mps
                )
                x_gi[sensor_id] = (X_new, Y_new)
            
            # Exchange x_gi values with neighbors (for L matrix multiplication)
            # ... (communication code here)
            
            # Second block: prox of indicator functions
            x_delta = {}
            for sensor_id in self.sensor_ids:
                # Get current v values and add L*x contribution
                v_X, v_Y = self.sensor_data[sensor_id].v_delta
                
                # Add contributions from first block (simplified for now)
                v_Y_adjusted = v_Y.copy()
                
                # Compute prox_indicator
                S_i = prox_ops.construct_Si(
                    self.sensor_data[sensor_id].X,
                    v_Y_adjusted,
                    self.problem.d
                )
                S_i_proj = prox_ops.prox_indicator_psd(S_i)
                X_new, Y_new = prox_ops.extract_from_Si(S_i_proj, sensor_id, self.sensor_data[sensor_id])
                
                x_delta[sensor_id] = (X_new, Y_new)
            
            # Update v^{k+1} = v^k - gamma * W * x^k
            for sensor_id in self.sensor_ids:
                # Update sensor data
                self.sensor_data[sensor_id].X = (x_gi[sensor_id][0] + x_delta[sensor_id][0]) / 2
                self.sensor_data[sensor_id].Y = (x_gi[sensor_id][1] + x_delta[sensor_id][1]) / 2
                
                # Update dual variables (simplified)
                gamma = self.problem.gamma
                self.sensor_data[sensor_id].v_gi = (
                    self.sensor_data[sensor_id].v_gi[0] - gamma * W_blocks[sensor_id][0, 0] * x_gi[sensor_id][0],
                    self.sensor_data[sensor_id].v_gi[1] - gamma * W_blocks[sensor_id] @ x_gi[sensor_id][1]
                )
                self.sensor_data[sensor_id].v_delta = (
                    self.sensor_data[sensor_id].v_delta[0] - gamma * W_blocks[sensor_id][0, 0] * x_delta[sensor_id][0],
                    self.sensor_data[sensor_id].v_delta[1] - gamma * W_blocks[sensor_id] @ x_delta[sensor_id][1]
                )
            
            # Compute objective value
            objective = self._compute_objective()
            objective_history.append(objective)
            
            # Early termination check
            if objective < min_objective:
                min_objective = objective
                iterations_since_min = 0
            else:
                iterations_since_min += 1
            
            if iterations_since_min >= self.problem.early_termination_window:
                if self.rank == 0:
                    logger.info(f"Early termination at iteration {iteration}")
                break
            
            # Convergence check
            max_change = 0
            for sensor_id in self.sensor_ids:
                change = np.linalg.norm(self.sensor_data[sensor_id].X - old_X[sensor_id])
                max_change = max(max_change, change)
            
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            converged = global_max_change < self.problem.tol
            
            iteration += 1
            
            if self.rank == 0 and iteration % 50 == 0:
                logger.info(f"MPS iteration {iteration}, objective: {objective:.6f}, max change: {global_max_change:.6f}")
        
        # Collect results
        results = {}
        for sensor_id in self.sensor_ids:
            results[sensor_id] = (
                self.sensor_data[sensor_id].X.copy(),
                self.sensor_data[sensor_id].Y.copy()
            )
        
        return results
    
    def admm_decentralized(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Implement decentralized ADMM for comparison
        Based on equations (32-34) from the paper
        """
        from proximal_operators import ProximalOperators
        prox_ops = ProximalOperators(self.problem)
        
        converged = False
        iteration = 0
        
        while not converged and iteration < self.problem.max_iter:
            # Store old values
            old_X = {}
            for sensor_id in self.sensor_ids:
                old_X[sensor_id] = self.sensor_data[sensor_id].X.copy()
            
            # U update: prox of f_i
            for sensor_id in self.sensor_ids:
                V_X, V_Y = self.sensor_data[sensor_id].V
                
                # Compute scaling based on neighborhood size
                K_i = len(self.sensor_data[sensor_id].neighbors) + 1  # +1 for self
                
                # Prox for g_i
                U_X, U_Y = prox_ops.prox_gi_admm(
                    sensor_id,
                    self.sensor_data[sensor_id],
                    V_X, V_Y,
                    self.anchor_positions,
                    self.problem.alpha_admm / K_i
                )
                
                # Prox for indicator (run in sequence)
                S_i = prox_ops.construct_Si(U_X, U_Y, self.problem.d)
                S_i_proj = prox_ops.prox_indicator_psd(S_i)
                U_X, U_Y = prox_ops.extract_from_Si(S_i_proj, sensor_id, self.sensor_data[sensor_id])
                
                self.sensor_data[sensor_id].U = (U_X, U_Y)
            
            # R update: average of neighbors' U values
            for sensor_id in self.sensor_ids:
                # Collect U values from neighbors
                neighbor_U_X = [self.sensor_data[sensor_id].U[0]]
                neighbor_U_Y = [self.sensor_data[sensor_id].U[1]]
                
                # Exchange U values with neighbors
                # ... (communication code here)
                
                # Average
                K_i = len(neighbor_U_X)
                R_X = np.mean(neighbor_U_X, axis=0)
                R_Y = np.mean(neighbor_U_Y, axis=0)
                
                self.sensor_data[sensor_id].R = (R_X, R_Y)
            
            # V update
            for sensor_id in self.sensor_ids:
                V_X_old, V_Y_old = self.sensor_data[sensor_id].V
                R_X, R_Y = self.sensor_data[sensor_id].R
                U_X, U_Y = self.sensor_data[sensor_id].U
                
                V_X_new = V_X_old + R_X - 0.5 * R_X - 0.5 * U_X
                V_Y_new = V_Y_old + R_Y - 0.5 * R_Y - 0.5 * U_Y
                
                self.sensor_data[sensor_id].V = (V_X_new, V_Y_new)
                
                # Update position estimate
                self.sensor_data[sensor_id].X = U_X
                self.sensor_data[sensor_id].Y = U_Y
            
            # Convergence check
            max_change = 0
            for sensor_id in self.sensor_ids:
                change = np.linalg.norm(self.sensor_data[sensor_id].X - old_X[sensor_id])
                max_change = max(max_change, change)
            
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            converged = global_max_change < self.problem.tol
            
            iteration += 1
            
            if self.rank == 0 and iteration % 50 == 0:
                logger.info(f"ADMM iteration {iteration}, max change: {global_max_change:.6f}")
        
        # Collect results
        results = {}
        for sensor_id in self.sensor_ids:
            results[sensor_id] = (
                self.sensor_data[sensor_id].X.copy(),
                self.sensor_data[sensor_id].Y.copy()
            )
        
        return results
    
    def _compute_objective(self) -> float:
        """Compute objective value for current solution"""
        total_obj = 0
        
        for sensor_id in self.sensor_ids:
            sensor = self.sensor_data[sensor_id]
            
            # Distance to neighbors
            for j, dist in sensor.distance_measurements.items():
                if j in self.sensor_data:  # Local neighbor
                    diff = dist**2 - np.linalg.norm(sensor.X - self.sensor_data[j].X)**2
                else:
                    # Need to get position from other process
                    diff = 0  # Simplified for now
                total_obj += abs(diff)
            
            # Distance to anchors
            for k, dist in sensor.anchor_distances.items():
                diff = dist**2 - np.linalg.norm(sensor.X - self.anchor_positions[k])**2
                total_obj += abs(diff)
        
        # Global sum
        global_obj = self.comm.allreduce(total_obj, op=MPI.SUM)
        return global_obj
    
    def compute_error(self, results: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute relative error from true positions"""
        # Gather all results to rank 0
        all_positions = self.comm.gather(results, root=0)
        
        if self.rank == 0:
            # Combine results
            estimated_positions = np.zeros((self.problem.n_sensors, self.problem.d))
            
            for proc_results in all_positions:
                if proc_results is not None:
                    for sensor_id, (X, Y) in proc_results.items():
                        estimated_positions[sensor_id] = X
            
            # Compute relative error
            error = np.linalg.norm(estimated_positions - self.true_positions, 'fro')
            error /= np.linalg.norm(self.true_positions, 'fro')
            
            return error
        else:
            return None
    
    def compare_algorithms(self) -> Dict[str, any]:
        """Run both algorithms and compare results"""
        if self.rank == 0:
            logger.info("Starting algorithm comparison...")
        
        # Run MPS
        start_time = time.time()
        mps_results = self.matrix_parametrized_splitting()
        mps_time = time.time() - start_time
        mps_error = self.compute_error(mps_results)
        
        # Reset variables for ADMM
        for sensor_id in self.sensor_ids:
            self.sensor_data[sensor_id].X = np.zeros(self.problem.d)
            self.sensor_data[sensor_id].Y = np.zeros_like(self.sensor_data[sensor_id].Y)
            self.sensor_data[sensor_id].U = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
            self.sensor_data[sensor_id].V = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
            self.sensor_data[sensor_id].R = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
        
        # Run ADMM
        start_time = time.time()
        admm_results = self.admm_decentralized()
        admm_time = time.time() - start_time
        admm_error = self.compute_error(admm_results)
        
        if self.rank == 0:
            results = {
                'mps_error': mps_error,
                'mps_time': mps_time,
                'admm_error': admm_error,
                'admm_time': admm_time,
                'error_ratio': admm_error / mps_error if mps_error > 0 else float('inf')
            }
            
            logger.info(f"MPS Error: {mps_error:.6f}, Time: {mps_time:.2f}s")
            logger.info(f"ADMM Error: {admm_error:.6f}, Time: {admm_time:.2f}s")
            logger.info(f"Error ratio (ADMM/MPS): {results['error_ratio']:.2f}")
            
            return results
        else:
            return None


if __name__ == "__main__":
    # Example usage
    problem = SNLProblem(
        n_sensors=30,
        n_anchors=6,
        communication_range=0.7,
        noise_factor=0.05,
        seed=42
    )
    
    snl = DistributedSNL(problem)
    snl.generate_network()
    
    results = snl.compare_algorithms()
    
    if snl.rank == 0 and results is not None:
        print("\nFinal Results:")
        print(json.dumps(results, indent=2))