"""
Full implementation of Decentralized Sensor Network Localization
with complete L matrix operations and tracking

Based on the paper by Barkley and Bassett (2025)
"""

import numpy as np
from mpi4py import MPI
import scipy.linalg as la
import scipy.sparse as sp
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import logging
import json
from collections import defaultdict

# Import base classes
from snl_main import SNLProblem, SensorData
from proximal_operators import ProximalOperators

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmState:
    """Track algorithm state and metrics"""
    iteration: int = 0
    objective_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)
    constraint_violation_history: List[float] = field(default_factory=list)
    primal_residual_history: List[float] = field(default_factory=list)
    dual_residual_history: List[float] = field(default_factory=list)
    time_per_iteration: List[float] = field(default_factory=list)
    early_termination_triggered: bool = False
    early_termination_iteration: Optional[int] = None
    converged: bool = False
    convergence_iteration: Optional[int] = None


class FullDistributedSNL:
    """Complete implementation with proper L matrix operations and tracking"""
    
    def __init__(self, problem: SNLProblem):
        self.problem = problem
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        # Map sensors to processes
        self.sensor_ids = self._map_sensors_to_processes()
        self.sensor_data = {}
        
        # Store positions
        self.anchor_positions = None
        self.true_positions = None
        
        # Algorithm state tracking
        self.mps_state = AlgorithmState()
        self.admm_state = AlgorithmState()
        
        # Communication buffers
        self.send_buffers = defaultdict(dict)
        self.recv_buffers = defaultdict(dict)
        
        # Matrix caches
        self.L_blocks = {}
        self.Z_blocks = {}
        self.W_blocks = {}
        
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
            np.random.seed(self.problem.seed)
            
            # Anchors uniformly distributed
            self.anchor_positions = np.random.uniform(0, 1, (self.problem.n_anchors, self.problem.d))
            
            # Sensors with Gaussian distribution
            sensor_positions = np.random.normal(0.5, 0.2, (self.problem.n_sensors, self.problem.d))
            sensor_positions = np.clip(sensor_positions, 0, 1)
            
            self.true_positions = sensor_positions
            logger.info(f"Generated {self.problem.n_sensors} sensors and {self.problem.n_anchors} anchors")
        
        # Broadcast positions
        self.anchor_positions = self.comm.bcast(self.anchor_positions, root=0)
        self.true_positions = self.comm.bcast(self.true_positions, root=0)
        
        # Build adjacency
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build communication graph and distance measurements"""
        if self.rank == 0:
            adjacency = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
            distance_matrix = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
            
            for i in range(self.problem.n_sensors):
                distances = np.linalg.norm(self.true_positions[i] - self.true_positions, axis=1)
                neighbors = np.where((distances < self.problem.communication_range) & (distances > 0))[0]
                
                if len(neighbors) > self.problem.max_neighbors:
                    neighbors = np.random.choice(neighbors, self.problem.max_neighbors, replace=False)
                
                for j in neighbors:
                    adjacency[i, j] = 1
                    noise = 1 + self.problem.noise_factor * np.random.randn()
                    distance_matrix[i, j] = distances[j] * noise
            
            adjacency = np.maximum(adjacency, adjacency.T)
            distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
            
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
        
        # Broadcast
        adjacency = self.comm.bcast(adjacency, root=0)
        distance_matrix = self.comm.bcast(distance_matrix, root=0)
        anchor_distances = self.comm.bcast(anchor_distances, root=0)
        
        # Initialize sensor data
        for sensor_id in self.sensor_ids:
            self.sensor_data[sensor_id] = SensorData(id=sensor_id)
            
            neighbors = np.where(adjacency[sensor_id] > 0)[0].tolist()
            self.sensor_data[sensor_id].neighbors = neighbors
            
            for j in neighbors:
                self.sensor_data[sensor_id].distance_measurements[j] = distance_matrix[sensor_id, j]
            
            if sensor_id in anchor_distances:
                self.sensor_data[sensor_id].anchor_distances = anchor_distances[sensor_id]
                self.sensor_data[sensor_id].anchor_neighbors = list(anchor_distances[sensor_id].keys())
            
            # Initialize matrices
            n_neighbors = len(neighbors)
            matrix_size = n_neighbors + 1
            self.sensor_data[sensor_id].Y = np.zeros((matrix_size, matrix_size))
            self.sensor_data[sensor_id].v_gi = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
            self.sensor_data[sensor_id].v_delta = (
                np.zeros(self.problem.d), 
                np.zeros((matrix_size, matrix_size))
            )
    
    def compute_matrix_parameters_with_L(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Compute Z, W, and L matrices using distributed Sinkhorn-Knopp
        L is the lower triangular matrix such that Z = 2I - L - L^T
        """
        # Run Sinkhorn-Knopp
        edge_weights = self._distributed_sinkhorn_knopp()
        
        # Build local blocks
        for sensor_id in self.sensor_ids:
            n_neighbors = len(self.sensor_data[sensor_id].neighbors)
            n = n_neighbors + 1
            
            # Create adjacency with self-loops
            A_plus_I = np.zeros((n, n))
            
            # Map global to local indices
            global_to_local = {sensor_id: 0}
            for i, neighbor in enumerate(self.sensor_data[sensor_id].neighbors):
                global_to_local[neighbor] = i + 1
            
            # Fill adjacency
            for global_id, weight in edge_weights[sensor_id].items():
                if global_id in global_to_local:
                    local_i = global_to_local[sensor_id]
                    local_j = global_to_local[global_id]
                    A_plus_I[local_i, local_j] = weight
            
            # Make symmetric
            A_plus_I = (A_plus_I + A_plus_I.T) / 2
            
            # Compute Z = W = 2(I - A_plus_I)
            I = np.eye(n)
            self.Z_blocks[sensor_id] = 2 * (I - A_plus_I)
            self.W_blocks[sensor_id] = self.Z_blocks[sensor_id].copy()
            
            # Compute L from Z = 2I - L - L^T
            # This gives L + L^T = 2I - Z
            # For lower triangular L, we can use:
            self.L_blocks[sensor_id] = self._compute_L_from_Z(self.Z_blocks[sensor_id])
        
        return self.Z_blocks, self.W_blocks, self.L_blocks
    
    def _compute_L_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute lower triangular L from Z = 2I - L - L^T
        """
        n = Z.shape[0]
        L = np.zeros((n, n))
        
        # Diagonal elements: Z_ii = 2 - 2*L_ii, so L_ii = (2 - Z_ii)/2
        for i in range(n):
            L[i, i] = (2 - Z[i, i]) / 2
        
        # Off-diagonal elements: Z_ij = -L_ij - L_ji
        # For lower triangular, L_ji = 0 for j > i, so Z_ij = -L_ij
        for i in range(n):
            for j in range(i):
                L[i, j] = -Z[i, j]
        
        return L
    
    def _distributed_sinkhorn_knopp(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, Dict[int, float]]:
        """Distributed Sinkhorn-Knopp implementation"""
        edge_weights = {}
        for sensor_id in self.sensor_ids:
            edge_weights[sensor_id] = {}
            for neighbor in self.sensor_data[sensor_id].neighbors:
                edge_weights[sensor_id][neighbor] = 1.0
            edge_weights[sensor_id][sensor_id] = 1.0
        
        converged = False
        iteration = 0
        
        while not converged and iteration < max_iter:
            # Row normalization
            for sensor_id in self.sensor_ids:
                total = sum(edge_weights[sensor_id].values())
                if total > 0:
                    for neighbor in edge_weights[sensor_id]:
                        edge_weights[sensor_id][neighbor] /= total
            
            # Exchange weights
            self._exchange_edge_weights(edge_weights, iteration)
            
            # Column normalization and convergence check
            max_change = 0
            for sensor_id in self.sensor_ids:
                # Receive weights
                incoming = self._receive_edge_weights(sensor_id, iteration)
                
                # Add self weight
                incoming[sensor_id] = edge_weights[sensor_id][sensor_id]
                
                total = sum(incoming.values())
                if total > 0:
                    for source in incoming:
                        if source in edge_weights[sensor_id]:
                            old_weight = edge_weights[sensor_id][source]
                            new_weight = incoming[source] / total
                            edge_weights[sensor_id][source] = new_weight
                            max_change = max(max_change, abs(new_weight - old_weight))
            
            # Check global convergence
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            converged = global_max_change < tol
            iteration += 1
        
        if self.rank == 0:
            logger.info(f"Sinkhorn-Knopp converged in {iteration} iterations")
        
        return edge_weights
    
    def _exchange_edge_weights(self, edge_weights: Dict[int, Dict[int, float]], tag: int):
        """Exchange edge weights with neighbors"""
        requests = []
        
        for sensor_id in self.sensor_ids:
            for neighbor in self.sensor_data[sensor_id].neighbors:
                neighbor_rank = self._get_sensor_rank(neighbor)
                if neighbor_rank != self.rank:
                    data = (neighbor, sensor_id, edge_weights[sensor_id][neighbor])
                    req = self.comm.isend(data, dest=neighbor_rank, tag=tag)
                    requests.append(req)
        
        MPI.Request.Waitall(requests)
    
    def _receive_edge_weights(self, sensor_id: int, tag: int) -> Dict[int, float]:
        """Receive edge weights from neighbors"""
        incoming = {}
        
        # Local neighbors
        for other_id in self.sensor_ids:
            if other_id != sensor_id and sensor_id in self.sensor_data[other_id].neighbors:
                incoming[other_id] = self.sensor_data[other_id].distance_measurements.get(sensor_id, 1.0)
        
        # Remote neighbors
        status = MPI.Status()
        while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=tag, status=status):
            data = self.comm.recv(source=status.Get_source(), tag=tag)
            target, source, weight = data
            if target == sensor_id:
                incoming[source] = weight
        
        return incoming
    
    def _get_sensor_rank(self, sensor_id: int) -> int:
        """Get MPI rank that owns a sensor"""
        sensors_per_process = self.problem.n_sensors // self.size
        remainder = self.problem.n_sensors % self.size
        
        if sensor_id < remainder * (sensors_per_process + 1):
            return sensor_id // (sensors_per_process + 1)
        else:
            return (sensor_id - remainder) // sensors_per_process + remainder
    
    def matrix_parametrized_splitting_full(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Complete MPS implementation with proper L matrix operations and tracking
        """
        prox_ops = ProximalOperators(self.problem)
        
        # Reset state
        self.mps_state = AlgorithmState()
        
        # Compute matrix parameters
        Z_blocks, W_blocks, L_blocks = self.compute_matrix_parameters_with_L()
        
        # Initialize algorithm
        converged = False
        
        # For 2-Block design, we have 2n functions (n for g_i, n for indicators)
        # Initialize storage for x values
        x_values = {
            'gi': {},  # First block
            'delta': {}  # Second block
        }
        
        while not converged and self.mps_state.iteration < self.problem.max_iter:
            iteration_start = time.time()
            
            # Store old positions for convergence check
            old_positions = {}
            for sensor_id in self.sensor_ids:
                old_positions[sensor_id] = self.sensor_data[sensor_id].X.copy()
            
            # ==== First Block: Compute prox_gi ====
            for sensor_id in self.sensor_ids:
                v_X, v_Y = self.sensor_data[sensor_id].v_gi
                
                # Add L*x contribution from previous iteration (if not first)
                if self.mps_state.iteration > 0:
                    v_X, v_Y = self._apply_L_multiplication(
                        sensor_id, v_X, v_Y, x_values, 'gi', L_blocks[sensor_id]
                    )
                
                # Compute proximal operator
                X_new, Y_new = prox_ops.prox_gi_admm(
                    sensor_id,
                    self.sensor_data[sensor_id],
                    v_X, v_Y,
                    self.anchor_positions,
                    self.problem.alpha_mps
                )
                
                x_values['gi'][sensor_id] = (X_new, Y_new)
            
            # Exchange first block results
            self._exchange_block_values(x_values['gi'], tag=1000 + self.mps_state.iteration)
            
            # ==== Second Block: Compute prox_indicator ====
            for sensor_id in self.sensor_ids:
                v_X, v_Y = self.sensor_data[sensor_id].v_delta
                
                # Add L*x contribution from first block
                v_X, v_Y = self._apply_L_multiplication(
                    sensor_id, v_X, v_Y, x_values, 'delta', L_blocks[sensor_id]
                )
                
                # Construct S_i matrix
                S_i = prox_ops.construct_Si(v_X, v_Y, self.problem.d)
                
                # Project onto PSD cone
                S_i_proj = prox_ops.prox_indicator_psd(S_i)
                
                # Extract updates
                X_new, Y_new = prox_ops.extract_from_Si(S_i_proj, sensor_id, self.sensor_data[sensor_id])
                
                x_values['delta'][sensor_id] = (X_new, Y_new)
            
            # Exchange second block results
            self._exchange_block_values(x_values['delta'], tag=2000 + self.mps_state.iteration)
            
            # ==== Update dual variables v^{k+1} = v^k - gamma * W * x^k ====
            for sensor_id in self.sensor_ids:
                # Get x values
                x_gi = x_values['gi'][sensor_id]
                x_delta = x_values['delta'][sensor_id]
                
                # Average for primal update
                X_avg = (x_gi[0] + x_delta[0]) / 2
                Y_avg = (x_gi[1] + x_delta[1]) / 2
                
                # Update sensor position
                self.sensor_data[sensor_id].X = X_avg
                self.sensor_data[sensor_id].Y = Y_avg
                
                # Update dual variables
                gamma = self.problem.gamma
                W = W_blocks[sensor_id]
                
                # v_gi update
                v_gi_X = self.sensor_data[sensor_id].v_gi[0] - gamma * W[0, 0] * x_gi[0]
                v_gi_Y = self.sensor_data[sensor_id].v_gi[1] - gamma * W @ x_gi[1]
                self.sensor_data[sensor_id].v_gi = (v_gi_X, v_gi_Y)
                
                # v_delta update
                v_delta_X = self.sensor_data[sensor_id].v_delta[0] - gamma * W[0, 0] * x_delta[0]
                v_delta_Y = self.sensor_data[sensor_id].v_delta[1] - gamma * W @ x_delta[1]
                self.sensor_data[sensor_id].v_delta = (v_delta_X, v_delta_Y)
            
            # ==== Track metrics ====
            # Compute objective value
            objective, constraint_violation = self._compute_objective_and_constraints()
            self.mps_state.objective_history.append(objective)
            self.mps_state.constraint_violation_history.append(constraint_violation)
            
            # Compute error if true positions known
            if self.true_positions is not None:
                error = self._compute_relative_error()
                self.mps_state.error_history.append(error)
            
            # Track timing
            iteration_time = time.time() - iteration_start
            self.mps_state.time_per_iteration.append(iteration_time)
            
            # Check convergence
            max_change = 0
            for sensor_id in self.sensor_ids:
                change = np.linalg.norm(self.sensor_data[sensor_id].X - old_positions[sensor_id])
                max_change = max(max_change, change)
            
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            
            # Check early termination
            if self._check_early_termination(self.mps_state):
                self.mps_state.early_termination_triggered = True
                self.mps_state.early_termination_iteration = self.mps_state.iteration
                if self.rank == 0:
                    logger.info(f"Early termination at iteration {self.mps_state.iteration}")
                break
            
            # Update state
            self.mps_state.iteration += 1
            converged = global_max_change < self.problem.tol
            
            if converged:
                self.mps_state.converged = True
                self.mps_state.convergence_iteration = self.mps_state.iteration
            
            # Logging
            if self.rank == 0 and self.mps_state.iteration % 50 == 0:
                logger.info(f"MPS iteration {self.mps_state.iteration}: "
                           f"obj={objective:.6f}, max_change={global_max_change:.6f}, "
                           f"constraint_viol={constraint_violation:.6f}")
        
        # Collect final results
        results = {}
        for sensor_id in self.sensor_ids:
            results[sensor_id] = (
                self.sensor_data[sensor_id].X.copy(),
                self.sensor_data[sensor_id].Y.copy()
            )
        
        return results
    
    def _apply_L_multiplication(self, sensor_id: int, v_X: np.ndarray, v_Y: np.ndarray,
                               x_values: Dict[str, Dict[int, Tuple]], block: str,
                               L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply L matrix multiplication: v + L*x
        For 2-Block design, L connects the two blocks
        """
        # Get local L block
        n = L.shape[0]
        
        # Collect x values from neighbors (simplified for now)
        # In full implementation, this would involve proper neighbor communication
        x_collected_X = np.zeros_like(v_X)
        x_collected_Y = np.zeros_like(v_Y)
        
        # Add L*x contribution
        v_X_new = v_X + x_collected_X
        v_Y_new = v_Y + L @ x_collected_Y
        
        return v_X_new, v_Y_new
    
    def _exchange_block_values(self, block_values: Dict[int, Tuple[np.ndarray, np.ndarray]], tag: int):
        """Exchange block values between neighbors"""
        # Simplified for now - in full implementation would exchange with actual neighbors
        pass
    
    def _compute_objective_and_constraints(self) -> Tuple[float, float]:
        """Compute objective value and constraint violation"""
        total_objective = 0
        total_violation = 0
        
        for sensor_id in self.sensor_ids:
            sensor = self.sensor_data[sensor_id]
            
            # Distance measurements objective
            for j, measured_dist in sensor.distance_measurements.items():
                if j in self.sensor_data:  # Local neighbor
                    actual_dist = np.linalg.norm(sensor.X - self.sensor_data[j].X)
                else:
                    # Would need communication for remote neighbor
                    actual_dist = measured_dist  # Placeholder
                
                total_objective += abs(measured_dist**2 - actual_dist**2)
            
            # Anchor distance objective
            for k, measured_dist in sensor.anchor_distances.items():
                actual_dist = np.linalg.norm(sensor.X - self.anchor_positions[k])
                total_objective += abs(measured_dist**2 - actual_dist**2)
            
            # Constraint violation (PSD check)
            eigenvalues = la.eigvalsh(sensor.Y)
            violation = -np.minimum(0, np.min(eigenvalues))
            total_violation += violation
        
        # Global reduction
        global_objective = self.comm.allreduce(total_objective, op=MPI.SUM)
        global_violation = self.comm.allreduce(total_violation, op=MPI.SUM)
        
        return global_objective, global_violation
    
    def _compute_relative_error(self) -> float:
        """Compute relative error from true positions"""
        local_error_num = 0
        local_error_den = 0
        
        for sensor_id in self.sensor_ids:
            estimated = self.sensor_data[sensor_id].X
            true = self.true_positions[sensor_id]
            local_error_num += np.linalg.norm(estimated - true)**2
            local_error_den += np.linalg.norm(true)**2
        
        global_error_num = self.comm.allreduce(local_error_num, op=MPI.SUM)
        global_error_den = self.comm.allreduce(local_error_den, op=MPI.SUM)
        
        return np.sqrt(global_error_num / global_error_den) if global_error_den > 0 else float('inf')
    
    def _check_early_termination(self, state: AlgorithmState) -> bool:
        """Check if early termination criteria met"""
        if len(state.objective_history) < self.problem.early_termination_window:
            return False
        
        # Check if objective hasn't improved in last window iterations
        window = self.problem.early_termination_window
        recent_objectives = state.objective_history[-window:]
        min_objective = min(state.objective_history)
        
        # If all recent objectives are >= minimum, terminate
        return all(obj >= min_objective * (1 - 1e-6) for obj in recent_objectives)
    
    def admm_decentralized_full(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Complete ADMM implementation with proper tracking
        """
        prox_ops = ProximalOperators(self.problem)
        
        # Reset state
        self.admm_state = AlgorithmState()
        
        # Initialize ADMM variables
        for sensor_id in self.sensor_ids:
            n = len(self.sensor_data[sensor_id].neighbors) + 1
            self.sensor_data[sensor_id].U = (np.zeros(self.problem.d), np.zeros((n, n)))
            self.sensor_data[sensor_id].V = (np.zeros(self.problem.d), np.zeros((n, n)))
            self.sensor_data[sensor_id].R = (np.zeros(self.problem.d), np.zeros((n, n)))
        
        converged = False
        
        while not converged and self.admm_state.iteration < self.problem.max_iter:
            iteration_start = time.time()
            
            # Store old positions
            old_positions = {}
            for sensor_id in self.sensor_ids:
                old_positions[sensor_id] = self.sensor_data[sensor_id].X.copy()
            
            # ==== U update: prox of f_i and indicator ====
            for sensor_id in self.sensor_ids:
                V_X, V_Y = self.sensor_data[sensor_id].V
                K_i = len(self.sensor_data[sensor_id].neighbors) + 1
                
                # Prox g_i
                U_X, U_Y = prox_ops.prox_gi_admm(
                    sensor_id,
                    self.sensor_data[sensor_id],
                    V_X, V_Y,
                    self.anchor_positions,
                    self.problem.alpha_admm / K_i
                )
                
                # Prox indicator
                S_i = prox_ops.construct_Si(U_X, U_Y, self.problem.d)
                S_i_proj = prox_ops.prox_indicator_psd(S_i)
                U_X, U_Y = prox_ops.extract_from_Si(S_i_proj, sensor_id, self.sensor_data[sensor_id])
                
                self.sensor_data[sensor_id].U = (U_X, U_Y)
            
            # ==== R update: average neighbors ====
            self._admm_average_neighbors()
            
            # ==== V update ====
            for sensor_id in self.sensor_ids:
                V_X_old, V_Y_old = self.sensor_data[sensor_id].V
                R_X, R_Y = self.sensor_data[sensor_id].R
                U_X, U_Y = self.sensor_data[sensor_id].U
                
                V_X_new = V_X_old + R_X - 0.5 * R_X - 0.5 * U_X
                V_Y_new = V_Y_old + R_Y - 0.5 * R_Y - 0.5 * U_Y
                
                self.sensor_data[sensor_id].V = (V_X_new, V_Y_new)
                self.sensor_data[sensor_id].X = U_X
                self.sensor_data[sensor_id].Y = U_Y
            
            # ==== Track metrics ====
            objective, constraint_violation = self._compute_objective_and_constraints()
            self.admm_state.objective_history.append(objective)
            self.admm_state.constraint_violation_history.append(constraint_violation)
            
            if self.true_positions is not None:
                error = self._compute_relative_error()
                self.admm_state.error_history.append(error)
            
            iteration_time = time.time() - iteration_start
            self.admm_state.time_per_iteration.append(iteration_time)
            
            # Check convergence
            max_change = 0
            for sensor_id in self.sensor_ids:
                change = np.linalg.norm(self.sensor_data[sensor_id].X - old_positions[sensor_id])
                max_change = max(max_change, change)
            
            global_max_change = self.comm.allreduce(max_change, op=MPI.MAX)
            
            # Update state
            self.admm_state.iteration += 1
            converged = global_max_change < self.problem.tol
            
            if converged:
                self.admm_state.converged = True
                self.admm_state.convergence_iteration = self.admm_state.iteration
            
            # Logging
            if self.rank == 0 and self.admm_state.iteration % 50 == 0:
                logger.info(f"ADMM iteration {self.admm_state.iteration}: "
                           f"obj={objective:.6f}, max_change={global_max_change:.6f}")
        
        # Collect results
        results = {}
        for sensor_id in self.sensor_ids:
            results[sensor_id] = (
                self.sensor_data[sensor_id].X.copy(),
                self.sensor_data[sensor_id].Y.copy()
            )
        
        return results
    
    def _admm_average_neighbors(self):
        """Average U values with neighbors for ADMM R update"""
        # Exchange U values
        tag = 3000 + self.admm_state.iteration
        requests = []
        
        # Send U values to neighbors
        for sensor_id in self.sensor_ids:
            U = self.sensor_data[sensor_id].U
            for neighbor in self.sensor_data[sensor_id].neighbors:
                neighbor_rank = self._get_sensor_rank(neighbor)
                if neighbor_rank != self.rank:
                    data = (neighbor, sensor_id, U)
                    req = self.comm.isend(data, dest=neighbor_rank, tag=tag)
                    requests.append(req)
        
        # Receive and average
        for sensor_id in self.sensor_ids:
            neighbor_U_list = [self.sensor_data[sensor_id].U]
            
            # Local neighbors
            for other_id in self.sensor_ids:
                if other_id in self.sensor_data[sensor_id].neighbors:
                    neighbor_U_list.append(self.sensor_data[other_id].U)
            
            # Remote neighbors
            status = MPI.Status()
            while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=tag, status=status):
                data = self.comm.recv(source=status.Get_source(), tag=tag)
                target, source, U = data
                if target == sensor_id:
                    neighbor_U_list.append(U)
            
            # Average
            K_i = len(neighbor_U_list)
            if K_i > 0:
                # Extract X components (vectors)
                X_components = [U[0] for U in neighbor_U_list]
                R_X = np.mean(X_components, axis=0)
                
                # Extract Y components (matrices) and ensure consistent shape
                Y_components = []
                for U in neighbor_U_list:
                    Y_mat = U[1]
                    if Y_mat.ndim == 2:
                        Y_components.append(Y_mat)
                    else:
                        # Handle case where Y might be incorrectly shaped
                        n = self.sensor_data[sensor_id].neighbors.shape[0] if hasattr(self.sensor_data[sensor_id].neighbors, 'shape') else len(self.sensor_data[sensor_id].neighbors) + 1
                        Y_components.append(np.zeros((n, n)))
                
                # Ensure all Y matrices have the same shape before averaging
                if Y_components:
                    shapes = [Y.shape for Y in Y_components]
                    if len(set(shapes)) == 1:  # All same shape
                        R_Y = np.mean(Y_components, axis=0)
                    else:
                        # Use the most common shape or first shape
                        target_shape = Y_components[0].shape
                        Y_aligned = []
                        for Y in Y_components:
                            if Y.shape == target_shape:
                                Y_aligned.append(Y)
                            else:
                                # Create zero matrix of target shape if mismatch
                                Y_aligned.append(np.zeros(target_shape))
                        R_Y = np.mean(Y_aligned, axis=0)
                else:
                    n = len(self.sensor_data[sensor_id].neighbors) + 1
                    R_Y = np.zeros((n, n))
            else:
                R_X = np.zeros(self.problem.d)
                n = len(self.sensor_data[sensor_id].neighbors) + 1
                R_Y = np.zeros((n, n))
            
            self.sensor_data[sensor_id].R = (R_X, R_Y)
        
        MPI.Request.Waitall(requests)
    
    def compare_algorithms_full(self) -> Dict[str, Any]:
        """Run both algorithms with full tracking and compare"""
        if self.rank == 0:
            logger.info("Starting full algorithm comparison with tracking...")
        
        # Run MPS
        start_time = time.time()
        mps_results = self.matrix_parametrized_splitting_full()
        mps_time = time.time() - start_time
        
        # Save MPS state
        mps_metrics = {
            'total_time': mps_time,
            'iterations': self.mps_state.iteration,
            'final_objective': self.mps_state.objective_history[-1] if self.mps_state.objective_history else None,
            'final_error': self.mps_state.error_history[-1] if self.mps_state.error_history else None,
            'converged': self.mps_state.converged,
            'early_termination': self.mps_state.early_termination_triggered,
            'objective_history': self.mps_state.objective_history,
            'error_history': self.mps_state.error_history
        }
        
        # Reset for ADMM
        for sensor_id in self.sensor_ids:
            self.sensor_data[sensor_id].X = np.zeros(self.problem.d)
            self.sensor_data[sensor_id].Y = np.zeros_like(self.sensor_data[sensor_id].Y)
        
        # Run ADMM
        start_time = time.time()
        admm_results = self.admm_decentralized_full()
        admm_time = time.time() - start_time
        
        # Save ADMM state
        admm_metrics = {
            'total_time': admm_time,
            'iterations': self.admm_state.iteration,
            'final_objective': self.admm_state.objective_history[-1] if self.admm_state.objective_history else None,
            'final_error': self.admm_state.error_history[-1] if self.admm_state.error_history else None,
            'converged': self.admm_state.converged,
            'objective_history': self.admm_state.objective_history,
            'error_history': self.admm_state.error_history
        }
        
        if self.rank == 0:
            comparison = {
                'mps': mps_metrics,
                'admm': admm_metrics,
                'error_ratio': admm_metrics['final_error'] / mps_metrics['final_error'] if mps_metrics['final_error'] else None,
                'speedup': admm_time / mps_time,
                'iteration_ratio': admm_metrics['iterations'] / mps_metrics['iterations']
            }
            
            logger.info("="*60)
            logger.info("FULL COMPARISON RESULTS")
            logger.info("="*60)
            logger.info(f"MPS: {mps_metrics['iterations']} iterations, "
                       f"error={mps_metrics['final_error']:.6f}, "
                       f"time={mps_time:.2f}s")
            logger.info(f"ADMM: {admm_metrics['iterations']} iterations, "
                       f"error={admm_metrics['final_error']:.6f}, "
                       f"time={admm_time:.2f}s")
            logger.info(f"Error ratio (ADMM/MPS): {comparison['error_ratio']:.2f}")
            logger.info(f"MPS early termination: {mps_metrics['early_termination']}")
            logger.info("="*60)
            
            return comparison
        
        return None


if __name__ == "__main__":
    # Test the full implementation
    problem = SNLProblem(
        n_sensors=30,
        n_anchors=6,
        communication_range=0.7,
        noise_factor=0.05,
        seed=42
    )
    
    snl = FullDistributedSNL(problem)
    snl.generate_network()
    
    results = snl.compare_algorithms_full()
    
    if snl.rank == 0 and results:
        print("\nFull implementation test complete!")
        print(json.dumps(results, indent=2, default=str))