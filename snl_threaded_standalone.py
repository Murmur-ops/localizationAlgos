"""
Standalone threaded implementation of Decentralized Sensor Network Localization
No MPI dependencies - pure threading implementation
"""

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import json
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


@dataclass
class SNLProblem:
    """Problem parameters for sensor network localization"""
    n_sensors: int = 30
    n_anchors: int = 6
    d: int = 2  # dimension
    communication_range: float = 0.7
    max_neighbors: int = 7
    noise_factor: float = 0.05
    gamma: float = 0.999
    alpha_mps: float = 10.0
    alpha_admm: float = 150.0
    max_iter: int = 1000
    tol: float = 1e-4
    early_termination_window: int = 100
    seed: int = None


@dataclass
class SensorData:
    """Data structure for a single sensor"""
    id: int
    neighbors: List[int] = field(default_factory=list)
    distance_measurements: Dict[int, float] = field(default_factory=dict)
    anchor_neighbors: List[int] = field(default_factory=list)
    anchor_distances: Dict[int, float] = field(default_factory=dict)
    X: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    v_gi: tuple = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    v_delta: tuple = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    U: tuple = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    V: tuple = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))
    R: tuple = field(default_factory=lambda: (np.zeros(2), np.zeros((1, 1))))


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


class ProximalOperators:
    """Proximal operator implementations for SNL"""
    
    def __init__(self, problem: SNLProblem):
        self.problem = problem
    
    def prox_gi_admm(self, sensor_id: int, sensor_data: SensorData, 
                     X_k: np.ndarray, Y_k: np.ndarray,
                     anchor_positions: np.ndarray, alpha: float, 
                     max_iter: int = 50, rho: float = 1.0, tol: float = 1e-6):
        """ADMM-based proximal operator for g_i"""
        d = self.problem.d
        n_i = len(sensor_data.neighbors) + 1
        
        # Initialize
        X = X_k.copy()
        Y = Y_k.copy()
        Z_X = X.copy()
        Z_Y = Y.copy()
        U_X = np.zeros_like(X)
        U_Y = np.zeros_like(Y)
        
        for iteration in range(max_iter):
            X_old = X.copy()
            Y_old = Y.copy()
            
            # X update
            X = self._solve_X_subproblem(sensor_id, sensor_data, Z_X - U_X, 
                                       anchor_positions, alpha, rho)
            
            # Y update
            Y = self._solve_Y_subproblem(sensor_id, sensor_data, Z_Y - U_Y, 
                                       anchor_positions, alpha, rho, X)
            
            # Z update (projection)
            Z_X = X + U_X
            Z_Y = Y + U_Y
            
            # U update
            U_X = U_X + X - Z_X
            U_Y = U_Y + Y - Z_Y
            
            # Check convergence
            primal_res = np.linalg.norm(X - X_old) + np.linalg.norm(Y - Y_old)
            if primal_res < tol:
                break
        
        return X, Y
    
    def _solve_X_subproblem(self, sensor_id: int, sensor_data: SensorData,
                           V_X: np.ndarray, anchor_positions: np.ndarray,
                           alpha: float, rho: float):
        """Solve X subproblem in ADMM"""
        d = self.problem.d
        
        # Build matrices
        D_i = self.construct_Di(sensor_id, sensor_data)
        c_i = self.construct_ci(sensor_id, sensor_data, anchor_positions)
        
        # Solve (D_i + rho*I)X = c_i + rho*V_X
        A = alpha * D_i + rho * np.eye(d)
        b = alpha * c_i + rho * V_X
        
        try:
            X = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            X = V_X
        
        return X
    
    def _solve_Y_subproblem(self, sensor_id: int, sensor_data: SensorData,
                           V_Y: np.ndarray, anchor_positions: np.ndarray,
                           alpha: float, rho: float, X: np.ndarray):
        """Solve Y subproblem in ADMM"""
        n_i = len(sensor_data.neighbors) + 1
        
        # Build matrices
        K_i = self.construct_Ki(sensor_id, sensor_data)
        N_i = self.construct_Ni(X)
        M_i = self.construct_Mi(sensor_id, sensor_data, anchor_positions, X)
        
        # Solve (K_i + rho*I)vec(Y) = vec(N_i - M_i + rho*V_Y)
        A = alpha * K_i + rho * np.eye(n_i * n_i)
        b = alpha * (N_i - M_i).flatten() + rho * V_Y.flatten()
        
        try:
            Y_vec = np.linalg.solve(A, b)
            Y = Y_vec.reshape(n_i, n_i)
        except np.linalg.LinAlgError:
            Y = V_Y
        
        return Y
    
    def construct_Di(self, sensor_id: int, sensor_data: SensorData):
        """Construct D_i matrix"""
        d = self.problem.d
        n_neighbors = len(sensor_data.neighbors)
        n_anchors = len(sensor_data.anchor_neighbors)
        
        D_i = (n_neighbors + n_anchors) * np.eye(d)
        return D_i
    
    def construct_Ki(self, sensor_id: int, sensor_data: SensorData):
        """Construct K_i matrix"""
        n_i = len(sensor_data.neighbors) + 1
        n_neighbors = len(sensor_data.neighbors)
        n_anchors = len(sensor_data.anchor_neighbors)
        
        # K_i = I ⊗ D_i
        D_i_scalar = n_neighbors + n_anchors
        K_i = D_i_scalar * np.eye(n_i * n_i)
        
        return K_i
    
    def construct_ci(self, sensor_id: int, sensor_data: SensorData, 
                    anchor_positions: np.ndarray):
        """Construct c_i vector"""
        d = self.problem.d
        c_i = np.zeros(d)
        
        # Anchor contributions
        for k in sensor_data.anchor_neighbors:
            c_i += anchor_positions[k]
        
        return c_i
    
    def construct_Ni(self, X: np.ndarray):
        """Construct N_i matrix"""
        d = len(X)
        n_i = 1  # Simplified for single sensor block
        
        N_i = np.zeros((n_i, n_i))
        N_i[0, 0] = np.dot(X, X)
        
        return N_i
    
    def construct_Mi(self, sensor_id: int, sensor_data: SensorData,
                    anchor_positions: np.ndarray, X: np.ndarray):
        """Construct M_i matrix"""
        n_i = len(sensor_data.neighbors) + 1
        M_i = np.zeros((n_i, n_i))
        
        # Neighbor distance measurements
        for idx, j in enumerate(sensor_data.neighbors):
            if j in sensor_data.distance_measurements:
                M_i[0, idx+1] = sensor_data.distance_measurements[j]**2
                M_i[idx+1, 0] = sensor_data.distance_measurements[j]**2
        
        # Anchor distance measurements  
        for k in sensor_data.anchor_neighbors:
            if k in sensor_data.anchor_distances:
                dist_sq = sensor_data.anchor_distances[k]**2
                anchor_pos = anchor_positions[k]
                correction = np.dot(anchor_pos, anchor_pos) - 2*np.dot(anchor_pos, X)
                M_i[0, 0] += dist_sq - correction
        
        return M_i
    
    def prox_indicator_psd(self, S: np.ndarray):
        """Project onto PSD cone"""
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def construct_Si(self, X: np.ndarray, Y: np.ndarray, d: int):
        """Construct Si matrix for PSD projection"""
        n = Y.shape[0]
        Si = np.zeros((d + n, d + n))
        
        Si[:d, :d] = np.eye(d)
        Si[:d, d:] = X.reshape(d, 1) @ np.ones((1, n))
        Si[d:, :d] = np.ones((n, 1)) @ X.reshape(1, d)
        Si[d:, d:] = Y
        
        return Si
    
    def extract_from_Si(self, Si: np.ndarray, sensor_id: int, sensor_data: SensorData):
        """Extract X and Y from Si matrix"""
        d = self.problem.d
        n = len(sensor_data.neighbors) + 1
        
        X = Si[:d, d:].mean(axis=1)
        Y = Si[d:, d:]
        
        return X, Y


class ThreadedCommunicator:
    """Manages thread-safe communication between sensors"""
    
    def __init__(self, n_sensors: int):
        self.n_sensors = n_sensors
        self.queues = {i: queue.Queue() for i in range(n_sensors)}
        self.barriers = {}
        self.lock = threading.Lock()
    
    def send(self, from_sensor: int, to_sensor: int, data: Any, tag: int):
        """Send data from one sensor to another"""
        message = {
            'from': from_sensor,
            'to': to_sensor,
            'data': data,
            'tag': tag
        }
        self.queues[to_sensor].put(message)
    
    def receive(self, sensor_id: int, tag: int, timeout: float = 1.0) -> List[Dict]:
        """Receive all messages for a sensor with given tag"""
        messages = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                    
                msg = self.queues[sensor_id].get(timeout=min(remaining, 0.01))
                if msg['tag'] == tag:
                    messages.append(msg)
                else:
                    # Put back if wrong tag
                    self.queues[sensor_id].put(msg)
            except queue.Empty:
                break
        
        return messages
    
    def barrier(self, name: str, n_participants: int):
        """Synchronization barrier"""
        with self.lock:
            if name not in self.barriers:
                self.barriers[name] = threading.Barrier(n_participants)
        
        self.barriers[name].wait()
    
    def broadcast(self, from_sensor: int, data: Any, tag: int, recipients: List[int]):
        """Broadcast data to multiple sensors"""
        for to_sensor in recipients:
            if to_sensor != from_sensor:
                self.send(from_sensor, to_sensor, data, tag)


class ThreadedSensor:
    """Sensor that runs in a thread with full algorithm implementation"""
    
    def __init__(self, sensor_id: int, sensor_data: SensorData, 
                 problem: SNLProblem, anchor_positions: np.ndarray,
                 communicator: ThreadedCommunicator,
                 true_position: Optional[np.ndarray] = None):
        self.sensor_id = sensor_id
        self.sensor_data = sensor_data
        self.problem = problem
        self.anchor_positions = anchor_positions
        self.communicator = communicator
        self.true_position = true_position
        
        self.prox_ops = ProximalOperators(problem)
        
        self.Z_block = None
        self.W_block = None
        self.L_block = None
        
        self.x_gi = None
        self.x_delta = None
        
        self.local_objective = 0
        self.local_constraint_violation = 0
    
    def compute_local_L_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Compute lower triangular L from Z = 2I - L - L^T"""
        n = Z.shape[0]
        L = np.zeros((n, n))
        
        for i in range(n):
            L[i, i] = (2 - Z[i, i]) / 2
        
        for i in range(n):
            for j in range(i):
                L[i, j] = -Z[i, j]
        
        return L
    
    def run_sinkhorn_knopp_phase(self, edge_weights: Dict[int, float], 
                                phase: str, iteration: int) -> Dict[int, float]:
        """Run one phase of Sinkhorn-Knopp"""
        if phase == 'row':
            total = sum(edge_weights.values())
            if total > 0:
                normalized = {k: v/total for k, v in edge_weights.items()}
            else:
                normalized = edge_weights.copy()
            
            tag = 1000 + iteration
            self.communicator.broadcast(
                self.sensor_id, 
                normalized[self.sensor_id],
                tag,
                self.sensor_data.neighbors
            )
            
            return normalized
            
        else:  # column normalization
            tag = 1000 + iteration
            messages = self.communicator.receive(self.sensor_id, tag, timeout=0.5)
            
            incoming = {self.sensor_id: edge_weights[self.sensor_id]}
            for msg in messages:
                incoming[msg['from']] = msg['data']
            
            total = sum(incoming.values())
            if total > 0:
                new_weights = {}
                for neighbor_id in edge_weights:
                    if neighbor_id in incoming:
                        new_weights[neighbor_id] = incoming[neighbor_id] / total
                    else:
                        new_weights[neighbor_id] = edge_weights[neighbor_id]
                return new_weights
            else:
                return edge_weights
    
    def run_mps_iteration(self, v_gi: Tuple[np.ndarray, np.ndarray],
                         v_delta: Tuple[np.ndarray, np.ndarray],
                         iteration: int) -> Dict[str, Any]:
        """Run one iteration of MPS algorithm"""
        # First block: compute prox_gi
        X_gi, Y_gi = self.prox_ops.prox_gi_admm(
            self.sensor_id,
            self.sensor_data,
            v_gi[0], v_gi[1],
            self.anchor_positions,
            self.problem.alpha_mps
        )
        self.x_gi = (X_gi, Y_gi)
        
        # Exchange first block results
        tag = 2000 + iteration
        self.communicator.broadcast(
            self.sensor_id,
            self.x_gi,
            tag,
            self.sensor_data.neighbors
        )
        
        # Synchronize
        self.communicator.barrier(f"mps_block1_{iteration}", self.problem.n_sensors)
        
        # Second block: compute prox_indicator
        v_delta_adjusted = v_delta  # Simplified
        
        S_i = self.prox_ops.construct_Si(v_delta_adjusted[0], v_delta_adjusted[1], self.problem.d)
        S_i_proj = self.prox_ops.prox_indicator_psd(S_i)
        X_delta, Y_delta = self.prox_ops.extract_from_Si(S_i_proj, self.sensor_id, self.sensor_data)
        self.x_delta = (X_delta, Y_delta)
        
        # Update position
        X_new = (X_gi + X_delta) / 2
        Y_new = (Y_gi + Y_delta) / 2
        
        # Update dual variables
        gamma = self.problem.gamma
        v_gi_new = (
            v_gi[0] - gamma * self.W_block[0, 0] * X_gi,
            v_gi[1] - gamma * self.W_block @ Y_gi
        )
        v_delta_new = (
            v_delta[0] - gamma * self.W_block[0, 0] * X_delta,
            v_delta[1] - gamma * self.W_block @ Y_delta
        )
        
        # Compute local metrics
        self._compute_local_metrics(X_new, Y_new)
        
        return {
            'X': X_new,
            'Y': Y_new,
            'v_gi': v_gi_new,
            'v_delta': v_delta_new,
            'objective': self.local_objective,
            'constraint_violation': self.local_constraint_violation
        }
    
    def run_admm_iteration(self, U: Tuple[np.ndarray, np.ndarray],
                          V: Tuple[np.ndarray, np.ndarray],
                          R: Tuple[np.ndarray, np.ndarray],
                          iteration: int) -> Dict[str, Any]:
        """Run one iteration of ADMM algorithm"""
        # U update
        V_X, V_Y = V
        K_i = len(self.sensor_data.neighbors) + 1
        
        # Prox g_i
        U_X, U_Y = self.prox_ops.prox_gi_admm(
            self.sensor_id,
            self.sensor_data,
            V_X, V_Y,
            self.anchor_positions,
            self.problem.alpha_admm / K_i
        )
        
        # Prox indicator
        S_i = self.prox_ops.construct_Si(U_X, U_Y, self.problem.d)
        S_i_proj = self.prox_ops.prox_indicator_psd(S_i)
        U_X, U_Y = self.prox_ops.extract_from_Si(S_i_proj, self.sensor_id, self.sensor_data)
        
        U_new = (U_X, U_Y)
        
        # Exchange U values
        tag = 3000 + iteration
        self.communicator.broadcast(
            self.sensor_id,
            U_new,
            tag,
            self.sensor_data.neighbors
        )
        
        # Receive neighbor U values
        messages = self.communicator.receive(self.sensor_id, tag, timeout=0.5)
        neighbor_U_list = [U_new]
        for msg in messages:
            if msg['from'] in self.sensor_data.neighbors:
                neighbor_U_list.append(msg['data'])
        
        # R update (average)
        K_i = len(neighbor_U_list)
        R_X = np.mean([u[0] for u in neighbor_U_list], axis=0)
        R_Y = np.mean([u[1] for u in neighbor_U_list], axis=0)
        R_new = (R_X, R_Y)
        
        # V update
        V_X_new = V[0] + R_new[0] - 0.5 * R_new[0] - 0.5 * U_new[0]
        V_Y_new = V[1] + R_new[1] - 0.5 * R_new[1] - 0.5 * U_new[1]
        V_new = (V_X_new, V_Y_new)
        
        # Compute local metrics
        self._compute_local_metrics(U_X, U_Y)
        
        return {
            'X': U_X,
            'Y': U_Y,
            'U': U_new,
            'V': V_new,
            'R': R_new,
            'objective': self.local_objective,
            'constraint_violation': self.local_constraint_violation
        }
    
    def _compute_local_metrics(self, X: np.ndarray, Y: np.ndarray):
        """Compute local contribution to objective and constraints"""
        self.local_objective = 0
        
        # Distance objective
        for j, measured_dist in self.sensor_data.distance_measurements.items():
            self.local_objective += abs(measured_dist**2 - np.linalg.norm(X)**2)
        
        # Anchor objective
        for k, measured_dist in self.sensor_data.anchor_distances.items():
            actual_dist = np.linalg.norm(X - self.anchor_positions[k])
            self.local_objective += abs(measured_dist**2 - actual_dist**2)
        
        # Constraint violation
        eigenvalues = np.linalg.eigvalsh(Y)
        self.local_constraint_violation = -np.minimum(0, np.min(eigenvalues))


class ThreadedSNLFull:
    """Full threaded implementation matching MPI version"""
    
    def __init__(self, problem: SNLProblem, n_threads: Optional[int] = None):
        self.problem = problem
        self.n_threads = n_threads or problem.n_sensors
        
        self.communicator = ThreadedCommunicator(problem.n_sensors)
        
        self.sensors = {}
        self.sensor_threads = {}
        
        self.anchor_positions = None
        self.true_positions = None
        
        self.executor = ThreadPoolExecutor(max_workers=self.n_threads)
        
        self.mps_state = AlgorithmState()
        self.admm_state = AlgorithmState()
        
        self.results_lock = threading.Lock()
        self.iteration_results = defaultdict(dict)
    
    def generate_network(self, seed: Optional[int] = None):
        """Generate network topology"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate positions
        self.anchor_positions = np.random.uniform(0, 1, (self.problem.n_anchors, self.problem.d))
        self.true_positions = np.random.normal(0.5, 0.2, (self.problem.n_sensors, self.problem.d))
        self.true_positions = np.clip(self.true_positions, 0, 1)
        
        # Build adjacency and distances
        adjacency, distance_matrix, anchor_distances = self._build_network_data()
        
        # Create sensors
        for i in range(self.problem.n_sensors):
            sensor_data = SensorData(id=i)
            sensor_data.neighbors = np.where(adjacency[i] > 0)[0].tolist()
            
            for j in sensor_data.neighbors:
                sensor_data.distance_measurements[j] = distance_matrix[i, j]
            
            if i in anchor_distances:
                sensor_data.anchor_distances = anchor_distances[i]
                sensor_data.anchor_neighbors = list(anchor_distances[i].keys())
            
            # Initialize matrices
            n_neighbors = len(sensor_data.neighbors)
            sensor_data.Y = np.zeros((n_neighbors + 1, n_neighbors + 1))
            sensor_data.X = np.zeros(self.problem.d)
            
            # Create sensor
            self.sensors[i] = ThreadedSensor(
                i, sensor_data, self.problem, 
                self.anchor_positions, self.communicator,
                self.true_positions[i]
            )
        
        logger.info(f"Created threaded network with {self.problem.n_sensors} sensors")
    
    def _build_network_data(self):
        """Build adjacency, distances, and anchor connections"""
        adjacency = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
        distance_matrix = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
        anchor_distances = {}
        
        for i in range(self.problem.n_sensors):
            distances = np.linalg.norm(self.true_positions[i] - self.true_positions, axis=1)
            neighbors = np.where((distances < self.problem.communication_range) & (distances > 0))[0]
            
            if len(neighbors) > self.problem.max_neighbors:
                neighbors = np.random.choice(neighbors, self.problem.max_neighbors, replace=False)
            
            for j in neighbors:
                adjacency[i, j] = 1
                noise = 1 + self.problem.noise_factor * np.random.randn()
                distance_matrix[i, j] = distances[j] * noise
        
        # Make symmetric
        adjacency = np.maximum(adjacency, adjacency.T)
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
        
        # Anchor distances
        for i in range(self.problem.n_sensors):
            anchor_distances[i] = {}
            for k in range(self.problem.n_anchors):
                dist = np.linalg.norm(self.true_positions[i] - self.anchor_positions[k])
                if dist < self.problem.communication_range:
                    noise = 1 + self.problem.noise_factor * np.random.randn()
                    anchor_distances[i][k] = dist * noise
        
        return adjacency, distance_matrix, anchor_distances
    
    def run_distributed_sinkhorn_knopp(self, max_iter: int = 100, tol: float = 1e-6):
        """Run Sinkhorn-Knopp using threads"""
        # Initialize edge weights
        edge_weights = {}
        for i, sensor in self.sensors.items():
            edge_weights[i] = {j: 1.0 for j in sensor.sensor_data.neighbors}
            edge_weights[i][i] = 1.0
        
        for iteration in range(max_iter):
            # Row normalization phase
            futures = []
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_sinkhorn_knopp_phase,
                    edge_weights[i], 'row', iteration
                )
                futures.append((i, future))
            
            # Collect row results
            for i, future in futures:
                edge_weights[i] = future.result()
            
            # Synchronize
            self.communicator.barrier(f"sk_row_{iteration}", self.problem.n_sensors)
            
            # Column normalization phase
            futures = []
            max_change = 0
            
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_sinkhorn_knopp_phase,
                    edge_weights[i], 'column', iteration
                )
                futures.append((i, future))
            
            # Collect column results and check convergence
            for i, future in futures:
                old_weights = edge_weights[i].copy()
                edge_weights[i] = future.result()
                
                for j, new_w in edge_weights[i].items():
                    old_w = old_weights.get(j, 0)
                    max_change = max(max_change, abs(new_w - old_w))
            
            if max_change < tol:
                logger.info(f"Sinkhorn-Knopp converged in {iteration + 1} iterations")
                break
        
        # Build matrix blocks
        for i, sensor in self.sensors.items():
            n = len(sensor.sensor_data.neighbors) + 1
            A_plus_I = np.zeros((n, n))
            
            # Map to local indices
            global_to_local = {i: 0}
            for idx, neighbor in enumerate(sensor.sensor_data.neighbors):
                global_to_local[neighbor] = idx + 1
            
            # Fill matrix
            for global_id, weight in edge_weights[i].items():
                if global_id in global_to_local:
                    local_i = 0
                    local_j = global_to_local[global_id]
                    A_plus_I[local_i, local_j] = weight
                    if global_id != i:
                        A_plus_I[local_j, local_i] = weight
            
            # Compute Z, W, L
            I = np.eye(n)
            sensor.Z_block = 2 * (I - A_plus_I)
            sensor.W_block = sensor.Z_block.copy()
            sensor.L_block = sensor.compute_local_L_from_Z(sensor.Z_block)
    
    def matrix_parametrized_splitting_threaded(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Run MPS algorithm using threads"""
        # Reset state
        self.mps_state = AlgorithmState()
        
        # Run Sinkhorn-Knopp
        self.run_distributed_sinkhorn_knopp()
        
        # Initialize dual variables
        v_gi = {}
        v_delta = {}
        for i, sensor in self.sensors.items():
            n = len(sensor.sensor_data.neighbors) + 1
            v_gi[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
            v_delta[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
        
        converged = False
        
        while not converged and self.mps_state.iteration < self.problem.max_iter:
            iteration_start = time.time()
            
            # Run iteration in parallel
            futures = []
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_mps_iteration,
                    v_gi[i], v_delta[i], self.mps_state.iteration
                )
                futures.append((i, future))
            
            # Collect results
            max_change = 0
            total_objective = 0
            total_violation = 0
            
            for i, future in futures:
                result = future.result()
                
                # Update sensor state
                old_X = self.sensors[i].sensor_data.X
                self.sensors[i].sensor_data.X = result['X']
                self.sensors[i].sensor_data.Y = result['Y']
                v_gi[i] = result['v_gi']
                v_delta[i] = result['v_delta']
                
                # Track change
                max_change = max(max_change, np.linalg.norm(result['X'] - old_X))
                
                # Accumulate metrics
                total_objective += result['objective']
                total_violation += result['constraint_violation']
            
            # Update state
            self.mps_state.objective_history.append(total_objective)
            self.mps_state.constraint_violation_history.append(total_violation)
            
            if self.true_positions is not None:
                error = self._compute_error()
                self.mps_state.error_history.append(error)
            
            iteration_time = time.time() - iteration_start
            self.mps_state.time_per_iteration.append(iteration_time)
            
            # Check early termination
            if self._check_early_termination(self.mps_state):
                self.mps_state.early_termination_triggered = True
                self.mps_state.early_termination_iteration = self.mps_state.iteration
                logger.info(f"Early termination at iteration {self.mps_state.iteration}")
                break
            
            # Update iteration
            self.mps_state.iteration += 1
            converged = max_change < self.problem.tol
            
            if converged:
                self.mps_state.converged = True
                self.mps_state.convergence_iteration = self.mps_state.iteration
            
            # Logging
            if self.mps_state.iteration % 50 == 0:
                logger.info(f"MPS iteration {self.mps_state.iteration}: "
                           f"obj={total_objective:.6f}, max_change={max_change:.6f}")
        
        # Return results
        results = {}
        for i, sensor in self.sensors.items():
            results[i] = (
                sensor.sensor_data.X.copy(),
                sensor.sensor_data.Y.copy()
            )
        
        return results
    
    def admm_decentralized_threaded(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Run ADMM algorithm using threads"""
        # Reset state
        self.admm_state = AlgorithmState()
        
        # Initialize variables
        U = {}
        V = {}
        R = {}
        for i, sensor in self.sensors.items():
            n = len(sensor.sensor_data.neighbors) + 1
            U[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
            V[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
            R[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
        
        converged = False
        
        while not converged and self.admm_state.iteration < self.problem.max_iter:
            iteration_start = time.time()
            
            # Run iteration in parallel
            futures = []
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_admm_iteration,
                    U[i], V[i], R[i], self.admm_state.iteration
                )
                futures.append((i, future))
            
            # Collect results
            max_change = 0
            total_objective = 0
            total_violation = 0
            
            for i, future in futures:
                result = future.result()
                
                # Update state
                old_X = self.sensors[i].sensor_data.X
                self.sensors[i].sensor_data.X = result['X']
                self.sensors[i].sensor_data.Y = result['Y']
                U[i] = result['U']
                V[i] = result['V']
                R[i] = result['R']
                
                # Track metrics
                max_change = max(max_change, np.linalg.norm(result['X'] - old_X))
                total_objective += result['objective']
                total_violation += result['constraint_violation']
            
            # Update state
            self.admm_state.objective_history.append(total_objective)
            self.admm_state.constraint_violation_history.append(total_violation)
            
            if self.true_positions is not None:
                error = self._compute_error()
                self.admm_state.error_history.append(error)
            
            iteration_time = time.time() - iteration_start
            self.admm_state.time_per_iteration.append(iteration_time)
            
            # Update iteration
            self.admm_state.iteration += 1
            converged = max_change < self.problem.tol
            
            if converged:
                self.admm_state.converged = True
                self.admm_state.convergence_iteration = self.admm_state.iteration
            
            # Logging
            if self.admm_state.iteration % 50 == 0:
                logger.info(f"ADMM iteration {self.admm_state.iteration}: "
                           f"obj={total_objective:.6f}, max_change={max_change:.6f}")
        
        # Return results
        results = {}
        for i, sensor in self.sensors.items():
            results[i] = (
                sensor.sensor_data.X.copy(),
                sensor.sensor_data.Y.copy()
            )
        
        return results
    
    def _compute_error(self) -> float:
        """Compute relative error from true positions"""
        estimated = np.zeros((self.problem.n_sensors, self.problem.d))
        for i, sensor in self.sensors.items():
            estimated[i] = sensor.sensor_data.X
        
        error = np.linalg.norm(estimated - self.true_positions, 'fro')
        error /= np.linalg.norm(self.true_positions, 'fro')
        
        return error
    
    def _check_early_termination(self, state: AlgorithmState) -> bool:
        """Check early termination criteria"""
        if len(state.objective_history) < self.problem.early_termination_window:
            return False
        
        window = self.problem.early_termination_window
        recent_objectives = state.objective_history[-window:]
        min_objective = min(state.objective_history)
        
        return all(obj >= min_objective * (1 - 1e-6) for obj in recent_objectives)
    
    def compare_algorithms_threaded(self) -> Dict[str, Any]:
        """Compare MPS and ADMM using threads"""
        logger.info("Starting threaded algorithm comparison...")
        
        # Run MPS
        start_time = time.time()
        mps_results = self.matrix_parametrized_splitting_threaded()
        mps_time = time.time() - start_time
        
        mps_metrics = {
            'total_time': mps_time,
            'iterations': self.mps_state.iteration,
            'final_objective': self.mps_state.objective_history[-1] if self.mps_state.objective_history else None,
            'final_error': self.mps_state.error_history[-1] if self.mps_state.error_history else None,
            'converged': self.mps_state.converged,
            'early_termination': self.mps_state.early_termination_triggered,
            'objective_history': self.mps_state.objective_history.copy(),
            'error_history': self.mps_state.error_history.copy()
        }
        
        # Reset for ADMM
        for sensor in self.sensors.values():
            sensor.sensor_data.X = np.zeros(self.problem.d)
            sensor.sensor_data.Y = np.zeros_like(sensor.sensor_data.Y)
        
        # Run ADMM
        start_time = time.time()
        admm_results = self.admm_decentralized_threaded()
        admm_time = time.time() - start_time
        
        admm_metrics = {
            'total_time': admm_time,
            'iterations': self.admm_state.iteration,
            'final_objective': self.admm_state.objective_history[-1] if self.admm_state.objective_history else None,
            'final_error': self.admm_state.error_history[-1] if self.admm_state.error_history else None,
            'converged': self.admm_state.converged,
            'objective_history': self.admm_state.objective_history.copy(),
            'error_history': self.admm_state.error_history.copy()
        }
        
        comparison = {
            'mps': mps_metrics,
            'admm': admm_metrics,
            'error_ratio': admm_metrics['final_error'] / mps_metrics['final_error'] if mps_metrics['final_error'] else None,
            'speedup': admm_time / mps_time,
            'iteration_ratio': admm_metrics['iterations'] / mps_metrics['iterations']
        }
        
        logger.info("="*60)
        logger.info("THREADED COMPARISON RESULTS")
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
    
    def shutdown(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


def demo():
    """Run a simple demonstration"""
    logging.basicConfig(level=logging.INFO)
    
    problem = SNLProblem(
        n_sensors=30,
        n_anchors=6,
        communication_range=0.7,
        noise_factor=0.05,
        max_iter=500,
        seed=42
    )
    
    logger.info("Testing standalone threaded implementation...")
    
    # Create and run
    snl = ThreadedSNLFull(problem)
    snl.generate_network(seed=42)
    
    comparison = snl.compare_algorithms_threaded()
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/standalone_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to test_results/standalone_comparison.json")
    
    snl.shutdown()
    
    return comparison


if __name__ == "__main__":
    results = demo()