"""
Simulation mode for testing without MPI
Uses threading to simulate distributed computation
"""

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from snl_main import SNLProblem, SensorData
from proximal_operators import ProximalOperators

logger = logging.getLogger(__name__)


class SimulatedSensor:
    """Simulated sensor that runs in its own thread"""
    
    def __init__(self, sensor_id: int, sensor_data: SensorData, 
                 problem: SNLProblem, anchor_positions: np.ndarray):
        self.sensor_id = sensor_id
        self.sensor_data = sensor_data
        self.problem = problem
        self.anchor_positions = anchor_positions
        
        # Communication queues
        self.inbox = queue.Queue()
        self.outboxes = {}  # neighbor_id -> queue
        
        # Algorithm state
        self.iteration = 0
        self.converged = False
        
        # Proximal operators
        self.prox_ops = ProximalOperators(problem)
    
    def connect_neighbor(self, neighbor_id: int, neighbor_inbox: queue.Queue):
        """Establish communication link with neighbor"""
        self.outboxes[neighbor_id] = neighbor_inbox
    
    def send_to_neighbor(self, neighbor_id: int, data: any):
        """Send data to a specific neighbor"""
        if neighbor_id in self.outboxes:
            self.outboxes[neighbor_id].put((self.sensor_id, data))
    
    def broadcast_to_neighbors(self, data: any):
        """Send data to all neighbors"""
        for neighbor_id in self.sensor_data.neighbors:
            self.send_to_neighbor(neighbor_id, data)
    
    def receive_from_neighbors(self, timeout: float = 1.0) -> Dict[int, any]:
        """Receive data from all neighbors"""
        received = {}
        expected = set(self.sensor_data.neighbors)
        
        deadline = time.time() + timeout
        while expected and time.time() < deadline:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                    
                sender_id, data = self.inbox.get(timeout=min(remaining, 0.1))
                if sender_id in expected:
                    received[sender_id] = data
                    expected.remove(sender_id)
            except queue.Empty:
                continue
        
        return received
    
    def run_sinkhorn_knopp_iteration(self, edge_weights: Dict[int, float]) -> Dict[int, float]:
        """One iteration of distributed Sinkhorn-Knopp"""
        # Row normalization
        total = sum(edge_weights.values()) + edge_weights.get(self.sensor_id, 1.0)
        normalized_weights = {}
        
        for neighbor_id, weight in edge_weights.items():
            normalized_weights[neighbor_id] = weight / total if total > 0 else 0
        normalized_weights[self.sensor_id] = edge_weights.get(self.sensor_id, 1.0) / total if total > 0 else 0
        
        # Send normalized weights to neighbors
        self.broadcast_to_neighbors(normalized_weights[self.sensor_id])
        
        # Receive weights from neighbors
        received_weights = self.receive_from_neighbors()
        
        # Column normalization
        incoming_weights = {self.sensor_id: normalized_weights[self.sensor_id]}
        for sender_id, weight in received_weights.items():
            incoming_weights[sender_id] = weight
        
        total_incoming = sum(incoming_weights.values())
        
        # Update edge weights
        new_weights = {}
        for neighbor_id in edge_weights:
            if neighbor_id in incoming_weights:
                new_weights[neighbor_id] = incoming_weights[neighbor_id] / total_incoming if total_incoming > 0 else 0
            else:
                new_weights[neighbor_id] = edge_weights[neighbor_id]
        
        return new_weights
    
    def run_mps_iteration(self, v_gi: Tuple[np.ndarray, np.ndarray], 
                         v_delta: Tuple[np.ndarray, np.ndarray],
                         W_block: np.ndarray, Z_block: np.ndarray,
                         gamma: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """One iteration of matrix-parametrized splitting"""
        # Compute prox_gi
        X_gi, Y_gi = self.prox_ops.prox_gi_admm(
            self.sensor_id, self.sensor_data,
            v_gi[0], v_gi[1],
            self.anchor_positions, alpha
        )
        
        # Exchange with neighbors for L matrix multiplication
        # (simplified - in reality need to implement full L multiplication)
        self.broadcast_to_neighbors((X_gi, Y_gi))
        neighbor_values = self.receive_from_neighbors()
        
        # Compute prox_indicator
        S_i = self.prox_ops.construct_Si(v_delta[0], v_delta[1], self.problem.d)
        S_i_proj = self.prox_ops.prox_indicator_psd(S_i)
        X_delta, Y_delta = self.prox_ops.extract_from_Si(S_i_proj, self.sensor_id, self.sensor_data)
        
        # Update dual variables
        v_gi_new = (
            v_gi[0] - gamma * W_block[0, 0] * X_gi,
            v_gi[1] - gamma * W_block @ Y_gi
        )
        
        v_delta_new = (
            v_delta[0] - gamma * W_block[0, 0] * X_delta,
            v_delta[1] - gamma * W_block @ Y_delta
        )
        
        # Update position estimate
        X_new = (X_gi + X_delta) / 2
        Y_new = (Y_gi + Y_delta) / 2
        
        return X_new, Y_new, v_gi_new, v_delta_new


class ThreadedSNLSimulator:
    """Simulate distributed SNL using threads instead of MPI"""
    
    def __init__(self, problem: SNLProblem):
        self.problem = problem
        self.sensors = {}
        self.anchor_positions = None
        self.true_positions = None
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=problem.n_sensors)
    
    def generate_network(self, seed: Optional[int] = None):
        """Generate network topology and initialize sensors"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate positions
        self.anchor_positions = np.random.uniform(0, 1, (self.problem.n_anchors, self.problem.d))
        self.true_positions = np.random.normal(0.5, 0.2, (self.problem.n_sensors, self.problem.d))
        self.true_positions = np.clip(self.true_positions, 0, 1)
        
        # Build adjacency
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
        
        # Make symmetric
        adjacency = np.maximum(adjacency, adjacency.T)
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
        
        # Create sensors
        for i in range(self.problem.n_sensors):
            sensor_data = SensorData(id=i)
            sensor_data.neighbors = np.where(adjacency[i] > 0)[0].tolist()
            
            # Set distance measurements
            for j in sensor_data.neighbors:
                sensor_data.distance_measurements[j] = distance_matrix[i, j]
            
            # Set anchor measurements
            for k in range(self.problem.n_anchors):
                dist = np.linalg.norm(self.true_positions[i] - self.anchor_positions[k])
                if dist < self.problem.communication_range:
                    noise = 1 + self.problem.noise_factor * np.random.randn()
                    sensor_data.anchor_distances[k] = dist * noise
                    sensor_data.anchor_neighbors.append(k)
            
            # Initialize matrices
            n_neighbors = len(sensor_data.neighbors)
            sensor_data.Y = np.zeros((n_neighbors + 1, n_neighbors + 1))
            sensor_data.X = np.zeros(self.problem.d)
            
            # Create simulated sensor
            self.sensors[i] = SimulatedSensor(i, sensor_data, self.problem, self.anchor_positions)
        
        # Connect sensors
        for i, sensor in self.sensors.items():
            for j in sensor.sensor_data.neighbors:
                if j in self.sensors:
                    sensor.connect_neighbor(j, self.sensors[j].inbox)
        
        logger.info(f"Created network with {self.problem.n_sensors} sensors")
    
    def run_distributed_sinkhorn_knopp(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, Dict[int, float]]:
        """Run Sinkhorn-Knopp in parallel using threads"""
        # Initialize edge weights
        edge_weights = {}
        for i, sensor in self.sensors.items():
            edge_weights[i] = {j: 1.0 for j in sensor.sensor_data.neighbors}
            edge_weights[i][i] = 1.0
        
        for iteration in range(max_iter):
            futures = []
            
            # Run iteration in parallel
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_sinkhorn_knopp_iteration,
                    edge_weights[i]
                )
                futures.append((i, future))
            
            # Collect results
            new_weights = {}
            max_change = 0
            
            for i, future in futures:
                new_weights[i] = future.result()
                
                # Check convergence
                for j, new_w in new_weights[i].items():
                    old_w = edge_weights[i].get(j, 0)
                    max_change = max(max_change, abs(new_w - old_w))
            
            edge_weights = new_weights
            
            if max_change < tol:
                logger.info(f"Sinkhorn-Knopp converged in {iteration + 1} iterations")
                break
        
        return edge_weights
    
    def run_mps(self, max_iter: int = 500, tol: float = 1e-6) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Run matrix-parametrized splitting using threads"""
        # Get matrix parameters from Sinkhorn-Knopp
        edge_weights = self.run_distributed_sinkhorn_knopp()
        
        # Build Z and W matrices for each sensor
        Z_blocks = {}
        W_blocks = {}
        
        for i, sensor in self.sensors.items():
            n = len(sensor.sensor_data.neighbors) + 1
            Z = 2 * np.eye(n)
            
            # Fill in edge weights
            for j, neighbor in enumerate(sensor.sensor_data.neighbors):
                if neighbor in edge_weights[i]:
                    Z[0, j+1] = Z[j+1, 0] = -edge_weights[i][neighbor]
            
            Z_blocks[i] = Z
            W_blocks[i] = Z.copy()
        
        # Initialize dual variables
        v_gi = {}
        v_delta = {}
        
        for i, sensor in self.sensors.items():
            n = len(sensor.sensor_data.neighbors) + 1
            v_gi[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
            v_delta[i] = (np.zeros(self.problem.d), np.zeros((n, n)))
        
        # Run iterations
        objective_history = []
        
        for iteration in range(max_iter):
            futures = []
            
            # Run MPS iteration in parallel
            for i, sensor in self.sensors.items():
                future = self.executor.submit(
                    sensor.run_mps_iteration,
                    v_gi[i], v_delta[i],
                    W_blocks[i], Z_blocks[i],
                    self.problem.gamma, self.problem.alpha_mps
                )
                futures.append((i, future))
            
            # Collect results
            max_change = 0
            total_objective = 0
            
            for i, future in futures:
                X_new, Y_new, v_gi_new, v_delta_new = future.result()
                
                # Check convergence
                max_change = max(max_change, np.linalg.norm(X_new - self.sensors[i].sensor_data.X))
                
                # Update sensor state
                self.sensors[i].sensor_data.X = X_new
                self.sensors[i].sensor_data.Y = Y_new
                v_gi[i] = v_gi_new
                v_delta[i] = v_delta_new
                
                # Compute objective contribution
                sensor = self.sensors[i]
                for j, dist in sensor.sensor_data.distance_measurements.items():
                    if j in self.sensors:
                        diff = dist**2 - np.linalg.norm(X_new - self.sensors[j].sensor_data.X)**2
                        total_objective += abs(diff)
                
                for k, dist in sensor.sensor_data.anchor_distances.items():
                    diff = dist**2 - np.linalg.norm(X_new - self.anchor_positions[k])**2
                    total_objective += abs(diff)
            
            objective_history.append(total_objective)
            
            # Early termination check
            if len(objective_history) > self.problem.early_termination_window:
                min_obj = min(objective_history)
                if all(obj >= min_obj for obj in objective_history[-self.problem.early_termination_window:]):
                    logger.info(f"Early termination at iteration {iteration}")
                    break
            
            if max_change < tol:
                logger.info(f"MPS converged in {iteration + 1} iterations")
                break
            
            if iteration % 50 == 0:
                logger.info(f"MPS iteration {iteration}, objective: {total_objective:.6f}, max change: {max_change:.6f}")
        
        # Return results
        results = {}
        for i, sensor in self.sensors.items():
            results[i] = (sensor.sensor_data.X.copy(), sensor.sensor_data.Y.copy())
        
        return results
    
    def compute_error(self, results: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute relative error from true positions"""
        estimated = np.zeros((self.problem.n_sensors, self.problem.d))
        
        for i, (X, Y) in results.items():
            estimated[i] = X
        
        error = np.linalg.norm(estimated - self.true_positions, 'fro')
        error /= np.linalg.norm(self.true_positions, 'fro')
        
        return error
    
    def shutdown(self):
        """Clean up thread pool"""
        self.executor.shutdown(wait=True)


def compare_with_mpi():
    """Compare threaded simulation with MPI implementation"""
    problem = SNLProblem(
        n_sensors=30,
        n_anchors=6,
        communication_range=0.7,
        noise_factor=0.05,
        seed=42
    )
    
    # Run threaded simulation
    logger.info("Running threaded simulation...")
    simulator = ThreadedSNLSimulator(problem)
    simulator.generate_network(seed=42)
    
    start_time = time.time()
    mps_results = simulator.run_mps()
    mps_time = time.time() - start_time
    
    mps_error = simulator.compute_error(mps_results)
    
    logger.info(f"Threaded MPS: Error={mps_error:.6f}, Time={mps_time:.2f}s")
    
    simulator.shutdown()
    
    # Compare with MPI if available
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            logger.info("\nCompare with MPI implementation by running:")
            logger.info("mpirun -np 30 python snl_main.py")
    except ImportError:
        logger.info("MPI not available for comparison")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compare_with_mpi()