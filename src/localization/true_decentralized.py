"""
True Decentralized Localization with Local Information Only
Each node only knows:
- Its own measurements to neighbors
- Position estimates received from direct neighbors
- No global information!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LocalMeasurement:
    """Local ranging measurement to a neighbor"""
    neighbor_id: int
    distance: float
    variance: float
    quality: float


class TrueDecentralizedNode:
    """
    Node that ONLY uses local information
    - Can only measure distances to direct neighbors
    - Can only communicate with direct neighbors
    - No global topology knowledge
    """
    
    def __init__(self, node_id: int, initial_position: np.ndarray,
                 is_anchor: bool = False, dimension: int = 2):
        """
        Initialize truly decentralized node
        
        Args:
            node_id: Unique identifier
            initial_position: Initial position estimate
            is_anchor: Whether this is an anchor (known position)
            dimension: 2D or 3D
        """
        self.node_id = node_id
        self.position = initial_position.copy()
        self.is_anchor = is_anchor
        self.d = dimension
        
        # Local information only
        self.local_measurements = {}  # neighbor_id -> measurement
        self.neighbor_positions = {}  # neighbor_id -> position estimate
        self.neighbor_types = {}  # neighbor_id -> is_anchor
        
        # For distributed optimization
        self.gradient = np.zeros(dimension)
        self.momentum = np.zeros(dimension)
        
        # Consensus variables for ADMM
        self.z_consensus = {}  # neighbor_id -> consensus position
        self.u_dual = {}  # neighbor_id -> dual variable
        
        # Parameters
        self.alpha = 0.5  # Gradient step size (increased for faster convergence)
        self.beta = 0.7   # Momentum (increased for smoother convergence)
        self.rho = 0.5    # ADMM penalty parameter (reduced to balance with measurements)
        
        # Track direct neighbors only
        self.direct_neighbors = set()
        
    def add_local_measurement(self, neighbor_id: int, distance: float, 
                             variance: float = 0.01, quality: float = 1.0):
        """
        Add measurement to a direct neighbor
        
        Args:
            neighbor_id: ID of the neighbor
            distance: Measured distance
            variance: Measurement variance
            quality: Measurement quality (0-1)
        """
        self.local_measurements[neighbor_id] = LocalMeasurement(
            neighbor_id=neighbor_id,
            distance=distance,
            variance=variance,
            quality=quality
        )
        self.direct_neighbors.add(neighbor_id)
        
        # Initialize consensus variables
        if neighbor_id not in self.z_consensus:
            self.z_consensus[neighbor_id] = self.position.copy()
            self.u_dual[neighbor_id] = np.zeros(self.d)
    
    def receive_neighbor_position(self, neighbor_id: int, position: np.ndarray, 
                                 is_anchor: bool = False):
        """
        Receive position update from a direct neighbor
        
        Args:
            neighbor_id: Neighbor's ID
            position: Neighbor's position estimate
            is_anchor: Whether neighbor is an anchor
        """
        if neighbor_id in self.direct_neighbors:
            self.neighbor_positions[neighbor_id] = position.copy()
            self.neighbor_types[neighbor_id] = is_anchor
    
    def compute_local_gradient(self) -> np.ndarray:
        """
        Compute gradient using ONLY local measurements
        
        Returns:
            Gradient vector based on local information
        """
        gradient = np.zeros(self.d)
        
        for neighbor_id, measurement in self.local_measurements.items():
            if neighbor_id in self.neighbor_positions:
                # Use neighbor's shared position estimate
                neighbor_pos = self.neighbor_positions[neighbor_id]
                
                # Current estimated distance
                diff = self.position - neighbor_pos
                est_dist = np.linalg.norm(diff)
                
                if est_dist > 1e-6:
                    # Gradient of squared error
                    error = est_dist - measurement.distance
                    weight = measurement.quality / (measurement.variance + 1e-6)
                    
                    # Gradient contribution
                    gradient += weight * error * (diff / est_dist)
        
        return gradient
    
    def update_position_admm(self) -> np.ndarray:
        """
        Update position using ADMM with consensus
        Each node solves: min f_i(x_i) + (rho/2)||x_i - z_i||^2
        
        Returns:
            Updated position
        """
        if self.is_anchor:
            return self.position
        
        # Build local optimization problem
        # minimize: sum_j w_ij*(||x_i - p_j|| - d_ij)^2 + (rho/2)*sum_j||x_i - z_ij + u_ij||^2
        
        # Newton-Raphson iteration
        H = np.zeros((self.d, self.d))  # Hessian
        g = np.zeros(self.d)  # Gradient
        
        # Measurement terms
        for neighbor_id, measurement in self.local_measurements.items():
            if neighbor_id in self.neighbor_positions:
                neighbor_pos = self.neighbor_positions[neighbor_id]
                diff = self.position - neighbor_pos
                dist = np.linalg.norm(diff)
                
                if dist > 1e-6:
                    weight = measurement.quality / (measurement.variance + 1e-6)
                    error = dist - measurement.distance
                    
                    # First derivative
                    grad_dist = diff / dist
                    g += weight * error * grad_dist
                    
                    # Second derivative (Gauss-Newton approximation)
                    H += weight * np.outer(grad_dist, grad_dist)
        
        # Consensus terms (ADMM)
        for neighbor_id in self.direct_neighbors:
            if neighbor_id in self.z_consensus:
                # Add consensus penalty: (rho/2)||x - z + u||^2
                g += self.rho * (self.position - self.z_consensus[neighbor_id] + self.u_dual[neighbor_id])
                H += self.rho * np.eye(self.d)
        
        # Solve H * delta = -g
        try:
            # Add regularization for stability
            H_reg = H + 0.01 * np.eye(self.d)
            delta = -np.linalg.solve(H_reg, g)
            
            # Apply momentum
            self.momentum = self.beta * self.momentum + (1 - self.beta) * delta
            
            # Update with step size
            self.position += self.alpha * self.momentum
            
        except np.linalg.LinAlgError:
            # If singular, use gradient descent
            self.position -= self.alpha * g
        
        return self.position
    
    def update_consensus_variables(self, neighbor_positions: Dict[int, np.ndarray]):
        """
        Update consensus variables (z and u) for ADMM
        
        Args:
            neighbor_positions: Current positions from neighbors
        """
        for neighbor_id in self.direct_neighbors:
            if neighbor_id in neighbor_positions:
                neighbor_pos = neighbor_positions[neighbor_id]
                
                # Update z (consensus variable) - average of both positions
                self.z_consensus[neighbor_id] = 0.5 * (self.position + neighbor_pos)
                
                # Update dual variable
                self.u_dual[neighbor_id] += self.rho * (self.position - self.z_consensus[neighbor_id])
    
    def get_message_for_neighbors(self) -> Dict:
        """
        Prepare message to send to direct neighbors
        
        Returns:
            Message containing position and metadata
        """
        return {
            'node_id': self.node_id,
            'position': self.position.copy(),
            'is_anchor': self.is_anchor
        }


class TrueDecentralizedSystem:
    """
    Truly decentralized system where nodes only communicate with direct neighbors
    """
    
    def __init__(self, dimension: int = 2):
        """
        Initialize decentralized system
        
        Args:
            dimension: 2D or 3D
        """
        self.d = dimension
        self.nodes = {}
        self.topology = {}  # node_id -> set of neighbor_ids
        
    def add_node(self, node_id: int, initial_position: np.ndarray,
                 is_anchor: bool = False) -> TrueDecentralizedNode:
        """
        Add node to system
        
        Args:
            node_id: Unique identifier
            initial_position: Initial position
            is_anchor: Whether this is an anchor
            
        Returns:
            Created node
        """
        node = TrueDecentralizedNode(node_id, initial_position, is_anchor, self.d)
        self.nodes[node_id] = node
        self.topology[node_id] = set()
        return node
    
    def add_edge(self, node_i: int, node_j: int, distance: float,
                 variance: float = 0.01, quality: float = 1.0):
        """
        Add edge (measurement) between two nodes
        
        Args:
            node_i, node_j: Node IDs
            distance: Measured distance
            variance: Measurement variance
            quality: Measurement quality
        """
        if node_i in self.nodes and node_j in self.nodes:
            # Add bidirectional measurement
            self.nodes[node_i].add_local_measurement(node_j, distance, variance, quality)
            self.nodes[node_j].add_local_measurement(node_i, distance, variance, quality)
            
            # Update topology
            self.topology[node_i].add(node_j)
            self.topology[node_j].add(node_i)
    
    def simulate_communication_round(self):
        """
        Simulate one round of neighbor-to-neighbor communication
        Each node broadcasts to its direct neighbors only
        """
        # Collect all messages
        messages = {}
        for node_id, node in self.nodes.items():
            messages[node_id] = node.get_message_for_neighbors()
        
        # Deliver messages to direct neighbors only
        for node_id, node in self.nodes.items():
            for neighbor_id in node.direct_neighbors:
                if neighbor_id in messages:
                    msg = messages[neighbor_id]
                    node.receive_neighbor_position(
                        msg['node_id'],
                        msg['position'],
                        msg['is_anchor']
                    )
    
    def iterate_admm(self) -> Dict[int, np.ndarray]:
        """
        One iteration of distributed ADMM
        
        Returns:
            Current positions
        """
        # Phase 1: Exchange positions with neighbors
        self.simulate_communication_round()
        
        # Phase 2: Local position updates
        new_positions = {}
        for node_id, node in self.nodes.items():
            new_positions[node_id] = node.update_position_admm()
        
        # Phase 3: Update consensus variables
        neighbor_positions_dict = {}
        for node_id, node in self.nodes.items():
            neighbor_pos = {}
            for neighbor_id in node.direct_neighbors:
                if neighbor_id in new_positions:
                    neighbor_pos[neighbor_id] = new_positions[neighbor_id]
            neighbor_positions_dict[node_id] = neighbor_pos
        
        for node_id, node in self.nodes.items():
            node.update_consensus_variables(neighbor_positions_dict[node_id])
        
        return new_positions
    
    def run(self, max_iterations: int = 100,
            convergence_threshold: float = 1e-3) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Run distributed ADMM until convergence
        
        Args:
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Final positions and convergence info
        """
        position_history = []
        cost_history = []
        
        for iteration in range(max_iterations):
            # Store previous positions
            prev_positions = {nid: node.position.copy() 
                             for nid, node in self.nodes.items()}
            
            # Run ADMM iteration
            current_positions = self.iterate_admm()
            position_history.append(current_positions.copy())
            
            # Compute total cost (for monitoring only - nodes don't know this!)
            total_cost = 0
            for node_id, node in self.nodes.items():
                for neighbor_id, measurement in node.local_measurements.items():
                    if neighbor_id in current_positions:
                        diff = current_positions[node_id] - current_positions[neighbor_id]
                        est_dist = np.linalg.norm(diff)
                        error = (est_dist - measurement.distance) ** 2
                        total_cost += error * measurement.quality / (2 * measurement.variance)
            
            cost_history.append(total_cost)
            
            # Check convergence
            max_change = 0
            for node_id, node in self.nodes.items():
                if not node.is_anchor:
                    change = np.linalg.norm(current_positions[node_id] - prev_positions[node_id])
                    max_change = max(max_change, change)
            
            if max_change < convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Adaptive parameter tuning
            if iteration > 0 and iteration % 20 == 0:
                if cost_history[-1] > cost_history[-20]:
                    # Not improving, reduce step size
                    for node in self.nodes.values():
                        node.alpha *= 0.8
                else:
                    # Good progress, slightly reduce for fine-tuning
                    for node in self.nodes.values():
                        node.alpha *= 0.95
        
        return current_positions, {
            'iterations': iteration + 1,
            'final_cost': cost_history[-1] if cost_history else float('inf'),
            'cost_history': cost_history,
            'position_history': position_history,
            'converged': iteration < max_iterations - 1
        }
    
    def compute_rmse(self, true_positions: Dict[int, np.ndarray]) -> float:
        """
        Compute RMSE (for evaluation only - nodes don't know true positions!)
        
        Args:
            true_positions: Ground truth
            
        Returns:
            RMSE
        """
        errors = []
        for node_id, node in self.nodes.items():
            if not node.is_anchor and node_id in true_positions:
                error = np.linalg.norm(node.position - true_positions[node_id])
                errors.append(error ** 2)
        
        return np.sqrt(np.mean(errors)) if errors else 0.0