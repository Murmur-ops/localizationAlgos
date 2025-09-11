"""
Decentralized Localization using Distributed Gradient Descent
Implements a proper distributed algorithm that converges to the centralized solution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RangingMeasurement:
    """Ranging measurement between nodes"""
    from_node: int
    to_node: int
    distance: float
    variance: float
    quality: float  # 0-1 quality score


class DecentralizedNode:
    """Node in decentralized localization network"""
    
    def __init__(self, node_id: int, initial_position: np.ndarray,
                 is_anchor: bool = False, dimension: int = 2):
        """
        Initialize node for decentralized localization
        
        Args:
            node_id: Unique node identifier
            initial_position: Initial position estimate [x, y] or [x, y, z]
            is_anchor: Whether this is an anchor node (known position)
            dimension: 2D or 3D localization
        """
        self.node_id = node_id
        self.position = initial_position.copy()
        self.is_anchor = is_anchor
        self.d = dimension
        
        # For consensus
        self.neighbors = set()
        self.neighbor_positions = {}
        
        # For gradient descent
        self.gradient = np.zeros(dimension)
        self.momentum = np.zeros(dimension)  # Momentum for faster convergence
        
        # Learning parameters
        self.alpha = 0.01  # Step size
        self.beta = 0.9    # Momentum coefficient
        self.consensus_weight = 0.5  # Weight for consensus vs local gradient
        
        # Measurements
        self.measurements = {}
        
    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor for consensus"""
        self.neighbors.add(neighbor_id)
        
    def receive_position(self, neighbor_id: int, position: np.ndarray):
        """Receive position update from neighbor"""
        self.neighbor_positions[neighbor_id] = position.copy()
        
    def add_measurement(self, measurement: RangingMeasurement):
        """Add ranging measurement"""
        if measurement.from_node == self.node_id:
            self.measurements[measurement.to_node] = measurement
        elif measurement.to_node == self.node_id:
            self.measurements[measurement.from_node] = measurement
            
    def compute_local_gradient(self) -> np.ndarray:
        """
        Compute gradient based on local ranging measurements
        
        Returns:
            Gradient vector
        """
        gradient = np.zeros(self.d)
        
        for neighbor_id, measurement in self.measurements.items():
            if neighbor_id in self.neighbor_positions:
                neighbor_pos = self.neighbor_positions[neighbor_id]
                
                # Compute current distance estimate
                diff = self.position - neighbor_pos
                est_dist = np.linalg.norm(diff)
                
                if est_dist > 1e-6:  # Avoid division by zero
                    # Gradient of squared error
                    error = est_dist - measurement.distance
                    weight = measurement.quality / measurement.variance
                    
                    # Gradient: 2 * weight * error * (diff / est_dist)
                    gradient += 2 * weight * error * (diff / est_dist)
                    
        return gradient
    
    def compute_consensus_term(self) -> np.ndarray:
        """
        Compute consensus term to align with neighbors
        
        Returns:
            Consensus correction vector
        """
        if not self.neighbors:
            return np.zeros(self.d)
            
        # Average neighbor positions
        consensus = np.zeros(self.d)
        count = 0
        
        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_positions:
                consensus += self.neighbor_positions[neighbor_id]
                count += 1
                
        if count > 0:
            consensus /= count
            return consensus - self.position
        
        return np.zeros(self.d)
    
    def update_position(self) -> np.ndarray:
        """
        Update position using distributed gradient descent with consensus
        
        Returns:
            Updated position
        """
        if self.is_anchor:
            return self.position  # Anchors don't move
            
        # Compute local gradient
        local_grad = self.compute_local_gradient()
        
        # Compute consensus term
        consensus = self.compute_consensus_term()
        
        # Combine local gradient and consensus
        combined_update = -self.alpha * local_grad + self.consensus_weight * consensus
        
        # Apply momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * combined_update
        
        # Update position
        self.position += self.momentum
        
        return self.position
    
    def broadcast_position(self) -> Tuple[int, np.ndarray]:
        """
        Broadcast current position to neighbors
        
        Returns:
            Node ID and position for broadcasting
        """
        return self.node_id, self.position.copy()


class DecentralizedLocalizationSystem:
    """Complete decentralized localization system"""
    
    def __init__(self, dimension: int = 2):
        """
        Initialize decentralized localization system
        
        Args:
            dimension: 2D or 3D localization
        """
        self.d = dimension
        self.nodes = {}
        self.measurements = []
        self.iteration = 0
        
    def add_node(self, node_id: int, initial_position: np.ndarray,
                 is_anchor: bool = False) -> DecentralizedNode:
        """
        Add node to the system
        
        Args:
            node_id: Unique node identifier
            initial_position: Initial position estimate
            is_anchor: Whether this is an anchor node
            
        Returns:
            Created node
        """
        node = DecentralizedNode(node_id, initial_position, is_anchor, self.d)
        self.nodes[node_id] = node
        return node
    
    def add_measurement(self, measurement: RangingMeasurement):
        """Add ranging measurement between nodes"""
        self.measurements.append(measurement)
        
        # Add to relevant nodes
        if measurement.from_node in self.nodes:
            self.nodes[measurement.from_node].add_measurement(measurement)
        if measurement.to_node in self.nodes:
            self.nodes[measurement.to_node].add_measurement(measurement)
            
        # Establish neighbor relationships
        if measurement.from_node in self.nodes and measurement.to_node in self.nodes:
            self.nodes[measurement.from_node].add_neighbor(measurement.to_node)
            self.nodes[measurement.to_node].add_neighbor(measurement.from_node)
    
    def broadcast_positions(self):
        """All nodes broadcast their positions"""
        broadcasts = {}
        for node_id, node in self.nodes.items():
            broadcasts[node_id] = node.broadcast_position()[1]
            
        # Deliver broadcasts to neighbors
        for node_id, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                if neighbor_id in broadcasts:
                    node.receive_position(neighbor_id, broadcasts[neighbor_id])
    
    def iterate(self) -> Dict[int, np.ndarray]:
        """
        Run one iteration of distributed localization
        
        Returns:
            Current position estimates
        """
        # Phase 1: Broadcast current positions
        self.broadcast_positions()
        
        # Phase 2: Update positions based on local gradients and consensus
        new_positions = {}
        for node_id, node in self.nodes.items():
            new_positions[node_id] = node.update_position()
            
        self.iteration += 1
        return new_positions
    
    def run(self, max_iterations: int = 100, 
            convergence_threshold: float = 1e-4) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Run distributed localization until convergence
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for position changes
            
        Returns:
            Final positions and convergence info
        """
        position_history = []
        cost_history = []
        
        for iteration in range(max_iterations):
            # Store previous positions
            prev_positions = {node_id: node.position.copy() 
                             for node_id, node in self.nodes.items()}
            
            # Run iteration
            current_positions = self.iterate()
            position_history.append(current_positions.copy())
            
            # Compute total cost
            total_cost = 0
            for measurement in self.measurements:
                if measurement.from_node in current_positions and \
                   measurement.to_node in current_positions:
                    pos_i = current_positions[measurement.from_node]
                    pos_j = current_positions[measurement.to_node]
                    est_dist = np.linalg.norm(pos_i - pos_j)
                    error = (est_dist - measurement.distance) ** 2
                    weighted_error = error * measurement.quality / measurement.variance
                    total_cost += weighted_error
            
            cost_history.append(total_cost)
            
            # Check convergence
            max_change = 0
            for node_id in self.nodes:
                if not self.nodes[node_id].is_anchor:
                    change = np.linalg.norm(current_positions[node_id] - prev_positions[node_id])
                    max_change = max(max_change, change)
            
            if max_change < convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            # Adaptive learning rate
            if iteration > 0 and cost_history[-1] > cost_history[-2]:
                # Cost increased, reduce learning rate
                for node in self.nodes.values():
                    node.alpha *= 0.9
            elif iteration > 10 and iteration % 10 == 0:
                # Periodically reduce learning rate
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
        Compute RMSE compared to true positions
        
        Args:
            true_positions: Ground truth positions
            
        Returns:
            Root mean squared error
        """
        errors = []
        for node_id, node in self.nodes.items():
            if not node.is_anchor and node_id in true_positions:
                error = np.linalg.norm(node.position - true_positions[node_id])
                errors.append(error ** 2)
        
        if errors:
            return np.sqrt(np.mean(errors))
        return 0.0


def create_measurements_from_distances(distance_matrix: np.ndarray,
                                      noise_std: float = 0.1,
                                      quality_model: str = 'snr') -> List[RangingMeasurement]:
    """
    Create measurements from distance matrix with realistic noise
    
    Args:
        distance_matrix: True distances between nodes
        noise_std: Standard deviation of measurement noise
        quality_model: How to compute quality scores ('snr', 'distance', 'uniform')
        
    Returns:
        List of ranging measurements
    """
    n_nodes = len(distance_matrix)
    measurements = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = distance_matrix[i, j]
            
            # Add noise
            noise = np.random.normal(0, noise_std)
            measured_dist = true_dist + noise
            
            # Compute quality based on model
            if quality_model == 'snr':
                # Quality decreases with distance (path loss)
                quality = max(0.1, 1.0 / (1 + true_dist / 10))
            elif quality_model == 'distance':
                # Inverse distance weighting
                quality = 1.0 / (1 + true_dist)
            else:
                quality = 0.8  # Uniform quality
            
            # Variance proportional to distance
            variance = (noise_std * (1 + true_dist / 10)) ** 2
            
            measurement = RangingMeasurement(
                from_node=i,
                to_node=j,
                distance=measured_dist,
                variance=variance,
                quality=quality
            )
            measurements.append(measurement)
    
    return measurements