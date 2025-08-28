"""
Belief Propagation for Distributed Sensor Localization
"""

import numpy as np
from numpy.linalg import inv, norm
from typing import Dict, Tuple, Optional
import networkx as nx


class BeliefPropagation:
    def __init__(self, n_sensors: int, n_anchors: int, communication_range: float,
                 noise_factor: float = 0.05, max_iter: int = 50, 
                 damping: float = 0.5, tol: float = 1e-4):
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        self.communication_range = communication_range
        self.noise_factor = noise_factor
        self.max_iter = max_iter
        self.damping = damping
        self.tol = tol
        
        # Graph structure
        self.graph = None
        self.edges = []
        
        # Beliefs: mean and precision (inverse covariance)
        self.beliefs = {}  # {node_id: {'mean': np.array, 'precision': np.array}}
        self.messages = {}  # {(from, to): {'mean': np.array, 'precision': np.array}}
        
        # Measurements
        self.distances = {}
        self.anchor_positions = None
        self.true_positions = None
        
    def generate_network(self, true_positions: Dict, anchor_positions: np.ndarray):
        """Build network from positions"""
        self.true_positions = true_positions
        self.anchor_positions = anchor_positions
        
        # Build graph
        self.graph = nx.Graph()
        for i in range(self.n_sensors):
            self.graph.add_node(i)
            
        # Add edges based on communication range
        self.edges = []
        for i in range(self.n_sensors):
            for j in range(i+1, self.n_sensors):
                dist = norm(true_positions[i] - true_positions[j])
                if dist <= self.communication_range:
                    self.graph.add_edge(i, j)
                    self.edges.append((i, j))
                    # Store noisy distance
                    noise = np.random.normal(0, self.noise_factor * dist)
                    self.distances[(i, j)] = dist + noise
                    self.distances[(j, i)] = dist + noise
                    
        # Add anchor measurements
        for i in range(self.n_sensors):
            for a in range(self.n_anchors):
                dist = norm(true_positions[i] - anchor_positions[a])
                if dist <= self.communication_range:
                    noise = np.random.normal(0, self.noise_factor * dist)
                    self.distances[(i, f'a{a}')] = dist + noise
                    
    def initialize_beliefs(self):
        """Initialize beliefs with random positions and low confidence"""
        for i in range(self.n_sensors):
            # Start with random position
            mean = np.random.uniform(0.2, 0.8, 2)
            # Low confidence (high variance = low precision)
            precision = np.eye(2) * 1.0  # Lower initial precision
            
            # Use all available anchor measurements for better init
            anchor_constraints = []
            for a in range(self.n_anchors):
                if (i, f'a{a}') in self.distances:
                    d = self.distances[(i, f'a{a}')]
                    anchor_constraints.append((self.anchor_positions[a], d))
            
            if len(anchor_constraints) >= 2:
                # Use first two anchors for trilateration-like init
                a1_pos, d1 = anchor_constraints[0]
                a2_pos, d2 = anchor_constraints[1]
                
                # Approximate position between anchors
                direction = a2_pos - a1_pos
                direction /= norm(direction)
                
                # Simple geometric estimate
                mean = a1_pos + direction * d1 * 0.5 + np.random.randn(2) * 0.1
                mean = np.clip(mean, 0, 1)
                precision = np.eye(2) * 10.0  # Higher confidence with anchors
                    
            self.beliefs[i] = {'mean': mean, 'precision': precision}
            
        # Initialize messages to zero information
        for edge in self.edges:
            self.messages[(edge[0], edge[1])] = {'mean': np.zeros(2), 'precision': np.zeros((2, 2))}
            self.messages[(edge[1], edge[0])] = {'mean': np.zeros(2), 'precision': np.zeros((2, 2))}
            
    def compute_message(self, from_node: int, to_node: int) -> Dict:
        """Compute message from from_node to to_node based on distance constraint"""
        # Get current beliefs
        from_belief = self.beliefs[from_node]
        to_belief = self.beliefs[to_node]
        
        # Measured distance
        if (from_node, to_node) in self.distances:
            d_measured = self.distances[(from_node, to_node)]
        else:
            d_measured = self.distances[(to_node, from_node)]
            
        # Current position estimates
        x_from = from_belief['mean']
        x_to = to_belief['mean']
        
        # Estimated distance
        d_est = norm(x_from - x_to)
        if d_est < 1e-6:
            # Avoid singularity
            return {'mean': np.zeros(2), 'precision': np.zeros((2, 2))}
            
        # Linearization: Jacobian of distance w.r.t. to_node position
        H = (x_to - x_from) / d_est
        
        # Measurement variance
        sigma2 = (self.noise_factor * d_measured) ** 2
        
        # Information form message
        # precision = H^T H / sigma^2
        precision = np.outer(H, H) / sigma2
        
        # Limit precision to avoid numerical issues
        max_precision = 1000.0
        precision = np.clip(precision, -max_precision, max_precision)
        
        # innovation = (measured - estimated)
        innovation = (d_measured - d_est)
        
        # Limit innovation to reasonable range
        innovation = np.clip(innovation, -0.2, 0.2)
        
        # Information vector = precision * (position + correction)
        correction = innovation * H 
        info_vector = precision @ (x_to + correction)
        
        return {'mean': info_vector, 'precision': precision}
        
    def update_belief(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """Update belief by combining incoming messages and anchor constraints"""
        # Start with weak prior
        total_precision = np.eye(2) * 1.0
        total_info_vector = np.zeros(2)
        
        # Add anchor constraints
        for a in range(self.n_anchors):
            if (node, f'a{a}') in self.distances:
                d_measured = self.distances[(node, f'a{a}')]
                anchor_pos = self.anchor_positions[a]
                
                # Current estimate
                x_current = self.beliefs[node]['mean']
                d_est = norm(x_current - anchor_pos)
                
                if d_est > 1e-6:
                    # Jacobian
                    H = (x_current - anchor_pos) / d_est
                    
                    # Add anchor information
                    sigma2 = (self.noise_factor * d_measured) ** 2
                    anchor_precision = np.outer(H, H) / sigma2
                    
                    # Innovation
                    innovation = (d_measured - d_est)
                    correction = innovation * H / d_est
                    
                    total_precision += anchor_precision
                    total_info_vector += anchor_precision @ (x_current + correction)
                    
        # Add messages from neighbors
        for neighbor in self.graph.neighbors(node):
            if (neighbor, node) in self.messages:
                msg = self.messages[(neighbor, node)]
                if np.any(msg['precision']):  # Non-zero message
                    total_precision += msg['precision']
                    total_info_vector += msg['mean']
                    
        # Compute new belief
        try:
            new_cov = inv(total_precision + np.eye(2) * 1e-6)
            new_mean = new_cov @ total_info_vector
        except:
            # Numerical issues, keep old belief
            return self.beliefs[node]['mean'], inv(self.beliefs[node]['precision'])
            
        # Apply damping
        old_mean = self.beliefs[node]['mean']
        old_cov = inv(self.beliefs[node]['precision'] + np.eye(2) * 1e-6)
        
        damped_mean = self.damping * old_mean + (1 - self.damping) * new_mean
        damped_cov = self.damping * old_cov + (1 - self.damping) * new_cov
        
        # Update precision
        damped_precision = inv(damped_cov + np.eye(2) * 1e-6)
        
        return damped_mean, damped_precision
        
    def run(self) -> Dict:
        """Run belief propagation"""
        # Initialize
        self.initialize_beliefs()
        
        errors = []
        for iteration in range(self.max_iter):
            # Store old beliefs for convergence check
            old_beliefs = {i: self.beliefs[i]['mean'].copy() for i in range(self.n_sensors)}
            
            # Message passing phase
            new_messages = {}
            for edge in self.edges:
                # Message from i to j
                msg_ij = self.compute_message(edge[0], edge[1])
                new_messages[(edge[0], edge[1])] = msg_ij
                
                # Message from j to i  
                msg_ji = self.compute_message(edge[1], edge[0])
                new_messages[(edge[1], edge[0])] = msg_ji
                
            self.messages = new_messages
            
            # Belief update phase
            for i in range(self.n_sensors):
                new_mean, new_precision = self.update_belief(i)
                self.beliefs[i] = {'mean': new_mean, 'precision': new_precision}
                
            # Check convergence
            max_change = max([norm(self.beliefs[i]['mean'] - old_beliefs[i]) 
                             for i in range(self.n_sensors)])
            
            # Calculate current error
            if self.true_positions:
                error = np.sqrt(np.mean([norm(self.beliefs[i]['mean'] - self.true_positions[i])**2 
                                         for i in range(self.n_sensors)]))
                errors.append(error)
                
            if max_change < self.tol:
                print(f"BP converged in {iteration} iterations")
                break
                
        # Extract final positions
        final_positions = {i: self.beliefs[i]['mean'] for i in range(self.n_sensors)}
        
        # Calculate final error
        if self.true_positions:
            final_error = np.sqrt(np.mean([norm(final_positions[i] - self.true_positions[i])**2 
                                           for i in range(self.n_sensors)]))
        else:
            final_error = 0
            
        return {
            'positions': final_positions,
            'final_error': final_error,
            'iterations': iteration + 1,
            'errors': errors,
            'beliefs': self.beliefs
        }