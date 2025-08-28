"""
Consensus Optimization with Acceleration
Implements accelerated consensus for distributed localization
"""

import numpy as np
from numpy.linalg import norm, eigh
from typing import Dict, List, Optional
import networkx as nx


class ConsensusOptimizer:
    def __init__(self, graph: nx.Graph, n_sensors: int, n_anchors: int):
        """
        Initialize consensus optimizer
        
        Args:
            graph: Network connectivity graph  
            n_sensors: Number of sensors
            n_anchors: Number of anchors
        """
        self.graph = graph
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        
        # Consensus parameters
        self.weight_matrix = None
        self.momentum = 0.9  # Nesterov momentum
        self.step_size = None
        
        self._initialize_consensus_weights()
        
    def _initialize_consensus_weights(self):
        """Initialize optimized consensus weight matrix"""
        n = self.n_sensors
        W = np.zeros((n, n))
        
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        
        # Compute degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        
        # Metropolis-Hastings weights (optimal for consensus)
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self weight
                    W[i, i] = 1.0
                    for k in range(n):
                        if adj_matrix[i, k] > 0 and k != i:
                            W[i, i] -= 1.0 / (max(degrees[i], degrees[k]) + 1)
                elif adj_matrix[i, j] > 0:
                    # Neighbor weight
                    W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1)
                    
        # Ensure doubly stochastic
        W = self._make_doubly_stochastic(W)
        
        # Compute optimal step size from spectral radius
        eigenvalues = np.linalg.eigvals(W - np.eye(n))
        spectral_radius = np.max(np.abs(eigenvalues))
        self.step_size = 1.0 / (1.0 + spectral_radius) if spectral_radius > 0 else 0.5
        
        self.weight_matrix = W
        
    def _make_doubly_stochastic(self, W: np.ndarray) -> np.ndarray:
        """Ensure weight matrix is doubly stochastic"""
        n = W.shape[0]
        
        # Sinkhorn-Knopp algorithm
        for _ in range(10):
            # Normalize rows
            row_sums = W.sum(axis=1)
            W = W / (row_sums[:, np.newaxis] + 1e-10)
            
            # Normalize columns  
            col_sums = W.sum(axis=0)
            W = W / (col_sums + 1e-10)
            
        return W
        
    def consensus_update(self, positions: Dict[int, np.ndarray], 
                        velocities: Optional[Dict[int, np.ndarray]] = None,
                        use_acceleration: bool = True) -> Dict[int, np.ndarray]:
        """
        Perform consensus update with optional acceleration
        
        Args:
            positions: Current position estimates
            velocities: Current velocities (for momentum)
            use_acceleration: Use Nesterov acceleration
            
        Returns:
            Updated positions
        """
        n = self.n_sensors
        
        # Convert to matrix form
        X = np.zeros((n, 2))
        for i in range(n):
            if i in positions:
                X[i] = positions[i]
                
        # Initialize velocities if needed
        if velocities is None:
            velocities = {i: np.zeros(2) for i in range(n)}
            
        V = np.zeros((n, 2))
        for i in range(n):
            if i in velocities:
                V[i] = velocities[i]
                
        if use_acceleration:
            # Nesterov accelerated consensus
            # Look-ahead position
            Y = X + self.momentum * V
            
            # Consensus step on look-ahead
            X_consensus = self.weight_matrix @ Y
            
            # Update velocity
            V_new = self.momentum * V + self.step_size * (X_consensus - Y)
            
            # Update position
            X_new = X + V_new
            
            # Store velocities for next iteration
            new_velocities = {i: V_new[i] for i in range(n)}
            
        else:
            # Standard consensus
            X_new = self.weight_matrix @ X
            new_velocities = velocities
            
        # Convert back to dict
        new_positions = {i: X_new[i] for i in range(n)}
        
        return new_positions, new_velocities
        
    def distributed_gradient_consensus(self, positions: Dict[int, np.ndarray],
                                      gradients: Dict[int, np.ndarray],
                                      K: int = 50) -> Dict[int, np.ndarray]:
        """
        Distributed gradient consensus
        Combines local gradients with consensus
        
        Args:
            positions: Current positions
            gradients: Local gradients at each node
            K: Number of consensus rounds
            
        Returns:
            Consensus positions
        """
        n = self.n_sensors
        
        # Initialize
        X = np.zeros((n, 2))
        G = np.zeros((n, 2))
        
        for i in range(n):
            if i in positions:
                X[i] = positions[i]
            if i in gradients:
                G[i] = gradients[i]
                
        # Gradient tracking variables
        Y = X.copy()  # Gradient tracker
        S = G.copy()  # Gradient sum tracker
        
        # Learning rate schedule
        alpha_0 = 0.01
        
        for k in range(K):
            # Decaying step size
            alpha = alpha_0 / (1 + k * 0.01)
            
            # Gradient tracking consensus
            Y_new = self.weight_matrix @ Y
            S_new = self.weight_matrix @ S
            
            # Position update with tracked gradient
            X_new = self.weight_matrix @ X - alpha * S_new
            
            # Update trackers
            Y = Y_new
            S = S_new
            X = X_new
            
            # Project to feasible region [0, 1]²
            X = np.clip(X, 0, 1)
            
        return {i: X[i] for i in range(n)}
        
    def adaptive_consensus(self, positions: Dict[int, np.ndarray],
                          measurements: Dict, anchor_positions: np.ndarray,
                          noise_factor: float, K: int = 100) -> Dict[int, np.ndarray]:
        """
        Adaptive consensus with measurement incorporation
        
        Args:
            positions: Initial positions
            measurements: Distance measurements
            anchor_positions: Anchor positions
            noise_factor: Noise level
            K: Number of rounds
            
        Returns:
            Final positions
        """
        velocities = {i: np.zeros(2) for i in range(self.n_sensors)}
        
        for round in range(K):
            # Compute local gradients based on measurements
            gradients = self._compute_measurement_gradients(
                positions, measurements, anchor_positions, noise_factor
            )
            
            # Consensus step with acceleration
            positions, velocities = self.consensus_update(
                positions, velocities, use_acceleration=True
            )
            
            # Gradient correction step
            step_size = 0.01 / (1 + round * 0.001)
            for i in range(self.n_sensors):
                if i in gradients:
                    positions[i] -= step_size * gradients[i]
                    positions[i] = np.clip(positions[i], 0, 1)
                    
            # Check convergence
            if round > 10 and round % 10 == 0:
                max_vel = max([norm(v) for v in velocities.values()])
                if max_vel < 1e-5:
                    break
                    
        return positions
        
    def _compute_measurement_gradients(self, positions: Dict, measurements: Dict,
                                      anchor_positions: np.ndarray, 
                                      noise_factor: float) -> Dict:
        """Compute gradients from measurement errors"""
        gradients = {}
        
        for i in range(self.n_sensors):
            if i not in positions:
                continue
                
            grad = np.zeros(2)
            count = 0
            
            # Anchor measurements
            for a in range(self.n_anchors):
                key = (i, f'a{a}')
                if key in measurements:
                    measured_dist = measurements[key]
                    anchor_pos = anchor_positions[a]
                    
                    diff = positions[i] - anchor_pos
                    actual_dist = norm(diff)
                    
                    if actual_dist > 1e-6:
                        # Gradient of distance error
                        error = actual_dist - measured_dist
                        weight = 1.0 / (noise_factor * measured_dist + 0.01)
                        grad += weight * error * diff / actual_dist
                        count += 1
                        
            # Neighbor measurements
            for j in self.graph.neighbors(i):
                if j not in positions:
                    continue
                    
                if (i, j) in measurements:
                    measured_dist = measurements[(i, j)]
                elif (j, i) in measurements:
                    measured_dist = measurements[(j, i)]
                else:
                    continue
                    
                diff = positions[i] - positions[j]
                actual_dist = norm(diff)
                
                if actual_dist > 1e-6:
                    error = actual_dist - measured_dist
                    weight = 0.5 / (noise_factor * measured_dist + 0.01)
                    grad += weight * error * diff / actual_dist
                    count += 1
                    
            if count > 0:
                gradients[i] = grad / count
            else:
                gradients[i] = np.zeros(2)
                
        return gradients
        
    def get_consensus_parameters(self) -> Dict:
        """Get consensus algorithm parameters"""
        # Analyze weight matrix
        eigenvalues = np.linalg.eigvals(self.weight_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # Second largest eigenvalue determines convergence rate
        if len(eigenvalues) > 1:
            convergence_rate = eigenvalues[1]
        else:
            convergence_rate = 0.5
            
        return {
            'step_size': self.step_size,
            'momentum': self.momentum,
            'convergence_rate': convergence_rate,
            'spectral_gap': 1.0 - convergence_rate,
            'weight_matrix_norm': np.linalg.norm(self.weight_matrix)
        }