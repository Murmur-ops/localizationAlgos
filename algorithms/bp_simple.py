"""
Simplified Stable Belief Propagation for Sensor Localization
"""

import numpy as np
from numpy.linalg import norm
from typing import Dict
import networkx as nx


class SimpleBeliefPropagation:
    def __init__(self, n_sensors: int, n_anchors: int, communication_range: float,
                 noise_factor: float = 0.05, max_iter: int = 50):
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        self.communication_range = communication_range
        self.noise_factor = noise_factor
        self.max_iter = max_iter
        
        # Positions and measurements
        self.positions = {}  # Current position estimates
        self.distances = {}
        self.anchor_positions = None
        self.true_positions = None
        self.graph = None
        
    def generate_network(self, true_positions: Dict, anchor_positions: np.ndarray):
        """Build network from positions"""
        self.true_positions = true_positions
        self.anchor_positions = anchor_positions
        self.graph = nx.Graph()
        
        # Add nodes
        for i in range(self.n_sensors):
            self.graph.add_node(i)
            
        # Add edges and measurements
        for i in range(self.n_sensors):
            for j in range(i+1, self.n_sensors):
                dist = norm(true_positions[i] - true_positions[j])
                if dist <= self.communication_range:
                    self.graph.add_edge(i, j)
                    noise = np.random.normal(0, self.noise_factor * dist)
                    self.distances[(i, j)] = dist + noise
                    self.distances[(j, i)] = dist + noise
                    
            # Anchor measurements
            for a in range(self.n_anchors):
                dist = norm(true_positions[i] - anchor_positions[a])
                if dist <= self.communication_range:
                    noise = np.random.normal(0, self.noise_factor * dist)
                    self.distances[(i, f'a{a}')] = dist + noise
                    
    def initialize_positions(self):
        """Initialize with simple heuristic"""
        for i in range(self.n_sensors):
            # Check anchors
            anchor_dists = []
            for a in range(self.n_anchors):
                if (i, f'a{a}') in self.distances:
                    anchor_dists.append((a, self.distances[(i, f'a{a}')]))
            
            if anchor_dists:
                # Place near first anchor with some noise
                a_idx, d = anchor_dists[0]
                angle = np.random.uniform(0, 2*np.pi)
                self.positions[i] = self.anchor_positions[a_idx] + \
                                   d * np.array([np.cos(angle), np.sin(angle)])
                self.positions[i] = np.clip(self.positions[i], 0, 1)
            else:
                # Random initialization
                self.positions[i] = np.random.uniform(0.2, 0.8, 2)
                
    def update_position(self, node: int) -> np.ndarray:
        """Update position using weighted average of constraints"""
        forces = []
        weights = []
        
        # Anchor constraints
        for a in range(self.n_anchors):
            if (node, f'a{a}') in self.distances:
                anchor_pos = self.anchor_positions[a]
                measured_dist = self.distances[(node, f'a{a}')]
                current_dist = norm(self.positions[node] - anchor_pos)
                
                if current_dist > 1e-6:
                    # Direction from anchor to node
                    direction = (self.positions[node] - anchor_pos) / current_dist
                    # Target position
                    target = anchor_pos + direction * measured_dist
                    forces.append(target)
                    # Weight inversely proportional to measurement variance
                    weights.append(1.0 / (self.noise_factor * measured_dist + 0.01))
        
        # Neighbor constraints
        for neighbor in self.graph.neighbors(node):
            if (node, neighbor) in self.distances:
                measured_dist = self.distances[(node, neighbor)]
            else:
                measured_dist = self.distances[(neighbor, node)]
                
            neighbor_pos = self.positions[neighbor]
            current_dist = norm(self.positions[node] - neighbor_pos)
            
            if current_dist > 1e-6:
                # Direction from neighbor to node
                direction = (self.positions[node] - neighbor_pos) / current_dist
                # Target position
                target = neighbor_pos + direction * measured_dist
                forces.append(target)
                # Lower weight for neighbor constraints
                weights.append(0.5 / (self.noise_factor * measured_dist + 0.01))
        
        if forces:
            # Weighted average
            weights = np.array(weights)
            weights /= weights.sum()
            new_pos = np.zeros(2)
            for f, w in zip(forces, weights):
                new_pos += w * f
            
            # Damping for stability
            damping = 0.7
            new_pos = damping * self.positions[node] + (1 - damping) * new_pos
            
            # Keep in bounds
            new_pos = np.clip(new_pos, 0, 1)
            return new_pos
        else:
            return self.positions[node]
            
    def run(self) -> Dict:
        """Run simple BP"""
        self.initialize_positions()
        
        errors = []
        for iteration in range(self.max_iter):
            # Random order update
            update_order = np.random.permutation(self.n_sensors)
            
            max_change = 0
            for node in update_order:
                old_pos = self.positions[node].copy()
                self.positions[node] = self.update_position(node)
                max_change = max(max_change, norm(self.positions[node] - old_pos))
            
            # Calculate error
            if self.true_positions:
                error = np.sqrt(np.mean([norm(self.positions[i] - self.true_positions[i])**2 
                                         for i in range(self.n_sensors)]))
                errors.append(error)
                
            # Check convergence
            if max_change < 1e-4:
                break
                
        # Final error
        final_error = 0
        if self.true_positions:
            final_error = np.sqrt(np.mean([norm(self.positions[i] - self.true_positions[i])**2 
                                           for i in range(self.n_sensors)]))
            
        return {
            'positions': self.positions,
            'final_error': final_error,
            'iterations': iteration + 1,
            'errors': errors
        }