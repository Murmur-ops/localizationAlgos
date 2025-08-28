"""
Unified Localizer: Integration of all advanced methods
Combines BP, Hierarchical, Adaptive Weighting, and Consensus
"""

import numpy as np
from numpy.linalg import norm
from typing import Dict, Optional
import networkx as nx

from algorithms.bp_simple import SimpleBeliefPropagation
from algorithms.hierarchical_processing import HierarchicalProcessor
from algorithms.adaptive_weighting import AdaptiveWeighting
from algorithms.consensus_optimizer import ConsensusOptimizer


class UnifiedLocalizer:
    def __init__(self, n_sensors: int, n_anchors: int, communication_range: float,
                 noise_factor: float = 0.05):
        """
        Initialize unified localization system
        
        Args:
            n_sensors: Number of sensors
            n_anchors: Number of anchors  
            communication_range: Communication range
            noise_factor: Measurement noise level
        """
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        self.communication_range = communication_range
        self.noise_factor = noise_factor
        
        # Component algorithms
        self.bp = None
        self.hierarchical = None
        self.adaptive = None
        self.consensus = None
        
        # Network data
        self.graph = None
        self.distances = {}
        self.anchor_positions = None
        self.true_positions = None
        
    def generate_network(self, true_positions: Dict, anchor_positions: np.ndarray):
        """Build network and initialize all components"""
        self.true_positions = true_positions
        self.anchor_positions = anchor_positions
        
        # Build graph
        self.graph = nx.Graph()
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
                    
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all algorithm components"""
        # Adaptive weighting system
        self.adaptive = AdaptiveWeighting(self.graph, self.n_anchors, self.anchor_positions)
        self.adaptive.distances = self.distances  # Share measurements
        
        # Get adaptive parameters
        params = self.adaptive.get_algorithm_parameters()
        
        # Simple BP with adaptive parameters
        self.bp = SimpleBeliefPropagation(
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors,
            communication_range=self.communication_range,
            noise_factor=self.noise_factor,
            max_iter=params['max_iterations']
        )
        
        # Hierarchical processor
        self.hierarchical = HierarchicalProcessor(
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors,
            communication_range=self.communication_range,
            noise_factor=self.noise_factor,
            n_tiers=3
        )
        
        # Consensus optimizer
        self.consensus = ConsensusOptimizer(
            graph=self.graph,
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors
        )
        
    def run(self, max_iter: int = 100) -> Dict:
        """
        Run unified localization
        
        Returns:
            Dictionary with results
        """
        print("Running Unified Localizer...")
        
        # Phase 1: Initial estimate with BP
        print("  Phase 1: Belief Propagation...")
        self.bp.generate_network(self.true_positions, self.anchor_positions)
        bp_result = self.bp.run()
        positions = bp_result['positions']
        
        # Phase 2: Hierarchical refinement
        print("  Phase 2: Hierarchical Processing...")
        self.hierarchical.generate_network(self.true_positions, self.anchor_positions)
        
        # Use BP result as initialization
        self.hierarchical.positions = positions
        hier_result = self.hierarchical.run(max_iter=30)
        positions = hier_result['positions']
        
        # Phase 3: Consensus optimization with adaptive weights
        print("  Phase 3: Adaptive Consensus...")
        
        # Get adaptive parameters
        params = self.adaptive.get_algorithm_parameters()
        damping = params['damping']
        
        # Run adaptive consensus
        consensus_positions = self.consensus.adaptive_consensus(
            positions=positions,
            measurements=self.distances,
            anchor_positions=self.anchor_positions,
            noise_factor=self.noise_factor,
            K=50  # Bounded rounds
        )
        
        # Phase 4: Final refinement with all methods
        print("  Phase 4: Final Refinement...")
        positions = self._unified_refinement(consensus_positions, max_iter=20)
        
        # Calculate final error
        if self.true_positions:
            errors = [norm(positions[i] - self.true_positions[i]) 
                     for i in range(self.n_sensors)]
            final_error = np.sqrt(np.mean(np.square(errors)))
        else:
            final_error = 0
            
        return {
            'positions': positions,
            'final_error': final_error,
            'bp_error': bp_result['final_error'],
            'hierarchical_error': hier_result['final_error'],
            'fiedler_value': self.adaptive.fiedler_value,
            'convergence_rate': params['convergence_rate']
        }
        
    def _unified_refinement(self, positions: Dict, max_iter: int = 20) -> Dict:
        """Final refinement using all methods together"""
        velocities = {i: np.zeros(2) for i in range(self.n_sensors)}
        
        for iteration in range(max_iter):
            old_positions = {i: pos.copy() for i, pos in positions.items()}
            
            # Compute weighted updates from each sensor
            for i in range(self.n_sensors):
                if i not in positions:
                    positions[i] = np.random.uniform(0.2, 0.8, 2)
                    
                forces = []
                weights = []
                
                # Anchor constraints with adaptive weights
                for a in range(self.n_anchors):
                    if (i, f'a{a}') in self.distances:
                        anchor_pos = self.anchor_positions[a]
                        measured_dist = self.distances[(i, f'a{a}')] 
                        current_dist = norm(positions[i] - anchor_pos)
                        
                        if current_dist > 1e-6:
                            direction = (positions[i] - anchor_pos) / current_dist
                            target = anchor_pos + direction * measured_dist
                            
                            # Use adaptive anchor weight
                            anchor_weight = self.adaptive.get_anchor_weight(i, a)
                            forces.append(target)
                            weights.append(anchor_weight)
                            
                # Neighbor constraints with adaptive edge weights
                for j in self.graph.neighbors(i):
                    if j in positions:
                        if (i, j) in self.distances:
                            measured_dist = self.distances[(i, j)]
                        else:
                            measured_dist = self.distances[(j, i)]
                            
                        neighbor_pos = positions[j]
                        current_dist = norm(positions[i] - neighbor_pos)
                        
                        if current_dist > 1e-6:
                            direction = (positions[i] - neighbor_pos) / current_dist
                            target = neighbor_pos + direction * measured_dist
                            
                            # Use adaptive edge weight
                            edge_weight = self.adaptive.get_edge_weight(i, j)
                            forces.append(target)
                            weights.append(edge_weight)
                            
                # Weighted average with adaptive damping
                if forces:
                    weights = np.array(weights)
                    weights /= weights.sum()
                    new_pos = np.zeros(2)
                    for f, w in zip(forces, weights):
                        new_pos += w * f
                        
                    # Adaptive damping
                    damping = self.adaptive.get_damping_factor()
                    positions[i] = damping * positions[i] + (1 - damping) * new_pos
                    positions[i] = np.clip(positions[i], 0, 1)
                    
            # Apply consensus step every few iterations
            if iteration % 3 == 0:
                positions, velocities = self.consensus.consensus_update(
                    positions, velocities, use_acceleration=True
                )
                
            # Check convergence
            max_change = max([norm(positions[i] - old_positions[i]) 
                            for i in range(self.n_sensors)])
            
            if max_change < 1e-5:
                break
                
        return positions