#!/usr/bin/env python3
"""
Distributed Distance Consensus Protocol

Implements a fully distributed protocol for finding the maximum measured distance
in a sensor network, allowing for practical normalization without knowing true positions.

Real-world protocol:
1. Nodes discover neighbors through broadcasts
2. Perform TWTT to measure distances
3. Find local maximum distances
4. Use consensus to determine global maximum
5. All nodes normalize by the agreed maximum
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import logging
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class DistanceConsensusState:
    """State for distributed distance consensus"""
    node_id: int
    local_max_distance: float = 0.0
    consensus_max_distance: float = 0.0
    measured_distances: Dict[int, float] = field(default_factory=dict)
    neighbor_max_distances: Dict[int, float] = field(default_factory=dict)
    iteration: int = 0
    converged: bool = False
    
    def update_local_max(self):
        """Update local maximum from measured distances"""
        if self.measured_distances:
            self.local_max_distance = max(self.measured_distances.values())
        return self.local_max_distance


class DistanceConsensus:
    """
    Distributed consensus protocol for finding maximum network distance
    
    This allows practical normalization in real deployments where true
    positions are unknown. Each node only needs local measurements.
    """
    
    def __init__(self, n_nodes: int, adjacency_matrix: np.ndarray,
                 mixing_parameter: float = 0.5,
                 convergence_threshold: float = 0.001):
        """
        Initialize distance consensus protocol
        
        Args:
            n_nodes: Number of nodes in network
            adjacency_matrix: Network connectivity (1 if connected, 0 otherwise)
            mixing_parameter: Weight for consensus updates (0-1)
            convergence_threshold: Threshold for convergence detection
        """
        self.n_nodes = n_nodes
        self.adjacency = adjacency_matrix
        self.mixing_parameter = mixing_parameter
        self.convergence_threshold = convergence_threshold
        
        # Initialize states for all nodes
        self.states = {i: DistanceConsensusState(i) for i in range(n_nodes)}
        
        # Track global metrics
        self.global_max_distance = 0.0
        self.consensus_achieved = False
        self.iterations_to_converge = 0
        
        logger.info(f"DistanceConsensus initialized: {n_nodes} nodes")
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get list of neighbors for a node"""
        neighbors = []
        for j in range(self.n_nodes):
            if j != node_id and self.adjacency[node_id, j] > 0:
                neighbors.append(j)
        return neighbors
    
    def add_distance_measurement(self, node1: int, node2: int, distance: float):
        """
        Add a distance measurement between two nodes
        
        In practice, this would come from TWTT or other ranging method
        """
        self.states[node1].measured_distances[node2] = distance
        self.states[node2].measured_distances[node1] = distance
        
        # Update local maximums
        self.states[node1].update_local_max()
        self.states[node2].update_local_max()
    
    def consensus_iteration(self) -> float:
        """
        Perform one iteration of max-consensus
        
        Each node:
        1. Shares its current max with neighbors
        2. Updates its estimate to the maximum seen
        3. Applies mixing for stability
        
        Returns:
            Maximum change in consensus values
        """
        new_max_estimates = {}
        max_change = 0.0
        
        for node_id in range(self.n_nodes):
            state = self.states[node_id]
            neighbors = self.get_neighbors(node_id)
            
            if not neighbors:
                # Isolated node keeps its value
                new_max_estimates[node_id] = state.consensus_max_distance
                continue
            
            # Collect maximum values from neighbors
            neighbor_maxes = [state.local_max_distance]  # Include own measurement
            
            for neighbor_id in neighbors:
                neighbor_state = self.states[neighbor_id]
                # Get neighbor's current consensus estimate
                neighbor_max = neighbor_state.consensus_max_distance
                if neighbor_max == 0:  # Not initialized yet
                    neighbor_max = neighbor_state.local_max_distance
                neighbor_maxes.append(neighbor_max)
                
                # Track what we learned from this neighbor
                state.neighbor_max_distances[neighbor_id] = neighbor_max
            
            # Consensus step: take maximum (not average for max-finding)
            observed_max = max(neighbor_maxes)
            
            # Apply mixing parameter for stability
            if state.consensus_max_distance == 0:
                # First iteration - take the observed max directly
                new_max = observed_max
            else:
                # Mix with previous estimate for stability
                new_max = max(
                    state.consensus_max_distance,  # Keep if already larger
                    (1 - self.mixing_parameter) * state.consensus_max_distance + 
                    self.mixing_parameter * observed_max
                )
            
            # Track convergence
            change = abs(new_max - state.consensus_max_distance)
            max_change = max(max_change, change)
            
            new_max_estimates[node_id] = new_max
            
            logger.debug(f"Node {node_id}: max {state.consensus_max_distance:.3f} -> "
                        f"{new_max:.3f} (local={state.local_max_distance:.3f})")
        
        # Apply updates
        for node_id, new_max in new_max_estimates.items():
            self.states[node_id].consensus_max_distance = new_max
            self.states[node_id].iteration += 1
        
        return max_change
    
    def run_consensus(self, max_iterations: int = 100) -> Dict:
        """
        Run consensus protocol to convergence
        
        Returns:
            Dictionary with consensus results
        """
        logger.info("Starting distance consensus protocol")
        
        # Initialize consensus values with local measurements
        for node_id in range(self.n_nodes):
            state = self.states[node_id]
            state.consensus_max_distance = state.local_max_distance
        
        # Run consensus iterations
        convergence_history = []
        
        for iteration in range(max_iterations):
            max_change = self.consensus_iteration()
            convergence_history.append(max_change)
            
            # Check convergence
            if max_change < self.convergence_threshold:
                self.consensus_achieved = True
                self.iterations_to_converge = iteration + 1
                logger.info(f"Consensus achieved in {iteration + 1} iterations")
                break
            
            if iteration % 10 == 0:
                # Log progress
                current_estimates = [s.consensus_max_distance for s in self.states.values()]
                logger.debug(f"Iteration {iteration}: max_change={max_change:.6f}, "
                           f"estimates range=[{min(current_estimates):.3f}, "
                           f"{max(current_estimates):.3f}]")
        
        # Extract final consensus value (should be same for all nodes)
        final_estimates = [s.consensus_max_distance for s in self.states.values()]
        self.global_max_distance = max(final_estimates)
        
        # Check agreement
        estimate_variance = np.var(final_estimates)
        all_agree = estimate_variance < self.convergence_threshold
        
        results = {
            'converged': self.consensus_achieved,
            'iterations': self.iterations_to_converge,
            'global_max_distance': self.global_max_distance,
            'all_nodes_agree': all_agree,
            'estimate_variance': estimate_variance,
            'convergence_history': convergence_history,
            'final_estimates': final_estimates
        }
        
        logger.info(f"Consensus complete: max_distance={self.global_max_distance:.3f}, "
                   f"converged={self.consensus_achieved}, all_agree={all_agree}")
        
        return results
    
    def get_normalization_factors(self) -> Dict[int, float]:
        """
        Get normalization factor for each node
        
        Returns:
            Dictionary mapping node_id to its normalization factor
        """
        factors = {}
        for node_id in range(self.n_nodes):
            consensus_max = self.states[node_id].consensus_max_distance
            if consensus_max > 0:
                factors[node_id] = 1.0 / consensus_max
            else:
                factors[node_id] = 1.0
        return factors
    
    def normalize_distances(self) -> Dict[Tuple[int, int], float]:
        """
        Return all distances normalized by consensus maximum
        
        Returns:
            Dictionary of normalized distances
        """
        normalized = {}
        
        for node_id in range(self.n_nodes):
            state = self.states[node_id]
            norm_factor = 1.0 / self.global_max_distance if self.global_max_distance > 0 else 1.0
            
            for neighbor_id, distance in state.measured_distances.items():
                key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if key not in normalized:
                    normalized[key] = distance * norm_factor
        
        return normalized


def simulate_distance_consensus(n_sensors: int = 20, 
                               communication_range: float = 0.35,
                               seed: int = 42) -> Dict:
    """
    Simulate the distance consensus protocol with a random network
    
    This demonstrates how the protocol would work in practice
    """
    np.random.seed(seed)
    
    # Generate random positions (unknown to nodes!)
    true_positions = np.random.uniform(0, 1, (n_sensors, 2))
    
    # Build adjacency based on communication range
    adjacency = np.zeros((n_sensors, n_sensors))
    true_distances = {}
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= communication_range:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                # Add noise to simulate measurement error
                measured_dist = true_dist * (1 + 0.01 * np.random.randn())
                true_distances[(i, j)] = measured_dist
    
    # Initialize consensus protocol
    consensus = DistanceConsensus(n_sensors, adjacency)
    
    # Add distance measurements (simulating TWTT)
    for (i, j), dist in true_distances.items():
        consensus.add_distance_measurement(i, j, dist)
    
    # Run consensus
    results = consensus.run_consensus()
    
    # Compare with true maximum
    true_max = 0
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_max = max(true_max, np.linalg.norm(true_positions[i] - true_positions[j]))
    
    results['true_max_distance'] = true_max
    results['error_percentage'] = abs(results['global_max_distance'] - true_max) / true_max * 100
    
    logger.info(f"Consensus max: {results['global_max_distance']:.3f}, "
               f"True max: {true_max:.3f}, "
               f"Error: {results['error_percentage']:.1f}%")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("DISTRIBUTED DISTANCE CONSENSUS PROTOCOL")
    print("="*60)
    
    # Run simulation
    results = simulate_distance_consensus(n_sensors=20)
    
    print(f"\nResults:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Consensus max distance: {results['global_max_distance']:.3f}")
    print(f"  True max distance: {results['true_max_distance']:.3f}")
    print(f"  Error: {results['error_percentage']:.1f}%")
    print(f"  All nodes agree: {results['all_nodes_agree']}")
    
    print("\n✓ Protocol enables distributed normalization without knowing true positions!")