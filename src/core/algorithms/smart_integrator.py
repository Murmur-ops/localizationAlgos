"""
Smart Integrator: Intelligent combination of localization algorithms
"""

import numpy as np
from numpy.linalg import norm
from typing import Dict, List, Optional, Tuple
import networkx as nx

from src.core.algorithms.node_analyzer import NodeAnalyzer, NodeType
from src.core.algorithms.estimate_fusion import EstimateFusion, EstimateWithUncertainty
from src.core.algorithms.bp_simple import SimpleBeliefPropagation
from src.core.algorithms.hierarchical_processing import HierarchicalProcessor
from src.core.algorithms.consensus_optimizer import ConsensusOptimizer
from src.core.algorithms.adaptive_weighting import AdaptiveWeighting


class SmartIntegrator:
    def __init__(self, n_sensors: int, n_anchors: int, 
                 communication_range: float, noise_factor: float):
        """
        Initialize smart integration system
        
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
        
        # Component systems
        self.node_analyzer = None
        self.estimate_fusion = EstimateFusion()
        self.adaptive_weights = None
        
        # Algorithm instances
        self.bp = None
        self.hierarchical = None
        self.consensus = None
        
        # State tracking
        self.node_estimates = {}  # {node_id: {algorithm: EstimateWithUncertainty}}
        self.final_estimates = {}  # {node_id: EstimateWithUncertainty}
        
    def initialize(self, graph: nx.Graph, distances: Dict, 
                  anchor_positions: np.ndarray, true_positions: Dict = None):
        """Initialize all components with network data"""
        # Analyze network
        self.node_analyzer = NodeAnalyzer(graph, self.n_anchors, 
                                         anchor_positions, distances)
        
        # Adaptive weighting
        self.adaptive_weights = AdaptiveWeighting(graph, self.n_anchors, 
                                                 anchor_positions)
        self.adaptive_weights.distances = distances
        
        # Initialize algorithms
        self.bp = SimpleBeliefPropagation(
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors,
            communication_range=self.communication_range,
            noise_factor=self.noise_factor,
            max_iter=100
        )
        
        self.hierarchical = HierarchicalProcessor(
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors,
            communication_range=self.communication_range,
            noise_factor=self.noise_factor
        )
        
        self.consensus = ConsensusOptimizer(
            graph=graph,
            n_sensors=self.n_sensors,
            n_anchors=self.n_anchors
        )
        
        # Store network data
        self.graph = graph
        self.distances = distances
        self.anchor_positions = anchor_positions
        self.true_positions = true_positions
        
    def run_smart_localization(self, max_iter: int = 100) -> Dict:
        """
        Run smart integrated localization
        
        Returns:
            Dictionary with final positions and metrics
        """
        print("Smart Integrated Localization")
        print("-" * 40)
        
        # Get processing strategy
        processing_tiers = self.node_analyzer.get_processing_order()
        analysis_summary = self.node_analyzer.get_analysis_summary()
        
        print(f"Network Analysis:")
        print(f"  Node types: {analysis_summary['node_type_distribution']}")
        print(f"  Average confidence: {analysis_summary['average_confidence']:.2f}")
        print(f"  Processing tiers: {len(processing_tiers)}")
        
        # Process each tier
        for tier_idx, tier_nodes in enumerate(processing_tiers):
            print(f"\nProcessing Tier {tier_idx + 1} ({len(tier_nodes)} nodes)")
            self._process_tier(tier_nodes, tier_idx)
            
        # Final fusion and refinement
        print("\nFinal fusion and refinement...")
        self._final_refinement(max_iter=20)
        
        # Extract final positions
        final_positions = {}
        for node_id, estimate in self.final_estimates.items():
            final_positions[node_id] = estimate.position
            
        # Calculate error if ground truth available
        final_error = 0
        if self.true_positions:
            errors = []
            for node_id in range(self.n_sensors):
                if node_id in final_positions:
                    error = norm(final_positions[node_id] - self.true_positions[node_id])
                    errors.append(error)
            final_error = np.sqrt(np.mean(np.square(errors)))
            
        return {
            'positions': final_positions,
            'final_error': final_error,
            'estimates': self.final_estimates,
            'analysis': analysis_summary
        }
        
    def _process_tier(self, nodes: List[int], tier_idx: int):
        """Process a tier of nodes with appropriate algorithms"""
        
        for node in nodes:
            node_type = self.node_analyzer.get_node_type(node)
            strategy = self.node_analyzer.get_processing_strategy(node)
            
            estimates = []
            
            # Run BP if weight > 0
            if strategy['bp_weight'] > 0:
                bp_est = self._run_bp_for_node(node, strategy)
                if bp_est:
                    estimates.append(bp_est)
                    
            # Run hierarchical if weight > 0 and not well-anchored
            if (strategy['hierarchical_weight'] > 0 and 
                node_type != NodeType.WELL_ANCHORED):
                hier_est = self._run_hierarchical_for_node(node, strategy)
                if hier_est:
                    estimates.append(hier_est)
                    
            # Run consensus if weight > 0 and has neighbors
            if (strategy['consensus_weight'] > 0 and 
                self.graph.degree(node) > 0):
                consensus_est = self._run_consensus_for_node(node, strategy)
                if consensus_est:
                    estimates.append(consensus_est)
                    
            # Fuse estimates
            if estimates:
                # Use weighted fusion based on strategy
                weights = {
                    'BP': strategy['bp_weight'],
                    'Hierarchical': strategy['hierarchical_weight'],
                    'Consensus': strategy['consensus_weight']
                }
                
                fused = self.estimate_fusion.weighted_fusion(estimates, weights)
                self.final_estimates[node] = fused
                
                # Store individual estimates for analysis
                self.node_estimates[node] = {e.source: e for e in estimates}
                
    def _run_bp_for_node(self, node: int, strategy: Dict) -> Optional[EstimateWithUncertainty]:
        """Run BP for a specific node"""
        # Generate network if not done
        if not hasattr(self.bp, 'graph') or self.bp.graph is None:
            self.bp.generate_network(self.true_positions, self.anchor_positions)
            
        # Run BP (simplified - would need node-specific in practice)
        result = self.bp.run()
        
        if node in result['positions']:
            # Estimate covariance based on convergence
            base_variance = (self.noise_factor * self.communication_range) ** 2
            covariance = np.eye(2) * base_variance
            
            # Confidence based on anchor coverage
            confidence = self.node_analyzer.get_node_confidence(node)
            
            return EstimateWithUncertainty(
                position=result['positions'][node],
                covariance=covariance,
                confidence=confidence * strategy['bp_weight'],
                source='BP',
                convergence_rate=0.9,
                iterations=result.get('iterations', 50)
            )
        return None
        
    def _run_hierarchical_for_node(self, node: int, strategy: Dict) -> Optional[EstimateWithUncertainty]:
        """Run hierarchical processing for a specific node"""
        if not hasattr(self.hierarchical, 'graph') or self.hierarchical.graph is None:
            self.hierarchical.generate_network(self.true_positions, self.anchor_positions)
            
        result = self.hierarchical.run(max_iter=strategy['iterations'])
        
        if node in result['positions']:
            # Higher uncertainty for hierarchical
            base_variance = (self.noise_factor * self.communication_range * 1.5) ** 2
            covariance = np.eye(2) * base_variance
            
            confidence = 0.7 * strategy['hierarchical_weight']
            
            return EstimateWithUncertainty(
                position=result['positions'][node],
                covariance=covariance,
                confidence=confidence,
                source='Hierarchical',
                convergence_rate=0.7,
                iterations=strategy['iterations']
            )
        return None
        
    def _run_consensus_for_node(self, node: int, strategy: Dict) -> Optional[EstimateWithUncertainty]:
        """Run consensus for a specific node"""
        # Start with current best estimate
        if node in self.final_estimates:
            initial_pos = {node: self.final_estimates[node].position}
        else:
            initial_pos = {node: np.random.uniform(0, 1, 2)}
            
        # Add neighbor positions
        for neighbor in self.graph.neighbors(node):
            if neighbor in self.final_estimates:
                initial_pos[neighbor] = self.final_estimates[neighbor].position
                
        # Run consensus
        K_rounds = min(50, int(strategy['iterations'] * 0.5))
        final_pos = self.consensus.adaptive_consensus(
            positions=initial_pos,
            measurements=self.distances,
            anchor_positions=self.anchor_positions,
            noise_factor=self.noise_factor,
            K=K_rounds
        )
        
        if node in final_pos:
            # Consensus typically has moderate uncertainty
            base_variance = (self.noise_factor * self.communication_range * 1.2) ** 2
            covariance = np.eye(2) * base_variance
            
            confidence = 0.8 * strategy['consensus_weight']
            
            return EstimateWithUncertainty(
                position=final_pos[node],
                covariance=covariance,
                confidence=confidence,
                source='Consensus',
                convergence_rate=0.8,
                iterations=K_rounds
            )
        return None
        
    def _final_refinement(self, max_iter: int = 20):
        """Final refinement pass using all information"""
        
        for iteration in range(max_iter):
            updates = {}
            
            for node in range(self.n_sensors):
                if node not in self.final_estimates:
                    # Initialize if needed
                    self.final_estimates[node] = EstimateWithUncertainty(
                        position=np.random.uniform(0.2, 0.8, 2),
                        covariance=np.eye(2) * 0.1,
                        confidence=0.1,
                        source='Init',
                        convergence_rate=0.0,
                        iterations=0
                    )
                    
                # Compute update based on measurements
                current_est = self.final_estimates[node]
                forces = []
                weights = []
                
                # Anchor constraints
                for a in range(self.n_anchors):
                    if (node, f'a{a}') in self.distances:
                        anchor_pos = self.anchor_positions[a]
                        measured_dist = self.distances[(node, f'a{a}')] 
                        current_dist = norm(current_est.position - anchor_pos)
                        
                        if current_dist > 1e-6:
                            direction = (current_est.position - anchor_pos) / current_dist
                            target = anchor_pos + direction * measured_dist
                            
                            # Adaptive weight
                            weight = self.adaptive_weights.get_anchor_weight(node, a)
                            forces.append(target)
                            weights.append(weight * current_est.confidence)
                            
                # Neighbor constraints
                for neighbor in self.graph.neighbors(node):
                    if neighbor in self.final_estimates:
                        key = (node, neighbor) if (node, neighbor) in self.distances else (neighbor, node)
                        if key in self.distances:
                            measured_dist = self.distances[key]
                            neighbor_pos = self.final_estimates[neighbor].position
                            current_dist = norm(current_est.position - neighbor_pos)
                            
                            if current_dist > 1e-6:
                                direction = (current_est.position - neighbor_pos) / current_dist
                                target = neighbor_pos + direction * measured_dist
                                
                                # Adaptive weight
                                weight = self.adaptive_weights.get_edge_weight(node, neighbor)
                                neighbor_conf = self.final_estimates[neighbor].confidence
                                forces.append(target)
                                weights.append(weight * neighbor_conf * 0.5)
                                
                # Compute update
                if forces:
                    weights = np.array(weights)
                    weights /= weights.sum() + 1e-10
                    
                    new_pos = np.zeros(2)
                    for f, w in zip(forces, weights):
                        new_pos += w * f
                        
                    # Adaptive damping
                    damping = self.adaptive_weights.get_damping_factor()
                    updated_pos = damping * current_est.position + (1 - damping) * new_pos
                    updated_pos = np.clip(updated_pos, 0, 1)
                    
                    # Update estimate
                    updates[node] = EstimateWithUncertainty(
                        position=updated_pos,
                        covariance=current_est.covariance * 0.95,  # Reduce uncertainty
                        confidence=min(1.0, current_est.confidence * 1.05),  # Increase confidence
                        source='Refined',
                        convergence_rate=current_est.convergence_rate,
                        iterations=current_est.iterations + 1
                    )
                    
            # Apply updates
            for node, update in updates.items():
                self.final_estimates[node] = update
                
            # Check convergence
            if iteration > 5:
                max_change = max([norm(updates[n].position - self.final_estimates[n].position) 
                                for n in updates.keys()] + [0])
                if max_change < 1e-5:
                    break