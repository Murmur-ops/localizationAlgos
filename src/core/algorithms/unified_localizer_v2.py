"""
Unified Localizer V2: Improved integration with smart fusion
"""

import numpy as np
from numpy.linalg import norm
from typing import Dict, Optional
import networkx as nx

from algorithms.smart_integrator import SmartIntegrator
from algorithms.estimate_fusion import EstimateWithUncertainty


class UnifiedLocalizerV2:
    def __init__(self, n_sensors: int, n_anchors: int, 
                 communication_range: float, noise_factor: float = 0.05):
        """
        Initialize improved unified localization system
        
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
        
        # Smart integration system
        self.integrator = SmartIntegrator(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            communication_range=communication_range,
            noise_factor=noise_factor
        )
        
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
                    
        # Initialize smart integrator
        self.integrator.initialize(
            graph=self.graph,
            distances=self.distances,
            anchor_positions=anchor_positions,
            true_positions=true_positions
        )
        
    def run(self, max_iter: int = 100, verbose: bool = True) -> Dict:
        """
        Run improved unified localization
        
        Args:
            max_iter: Maximum iterations
            verbose: Print progress
            
        Returns:
            Dictionary with results and metrics
        """
        if verbose:
            print("\n" + "="*50)
            print("Unified Localizer V2 - Smart Integration")
            print("="*50)
            
        # Run smart integrated localization
        result = self.integrator.run_smart_localization(max_iter=max_iter)
        
        # Calculate additional metrics
        positions = result['positions']
        
        # Per-node confidence scores
        confidence_scores = {}
        for node_id, estimate in result['estimates'].items():
            confidence_scores[node_id] = estimate.confidence
            
        avg_confidence = np.mean(list(confidence_scores.values()))
        
        # Calculate RMSE if ground truth available
        if self.true_positions:
            errors = []
            for i in range(self.n_sensors):
                if i in positions:
                    error = norm(positions[i] - self.true_positions[i])
                    errors.append(error)
            final_error = np.sqrt(np.mean(np.square(errors)))
        else:
            final_error = 0
            
        # Get analysis summary
        analysis = result['analysis']
        
        if verbose:
            print(f"\nResults:")
            print(f"  Final RMSE: {final_error:.4f}")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Fiedler value: {analysis.get('fiedler_value', 0):.3f}")
            
        return {
            'positions': positions,
            'final_error': final_error,
            'confidence_scores': confidence_scores,
            'average_confidence': avg_confidence,
            'estimates': result['estimates'],
            'analysis': analysis
        }
        
    def get_detailed_results(self) -> Dict:
        """Get detailed results for analysis"""
        if not hasattr(self.integrator, 'node_estimates'):
            return {}
            
        detailed = {}
        
        for node_id, estimates in self.integrator.node_estimates.items():
            node_detail = {
                'node_type': self.integrator.node_analyzer.get_node_type(node_id).value,
                'confidence': self.integrator.node_analyzer.get_node_confidence(node_id),
                'estimates': {}
            }
            
            for source, estimate in estimates.items():
                node_detail['estimates'][source] = {
                    'position': estimate.position.tolist(),
                    'confidence': estimate.confidence,
                    'convergence_rate': estimate.convergence_rate
                }
                
            if node_id in self.integrator.final_estimates:
                final = self.integrator.final_estimates[node_id]
                node_detail['final'] = {
                    'position': final.position.tolist(),
                    'confidence': final.confidence,
                    'source': final.source
                }
                
                # Calculate error if ground truth available
                if self.true_positions and node_id in self.true_positions:
                    error = norm(final.position - self.true_positions[node_id])
                    node_detail['error'] = error
                    
            detailed[node_id] = node_detail
            
        return detailed