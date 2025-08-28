"""
Node Analyzer: Classify nodes for optimal algorithm selection
"""

import numpy as np
from numpy.linalg import norm, eigh
from typing import Dict, List, Set, Tuple
import networkx as nx
from enum import Enum


class NodeType(Enum):
    """Node classification based on network properties"""
    WELL_ANCHORED = "well_anchored"  # ≥2 anchors, good connectivity
    BOTTLENECK = "bottleneck"        # Critical for network connectivity
    ISOLATED = "isolated"            # Poor connectivity, few neighbors
    BRIDGE = "bridge"                # Connects clusters
    NORMAL = "normal"                # Standard node


class NodeAnalyzer:
    def __init__(self, graph: nx.Graph, n_anchors: int, 
                 anchor_positions: np.ndarray, distances: Dict):
        """
        Initialize node analyzer
        
        Args:
            graph: Network connectivity graph
            n_anchors: Number of anchors
            anchor_positions: Anchor positions
            distances: Distance measurements
        """
        self.graph = graph
        self.n_anchors = n_anchors
        self.anchor_positions = anchor_positions
        self.distances = distances
        
        # Analysis results
        self.node_types = {}
        self.node_scores = {}
        self.fiedler_vector = None
        self.betweenness = {}
        self.anchor_coverage = {}
        
        # Perform analysis
        self._analyze_network()
        
    def _analyze_network(self):
        """Comprehensive network analysis"""
        # Compute Fiedler vector (algebraic connectivity)
        self._compute_fiedler()
        
        # Compute betweenness centrality
        self.betweenness = nx.betweenness_centrality(self.graph)
        
        # Analyze anchor coverage
        self._analyze_anchor_coverage()
        
        # Classify each node
        self._classify_nodes()
        
    def _compute_fiedler(self):
        """Compute Fiedler eigenvalue and eigenvector"""
        L = nx.laplacian_matrix(self.graph).toarray()
        n = self.graph.number_of_nodes()
        
        if n < 2:
            self.fiedler_vector = np.ones(n)
            return
            
        try:
            eigenvalues, eigenvectors = eigh(L)
            idx = np.argsort(eigenvalues)
            
            # Second smallest eigenvalue and its eigenvector
            if len(idx) > 1:
                self.fiedler_value = eigenvalues[idx[1]]
                self.fiedler_vector = np.abs(eigenvectors[:, idx[1]])
            else:
                self.fiedler_value = 1.0
                self.fiedler_vector = np.ones(n)
        except:
            self.fiedler_value = 1.0
            self.fiedler_vector = np.ones(n)
            
    def _analyze_anchor_coverage(self):
        """Analyze anchor measurement coverage for each node"""
        for node in self.graph.nodes():
            anchor_count = 0
            anchor_quality = 0
            
            for a in range(self.n_anchors):
                if (node, f'a{a}') in self.distances:
                    anchor_count += 1
                    # Quality inversely proportional to distance
                    dist = self.distances[(node, f'a{a}')]
                    anchor_quality += 1.0 / (dist + 0.1)
                    
            self.anchor_coverage[node] = {
                'count': anchor_count,
                'quality': anchor_quality
            }
            
    def _classify_nodes(self):
        """Classify each node based on its properties"""
        for node in self.graph.nodes():
            # Get node properties
            degree = self.graph.degree(node)
            anchor_count = self.anchor_coverage[node]['count']
            anchor_quality = self.anchor_coverage[node]['quality']
            betweenness_score = self.betweenness[node]
            
            # Fiedler component (bottleneck indicator)
            if node < len(self.fiedler_vector):
                fiedler_score = self.fiedler_vector[node]
            else:
                fiedler_score = 0
                
            # Classification logic
            if anchor_count >= 2 and anchor_quality > 3.0:
                node_type = NodeType.WELL_ANCHORED
                confidence = 0.9
                
            elif degree <= 2:
                node_type = NodeType.ISOLATED
                confidence = 0.3
                
            elif betweenness_score > 0.2:
                node_type = NodeType.BRIDGE
                confidence = 0.5
                
            elif fiedler_score > np.percentile(self.fiedler_vector, 80):
                node_type = NodeType.BOTTLENECK
                confidence = 0.4
                
            else:
                node_type = NodeType.NORMAL
                confidence = 0.6
                
            self.node_types[node] = node_type
            
            # Compute overall score for the node
            self.node_scores[node] = {
                'type': node_type,
                'confidence': confidence,
                'degree': degree,
                'anchor_count': anchor_count,
                'anchor_quality': anchor_quality,
                'betweenness': betweenness_score,
                'fiedler': fiedler_score
            }
            
    def get_node_type(self, node: int) -> NodeType:
        """Get classification for a specific node"""
        return self.node_types.get(node, NodeType.NORMAL)
        
    def get_node_confidence(self, node: int) -> float:
        """Get confidence score for a node"""
        if node in self.node_scores:
            return self.node_scores[node]['confidence']
        return 0.5
        
    def get_processing_strategy(self, node: int) -> Dict:
        """
        Get recommended processing strategy for a node
        
        Returns:
            Dictionary with algorithm weights and parameters
        """
        node_type = self.get_node_type(node)
        scores = self.node_scores.get(node, {})
        
        if node_type == NodeType.WELL_ANCHORED:
            # Strong anchor coverage - BP is sufficient
            return {
                'bp_weight': 1.0,
                'hierarchical_weight': 0.0,
                'consensus_weight': 0.2,
                'damping': 0.5,
                'iterations': 50,
                'confidence_threshold': 0.8
            }
            
        elif node_type == NodeType.ISOLATED:
            # Poor connectivity - needs hierarchical help
            return {
                'bp_weight': 0.3,
                'hierarchical_weight': 0.5,
                'consensus_weight': 0.2,
                'damping': 0.8,  # High damping for stability
                'iterations': 100,
                'confidence_threshold': 0.4
            }
            
        elif node_type == NodeType.BOTTLENECK:
            # Critical node - needs consensus
            return {
                'bp_weight': 0.4,
                'hierarchical_weight': 0.1,
                'consensus_weight': 0.5,
                'damping': 0.7,
                'iterations': 80,
                'confidence_threshold': 0.5
            }
            
        elif node_type == NodeType.BRIDGE:
            # Connects clusters - balanced approach
            return {
                'bp_weight': 0.4,
                'hierarchical_weight': 0.3,
                'consensus_weight': 0.3,
                'damping': 0.6,
                'iterations': 70,
                'confidence_threshold': 0.6
            }
            
        else:  # NORMAL
            # Standard processing
            return {
                'bp_weight': 0.5,
                'hierarchical_weight': 0.2,
                'consensus_weight': 0.3,
                'damping': 0.6,
                'iterations': 60,
                'confidence_threshold': 0.6
            }
            
    def get_critical_nodes(self) -> List[int]:
        """Get list of critical nodes that need special attention"""
        critical = []
        
        for node, scores in self.node_scores.items():
            # Critical if: bottleneck, bridge, or very isolated
            if (scores['type'] in [NodeType.BOTTLENECK, NodeType.BRIDGE] or
                (scores['type'] == NodeType.ISOLATED and scores['degree'] <= 1)):
                critical.append(node)
                
        return critical
        
    def get_processing_order(self) -> List[List[int]]:
        """
        Get recommended processing order (in tiers)
        
        Returns:
            List of node groups to process in order
        """
        tiers = []
        
        # Tier 1: Well-anchored nodes (high confidence)
        tier1 = [n for n, t in self.node_types.items() 
                if t == NodeType.WELL_ANCHORED]
        if tier1:
            tiers.append(tier1)
            
        # Tier 2: Normal and bridge nodes
        tier2 = [n for n, t in self.node_types.items() 
                if t in [NodeType.NORMAL, NodeType.BRIDGE]]
        if tier2:
            tiers.append(tier2)
            
        # Tier 3: Bottleneck nodes (need neighbor info)
        tier3 = [n for n, t in self.node_types.items() 
                if t == NodeType.BOTTLENECK]
        if tier3:
            tiers.append(tier3)
            
        # Tier 4: Isolated nodes (most difficult)
        tier4 = [n for n, t in self.node_types.items() 
                if t == NodeType.ISOLATED]
        if tier4:
            tiers.append(tier4)
            
        return tiers
        
    def get_analysis_summary(self) -> Dict:
        """Get summary of network analysis"""
        type_counts = {}
        for node_type in NodeType:
            type_counts[node_type.value] = sum(1 for t in self.node_types.values() 
                                              if t == node_type)
            
        avg_confidence = np.mean([s['confidence'] for s in self.node_scores.values()])
        critical_count = len(self.get_critical_nodes())
        
        return {
            'node_type_distribution': type_counts,
            'average_confidence': avg_confidence,
            'critical_nodes': critical_count,
            'fiedler_value': self.fiedler_value,
            'processing_tiers': len(self.get_processing_order())
        }