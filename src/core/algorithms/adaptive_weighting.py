"""
Adaptive Weighting using Fiedler Eigenvalue
Adjusts algorithm parameters based on graph connectivity
"""

import numpy as np
from numpy.linalg import eigh, norm
from typing import Dict, Tuple
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


class AdaptiveWeighting:
    def __init__(self, graph: nx.Graph, n_anchors: int, anchor_positions: np.ndarray):
        """
        Initialize adaptive weighting system
        
        Args:
            graph: Network connectivity graph
            n_anchors: Number of anchors
            anchor_positions: Anchor positions
        """
        self.graph = graph
        self.n_anchors = n_anchors
        self.anchor_positions = anchor_positions
        
        # Compute graph properties
        self.fiedler_value = None
        self.fiedler_vector = None
        self.node_centrality = {}
        self.edge_weights = {}
        
        self._analyze_graph()
        
    def _analyze_graph(self):
        """Analyze graph structure for adaptive weighting"""
        # Compute Laplacian
        L = nx.laplacian_matrix(self.graph).astype(float)
        n = self.graph.number_of_nodes()
        
        if n < 2:
            self.fiedler_value = 1.0
            self.fiedler_vector = np.ones(n)
            return
            
        # Compute Fiedler eigenvalue and eigenvector
        try:
            # Get second smallest eigenvalue (Fiedler value)
            if n < 10:
                # Small graph: use dense computation
                L_dense = L.toarray()
                eigenvalues, eigenvectors = eigh(L_dense)
                idx = np.argsort(eigenvalues)
                self.fiedler_value = eigenvalues[idx[1]] if len(idx) > 1 else eigenvalues[0]
                self.fiedler_vector = eigenvectors[:, idx[1]] if len(idx) > 1 else eigenvectors[:, 0]
            else:
                # Large graph: use sparse computation
                eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
                self.fiedler_value = max(eigenvalues)
                idx = np.argmax(eigenvalues)
                self.fiedler_vector = eigenvectors[:, idx]
                
        except:
            # Fallback
            self.fiedler_value = 1.0
            self.fiedler_vector = np.ones(n) / np.sqrt(n)
            
        # Compute node centrality measures
        self._compute_centrality()
        
        # Compute adaptive edge weights
        self._compute_edge_weights()
        
    def _compute_centrality(self):
        """Compute node centrality for importance weighting"""
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Closeness centrality (if connected)
        try:
            closeness_cent = nx.closeness_centrality(self.graph)
        except:
            closeness_cent = degree_cent
            
        # Combine metrics
        for node in self.graph.nodes():
            # Base centrality from graph structure
            structural_score = 0.5 * degree_cent[node] + 0.5 * closeness_cent.get(node, 0)
            
            # Boost for Fiedler vector (indicates bottlenecks)
            fiedler_score = abs(self.fiedler_vector[node]) if node < len(self.fiedler_vector) else 0
            
            # Combine scores
            self.node_centrality[node] = structural_score + 0.3 * fiedler_score
            
    def _compute_edge_weights(self):
        """Compute adaptive weights for edges"""
        # Base weight inversely proportional to Fiedler value
        # Low Fiedler = poorly connected = need stronger weights
        base_weight = 1.0 / (self.fiedler_value + 0.1)
        
        for edge in self.graph.edges():
            i, j = edge
            
            # Weight based on node centralities
            centrality_weight = (self.node_centrality[i] + self.node_centrality[j]) / 2
            
            # Weight based on Fiedler vector difference (larger diff = weaker connection)
            if i < len(self.fiedler_vector) and j < len(self.fiedler_vector):
                fiedler_diff = abs(self.fiedler_vector[i] - self.fiedler_vector[j])
                connectivity_weight = 1.0 / (1.0 + fiedler_diff)
            else:
                connectivity_weight = 1.0
                
            # Combine weights
            self.edge_weights[edge] = base_weight * centrality_weight * connectivity_weight
            self.edge_weights[(j, i)] = self.edge_weights[edge]
            
    def get_node_weight(self, node: int) -> float:
        """Get adaptive weight for a node"""
        # Base weight from centrality
        base = self.node_centrality.get(node, 1.0)
        
        # Boost based on connectivity (inverse of Fiedler)
        connectivity_boost = 2.0 / (self.fiedler_value + 1.0)
        
        return base * connectivity_boost
        
    def get_edge_weight(self, i: int, j: int) -> float:
        """Get adaptive weight for an edge"""
        if (i, j) in self.edge_weights:
            return self.edge_weights[(i, j)]
        elif (j, i) in self.edge_weights:
            return self.edge_weights[(j, i)]
        else:
            # Default weight for non-edges
            return 0.5 / (self.fiedler_value + 1.0)
            
    def get_anchor_weight(self, node: int, anchor_id: int) -> float:
        """Get adaptive weight for anchor measurement"""
        # Base weight from node centrality
        node_weight = self.node_centrality.get(node, 1.0)
        
        # Boost for nodes with few anchors
        anchor_count = 0
        if hasattr(self, 'distances'):
            anchor_count = sum(1 for a in range(self.n_anchors) 
                             if (node, f'a{a}') in self.distances)
                             
        scarcity_boost = 3.0 / (anchor_count + 1) if anchor_count > 0 else 2.0
        
        # Weight based on Fiedler (poor connectivity needs anchors more)
        connectivity_factor = 2.0 / (self.fiedler_value + 0.5)
        
        return node_weight * scarcity_boost * connectivity_factor
        
    def get_convergence_rate(self) -> float:
        """Estimate convergence rate based on Fiedler value"""
        # Higher Fiedler = better connectivity = faster convergence
        # Rate between 0.5 (slow) and 0.95 (fast)
        rate = 0.5 + 0.45 * np.tanh(self.fiedler_value)
        return rate
        
    def get_damping_factor(self) -> float:
        """Get adaptive damping factor"""
        # Poor connectivity needs more damping for stability
        # Good connectivity can use less damping for speed
        if self.fiedler_value < 0.1:
            return 0.8  # High damping for poor connectivity
        elif self.fiedler_value < 0.5:
            return 0.7
        elif self.fiedler_value < 1.0:
            return 0.6
        else:
            return 0.5  # Low damping for good connectivity
            
    def get_iteration_count(self, base_iterations: int = 100) -> int:
        """Get adaptive iteration count"""
        # Poor connectivity needs more iterations
        if self.fiedler_value < 0.1:
            return int(base_iterations * 2.0)
        elif self.fiedler_value < 0.5:
            return int(base_iterations * 1.5)
        elif self.fiedler_value < 1.0:
            return base_iterations
        else:
            return int(base_iterations * 0.8)
            
    def apply_to_belief_update(self, node: int, neighbor_messages: Dict, 
                              anchor_measurements: Dict) -> Tuple[Dict, Dict]:
        """
        Apply adaptive weighting to belief propagation messages
        
        Args:
            node: Current node
            neighbor_messages: Messages from neighbors
            anchor_measurements: Measurements to anchors
            
        Returns:
            Weighted messages and measurements
        """
        weighted_messages = {}
        weighted_anchors = {}
        
        # Weight neighbor messages
        for neighbor, message in neighbor_messages.items():
            edge_weight = self.get_edge_weight(node, neighbor)
            weighted_messages[neighbor] = {
                'mean': message['mean'],
                'precision': message['precision'] * edge_weight
            }
            
        # Weight anchor measurements
        for anchor_id, measurement in anchor_measurements.items():
            anchor_weight = self.get_anchor_weight(node, anchor_id)
            weighted_anchors[anchor_id] = {
                'distance': measurement['distance'],
                'weight': anchor_weight
            }
            
        return weighted_messages, weighted_anchors
        
    def get_algorithm_parameters(self) -> Dict:
        """Get all adaptive parameters for algorithms"""
        return {
            'damping': self.get_damping_factor(),
            'max_iterations': self.get_iteration_count(),
            'convergence_rate': self.get_convergence_rate(),
            'fiedler_value': self.fiedler_value,
            'connectivity_score': np.tanh(self.fiedler_value),  # 0-1 score
            'node_weights': self.node_centrality,
            'edge_weights': self.edge_weights
        }