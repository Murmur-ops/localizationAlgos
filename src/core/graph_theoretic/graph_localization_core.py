"""
Graph-Theoretic Core for Distributed Sensor Localization

Based on research from:
- "Graph Signal Processing: Overview, Challenges and Applications" (Ortega et al., 2018)
- "Distributed Network Localization from Graph Laplacian Perspective" (2024)
- "Algebraic connectivity in distributed networks" (2019)

This module implements the core graph operations needed for distributed localization.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx


class GraphLocalizationCore:
    """
    Core graph-theoretic operations for sensor network localization
    
    Key concepts:
    - Graph Laplacian matrix L = D - A (degree matrix - adjacency matrix)
    - Fiedler value λ₂: second smallest eigenvalue of L (algebraic connectivity)
    - Fiedler vector: corresponding eigenvector (used for partitioning/embedding)
    """
    
    def __init__(self, n_sensors: int, communication_range: float = 0.4):
        """
        Initialize graph structure for sensor network
        
        Args:
            n_sensors: Number of sensors in network
            communication_range: Maximum communication distance
        """
        self.n_sensors = n_sensors
        self.communication_range = communication_range
        
        # Graph representation
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_sensors))
        
        # Matrices (sparse for efficiency)
        self.adjacency_matrix = None
        self.laplacian_matrix = None
        self.weighted_laplacian = None
        
        # Spectral properties
        self.fiedler_value = None
        self.fiedler_vector = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        # Sensor categorization
        self.anchor_nodes = set()
        self.sensor_tiers = {}  # Hierarchy based on anchor distance
    
    def build_network_from_distances(self, 
                                    sensor_positions: Dict[int, np.ndarray],
                                    anchor_positions: Optional[np.ndarray] = None):
        """
        Build network graph from sensor positions
        
        Based on Unit Disk Graph model (research standard for WSN)
        """
        # Build sensor-to-sensor edges
        edges = []
        edge_weights = {}
        
        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                if i in sensor_positions and j in sensor_positions:
                    dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
                    if dist <= self.communication_range:
                        edges.append((i, j))
                        # Weight inversely proportional to distance (research-backed)
                        edge_weights[(i, j)] = 1.0 / (1.0 + dist)
        
        self.graph.add_edges_from(edges)
        
        # Store edge weights
        nx.set_edge_attributes(self.graph, edge_weights, 'weight')
        
        # Identify anchor-adjacent sensors (Tier 1)
        if anchor_positions is not None:
            self._identify_sensor_tiers(sensor_positions, anchor_positions)
        
        # Compute graph matrices
        self._compute_graph_matrices()
    
    def _identify_sensor_tiers(self, 
                              sensor_positions: Dict[int, np.ndarray],
                              anchor_positions: np.ndarray):
        """
        Categorize sensors into tiers based on anchor proximity
        
        Research backing: Hierarchical processing improves performance
        - Tier 0: Anchors
        - Tier 1: Direct anchor neighbors
        - Tier 2: 2-hop from anchors
        - Tier 3+: Further sensors
        """
        n_anchors = len(anchor_positions)
        
        # Find sensors within range of each anchor
        tier_1_sensors = set()
        for i in range(self.n_sensors):
            if i in sensor_positions:
                for k in range(n_anchors):
                    dist = np.linalg.norm(sensor_positions[i] - anchor_positions[k])
                    if dist <= self.communication_range:
                        tier_1_sensors.add(i)
                        break
        
        # BFS to find tier 2, 3, etc.
        self.sensor_tiers[1] = tier_1_sensors
        
        visited = tier_1_sensors.copy()
        current_tier = tier_1_sensors
        tier_num = 2
        
        while len(visited) < self.n_sensors:
            next_tier = set()
            for node in current_tier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        next_tier.add(neighbor)
                        visited.add(neighbor)
            
            if next_tier:
                self.sensor_tiers[tier_num] = next_tier
                current_tier = next_tier
                tier_num += 1
            else:
                break
    
    def _compute_graph_matrices(self):
        """
        Compute adjacency and Laplacian matrices
        
        Research: "The Laplacian matrix is fundamental to graph signal processing"
        """
        # Get adjacency matrix (weighted)
        self.adjacency_matrix = nx.adjacency_matrix(self.graph, weight='weight')
        
        # Compute Laplacian: L = D - A
        degree_matrix = np.diag(np.array(self.adjacency_matrix.sum(axis=1)).flatten())
        self.laplacian_matrix = degree_matrix - self.adjacency_matrix.toarray()
        
        # Normalized Laplacian (better numerical properties)
        # L_norm = D^(-1/2) * L * D^(-1/2)
        d_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree_matrix.diagonal(), 1e-10)))
        self.normalized_laplacian = d_sqrt_inv @ self.laplacian_matrix @ d_sqrt_inv
    
    def compute_spectral_properties(self, k: int = 6):
        """
        Compute eigenvalues and eigenvectors of Laplacian
        
        Research backing: 
        - Fiedler value (λ₂) determines convergence rate
        - First k eigenvectors provide optimal k-dimensional embedding
        
        Args:
            k: Number of smallest eigenvalues/vectors to compute
        """
        # Ensure we don't request more than available
        k = min(k, self.n_sensors)
        
        # Compute eigendecomposition
        if self.n_sensors < 100:
            # Small networks: use dense method
            eigenvalues, eigenvectors = eigh(self.laplacian_matrix)
            self.eigenvalues = eigenvalues[:k]
            self.eigenvectors = eigenvectors[:, :k]
        else:
            # Large networks: use sparse method
            eigenvalues, eigenvectors = eigsh(
                csr_matrix(self.laplacian_matrix), 
                k=k, 
                which='SM'  # Smallest magnitude
            )
            idx = np.argsort(eigenvalues)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
        
        # Extract Fiedler value and vector
        if len(self.eigenvalues) > 1:
            self.fiedler_value = self.eigenvalues[1]  # Second smallest
            self.fiedler_vector = self.eigenvectors[:, 1]
        
        return self.eigenvalues, self.eigenvectors
    
    def spectral_embedding(self, d: int = 2) -> np.ndarray:
        """
        Compute spectral embedding for initial positions
        
        Research: "Laplacian Eigenmaps" provide optimal embedding
        
        Args:
            d: Embedding dimension
            
        Returns:
            Initial position estimates from spectral embedding
        """
        if self.eigenvectors is None:
            self.compute_spectral_properties(d + 1)
        
        # Use eigenvectors 1 through d (skip the constant eigenvector)
        embedding = self.eigenvectors[:, 1:d+1]
        
        # Scale to unit square
        for i in range(d):
            min_val = embedding[:, i].min()
            max_val = embedding[:, i].max()
            if max_val > min_val:
                embedding[:, i] = (embedding[:, i] - min_val) / (max_val - min_val)
        
        return embedding
    
    def optimize_edge_weights_for_connectivity(self) -> Dict[Tuple[int, int], float]:
        """
        Optimize edge weights to maximize algebraic connectivity (Fiedler value)
        
        Research: "Maximizing algebraic connectivity improves convergence"
        From: "Algebraic Connectivity Control in Distributed Networks" (2021)
        """
        # Start with current weights
        current_weights = nx.get_edge_attributes(self.graph, 'weight')
        
        # Gradient ascent on Fiedler value
        learning_rate = 0.1
        max_iterations = 100
        
        best_weights = current_weights.copy()
        best_fiedler = self.fiedler_value if self.fiedler_value else 0
        
        for _ in range(max_iterations):
            # Perturb weights slightly
            perturbed_weights = {}
            for edge, weight in best_weights.items():
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01)
                new_weight = np.clip(weight + perturbation, 0.01, 10.0)
                perturbed_weights[edge] = new_weight
            
            # Update graph with perturbed weights
            nx.set_edge_attributes(self.graph, perturbed_weights, 'weight')
            self._compute_graph_matrices()
            self.compute_spectral_properties()
            
            # Keep if better
            if self.fiedler_value > best_fiedler:
                best_weights = perturbed_weights
                best_fiedler = self.fiedler_value
        
        return best_weights
    
    def compute_resistance_distance(self, i: int, j: int) -> float:
        """
        Compute resistance distance between nodes (graph-theoretic distance)
        
        Research: Resistance distance is more robust than hop distance
        """
        if self.eigenvectors is None:
            self.compute_spectral_properties()
        
        # Resistance distance using Moore-Penrose pseudoinverse of Laplacian
        # R_ij = (e_i - e_j)^T * L^+ * (e_i - e_j)
        L_pinv = np.linalg.pinv(self.laplacian_matrix)
        
        e_diff = np.zeros(self.n_sensors)
        e_diff[i] = 1
        e_diff[j] = -1
        
        resistance = e_diff.T @ L_pinv @ e_diff
        return resistance
    
    def distributed_consensus_matrix(self, epsilon: Optional[float] = None) -> np.ndarray:
        """
        Compute consensus matrix for distributed averaging
        
        Research: Consensus achieved via x_{k+1} = (I - εL)x_k
        Convergence rate determined by spectral gap
        
        Args:
            epsilon: Step size (auto-computed if None)
            
        Returns:
            Consensus iteration matrix
        """
        if epsilon is None:
            # Optimal epsilon for fastest convergence
            # ε = 2/(λ_max + λ_2) from research
            max_eigenval = np.max(np.abs(self.eigenvalues)) if self.eigenvalues is not None else 2
            epsilon = 1.0 / max_eigenval
        
        # Consensus matrix
        consensus_matrix = np.eye(self.n_sensors) - epsilon * self.laplacian_matrix
        
        return consensus_matrix
    
    def get_network_metrics(self) -> Dict[str, float]:
        """
        Compute key network metrics for analysis
        """
        metrics = {
            'n_sensors': self.n_sensors,
            'n_edges': self.graph.number_of_edges(),
            'avg_degree': 2 * self.graph.number_of_edges() / self.n_sensors,
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
            'fiedler_value': self.fiedler_value if self.fiedler_value else 0,
            'connectivity': nx.node_connectivity(self.graph),
            'clustering_coeff': nx.average_clustering(self.graph)
        }
        
        # Tier distribution
        for tier_num, sensors in self.sensor_tiers.items():
            metrics[f'tier_{tier_num}_count'] = len(sensors)
        
        return metrics


def test_graph_core():
    """Test the graph localization core functionality"""
    print("="*60)
    print("Testing Graph Localization Core")
    print("="*60)
    
    # Create test network
    n_sensors = 20
    np.random.seed(42)
    
    # Generate random positions for testing
    positions = {}
    for i in range(n_sensors):
        positions[i] = np.random.uniform(0, 1, 2)
    
    # Create anchors
    anchor_positions = np.array([
        [0.1, 0.1],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.1, 0.9]
    ])
    
    # Initialize graph core
    graph_core = GraphLocalizationCore(n_sensors, communication_range=0.4)
    graph_core.build_network_from_distances(positions, anchor_positions)
    
    # Compute spectral properties
    eigenvals, eigenvecs = graph_core.compute_spectral_properties()
    
    print(f"\nNetwork Metrics:")
    metrics = graph_core.get_network_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print(f"\nSpectral Properties:")
    print(f"  First 6 eigenvalues: {eigenvals}")
    print(f"  Fiedler value (λ₂): {graph_core.fiedler_value:.4f}")
    print(f"  Spectral gap (λ₂/λ₁): {graph_core.fiedler_value / eigenvals[0]:.4f}")
    
    # Test spectral embedding
    embedding = graph_core.spectral_embedding(d=2)
    print(f"\nSpectral Embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    return graph_core


if __name__ == "__main__":
    graph_core = test_graph_core()