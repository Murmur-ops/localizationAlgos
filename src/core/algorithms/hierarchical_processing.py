"""
Hierarchical Processing for Distributed Sensor Localization
Implements tier-based optimization to improve efficiency
"""

import numpy as np
from numpy.linalg import norm, eigh
from typing import Dict, List, Set, Tuple
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class HierarchicalProcessor:
    def __init__(self, n_sensors: int, n_anchors: int, communication_range: float,
                 noise_factor: float = 0.05, n_tiers: int = 3):
        """
        Initialize hierarchical processor
        
        Args:
            n_sensors: Number of sensors
            n_anchors: Number of anchors
            communication_range: Communication range
            noise_factor: Measurement noise level
            n_tiers: Number of hierarchical tiers
        """
        self.n_sensors = n_sensors
        self.n_anchors = n_anchors
        self.communication_range = communication_range
        self.noise_factor = noise_factor
        self.n_tiers = n_tiers
        
        # Hierarchical structure
        self.clusters = []
        self.cluster_heads = []
        self.sensor_to_cluster = {}
        
        # Network data
        self.graph = None
        self.distances = {}
        self.anchor_positions = None
        self.true_positions = None
        
    def generate_network(self, true_positions: Dict, anchor_positions: np.ndarray):
        """Build network and hierarchical structure"""
        self.true_positions = true_positions
        self.anchor_positions = anchor_positions
        
        # Build connectivity graph
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
                    
        # Build hierarchical clusters
        self._build_hierarchy()
        
    def _build_hierarchy(self):
        """Build hierarchical cluster structure"""
        # Use graph connectivity for clustering
        adjacency = nx.adjacency_matrix(self.graph)
        n_components, labels = connected_components(csgraph=adjacency, directed=False)
        
        # Group sensors by connected component
        components = {}
        for i, label in enumerate(labels):
            if label not in components:
                components[label] = []
            components[label].append(i)
            
        # Further subdivide large components
        self.clusters = []
        for component in components.values():
            if len(component) > self.n_sensors // self.n_tiers:
                # Split large component using spectral clustering
                sub_clusters = self._spectral_cluster(component)
                self.clusters.extend(sub_clusters)
            else:
                self.clusters.append(component)
                
        # Select cluster heads (best connected nodes)
        self.cluster_heads = []
        for cluster_id, cluster in enumerate(self.clusters):
            # Find node with most connections within cluster
            best_node = None
            best_degree = -1
            
            for node in cluster:
                degree = sum(1 for neighbor in self.graph.neighbors(node) 
                           if neighbor in cluster)
                # Bonus for anchor connections
                anchor_bonus = sum(1 for a in range(self.n_anchors) 
                                 if (node, f'a{a}') in self.distances)
                degree += anchor_bonus * 2
                
                if degree > best_degree:
                    best_degree = degree
                    best_node = node
                    
            self.cluster_heads.append(best_node)
            
            # Map sensors to clusters
            for node in cluster:
                self.sensor_to_cluster[node] = cluster_id
                
    def _spectral_cluster(self, nodes: List[int], n_clusters: int = None) -> List[List[int]]:
        """Use spectral clustering to split nodes"""
        if n_clusters is None:
            n_clusters = min(3, len(nodes) // 4)
            
        if n_clusters <= 1 or len(nodes) <= n_clusters:
            return [nodes]
            
        # Build subgraph adjacency
        subgraph = self.graph.subgraph(nodes)
        adjacency = nx.adjacency_matrix(subgraph)
        
        # Compute Laplacian
        degree = np.array(adjacency.sum(axis=1)).flatten()
        D = np.diag(degree)
        L = D - adjacency.toarray()
        
        # Get eigenvectors
        try:
            eigenvalues, eigenvectors = eigh(L)
            # Use first n_clusters eigenvectors
            embedding = eigenvectors[:, :n_clusters]
            
            # Simple clustering without sklearn
            # Initialize cluster centers randomly
            centers = embedding[np.random.choice(len(nodes), n_clusters, replace=False)]
            
            # Simple k-means iteration
            for _ in range(10):
                # Assign points to nearest center
                labels = []
                for point in embedding:
                    distances = [norm(point - center) for center in centers]
                    labels.append(np.argmin(distances))
                    
                # Update centers
                new_centers = []
                for k in range(n_clusters):
                    cluster_points = [embedding[i] for i, l in enumerate(labels) if l == k]
                    if cluster_points:
                        new_centers.append(np.mean(cluster_points, axis=0))
                    else:
                        new_centers.append(centers[k])
                centers = np.array(new_centers)
                
            # Group nodes by label
            clusters = [[] for _ in range(n_clusters)]
            for i, node in enumerate(nodes):
                clusters[labels[i]].append(node)
                
            return [c for c in clusters if c]  # Remove empty clusters
            
        except:
            # Fallback: simple split
            chunk_size = len(nodes) // n_clusters
            return [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
            
    def process_tier(self, tier_nodes: List[int], positions: Dict) -> Dict:
        """Process a single tier of nodes"""
        tier_positions = {}
        
        for node in tier_nodes:
            # Collect constraints
            forces = []
            weights = []
            
            # Anchor constraints
            for a in range(self.n_anchors):
                if (node, f'a{a}') in self.distances:
                    anchor_pos = self.anchor_positions[a]
                    measured_dist = self.distances[(node, f'a{a}')] 
                    
                    if node in positions:
                        current_pos = positions[node]
                        current_dist = norm(current_pos - anchor_pos)
                        
                        if current_dist > 1e-6:
                            direction = (current_pos - anchor_pos) / current_dist
                            target = anchor_pos + direction * measured_dist
                            forces.append(target)
                            weights.append(2.0 / (self.noise_factor * measured_dist + 0.01))
                    else:
                        # Initialize with anchor constraint
                        angle = np.random.uniform(0, 2*np.pi)
                        target = anchor_pos + measured_dist * np.array([np.cos(angle), np.sin(angle)])
                        forces.append(target)
                        weights.append(1.0)
                        
            # Neighbor constraints within tier
            for neighbor in self.graph.neighbors(node):
                if neighbor in tier_nodes and neighbor in positions:
                    if (node, neighbor) in self.distances:
                        measured_dist = self.distances[(node, neighbor)]
                    else:
                        measured_dist = self.distances[(neighbor, node)]
                        
                    neighbor_pos = positions[neighbor]
                    
                    if node in positions:
                        current_pos = positions[node]
                        current_dist = norm(current_pos - neighbor_pos)
                        
                        if current_dist > 1e-6:
                            direction = (current_pos - neighbor_pos) / current_dist
                            target = neighbor_pos + direction * measured_dist
                            forces.append(target)
                            weights.append(1.0 / (self.noise_factor * measured_dist + 0.01))
                            
            # Compute weighted average
            if forces:
                weights = np.array(weights)
                weights /= weights.sum()
                new_pos = np.zeros(2)
                for f, w in zip(forces, weights):
                    new_pos += w * f
                    
                # Apply damping if updating
                if node in positions:
                    damping = 0.7
                    new_pos = damping * positions[node] + (1 - damping) * new_pos
                    
                tier_positions[node] = np.clip(new_pos, 0, 1)
            elif node in positions:
                tier_positions[node] = positions[node]
            else:
                tier_positions[node] = np.random.uniform(0.2, 0.8, 2)
                
        return tier_positions
        
    def run(self, max_iter: int = 50) -> Dict:
        """Run hierarchical processing"""
        # Initialize positions
        positions = {}
        
        # Process each tier
        for tier in range(self.n_tiers):
            # Determine nodes for this tier
            if tier == 0:
                # Tier 0: Cluster heads and well-connected nodes
                tier_nodes = set(self.cluster_heads)
                
                # Add nodes with anchor connections
                for node in range(self.n_sensors):
                    anchor_count = sum(1 for a in range(self.n_anchors)
                                     if (node, f'a{a}') in self.distances)
                    if anchor_count >= 2:
                        tier_nodes.add(node)
                        
            elif tier == 1:
                # Tier 1: Neighbors of tier 0
                tier_nodes = set()
                for node in range(self.n_sensors):
                    if node not in positions:
                        # Check if connected to tier 0
                        for neighbor in self.graph.neighbors(node):
                            if neighbor in positions:
                                tier_nodes.add(node)
                                break
                                
            else:
                # Tier 2+: Remaining nodes
                tier_nodes = set(range(self.n_sensors)) - set(positions.keys())
                
            if not tier_nodes:
                continue
                
            # Iterate within tier
            for iteration in range(max_iter // self.n_tiers):
                old_positions = {n: positions.get(n, np.random.uniform(0, 1, 2)).copy() 
                               for n in tier_nodes}
                
                # Update positions in tier
                tier_updates = self.process_tier(list(tier_nodes), positions)
                positions.update(tier_updates)
                
                # Check convergence
                max_change = max([norm(positions[n] - old_positions[n]) 
                                for n in tier_nodes if n in old_positions])
                
                if max_change < 1e-4:
                    break
                    
        # Final refinement pass
        for _ in range(10):
            old_pos = {i: positions[i].copy() for i in range(self.n_sensors)}
            
            for cluster_id, cluster in enumerate(self.clusters):
                cluster_updates = self.process_tier(cluster, positions)
                positions.update(cluster_updates)
                
            max_change = max([norm(positions[i] - old_pos[i]) 
                            for i in range(self.n_sensors)])
            
            if max_change < 1e-4:
                break
                
        # Calculate error
        if self.true_positions:
            errors = [norm(positions[i] - self.true_positions[i]) 
                     for i in range(self.n_sensors)]
            final_error = np.sqrt(np.mean(np.square(errors)))
        else:
            final_error = 0
            
        return {
            'positions': positions,
            'final_error': final_error,
            'n_clusters': len(self.clusters),
            'cluster_sizes': [len(c) for c in self.clusters]
        }