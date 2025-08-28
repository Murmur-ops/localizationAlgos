"""
Matrix operations for sensor network localization
Including L-matrix operations and Sinkhorn-Knopp algorithm
"""

import numpy as np
from scipy.sparse import csr_matrix, identity
from typing import Tuple, Optional, Dict, List


class MatrixOperations:
    """Matrix operations for distributed optimization"""
    
    @staticmethod
    def create_laplacian_matrix(adjacency: np.ndarray) -> np.ndarray:
        """
        Create graph Laplacian matrix L = D - A
        
        Args:
            adjacency: Adjacency matrix (symmetric)
            
        Returns:
            Laplacian matrix
        """
        # Degree matrix
        degree = np.diag(np.sum(adjacency, axis=1))
        
        # Laplacian
        L = degree - adjacency
        
        return L
    
    @staticmethod
    def create_incidence_matrix(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create incidence matrix for graph
        
        Args:
            n_nodes: Number of nodes
            edges: List of edges as (i, j) tuples
            
        Returns:
            Incidence matrix (n_nodes x n_edges)
        """
        n_edges = len(edges)
        B = np.zeros((n_nodes, n_edges))
        
        for k, (i, j) in enumerate(edges):
            B[i, k] = 1
            B[j, k] = -1
        
        return B
    
    @staticmethod
    def sinkhorn_knopp(A: np.ndarray, 
                      max_iter: int = 100,
                      tol: float = 1e-6) -> np.ndarray:
        """
        Sinkhorn-Knopp algorithm to create doubly stochastic matrix
        
        Args:
            A: Input non-negative matrix
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Doubly stochastic matrix
        """
        n = A.shape[0]
        
        # Ensure non-negative
        A = np.abs(A) + 1e-10
        
        for iteration in range(max_iter):
            # Row normalization
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / np.maximum(row_sums, 1e-10)
            
            # Column normalization  
            col_sums = A.sum(axis=0, keepdims=True)
            A = A / np.maximum(col_sums, 1e-10)
            
            # Check convergence
            row_err = np.abs(A.sum(axis=1) - 1).max()
            col_err = np.abs(A.sum(axis=0) - 1).max()
            
            if row_err < tol and col_err < tol:
                break
        
        return A
    
    @staticmethod
    def create_consensus_matrix(adjacency: np.ndarray,
                               gamma: float = 0.999) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 2-block consensus matrix for MPS algorithm
        
        Args:
            adjacency: Adjacency matrix
            gamma: Mixing parameter (0 < gamma < 1)
            
        Returns:
            Z matrix, W matrix for 2-block structure
        """
        n = adjacency.shape[0]
        
        # Create base doubly stochastic matrix
        A = adjacency + np.eye(n) * (adjacency.sum(axis=1).max())
        A = MatrixOperations.sinkhorn_knopp(A)
        
        # Create 2-block structure
        Z = np.block([
            [gamma * A, (1 - gamma) * A],
            [(1 - gamma) * A, gamma * A]
        ])
        
        # W matrix (can be identity or weighted)
        W = np.eye(2 * n)
        
        return Z, W
    
    @staticmethod
    def project_onto_consensus_space(X: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Project onto consensus space (null space of L)
        
        Args:
            X: Current iterate (n x d)
            L: Laplacian matrix (n x n)
            
        Returns:
            Projected matrix
        """
        # Compute average (consensus value)
        n = L.shape[0]
        ones = np.ones((n, 1))
        
        # Average each column
        X_avg = (ones @ ones.T @ X) / n
        
        # Project by replacing with average
        # This ensures L @ X_proj = 0
        return X_avg
    
    @staticmethod
    def compute_effective_resistance(L: np.ndarray) -> np.ndarray:
        """
        Compute effective resistance matrix from Laplacian
        
        Args:
            L: Laplacian matrix
            
        Returns:
            Effective resistance matrix
        """
        n = L.shape[0]
        
        # Compute Moore-Penrose pseudoinverse
        L_pinv = np.linalg.pinv(L)
        
        # Effective resistance between nodes i and j
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                e_ij = np.zeros(n)
                e_ij[i] = 1
                e_ij[j] = -1
                R[i, j] = R[j, i] = e_ij.T @ L_pinv @ e_ij
        
        return R
    
    @staticmethod
    def metropolis_hastings_weights(adjacency: np.ndarray) -> np.ndarray:
        """
        Compute Metropolis-Hastings weights for consensus
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Weight matrix
        """
        n = adjacency.shape[0]
        degrees = adjacency.sum(axis=1)
        W = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] > 0 and i != j:
                    W[i, j] = 1 / (1 + max(degrees[i], degrees[j]))
                elif i == j:
                    W[i, i] = 1 - np.sum(W[i, :])
        
        return W
    
    @staticmethod
    def max_degree_weights(adjacency: np.ndarray) -> np.ndarray:
        """
        Compute max-degree weights for consensus
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Weight matrix
        """
        n = adjacency.shape[0]
        max_degree = adjacency.sum(axis=1).max()
        
        W = np.eye(n) - adjacency / max_degree
        
        return W
    
    @staticmethod
    def compute_algebraic_connectivity(L: np.ndarray) -> float:
        """
        Compute algebraic connectivity (second smallest eigenvalue of Laplacian)
        
        Args:
            L: Laplacian matrix
            
        Returns:
            Algebraic connectivity (Fiedler value)
        """
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues.sort()
        
        # Second smallest eigenvalue
        return eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    
    @staticmethod
    def create_stress_matrix(distances: Dict[Tuple[int, int], float],
                            n_nodes: int) -> np.ndarray:
        """
        Create stress matrix for distance-based localization
        
        Args:
            distances: Dictionary of (i, j) -> distance
            n_nodes: Number of nodes
            
        Returns:
            Stress matrix
        """
        S = np.zeros((n_nodes, n_nodes))
        
        for (i, j), dist in distances.items():
            if i != j:
                weight = 1.0 / (dist ** 2 + 1e-6)
                S[i, j] = -weight
                S[j, i] = -weight
                S[i, i] += weight
                S[j, j] += weight
        
        return S