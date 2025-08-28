"""
Core matrix operations for MPS algorithm
Simplified implementation focusing on essential operations from the paper
"""

import numpy as np
from typing import Tuple, Dict, Set


class MatrixOperations:
    """Essential matrix operations for distributed optimization"""
    
    @staticmethod
    def create_laplacian(adjacency: np.ndarray) -> np.ndarray:
        """
        Create graph Laplacian matrix L = D - A
        
        Args:
            adjacency: Adjacency matrix (symmetric)
            
        Returns:
            Laplacian matrix
        """
        degree = np.diag(np.sum(adjacency, axis=1))
        return degree - adjacency
    
    @staticmethod
    def sinkhorn_knopp(A: np.ndarray, 
                      max_iter: int = 50,
                      tol: float = 1e-6) -> np.ndarray:
        """
        Simplified Sinkhorn-Knopp algorithm for doubly stochastic matrix
        
        Args:
            A: Input non-negative matrix  
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Doubly stochastic matrix
        """
        # Ensure non-negative
        A = np.abs(A) + 1e-10
        
        for _ in range(max_iter):
            # Row normalization
            A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-10)
            
            # Column normalization
            A = A / np.maximum(A.sum(axis=0, keepdims=True), 1e-10)
            
            # Check convergence
            if (np.abs(A.sum(axis=1) - 1).max() < tol and 
                np.abs(A.sum(axis=0) - 1).max() < tol):
                break
        
        return A
    
    @staticmethod
    def create_consensus_matrix(adjacency: np.ndarray,
                               gamma: float = 0.99) -> np.ndarray:
        """
        Create consensus matrix for MPS algorithm (2-block structure from paper)
        
        Args:
            adjacency: Adjacency matrix
            gamma: Mixing parameter (0 < gamma < 1)
            
        Returns:
            Z matrix for consensus operation
        """
        n = adjacency.shape[0]
        
        # Create weight matrix from adjacency
        W = adjacency.copy()
        W = W + np.eye(n)  # Add self-loops
        
        # Make doubly stochastic
        W = MatrixOperations.sinkhorn_knopp(W)
        
        # Create 2-block consensus matrix as in paper
        # Z = [γW, (1-γ)I; (1-γ)I, γW]
        Z = np.zeros((2*n, 2*n))
        Z[:n, :n] = gamma * W
        Z[:n, n:] = (1 - gamma) * np.eye(n)
        Z[n:, :n] = (1 - gamma) * np.eye(n)
        Z[n:, n:] = gamma * W
        
        return Z
    
    @staticmethod
    def build_adjacency(positions: Dict[int, np.ndarray],
                       communication_range: float) -> np.ndarray:
        """
        Build adjacency matrix from positions and communication range
        
        Args:
            positions: Dictionary of sensor positions
            communication_range: Maximum communication distance
            
        Returns:
            Adjacency matrix
        """
        n = len(positions)
        adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= communication_range:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
        
        return adjacency