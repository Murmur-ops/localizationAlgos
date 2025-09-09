"""
Sinkhorn-Knopp Algorithm for Matrix Parameter Selection
Implements decentralized doubly stochastic matrix generation
for the Matrix-Parametrized Proximal Splitting algorithm
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SinkhornKnopp:
    """
    Sinkhorn-Knopp algorithm for creating doubly stochastic matrices
    that adhere to communication graph structure
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-10,
                 check_support: bool = True):
        """
        Initialize Sinkhorn-Knopp algorithm
        
        Args:
            adjacency_matrix: Binary adjacency matrix of communication graph
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            check_support: Whether to check for matrix support
        """
        self.A = adjacency_matrix.astype(float)
        self.n = adjacency_matrix.shape[0]
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Check if matrix has support (necessary condition)
        if check_support and not self._has_support():
            # Try to make it connected by adding minimal edges
            self.A = self._make_connected(self.A)
    
    def _has_support(self) -> bool:
        """
        Check if the matrix has support (admits a perfect matching)
        For symmetric matrices, this means the graph is connected
        """
        # For undirected graphs, check connectivity
        # Use breadth-first search from node 0
        visited = np.zeros(self.n, dtype=bool)
        queue = [0]
        visited[0] = True
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(self.n):
                if self.A[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return np.all(visited)
    
    def _make_connected(self, A: np.ndarray) -> np.ndarray:
        """
        Make a disconnected graph connected by adding minimal edges
        
        Args:
            A: Adjacency matrix
            
        Returns:
            Connected adjacency matrix
        """
        # Find connected components
        n = A.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = []
        
        for start in range(n):
            if not visited[start]:
                component = []
                queue = [start]
                visited[start] = True
                
                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    for neighbor in range(n):
                        if A[node, neighbor] > 0 and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                components.append(component)
        
        # Connect components with minimal edges
        A_connected = A.copy()
        for i in range(len(components) - 1):
            # Connect component i to component i+1
            node1 = components[i][0]
            node2 = components[i+1][0]
            A_connected[node1, node2] = 1
            A_connected[node2, node1] = 1
        
        return A_connected
    
    def compute_doubly_stochastic(self) -> np.ndarray:
        """
        Compute doubly stochastic matrix using Sinkhorn-Knopp algorithm
        
        Returns:
            Doubly stochastic matrix with same sparsity pattern as A
        """
        # Initialize with adjacency matrix
        B = self.A.copy()
        
        for iteration in range(self.max_iterations):
            # Store previous matrix for convergence check
            B_prev = B.copy()
            
            # Row normalization
            row_sums = B.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            B = B / row_sums[:, np.newaxis]
            
            # Column normalization
            col_sums = B.sum(axis=0)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            B = B / col_sums[np.newaxis, :]
            
            # Check convergence
            if np.max(np.abs(B - B_prev)) < self.tolerance:
                logger.debug(f"Sinkhorn-Knopp converged in {iteration + 1} iterations")
                break
        else:
            logger.warning(f"Sinkhorn-Knopp did not converge in {self.max_iterations} iterations")
        
        # Ensure exact symmetry for undirected graphs
        if np.allclose(self.A, self.A.T):
            B = (B + B.T) / 2
        
        return B
    
    def compute_2block_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Z and W matrices for 2-Block design
        Following Corollary 1.1 from the paper
        
        Returns:
            Z, W matrices satisfying the requirements for proximal splitting
        """
        # Add identity to ensure support (A + I)
        A_plus_I = self.A + np.eye(self.n)
        
        # Compute doubly stochastic version
        sk_alg = SinkhornKnopp(A_plus_I, self.max_iterations, self.tolerance)
        SK_A = sk_alg.compute_doubly_stochastic()
        
        # Construct 2-Block matrices
        # Z = W = 2 * [I   -SK(A+I)]
        #            [-SK(A+I)   I]
        Z = np.zeros((2*self.n, 2*self.n))
        
        # Upper left block: I
        Z[:self.n, :self.n] = np.eye(self.n)
        
        # Upper right block: -SK(A+I)
        Z[:self.n, self.n:] = -SK_A
        
        # Lower left block: -SK(A+I)
        Z[self.n:, :self.n] = -SK_A
        
        # Lower right block: I
        Z[self.n:, self.n:] = np.eye(self.n)
        
        # Scale by 2
        Z = 2 * Z
        
        # For this design, W = Z
        W = Z.copy()
        
        # Verify requirements
        self._verify_matrix_requirements(Z, W)
        
        return Z, W
    
    def _verify_matrix_requirements(self, Z: np.ndarray, W: np.ndarray):
        """
        Verify that Z and W satisfy the requirements from equation (8) in the paper:
        - Z ⪰ W (Z - W is PSD)
        - null(W) = span(1)
        - diag(Z) = 2*1
        - 1^T Z 1 = 0
        """
        tol = 1e-8
        ones = np.ones(Z.shape[0])
        
        # Check Z ⪰ W
        diff = Z - W
        eigenvalues = np.linalg.eigvalsh(diff)
        if np.min(eigenvalues) < -tol:
            logger.warning(f"Z - W not PSD: min eigenvalue = {np.min(eigenvalues)}")
        
        # Check null(W) = span(1)
        W_ones = W @ ones
        if np.linalg.norm(W_ones) > tol:
            logger.warning(f"W*1 != 0: norm = {np.linalg.norm(W_ones)}")
        
        # Check diag(Z) = 2*1
        diag_Z = np.diag(Z)
        if not np.allclose(diag_Z, 2 * np.ones_like(diag_Z), atol=tol):
            logger.warning(f"diag(Z) != 2*1: max diff = {np.max(np.abs(diag_Z - 2))}")
        
        # Check 1^T Z 1 = 0
        ones_Z_ones = ones.T @ Z @ ones
        if abs(ones_Z_ones) > tol:
            logger.warning(f"1^T Z 1 != 0: value = {ones_Z_ones}")
    
    @staticmethod
    def decentralized_computation(adjacency_matrix: np.ndarray,
                                 max_iterations: int = 1000,
                                 tolerance: float = 1e-10) -> np.ndarray:
        """
        Simulate decentralized computation of Sinkhorn-Knopp algorithm
        Each node only communicates with its neighbors
        
        Args:
            adjacency_matrix: Communication graph structure
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Doubly stochastic matrix
        """
        n = adjacency_matrix.shape[0]
        
        # Initialize edge weights (each node stores weights to neighbors)
        edge_weights = adjacency_matrix.copy().astype(float)
        
        for iteration in range(max_iterations):
            edge_weights_prev = edge_weights.copy()
            
            # Each node normalizes its outgoing edges (row normalization)
            for i in range(n):
                neighbors = np.where(adjacency_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    # Node i normalizes weights to its neighbors
                    total = edge_weights[i, neighbors].sum()
                    if total > 0:
                        edge_weights[i, neighbors] /= total
            
            # Communication step: each node sends its weights to neighbors
            # and receives weights from neighbors
            received_weights = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if adjacency_matrix[i, j] > 0:
                        # Node j receives weight from node i
                        received_weights[j, i] = edge_weights[i, j]
            
            # Each node normalizes incoming edges (column normalization)
            for j in range(n):
                neighbors = np.where(adjacency_matrix[:, j] > 0)[0]
                if len(neighbors) > 0:
                    total = received_weights[neighbors, j].sum()
                    if total > 0:
                        scale = 1.0 / total
                        # Send scaling factor back to neighbors
                        for i in neighbors:
                            edge_weights[i, j] *= scale
            
            # Check convergence
            if np.max(np.abs(edge_weights - edge_weights_prev)) < tolerance:
                logger.debug(f"Decentralized Sinkhorn-Knopp converged in {iteration + 1} iterations")
                break
        
        return edge_weights


class MatrixParameterGenerator:
    """
    Generate matrix parameters for proximal splitting algorithm
    using various methods including Sinkhorn-Knopp
    """
    
    @staticmethod
    def generate_from_communication_graph(adjacency_matrix: np.ndarray,
                                         method: str = 'sinkhorn-knopp',
                                         block_design: str = '2-block') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Z and W matrices from communication graph
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            method: Method to use ('sinkhorn-knopp', 'uniform', 'degree-normalized')
            block_design: Block structure ('2-block', 'full')
            
        Returns:
            Z, W matrices for proximal splitting
        """
        if method == 'sinkhorn-knopp':
            if block_design == '2-block':
                sk = SinkhornKnopp(adjacency_matrix)
                return sk.compute_2block_parameters()
            else:
                # Full design
                sk = SinkhornKnopp(adjacency_matrix)
                DS = sk.compute_doubly_stochastic()
                # Graph Laplacian construction
                Z = 2 * (np.eye(adjacency_matrix.shape[0]) - DS)
                W = Z.copy()
                return Z, W
                
        elif method == 'uniform':
            # Simple uniform weights on edges
            n = adjacency_matrix.shape[0]
            degrees = adjacency_matrix.sum(axis=1)
            
            if block_design == '2-block':
                # Uniform 2-block design
                Z = np.zeros((2*n, 2*n))
                Z[:n, :n] = np.eye(n)
                Z[n:, n:] = np.eye(n)
                
                # Off-diagonal blocks with uniform weights
                for i in range(n):
                    if degrees[i] > 0:
                        neighbors = np.where(adjacency_matrix[i] > 0)[0]
                        weight = 1.0 / degrees[i]
                        Z[i, n + neighbors] = -weight
                        Z[n + neighbors, i] = -weight
                
                Z = 2 * Z
                W = Z.copy()
                return Z, W
            else:
                # Full uniform design
                Z = np.zeros((n, n))
                for i in range(n):
                    Z[i, i] = 2.0
                    if degrees[i] > 0:
                        neighbors = np.where(adjacency_matrix[i] > 0)[0]
                        weight = 2.0 / degrees[i]
                        Z[i, neighbors] = -weight
                
                W = Z.copy()
                return Z, W
                
        elif method == 'degree-normalized':
            # Degree-normalized Laplacian
            n = adjacency_matrix.shape[0]
            degrees = adjacency_matrix.sum(axis=1)
            D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1)))
            L_norm = np.eye(n) - D_sqrt_inv @ adjacency_matrix @ D_sqrt_inv
            
            if block_design == '2-block':
                Z = np.zeros((2*n, 2*n))
                Z[:n, :n] = L_norm
                Z[n:, n:] = L_norm
                Z[:n, n:] = -np.eye(n) + L_norm
                Z[n:, :n] = -np.eye(n) + L_norm
                W = Z.copy()
                return Z, W
            else:
                Z = 2 * L_norm
                W = Z.copy()
                return Z, W
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def compute_lower_triangular_L(Z: np.ndarray) -> np.ndarray:
        """
        Compute lower triangular matrix L such that Z = 2I - L - L^T
        
        From the paper: The L matrix encodes the sequential dependencies
        in the proximal evaluations. It must be strictly lower triangular
        to ensure proper sequential structure.
        
        Args:
            Z: Matrix parameter (must satisfy diag(Z) = 2)
            
        Returns:
            Lower triangular matrix L
        """
        n = Z.shape[0]
        L = np.zeros((n, n))
        
        # From Z = 2I - L - L^T, we have:
        # - Diagonal: Z_ii = 2 - L_ii - L_ii = 2 - 2*L_ii
        #   Therefore: L_ii = (2 - Z_ii) / 2 = 0 (since Z_ii = 2)
        # - Off-diagonal: Z_ij = -L_ij - L_ji
        #   For i > j (lower triangular): L_ij = -Z_ij/2, L_ji = -Z_ij/2
        
        # Diagonal entries should be 0 for strictly lower triangular
        # (The paper uses strictly lower triangular L)
        
        # Off-diagonal entries (strictly lower triangular)
        for i in range(n):
            for j in range(i):
                # L is strictly lower triangular
                # From Z = 2I - L - L^T, for i > j:
                # Z_ij = -L_ij - L_ji = -L_ij (since L_ji = 0 for j > i)
                L[i, j] = -Z[i, j]
        
        return L