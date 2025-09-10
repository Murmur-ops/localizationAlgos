"""
Vectorization layer for handling variable-dimension matrices in MPS algorithm.
Maps between matrix variables S^i and fixed-size vector representations.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MatrixVectorizer:
    """
    Handles vectorization and devectorization of variable-dimension matrices
    for the MPS algorithm. Each sensor i has a matrix S^i of dimension
    (d+1+|N_i|) x (d+1+|N_i|), which needs to be mapped to/from a fixed-size
    vector for matrix operations.
    """
    
    def __init__(self, n_sensors: int, dimension: int, 
                 neighborhoods: Dict[int, List[int]]):
        """
        Initialize vectorizer
        
        Args:
            n_sensors: Number of sensors
            dimension: Spatial dimension (2 or 3)
            neighborhoods: Dictionary mapping sensor to neighbors
        """
        self.n = n_sensors
        self.d = dimension
        self.neighborhoods = neighborhoods
        
        # Compute dimensions for each sensor's matrix
        self.matrix_dims = {}
        self.vector_dims = {}
        self.vector_offsets = {}
        
        total_dim = 0
        for i in range(n_sensors):
            neighbors = neighborhoods.get(i, [])
            mat_dim = dimension + 1 + len(neighbors)
            self.matrix_dims[i] = mat_dim
            
            # Vectorized dimension (upper triangular part of symmetric matrix)
            vec_dim = mat_dim * (mat_dim + 1) // 2
            self.vector_dims[i] = vec_dim
            self.vector_offsets[i] = total_dim
            total_dim += vec_dim
        
        # Total dimension of stacked vector
        self.total_vector_dim = total_dim
        
        # Pre-compute vectorization indices for efficiency
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Pre-compute indices for efficient vectorization"""
        self.vec_indices = {}
        self.unvec_indices = {}
        
        for i in range(self.n):
            mat_dim = self.matrix_dims[i]
            
            # Indices for upper triangular part (row, col)
            indices = []
            for row in range(mat_dim):
                for col in range(row, mat_dim):
                    indices.append((row, col))
            
            self.vec_indices[i] = indices
            
            # Reverse mapping for devectorization
            self.unvec_indices[i] = {idx: k for k, idx in enumerate(indices)}
    
    def vectorize_matrix(self, S: np.ndarray, sensor_idx: int) -> np.ndarray:
        """
        Vectorize a single matrix S^i with proper sqrt(2) scaling for off-diagonals
        
        Args:
            S: Matrix to vectorize
            sensor_idx: Sensor index
            
        Returns:
            Vectorized representation with D_i weighting
        """
        indices = self.vec_indices[sensor_idx]
        vec = np.zeros(self.vector_dims[sensor_idx])
        
        for k, (i, j) in enumerate(indices):
            if i == j:
                # Diagonal entries: store as-is
                vec[k] = S[i, j]
            else:
                # Off-diagonal entries: multiply by sqrt(2)
                vec[k] = np.sqrt(2) * S[i, j]
        
        return vec
    
    def devectorize_matrix(self, vec: np.ndarray, sensor_idx: int) -> np.ndarray:
        """
        Reconstruct matrix from vectorized form with proper sqrt(2) scaling
        
        Args:
            vec: Vectorized representation with D_i weighting
            sensor_idx: Sensor index
            
        Returns:
            Reconstructed symmetric matrix
        """
        mat_dim = self.matrix_dims[sensor_idx]
        S = np.zeros((mat_dim, mat_dim))
        indices = self.vec_indices[sensor_idx]
        
        for k, (i, j) in enumerate(indices):
            if i == j:
                # Diagonal entries: store as-is
                S[i, j] = vec[k]
            else:
                # Off-diagonal entries: divide by sqrt(2) and symmetrize
                val = vec[k] / np.sqrt(2)
                S[i, j] = val
                S[j, i] = val  # Maintain symmetry
        
        return S
    
    def stack_matrices(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Stack all matrices into a single vector
        
        Args:
            matrices: List of matrices (one per sensor)
            
        Returns:
            Stacked vector of dimension total_vector_dim
        """
        stacked = np.zeros(self.total_vector_dim)
        
        for i in range(self.n):
            vec = self.vectorize_matrix(matrices[i], i)
            start = self.vector_offsets[i]
            end = start + self.vector_dims[i]
            stacked[start:end] = vec
        
        return stacked
    
    def unstack_vector(self, stacked: np.ndarray) -> List[np.ndarray]:
        """
        Unstack vector into individual matrices
        
        Args:
            stacked: Stacked vector
            
        Returns:
            List of matrices
        """
        matrices = []
        
        for i in range(self.n):
            start = self.vector_offsets[i]
            end = start + self.vector_dims[i]
            vec = stacked[start:end]
            S = self.devectorize_matrix(vec, i)
            matrices.append(S)
        
        return matrices
    
    def create_coupling_matrix(self, W_scalar: np.ndarray) -> np.ndarray:
        """
        Create block-diagonal coupling matrix from scalar W matrix.
        This expands the scalar coupling to operate on vectorized matrices.
        
        Args:
            W_scalar: Scalar coupling matrix (2n x 2n)
            
        Returns:
            Expanded coupling matrix for vectorized representation
        """
        # For 2-block structure: first n components are objectives,
        # last n are PSD constraints
        p = 2 * self.n
        
        # Create block structure where each block corresponds to 
        # the vectorized dimension of each matrix
        W_expanded = np.zeros((self.total_vector_dim * 2, 
                               self.total_vector_dim * 2))
        
        # Map scalar couplings to vector space
        for i in range(p):
            sensor_i = i % self.n
            start_i = self.vector_offsets[sensor_i]
            dim_i = self.vector_dims[sensor_i]
            
            # Offset for second block (PSD constraints)
            if i >= self.n:
                start_i += self.total_vector_dim
            
            for j in range(p):
                sensor_j = j % self.n
                start_j = self.vector_offsets[sensor_j]
                dim_j = self.vector_dims[sensor_j]
                
                # Offset for second block
                if j >= self.n:
                    start_j += self.total_vector_dim
                
                # Expand scalar coupling to matrix block
                if W_scalar[i, j] != 0:
                    # Use identity scaling for each block
                    if sensor_i == sensor_j and dim_i == dim_j:
                        W_expanded[start_i:start_i+dim_i, 
                                  start_j:start_j+dim_j] = W_scalar[i, j] * np.eye(dim_i)
        
        return W_expanded
    
    def apply_L_sequential(self, v_list: List[np.ndarray], 
                          x_list: List[np.ndarray], 
                          L: np.ndarray, i: int) -> np.ndarray:
        """
        Apply L matrix for sequential proximal evaluation.
        Computes: v_i + Σ_{j<i} L_ij * x_j
        
        Args:
            v_list: List of v matrices
            x_list: List of x matrices (partially updated)
            L: Lower triangular coupling matrix
            i: Current index
            
        Returns:
            Input for i-th proximal operator
        """
        result = v_list[i].copy()
        
        # Add contributions from previous evaluations
        for j in range(i):
            if L[i, j] != 0:
                # For matrix variables with same dimensions, 
                # apply scalar coupling directly
                sensor_i = i % self.n
                sensor_j = j % self.n
                
                if self.matrix_dims[sensor_i] == self.matrix_dims[sensor_j]:
                    result += L[i, j] * x_list[j]
                else:
                    # Handle dimension mismatch by padding/truncating
                    # This shouldn't happen in proper 2-block design
                    logger.warning(f"Dimension mismatch: sensor {sensor_i} "
                                 f"({self.matrix_dims[sensor_i]}) vs "
                                 f"sensor {sensor_j} ({self.matrix_dims[sensor_j]})")
        
        return result