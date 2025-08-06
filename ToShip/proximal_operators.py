"""
Proximal operators for Sensor Network Localization
Implements the proximal operators needed for the MPS algorithm
"""

import numpy as np
import scipy.linalg as la
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ProximalOperators:
    """Class containing proximal operator implementations"""
    
    def __init__(self, problem):
        self.problem = problem
        self.cholesky_cache = {}  # Cache Cholesky factorizations
    
    def construct_vectorization(self, sensor_data, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Implement vectorization for the algorithm
        vec_i = [Yii, Yj1j1, Yij1, ..., Yj|Ni|j|Ni|, Yij|Ni|, Xi]
        """
        neighbors = sensor_data.neighbors
        n_neighbors = len(neighbors)
        
        # Build vectorization
        vec_parts = [Y[0, 0]]  # Yii
        
        # Add Yjj and Yij for each neighbor
        for j_idx, j in enumerate(neighbors):
            vec_parts.append(Y[j_idx + 1, j_idx + 1])  # Yjj
            vec_parts.append(Y[0, j_idx + 1])  # Yij
        
        # Add Xi (position)
        vec = np.concatenate([vec_parts, X])
        
        return vec
    
    def construct_matrices(self, sensor_id: int, sensor_data, anchor_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build Di, Ki, ci, Ni, Mi matrices from equations (19-22)
        """
        neighbors = sensor_data.neighbors
        anchor_neighbors = sensor_data.anchor_neighbors
        n_neighbors = len(neighbors)
        n_anchors = len(anchor_neighbors)
        d = self.problem.d
        
        # Di matrix - doubles off-diagonal entries (equation 19)
        vec_size = 1 + 2 * n_neighbors + d
        Di_diag = [1.0]  # Yii
        for j in range(n_neighbors):
            Di_diag.extend([1.0, np.sqrt(2)])  # Yjj, Yij
        Di_diag.extend([np.sqrt(2)] * d)  # Xi components
        Di = np.diag(Di_diag)
        
        # Ni matrix (equation 20)
        if n_neighbors > 0:
            Ni = np.zeros((n_neighbors, 2 * n_neighbors))
            for i in range(n_neighbors):
                Ni[i, 2*i] = 1      # coefficient for Yjj
                Ni[i, 2*i + 1] = -2  # coefficient for Yij
        else:
            Ni = np.zeros((0, 0))
        
        # Mi matrix - anchor positions (equation 21)
        if n_anchors > 0:
            Mi = np.zeros((n_anchors, d))
            for i, k in enumerate(anchor_neighbors):
                Mi[i] = anchor_positions[k]
        else:
            Mi = np.zeros((0, d))
        
        # Ki matrix (equation 22)
        Ki = np.zeros((n_neighbors + n_anchors, vec_size))
        
        # First n_neighbors rows: [1, Ni, 0]
        if n_neighbors > 0:
            Ki[:n_neighbors, 0] = 1  # coefficient for Yii
            Ki[:n_neighbors, 1:1+2*n_neighbors] = Ni
        
        # Next n_anchors rows: [1, 0, Mi]
        if n_anchors > 0:
            Ki[n_neighbors:, 0] = 1  # coefficient for Yii
            Ki[n_neighbors:, -d:] = Mi
        
        # ci vector (equation 18)
        ci = np.zeros(n_neighbors + n_anchors)
        
        # Distance measurements to neighbors
        for i, j in enumerate(neighbors):
            if j in sensor_data.distance_measurements:
                ci[i] = sensor_data.distance_measurements[j]**2
        
        # Distance measurements to anchors
        for i, k in enumerate(anchor_neighbors):
            if k in sensor_data.anchor_distances:
                dist = sensor_data.anchor_distances[k]
                anchor_pos = anchor_positions[k]
                ci[n_neighbors + i] = dist**2 - np.linalg.norm(anchor_pos)**2
        
        return Di, Ki, ci, Ni, Mi
    
    def prox_gi_admm(self, sensor_id: int, sensor_data, X_k: np.ndarray, Y_k: np.ndarray,
                     anchor_positions: np.ndarray, alpha: float, 
                     max_iter: int = 50, rho: float = 1.0, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve proximal operator for gi using ADMM
        Implements the distance constraint optimization
        
        Solves: min_w ||c^k - K_i w||_1 + (1/2)||D_i w||^2
        where w = vec_i(X,Y) - vec_i(X^k, Y^k)
        """
        # Construct matrices
        Di, Ki, ci, Ni, Mi = self.construct_matrices(sensor_id, sensor_data, anchor_positions)
        
        # Vectorize current point
        vec_k = self.construct_vectorization(sensor_data, X_k, Y_k)
        
        # Scale by alpha
        Di_scaled = Di / np.sqrt(alpha)
        
        # Update ci based on current point
        ci_k = ci + Ki @ vec_k
        
        # Initialize ADMM variables
        n_measurements = len(ci_k)
        vec_size = len(vec_k)
        
        if n_measurements == 0:  # No measurements
            return X_k, Y_k
        
        # Check cache for Cholesky factorization
        cache_key = (sensor_id, vec_size)
        if cache_key not in self.cholesky_cache:
            # Compute and cache Cholesky factorization
            A = rho * Ki.T @ Ki + Di_scaled.T @ Di_scaled
            try:
                L = la.cholesky(A, lower=True)
                self.cholesky_cache[cache_key] = L
            except la.LinAlgError:
                # Add regularization if not positive definite
                reg = 1e-6
                while reg < 1.0:
                    try:
                        A_reg = A + reg * np.eye(vec_size)
                        L = la.cholesky(A_reg, lower=True)
                        self.cholesky_cache[cache_key] = L
                        break
                    except la.LinAlgError:
                        reg *= 10
                if reg >= 1.0:
                    logger.warning(f"Cholesky failed for sensor {sensor_id}, using identity")
                    L = np.eye(vec_size)
                    self.cholesky_cache[cache_key] = L
        else:
            L = self.cholesky_cache[cache_key]
        
        # Initialize ADMM variables
        lambda_vec = np.zeros(n_measurements)
        y = np.zeros(n_measurements)
        w = np.zeros(vec_size)
        
        # Track convergence
        primal_residuals = []
        dual_residuals = []
        
        # ADMM iterations
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # w-update (equation 25)
            b = rho * Ki.T @ (ci_k - y + lambda_vec)
            w = la.solve_triangular(L.T, la.solve_triangular(L, b, lower=True))
            
            # y-update with soft thresholding (equation 26)
            z = ci_k - Ki @ w + lambda_vec
            y_old = y.copy()
            y = np.sign(z) * np.maximum(np.abs(z) - 1/(alpha * rho), 0)
            
            # Lambda update (equation 27)
            primal_residual = y + Ki @ w - ci_k
            lambda_vec = lambda_vec - primal_residual
            
            # Compute dual residual
            dual_residual = rho * Ki.T @ (y - y_old)
            
            # Store residuals
            primal_residuals.append(np.linalg.norm(primal_residual))
            dual_residuals.append(np.linalg.norm(dual_residual))
            
            # Check convergence
            eps_pri = np.sqrt(n_measurements) * tol + tol * max(
                np.linalg.norm(Ki @ w), np.linalg.norm(y), np.linalg.norm(ci_k)
            )
            eps_dual = np.sqrt(vec_size) * tol + tol * np.linalg.norm(rho * Ki.T @ lambda_vec)
            
            if primal_residuals[-1] < eps_pri and dual_residuals[-1] < eps_dual:
                break
        
        # Extract X and Y from solution
        n_neighbors = len(sensor_data.neighbors)
        
        # Update from vec_k - w
        vec_new = vec_k - w
        
        # Extract components
        X_new = vec_new[-self.problem.d:]
        
        # Reconstruct Y matrix
        Y_new = np.zeros((n_neighbors + 1, n_neighbors + 1))
        Y_new[0, 0] = vec_new[0]  # Yii
        
        idx = 1
        for j_idx in range(n_neighbors):
            Y_new[j_idx + 1, j_idx + 1] = vec_new[idx]  # Yjj
            Y_new[0, j_idx + 1] = vec_new[idx + 1]  # Yij
            Y_new[j_idx + 1, 0] = vec_new[idx + 1]  # Yji (symmetric)
            idx += 2
        
        return X_new, Y_new
    
    def prox_indicator_psd(self, S: np.ndarray) -> np.ndarray:
        """
        Project onto PSD cone using eigendecomposition
        """
        # Ensure symmetric
        S = (S + S.T) / 2
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = la.eigh(S)
        except la.LinAlgError:
            # If decomposition fails, return closest diagonal PSD matrix
            logger.warning("Eigendecomposition failed, using diagonal approximation")
            diag_S = np.diag(np.maximum(np.diag(S), 0))
            return diag_S
        
        # Project eigenvalues to non-negative
        eigenvalues_plus = np.maximum(eigenvalues, 0)
        
        # Reconstruct
        S_proj = eigenvectors @ np.diag(eigenvalues_plus) @ eigenvectors.T
        
        # Ensure symmetric
        S_proj = (S_proj + S_proj.T) / 2
        
        return S_proj
    
    def construct_Si(self, X: np.ndarray, Y: np.ndarray, d: int) -> np.ndarray:
        """
        Build principal submatrix Si for sensor i
        Constructs the matrix for distance constraints
        """
        n = Y.shape[0]  # 1 + number of neighbors
        Si = np.zeros((d + n, d + n))
        
        # Upper left: Id
        Si[:d, :d] = np.eye(d)
        
        # Upper right: X^T (each column is a position)
        # For sensor i and its neighbors
        Si[:d, d] = X  # Position of sensor i
        
        # Lower left: X
        Si[d, :d] = X
        
        # Lower right: Y
        Si[d:, d:] = Y
        
        return Si
    
    def extract_from_Si(self, Si_proj: np.ndarray, sensor_id: int, sensor_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X and Y updates from projected Si matrix
        """
        d = self.problem.d
        
        # Extract X (position of sensor i)
        X_new = Si_proj[d, :d].copy()
        
        # Extract Y
        Y_new = Si_proj[d:, d:].copy()
        
        # Ensure Y is symmetric
        Y_new = (Y_new + Y_new.T) / 2
        
        return X_new, Y_new


class WarmStart:
    """Class for warm starting the algorithms"""
    
    @staticmethod
    def from_previous_solution(X_prev: np.ndarray, Y_prev: np.ndarray, 
                             velocity: Optional[np.ndarray] = None, 
                             dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warm start from previous solution, optionally with velocity estimate
        """
        if velocity is not None:
            X_warm = X_prev + velocity * dt
        else:
            X_warm = X_prev.copy()
        
        # Simple warm start for Y
        Y_warm = Y_prev.copy()
        
        return X_warm, Y_warm
    
    @staticmethod
    def from_rough_estimate(X_rough: np.ndarray, d: int, n_neighbors: int, 
                          noise_std: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize from rough position estimate with added noise
        """
        # Add Gaussian noise to position
        X_warm = X_rough + np.random.normal(0, noise_std, size=d)
        
        # Initialize Y based on rough positions
        Y_warm = np.zeros((n_neighbors + 1, n_neighbors + 1))
        
        # Diagonal elements (squared norms)
        Y_warm[0, 0] = np.linalg.norm(X_warm)**2
        
        return X_warm, Y_warm


class MatrixParameterVerification:
    """Verify that matrix parameters satisfy required conditions"""
    
    @staticmethod
    def verify_equation_8(Z: np.ndarray, W: np.ndarray, tol: float = 1e-6) -> dict:
        """
        Check all required conditions for the algorithm:
        - Z ⪰ W
        - null(W) = span(1)
        - diag(Z) = 2*1
        - 1^T Z 1 = 0
        """
        results = {
            'Z_succeq_W': False,
            'null_W_correct': False,
            'diag_Z_correct': False,
            'sum_Z_zero': False,
            'all_satisfied': False
        }
        
        n = Z.shape[0]
        ones = np.ones(n)
        
        # Check Z ⪰ W
        try:
            eigenvalues = la.eigvalsh(Z - W)
            results['Z_succeq_W'] = np.all(eigenvalues >= -tol)
        except la.LinAlgError:
            results['Z_succeq_W'] = False
        
        # Check null(W) = span(1)
        W_eigenvalues = la.eigvalsh(W)
        null_dim = np.sum(np.abs(W_eigenvalues) < tol)
        W_ones = W @ ones
        results['null_W_correct'] = (null_dim == 1) and (np.linalg.norm(W_ones) < tol)
        
        # Check diag(Z) = 2*1
        results['diag_Z_correct'] = np.allclose(np.diag(Z), 2 * ones, atol=tol)
        
        # Check 1^T Z 1 = 0
        sum_Z = ones.T @ Z @ ones
        results['sum_Z_zero'] = abs(sum_Z) < tol
        
        # Overall check
        results['all_satisfied'] = all([
            results['Z_succeq_W'],
            results['null_W_correct'],
            results['diag_Z_correct'],
            results['sum_Z_zero']
        ])
        
        return results
    
    @staticmethod
    def verify_2block_structure(Z: np.ndarray, W: np.ndarray, 
                               adjacency: np.ndarray) -> bool:
        """
        Verify that Z and W adhere to 2-block structure and communication graph
        """
        n = adjacency.shape[0]
        n_half = n // 2
        
        # Check block structure
        for i in range(n_half):
            for j in range(n_half):
                if i != j and adjacency[i, j] == 0:
                    # Check that corresponding entries are zero
                    if (abs(Z[i, j]) > 1e-10 or abs(Z[i+n_half, j+n_half]) > 1e-10 or
                        abs(W[i, j]) > 1e-10 or abs(W[i+n_half, j+n_half]) > 1e-10):
                        return False
        
        return True