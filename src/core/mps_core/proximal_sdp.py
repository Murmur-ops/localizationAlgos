"""
Advanced Proximal Operators for SDP-based MPS Algorithm
Includes ADMM solver for regularized least absolute deviation
"""

import numpy as np
from scipy.linalg import eigh, cholesky, solve_triangular
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ProximalADMMSolver:
    """
    ADMM solver for the proximal operator of g_i
    Solves regularized least absolute deviation problem
    """
    
    def __init__(self, rho: float = 1.0, max_iterations: int = 100,
                 tolerance: float = 1e-6, warm_start: bool = True,
                 adaptive_penalty: bool = False):  # Disable adaptive by default for stability
        """
        Initialize ADMM solver
        
        Args:
            rho: ADMM penalty parameter
            max_iterations: Maximum ADMM iterations
            tolerance: Convergence tolerance
            warm_start: Whether to use warm starting
            adaptive_penalty: Whether to use adaptive penalty parameter
        """
        self.rho = rho
        self.initial_rho = rho
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.warm_start = warm_start
        self.adaptive_penalty = adaptive_penalty
        
        # Warm start variables
        self.lambda_prev = None
        self.y_prev = None
        
        # Cholesky factorization cache
        self.cholesky_cache = {}
        
        # Adaptive penalty parameters (Boyd et al.)
        self.mu = 10.0  # Ratio threshold
        self.tau_incr = 2.0  # Increase factor
        self.tau_decr = 2.0  # Decrease factor
    
    def setup_problem_matrices(self, sensor_idx: int, neighbors: List[int],
                              anchors: List[int], dimension: int) -> Dict:
        """
        Setup matrices for ADMM formulation as in equations (17)-(22) of paper
        
        Args:
            sensor_idx: Index of sensor i
            neighbors: List of neighbor indices
            anchors: List of anchor indices  
            dimension: Spatial dimension (2 or 3)
            
        Returns:
            Dictionary with problem matrices
        """
        n_neighbors = len(neighbors)
        n_anchors = len(anchors)
        
        # Vectorization dimension
        vec_dim = 1 + 2*n_neighbors + dimension
        
        # D matrix: doubles off-diagonal entries
        D_entries = [1]  # Y_ii
        for j in range(n_neighbors):
            D_entries.extend([1, np.sqrt(2)])  # Y_jj, Y_ij
        D_entries.extend([np.sqrt(2)] * dimension)  # X_i components
        D = np.diag(D_entries)
        
        # N matrix: extracts neighbor distance terms
        N = np.zeros((n_neighbors, 2*n_neighbors))
        for j in range(n_neighbors):
            N[j, 2*j] = 1      # Y_jj coefficient
            N[j, 2*j+1] = -2   # Y_ij coefficient
        N = np.hstack([np.ones((n_neighbors, 1)), N, np.zeros((n_neighbors, dimension))])
        
        # M matrix: anchor positions
        M = np.zeros((n_anchors, dimension))  # Will be filled with actual anchor positions
        
        # K matrix: combines sensor and anchor constraints
        K = np.vstack([N, np.hstack([np.ones((n_anchors, 1)), 
                                     np.zeros((n_anchors, 2*n_neighbors)), M])])
        
        return {
            'D': D,
            'N': N,
            'M': M,
            'K': K,
            'vec_dim': vec_dim,
            'n_constraints': n_neighbors + n_anchors
        }
    
    def solve(self, X_prev: np.ndarray, Y_prev: np.ndarray,
              sensor_idx: int, neighbors: List[int], anchors: List[int],
              distances_sensors: Dict, distances_anchors: Dict,
              anchor_positions: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve proximal operator using ADMM
        
        Args:
            X_prev: Previous X estimate
            Y_prev: Previous Y estimate
            sensor_idx: Sensor index
            neighbors: Neighbor indices
            anchors: Anchor indices
            distances_sensors: Distance measurements to sensors
            distances_anchors: Distance measurements to anchors
            anchor_positions: Anchor positions
            alpha: Proximal parameter
            
        Returns:
            Updated (X, Y) estimates
        """
        if len(neighbors) == 0 and len(anchors) == 0:
            return X_prev.copy(), Y_prev.copy()
        
        dimension = X_prev.shape[1]
        
        # Setup problem matrices
        matrices = self.setup_problem_matrices(sensor_idx, neighbors, anchors, dimension)
        
        # Fill in anchor positions
        for k_idx, k in enumerate(anchors):
            if k < len(anchor_positions):
                matrices['M'][k_idx] = anchor_positions[k]
        
        # Vectorize current estimates
        w_prev = self._vectorize_variables(X_prev[sensor_idx], Y_prev, sensor_idx, neighbors)
        
        # Setup distance vector
        c = np.zeros(matrices['n_constraints'])
        for j_idx, j in enumerate(neighbors):
            if j in distances_sensors:
                c[j_idx] = distances_sensors[j]**2
        for k_idx, k in enumerate(anchors):
            if k in distances_anchors:
                # Make sure anchor index is valid
                if k < len(anchor_positions):
                    c[len(neighbors) + k_idx] = distances_anchors[k]**2 - np.dot(anchor_positions[k], anchor_positions[k])
                else:
                    c[len(neighbors) + k_idx] = distances_anchors[k]**2
        
        # ADMM iterations
        K = matrices['K']
        D = matrices['D']
        vec_dim = matrices['vec_dim']
        
        # Compute Cholesky factorization with preconditioning
        cache_key = (sensor_idx, tuple(neighbors), tuple(anchors), self.rho)
        if cache_key not in self.cholesky_cache or self.adaptive_penalty:
            # Form matrix: rho * K^T K + D^T D / alpha
            A = self.rho * K.T @ K + D.T @ D / alpha
            
            # Add Tikhonov regularization for stability
            trace_A = np.trace(A)
            if trace_A > 0:
                lambda_reg = 1e-6 * trace_A / vec_dim
            else:
                lambda_reg = 1e-6
            A += lambda_reg * np.eye(vec_dim)
            
            # Check condition number
            try:
                cond = np.linalg.cond(A)
                if cond > 1e6:
                    # Apply diagonal preconditioning
                    D_precond = np.diag(1.0 / np.sqrt(np.maximum(np.diag(A), 1e-10)))
                    A = D_precond @ A @ D_precond
                    logger.debug(f"Applied preconditioning, condition number: {cond:.2e} -> {np.linalg.cond(A):.2e}")
            except:
                pass
            
            try:
                L_chol = cholesky(A, lower=True)
                if not self.adaptive_penalty:
                    self.cholesky_cache[cache_key] = L_chol
            except np.linalg.LinAlgError:
                # Fall back to stronger regularization
                A += 1e-4 * np.eye(vec_dim)
                L_chol = cholesky(A, lower=True)
        else:
            L_chol = self.cholesky_cache[cache_key]
        
        # Initialize ADMM variables with warm-starting
        # Check if warm start variables have correct dimensions
        if (self.warm_start and self.lambda_prev is not None and 
            len(self.lambda_prev) == matrices['n_constraints']):
            lambda_admm = self.lambda_prev.copy()
            y = self.y_prev.copy()
            logger.debug(f"Warm-starting ADMM for sensor {sensor_idx} "
                        f"(||lambda||={np.linalg.norm(lambda_admm):.3f})")
        else:
            lambda_admm = np.zeros(matrices['n_constraints'])
            y = np.zeros(matrices['n_constraints'])
            if self.warm_start and self.lambda_prev is not None:
                logger.debug(f"Warm-start dimension mismatch for sensor {sensor_idx}: "
                           f"expected {matrices['n_constraints']}, got {len(self.lambda_prev)}")
        
        w = w_prev.copy()
        
        # ADMM iterations
        for admm_iter in range(self.max_iterations):
            # w-update (equation 25 in paper)
            rhs = D.T @ D @ w_prev / alpha + self.rho * K.T @ (lambda_admm + c - y)
            w = solve_triangular(L_chol, rhs, lower=True)
            w = solve_triangular(L_chol.T, w, lower=False)
            
            # y-update (soft thresholding, equation 26)
            y = self._soft_threshold(c - K @ w + lambda_admm, 1.0/self.rho)
            
            # lambda-update (equation 27)
            lambda_new = lambda_admm - y - K @ w + c
            
            # Check convergence
            primal_residual = np.linalg.norm(y + K @ w - c)
            dual_residual = np.linalg.norm(lambda_new - lambda_admm)
            
            if primal_residual < self.tolerance and dual_residual < self.tolerance:
                logger.debug(f"ADMM converged at iteration {admm_iter+1} "
                           f"(primal={primal_residual:.2e}, dual={dual_residual:.2e})")
                break
            
            # Adaptive penalty update (Boyd et al.)
            if self.adaptive_penalty and admm_iter > 0 and admm_iter % 10 == 0:
                if primal_residual > self.mu * dual_residual:
                    self.rho = min(self.rho * self.tau_incr, 1e4)  # Cap at 1e4
                    # Need to recompute Cholesky with new rho
                    cache_key = None
                elif dual_residual > self.mu * primal_residual:
                    self.rho = max(self.rho / self.tau_decr, 1e-4)  # Floor at 1e-4
                    # Need to recompute Cholesky with new rho
                    cache_key = None
            
            lambda_admm = lambda_new
        
        # Store for warm start
        if self.warm_start:
            self.lambda_prev = lambda_admm.copy()
            self.y_prev = y.copy()
            logger.debug(f"Stored warm-start for sensor {sensor_idx} "
                        f"(iterations={admm_iter+1})")
        
        # Extract updated variables
        X_new = X_prev.copy()
        Y_new = Y_prev.copy()
        
        X_new[sensor_idx], Y_new = self._devectorize_variables(
            w, X_prev.shape, sensor_idx, neighbors
        )
        
        return X_new, Y_new
    
    def _vectorize_variables(self, x_i: np.ndarray, Y: np.ndarray,
                            sensor_idx: int, neighbors: List[int]) -> np.ndarray:
        """
        Vectorize variables for ADMM solver
        Format: [Y_ii, Y_j1j1, Y_ij1, ..., Y_jnjn, Y_ijn, X_i]
        """
        vec = [Y[sensor_idx, sensor_idx]]
        
        for j in neighbors:
            vec.append(Y[j, j])
            vec.append(Y[sensor_idx, j])
        
        vec.extend(x_i)
        
        return np.array(vec)
    
    def _devectorize_variables(self, w: np.ndarray, X_shape: Tuple,
                              sensor_idx: int, neighbors: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X and Y from vectorized form
        """
        n_sensors, dimension = X_shape
        Y = np.zeros((n_sensors, n_sensors))
        
        # Extract Y_ii
        Y[sensor_idx, sensor_idx] = w[0]
        
        # Extract neighbor terms
        idx = 1
        for j in neighbors:
            Y[j, j] = w[idx]
            Y[sensor_idx, j] = w[idx + 1]
            Y[j, sensor_idx] = w[idx + 1]  # Maintain symmetry
            idx += 2
        
        # Extract X_i
        x_i = w[idx:idx + dimension]
        
        return x_i, Y
    
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft thresholding operator
        soft_τ(x) = sign(x) * max(|x| - τ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _huber_prox(self, x: np.ndarray, delta: float, lambda_param: float) -> np.ndarray:
        """
        Proximal operator for Huber loss
        Provides robustness to outliers
        
        Args:
            x: Input vector
            delta: Huber threshold
            lambda_param: Proximal parameter
            
        Returns:
            Proximal operator result
        """
        # For |x| <= delta + lambda: quadratic region
        # For |x| > delta + lambda: linear region
        abs_x = np.abs(x)
        threshold = delta + lambda_param
        
        result = np.zeros_like(x)
        
        # Quadratic region
        quad_mask = abs_x <= threshold
        result[quad_mask] = x[quad_mask] / (1 + lambda_param)
        
        # Linear region
        lin_mask = abs_x > threshold
        result[lin_mask] = x[lin_mask] - lambda_param * np.sign(x[lin_mask])
        
        return result


class ProximalOperatorsPSD:
    """
    Complete set of proximal operators for SDP-based algorithm
    """
    
    def __init__(self, admm_solver: Optional[ProximalADMMSolver] = None):
        """
        Initialize proximal operators
        
        Args:
            admm_solver: ADMM solver instance (creates default if None)
        """
        self.admm_solver = admm_solver or ProximalADMMSolver()
    
    @staticmethod
    def project_psd_cone(matrix: np.ndarray, regularization: float = 0.0) -> np.ndarray:
        """
        Project matrix onto positive semidefinite cone
        
        Args:
            matrix: Input matrix (must be symmetric)
            regularization: Small positive value to ensure strict positivity
            
        Returns:
            Projected PSD matrix
        """
        # Ensure symmetry
        matrix_sym = (matrix + matrix.T) / 2
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = eigh(matrix_sym)
        except np.linalg.LinAlgError:
            # Fall back to SVD if eigendecomposition fails
            U, S, Vt = np.linalg.svd(matrix_sym)
            eigenvalues = S
            eigenvectors = U
        
        # Keep only non-negative eigenvalues (with optional regularization)
        eigenvalues = np.maximum(eigenvalues, regularization)
        
        # Reconstruct matrix
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def prox_objective_gi(self, X: np.ndarray, Y: np.ndarray,
                         sensor_idx: int, neighbors: List[int],
                         anchors: List[int], distances_sensors: Dict,
                         distances_anchors: Dict, anchor_positions: np.ndarray,
                         alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proximal operator for objective g_i using ADMM solver
        
        Args:
            X: Current position estimates
            Y: Current Y matrix
            sensor_idx: Sensor index
            neighbors: Neighbor indices
            anchors: Anchor indices
            distances_sensors: Distance measurements to sensors
            distances_anchors: Distance measurements to anchors
            anchor_positions: Anchor positions
            alpha: Proximal parameter
            
        Returns:
            Updated (X, Y) estimates
        """
        return self.admm_solver.solve(
            X, Y, sensor_idx, neighbors, anchors,
            distances_sensors, distances_anchors,
            anchor_positions, alpha
        )
    
    @staticmethod
    def prox_consensus(X: np.ndarray, Z: np.ndarray, U: np.ndarray,
                      rho: float = 1.0) -> np.ndarray:
        """
        Proximal operator for consensus constraint
        
        Args:
            X: Current variable
            Z: Consensus variable
            U: Dual variable
            rho: Penalty parameter
            
        Returns:
            Updated consensus variable
        """
        return (X + U) / 2
    
    @staticmethod
    def prox_nuclear_norm(matrix: np.ndarray, threshold: float) -> np.ndarray:
        """
        Proximal operator for nuclear norm (sum of singular values)
        Used for low-rank approximation
        
        Args:
            matrix: Input matrix
            threshold: Soft threshold parameter
            
        Returns:
            Soft-thresholded matrix
        """
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S_threshold = np.maximum(S - threshold, 0)
        return U @ np.diag(S_threshold) @ Vt
    
    @staticmethod
    def prox_frobenius_ball(matrix: np.ndarray, center: np.ndarray,
                           radius: float) -> np.ndarray:
        """
        Project onto Frobenius norm ball
        
        Args:
            matrix: Input matrix
            center: Center of ball
            radius: Radius of ball
            
        Returns:
            Projected matrix
        """
        diff = matrix - center
        norm = np.linalg.norm(diff, 'fro')
        
        if norm <= radius:
            return matrix
        else:
            return center + radius * diff / norm
    
    @staticmethod
    def prox_box_constraint(x: np.ndarray, lower: Optional[np.ndarray] = None,
                          upper: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project onto box constraints
        
        Args:
            x: Input vector/matrix
            lower: Lower bounds
            upper: Upper bounds
            
        Returns:
            Projected vector/matrix
        """
        x_proj = x.copy()
        
        if lower is not None:
            x_proj = np.maximum(x_proj, lower)
        
        if upper is not None:
            x_proj = np.minimum(x_proj, upper)
        
        return x_proj