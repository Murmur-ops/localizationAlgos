"""
Proximal operators for sensor network localization
Real implementations without any mock data
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional


class ProximalOperators:
    """Proximal operators for MPS and ADMM algorithms"""
    
    @staticmethod
    def project_psd(X: np.ndarray, min_eigenvalue: float = 0.0) -> np.ndarray:
        """
        Project matrix onto positive semidefinite cone
        
        Args:
            X: Input matrix (must be square and symmetric)
            min_eigenvalue: Minimum eigenvalue threshold (default 0)
            
        Returns:
            Projected PSD matrix
        """
        # Ensure symmetry
        X = (X + X.T) / 2
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(X)
        
        # Project eigenvalues to be non-negative
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct matrix
        X_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return X_psd
    
    @staticmethod
    def prox_distance_constraint(x: np.ndarray, 
                                 target: np.ndarray, 
                                 measured_dist: float,
                                 alpha: float = 1.0,
                                 max_iter: int = 10) -> np.ndarray:
        """
        Proximal operator for distance constraint using gradient descent
        
        Minimizes: ||x - target||_2 ≈ measured_dist
        
        Args:
            x: Current position estimate
            target: Target position (sensor or anchor)
            measured_dist: Measured distance
            alpha: Step size parameter
            max_iter: Maximum gradient descent iterations
            
        Returns:
            Updated position
        """
        x_new = x.copy()
        
        for _ in range(max_iter):
            # Current distance to target
            direction = x_new - target
            current_dist = np.linalg.norm(direction)
            
            if current_dist < 1e-10:
                # Avoid division by zero
                break
            
            # Gradient of distance error
            error = current_dist - measured_dist
            grad = error * direction / current_dist
            
            # Gradient descent step
            x_new = x_new - alpha * grad
            
            # Check convergence
            if abs(error) < 1e-6:
                break
        
        return x_new
    
    @staticmethod
    def prox_consensus(X: np.ndarray, 
                      Y: np.ndarray,
                      rho: float = 1.0) -> np.ndarray:
        """
        Proximal operator for consensus constraint
        
        Minimizes: (1/2)||X - Y||^2 + rho * indicator(X in consensus)
        
        Args:
            X: Current variable
            Y: Target/auxiliary variable
            rho: Penalty parameter
            
        Returns:
            Consensus projection
        """
        # Simple averaging for consensus
        return (X + rho * Y) / (1 + rho)
    
    @staticmethod
    def prox_l1(x: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Proximal operator for L1 norm (soft thresholding)
        
        Args:
            x: Input vector/matrix
            lambda_: Regularization parameter
            
        Returns:
            Soft-thresholded result
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    @staticmethod
    def prox_l2_ball(x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """
        Project onto L2 ball
        
        Args:
            x: Point to project
            center: Center of ball
            radius: Radius of ball
            
        Returns:
            Projected point
        """
        diff = x - center
        norm_diff = np.linalg.norm(diff)
        
        if norm_diff <= radius:
            return x
        else:
            return center + radius * diff / norm_diff
    
    @staticmethod
    def prox_box_constraint(x: np.ndarray, 
                           lower: float = 0.0, 
                           upper: float = 1.0) -> np.ndarray:
        """
        Project onto box constraints
        
        Args:
            x: Input vector
            lower: Lower bound
            upper: Upper bound
            
        Returns:
            Projected vector
        """
        return np.clip(x, lower, upper)
    
    @staticmethod
    def prox_nuclear_norm(X: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Proximal operator for nuclear norm (singular value thresholding)
        
        Args:
            X: Input matrix
            lambda_: Regularization parameter
            
        Returns:
            Thresholded matrix
        """
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - lambda_, 0)
        return U @ np.diag(s_thresh) @ Vt
    
    @staticmethod
    def prox_indicator_affine(x: np.ndarray, 
                             A: np.ndarray, 
                             b: np.ndarray) -> np.ndarray:
        """
        Projection onto affine constraint Ax = b
        
        Args:
            x: Current point
            A: Constraint matrix
            b: Constraint vector
            
        Returns:
            Projected point
        """
        # Using pseudo-inverse for projection
        # x_proj = x - A.T @ (A @ A.T)^-1 @ (A @ x - b)
        
        Ax_minus_b = A @ x - b
        
        # Solve the system more stably
        try:
            # Use least squares for stability
            correction = np.linalg.lstsq(A @ A.T, Ax_minus_b, rcond=None)[0]
            x_proj = x - A.T @ correction
        except np.linalg.LinAlgError:
            # If singular, return original
            x_proj = x
        
        return x_proj
    
    @staticmethod
    def prox_elastic_net(x: np.ndarray, 
                        lambda1: float, 
                        lambda2: float) -> np.ndarray:
        """
        Proximal operator for elastic net regularization
        
        Minimizes: lambda1 * ||x||_1 + (lambda2/2) * ||x||_2^2
        
        Args:
            x: Input vector
            lambda1: L1 regularization
            lambda2: L2 regularization
            
        Returns:
            Regularized vector
        """
        # Soft thresholding with L2 scaling
        scale = 1 / (1 + lambda2)
        return scale * ProximalOperators.prox_l1(x, lambda1)