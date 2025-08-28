"""
Essential proximal operators for MPS algorithm
Simplified version with only necessary operators from the paper
"""

import numpy as np
from typing import Optional


class ProximalOperators:
    """Core proximal operators for sensor network localization"""
    
    @staticmethod
    def prox_distance(x: np.ndarray, 
                     target: np.ndarray, 
                     measured_dist: float,
                     alpha: float = 1.0) -> np.ndarray:
        """
        Proximal operator for distance constraint
        Projects x to satisfy ||x - target|| ≈ measured_dist
        
        Args:
            x: Current position estimate
            target: Target position (sensor or anchor)
            measured_dist: Measured distance
            alpha: Step size parameter
            
        Returns:
            Updated position
        """
        direction = x - target
        current_dist = np.linalg.norm(direction)
        
        if current_dist < 1e-10:
            # Handle zero distance case
            return target + measured_dist * np.random.randn(len(x)) / np.sqrt(len(x))
        
        # Project onto sphere of radius measured_dist centered at target
        error = (current_dist - measured_dist) / (current_dist + 1e-10)
        x_new = x - alpha * error * direction
        
        return x_new
    
    @staticmethod
    def prox_consensus(X: np.ndarray, Z: np.ndarray, U: np.ndarray, 
                      rho: float = 1.0) -> np.ndarray:
        """
        Proximal operator for consensus constraint in ADMM form
        
        Args:
            X: Current variable
            Z: Consensus variable  
            U: Dual variable
            rho: Penalty parameter
            
        Returns:
            Updated consensus variable
        """
        # Simple averaging for consensus
        return (X + U) / 2
    
    @staticmethod
    def prox_box_constraint(x: np.ndarray, 
                           lower: Optional[float] = None,
                           upper: Optional[float] = None) -> np.ndarray:
        """
        Project onto box constraints [lower, upper]
        
        Args:
            x: Input vector
            lower: Lower bound
            upper: Upper bound
            
        Returns:
            Projected vector
        """
        x_proj = x.copy()
        if lower is not None:
            x_proj = np.maximum(x_proj, lower)
        if upper is not None:
            x_proj = np.minimum(x_proj, upper)
        return x_proj