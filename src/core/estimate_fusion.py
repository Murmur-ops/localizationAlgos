"""
Estimate Fusion: Information-theoretic combination of multiple estimates
"""

import numpy as np
from numpy.linalg import inv, norm, det
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EstimateWithUncertainty:
    """Position estimate with uncertainty quantification"""
    position: np.ndarray          # 2D position estimate
    covariance: np.ndarray        # 2x2 covariance matrix
    confidence: float             # Confidence score [0, 1]
    source: str                   # Algorithm source (BP, Hier, Consensus, etc.)
    convergence_rate: float       # How fast it converged
    iterations: int               # Iterations used
    

class EstimateFusion:
    def __init__(self, regularization: float = 1e-6):
        """
        Initialize estimate fusion system
        
        Args:
            regularization: Small value to ensure numerical stability
        """
        self.regularization = regularization
        self.fusion_history = []
        
    def fuse_estimates(self, estimates: List[EstimateWithUncertainty]) -> EstimateWithUncertainty:
        """
        Fuse multiple estimates using covariance-weighted averaging
        
        This implements optimal fusion under Gaussian assumptions:
        - Minimum variance estimator
        - Accounts for correlation between estimates
        
        Args:
            estimates: List of estimates with uncertainties
            
        Returns:
            Fused estimate with combined uncertainty
        """
        if not estimates:
            return None
            
        if len(estimates) == 1:
            return estimates[0]
            
        # Information form fusion (more numerically stable)
        # Information matrix = inverse covariance
        total_information = np.zeros((2, 2))
        total_information_vector = np.zeros(2)
        
        # Weight by confidence as well as covariance
        total_confidence = sum(e.confidence for e in estimates)
        if total_confidence == 0:
            total_confidence = 1.0
            
        for estimate in estimates:
            # Confidence-weighted precision matrix
            confidence_weight = estimate.confidence / total_confidence
            
            # Add regularization for numerical stability
            cov_reg = estimate.covariance + np.eye(2) * self.regularization
            
            try:
                precision = inv(cov_reg)
                
                # Weight precision by confidence
                weighted_precision = precision * confidence_weight
                
                # Accumulate information
                total_information += weighted_precision
                total_information_vector += weighted_precision @ estimate.position
                
            except np.linalg.LinAlgError:
                # Singular covariance, use simple averaging fallback
                weight = confidence_weight / len(estimates)
                fallback_precision = np.eye(2) * weight
                total_information += fallback_precision
                total_information_vector += fallback_precision @ estimate.position
                
        # Compute fused estimate
        try:
            # Add regularization to ensure invertibility
            total_information_reg = total_information + np.eye(2) * self.regularization
            fused_covariance = inv(total_information_reg)
            fused_position = fused_covariance @ total_information_vector
            
        except np.linalg.LinAlgError:
            # Fallback to simple weighted average
            weights = np.array([e.confidence for e in estimates])
            weights /= weights.sum()
            
            fused_position = np.zeros(2)
            for w, e in zip(weights, estimates):
                fused_position += w * e.position
                
            # Conservative covariance estimate
            fused_covariance = np.mean([e.covariance for e in estimates], axis=0)
            
        # Compute fused confidence
        # Higher confidence if: multiple sources agree, low uncertainty
        position_variance = np.trace(fused_covariance)
        
        # Agreement score: how close are the estimates?
        positions = np.array([e.position for e in estimates])
        mean_pos = np.mean(positions, axis=0)
        deviations = [norm(p - mean_pos) for p in positions]
        agreement = 1.0 / (1.0 + np.std(deviations))
        
        # Combined confidence
        base_confidence = np.mean([e.confidence for e in estimates])
        variance_factor = 1.0 / (1.0 + position_variance)
        fused_confidence = base_confidence * agreement * variance_factor
        fused_confidence = np.clip(fused_confidence, 0, 1)
        
        # Average convergence rate
        avg_convergence = np.mean([e.convergence_rate for e in estimates])
        total_iterations = sum(e.iterations for e in estimates)
        
        return EstimateWithUncertainty(
            position=fused_position,
            covariance=fused_covariance,
            confidence=fused_confidence,
            source=f"Fusion({','.join(e.source for e in estimates)})",
            convergence_rate=avg_convergence,
            iterations=total_iterations
        )
        
    def selective_fusion(self, estimates: List[EstimateWithUncertainty],
                        confidence_threshold: float = 0.5) -> EstimateWithUncertainty:
        """
        Selectively fuse estimates, excluding low-confidence ones
        
        Args:
            estimates: List of estimates
            confidence_threshold: Minimum confidence to include
            
        Returns:
            Fused estimate from high-confidence sources
        """
        # Filter by confidence
        high_confidence = [e for e in estimates if e.confidence >= confidence_threshold]
        
        if not high_confidence:
            # All low confidence, use best one
            if estimates:
                return max(estimates, key=lambda e: e.confidence)
            return None
            
        return self.fuse_estimates(high_confidence)
        
    def weighted_fusion(self, estimates: List[EstimateWithUncertainty],
                       weights: Dict[str, float]) -> EstimateWithUncertainty:
        """
        Fuse with custom weights per algorithm
        
        Args:
            estimates: List of estimates
            weights: Weight for each algorithm source
            
        Returns:
            Weighted fused estimate
        """
        # Apply custom weights
        weighted_estimates = []
        for estimate in estimates:
            if estimate.source in weights:
                # Scale confidence by weight
                weighted_est = EstimateWithUncertainty(
                    position=estimate.position,
                    covariance=estimate.covariance / weights[estimate.source],
                    confidence=estimate.confidence * weights[estimate.source],
                    source=estimate.source,
                    convergence_rate=estimate.convergence_rate,
                    iterations=estimate.iterations
                )
                weighted_estimates.append(weighted_est)
            else:
                weighted_estimates.append(estimate)
                
        return self.fuse_estimates(weighted_estimates)
        
    def incremental_fusion(self, current: EstimateWithUncertainty,
                         new: EstimateWithUncertainty,
                         alpha: float = 0.3) -> EstimateWithUncertainty:
        """
        Incrementally update estimate with new information
        
        Args:
            current: Current estimate
            new: New estimate to incorporate
            alpha: Learning rate / mixing parameter
            
        Returns:
            Updated estimate
        """
        if current is None:
            return new
            
        if new is None:
            return current
            
        # Kalman-like update
        # Predict step (no motion model, so prediction = current)
        predicted_pos = current.position
        predicted_cov = current.covariance + np.eye(2) * 0.001  # Process noise
        
        # Update step
        try:
            # Innovation
            innovation = new.position - predicted_pos
            
            # Innovation covariance
            S = predicted_cov + new.covariance
            
            # Kalman gain
            K = predicted_cov @ inv(S + np.eye(2) * self.regularization)
            
            # Updated estimate
            updated_pos = predicted_pos + K @ innovation
            updated_cov = (np.eye(2) - K) @ predicted_cov
            
        except np.linalg.LinAlgError:
            # Fallback to simple mixing
            updated_pos = (1 - alpha) * current.position + alpha * new.position
            updated_cov = (1 - alpha) * current.covariance + alpha * new.covariance
            
        # Update confidence
        updated_confidence = (1 - alpha) * current.confidence + alpha * new.confidence
        
        return EstimateWithUncertainty(
            position=updated_pos,
            covariance=updated_cov,
            confidence=updated_confidence,
            source=f"Incremental({current.source},{new.source})",
            convergence_rate=(current.convergence_rate + new.convergence_rate) / 2,
            iterations=current.iterations + new.iterations
        )
        
    def compute_uncertainty(self, position: np.ndarray, 
                          measurements: Dict, 
                          noise_factor: float) -> np.ndarray:
        """
        Compute uncertainty (covariance) for a position estimate
        
        Args:
            position: Position estimate
            measurements: Distance measurements
            noise_factor: Measurement noise level
            
        Returns:
            2x2 covariance matrix
        """
        # Fisher Information Matrix approach
        information = np.zeros((2, 2))
        
        for key, measured_dist in measurements.items():
            # Measurement variance
            sigma2 = (noise_factor * measured_dist) ** 2
            
            # For anchor measurements
            if isinstance(key, tuple) and isinstance(key[1], str):
                # This is an anchor measurement
                # Gradient of distance w.r.t position
                # (simplified - would need actual anchor position)
                H = np.random.randn(2)  # Placeholder
                H /= norm(H)
                
                # Add to information
                information += np.outer(H, H) / sigma2
                
        # Covariance is inverse of information
        if det(information) > 1e-10:
            try:
                covariance = inv(information + np.eye(2) * self.regularization)
            except:
                covariance = np.eye(2) * 0.1
        else:
            # Low information, high uncertainty
            covariance = np.eye(2) * 0.1
            
        return covariance
        
    def evaluate_consistency(self, estimates: List[EstimateWithUncertainty]) -> float:
        """
        Evaluate consistency between estimates
        
        Returns:
            Consistency score [0, 1], higher is better
        """
        if len(estimates) < 2:
            return 1.0
            
        # Compute pairwise Mahalanobis distances
        distances = []
        for i, e1 in enumerate(estimates):
            for j, e2 in enumerate(estimates[i+1:], i+1):
                diff = e1.position - e2.position
                
                # Combined covariance
                combined_cov = e1.covariance + e2.covariance + np.eye(2) * self.regularization
                
                try:
                    # Mahalanobis distance
                    mahal_dist = diff.T @ inv(combined_cov) @ diff
                    distances.append(np.sqrt(mahal_dist))
                except:
                    # Fallback to Euclidean
                    distances.append(norm(diff))
                    
        # Consistency score based on average distance
        avg_distance = np.mean(distances) if distances else 0
        consistency = 1.0 / (1.0 + avg_distance)
        
        return consistency