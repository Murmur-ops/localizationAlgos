"""
Integer Ambiguity Resolution for Carrier Phase Measurements
Combines TWTT coarse ranging with carrier phase for unambiguous millimeter accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .phase_measurement import PhaseMeasurement, CarrierPhaseConfig
from .wide_lane import MelbourneWubbenaResolver, DualFrequencyConfig, WideLaneMeasurement

logger = logging.getLogger(__name__)


@dataclass
class AmbiguityResolutionResult:
    """Result of integer ambiguity resolution"""
    integer_cycles: int
    refined_distance_m: float
    confidence: float  # Confidence in resolution [0, 1]
    alternatives: List[Tuple[int, float, float]]  # (cycles, distance, probability)
    method: str  # Resolution method used


class IntegerAmbiguityResolver:
    """
    Resolves integer cycle ambiguity in carrier phase measurements
    using multiple techniques
    """
    
    def __init__(self, config: CarrierPhaseConfig = None):
        """
        Initialize ambiguity resolver
        
        Args:
            config: Carrier phase configuration
        """
        self.config = config or CarrierPhaseConfig()
        self.wavelength = self.config.wavelength
        
        # History for temporal consistency
        self.resolution_history: Dict[Tuple[int, int], List[int]] = {}
        
        # Initialize wide-lane resolver if dual-frequency enabled
        self.wide_lane_resolver = None
        if self.config.use_dual_frequency and self.config.frequency_l2_hz:
            dual_config = DualFrequencyConfig(
                frequency_l1_hz=self.config.frequency_hz,
                frequency_l2_hz=self.config.frequency_l2_hz,
                phase_noise_l1_rad=self.config.phase_noise_rad,
                phase_noise_l2_rad=self.config.phase_noise_rad * 1.5,
                snr_db=self.config.snr_db
            )
            self.wide_lane_resolver = MelbourneWubbenaResolver(dual_config)
            logger.info(f"Dual-frequency mode enabled: WL wavelength = {dual_config.wavelength_wide_lane*100:.1f}cm")
        
    def resolve_single_baseline(self, phase_rad: float,
                               coarse_distance: float,
                               coarse_std: float,
                               phase_l2_rad: Optional[float] = None,
                               code_l2: Optional[float] = None) -> AmbiguityResolutionResult:
        """
        Resolve ambiguity for single baseline using coarse distance
        
        Args:
            phase_rad: Measured carrier phase [0, 2π) for L1
            coarse_distance: Coarse distance from TWTT (meters)
            coarse_std: Standard deviation of coarse measurement
            phase_l2_rad: Optional L2 phase for dual-frequency
            code_l2: Optional L2 code measurement
            
        Returns:
            Ambiguity resolution result
        """
        # Use wide-lane if dual-frequency data available
        if (self.wide_lane_resolver and phase_l2_rad is not None and 
            code_l2 is not None):
            return self._resolve_with_wide_lane(
                phase_rad, phase_l2_rad, coarse_distance, code_l2, coarse_std
            )
        # Fractional wavelength from phase
        fractional = phase_rad / (2 * np.pi)
        fine_distance = fractional * self.wavelength
        
        # Most likely integer cycles
        n_center = np.round(coarse_distance / self.wavelength)
        
        # Consider alternatives within 3σ of coarse measurement
        n_range = int(np.ceil(3 * coarse_std / self.wavelength)) + 1
        alternatives = []
        
        for n in range(int(n_center - n_range), int(n_center + n_range + 1)):
            if n < 0:
                continue
                
            distance = n * self.wavelength + fine_distance
            
            # Calculate probability based on coarse measurement
            error = distance - coarse_distance
            prob = np.exp(-0.5 * (error / coarse_std) ** 2)
            prob /= (coarse_std * np.sqrt(2 * np.pi))  # Normalize
            
            alternatives.append((n, distance, prob))
        
        # Sort by probability
        alternatives.sort(key=lambda x: x[2], reverse=True)
        
        # Best solution
        best_n, best_dist, best_prob = alternatives[0]
        
        # Calculate confidence
        total_prob = sum(p for _, _, p in alternatives)
        confidence = best_prob / total_prob if total_prob > 0 else 0
        
        return AmbiguityResolutionResult(
            integer_cycles=best_n,
            refined_distance_m=best_dist,
            confidence=confidence,
            alternatives=alternatives[:5],  # Keep top 5
            method="single_baseline"
        )
    
    def _resolve_with_wide_lane(self, phase_l1: float, phase_l2: float,
                               code_l1: float, code_l2: float,
                               coarse_std: float) -> AmbiguityResolutionResult:
        """
        Resolve using wide-lane combination for robust resolution
        
        Args:
            phase_l1: L1 phase measurement (radians)
            phase_l2: L2 phase measurement (radians)
            code_l1: L1 code/TWTT measurement (meters)
            code_l2: L2 code measurement (meters)
            coarse_std: Coarse measurement standard deviation
            
        Returns:
            Ambiguity resolution result with high confidence
        """
        # Resolve using Melbourne-Wübbena
        wl_result = self.wide_lane_resolver.resolve_dual_frequency(
            phase_l1, phase_l2, code_l1, code_l2
        )
        
        # Validate solution
        valid = self.wide_lane_resolver.validate_solution(wl_result)
        
        if not valid:
            # Fall back to single frequency
            logger.warning("Wide-lane validation failed, falling back to single frequency")
            return self.resolve_single_baseline(phase_l1, code_l1, coarse_std)
        
        # Build alternatives list for compatibility
        alternatives = [
            (wl_result.l1_ambiguity, wl_result.refined_distance_m, wl_result.confidence)
        ]
        
        # Add nearby alternatives with lower confidence
        for delta in [-1, 1]:
            alt_n = wl_result.l1_ambiguity + delta
            alt_dist = (alt_n + phase_l1/(2*np.pi)) * self.wavelength
            alt_conf = wl_result.confidence * 0.1  # Much lower confidence
            alternatives.append((alt_n, alt_dist, alt_conf))
        
        return AmbiguityResolutionResult(
            integer_cycles=wl_result.l1_ambiguity,
            refined_distance_m=wl_result.refined_distance_m,
            confidence=wl_result.confidence,
            alternatives=alternatives,
            method="wide_lane"
        )
    
    def resolve_with_geometry(self, measurements: List[PhaseMeasurement],
                             positions: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Resolve ambiguities using geometric constraints
        
        Args:
            measurements: List of phase measurements
            positions: Optional known positions for some nodes
            
        Returns:
            Dictionary of resolved measurements by node pair
        """
        results = {}
        
        # Group measurements by node
        node_measurements = {}
        for m in measurements:
            if m.node_i not in node_measurements:
                node_measurements[m.node_i] = []
            if m.node_j not in node_measurements:
                node_measurements[m.node_j] = []
            node_measurements[m.node_i].append(m)
            node_measurements[m.node_j].append(m)
        
        # Use geometric constraints if we have multiple measurements
        for m in measurements:
            pair = (m.node_i, m.node_j)
            
            # First try single baseline resolution
            result = self.resolve_single_baseline(
                m.measured_phase_rad,
                m.coarse_distance_m,
                np.sqrt(m.coarse_variance)
            )
            
            # If confidence is low, try to use geometry
            if result.confidence < 0.9 and len(node_measurements) >= 3:
                result = self._improve_with_geometry(
                    m, node_measurements, result, positions
                )
            
            results[pair] = result
            
            # Update history for temporal tracking
            if pair not in self.resolution_history:
                self.resolution_history[pair] = []
            self.resolution_history[pair].append(result.integer_cycles)
            
        return results
    
    def _improve_with_geometry(self, measurement: PhaseMeasurement,
                               node_measurements: Dict,
                               initial_result: AmbiguityResolutionResult,
                               positions: Optional[Dict] = None) -> AmbiguityResolutionResult:
        """
        Improve ambiguity resolution using geometric constraints
        
        Triangle inequality and other geometric constraints can help
        resolve ambiguities when single baseline confidence is low
        """
        # Find common neighbors
        i_measurements = node_measurements.get(measurement.node_i, [])
        j_measurements = node_measurements.get(measurement.node_j, [])
        
        common_neighbors = set()
        for m in i_measurements:
            other = m.node_j if m.node_i == measurement.node_i else m.node_i
            common_neighbors.add(other)
        for m in j_measurements:
            other = m.node_j if m.node_i == measurement.node_j else m.node_i
            common_neighbors.add(other)
        
        # Remove the nodes themselves
        common_neighbors.discard(measurement.node_i)
        common_neighbors.discard(measurement.node_j)
        
        if not common_neighbors:
            return initial_result
        
        # Check triangle inequality for each alternative
        improved_scores = []
        
        for n_cycles, distance, prob in initial_result.alternatives[:5]:
            geometry_score = 1.0
            
            for k in list(common_neighbors)[:3]:  # Use up to 3 common neighbors
                # Find measurements i-k and j-k
                d_ik = None
                d_jk = None
                
                for m in i_measurements:
                    if k in [m.node_i, m.node_j]:
                        d_ik = m.refined_distance_m or m.coarse_distance_m
                        break
                
                for m in j_measurements:
                    if k in [m.node_i, m.node_j]:
                        d_jk = m.refined_distance_m or m.coarse_distance_m
                        break
                
                if d_ik and d_jk:
                    # Check triangle inequality
                    # |d_ik - d_jk| <= d_ij <= d_ik + d_jk
                    lower = abs(d_ik - d_jk)
                    upper = d_ik + d_jk
                    
                    if lower <= distance <= upper:
                        # Good fit - boost score
                        margin = min(distance - lower, upper - distance)
                        geometry_score *= (1 + margin / distance)
                    else:
                        # Violation - reduce score
                        if distance < lower:
                            violation = lower - distance
                        else:
                            violation = distance - upper
                        geometry_score *= np.exp(-violation / self.wavelength)
            
            improved_scores.append((n_cycles, distance, prob * geometry_score))
        
        # Re-sort by improved scores
        improved_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Update result
        best_n, best_dist, best_score = improved_scores[0]
        total_score = sum(s for _, _, s in improved_scores)
        
        return AmbiguityResolutionResult(
            integer_cycles=best_n,
            refined_distance_m=best_dist,
            confidence=best_score / total_score if total_score > 0 else 0,
            alternatives=[(n, d, s/total_score) for n, d, s in improved_scores],
            method="geometric_constraints"
        )
    
    def resolve_with_kalman(self, measurement: PhaseMeasurement,
                           pair: Tuple[int, int]) -> AmbiguityResolutionResult:
        """
        Use Kalman filter for temporal tracking of integer ambiguities
        
        Args:
            measurement: New phase measurement
            pair: Node pair (i, j)
            
        Returns:
            Resolved ambiguity with temporal consistency
        """
        # Get history for this pair
        history = self.resolution_history.get(pair, [])
        
        # First measurement - use single baseline
        if not history:
            return self.resolve_single_baseline(
                measurement.measured_phase_rad,
                measurement.coarse_distance_m,
                np.sqrt(measurement.coarse_variance)
            )
        
        # Predict based on history
        if len(history) >= 3:
            # Use trend for prediction
            recent = history[-3:]
            if len(set(recent)) == 1:
                # Stable - high confidence in previous value
                predicted_n = recent[-1]
                prediction_confidence = 0.9
            else:
                # Changing - lower confidence
                predicted_n = recent[-1]
                prediction_confidence = 0.5
        else:
            predicted_n = history[-1]
            prediction_confidence = 0.7
        
        # Get single baseline result
        baseline_result = self.resolve_single_baseline(
            measurement.measured_phase_rad,
            measurement.coarse_distance_m,
            np.sqrt(measurement.coarse_variance)
        )
        
        # Combine prediction with measurement
        combined_alternatives = []
        
        for n_cycles, distance, prob in baseline_result.alternatives:
            # Boost probability if matches prediction
            if n_cycles == predicted_n:
                combined_prob = prob * (1 + prediction_confidence)
            else:
                # Reduce if different from prediction
                cycle_diff = abs(n_cycles - predicted_n)
                combined_prob = prob * np.exp(-cycle_diff)
            
            combined_alternatives.append((n_cycles, distance, combined_prob))
        
        # Re-normalize
        combined_alternatives.sort(key=lambda x: x[2], reverse=True)
        total_prob = sum(p for _, _, p in combined_alternatives)
        
        if total_prob > 0:
            combined_alternatives = [(n, d, p/total_prob) 
                                    for n, d, p in combined_alternatives]
        
        best_n, best_dist, best_prob = combined_alternatives[0]
        
        return AmbiguityResolutionResult(
            integer_cycles=best_n,
            refined_distance_m=best_dist,
            confidence=best_prob,
            alternatives=combined_alternatives[:5],
            method="kalman_tracking"
        )
    
    def batch_resolve(self, measurements: List[PhaseMeasurement]) -> Dict:
        """
        Resolve ambiguities for batch of measurements jointly
        
        Uses all available information for optimal resolution
        
        Args:
            measurements: List of phase measurements
            
        Returns:
            Dictionary of resolution results by node pair
        """
        results = {}
        
        # First pass: single baseline resolution
        for m in measurements:
            pair = (m.node_i, m.node_j)
            result = self.resolve_single_baseline(
                m.measured_phase_rad,
                m.coarse_distance_m,
                np.sqrt(m.coarse_variance)
            )
            results[pair] = result
        
        # Second pass: improve with geometry
        improved = self.resolve_with_geometry(measurements)
        results.update(improved)
        
        # Third pass: temporal consistency for pairs with history
        for m in measurements:
            pair = (m.node_i, m.node_j)
            if pair in self.resolution_history and len(self.resolution_history[pair]) >= 2:
                kalman_result = self.resolve_with_kalman(m, pair)
                
                # Use Kalman result if more confident
                if kalman_result.confidence > results[pair].confidence:
                    results[pair] = kalman_result
        
        # Update measurements with resolved values
        for m in measurements:
            pair = (m.node_i, m.node_j)
            if pair in results:
                m.integer_cycles = results[pair].integer_cycles
                m.refined_distance_m = results[pair].refined_distance_m
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get resolution statistics"""
        
        if not self.resolution_history:
            return {"status": "No resolution history"}
        
        stats = {
            "num_pairs": len(self.resolution_history),
            "total_resolutions": sum(len(h) for h in self.resolution_history.values()),
        }
        
        # Check stability of resolutions
        stable_pairs = 0
        changing_pairs = 0
        
        for pair, history in self.resolution_history.items():
            if len(history) >= 3:
                recent = history[-5:]
                if len(set(recent)) == 1:
                    stable_pairs += 1
                else:
                    changing_pairs += 1
        
        stats["stable_pairs"] = stable_pairs
        stats["changing_pairs"] = changing_pairs
        
        return stats


def test_ambiguity_resolver():
    """Test integer ambiguity resolver"""
    
    print("="*60)
    print("INTEGER AMBIGUITY RESOLVER TEST")
    print("="*60)
    
    config = CarrierPhaseConfig(frequency_hz=2.4e9)
    resolver = IntegerAmbiguityResolver(config)
    
    print(f"\nWavelength: {config.wavelength*100:.1f} cm")
    print("-"*40)
    
    # Test cases with different scenarios
    test_cases = [
        # (true_distance, coarse_distance, coarse_std)
        (1.0, 1.05, 0.3),  # Easy case
        (2.5, 2.3, 0.5),   # Moderate uncertainty
        (5.0, 5.5, 1.0),   # High uncertainty
        (0.3, 0.25, 0.2),  # Short distance
    ]
    
    for true_dist, coarse_dist, coarse_std in test_cases:
        # Simulate phase measurement
        true_phase = (true_dist / config.wavelength) * 2 * np.pi
        measured_phase = true_phase % (2 * np.pi)
        
        # Add small phase noise
        measured_phase += np.random.normal(0, 0.001)
        measured_phase = measured_phase % (2 * np.pi)
        
        # Resolve ambiguity
        result = resolver.resolve_single_baseline(
            measured_phase, coarse_dist, coarse_std
        )
        
        error_mm = abs(result.refined_distance_m - true_dist) * 1000
        
        print(f"\nTrue distance: {true_dist:.3f} m")
        print(f"Coarse (TWTT): {coarse_dist:.3f} ± {coarse_std:.3f} m")
        print(f"Phase: {measured_phase:.3f} rad")
        print(f"Resolved cycles: {result.integer_cycles}")
        print(f"Refined distance: {result.refined_distance_m:.6f} m")
        print(f"Error: {error_mm:.2f} mm")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Method: {result.method}")
        
        if result.alternatives:
            print("Top alternatives:")
            for n, d, p in result.alternatives[:3]:
                print(f"  n={n}: {d:.3f}m (p={p:.3f})")
    
    print("\n" + "="*60)
    
    return resolver


if __name__ == "__main__":
    test_ambiguity_resolver()