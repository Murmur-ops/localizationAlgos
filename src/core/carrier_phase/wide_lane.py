"""
Wide-Lane Carrier Phase Combinations for Robust Ambiguity Resolution
Based on Melbourne-Wübbena combination and GPS RTK techniques
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DualFrequencyConfig:
    """Configuration for dual-frequency carrier phase measurements"""
    # L1 frequency (S-band)
    frequency_l1_hz: float = 2.4e9  # 2.4 GHz
    # L2 frequency (lower S-band)  
    frequency_l2_hz: float = 1.9e9  # 1.9 GHz
    
    # Phase noise for each frequency
    phase_noise_l1_rad: float = 0.001  # 1 mrad
    phase_noise_l2_rad: float = 0.0015  # Slightly worse at lower frequency
    
    # Code/pseudorange noise
    code_noise_l1_m: float = 0.3  # 30cm for TWTT
    code_noise_l2_m: float = 0.4  # Slightly worse
    
    snr_db: float = 40.0
    
    @property
    def wavelength_l1(self) -> float:
        """L1 wavelength in meters"""
        c = 299792458
        return c / self.frequency_l1_hz
    
    @property
    def wavelength_l2(self) -> float:
        """L2 wavelength in meters"""
        c = 299792458
        return c / self.frequency_l2_hz
    
    @property
    def wavelength_wide_lane(self) -> float:
        """Wide-lane wavelength in meters"""
        c = 299792458
        f_wl = self.frequency_l1_hz - self.frequency_l2_hz
        return c / f_wl if f_wl > 0 else float('inf')
    
    @property
    def wavelength_narrow_lane(self) -> float:
        """Narrow-lane wavelength in meters"""
        c = 299792458
        f_nl = self.frequency_l1_hz + self.frequency_l2_hz
        return c / f_nl
    
    def get_lane_info(self) -> Dict:
        """Get information about all lane combinations"""
        return {
            'L1': {
                'frequency_hz': self.frequency_l1_hz,
                'wavelength_m': self.wavelength_l1,
                'wavelength_cm': self.wavelength_l1 * 100,
                'max_twtt_error_m': self.wavelength_l1 / 2
            },
            'L2': {
                'frequency_hz': self.frequency_l2_hz,
                'wavelength_m': self.wavelength_l2,
                'wavelength_cm': self.wavelength_l2 * 100,
                'max_twtt_error_m': self.wavelength_l2 / 2
            },
            'wide_lane': {
                'frequency_hz': self.frequency_l1_hz - self.frequency_l2_hz,
                'wavelength_m': self.wavelength_wide_lane,
                'wavelength_cm': self.wavelength_wide_lane * 100,
                'max_twtt_error_m': self.wavelength_wide_lane / 2
            },
            'narrow_lane': {
                'frequency_hz': self.frequency_l1_hz + self.frequency_l2_hz,
                'wavelength_m': self.wavelength_narrow_lane,
                'wavelength_cm': self.wavelength_narrow_lane * 100,
                'max_twtt_error_m': self.wavelength_narrow_lane / 2
            }
        }


@dataclass
class WideLaneMeasurement:
    """Wide-lane carrier phase measurement result"""
    node_i: int
    node_j: int
    
    # Phase measurements (radians)
    phase_l1_rad: float
    phase_l2_rad: float
    phase_wide_lane_rad: float
    
    # Code measurements (meters)
    code_l1_m: float
    code_l2_m: float
    
    # Melbourne-Wübbena observable
    mw_observable: float
    mw_cycles: Optional[int] = None
    
    # Resolved ambiguities
    wide_lane_ambiguity: Optional[int] = None
    l1_ambiguity: Optional[int] = None
    l2_ambiguity: Optional[int] = None
    
    # Final distance
    refined_distance_m: Optional[float] = None
    confidence: float = 0.0


class MelbourneWubbenaResolver:
    """
    Melbourne-Wübbena wide-lane ambiguity resolver
    Implements the proven GPS RTK technique for robust ambiguity resolution
    """
    
    def __init__(self, config: Optional[DualFrequencyConfig] = None):
        """Initialize resolver with dual-frequency configuration"""
        self.config = config or DualFrequencyConfig()
        
        # Cache wavelengths
        self.lambda_1 = self.config.wavelength_l1
        self.lambda_2 = self.config.wavelength_l2
        self.lambda_wl = self.config.wavelength_wide_lane
        self.lambda_nl = self.config.wavelength_narrow_lane
        
        # Frequencies
        self.f1 = self.config.frequency_l1_hz
        self.f2 = self.config.frequency_l2_hz
        
        # MW averaging window
        self.mw_history: Dict[Tuple[int, int], List[float]] = {}
        self.mw_window_size = 10  # Average over 10 epochs
        
        logger.info(f"Melbourne-Wübbena Resolver initialized:")
        logger.info(f"  L1: {self.lambda_1*100:.1f}cm @ {self.f1/1e9:.2f}GHz")
        logger.info(f"  L2: {self.lambda_2*100:.1f}cm @ {self.f2/1e9:.2f}GHz")
        logger.info(f"  Wide-lane: {self.lambda_wl*100:.1f}cm")
        logger.info(f"  Max TWTT error tolerance: {self.lambda_wl/2*100:.1f}cm")
    
    def compute_mw_observable(self, phase_l1: float, phase_l2: float,
                             code_l1: float, code_l2: float) -> float:
        """
        Compute Melbourne-Wübbena linear combination
        
        This observable is:
        - Geometry-free (cancels geometric range)
        - Ionosphere-free (cancels ionospheric delay)
        - Ideal for wide-lane ambiguity resolution
        
        Args:
            phase_l1: L1 phase measurement (radians)
            phase_l2: L2 phase measurement (radians)
            code_l1: L1 code/TWTT measurement (meters)
            code_l2: L2 code/TWTT measurement (meters)
            
        Returns:
            MW observable in wide-lane cycles
        """
        # Convert phases to cycles
        phi_1_cycles = phase_l1 / (2 * np.pi)
        phi_2_cycles = phase_l2 / (2 * np.pi)
        
        # Wide-lane phase combination (cycles)
        phi_wl = (self.f1 * phi_1_cycles - self.f2 * phi_2_cycles) / (self.f1 - self.f2)
        
        # Narrow-lane code combination (meters -> cycles)
        p_nl = (self.f1 * code_l1 + self.f2 * code_l2) / (self.f1 + self.f2)
        p_nl_cycles = p_nl / self.lambda_wl
        
        # Melbourne-Wübbena combination
        mw = phi_wl - p_nl_cycles
        
        return mw
    
    def resolve_wide_lane_ambiguity(self, mw_observable: float,
                                   pair: Optional[Tuple[int, int]] = None) -> Tuple[int, float]:
        """
        Resolve wide-lane integer ambiguity
        
        Args:
            mw_observable: Melbourne-Wübbena observable
            pair: Optional node pair for averaging
            
        Returns:
            (wide_lane_ambiguity, confidence)
        """
        if pair and pair in self.mw_history:
            # Add to history
            history = self.mw_history[pair]
            history.append(mw_observable)
            
            # Keep window size
            if len(history) > self.mw_window_size:
                history.pop(0)
            
            # Average for robust estimation
            mw_avg = np.mean(history)
            mw_std = np.std(history) if len(history) > 1 else 0.5
        else:
            # Single epoch resolution
            mw_avg = mw_observable
            mw_std = 0.5
            
            if pair:
                self.mw_history[pair] = [mw_observable]
        
        # Round to nearest integer
        n_wl = np.round(mw_avg)
        
        # Calculate confidence based on distance to integer
        residual = abs(mw_avg - n_wl)
        confidence = np.exp(-residual / 0.1)  # High confidence if close to integer
        
        # Reduce confidence if high variance
        if mw_std > 0.2:
            confidence *= np.exp(-mw_std)
        
        return int(n_wl), confidence
    
    def resolve_l1_ambiguity(self, phase_l1: float, phase_l2: float,
                            n_wl: int, coarse_distance: float) -> Tuple[int, float]:
        """
        Resolve L1 ambiguity using resolved wide-lane
        
        Args:
            phase_l1: L1 phase (radians)
            phase_l2: L2 phase (radians)
            n_wl: Resolved wide-lane ambiguity
            coarse_distance: Coarse distance estimate (meters)
            
        Returns:
            (l1_ambiguity, refined_distance)
        """
        # Convert phases to cycles
        phi_1 = phase_l1 / (2 * np.pi)
        phi_2 = phase_l2 / (2 * np.pi)
        
        # Use wide-lane constraint: N1 - N2 = N_wl
        # The key insight: once we know N_wl, we can determine N1 more accurately
        
        # Narrow-lane combination for better precision
        # φ_NL = (f1*φ1 + f2*φ2)/(f1+f2)
        phi_nl = (self.f1 * phi_1 + self.f2 * phi_2) / (self.f1 + self.f2)
        
        # Initial estimate from narrow-lane
        n_nl_float = coarse_distance / self.lambda_nl - phi_nl
        
        # Now use the fact that:
        # N1 = (f1+f2)/(2*f1) * N_NL + (f1-f2)/(2*f1) * N_WL
        # N2 = (f1+f2)/(2*f2) * N_NL - (f1-f2)/(2*f2) * N_WL
        
        # Since we know N_WL, we can find N_NL and then N1, N2
        # For simplicity, search around the expected value
        
        best_n1 = None
        best_error = float('inf')
        
        # Search for N1 that satisfies both WL constraint and minimizes residual
        n1_estimate = coarse_distance / self.lambda_1
        search_range = max(3, int(0.5 / self.lambda_1))  # At least ±50cm search
        
        for n1 in range(int(n1_estimate - search_range), int(n1_estimate + search_range + 1)):
            # Apply wide-lane constraint
            n2 = n1 - n_wl
            
            # Calculate distances using both frequencies
            dist_1 = (n1 + phi_1) * self.lambda_1
            dist_2 = (n2 + phi_2) * self.lambda_2
            
            # Both should give same distance (within noise)
            dist_avg = (dist_1 + dist_2) / 2
            
            # Error from coarse measurement
            coarse_error = abs(dist_avg - coarse_distance)
            
            # Consistency error between L1 and L2
            consistency_error = abs(dist_1 - dist_2)
            
            # Combined error metric
            total_error = coarse_error + 10 * consistency_error  # Weight consistency heavily
            
            if total_error < best_error:
                best_error = total_error
                best_n1 = n1
        
        # Calculate final distance using both frequencies
        n2_final = best_n1 - n_wl
        dist_1_final = (best_n1 + phi_1) * self.lambda_1
        dist_2_final = (n2_final + phi_2) * self.lambda_2
        
        # Use weighted average based on wavelengths (shorter wavelength = higher precision)
        w1 = 1.0 / self.lambda_1
        w2 = 1.0 / self.lambda_2
        refined_distance = (w1 * dist_1_final + w2 * dist_2_final) / (w1 + w2)
        
        return best_n1, refined_distance
    
    def resolve_dual_frequency(self, phase_l1: float, phase_l2: float,
                              code_l1: float, code_l2: float,
                              pair: Optional[Tuple[int, int]] = None) -> WideLaneMeasurement:
        """
        Complete dual-frequency ambiguity resolution
        
        Args:
            phase_l1, phase_l2: Phase measurements (radians)
            code_l1, code_l2: Code measurements (meters)
            pair: Node pair for history tracking
            
        Returns:
            Complete wide-lane measurement with resolved ambiguities
        """
        # Step 1: Compute Melbourne-Wübbena observable
        mw_obs = self.compute_mw_observable(phase_l1, phase_l2, code_l1, code_l2)
        
        # Step 2: Resolve wide-lane ambiguity
        n_wl, wl_confidence = self.resolve_wide_lane_ambiguity(mw_obs, pair)
        
        # Step 3: Resolve L1 ambiguity using wide-lane constraint
        coarse_dist = (code_l1 + code_l2) / 2  # Average code measurement
        n1, refined_dist = self.resolve_l1_ambiguity(phase_l1, phase_l2, n_wl, coarse_dist)
        
        # Step 4: Calculate L2 ambiguity
        n2 = n1 - n_wl
        
        # Create measurement result
        result = WideLaneMeasurement(
            node_i=pair[0] if pair else 0,
            node_j=pair[1] if pair else 1,
            phase_l1_rad=phase_l1,
            phase_l2_rad=phase_l2,
            phase_wide_lane_rad=(phase_l1 - phase_l2),
            code_l1_m=code_l1,
            code_l2_m=code_l2,
            mw_observable=mw_obs,
            mw_cycles=n_wl,
            wide_lane_ambiguity=n_wl,
            l1_ambiguity=n1,
            l2_ambiguity=n2,
            refined_distance_m=refined_dist,
            confidence=wl_confidence
        )
        
        return result
    
    def validate_solution(self, measurement: WideLaneMeasurement) -> bool:
        """
        Validate resolved ambiguities using multiple checks
        
        Args:
            measurement: Wide-lane measurement with resolved ambiguities
            
        Returns:
            True if solution passes all validation checks
        """
        if measurement.l1_ambiguity is None or measurement.l2_ambiguity is None:
            return False
        
        # Check 1: Wide-lane constraint
        n_wl_check = measurement.l1_ambiguity - measurement.l2_ambiguity
        if n_wl_check != measurement.wide_lane_ambiguity:
            logger.debug(f"Wide-lane constraint violated: {n_wl_check} != {measurement.wide_lane_ambiguity}")
            return False
        
        # Check 2: Distance consistency (relaxed for noisy measurements)
        phi_1 = measurement.phase_l1_rad / (2 * np.pi)
        phi_2 = measurement.phase_l2_rad / (2 * np.pi)
        
        dist_1 = (measurement.l1_ambiguity + phi_1) * self.lambda_1
        dist_2 = (measurement.l2_ambiguity + phi_2) * self.lambda_2
        
        # Allow larger tolerance due to phase noise
        # Typical phase noise of 1mrad = ~0.2mm at L1, ~0.25mm at L2
        # But integer errors can be large, so check relative to wavelength
        consistency_tolerance = 0.05 * max(self.lambda_1, self.lambda_2)  # 5% of wavelength
        
        if abs(dist_1 - dist_2) > consistency_tolerance:
            logger.debug(f"Distance inconsistency: L1={dist_1:.3f}m, L2={dist_2:.3f}m, diff={abs(dist_1-dist_2)*1000:.1f}mm")
            return False
        
        # Check 3: Reasonable distance
        if measurement.refined_distance_m < 0 or measurement.refined_distance_m > 1000:
            logger.debug(f"Unreasonable distance: {measurement.refined_distance_m:.3f}m")
            return False
        
        return True


def test_wide_lane_resolution():
    """Test wide-lane ambiguity resolution with various error levels"""
    
    print("="*60)
    print("WIDE-LANE AMBIGUITY RESOLUTION TEST")
    print("="*60)
    
    config = DualFrequencyConfig()
    resolver = MelbourneWubbenaResolver(config)
    
    # Print configuration
    print("\nConfiguration:")
    lane_info = config.get_lane_info()
    for lane_name, info in lane_info.items():
        print(f"  {lane_name}:")
        print(f"    Wavelength: {info['wavelength_cm']:.1f}cm")
        print(f"    Max TWTT error: {info['max_twtt_error_m']*100:.1f}cm")
    
    print("\n" + "-"*40)
    print("Testing with various TWTT errors:")
    print("-"*40)
    
    # Test scenarios
    test_cases = [
        (1.5, 0.05),   # 5cm error - should work with L1
        (2.0, 0.10),   # 10cm error - marginal for L1
        (3.0, 0.20),   # 20cm error - needs wide-lane
        (5.0, 0.30),   # 30cm error - still OK with wide-lane
        (10.0, 0.40),  # 40cm error - approaching WL limit
    ]
    
    results = []
    
    for true_dist, twtt_error in test_cases:
        # Simulate measurements
        # True phases
        true_phase_l1 = (true_dist / config.wavelength_l1) * 2 * np.pi
        true_phase_l2 = (true_dist / config.wavelength_l2) * 2 * np.pi
        
        # Add phase noise
        phase_l1 = (true_phase_l1 + np.random.normal(0, 0.001)) % (2 * np.pi)
        phase_l2 = (true_phase_l2 + np.random.normal(0, 0.0015)) % (2 * np.pi)
        
        # Code measurements with error
        code_l1 = true_dist + np.random.normal(0, twtt_error)
        code_l2 = true_dist + np.random.normal(0, twtt_error * 1.2)
        
        # Resolve
        measurement = resolver.resolve_dual_frequency(
            phase_l1, phase_l2, code_l1, code_l2, pair=(0, 1)
        )
        
        # Validate
        valid = resolver.validate_solution(measurement)
        
        # Calculate error
        error_mm = abs(measurement.refined_distance_m - true_dist) * 1000
        
        print(f"\nTrue distance: {true_dist:.3f}m")
        print(f"  TWTT error: ±{twtt_error*100:.0f}cm")
        print(f"  Code measurements: L1={code_l1:.3f}m, L2={code_l2:.3f}m")
        print(f"  MW observable: {measurement.mw_observable:.3f} cycles")
        print(f"  Wide-lane ambiguity: {measurement.wide_lane_ambiguity}")
        print(f"  L1 ambiguity: {measurement.l1_ambiguity}")
        print(f"  L2 ambiguity: {measurement.l2_ambiguity}")
        print(f"  Refined distance: {measurement.refined_distance_m:.6f}m")
        print(f"  Error: {error_mm:.2f}mm")
        print(f"  Confidence: {measurement.confidence:.3f}")
        print(f"  Valid: {'✓' if valid else '✗'}")
        
        results.append({
            'true_dist': true_dist,
            'twtt_error': twtt_error,
            'error_mm': error_mm,
            'valid': valid,
            'confidence': measurement.confidence
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        errors = [r['error_mm'] for r in valid_results]
        print(f"Valid resolutions: {len(valid_results)}/{len(results)}")
        print(f"Mean error: {np.mean(errors):.2f}mm")
        print(f"Max error: {np.max(errors):.2f}mm")
        print(f"RMSE: {np.sqrt(np.mean(np.square(errors))):.2f}mm")
        
        if np.sqrt(np.mean(np.square(errors))) < 15:
            print("\n✓ TARGET ACHIEVED: <15mm RMSE")
        else:
            print("\n⚠ Further tuning needed")
    else:
        print("✗ No valid resolutions")
    
    print("="*60)


if __name__ == "__main__":
    test_wide_lane_resolution()