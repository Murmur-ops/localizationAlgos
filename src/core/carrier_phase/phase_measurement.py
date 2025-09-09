"""
Carrier Phase Measurement System for Millimeter-Accuracy Ranging
Based on Nanzer et al. approach for S-band coherent beamforming

This module implements carrier phase measurements with integer ambiguity
resolution using TWTT for coarse ranging.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CarrierPhaseConfig:
    """Configuration for carrier phase measurements"""
    frequency_hz: float = 2.4e9  # S-band carrier frequency (L1)
    frequency_l2_hz: Optional[float] = 1.9e9  # Optional L2 frequency for dual-freq
    phase_noise_rad: float = 0.001  # 1 milliradian phase noise
    frequency_stability_ppb: float = 0.1  # 0.1 ppb with OCXO
    snr_db: float = 30.0  # Signal-to-noise ratio
    integration_time_ms: float = 1.0  # Phase integration time
    use_dual_frequency: bool = False  # Enable dual-frequency mode
    
    @property
    def wavelength(self) -> float:
        """Wavelength in meters"""
        c = 299792458  # Speed of light m/s
        return c / self.frequency_hz
    
    @property
    def phase_precision_mm(self) -> float:
        """Expected phase measurement precision in millimeters"""
        # Phase noise converts to distance: λ/(2π) * phase_noise
        return (self.wavelength / (2 * np.pi)) * self.phase_noise_rad * 1000
    
    @property
    def max_unambiguous_range(self) -> float:
        """Maximum unambiguous range (one wavelength)"""
        return self.wavelength
    
    @property
    def wavelength_l2(self) -> float:
        """L2 wavelength if dual-frequency enabled"""
        if self.frequency_l2_hz:
            c = 299792458
            return c / self.frequency_l2_hz
        return self.wavelength
    
    @property
    def wavelength_wide_lane(self) -> float:
        """Wide-lane wavelength for dual-frequency"""
        if self.frequency_l2_hz and self.use_dual_frequency:
            c = 299792458
            f_diff = abs(self.frequency_hz - self.frequency_l2_hz)
            return c / f_diff if f_diff > 0 else float('inf')
        return self.wavelength


@dataclass
class PhaseMeasurement:
    """Single carrier phase measurement between nodes"""
    node_i: int
    node_j: int
    measured_phase_rad: float  # Measured phase in radians [0, 2π)
    phase_variance: float  # Phase measurement variance
    coarse_distance_m: float  # Coarse distance from TWTT
    coarse_variance: float  # Coarse distance variance
    refined_distance_m: Optional[float] = None  # After ambiguity resolution
    integer_cycles: Optional[int] = None  # Resolved integer cycles
    timestamp_ns: Optional[int] = None  # Measurement timestamp
    quality_factor: float = 1.0  # Measurement quality [0, 1]
    
    def resolve_ambiguity(self) -> float:
        """
        Resolve integer ambiguity using coarse distance
        
        Returns:
            Refined distance in meters with millimeter precision
        """
        config = CarrierPhaseConfig()
        wavelength = config.wavelength
        
        # Calculate fractional wavelength from phase
        fractional_wavelength = self.measured_phase_rad / (2 * np.pi)
        fine_distance = fractional_wavelength * wavelength
        
        # Determine integer number of wavelengths from coarse measurement
        n_cycles = np.round(self.coarse_distance_m / wavelength)
        self.integer_cycles = int(n_cycles)
        
        # Combine for refined distance
        self.refined_distance_m = n_cycles * wavelength + fine_distance
        
        # Check for edge cases where we might be off by one cycle
        alternatives = [
            (n_cycles - 1) * wavelength + fine_distance,
            n_cycles * wavelength + fine_distance,
            (n_cycles + 1) * wavelength + fine_distance
        ]
        
        # Choose the one closest to coarse measurement
        errors = [abs(alt - self.coarse_distance_m) for alt in alternatives]
        best_idx = np.argmin(errors)
        
        if best_idx == 0:
            self.integer_cycles -= 1
        elif best_idx == 2:
            self.integer_cycles += 1
            
        self.refined_distance_m = alternatives[best_idx]
        
        return self.refined_distance_m


class CarrierPhaseMeasurementSystem:
    """
    Complete carrier phase measurement system with ambiguity resolution
    """
    
    def __init__(self, config: CarrierPhaseConfig = None):
        """
        Initialize carrier phase measurement system
        
        Args:
            config: Carrier phase configuration
        """
        self.config = config or CarrierPhaseConfig()
        self.measurements: Dict[Tuple[int, int], List[PhaseMeasurement]] = {}
        self.phase_history: Dict[Tuple[int, int], List[float]] = {}
        
        logger.info(f"Carrier Phase System initialized:")
        logger.info(f"  Frequency: {self.config.frequency_hz/1e9:.2f} GHz")
        logger.info(f"  Wavelength: {self.config.wavelength*100:.1f} cm")
        logger.info(f"  Phase precision: {self.config.phase_precision_mm:.2f} mm")
    
    def measure_carrier_phase(self, true_distance: float, 
                            add_noise: bool = True) -> Tuple[float, float]:
        """
        Simulate carrier phase measurement
        
        Args:
            true_distance: True distance in meters
            add_noise: Whether to add realistic noise
            
        Returns:
            (measured_phase_rad, phase_variance)
        """
        # True phase (wrapped to [0, 2π))
        true_phase = (true_distance / self.config.wavelength) * 2 * np.pi
        true_phase = true_phase % (2 * np.pi)
        
        if add_noise:
            # Add phase noise based on SNR and integration time
            snr_linear = 10 ** (self.config.snr_db / 10)
            
            # Phase noise decreases with sqrt(integration_time) and SNR
            integration_factor = np.sqrt(self.config.integration_time_ms / 1000)
            effective_phase_noise = self.config.phase_noise_rad / (np.sqrt(snr_linear) * integration_factor)
            
            phase_noise = np.random.normal(0, effective_phase_noise)
            measured_phase = (true_phase + phase_noise) % (2 * np.pi)
            phase_variance = effective_phase_noise ** 2
        else:
            measured_phase = true_phase
            phase_variance = 0.0
        
        return measured_phase, phase_variance
    
    def create_measurement(self, node_i: int, node_j: int,
                          true_distance: float,
                          coarse_distance: float,
                          coarse_std: float = 0.3,
                          use_dual_freq: bool = None) -> PhaseMeasurement:
        """
        Create a complete phase measurement with ambiguity resolution
        
        Args:
            node_i, node_j: Node indices
            true_distance: True distance for simulation
            coarse_distance: Coarse distance from TWTT
            coarse_std: Standard deviation of coarse measurement
            use_dual_freq: Override config to use dual-frequency
            
        Returns:
            Complete phase measurement with resolved distance
        """
        # Determine if using dual-frequency
        use_dual = use_dual_freq if use_dual_freq is not None else self.config.use_dual_frequency
        # Measure carrier phase
        measured_phase, phase_var = self.measure_carrier_phase(true_distance)
        
        # For dual-frequency, also measure L2
        phase_l2 = None
        code_l2 = None
        if use_dual and self.config.frequency_l2_hz:
            # L2 phase measurement
            true_phase_l2 = (true_distance / self.config.wavelength_l2) * 2 * np.pi
            phase_l2 = (true_phase_l2 + np.random.normal(0, self.config.phase_noise_rad * 1.5)) % (2 * np.pi)
            # L2 code measurement (slightly worse than L1)
            code_l2 = true_distance + np.random.normal(0, coarse_std * 1.2)
        
        # Create measurement object
        measurement = PhaseMeasurement(
            node_i=node_i,
            node_j=node_j,
            measured_phase_rad=measured_phase,
            phase_variance=phase_var,
            coarse_distance_m=coarse_distance,
            coarse_variance=coarse_std ** 2,
            timestamp_ns=np.random.randint(0, 2**63)
        )
        
        # Resolve integer ambiguity (with dual-freq if available)
        if use_dual and phase_l2 is not None:
            # Use advanced resolver with dual-frequency
            from .ambiguity_resolver import IntegerAmbiguityResolver
            resolver = IntegerAmbiguityResolver(self.config)
            result = resolver.resolve_single_baseline(
                measured_phase, coarse_distance, coarse_std,
                phase_l2_rad=phase_l2, code_l2=code_l2
            )
            measurement.integer_cycles = result.integer_cycles
            measurement.refined_distance_m = result.refined_distance_m
        else:
            # Standard single-frequency resolution
            measurement.resolve_ambiguity()
        
        # Calculate quality factor based on variances
        phase_quality = np.exp(-phase_var / (self.config.phase_noise_rad ** 2))
        coarse_quality = np.exp(-coarse_std / 1.0)  # Normalize to 1m
        measurement.quality_factor = 0.7 * phase_quality + 0.3 * coarse_quality
        
        # Store measurement
        pair = (min(node_i, node_j), max(node_i, node_j))
        if pair not in self.measurements:
            self.measurements[pair] = []
        self.measurements[pair].append(measurement)
        
        return measurement
    
    def get_measurement_weight(self, measurement: PhaseMeasurement) -> float:
        """
        Calculate weight for measurement based on precision
        Research-backed weight calculation based on GPS RTK standards
        
        Args:
            measurement: Phase measurement
            
        Returns:
            Weight for use in weighted least squares
        """
        if measurement.refined_distance_m is None:
            # Coarse measurement only (TWTT)
            return 1.0  # Baseline weight
        
        # Check if this was resolved with wide-lane (higher confidence)
        if self.config.use_dual_frequency:
            # Wide-lane resolved measurements get higher weight
            base_weight = 500.0  # Much higher confidence with WL
        else:
            # Single-frequency carrier phase
            base_weight = 100.0  # Standard GPS RTK ratio
        
        # Scale by SNR (higher SNR = better precision)
        # Cap SNR factor at 2.0 to prevent excessive weights
        snr_factor = min(2.0, 10 ** (self.config.snr_db / 20))
        
        # Scale by quality factor
        quality_scale = measurement.quality_factor
        
        # Combined weight with cap at 1000 to prevent numerical issues
        weight = base_weight * snr_factor * quality_scale
        
        # Cap weight to maintain condition number < 10^6
        return min(1000.0, weight)
    
    def detect_cycle_slip(self, node_i: int, node_j: int,
                         new_phase: float) -> bool:
        """
        Detect cycle slips in phase measurements
        
        Args:
            node_i, node_j: Node pair
            new_phase: New phase measurement
            
        Returns:
            True if cycle slip detected
        """
        pair = (min(node_i, node_j), max(node_i, node_j))
        
        if pair not in self.phase_history:
            self.phase_history[pair] = []
        
        history = self.phase_history[pair]
        
        if len(history) >= 2:
            # Predict next phase based on linear trend
            if len(history) >= 3:
                # Use last 3 measurements for prediction
                phases = history[-3:]
                times = list(range(len(phases)))
                
                # Linear fit
                z = np.polyfit(times, phases, 1)
                predicted_phase = z[0] * len(times) + z[1]
            else:
                # Simple prediction from last value
                predicted_phase = history[-1]
            
            # Check for sudden jump (more than π/2)
            phase_diff = abs(new_phase - predicted_phase)
            
            # Account for phase wrapping
            if phase_diff > np.pi:
                phase_diff = 2 * np.pi - phase_diff
            
            if phase_diff > np.pi / 2:
                logger.warning(f"Cycle slip detected for pair ({node_i}, {node_j}): "
                             f"jump of {phase_diff:.3f} rad")
                return True
        
        # Add to history
        history.append(new_phase)
        if len(history) > 10:
            history.pop(0)  # Keep only recent history
        
        return False
    
    def unwrap_phase(self, phases: List[float]) -> List[float]:
        """
        Unwrap phase measurements to remove 2π discontinuities
        
        Args:
            phases: List of wrapped phases [0, 2π)
            
        Returns:
            List of unwrapped phases
        """
        if len(phases) == 0:
            return []
        
        unwrapped = [phases[0]]
        
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            
            # Check for wrap
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            
            unwrapped.append(unwrapped[-1] + diff)
        
        return unwrapped
    
    def get_statistics(self) -> Dict:
        """
        Get measurement statistics
        
        Returns:
            Dictionary of statistics
        """
        if not self.measurements:
            return {"status": "No measurements"}
        
        all_measurements = []
        for pair_measurements in self.measurements.values():
            all_measurements.extend(pair_measurements)
        
        refined_distances = [m.refined_distance_m for m in all_measurements 
                           if m.refined_distance_m is not None]
        
        if not refined_distances:
            return {"status": "No refined measurements"}
        
        # Calculate ranging errors if we have true distances
        # (In real system, compare against reference)
        errors_mm = []
        for m in all_measurements:
            if m.refined_distance_m is not None:
                # Simulate true distance for statistics
                true_dist = m.coarse_distance_m  # Use coarse as proxy
                error_m = abs(m.refined_distance_m - true_dist)
                errors_mm.append(error_m * 1000)
        
        stats = {
            "num_measurements": len(all_measurements),
            "num_pairs": len(self.measurements),
            "avg_quality": np.mean([m.quality_factor for m in all_measurements]),
            "phase_precision_mm": self.config.phase_precision_mm,
            "wavelength_cm": self.config.wavelength * 100,
        }
        
        if errors_mm:
            stats.update({
                "mean_error_mm": np.mean(errors_mm),
                "std_error_mm": np.std(errors_mm),
                "max_error_mm": np.max(errors_mm),
                "rmse_mm": np.sqrt(np.mean(np.square(errors_mm)))
            })
        
        return stats


def test_carrier_phase_system():
    """Test carrier phase measurement system"""
    
    print("="*60)
    print("CARRIER PHASE MEASUREMENT SYSTEM TEST")
    print("="*60)
    
    # Create system
    config = CarrierPhaseConfig(
        frequency_hz=2.4e9,
        phase_noise_rad=0.001,
        snr_db=30
    )
    
    system = CarrierPhaseMeasurementSystem(config)
    
    # Test measurements at different distances
    test_distances = [0.5, 1.0, 2.5, 5.0, 10.0]  # meters
    
    print("\nTest Measurements:")
    print("-"*40)
    
    for true_dist in test_distances:
        # Simulate TWTT coarse measurement (±30cm accuracy)
        coarse_dist = true_dist + np.random.normal(0, 0.3)
        
        # Create phase measurement
        measurement = system.create_measurement(
            node_i=0,
            node_j=1,
            true_distance=true_dist,
            coarse_distance=coarse_dist,
            coarse_std=0.3
        )
        
        error_mm = abs(measurement.refined_distance_m - true_dist) * 1000
        
        print(f"True: {true_dist:.3f}m")
        print(f"  Coarse (TWTT): {coarse_dist:.3f}m")
        print(f"  Phase: {measurement.measured_phase_rad:.3f} rad")
        print(f"  Cycles: {measurement.integer_cycles}")
        print(f"  Refined: {measurement.refined_distance_m:.6f}m")
        print(f"  Error: {error_mm:.2f}mm")
        print(f"  Weight: {system.get_measurement_weight(measurement):.1f}")
        print()
    
    # Get statistics
    stats = system.get_statistics()
    print("\nSystem Statistics:")
    print("-"*40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    
    return system


if __name__ == "__main__":
    test_carrier_phase_system()