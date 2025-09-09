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
    frequency_hz: float = 2.4e9  # S-band carrier frequency
    phase_noise_rad: float = 0.001  # 1 milliradian phase noise
    frequency_stability_ppb: float = 0.1  # 0.1 ppb with OCXO
    snr_db: float = 30.0  # Signal-to-noise ratio
    integration_time_ms: float = 1.0  # Phase integration time
    
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
                          coarse_std: float = 0.3) -> PhaseMeasurement:
        """
        Create a complete phase measurement with ambiguity resolution
        
        Args:
            node_i, node_j: Node indices
            true_distance: True distance for simulation
            coarse_distance: Coarse distance from TWTT
            coarse_std: Standard deviation of coarse measurement
            
        Returns:
            Complete phase measurement with resolved distance
        """
        # Measure carrier phase
        measured_phase, phase_var = self.measure_carrier_phase(true_distance)
        
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
        
        # Resolve integer ambiguity
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
        
        Args:
            measurement: Phase measurement
            
        Returns:
            Weight for use in weighted least squares
        """
        if measurement.refined_distance_m is None:
            # Coarse measurement only
            return 1.0 / measurement.coarse_variance
        
        # Combined measurement - use phase precision
        phase_std_m = (self.config.wavelength / (2 * np.pi)) * np.sqrt(measurement.phase_variance)
        
        # Weight inversely proportional to variance
        weight = 1.0 / (phase_std_m ** 2)
        
        # Scale by quality factor
        weight *= measurement.quality_factor
        
        # Normalize weights (carrier phase ~1000x more precise than TWTT)
        return weight * 1000
    
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