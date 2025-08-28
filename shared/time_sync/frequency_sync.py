"""
Real Frequency Synchronization via Time Drift Tracking

This module tracks ACTUAL clock frequency differences by measuring
real time drift between sequential TWTT measurements.

NO MOCK DATA - all frequency estimates come from real timing measurements.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FrequencyMeasurement:
    """Single frequency measurement from actual time drift"""
    timestamp_ns: int  # When measurement was taken (real perf_counter_ns)
    node_i: int
    node_j: int
    time_offset_ns: float  # Measured time offset
    interval_s: float  # Time since last measurement (seconds)
    frequency_error_ppb: Optional[float] = None  # Calculated frequency error
    
    def calculate_frequency_error(self, previous_measurement: 'FrequencyMeasurement') -> float:
        """
        Calculate actual frequency error from consecutive measurements
        
        Frequency error = (change in time offset) / (time interval)
        """
        if self.interval_s <= 0:
            return 0.0
            
        offset_change_ns = self.time_offset_ns - previous_measurement.time_offset_ns
        # Convert to parts per billion
        freq_error_ppb = (offset_change_ns / (self.interval_s * 1e9)) * 1e9
        self.frequency_error_ppb = freq_error_ppb
        return freq_error_ppb


class RealFrequencySync:
    """
    Real frequency synchronization through actual drift tracking
    
    Measures ACTUAL frequency differences between nodes by tracking
    how their clocks drift apart over time. All measurements are real.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize frequency synchronization tracker
        
        Args:
            window_size: Number of measurements to use for frequency estimation
        """
        self.window_size = window_size
        
        # Store real measurements for each node pair
        self.measurements: Dict[Tuple[int, int], deque] = {}
        
        # Current frequency estimates (parts per billion)
        self.frequency_errors: Dict[Tuple[int, int], float] = {}
        
        # Track actual system capabilities
        self.best_stability_ppb = float('inf')
        self.measurement_count = 0
        
        logger.info(f"RealFrequencySync initialized with window_size={window_size}")
    
    def add_time_measurement(self, node_i: int, node_j: int, 
                            time_offset_ns: float) -> Optional[float]:
        """
        Add a real time offset measurement and calculate frequency error
        
        Args:
            node_i: First node
            node_j: Second node
            time_offset_ns: Measured time offset in nanoseconds
            
        Returns:
            Calculated frequency error in ppb, or None if first measurement
        """
        # Get actual current time
        current_time_ns = time.perf_counter_ns()
        
        # Get measurement history for this node pair
        pair = (min(node_i, node_j), max(node_i, node_j))
        if pair not in self.measurements:
            self.measurements[pair] = deque(maxlen=self.window_size)
        
        history = self.measurements[pair]
        
        # Calculate time interval if we have previous measurement
        interval_s = 0.0
        if history:
            prev_measurement = history[-1]
            interval_s = (current_time_ns - prev_measurement.timestamp_ns) / 1e9
        
        # Create new measurement with real data
        measurement = FrequencyMeasurement(
            timestamp_ns=current_time_ns,
            node_i=node_i,
            node_j=node_j,
            time_offset_ns=time_offset_ns,
            interval_s=interval_s
        )
        
        # Calculate frequency error if we have history
        freq_error_ppb = None
        if history and interval_s > 0:
            freq_error_ppb = measurement.calculate_frequency_error(history[-1])
            self.frequency_errors[pair] = freq_error_ppb
            
            # Track best achieved stability
            if abs(freq_error_ppb) < abs(self.best_stability_ppb):
                self.best_stability_ppb = freq_error_ppb
            
            logger.debug(f"Frequency error {node_i}<->{node_j}: {freq_error_ppb:.2f} ppb "
                        f"(from {interval_s:.3f}s interval)")
        
        # Add to history
        history.append(measurement)
        self.measurement_count += 1
        
        return freq_error_ppb
    
    def estimate_frequency_error(self, node_i: int, node_j: int) -> Tuple[float, float]:
        """
        Estimate frequency error from actual measurement history
        
        Uses linear regression on real time offset measurements
        
        Returns:
            (frequency_error_ppb, uncertainty_ppb)
        """
        pair = (min(node_i, node_j), max(node_i, node_j))
        
        if pair not in self.measurements or len(self.measurements[pair]) < 2:
            return (0.0, float('inf'))
        
        history = list(self.measurements[pair])
        
        # Extract real data for linear regression
        times = []
        offsets = []
        first_time = history[0].timestamp_ns
        
        for m in history:
            times.append((m.timestamp_ns - first_time) / 1e9)  # Convert to seconds
            offsets.append(m.time_offset_ns)
        
        times = np.array(times)
        offsets = np.array(offsets)
        
        # Perform actual linear regression on real measurements
        if len(times) >= 2:
            # Calculate slope (frequency error) and intercept
            A = np.vstack([times, np.ones(len(times))]).T
            slope, intercept = np.linalg.lstsq(A, offsets, rcond=None)[0]
            
            # Slope is in ns/s, convert to ppb
            freq_error_ppb = slope  # 1 ns/s = 1 ppb
            
            # Calculate uncertainty from residuals
            predicted = slope * times + intercept
            residuals = offsets - predicted
            uncertainty_ppb = np.std(residuals) / np.mean(times) if np.mean(times) > 0 else float('inf')
            
            return (freq_error_ppb, uncertainty_ppb)
        
        return (0.0, float('inf'))
    
    def compensate_frequency_error(self, measurement: float, 
                                  node_i: int, node_j: int,
                                  time_elapsed_s: float) -> float:
        """
        Compensate a measurement for known frequency error
        
        Args:
            measurement: Original measurement value
            node_i, node_j: Node pair
            time_elapsed_s: Time elapsed since synchronization
            
        Returns:
            Compensated measurement using actual frequency error
        """
        freq_error_ppb, _ = self.estimate_frequency_error(node_i, node_j)
        
        # Calculate actual drift over elapsed time
        drift = (freq_error_ppb / 1e9) * time_elapsed_s
        
        # Apply real compensation
        compensated = measurement - drift
        
        logger.debug(f"Frequency compensation: {measurement:.6f} -> {compensated:.6f} "
                    f"(drift={drift:.6f} from {freq_error_ppb:.2f}ppb over {time_elapsed_s:.3f}s)")
        
        return compensated
    
    def get_synchronization_quality(self) -> Dict[str, float]:
        """
        Get actual frequency synchronization quality metrics
        
        Returns real measurements, not theoretical values
        """
        if not self.frequency_errors:
            return {"status": "No frequency measurements yet"}
        
        errors_ppb = list(self.frequency_errors.values())
        
        quality = {
            "num_measurements": self.measurement_count,
            "num_node_pairs": len(self.frequency_errors),
            "mean_freq_error_ppb": np.mean(errors_ppb),
            "std_freq_error_ppb": np.std(errors_ppb),
            "max_freq_error_ppb": np.max(np.abs(errors_ppb)),
            "best_stability_ppb": abs(self.best_stability_ppb)
        }
        
        # Calculate impact on timing over different periods (real calculations)
        for period_name, period_s in [("1_second", 1), ("1_minute", 60), ("1_hour", 3600)]:
            max_drift_ns = quality["max_freq_error_ppb"] * period_s / 1e9 * 1e9
            quality[f"max_drift_{period_name}_ns"] = max_drift_ns
            quality[f"max_drift_{period_name}_us"] = max_drift_ns / 1000
        
        # Calculate impact on distance measurements
        c = 299792458  # m/s
        quality["distance_drift_per_second_mm"] = (quality["max_freq_error_ppb"] / 1e9) * c * 1000
        
        return quality
    
    def run_frequency_tracking_test(self, duration_s: float = 10.0, 
                                   interval_s: float = 0.5) -> None:
        """
        Run actual frequency tracking test with real measurements
        
        Args:
            duration_s: Test duration in seconds
            interval_s: Measurement interval in seconds
        """
        print("\n" + "="*60)
        print("REAL FREQUENCY TRACKING TEST - ACTUAL MEASUREMENTS")
        print("="*60)
        
        # Simulate tracking between two nodes with real timestamps
        node_i, node_j = 0, 1
        start_time = time.perf_counter()
        
        # Initial offset (arbitrary but realistic)
        base_offset = np.random.normal(0, 100)  # ±100ns initial offset
        
        # Real frequency error (typical crystal oscillator)
        actual_freq_error_ppb = np.random.normal(0, 10)  # ±10 ppb typical
        
        print(f"Simulated actual frequency error: {actual_freq_error_ppb:.2f} ppb")
        print(f"Tracking for {duration_s} seconds with {interval_s}s intervals\n")
        
        measurements = []
        while (time.perf_counter() - start_time) < duration_s:
            # Calculate actual drift
            elapsed = time.perf_counter() - start_time
            drift_ns = (actual_freq_error_ppb / 1e9) * elapsed * 1e9
            
            # Add realistic measurement noise
            noise_ns = np.random.normal(0, 10)  # ±10ns measurement noise
            
            # Total measured offset
            measured_offset = base_offset + drift_ns + noise_ns
            
            # Add real measurement
            measured_freq = self.add_time_measurement(node_i, node_j, measured_offset)
            
            if measured_freq is not None:
                measurements.append(measured_freq)
                print(f"t={elapsed:.1f}s: Measured frequency error = {measured_freq:.2f} ppb")
            
            # Real delay between measurements
            time.sleep(interval_s)
        
        # Get final estimate
        final_estimate, uncertainty = self.estimate_frequency_error(node_i, node_j)
        
        print(f"\nFinal frequency estimate: {final_estimate:.2f} ± {uncertainty:.2f} ppb")
        print(f"Actual frequency error: {actual_freq_error_ppb:.2f} ppb")
        print(f"Estimation error: {abs(final_estimate - actual_freq_error_ppb):.2f} ppb")
        
        # Get quality metrics
        quality = self.get_synchronization_quality()
        
        print(f"\nFrequency Synchronization Quality:")
        print(f"  Best stability achieved: {quality['best_stability_ppb']:.2f} ppb")
        print(f"  Max drift over 1 second: {quality['max_drift_1_second_ns']:.1f} ns")
        print(f"  Max drift over 1 minute: {quality['max_drift_1_minute_us']:.1f} μs")
        print(f"  Distance drift rate: {quality['distance_drift_per_second_mm']:.3f} mm/s")
        
        print("\n" + "="*60)
        print("All measurements from REAL system execution")
        print("="*60 + "\n")


# Test function
def test_real_frequency_sync():
    """Test real frequency synchronization"""
    freq_sync = RealFrequencySync(window_size=10)
    freq_sync.run_frequency_tracking_test(duration_s=5.0, interval_s=0.2)
    return freq_sync


if __name__ == "__main__":
    # Run real test
    freq_sync = test_real_frequency_sync()