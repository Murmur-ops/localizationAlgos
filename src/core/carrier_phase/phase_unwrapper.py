"""
Phase Unwrapping and Cycle Slip Detection for Carrier Phase Measurements
Ensures continuous phase tracking for dynamic scenarios
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PhaseState:
    """State of phase tracking for a node pair"""
    wrapped_phases: deque  # Recent wrapped phase measurements [0, 2π)
    unwrapped_phases: deque  # Continuous unwrapped phases
    timestamps_ns: deque  # Measurement timestamps
    cycle_slips: List[int]  # Indices where cycle slips occurred
    phase_rates: deque  # Phase rate estimates (rad/s)
    integer_offset: int  # Current integer cycle offset


class PhaseUnwrapper:
    """
    Unwraps carrier phase measurements and detects cycle slips
    Critical for maintaining millimeter accuracy in dynamic scenarios
    """
    
    def __init__(self, window_size: int = 20,
                 cycle_slip_threshold: float = np.pi/2):
        """
        Initialize phase unwrapper
        
        Args:
            window_size: Number of measurements to keep in history
            cycle_slip_threshold: Phase jump threshold for cycle slip detection (rad)
        """
        self.window_size = window_size
        self.cycle_slip_threshold = cycle_slip_threshold
        
        # Track phase state for each node pair
        self.phase_states: Dict[Tuple[int, int], PhaseState] = {}
        
        # Statistics
        self.total_measurements = 0
        self.total_cycle_slips = 0
        
    def process_measurement(self, node_i: int, node_j: int,
                           wrapped_phase: float,
                           timestamp_ns: int) -> Tuple[float, bool]:
        """
        Process new phase measurement with unwrapping and cycle slip detection
        
        Args:
            node_i, node_j: Node pair
            wrapped_phase: Measured phase [0, 2π)
            timestamp_ns: Measurement timestamp
            
        Returns:
            (unwrapped_phase, cycle_slip_detected)
        """
        pair = (min(node_i, node_j), max(node_i, node_j))
        
        # Initialize state if needed
        if pair not in self.phase_states:
            self.phase_states[pair] = PhaseState(
                wrapped_phases=deque(maxlen=self.window_size),
                unwrapped_phases=deque(maxlen=self.window_size),
                timestamps_ns=deque(maxlen=self.window_size),
                cycle_slips=[],
                phase_rates=deque(maxlen=self.window_size),
                integer_offset=0
            )
        
        state = self.phase_states[pair]
        self.total_measurements += 1
        
        # First measurement - no unwrapping needed
        if not state.wrapped_phases:
            state.wrapped_phases.append(wrapped_phase)
            state.unwrapped_phases.append(wrapped_phase)
            state.timestamps_ns.append(timestamp_ns)
            return wrapped_phase, False
        
        # Predict phase based on history
        predicted_phase = self._predict_phase(state, timestamp_ns)
        
        # Check for cycle slip
        cycle_slip = self._detect_cycle_slip(
            wrapped_phase, predicted_phase, state
        )
        
        if cycle_slip:
            self.total_cycle_slips += 1
            state.cycle_slips.append(len(state.wrapped_phases))
            
            # Correct integer offset
            state.integer_offset = self._correct_integer_offset(
                wrapped_phase, predicted_phase, state.integer_offset
            )
            
            logger.warning(f"Cycle slip detected for pair {pair}: "
                         f"offset corrected to {state.integer_offset}")
        
        # Unwrap phase
        unwrapped = self._unwrap_single(
            wrapped_phase,
            state.unwrapped_phases[-1] if state.unwrapped_phases else wrapped_phase,
            state.integer_offset
        )
        
        # Update phase rate
        if len(state.timestamps_ns) > 0:
            dt_s = (timestamp_ns - state.timestamps_ns[-1]) / 1e9
            if dt_s > 0:
                phase_rate = (unwrapped - state.unwrapped_phases[-1]) / dt_s
                state.phase_rates.append(phase_rate)
        
        # Store measurement
        state.wrapped_phases.append(wrapped_phase)
        state.unwrapped_phases.append(unwrapped)
        state.timestamps_ns.append(timestamp_ns)
        
        return unwrapped, cycle_slip
    
    def _predict_phase(self, state: PhaseState, timestamp_ns: int) -> float:
        """
        Predict next phase based on history
        
        Uses linear or quadratic prediction depending on available data
        """
        if len(state.unwrapped_phases) < 2:
            # Not enough history - return last value
            return state.unwrapped_phases[-1] if state.unwrapped_phases else 0
        
        # Time since last measurement
        dt_s = (timestamp_ns - state.timestamps_ns[-1]) / 1e9
        
        if len(state.phase_rates) >= 3:
            # Use quadratic prediction (accounts for acceleration)
            times = np.array([0, 1, 2])  # Relative times
            phases = np.array(list(state.unwrapped_phases)[-3:])
            
            # Fit quadratic
            coeffs = np.polyfit(times, phases, 2)
            predicted = np.polyval(coeffs, 3)  # Predict next
            
        elif len(state.phase_rates) >= 1:
            # Use linear prediction with average rate
            avg_rate = np.mean(list(state.phase_rates)[-3:])
            predicted = state.unwrapped_phases[-1] + avg_rate * dt_s
            
        else:
            # Simple linear extrapolation
            phase_diff = state.unwrapped_phases[-1] - state.unwrapped_phases[-2]
            time_diff = (state.timestamps_ns[-1] - state.timestamps_ns[-2]) / 1e9
            
            if time_diff > 0:
                rate = phase_diff / time_diff
                predicted = state.unwrapped_phases[-1] + rate * dt_s
            else:
                predicted = state.unwrapped_phases[-1]
        
        return predicted
    
    def _detect_cycle_slip(self, wrapped_phase: float,
                          predicted_phase: float,
                          state: PhaseState) -> bool:
        """
        Detect cycle slip by comparing measurement with prediction
        
        Returns:
            True if cycle slip detected
        """
        # Unwrap predicted phase to same cycle as measurement
        predicted_wrapped = predicted_phase % (2 * np.pi)
        
        # Calculate phase difference
        diff = wrapped_phase - predicted_wrapped
        
        # Account for wrapping at 2π
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        
        # Check if difference exceeds threshold
        if abs(diff) > self.cycle_slip_threshold:
            # Additional check: is this consistent with recent rates?
            if len(state.phase_rates) >= 2:
                recent_rates = list(state.phase_rates)[-2:]
                rate_change = abs(recent_rates[-1] - recent_rates[-2])
                
                # If rate is changing rapidly, might not be a slip
                if rate_change > 10:  # rad/s threshold
                    return False
            
            return True
        
        return False
    
    def _correct_integer_offset(self, wrapped_phase: float,
                               predicted_phase: float,
                               current_offset: int) -> int:
        """
        Correct integer offset after cycle slip detection
        
        Args:
            wrapped_phase: New wrapped measurement
            predicted_phase: Predicted unwrapped phase
            current_offset: Current integer offset
            
        Returns:
            Corrected integer offset
        """
        # Find nearest integer multiple of 2π
        phase_diff = predicted_phase - wrapped_phase
        n_cycles = np.round(phase_diff / (2 * np.pi))
        
        return current_offset + int(n_cycles)
    
    def _unwrap_single(self, wrapped_phase: float,
                      prev_unwrapped: float,
                      integer_offset: int) -> float:
        """
        Unwrap single phase measurement
        
        Args:
            wrapped_phase: New wrapped phase [0, 2π)
            prev_unwrapped: Previous unwrapped phase
            integer_offset: Current integer offset
            
        Returns:
            Unwrapped phase
        """
        # Add integer offset
        base_unwrapped = wrapped_phase + integer_offset * 2 * np.pi
        
        # Find the unwrapped value closest to previous
        candidates = [
            base_unwrapped - 2 * np.pi,
            base_unwrapped,
            base_unwrapped + 2 * np.pi
        ]
        
        diffs = [abs(c - prev_unwrapped) for c in candidates]
        best_idx = np.argmin(diffs)
        
        return candidates[best_idx]
    
    def unwrap_batch(self, wrapped_phases: List[float]) -> List[float]:
        """
        Unwrap batch of phase measurements
        
        Classical unwrapping for offline processing
        
        Args:
            wrapped_phases: List of wrapped phases [0, 2π)
            
        Returns:
            List of unwrapped phases
        """
        if not wrapped_phases:
            return []
        
        unwrapped = [wrapped_phases[0]]
        
        for i in range(1, len(wrapped_phases)):
            diff = wrapped_phases[i] - wrapped_phases[i-1]
            
            # Check for 2π jumps
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            
            unwrapped.append(unwrapped[-1] + diff)
        
        return unwrapped
    
    def detect_and_correct_cycle_slips(self, pair: Tuple[int, int]) -> int:
        """
        Detect and correct cycle slips in stored measurements
        
        Args:
            pair: Node pair
            
        Returns:
            Number of cycle slips corrected
        """
        if pair not in self.phase_states:
            return 0
        
        state = self.phase_states[pair]
        
        if len(state.wrapped_phases) < 3:
            return 0
        
        # Re-process all measurements
        wrapped = list(state.wrapped_phases)
        timestamps = list(state.timestamps_ns)
        
        # Clear and rebuild
        state.unwrapped_phases.clear()
        state.cycle_slips.clear()
        state.integer_offset = 0
        
        corrections = 0
        
        for i, (phase, ts) in enumerate(zip(wrapped, timestamps)):
            if i == 0:
                state.unwrapped_phases.append(phase)
            else:
                # Predict based on history
                if i >= 2:
                    predicted = self._predict_phase(state, ts)
                    
                    if self._detect_cycle_slip(phase, predicted, state):
                        state.integer_offset = self._correct_integer_offset(
                            phase, predicted, state.integer_offset
                        )
                        state.cycle_slips.append(i)
                        corrections += 1
                
                # Unwrap with current offset
                unwrapped = self._unwrap_single(
                    phase,
                    state.unwrapped_phases[-1],
                    state.integer_offset
                )
                state.unwrapped_phases.append(unwrapped)
        
        return corrections
    
    def get_phase_quality(self, pair: Tuple[int, int]) -> Dict:
        """
        Assess phase tracking quality for a node pair
        
        Args:
            pair: Node pair
            
        Returns:
            Quality metrics
        """
        if pair not in self.phase_states:
            return {"status": "No measurements"}
        
        state = self.phase_states[pair]
        
        if len(state.unwrapped_phases) < 2:
            return {"status": "Insufficient measurements"}
        
        # Calculate metrics
        unwrapped = np.array(list(state.unwrapped_phases))
        
        # Phase continuity (should be smooth)
        phase_diffs = np.diff(unwrapped)
        continuity = np.std(phase_diffs)
        
        # Rate stability
        if state.phase_rates:
            rates = np.array(list(state.phase_rates))
            rate_stability = np.std(rates)
        else:
            rate_stability = 0
        
        quality = {
            "num_measurements": len(state.unwrapped_phases),
            "num_cycle_slips": len(state.cycle_slips),
            "phase_continuity": continuity,
            "rate_stability": rate_stability,
            "integer_offset": state.integer_offset,
            "quality_score": np.exp(-continuity) * np.exp(-len(state.cycle_slips))
        }
        
        return quality
    
    def get_statistics(self) -> Dict:
        """Get overall unwrapping statistics"""
        
        stats = {
            "num_pairs": len(self.phase_states),
            "total_measurements": self.total_measurements,
            "total_cycle_slips": self.total_cycle_slips,
            "cycle_slip_rate": self.total_cycle_slips / max(self.total_measurements, 1)
        }
        
        # Quality scores
        quality_scores = []
        for pair in self.phase_states:
            q = self.get_phase_quality(pair)
            if "quality_score" in q:
                quality_scores.append(q["quality_score"])
        
        if quality_scores:
            stats["avg_quality"] = np.mean(quality_scores)
            stats["min_quality"] = np.min(quality_scores)
            stats["max_quality"] = np.max(quality_scores)
        
        return stats


def test_phase_unwrapper():
    """Test phase unwrapping and cycle slip detection"""
    
    print("="*60)
    print("PHASE UNWRAPPER TEST")
    print("="*60)
    
    unwrapper = PhaseUnwrapper()
    
    # Simulate dynamic scenario with phase measurements
    n_measurements = 50
    dt = 0.1  # 100ms intervals
    
    # True phase evolution (with some dynamics)
    true_phases = []
    phase = 0
    rate = 0.5  # rad/s initial rate
    
    for i in range(n_measurements):
        # Add some dynamics
        if i == 20:
            rate = 2.0  # Speed up
        if i == 35:
            rate = -1.0  # Reverse
        if i == 40:
            # Simulate cycle slip
            phase += 2 * np.pi
        
        phase += rate * dt
        true_phases.append(phase)
    
    print("\nProcessing measurements with dynamics and cycle slip...")
    print("-"*40)
    
    detected_slips = []
    unwrapped_phases = []
    
    for i, true_phase in enumerate(true_phases):
        # Wrap phase
        wrapped = true_phase % (2 * np.pi)
        
        # Add measurement noise
        wrapped += np.random.normal(0, 0.01)
        wrapped = wrapped % (2 * np.pi)
        
        # Process measurement
        timestamp = int(i * dt * 1e9)
        unwrapped, slip = unwrapper.process_measurement(0, 1, wrapped, timestamp)
        
        unwrapped_phases.append(unwrapped)
        
        if slip:
            detected_slips.append(i)
            print(f"Cycle slip detected at measurement {i}")
    
    # Get quality metrics
    quality = unwrapper.get_phase_quality((0, 1))
    
    print(f"\nPhase Tracking Quality:")
    print(f"  Measurements: {quality['num_measurements']}")
    print(f"  Cycle slips: {quality['num_cycle_slips']}")
    print(f"  Continuity: {quality['phase_continuity']:.3f}")
    print(f"  Rate stability: {quality['rate_stability']:.3f}")
    print(f"  Quality score: {quality['quality_score']:.3f}")
    
    # Check unwrapping accuracy
    unwrapped_array = np.array(unwrapped_phases)
    true_array = np.array(true_phases)
    
    # Remove cycle slip offset for comparison
    if detected_slips:
        slip_idx = detected_slips[0]
        true_array[slip_idx:] -= 2 * np.pi
    
    errors = unwrapped_array - true_array
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"\nUnwrapping Accuracy:")
    print(f"  RMSE: {rmse:.3f} rad")
    print(f"  Max error: {np.max(np.abs(errors)):.3f} rad")
    
    # Overall statistics
    stats = unwrapper.get_statistics()
    print(f"\nOverall Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    return unwrapper


if __name__ == "__main__":
    test_phase_unwrapper()