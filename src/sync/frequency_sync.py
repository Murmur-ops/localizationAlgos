"""
Frequency and Time Synchronization
Implements PLL for CFO/SRO and PTP-style time sync
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
from collections import deque


@dataclass
class PLLConfig:
    """Phase-Locked Loop configuration"""
    loop_bandwidth_hz: float = 100.0  # Loop bandwidth
    damping_factor: float = 0.707  # Critical damping
    sample_rate_hz: float = 200e6
    pilot_frequencies_mhz: List[float] = field(default_factory=lambda: [0, 1, 3, 5])
    
    def __post_init__(self):
        # Calculate loop filter coefficients
        wn = 2 * np.pi * self.loop_bandwidth_hz
        K = wn**2
        self.alpha = 2 * self.damping_factor * wn  # Proportional gain
        self.beta = wn**2  # Integral gain


class PilotPLL:
    """PLL for carrier frequency offset (CFO) and sample rate offset (SRO) tracking"""
    
    def __init__(self, config: PLLConfig):
        self.config = config
        self.dt = 1.0 / config.sample_rate_hz
        
        # State variables
        self.phase = 0.0
        self.frequency = 0.0  # CFO estimate
        self.phase_error_integral = 0.0
        
        # SRO estimation
        self.sro_ppm = 0.0
        self.timing_offset = 0.0
        
        # Tracking metrics
        self.phase_variance = 1.0
        self.frequency_variance = 1.0
        self.locked = False
        
        # History for variance estimation
        self.phase_error_history = deque(maxlen=100)
        self.frequency_history = deque(maxlen=100)
        
    def process_pilot(self, received_signal: np.ndarray) -> dict:
        """Process received pilot signal through PLL"""
        
        results = {
            'cfo_hz': [],
            'sro_ppm': [],
            'phase': [],
            'locked': []
        }
        
        for i, sample in enumerate(received_signal):
            # Phase detector (assuming complex baseband)
            phase_error = np.angle(sample * np.exp(-1j * self.phase))
            
            # Wrap phase error to [-π, π]
            phase_error = np.angle(np.exp(1j * phase_error))
            
            # Loop filter (PI controller)
            self.phase_error_integral += phase_error * self.dt
            frequency_correction = (self.config.alpha * phase_error + 
                                   self.config.beta * self.phase_error_integral)
            
            # Update NCO
            self.frequency += frequency_correction * self.dt
            self.phase += 2 * np.pi * self.frequency * self.dt
            
            # Wrap phase
            self.phase = np.angle(np.exp(1j * self.phase))
            
            # Track history
            self.phase_error_history.append(phase_error)
            self.frequency_history.append(self.frequency)
            
            # Estimate variance
            if len(self.phase_error_history) >= 50:
                self.phase_variance = np.var(self.phase_error_history)
                self.frequency_variance = np.var(self.frequency_history)
                
                # Lock detection
                self.locked = (self.phase_variance < 0.1 and 
                              self.frequency_variance < 100.0)
            
            # Estimate SRO from frequency drift rate
            if i > 1000 and len(results['cfo_hz']) > 0:
                # Look back at most 10 samples (1000 signal samples each)
                lookback_idx = max(0, len(results['cfo_hz']) - 10)
                if lookback_idx < len(results['cfo_hz']):
                    time_elapsed = (i - lookback_idx * 100) * self.dt
                    if time_elapsed > 0:
                        freq_drift = (self.frequency - results['cfo_hz'][lookback_idx]) / time_elapsed
                        self.sro_ppm = freq_drift / 1e6  # Convert to ppm
            
            # Store results
            if i % 100 == 0:  # Decimate output
                results['cfo_hz'].append(self.frequency)
                results['sro_ppm'].append(self.sro_ppm)
                results['phase'].append(self.phase)
                results['locked'].append(self.locked)
        
        return {
            'cfo_hz': self.frequency,
            'sro_ppm': self.sro_ppm,
            'phase_var': self.phase_variance,
            'freq_var': self.frequency_variance,
            'locked': self.locked,
            'history': results
        }


@dataclass
class TimeSyncConfig:
    """Configuration for time synchronization"""
    turnaround_time_ns: float = 1000  # Processing delay
    timestamp_jitter_ns: float = 10   # Hardware timestamp uncertainty
    kalman_process_noise: float = 1e-9
    kalman_measurement_noise: float = 1e-6


class HardwareTimestampSimulator:
    """Simulates hardware timestamping at MAC/PHY boundary"""
    
    def __init__(self, node_id: int, clock_offset_ns: float = 0, clock_skew_ppm: float = 0):
        self.node_id = node_id
        self.clock_offset_ns = clock_offset_ns
        self.clock_skew_ppm = clock_skew_ppm
        self.time_base_ns = 0
        
    def get_timestamp(self, true_time_ns: float, is_tx: bool = True) -> int:
        """Get hardware timestamp with realistic imperfections"""
        # Apply clock skew
        local_time = true_time_ns * (1 + self.clock_skew_ppm * 1e-6) + self.clock_offset_ns
        
        # Add jitter (different for TX/RX)
        jitter_std = 5 if is_tx else 10  # RX has more jitter
        jitter = np.random.normal(0, jitter_std)
        
        # Quantize to nanoseconds
        return int(local_time + jitter)


class PTPTimeSync:
    """PTP-style two-way time synchronization"""
    
    def __init__(self, config: TimeSyncConfig, node_id: int):
        self.config = config
        self.node_id = node_id
        
        # Kalman filter state for each neighbor
        self.neighbor_states = {}  # neighbor_id -> KalmanState
        
    def process_sync_exchange(self, neighbor_id: int, 
                             t1: int, t2: int, t3: int, t4: int) -> dict:
        """Process four timestamps from sync exchange"""
        
        # Calculate offset and delay
        # offset = ((t2 - t1) - (t4 - t3)) / 2
        # delay = ((t4 - t1) + (t3 - t2)) / 2
        
        offset_ns = ((t2 - t1) - (t4 - t3)) / 2.0
        rtt_ns = (t4 - t1) - (t3 - t2)
        one_way_delay_ns = rtt_ns / 2.0
        
        # Initialize Kalman filter for this neighbor if needed
        if neighbor_id not in self.neighbor_states:
            self.neighbor_states[neighbor_id] = KalmanTimeFilter(self.config)
        
        # Update Kalman filter
        kf = self.neighbor_states[neighbor_id]
        kf.update(offset_ns, rtt_ns)
        
        return {
            'neighbor_id': neighbor_id,
            'offset_ns': offset_ns,
            'rtt_ns': rtt_ns,
            'one_way_delay_ns': one_way_delay_ns,
            'filtered_offset_ns': kf.offset_estimate,
            'filtered_skew_ppm': kf.skew_estimate * 1e6,
            'offset_variance': kf.offset_variance,
            'timestamps': {'t1': t1, 't2': t2, 't3': t3, 't4': t4}
        }


class KalmanTimeFilter:
    """Kalman filter for time offset and skew estimation"""
    
    def __init__(self, config: TimeSyncConfig):
        self.config = config
        
        # State: [offset, skew]
        self.state = np.array([0.0, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 1e6
        
        # Process noise
        self.Q = np.array([[config.kalman_process_noise, 0],
                          [0, config.kalman_process_noise * 1e-6]])
        
        # Measurement noise
        self.R = config.kalman_measurement_noise
        
        # Measurement count (for skew estimation)
        self.measurement_count = 0
        self.last_time_ns = 0
        
    @property
    def offset_estimate(self) -> float:
        return self.state[0]
    
    @property
    def skew_estimate(self) -> float:
        return self.state[1]
    
    @property
    def offset_variance(self) -> float:
        return self.P[0, 0]
    
    def update(self, measured_offset_ns: float, rtt_ns: float):
        """Update filter with new measurement"""
        
        current_time_ns = measured_offset_ns  # Simplified
        
        if self.measurement_count > 0:
            # Time since last update
            dt = (current_time_ns - self.last_time_ns) * 1e-9  # Convert to seconds
            
            # State transition matrix
            F = np.array([[1, dt],
                         [0, 1]])
            
            # Predict
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q
            
            # Measurement matrix (we only measure offset)
            H = np.array([1, 0])
            
            # Innovation
            y = measured_offset_ns - H @ self.state
            S = H @ self.P @ H.T + self.R
            
            # Kalman gain
            K = self.P @ H.T / S
            
            # Update
            self.state = self.state + K * y
            self.P = (np.eye(2) - np.outer(K, H)) @ self.P
            
        else:
            # First measurement - initialize
            self.state[0] = measured_offset_ns
        
        self.measurement_count += 1
        self.last_time_ns = current_time_ns


class DistributedFrequencyConsensus:
    """DFAC - Distributed Frequency Agreement via Consensus"""
    
    def __init__(self, node_id: int, neighbor_ids: List[int]):
        self.node_id = node_id
        self.neighbor_ids = neighbor_ids
        
        # Local estimates
        self.local_cfo = 0.0
        self.local_sro = 0.0
        
        # Neighbor estimates
        self.neighbor_cfos = {nid: 0.0 for nid in neighbor_ids}
        self.neighbor_sros = {nid: 0.0 for nid in neighbor_ids}
        
        # Consensus weights (doubly stochastic)
        n = len(neighbor_ids) + 1
        self.weight_self = 1.0 / n
        self.weight_neighbor = 1.0 / n
        
    def update_local_estimate(self, cfo_hz: float, sro_ppm: float):
        """Update local frequency estimates from PLL"""
        self.local_cfo = cfo_hz
        self.local_sro = sro_ppm
    
    def receive_neighbor_estimate(self, neighbor_id: int, cfo_hz: float, sro_ppm: float):
        """Receive frequency estimate from neighbor"""
        if neighbor_id in self.neighbor_cfos:
            self.neighbor_cfos[neighbor_id] = cfo_hz
            self.neighbor_sros[neighbor_id] = sro_ppm
    
    def consensus_update(self) -> Tuple[float, float]:
        """Perform consensus update (weighted average)"""
        # CFO consensus
        new_cfo = self.weight_self * self.local_cfo
        for nid in self.neighbor_ids:
            new_cfo += self.weight_neighbor * self.neighbor_cfos[nid]
        
        # SRO consensus  
        new_sro = self.weight_self * self.local_sro
        for nid in self.neighbor_ids:
            new_sro += self.weight_neighbor * self.neighbor_sros[nid]
        
        # Update local estimates
        self.local_cfo = new_cfo
        self.local_sro = new_sro
        
        return new_cfo, new_sro


if __name__ == "__main__":
    # Test PLL
    print("Testing PLL...")
    pll_config = PLLConfig()
    pll = PilotPLL(pll_config)
    
    # Generate test signal with CFO
    t = np.arange(10000) / pll_config.sample_rate_hz
    cfo_true = 1000  # 1 kHz offset
    signal_test = np.exp(2j * np.pi * cfo_true * t)
    
    result = pll.process_pilot(signal_test)
    print(f"  True CFO: {cfo_true} Hz")
    print(f"  Estimated CFO: {result['cfo_hz']:.1f} Hz")
    print(f"  Locked: {result['locked']}")
    
    # Test time sync
    print("\nTesting Time Sync...")
    ts_config = TimeSyncConfig()
    
    # Create two nodes with clock offsets
    node_a = HardwareTimestampSimulator(1, clock_offset_ns=1000, clock_skew_ppm=10)
    node_b = HardwareTimestampSimulator(2, clock_offset_ns=-500, clock_skew_ppm=-5)
    
    sync_a = PTPTimeSync(ts_config, 1)
    
    # Simulate sync exchange
    true_time = 1_000_000_000  # 1 second
    
    t1 = node_a.get_timestamp(true_time, is_tx=True)
    t2 = node_b.get_timestamp(true_time + 1000, is_tx=False)  # 1µs propagation
    t3 = node_b.get_timestamp(true_time + 2000, is_tx=True)
    t4 = node_a.get_timestamp(true_time + 3000, is_tx=False)
    
    result = sync_a.process_sync_exchange(2, t1, t2, t3, t4)
    
    print(f"  Clock offset (true): {1000 - (-500)} ns")
    print(f"  Estimated offset: {result['offset_ns']:.1f} ns")
    print(f"  RTT: {result['rtt_ns']:.1f} ns")