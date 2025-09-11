"""
Channel Propagation Models
Implements realistic RF propagation including path loss, multipath, and NLOS
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum


class PropagationType(Enum):
    """Types of propagation conditions"""
    LOS = "line_of_sight"
    NLOS = "non_line_of_sight"
    OBSTRUCTED = "obstructed"


@dataclass
class ChannelConfig:
    """Channel model configuration"""
    carrier_freq_hz: float = 2.4e9  # 2.4 GHz
    bandwidth_hz: float = 100e6     # 100 MHz
    
    # Path loss model parameters
    path_loss_exponent: float = 2.0  # Free space
    reference_distance_m: float = 1.0
    reference_loss_db: float = 40.0
    
    # Multipath parameters
    enable_multipath: bool = True
    rician_k_factor_db: float = 10.0  # LOS dominance
    delay_spread_ns: float = 50.0     # RMS delay spread
    
    # NLOS parameters
    nlos_bias_mean_m: float = 5.0
    nlos_bias_std_m: float = 2.0
    nlos_excess_loss_db: float = 10.0
    
    # Noise floor
    noise_figure_db: float = 6.0
    thermal_noise_dbm: float = -174.0  # per Hz at 290K


class PathLossModel:
    """Path loss models for different environments"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.wavelength = 3e8 / config.carrier_freq_hz
    
    def free_space(self, distance_m: float) -> float:
        """Free space path loss (Friis equation)"""
        if distance_m < self.config.reference_distance_m:
            distance_m = self.config.reference_distance_m
        
        # PL = 20*log10(4*pi*d/lambda)
        path_loss_db = (20 * np.log10(4 * np.pi * distance_m / self.wavelength))
        return path_loss_db
    
    def log_distance(self, distance_m: float, environment: str = "urban") -> float:
        """Log-distance path loss model"""
        if distance_m < self.config.reference_distance_m:
            distance_m = self.config.reference_distance_m
        
        # Environment-specific path loss exponents
        exponents = {
            "free_space": 2.0,
            "urban": 3.5,
            "indoor_los": 1.8,
            "indoor_nlos": 4.0,
            "suburban": 3.0
        }
        
        n = exponents.get(environment, self.config.path_loss_exponent)
        
        # PL = PL(d0) + 10*n*log10(d/d0)
        path_loss_db = (self.config.reference_loss_db + 
                       10 * n * np.log10(distance_m / self.config.reference_distance_m))
        
        return path_loss_db
    
    def two_ray(self, distance_m: float, tx_height_m: float = 10.0, 
                rx_height_m: float = 1.5) -> float:
        """Two-ray ground reflection model"""
        if distance_m < self.config.reference_distance_m:
            distance_m = self.config.reference_distance_m
        
        # Breakpoint distance
        d_break = 4 * tx_height_m * rx_height_m / self.wavelength
        
        if distance_m < d_break:
            # Use free space below breakpoint
            return self.free_space(distance_m)
        else:
            # Two-ray model: PL = 40*log10(d) - 10*log10(ht^2 * hr^2)
            path_loss_db = (40 * np.log10(distance_m) - 
                           10 * np.log10(tx_height_m**2 * rx_height_m**2))
            return path_loss_db


class MultipathChannel:
    """Multipath fading channel model"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.sample_rate = config.bandwidth_hz * 2  # Nyquist
        
    def generate_channel_response(self, distance_m: float, 
                                 prop_type: PropagationType) -> Dict:
        """Generate channel impulse response with multipath"""
        
        # Convert RMS delay spread to samples
        delay_spread_samples = int(self.config.delay_spread_ns * 1e-9 * self.sample_rate)
        
        if prop_type == PropagationType.LOS:
            # Rician fading for LOS
            k_linear = 10**(self.config.rician_k_factor_db / 10)
            
            # Generate channel taps
            n_taps = max(1, delay_spread_samples * 3)
            delays_ns = np.arange(n_taps) * (1e9 / self.sample_rate)
            
            # Power delay profile (exponential decay)
            tau_rms = self.config.delay_spread_ns
            tap_powers = np.exp(-delays_ns / tau_rms)
            tap_powers = tap_powers / np.sum(tap_powers)
            
            # LOS component (first tap)
            los_power = k_linear / (k_linear + 1)
            nlos_power = 1 / (k_linear + 1)
            
            # Generate complex channel coefficients
            h = np.zeros(n_taps, dtype=complex)
            h[0] = np.sqrt(los_power)  # LOS component
            
            # NLOS components (Rayleigh)
            for i in range(1, n_taps):
                h[i] = np.sqrt(tap_powers[i] * nlos_power / 2) * (
                    np.random.randn() + 1j * np.random.randn())
            
        else:  # NLOS or OBSTRUCTED
            # Rayleigh fading for NLOS
            n_taps = max(1, delay_spread_samples * 3)
            delays_ns = np.arange(n_taps) * (1e9 / self.sample_rate)
            
            # Power delay profile
            tau_rms = self.config.delay_spread_ns * 2  # Larger spread for NLOS
            tap_powers = np.exp(-delays_ns / tau_rms)
            tap_powers = tap_powers / np.sum(tap_powers)
            
            # Generate Rayleigh fading taps
            h = np.sqrt(tap_powers / 2) * (np.random.randn(n_taps) + 
                                           1j * np.random.randn(n_taps))
        
        return {
            'impulse_response': h,
            'delays_ns': delays_ns,
            'tap_powers_db': 10 * np.log10(np.abs(h)**2 + 1e-10),
            'coherence_bandwidth_hz': 1 / (2 * np.pi * tau_rms * 1e-9),
            'rms_delay_spread_ns': tau_rms
        }
    
    def apply_multipath(self, signal: np.ndarray, channel: Dict) -> np.ndarray:
        """Apply multipath channel to signal"""
        # Convolve signal with channel impulse response
        output = np.convolve(signal, channel['impulse_response'], mode='same')
        return output


class RangingChannel:
    """Complete channel model for ranging measurements"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.path_loss = PathLossModel(config)
        self.multipath = MultipathChannel(config)
        
    def calculate_snr(self, distance_m: float, tx_power_dbm: float = 20.0,
                     environment: str = "urban") -> float:
        """Calculate SNR based on distance and environment"""
        # Path loss
        pl_db = self.path_loss.log_distance(distance_m, environment)
        
        # Received power
        rx_power_dbm = tx_power_dbm - pl_db
        
        # Noise power
        noise_power_dbm = (self.config.thermal_noise_dbm + 
                          10 * np.log10(self.config.bandwidth_hz) +
                          self.config.noise_figure_db)
        
        # SNR
        snr_db = rx_power_dbm - noise_power_dbm
        return snr_db
    
    def generate_measurement(self, true_distance_m: float, 
                           prop_type: PropagationType = PropagationType.LOS,
                           environment: str = "urban") -> Dict:
        """Generate ranging measurement with all channel effects"""
        
        # Calculate SNR
        snr_db = self.calculate_snr(true_distance_m, environment=environment)
        
        # Add NLOS bias if applicable
        measured_distance_m = true_distance_m
        if prop_type == PropagationType.NLOS:
            # NLOS always adds positive bias
            bias_m = np.abs(np.random.normal(self.config.nlos_bias_mean_m, 
                                            self.config.nlos_bias_std_m))
            measured_distance_m += bias_m
            snr_db -= self.config.nlos_excess_loss_db
        
        elif prop_type == PropagationType.OBSTRUCTED:
            # Heavy obstruction
            bias_m = np.abs(np.random.normal(self.config.nlos_bias_mean_m * 2, 
                                            self.config.nlos_bias_std_m * 2))
            measured_distance_m += bias_m
            snr_db -= self.config.nlos_excess_loss_db * 2
        
        # Calculate measurement variance based on Cramér-Rao bound
        c = 3e8
        snr_linear = 10**(snr_db / 10)
        beta_squared = (self.config.bandwidth_hz)**2 / 12
        
        # Range variance in meters
        range_std_m = c / (2 * np.sqrt(beta_squared * snr_linear))
        
        # Add measurement noise
        noise_m = np.random.normal(0, range_std_m)
        measured_distance_m += noise_m
        
        # Generate multipath channel response
        channel = self.multipath.generate_channel_response(true_distance_m, prop_type)
        
        # Calculate quality score (0-1, higher is better)
        quality_score = self._calculate_quality_score(snr_db, prop_type, channel)
        
        return {
            'true_distance_m': true_distance_m,
            'measured_distance_m': max(0, measured_distance_m),  # Can't be negative
            'snr_db': snr_db,
            'propagation_type': prop_type,
            'measurement_std_m': range_std_m,
            'quality_score': quality_score,
            'channel_response': channel,
            'nlos_bias_m': measured_distance_m - true_distance_m - noise_m if prop_type != PropagationType.LOS else 0
        }
    
    def _calculate_quality_score(self, snr_db: float, prop_type: PropagationType,
                                channel: Dict) -> float:
        """Calculate measurement quality score"""
        # SNR contribution (sigmoid)
        snr_score = 1 / (1 + np.exp(-(snr_db - 10) / 5))
        
        # Propagation type contribution
        prop_scores = {
            PropagationType.LOS: 1.0,
            PropagationType.NLOS: 0.5,
            PropagationType.OBSTRUCTED: 0.2
        }
        prop_score = prop_scores[prop_type]
        
        # Multipath contribution (based on delay spread)
        delay_spread_score = np.exp(-channel['rms_delay_spread_ns'] / 100)
        
        # Combined score
        quality = 0.5 * snr_score + 0.3 * prop_score + 0.2 * delay_spread_score
        return np.clip(quality, 0, 1)


class OutlierDetector:
    """Detect and handle NLOS/outlier measurements"""
    
    def __init__(self, innovation_threshold: float = 3.0):
        self.innovation_threshold = innovation_threshold
        self.measurement_history = {}
        
    def is_outlier(self, node_id: int, neighbor_id: int, 
                   measurement: Dict) -> bool:
        """Detect if measurement is an outlier"""
        
        key = (node_id, neighbor_id)
        
        # Initialize history if needed
        if key not in self.measurement_history:
            self.measurement_history[key] = []
        
        history = self.measurement_history[key]
        
        if len(history) < 5:
            # Not enough history, accept measurement
            history.append(measurement['measured_distance_m'])
            return False
        
        # Calculate innovation (prediction error)
        predicted = np.mean(history[-5:])
        innovation = abs(measurement['measured_distance_m'] - predicted)
        
        # Calculate threshold based on historical variance
        historical_std = np.std(history[-10:]) if len(history) >= 10 else np.std(history)
        threshold = self.innovation_threshold * historical_std
        
        # Check if outlier
        is_outlier = innovation > threshold
        
        # Update history (even for outliers, but with lower weight)
        if is_outlier:
            # Add with reduced weight
            history.append(0.3 * measurement['measured_distance_m'] + 0.7 * predicted)
        else:
            history.append(measurement['measured_distance_m'])
        
        # Keep history bounded
        if len(history) > 20:
            history.pop(0)
        
        return is_outlier


if __name__ == "__main__":
    # Test channel models
    config = ChannelConfig()
    channel = RangingChannel(config)
    
    print("Testing Channel Models...")
    print("=" * 50)
    
    # Test different scenarios
    test_cases = [
        (10, PropagationType.LOS, "free_space"),
        (50, PropagationType.LOS, "urban"),
        (100, PropagationType.NLOS, "urban"),
        (200, PropagationType.OBSTRUCTED, "indoor_nlos")
    ]
    
    for distance, prop_type, environment in test_cases:
        measurement = channel.generate_measurement(distance, prop_type, environment)
        
        print(f"\nDistance: {distance}m, Type: {prop_type.value}, Env: {environment}")
        print(f"  Measured: {measurement['measured_distance_m']:.2f}m")
        print(f"  Error: {measurement['measured_distance_m'] - distance:.2f}m")
        print(f"  SNR: {measurement['snr_db']:.1f}dB")
        print(f"  Std: {measurement['measurement_std_m']:.2f}m")
        print(f"  Quality: {measurement['quality_score']:.2f}")
        if prop_type != PropagationType.LOS:
            print(f"  NLOS bias: {measurement['nlos_bias_m']:.2f}m")