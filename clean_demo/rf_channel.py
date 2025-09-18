"""
Realistic RF Channel Model for FTL System
Integrates physics-based propagation from RadarSim
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class PropagationMode(Enum):
    """RF propagation modes"""
    FREE_SPACE = "free_space"
    TWO_RAY = "two_ray"
    MULTIPATH = "multipath"


@dataclass
class ChannelConfig:
    """RF channel configuration"""
    frequency_hz: float = 2.4e9  # Operating frequency
    bandwidth_hz: float = 100e6  # Signal bandwidth
    temperature_k: float = 290   # Ambient temperature

    # Hardware impairments
    iq_amplitude_imbalance_db: float = 0.5  # I/Q amplitude mismatch
    iq_phase_imbalance_deg: float = 5.0      # I/Q phase mismatch
    phase_noise_dbc_hz: float = -80          # Phase noise at 1kHz offset
    adc_bits: int = 12                       # ADC resolution

    # Environmental
    humidity_percent: float = 50.0
    atmospheric_pressure_kpa: float = 101.325

    # Multipath
    enable_multipath: bool = True
    ground_reflection_coefficient: float = -0.7  # Ground reflection


class RealisticRFChannel:
    """Physics-based RF channel model with hardware impairments"""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.c = 3e8  # Speed of light
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant

        # Wavelength
        self.wavelength = self.c / config.frequency_hz

        # ADC parameters
        self.adc_levels = 2**config.adc_bits
        self.adc_max_voltage = 1.0
        self.quantization_step = 2 * self.adc_max_voltage / self.adc_levels

    def propagate(self,
                  distance_m: float,
                  signal: np.ndarray,
                  velocity_mps: float = 0,
                  snr_db: float = 20,
                  enable_hardware_impairments: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Propagate signal through realistic channel

        Args:
            distance_m: Propagation distance in meters
            signal: Complex baseband signal
            velocity_mps: Relative velocity for Doppler
            snr_db: Target SNR in dB
            enable_hardware_impairments: Apply hardware effects

        Returns:
            Received signal and channel info dictionary
        """
        received = signal.copy().astype(complex)
        channel_info = {}

        # 1. Path loss (including R^4 for radar ranging)
        path_loss_db = self.calculate_path_loss(distance_m)
        path_loss_linear = 10**(-path_loss_db / 20)
        received *= path_loss_linear
        channel_info['path_loss_db'] = path_loss_db

        # 2. Doppler shift
        if velocity_mps != 0:
            doppler_hz = self.calculate_doppler_shift(velocity_mps)
            t = np.arange(len(signal)) / self.config.bandwidth_hz
            doppler_rotation = np.exp(2j * np.pi * doppler_hz * t)
            received *= doppler_rotation
            channel_info['doppler_hz'] = doppler_hz

        # 3. Multipath (two-ray ground reflection model)
        if self.config.enable_multipath:
            received = self.add_multipath(received, distance_m)
            channel_info['multipath_added'] = True

        # 4. Hardware impairments
        if enable_hardware_impairments:
            # I/Q imbalance
            received = self.apply_iq_imbalance(received)

            # Phase noise
            received = self.apply_phase_noise(received)

            # ADC quantization
            received = self.apply_adc_quantization(received)
            channel_info['hardware_impairments'] = True

        # 5. Atmospheric attenuation
        atm_loss_db = self.calculate_atmospheric_loss(distance_m)
        atm_loss_linear = 10**(-atm_loss_db / 20)
        received *= atm_loss_linear
        channel_info['atmospheric_loss_db'] = atm_loss_db

        # 6. Add noise based on SNR
        signal_power = np.mean(np.abs(received)**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(received)) +
                                            1j * np.random.randn(len(received)))
        received += noise
        channel_info['noise_power'] = noise_power

        # 7. Propagation delay
        propagation_delay_s = distance_m / self.c
        channel_info['propagation_delay_ns'] = propagation_delay_s * 1e9

        return received, channel_info

    def calculate_path_loss(self, distance_m: float) -> float:
        """
        Calculate path loss for ranging (R^4 dependence)

        Two-way propagation for ranging: R^4 instead of R^2
        """
        if distance_m <= 0:
            return 0

        # Two-way path loss for ranging
        # FSPL = 20*log10(4π*d/λ) for one-way
        # For two-way (radar): 40*log10(4π*d/λ)
        path_loss_db = 40 * np.log10(4 * np.pi * distance_m / self.wavelength)

        return path_loss_db

    def calculate_doppler_shift(self, velocity_mps: float) -> float:
        """
        Calculate Doppler frequency shift

        For two-way propagation (ranging): fd = 2 * v * f0 / c
        """
        doppler_hz = 2 * velocity_mps * self.config.frequency_hz / self.c
        return doppler_hz

    def add_multipath(self, signal: np.ndarray, distance_m: float) -> np.ndarray:
        """
        Add multipath using two-ray ground reflection model
        """
        # Assume transmitter and receiver at same height (h = 2m)
        height_m = 2.0

        # Direct path
        direct_distance = distance_m

        # Reflected path (image method)
        reflected_distance = np.sqrt(distance_m**2 + (2 * height_m)**2)

        # Path difference
        path_diff_m = reflected_distance - direct_distance
        delay_samples = int(path_diff_m / self.c * self.config.bandwidth_hz)

        if delay_samples > 0 and delay_samples < len(signal):
            # Create reflected component with ground reflection coefficient
            reflected = np.zeros_like(signal)
            reflected[delay_samples:] = signal[:-delay_samples] * self.config.ground_reflection_coefficient

            # Additional phase shift due to path difference
            phase_shift = 2 * np.pi * path_diff_m / self.wavelength
            reflected *= np.exp(1j * phase_shift)

            # Combine direct and reflected
            return signal + reflected

        return signal

    def apply_iq_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply I/Q imbalance impairments
        """
        # Amplitude imbalance
        amplitude_imbalance = 10**(self.config.iq_amplitude_imbalance_db / 20)

        # Phase imbalance
        phase_imbalance_rad = np.deg2rad(self.config.iq_phase_imbalance_deg)

        # Apply to I and Q separately
        i_component = np.real(signal)
        q_component = np.imag(signal) * amplitude_imbalance

        # Apply phase rotation to Q
        q_rotated = q_component * np.cos(phase_imbalance_rad) + i_component * np.sin(phase_imbalance_rad)

        return i_component + 1j * q_rotated

    def apply_phase_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply oscillator phase noise
        """
        # Simple phase noise model: random walk
        # More sophisticated: use PSD and filter

        # Phase noise power from dBc/Hz specification
        phase_noise_variance = 10**(self.config.phase_noise_dbc_hz / 10)

        # Generate phase noise as integrated white noise (random walk)
        white_noise = np.random.randn(len(signal)) * np.sqrt(phase_noise_variance)
        phase_noise = np.cumsum(white_noise) / np.sqrt(len(signal))

        # Apply phase modulation
        return signal * np.exp(1j * phase_noise)

    def apply_adc_quantization(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply ADC quantization effects
        """
        # Normalize to ADC range
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            normalized = signal / max_amplitude * self.adc_max_voltage
        else:
            normalized = signal

        # Quantize I and Q separately
        i_quantized = np.round(np.real(normalized) / self.quantization_step) * self.quantization_step
        q_quantized = np.round(np.imag(normalized) / self.quantization_step) * self.quantization_step

        # Restore original scale
        quantized = (i_quantized + 1j * q_quantized) * max_amplitude / self.adc_max_voltage

        return quantized

    def calculate_atmospheric_loss(self, distance_m: float) -> float:
        """
        Calculate atmospheric attenuation (simplified ITU-R P.676 model)
        """
        # Oxygen absorption (simplified)
        if self.config.frequency_hz < 57e9:
            oxygen_db_per_km = 0.01  # Below 57 GHz
        else:
            oxygen_db_per_km = 15.0  # Around 60 GHz peak

        # Water vapor absorption (simplified)
        water_vapor_db_per_km = 0.002 * (self.config.frequency_hz / 1e9)**2 * \
                                (self.config.humidity_percent / 100)

        # Total atmospheric loss
        total_db_per_km = oxygen_db_per_km + water_vapor_db_per_km
        atmospheric_loss_db = total_db_per_km * (distance_m / 1000)

        return atmospheric_loss_db


class RangingChannel(RealisticRFChannel):
    """Specialized channel for ranging/localization signals"""

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.ranging_cache = {}

    def process_ranging_signal(self,
                               tx_signal: np.ndarray,
                               true_distance_m: float,
                               true_velocity_mps: float = 0,
                               clock_offset_ns: float = 0,
                               freq_offset_hz: float = 0,
                               snr_db: float = 20) -> Tuple[np.ndarray, float, dict]:
        """
        Process ranging signal with all realistic effects

        Returns:
            Received signal, measured ToA in ns, channel info
        """
        # Apply frequency offset to transmitted signal
        if freq_offset_hz != 0:
            t = np.arange(len(tx_signal)) / self.config.bandwidth_hz
            tx_signal = tx_signal * np.exp(2j * np.pi * freq_offset_hz * t)

        # Propagate through channel
        rx_signal, channel_info = self.propagate(
            distance_m=true_distance_m,
            signal=tx_signal,
            velocity_mps=true_velocity_mps,
            snr_db=snr_db,
            enable_hardware_impairments=True
        )

        # True time of arrival
        true_toa_ns = (true_distance_m / self.c) * 1e9

        # Add clock offset error to measurement
        measured_toa_ns = true_toa_ns + clock_offset_ns

        # Add measurement noise based on SNR (Cramér-Rao bound)
        # σ_τ = 1 / (2π * BW * SNR^0.5)
        bandwidth_hz = self.config.bandwidth_hz
        snr_linear = 10**(snr_db / 10)
        toa_std_ns = 1e9 / (2 * np.pi * bandwidth_hz * np.sqrt(snr_linear))
        toa_noise_ns = np.random.normal(0, toa_std_ns)

        measured_toa_ns += toa_noise_ns

        # Store in channel info
        channel_info['true_toa_ns'] = true_toa_ns
        channel_info['measured_toa_ns'] = measured_toa_ns
        channel_info['toa_error_ns'] = measured_toa_ns - true_toa_ns
        channel_info['cramer_rao_bound_ns'] = toa_std_ns

        return rx_signal, measured_toa_ns, channel_info


def test_channel():
    """Test the realistic channel model"""
    print("Testing Realistic RF Channel Model")
    print("="*50)

    # Configure channel
    config = ChannelConfig(
        frequency_hz=2.4e9,
        bandwidth_hz=100e6,
        enable_multipath=True
    )

    channel = RangingChannel(config)

    # Create test signal (simple pulse)
    signal_length = 1000
    tx_signal = np.ones(signal_length) + 0j

    # Test at different distances
    distances = [10, 100, 1000]  # meters

    for dist in distances:
        rx_signal, toa_ns, info = channel.process_ranging_signal(
            tx_signal=tx_signal,
            true_distance_m=dist,
            clock_offset_ns=50,  # 50ns clock error
            freq_offset_hz=1000,  # 1kHz frequency error
            snr_db=20
        )

        print(f"\nDistance: {dist}m")
        print(f"  Path loss: {info['path_loss_db']:.1f} dB")
        print(f"  Atmospheric loss: {info['atmospheric_loss_db']:.3f} dB")
        print(f"  True ToA: {info['true_toa_ns']:.1f} ns")
        print(f"  Measured ToA: {info['measured_toa_ns']:.1f} ns")
        print(f"  ToA error: {info['toa_error_ns']:.2f} ns")
        print(f"  Cramér-Rao bound: {info['cramer_rao_bound_ns']:.2f} ns")
        print(f"  Multipath: {info.get('multipath_added', False)}")


if __name__ == "__main__":
    test_channel()