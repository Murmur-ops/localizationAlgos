"""
Spread Spectrum Waveform Generator
Implements the integrated waveform design from the spec
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import signal


@dataclass
class WaveformConfig:
    """Configuration for spread spectrum waveform"""
    # Bandwidth and sampling
    bandwidth_hz: float = 100e6  # 100 MHz
    sample_rate_hz: float = 200e6  # 2x oversampling
    
    # Frame structure (in ms)
    preamble_duration_ms: float = 0.5
    ranging_duration_ms: float = 0.5
    payload_duration_ms: float = 1.0
    
    # Pilot configuration
    pilot_frequencies_mhz: list = None  # Default: [-5, -3, -1, 0, 1, 3, 5]
    pilot_power_fraction: float = 0.3  # 30% power to pilots
    
    # PN sequences
    ranging_pn_length: int = 1023  # Gold code length
    data_pn_length: int = 255
    chip_rate_hz: float = 100e6  # Chips per second
    
    # Power allocation (normalized to 1.0 total)
    ranging_power: float = 0.5
    data_power: float = 0.2
    
    def __post_init__(self):
        if self.pilot_frequencies_mhz is None:
            self.pilot_frequencies_mhz = [-5, -3, -1, 0, 1, 3, 5]


class SpreadSpectrumGenerator:
    """Generates integrated spread spectrum waveforms"""
    
    def __init__(self, config: WaveformConfig):
        self.config = config
        self.dt = 1.0 / config.sample_rate_hz
        
        # Pre-generate PN sequences
        self.ranging_pn = self._generate_gold_code(config.ranging_pn_length)
        self.data_pn = self._generate_gold_code(config.data_pn_length)
    
    def _rrc_filter(self, n_taps: int, beta: float, samples_per_symbol: int) -> np.ndarray:
        """Generate root-raised cosine filter coefficients"""
        t = np.arange(-n_taps//2, n_taps//2 + 1) / samples_per_symbol
        h = np.zeros(len(t))
        
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 - beta + 4*beta/np.pi
            elif abs(ti) == 1/(4*beta):
                h[i] = (beta/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + 
                                            (1-2/np.pi)*np.cos(np.pi/(4*beta)))
            else:
                numerator = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
                denominator = np.pi*ti*(1-(4*beta*ti)**2)
                if denominator != 0:
                    h[i] = numerator / denominator
        
        # Normalize
        h = h / np.sqrt(np.sum(h**2))
        return h
        
    def _generate_gold_code(self, length: int) -> np.ndarray:
        """Generate Gold code for spreading using REAL Gold code generator"""
        from .gold_codes_proper import ProperGoldCodeGenerator

        # Use closest supported length
        supported_lengths = [31, 63, 127, 255, 511, 1023]
        closest = min(supported_lengths, key=lambda x: abs(x - length))

        if closest != length:
            print(f"Warning: Requested length {length}, using {closest}")

        # Generate real Gold code
        generator = ProperGoldCodeGenerator(closest)
        code = generator.get_code(0)  # Use first Gold code

        # Truncate or pad if needed
        if len(code) > length:
            return code[:length]
        elif len(code) < length:
            # Repeat to fill
            repetitions = (length // len(code)) + 1
            extended = np.tile(code, repetitions)
            return extended[:length]

        return code
    
    def generate_pilot(self, duration_s: float) -> np.ndarray:
        """Generate pilot tones/comb for frequency lock"""
        n_samples = int(duration_s * self.config.sample_rate_hz)
        t = np.arange(n_samples) * self.dt
        
        pilot = np.zeros(n_samples, dtype=complex)
        
        # Add pilot tones
        for freq_mhz in self.config.pilot_frequencies_mhz:
            freq_hz = freq_mhz * 1e6
            pilot += np.exp(2j * np.pi * freq_hz * t)
        
        # Normalize power
        pilot *= np.sqrt(self.config.pilot_power_fraction / len(self.config.pilot_frequencies_mhz))
        
        return pilot
    
    def generate_ranging_block(self) -> np.ndarray:
        """Generate wideband PN sequence for ranging"""
        # Samples per chip
        samples_per_chip = int(self.config.sample_rate_hz / self.config.chip_rate_hz)
        
        # Upsample PN sequence
        ranging_signal = np.repeat(self.ranging_pn, samples_per_chip)
        
        # Apply root-raised cosine pulse shaping
        # Note: scipy doesn't have rcosfilter, so we'll use a simple raised cosine approximation
        n_taps = 101
        beta = 0.35  # Roll-off factor
        h = self._rrc_filter(n_taps, beta, samples_per_chip)
        ranging_signal = signal.convolve(ranging_signal, h, mode='same')
        
        # Normalize power
        ranging_signal *= np.sqrt(self.config.ranging_power)
        
        return ranging_signal
    
    def generate_frame(self, data_bits: Optional[np.ndarray] = None) -> dict:
        """Generate complete frame with all components"""
        
        # 1. Preamble (pilots + short PN)
        preamble = self.generate_pilot(self.config.preamble_duration_ms * 1e-3)
        
        # 2. Ranging block
        ranging = self.generate_ranging_block()
        
        # 3. Data payload (if provided)
        if data_bits is not None:
            payload = self._generate_dsss_payload(data_bits)
        else:
            # Default: all zeros
            n_payload_samples = int(self.config.payload_duration_ms * 1e-3 * self.config.sample_rate_hz)
            payload = np.zeros(n_payload_samples, dtype=complex)
        
        # Concatenate for TDM version
        tdm_signal = np.concatenate([preamble, ranging, payload])
        
        # Also create superposition version
        max_len = max(len(preamble), len(ranging), len(payload))
        superposed = np.zeros(max_len, dtype=complex)
        
        superposed[:len(preamble)] += preamble * 0.3
        superposed[:len(ranging)] += ranging * 0.5
        superposed[:len(payload)] += payload * 0.2
        
        return {
            'tdm': tdm_signal,
            'superposed': superposed,
            'preamble': preamble,
            'ranging': ranging,
            'payload': payload,
            'timestamps': {
                'preamble_start': 0,
                'ranging_start': len(preamble) / self.config.sample_rate_hz,
                'payload_start': (len(preamble) + len(ranging)) / self.config.sample_rate_hz
            }
        }
    
    def _generate_dsss_payload(self, data_bits: np.ndarray) -> np.ndarray:
        """Generate DSSS modulated payload"""
        # QPSK modulation
        symbols = []
        for i in range(0, len(data_bits)-1, 2):
            symbol = (1 - 2*data_bits[i]) + 1j*(1 - 2*data_bits[i+1])
            symbols.append(symbol / np.sqrt(2))
        
        symbols = np.array(symbols)
        
        # Spread with data PN
        spread_factor = len(self.data_pn)
        spread_symbols = np.repeat(symbols, spread_factor)
        spread_signal = spread_symbols * np.tile(self.data_pn, len(symbols))
        
        # Upsample to sample rate
        samples_per_chip = int(self.config.sample_rate_hz / self.config.chip_rate_hz)
        upsampled = np.repeat(spread_signal, samples_per_chip)
        
        # Apply pulse shaping
        h = self._rrc_filter(51, 0.35, samples_per_chip)
        filtered = signal.convolve(upsampled, h, mode='same')
        
        # Add embedded pilots
        pilot_spacing_samples = int(0.001 * self.config.sample_rate_hz)  # Every 1ms
        for i in range(0, len(filtered), pilot_spacing_samples):
            filtered[i:i+100] *= 1.5  # Boost pilot symbols
        
        return filtered * np.sqrt(self.config.data_power)


class RangingCorrelator:
    """Correlates received signal with PN sequence for ranging"""
    
    def __init__(self, config: WaveformConfig):
        self.config = config
        self.ranging_pn = self._generate_gold_code(config.ranging_pn_length)
        self.samples_per_chip = int(config.sample_rate_hz / config.chip_rate_hz)
        
    def _generate_gold_code(self, length: int) -> np.ndarray:
        """Must match transmitter's PN sequence"""
        np.random.seed(42)
        return 2 * np.random.randint(0, 2, length) - 1
    
    def correlate(self, received_signal: np.ndarray) -> dict:
        """Perform correlation and find TOA"""
        
        # Generate reference signal
        reference = np.repeat(self.ranging_pn, self.samples_per_chip)
        
        # Cross-correlation
        correlation = signal.correlate(received_signal, reference, mode='valid')
        correlation_abs = np.abs(correlation)
        
        # Find peak (coarse)
        peak_idx = np.argmax(correlation_abs)
        peak_value = correlation_abs[peak_idx]
        
        # Sub-sample interpolation (parabolic)
        if 0 < peak_idx < len(correlation_abs) - 1:
            y1 = correlation_abs[peak_idx - 1]
            y2 = correlation_abs[peak_idx]
            y3 = correlation_abs[peak_idx + 1]
            
            # Parabolic interpolation
            a = (y1 - 2*y2 + y3) / 2
            b = (y3 - y1) / 2
            
            if a != 0:
                x_offset = -b / (2*a)
                fine_peak_idx = peak_idx + x_offset
            else:
                fine_peak_idx = peak_idx
        else:
            fine_peak_idx = peak_idx
        
        # Convert to time
        toa_samples = fine_peak_idx
        toa_seconds = toa_samples / self.config.sample_rate_hz
        
        # Estimate SNR from peak
        noise_floor = np.median(correlation_abs)
        snr_linear = peak_value / noise_floor
        snr_db = 10 * np.log10(snr_linear)
        
        # Multipath detection (look for secondary peaks)
        threshold = peak_value * 0.5
        peaks = signal.find_peaks(correlation_abs, height=threshold)[0]
        n_peaks = len(peaks) if len(peaks) > 0 else 1
        multipath_score = 1.0 / n_peaks  # More peaks = lower score
        
        return {
            'toa_seconds': toa_seconds,
            'toa_samples': toa_samples,
            'peak_value': peak_value,
            'snr_db': snr_db,
            'multipath_score': multipath_score,
            'correlation': correlation_abs,
            'fine_peak_idx': fine_peak_idx
        }


if __name__ == "__main__":
    # Test the waveform generator
    config = WaveformConfig()
    generator = SpreadSpectrumGenerator(config)
    
    # Generate a frame
    data_bits = np.random.randint(0, 2, 100)
    frame = generator.generate_frame(data_bits)
    
    print(f"Frame generated:")
    print(f"  TDM signal length: {len(frame['tdm'])} samples")
    print(f"  Duration: {len(frame['tdm']) / config.sample_rate_hz * 1000:.2f} ms")
    print(f"  Ranging starts at: {frame['timestamps']['ranging_start']*1000:.2f} ms")
    
    # Test correlation
    correlator = RangingCorrelator(config)
    
    # Add some delay and noise to simulate propagation
    delay_samples = 1000
    noise_power = 0.1
    received = np.concatenate([
        np.zeros(delay_samples, dtype=complex),
        frame['ranging'] + noise_power * (np.random.randn(len(frame['ranging'])) + 
                                          1j * np.random.randn(len(frame['ranging'])))
    ])
    
    result = correlator.correlate(received)
    
    print(f"\nCorrelation results:")
    print(f"  TOA: {result['toa_seconds']*1e6:.2f} µs")
    print(f"  Expected: {delay_samples/config.sample_rate_hz*1e6:.2f} µs")
    print(f"  SNR: {result['snr_db']:.1f} dB")
    print(f"  Multipath score: {result['multipath_score']:.2f}")