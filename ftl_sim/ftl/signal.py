"""
Signal Generation Module
IEEE 802.15.4z HRP-UWB and Zadoff-Chu CAZAC sequences
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    # Signal type
    signal_type: str = "HRP_UWB"  # HRP_UWB or ZADOFF_CHU

    # HRP-UWB parameters (IEEE 802.15.4z)
    prf_mhz: float = 124.8  # Pulse repetition frequency (124.8 or 249.6 MHz)
    bandwidth_mhz: float = 499.2  # Channel bandwidth
    preamble_length: int = 64  # Number of symbols in preamble
    sfd_length: int = 8  # Start frame delimiter length

    # Zadoff-Chu parameters
    zc_length: int = 1023  # Sequence length (should be prime)
    zc_root: int = 7  # Root index (coprime with length)

    # Common parameters
    sample_rate_hz: float = 2e9  # 2 GS/s
    n_repeats: int = 8  # Number of repeats for CFO estimation
    pulse_shape: str = "RRC"  # RRC or GAUSSIAN

    # Modulation
    modulation: str = "BPSK"  # BPSK or 4M (4-ary modulation)


def generate_ternary_sequence(length: int, seed: int = 42) -> np.ndarray:
    """
    Generate ternary preamble sequence for HRP-UWB
    Values are {-1, 0, +1} with specific autocorrelation properties
    """
    np.random.seed(seed)

    # Generate with 1/3 probability for each value
    # Real HRP uses specific sequences, but we'll use random for simplicity
    sequence = np.random.choice([-1, 0, 1], size=length, p=[0.3, 0.4, 0.3])

    # Ensure good autocorrelation by adjusting zero density
    # More zeros = better autocorrelation sidelobes
    zero_indices = np.random.choice(length, size=int(0.4 * length), replace=False)
    sequence[zero_indices] = 0

    return sequence


def gen_hrp_burst(
    cfg: SignalConfig,
    n_repeats: Optional[int] = None
) -> np.ndarray:
    """
    Generate HRP-UWB sync burst per IEEE 802.15.4z HPRF mode

    The HRP (High Rate Pulse) preamble consists of:
    1. SYNC field with repeated ternary sequences
    2. SFD (Start Frame Delimiter) for frame sync
    3. Optional STS (Scrambled Timestamp Sequence) for security

    Args:
        cfg: Signal configuration
        n_repeats: Number of repetitions (overrides config)

    Returns:
        Complex baseband signal
    """
    if n_repeats is None:
        n_repeats = cfg.n_repeats

    # Time parameters
    dt = 1.0 / cfg.sample_rate_hz
    prf_period = 1.0 / (cfg.prf_mhz * 1e6)
    samples_per_symbol = int(prf_period / dt)

    # Generate ternary preamble sequence
    preamble = generate_ternary_sequence(cfg.preamble_length)

    # Generate SFD (typically a known pattern)
    sfd = np.array([1, -1, 1, 1, -1, -1, 1, -1])[:cfg.sfd_length]

    # Combine preamble and SFD
    full_sequence = np.concatenate([preamble, sfd])

    # Generate pulse shape
    if cfg.pulse_shape == "GAUSSIAN":
        # Gaussian pulse for UWB
        pulse_duration = 2e-9  # 2 ns pulse width
        pulse_samples = int(pulse_duration / dt)
        t_pulse = np.arange(pulse_samples) * dt - pulse_duration / 2
        sigma = pulse_duration / 6  # 3-sigma within duration
        pulse = np.exp(-(t_pulse**2) / (2 * sigma**2))
        pulse = pulse / np.linalg.norm(pulse)
    else:  # RRC
        # Root-raised cosine pulse
        pulse = generate_rrc_pulse(
            span=4,
            sps=samples_per_symbol,
            beta=0.5
        )

    # Generate modulated signal
    signal = np.zeros(len(full_sequence) * samples_per_symbol, dtype=complex)

    for i, symbol in enumerate(full_sequence):
        if symbol != 0:  # Only transmit for non-zero symbols
            start_idx = i * samples_per_symbol
            end_idx = start_idx + len(pulse)
            if end_idx <= len(signal):
                signal[start_idx:end_idx] += symbol * pulse

    # Repeat for CFO estimation
    repeated_signal = np.tile(signal, n_repeats)

    # Apply bandpass filtering to match channel bandwidth
    filtered_signal = bandpass_filter(
        repeated_signal,
        cfg.sample_rate_hz,
        cfg.bandwidth_mhz * 1e6
    )

    return filtered_signal


def gen_zc_burst(
    cfg: SignalConfig,
    n_repeats: Optional[int] = None,
    nzc: Optional[int] = None,
    u: Optional[int] = None
) -> np.ndarray:
    """
    Generate repeated Zadoff-Chu CAZAC burst

    ZC sequences have ideal autocorrelation (zero sidelobes) and
    constant amplitude, making them excellent for timing and CFO.

    Args:
        cfg: Signal configuration
        n_repeats: Number of repetitions
        nzc: Sequence length (should be prime)
        u: Root index (should be coprime with nzc)

    Returns:
        Complex baseband signal
    """
    if n_repeats is None:
        n_repeats = cfg.n_repeats
    if nzc is None:
        nzc = cfg.zc_length
    if u is None:
        u = cfg.zc_root

    # Check that u and nzc are coprime
    if np.gcd(u, nzc) != 1:
        raise ValueError(f"Root {u} must be coprime with length {nzc}")

    # Generate Zadoff-Chu sequence
    n = np.arange(nzc)
    if nzc % 2 == 0:
        # Even length
        zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / nzc)
    else:
        # Odd length
        zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / nzc)

    # Upsample to target sample rate
    samples_per_chip = int(cfg.sample_rate_hz / (cfg.prf_mhz * 1e6))
    upsampled = np.zeros(nzc * samples_per_chip, dtype=complex)
    upsampled[::samples_per_chip] = zc_sequence

    # Apply pulse shaping
    if cfg.pulse_shape == "RRC":
        pulse = generate_rrc_pulse(span=6, sps=samples_per_chip, beta=0.35)
        signal = np.convolve(upsampled, pulse, mode='same')
    else:
        # Simple rectangular pulse
        signal = upsampled

    # Add cyclic prefix for timing acquisition (10% of sequence)
    cp_length = int(0.1 * len(signal))
    signal_with_cp = np.concatenate([signal[-cp_length:], signal])

    # Repeat for CFO estimation
    repeated_signal = np.tile(signal_with_cp, n_repeats)

    # Normalize power
    repeated_signal = repeated_signal / np.sqrt(np.mean(np.abs(repeated_signal)**2))

    return repeated_signal


def generate_rrc_pulse(
    span: int = 6,
    sps: int = 8,
    beta: float = 0.35
) -> np.ndarray:
    """
    Generate root-raised cosine pulse for pulse shaping

    Args:
        span: Filter span in symbols
        sps: Samples per symbol
        beta: Roll-off factor (0 to 1)

    Returns:
        RRC pulse shape
    """
    N = span * sps
    t = np.arange(-N//2, N//2 + 1) / sps

    # Handle special cases
    pulse = np.zeros_like(t)

    # t = 0 case
    idx_zero = np.where(t == 0)[0]
    if len(idx_zero) > 0:
        pulse[idx_zero] = 1 + beta * (4/np.pi - 1)

    # t = ±1/(4β) case
    idx_special = np.where(np.abs(t) == 1/(4*beta))[0]
    if len(idx_special) > 0:
        pulse[idx_special] = (beta/np.sqrt(2)) * (
            (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
            (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
        )

    # General case
    idx_general = np.where((t != 0) & (np.abs(t) != 1/(4*beta)))[0]
    if len(idx_general) > 0:
        t_g = t[idx_general]
        numerator = np.sin(np.pi * t_g * (1 - beta)) + \
                   4 * beta * t_g * np.cos(np.pi * t_g * (1 + beta))
        denominator = np.pi * t_g * (1 - (4 * beta * t_g)**2)
        pulse[idx_general] = numerator / denominator

    # Normalize
    pulse = pulse / np.linalg.norm(pulse)

    return pulse


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    bandwidth: float,
    center_freq: float = 0
) -> np.ndarray:
    """
    Apply ideal bandpass filter (for complex baseband, this is lowpass)

    Args:
        signal: Input signal
        fs: Sample rate
        bandwidth: Filter bandwidth
        center_freq: Center frequency (0 for baseband)

    Returns:
        Filtered signal
    """
    # For complex baseband, just do lowpass filtering
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/fs)

    # FFT of signal
    signal_fft = np.fft.fft(signal)

    # Ideal rectangular filter
    filter_mask = np.abs(freq) <= bandwidth / 2

    # Apply filter
    filtered_fft = signal_fft * filter_mask

    # IFFT back to time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return filtered_signal


def add_pilot_tones(
    signal: np.ndarray,
    pilot_freqs_mhz: List[float],
    sample_rate: float,
    pilot_power_db: float = -10
) -> np.ndarray:
    """
    Add pilot tones for enhanced CFO tracking

    Args:
        signal: Input signal
        pilot_freqs_mhz: Pilot frequencies in MHz
        sample_rate: Sample rate in Hz
        pilot_power_db: Pilot power relative to signal

    Returns:
        Signal with pilots added
    """
    n = len(signal)
    t = np.arange(n) / sample_rate

    # Calculate pilot amplitude
    signal_power = np.mean(np.abs(signal)**2)

    # Handle case where signal is zero
    if signal_power == 0:
        # Use absolute power level (assume 0 dBm reference)
        pilot_power_linear = 10**(pilot_power_db / 10)
    else:
        pilot_power_linear = 10**(pilot_power_db / 10) * signal_power

    pilot_amplitude = np.sqrt(pilot_power_linear)

    # Add each pilot tone
    output = signal.copy()
    for freq_mhz in pilot_freqs_mhz:
        freq_hz = freq_mhz * 1e6
        pilot = pilot_amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)
        output += pilot

    return output


if __name__ == "__main__":
    # Test signal generation
    print("Testing Signal Generation...")
    print("=" * 50)

    cfg = SignalConfig()

    # Test HRP-UWB generation
    print("\nGenerating HRP-UWB burst...")
    hrp_signal = gen_hrp_burst(cfg, n_repeats=4)
    print(f"  Signal length: {len(hrp_signal)} samples")
    print(f"  Duration: {len(hrp_signal) / cfg.sample_rate_hz * 1e6:.1f} µs")
    print(f"  RMS power: {np.sqrt(np.mean(np.abs(hrp_signal)**2)):.3f}")

    # Check bandwidth
    fft = np.fft.fft(hrp_signal)
    freq = np.fft.fftfreq(len(hrp_signal), 1/cfg.sample_rate_hz)
    power_spectrum = np.abs(fft)**2
    # Find -3dB bandwidth
    peak_power = np.max(power_spectrum)
    bw_indices = np.where(power_spectrum > peak_power / 2)[0]
    if len(bw_indices) > 0:
        bandwidth = np.max(np.abs(freq[bw_indices])) * 2
        print(f"  Measured bandwidth: {bandwidth / 1e6:.1f} MHz")

    # Test Zadoff-Chu generation
    print("\nGenerating Zadoff-Chu burst...")
    zc_signal = gen_zc_burst(cfg, n_repeats=4)
    print(f"  Signal length: {len(zc_signal)} samples")
    print(f"  Duration: {len(zc_signal) / cfg.sample_rate_hz * 1e6:.1f} µs")
    print(f"  RMS power: {np.sqrt(np.mean(np.abs(zc_signal)**2)):.3f}")

    # Check constant amplitude property
    amplitude_variation = np.std(np.abs(zc_signal[100:-100]))  # Exclude edges
    print(f"  Amplitude variation: {amplitude_variation:.4f} (should be ~0 for CAZAC)")

    # Test autocorrelation of ZC sequence
    print("\nTesting Zadoff-Chu autocorrelation...")
    zc_short = gen_zc_burst(cfg, n_repeats=1)[:cfg.zc_length * 10]
    autocorr = np.correlate(zc_short, zc_short, mode='same')
    autocorr_normalized = np.abs(autocorr) / np.max(np.abs(autocorr))

    # Find sidelobe level
    center = len(autocorr) // 2
    peak = autocorr_normalized[center]
    sidelobes = np.concatenate([
        autocorr_normalized[:center-10],
        autocorr_normalized[center+10:]
    ])
    max_sidelobe = np.max(sidelobes) if len(sidelobes) > 0 else 0
    print(f"  Peak correlation: {peak:.3f}")
    print(f"  Max sidelobe: {max_sidelobe:.4f} (should be ~0 for ideal CAZAC)")