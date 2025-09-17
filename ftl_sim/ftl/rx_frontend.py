"""
Receiver Front-End Module
Matched filtering, ToA detection, CFO estimation, and CRLB calculations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal


def matched_filter(
    received: np.ndarray,
    template: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply matched filter to received signal

    Args:
        received: Received signal
        template: Template/reference signal
        normalize: Whether to normalize template energy

    Returns:
        Correlation output
    """
    # Normalize template for unit energy
    if normalize:
        template_norm = template / np.linalg.norm(template)
    else:
        template_norm = template

    # Matched filter is correlation with time-reversed conjugate template
    # For complex signals: h[n] = template*[-n]
    matched = scipy_signal.correlate(received, template_norm, mode='same')

    return matched


def detect_toa(
    correlation: np.ndarray,
    sample_rate: float,
    mode: str = 'peak',
    threshold_factor: float = 3.0,
    enable_subsample: bool = True
) -> Dict:
    """
    Detect Time of Arrival from correlation function

    Args:
        correlation: Correlation function output
        sample_rate: Sample rate in Hz
        mode: Detection mode ('peak' or 'leading_edge')
        threshold_factor: Threshold factor above noise floor
        enable_subsample: Enable sub-sample refinement

    Returns:
        Dictionary with ToA estimates and metrics
    """
    correlation_mag = np.abs(correlation)

    # Estimate noise floor (use median for robustness)
    noise_floor = np.median(correlation_mag)
    noise_std = np.median(np.abs(correlation_mag - noise_floor)) / 0.6745  # MAD estimate

    # Set detection threshold
    threshold = noise_floor + threshold_factor * noise_std

    if mode == 'peak':
        # Find strongest peak
        peak_idx = np.argmax(correlation_mag)
        peak_value = correlation_mag[peak_idx]

    elif mode == 'leading_edge':
        # Find first crossing above threshold (for NLOS mitigation)
        above_threshold = np.where(correlation_mag > threshold)[0]

        if len(above_threshold) > 0:
            peak_idx = above_threshold[0]
            peak_value = correlation_mag[peak_idx]
        else:
            # Fallback to peak detection
            peak_idx = np.argmax(correlation_mag)
            peak_value = correlation_mag[peak_idx]

    else:
        raise ValueError(f"Unknown detection mode: {mode}")

    # Sub-sample refinement using parabolic interpolation
    refined_idx = peak_idx
    if enable_subsample and 1 < peak_idx < len(correlation) - 2:
        # Use 3 points around peak for parabolic fit
        y1 = correlation_mag[peak_idx - 1]
        y2 = correlation_mag[peak_idx]
        y3 = correlation_mag[peak_idx + 1]

        # Parabolic interpolation
        if y1 < y2 > y3:  # Valid peak
            denominator = y1 - 2*y2 + y3
            if abs(denominator) > 1e-10:  # Avoid division by zero
                delta = 0.5 * (y1 - y3) / denominator
                if abs(delta) < 1:  # Sanity check
                    refined_idx = peak_idx + delta

    # Convert to time
    toa_seconds = refined_idx / sample_rate

    return {
        'toa_samples': peak_idx,
        'toa_refined_samples': refined_idx,
        'toa_seconds': toa_seconds,
        'peak_value': peak_value,
        'noise_floor': noise_floor,
        'noise_std': noise_std,
        'snr_db': 10 * np.log10(peak_value**2 / (noise_floor**2 + 1e-10))
    }


def estimate_cfo(
    blocks: List[np.ndarray],
    block_separation_s: float,
    method: str = 'ml'
) -> float:
    """
    Estimate CFO from repeated signal blocks

    Uses phase difference between repeated blocks to estimate frequency offset.
    Based on maximum likelihood estimation.

    Args:
        blocks: List of repeated signal blocks
        block_separation_s: Time separation between blocks
        method: Estimation method ('ml' for maximum likelihood)

    Returns:
        Estimated CFO in Hz
    """
    if len(blocks) < 2:
        return 0.0

    # Accumulate phase differences
    phase_diffs = []

    for i in range(len(blocks) - 1):
        # Correlate consecutive blocks
        block1 = blocks[i]
        block2 = blocks[i + 1]

        # Ensure same length
        min_len = min(len(block1), len(block2))
        block1 = block1[:min_len]
        block2 = block2[:min_len]

        # Compute correlation (ML estimator)
        correlation = np.sum(np.conj(block1) * block2)

        # Extract phase difference
        phase_diff = np.angle(correlation)
        phase_diffs.append(phase_diff)

    # Average phase differences
    avg_phase_diff = np.mean(phase_diffs)

    # Convert to frequency
    # Phase accumulated over block_separation_s: Δφ = 2π * CFO * T
    cfo_hz = avg_phase_diff / (2 * np.pi * block_separation_s)

    return cfo_hz


def toa_crlb(
    snr_linear: float,
    bandwidth_hz: float
) -> float:
    """
    Calculate Cramér-Rao Lower Bound for ToA estimation

    For a deterministic signal in AWGN:
    var(τ) ≥ 1 / (8π² * β² * SNR)

    where β² is the mean-square bandwidth

    Args:
        snr_linear: Linear SNR (not dB)
        bandwidth_hz: Signal bandwidth in Hz

    Returns:
        CRLB variance in seconds²
    """
    # For rectangular spectrum, RMS bandwidth ≈ BW / sqrt(12)
    # But for practical signals, use effective bandwidth
    beta_rms = bandwidth_hz / np.sqrt(3)

    # CRLB formula
    variance = 1.0 / (8 * np.pi**2 * beta_rms**2 * snr_linear)

    return variance


def extract_correlation_features(
    correlation: np.ndarray,
    peak_idx: int,
    window: int = 100
) -> Dict:
    """
    Extract features from correlation function for LOS/NLOS classification

    Args:
        correlation: Correlation function
        peak_idx: Index of main peak
        window: Window size around peak for analysis

    Returns:
        Dictionary of correlation shape features
    """
    correlation_mag = np.abs(correlation)

    # Extract window around peak
    start_idx = max(0, peak_idx - window)
    end_idx = min(len(correlation), peak_idx + window)
    window_corr = correlation_mag[start_idx:end_idx]

    # Peak value
    peak_value = correlation_mag[peak_idx]

    # Find sidelobes (other peaks)
    # Use scipy to find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(correlation_mag, height=peak_value * 0.3)

    # Remove main peak from list
    other_peaks = peaks[peaks != peak_idx]

    # Peak to sidelobe ratio
    if len(other_peaks) > 0:
        max_sidelobe = np.max(correlation_mag[other_peaks])
        peak_to_sidelobe = peak_value / max_sidelobe
    else:
        peak_to_sidelobe = float('inf')

    # RMS width (spread of correlation peak)
    # Normalize window
    if np.sum(window_corr) > 0:
        window_norm = window_corr / np.sum(window_corr)
        indices = np.arange(len(window_norm))
        mean_idx = np.sum(indices * window_norm)
        rms_width = np.sqrt(np.sum((indices - mean_idx)**2 * window_norm))
    else:
        rms_width = 0

    # Excess delay (how far is energy spread beyond main peak)
    energy_total = np.sum(correlation_mag**2)
    energy_main = correlation_mag[peak_idx]**2
    energy_after = np.sum(correlation_mag[peak_idx+1:peak_idx+50]**2)

    multipath_ratio = energy_after / (energy_main + 1e-10)
    excess_delay = 0

    # Find delay where 90% of energy is captured
    cumsum_energy = np.cumsum(correlation_mag[peak_idx:]**2)
    if len(cumsum_energy) > 0 and cumsum_energy[-1] > 0:
        idx_90 = np.where(cumsum_energy > 0.9 * cumsum_energy[-1])[0]
        if len(idx_90) > 0:
            excess_delay = idx_90[0]

    return {
        'peak_to_sidelobe_ratio': peak_to_sidelobe if peak_to_sidelobe != float('inf') else 100.0,
        'rms_width': rms_width,
        'excess_delay': excess_delay,
        'multipath_ratio': multipath_ratio
    }


def classify_propagation(
    correlation: np.ndarray,
    peak_threshold: float = 5.0
) -> Dict:
    """
    Classify propagation as LOS or NLOS based on correlation shape

    Args:
        correlation: Correlation function
        peak_threshold: Threshold for peak detection

    Returns:
        Classification result with confidence
    """
    correlation_mag = np.abs(correlation)

    # Find main peak
    peak_idx = np.argmax(correlation_mag)

    # Extract features
    features = extract_correlation_features(correlation, peak_idx)

    # Classification logic based on features
    score_los = 0
    score_nlos = 0

    # Sharp peak indicates LOS
    if features['rms_width'] < 3:  # Stricter threshold for LOS
        score_los += 2
    elif features['rms_width'] > 8:
        score_nlos += 2

    # High peak-to-sidelobe ratio indicates LOS
    if features['peak_to_sidelobe_ratio'] > 3:
        score_los += 1
    else:
        score_nlos += 1

    # Low multipath ratio indicates LOS
    if features['multipath_ratio'] < 0.2:  # Stricter threshold
        score_los += 2
    else:
        score_nlos += 2

    # Small excess delay indicates LOS
    if features['excess_delay'] < 5:  # Much stricter for LOS
        score_los += 1
    else:
        score_nlos += 1

    # Determine classification
    total_score = score_los + score_nlos
    if total_score > 0:
        confidence_los = score_los / total_score
        confidence_nlos = score_nlos / total_score
    else:
        confidence_los = 0.5
        confidence_nlos = 0.5

    if score_los > score_nlos:
        prop_type = 'LOS'
        confidence = confidence_los
    else:
        prop_type = 'NLOS'
        confidence = confidence_nlos

    return {
        'type': prop_type,
        'confidence': confidence,
        'score_los': score_los,
        'score_nlos': score_nlos,
        'features': features
    }


if __name__ == "__main__":
    # Test receiver front-end functions
    print("Testing Receiver Front-End...")
    print("=" * 50)

    # Test matched filter
    print("\n1. Matched Filter Test:")
    template = np.array([1, -1, 1, 1, -1], dtype=complex)
    received = np.zeros(100, dtype=complex)
    received[30:35] = template * 2  # Signal at index 30 with amplitude 2

    # Add noise
    noise = 0.1 * (np.random.randn(100) + 1j * np.random.randn(100))
    received += noise

    correlation = matched_filter(received, template)
    peak_idx = np.argmax(np.abs(correlation))
    print(f"  Peak at index: {peak_idx}")
    print(f"  Peak value: {np.abs(correlation[peak_idx]):.2f}")

    # Test ToA detection
    print("\n2. ToA Detection Test:")
    result = detect_toa(correlation, sample_rate=1e9)
    print(f"  ToA: {result['toa_seconds']*1e9:.2f} ns")
    print(f"  SNR: {result['snr_db']:.1f} dB")

    # Test CRLB
    print("\n3. CRLB Test:")
    snr_linear = 100  # 20 dB
    bandwidth = 500e6  # 500 MHz

    variance = toa_crlb(snr_linear, bandwidth)
    std_seconds = np.sqrt(variance)
    std_meters = 3e8 * std_seconds

    print(f"  Bandwidth: {bandwidth/1e6:.0f} MHz")
    print(f"  SNR: {10*np.log10(snr_linear):.1f} dB")
    print(f"  CRLB std: {std_meters*100:.2f} cm")

    print("\nTest completed successfully!")