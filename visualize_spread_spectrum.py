#!/usr/bin/env python3
"""
Visualize the Spread Spectrum Signal
Shows time domain, frequency domain, and correlation properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rf.spread_spectrum import SpreadSpectrumGenerator, RangingCorrelator, WaveformConfig

def visualize_spread_spectrum():
    """Generate and visualize the spread spectrum waveform"""
    
    # Configure waveform
    config = WaveformConfig()
    generator = SpreadSpectrumGenerator(config)
    correlator = RangingCorrelator(config)
    
    # Generate frame with random data
    np.random.seed(42)
    data_bits = np.random.randint(0, 2, 100)
    frame = generator.generate_frame(data_bits)
    
    # Extract components
    preamble = frame['preamble']
    ranging = frame['ranging']
    tdm_signal = frame['tdm']
    
    # Time parameters
    dt = 1.0 / config.sample_rate_hz
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ============= TOP ROW: WAVEFORM STRUCTURE =============
    
    # 1. Frame Structure Overview
    ax1 = plt.subplot(4, 3, 1)
    t_frame = np.arange(len(tdm_signal)) * dt * 1e3  # Convert to ms
    
    # Show different components in different colors
    ax1.plot(t_frame[:len(preamble)], np.abs(preamble), 'b-', label='Preamble (Pilots)', alpha=0.7)
    t_ranging_start = len(preamble) * dt * 1e3
    t_ranging = np.arange(len(ranging)) * dt * 1e3 + t_ranging_start
    ax1.plot(t_ranging, np.abs(ranging), 'r-', label='Ranging (PN)', alpha=0.7)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Complete Frame Structure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2])
    
    # 2. Gold Code Sequence (Chips)
    ax2 = plt.subplot(4, 3, 2)
    gold_code = generator.ranging_pn[:100]  # First 100 chips
    ax2.stem(range(len(gold_code)), gold_code, basefmt=' ')
    ax2.set_xlabel('Chip Index')
    ax2.set_ylabel('Chip Value')
    ax2.set_title(f'Gold Code Sequence (First 100 of {config.ranging_pn_length} chips)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.5, 1.5])
    
    # 3. Pulse Shaping
    ax3 = plt.subplot(4, 3, 3)
    # Show raw chips vs pulse-shaped signal
    samples_per_chip = int(config.sample_rate_hz / config.chip_rate_hz)
    raw_chips = np.repeat(gold_code[:20], samples_per_chip)
    shaped_signal = ranging[:len(raw_chips)]
    
    t_chips = np.arange(len(raw_chips)) * dt * 1e6  # Convert to microseconds
    ax3.plot(t_chips, raw_chips, 'b-', label='Raw Chips', alpha=0.5, linewidth=2)
    ax3.plot(t_chips, np.real(shaped_signal), 'r-', label='Pulse Shaped', alpha=0.8)
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Root Raised Cosine Pulse Shaping (β=0.35)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ============= SECOND ROW: FREQUENCY DOMAIN =============
    
    # 4. Pilot Tones Spectrum
    ax4 = plt.subplot(4, 3, 4)
    # FFT of preamble (pilots)
    N = len(preamble)
    frequencies = np.fft.fftfreq(N, dt) / 1e6  # Convert to MHz
    preamble_fft = np.fft.fft(preamble)
    preamble_spectrum = 20 * np.log10(np.abs(preamble_fft) + 1e-10)
    
    # Only show positive frequencies
    pos_freq = frequencies[:N//2]
    pos_spectrum = preamble_spectrum[:N//2]
    
    ax4.plot(pos_freq, pos_spectrum, 'b-', alpha=0.7)
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Power (dB)')
    ax4.set_title('Pilot Tones Spectrum')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-10, 10])
    
    # Mark pilot frequencies
    for freq_mhz in config.pilot_frequencies_mhz:
        ax4.axvline(x=freq_mhz, color='r', linestyle='--', alpha=0.5)
    
    # 5. Ranging Signal Spectrum
    ax5 = plt.subplot(4, 3, 5)
    # FFT of ranging signal
    N_ranging = len(ranging)
    freq_ranging = np.fft.fftfreq(N_ranging, dt) / 1e6  # MHz
    ranging_fft = np.fft.fft(ranging)
    ranging_spectrum = 20 * np.log10(np.abs(ranging_fft) + 1e-10)
    
    # Only show positive frequencies
    pos_freq_r = freq_ranging[:N_ranging//2]
    pos_spectrum_r = ranging_spectrum[:N_ranging//2]
    
    ax5.plot(pos_freq_r, pos_spectrum_r, 'r-', alpha=0.7)
    ax5.set_xlabel('Frequency (MHz)')
    ax5.set_ylabel('Power (dB)')
    ax5.set_title(f'Spread Spectrum (100 MHz Bandwidth)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([-60, 60])
    
    # Mark -3dB bandwidth
    max_power = np.max(pos_spectrum_r)
    bw_3db = max_power - 3
    ax5.axhline(y=bw_3db, color='g', linestyle='--', alpha=0.5, label='-3dB BW')
    ax5.legend()
    
    # 6. Power Spectral Density
    ax6 = plt.subplot(4, 3, 6)
    # Welch's method for PSD
    f_psd, psd = signal.welch(ranging, config.sample_rate_hz, nperseg=1024, return_onesided=True)
    ax6.semilogy(f_psd/1e6, psd)
    ax6.set_xlabel('Frequency (MHz)')
    ax6.set_ylabel('PSD (V²/Hz)')
    ax6.set_title('Power Spectral Density')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 100])
    
    # ============= THIRD ROW: CORRELATION PROPERTIES =============
    
    # 7. Autocorrelation of Gold Code
    ax7 = plt.subplot(4, 3, 7)
    # Autocorrelation
    gold_full = generator.ranging_pn
    autocorr = np.correlate(gold_full, gold_full, mode='same')
    lags = np.arange(-len(autocorr)//2, len(autocorr)//2)
    
    ax7.plot(lags, autocorr, 'b-', alpha=0.7)
    ax7.set_xlabel('Lag (chips)')
    ax7.set_ylabel('Correlation')
    ax7.set_title('Gold Code Autocorrelation')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([-100, 100])
    
    # Highlight main peak
    ax7.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # 8. Ranging Correlation Peak
    ax8 = plt.subplot(4, 3, 8)
    # Simulate received signal with delay
    delay_samples = 1000
    noise_power = 0.1
    received = np.concatenate([
        np.zeros(delay_samples, dtype=complex),
        ranging + noise_power * (np.random.randn(len(ranging)) + 1j * np.random.randn(len(ranging)))
    ])
    
    # Correlate
    result = correlator.correlate(received)
    correlation = result['correlation']
    
    # Plot around peak
    peak_idx = int(result['fine_peak_idx'])
    window = 200
    start = max(0, peak_idx - window)
    end = min(len(correlation), peak_idx + window)
    
    x_samples = np.arange(start, end)
    x_meters = (x_samples - peak_idx) * (3e8 / config.sample_rate_hz)  # Convert to meters
    
    ax8.plot(x_meters, correlation[start:end], 'b-', alpha=0.7)
    ax8.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Peak')
    ax8.set_xlabel('Distance Error (meters)')
    ax8.set_ylabel('Correlation')
    ax8.set_title(f'Ranging Correlation (SNR={result["snr_db"]:.1f}dB)')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # 9. Sub-sample Interpolation
    ax9 = plt.subplot(4, 3, 9)
    # Zoom in on peak for sub-sample detail
    fine_window = 5
    peak_start = max(0, peak_idx - fine_window)
    peak_end = min(len(correlation), peak_idx + fine_window)
    
    x_coarse = np.arange(peak_start, peak_end)
    y_coarse = correlation[peak_start:peak_end]
    
    ax9.plot(x_coarse, y_coarse, 'bo-', markersize=8, label='Samples')
    
    # Interpolated curve (parabolic fit)
    if peak_idx > 0 and peak_idx < len(correlation) - 1:
        y1, y2, y3 = correlation[peak_idx-1:peak_idx+2]
        a = (y1 - 2*y2 + y3) / 2
        b = (y3 - y1) / 2
        c = y2
        x_offset = np.linspace(-1, 1, 100)
        y_fine = a * x_offset**2 + b * x_offset + c
        x_fine = peak_idx + x_offset
        ax9.plot(x_fine, y_fine, 'r-', linewidth=2, label='Parabolic Fit')
    
    ax9.axvline(x=result['fine_peak_idx'], color='g', linestyle='--', alpha=0.5, label='Sub-sample Peak')
    ax9.set_xlabel('Sample Index')
    ax9.set_ylabel('Correlation')
    ax9.set_title('Sub-sample Interpolation Detail')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # ============= BOTTOM ROW: KEY PARAMETERS =============
    
    # 10. Eye Diagram (simulated)
    ax10 = plt.subplot(4, 3, 10)
    # Create eye diagram for ranging signal
    samples_per_symbol = samples_per_chip
    num_symbols = 50
    eye_data = np.real(ranging[:num_symbols * samples_per_symbol])
    
    for i in range(num_symbols - 2):
        segment = eye_data[i*samples_per_symbol:(i+2)*samples_per_symbol]
        t_eye = np.arange(len(segment)) / samples_per_symbol
        ax10.plot(t_eye, segment, 'b-', alpha=0.1)
    
    ax10.set_xlabel('Symbol Period')
    ax10.set_ylabel('Amplitude')
    ax10.set_title('Eye Diagram')
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim([0, 2])
    
    # 11. Constellation (QPSK for data)
    ax11 = plt.subplot(4, 3, 11)
    # Generate some QPSK symbols
    qpsk_symbols = []
    for i in range(0, len(data_bits)-1, 2):
        symbol = (1 - 2*data_bits[i]) + 1j*(1 - 2*data_bits[i+1])
        qpsk_symbols.append(symbol / np.sqrt(2))
    
    qpsk_symbols = np.array(qpsk_symbols)
    ax11.scatter(np.real(qpsk_symbols), np.imag(qpsk_symbols), alpha=0.6, s=50)
    ax11.set_xlabel('In-phase')
    ax11.set_ylabel('Quadrature')
    ax11.set_title('QPSK Constellation (Data)')
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim([-1.5, 1.5])
    ax11.set_ylim([-1.5, 1.5])
    ax11.set_aspect('equal')
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax11.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # 12. Key Parameters Summary
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    params_text = f"""
    WAVEFORM PARAMETERS
    ━━━━━━━━━━━━━━━━━━━
    
    Bandwidth: {config.bandwidth_hz/1e6:.0f} MHz
    Sample Rate: {config.sample_rate_hz/1e6:.0f} Msps
    Chip Rate: {config.chip_rate_hz/1e6:.0f} Mcps
    
    Gold Code Length: {config.ranging_pn_length} chips
    Spreading Factor: {config.ranging_pn_length}
    Processing Gain: {10*np.log10(config.ranging_pn_length):.1f} dB
    
    Pilot Frequencies: {config.pilot_frequencies_mhz} MHz
    Pilot Power: {config.pilot_power_fraction*100:.0f}%
    Ranging Power: {config.ranging_power*100:.0f}%
    
    Range Resolution: {3e8/(2*config.bandwidth_hz):.2f} m
    Sub-sample Factor: ~10x
    Effective Resolution: ~0.15 m
    
    Frame Duration: {(config.preamble_duration_ms + 
                      config.ranging_duration_ms + 
                      config.payload_duration_ms):.1f} ms
    """
    
    ax12.text(0.1, 0.9, params_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Spread Spectrum Signal Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('spread_spectrum_analysis.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to: spread_spectrum_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPREAD SPECTRUM SIGNAL VISUALIZATION")
    print("="*70)
    print("\nGenerating and analyzing spread spectrum waveform...")
    print("This shows the actual signal used for ranging in the FTL system.\n")
    
    visualize_spread_spectrum()
    
    print("\nKey observations:")
    print("• Gold codes provide good autocorrelation properties")
    print("• 100 MHz bandwidth spreads energy across spectrum")
    print("• Pilot tones enable frequency synchronization")
    print("• Sub-sample interpolation improves resolution ~10x")
    print("• Root-raised cosine shaping reduces spectral leakage")