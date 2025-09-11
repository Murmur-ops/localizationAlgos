#!/usr/bin/env python3
"""
Generate PDF with Spread Spectrum Signal Analysis
One figure per page, properly centered and formatted
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rf.spread_spectrum import SpreadSpectrumGenerator, RangingCorrelator, WaveformConfig

def create_spread_spectrum_pdf():
    """Generate multi-page PDF with spread spectrum analysis"""
    
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
    
    # Create PDF
    pdf_filename = 'spread_spectrum_analysis.pdf'
    with PdfPages(pdf_filename) as pdf:
        
        # ==================== PAGE 1: FRAME STRUCTURE ====================
        fig = plt.figure(figsize=(11, 8.5))
        
        # Main plot
        ax = plt.subplot(2, 1, 1)
        t_frame = np.arange(len(tdm_signal)) * dt * 1e3  # Convert to ms
        
        # Show different components
        ax.plot(t_frame[:len(preamble)], np.abs(preamble), 'b-', label='Preamble (Pilots)', alpha=0.8, linewidth=2)
        t_ranging_start = len(preamble) * dt * 1e3
        t_ranging = np.arange(len(ranging)) * dt * 1e3 + t_ranging_start
        ax.plot(t_ranging[:1000], np.abs(ranging[:1000]), 'r-', label='Ranging (PN)', alpha=0.8, linewidth=1)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Complete Spread Spectrum Frame Structure', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])
        
        # Zoomed in view
        ax2 = plt.subplot(2, 1, 2)
        t_zoom = np.arange(2000) * dt * 1e6  # microseconds
        ax2.plot(t_zoom, np.real(ranging[:2000]), 'r-', alpha=0.7)
        ax2.set_xlabel('Time (μs)', fontsize=12)
        ax2.set_ylabel('Amplitude', fontsize=12)
        ax2.set_title('Ranging Signal Detail (First 10 μs)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 10])
        
        plt.suptitle('Page 1: Waveform Time Domain Structure', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ==================== PAGE 2: GOLD CODE & SPREADING ====================
        fig = plt.figure(figsize=(11, 8.5))
        
        # Gold code sequence
        ax1 = plt.subplot(3, 1, 1)
        gold_code = generator.ranging_pn[:200]
        ax1.stem(range(len(gold_code)), gold_code, basefmt=' ', markerfmt='bo', linefmt='b-')
        ax1.set_xlabel('Chip Index', fontsize=12)
        ax1.set_ylabel('Chip Value', fontsize=12)
        ax1.set_title(f'Gold Code Sequence (First 200 of {config.ranging_pn_length} chips)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-1.5, 1.5])
        
        # Autocorrelation
        ax2 = plt.subplot(3, 1, 2)
        gold_full = generator.ranging_pn
        autocorr = np.correlate(gold_full, gold_full, mode='same')
        lags = np.arange(-len(autocorr)//2, len(autocorr)//2)
        ax2.plot(lags, autocorr, 'b-', alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('Lag (chips)', fontsize=12)
        ax2.set_ylabel('Correlation', fontsize=12)
        ax2.set_title('Gold Code Autocorrelation Function', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-200, 200])
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
        
        # Pulse shaping
        ax3 = plt.subplot(3, 1, 3)
        samples_per_chip = int(config.sample_rate_hz / config.chip_rate_hz)
        raw_chips = np.repeat(gold_code[:30], samples_per_chip)
        shaped_signal = ranging[:len(raw_chips)]
        t_chips = np.arange(len(raw_chips)) * dt * 1e6
        ax3.plot(t_chips, raw_chips, 'b-', label='Raw Chips', alpha=0.5, linewidth=2)
        ax3.plot(t_chips, np.real(shaped_signal), 'r-', label='Pulse Shaped (RRC β=0.35)', alpha=0.8, linewidth=1.5)
        ax3.set_xlabel('Time (μs)', fontsize=12)
        ax3.set_ylabel('Amplitude', fontsize=12)
        ax3.set_title('Root Raised Cosine Pulse Shaping', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Page 2: Gold Code Properties & Pulse Shaping', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ==================== PAGE 3: FREQUENCY DOMAIN ====================
        fig = plt.figure(figsize=(11, 8.5))
        
        # Pilot tones spectrum
        ax1 = plt.subplot(3, 1, 1)
        N = len(preamble)
        frequencies = np.fft.fftfreq(N, dt) / 1e6
        preamble_fft = np.fft.fft(preamble)
        preamble_spectrum = 20 * np.log10(np.abs(preamble_fft) + 1e-10)
        ax1.plot(frequencies[:N//2], preamble_spectrum[:N//2], 'b-', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Frequency (MHz)', fontsize=12)
        ax1.set_ylabel('Power (dB)', fontsize=12)
        ax1.set_title('Pilot Tones for Frequency Synchronization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 10])
        for freq_mhz in config.pilot_frequencies_mhz:
            if freq_mhz >= 0:
                ax1.axvline(x=freq_mhz, color='r', linestyle='--', alpha=0.5)
                ax1.text(freq_mhz, ax1.get_ylim()[1]*0.9, f'{freq_mhz}MHz', 
                        rotation=90, va='top', fontsize=8)
        
        # Ranging signal spectrum
        ax2 = plt.subplot(3, 1, 2)
        N_ranging = len(ranging)
        freq_ranging = np.fft.fftfreq(N_ranging, dt) / 1e6
        ranging_fft = np.fft.fft(ranging)
        ranging_spectrum = 20 * np.log10(np.abs(ranging_fft) + 1e-10)
        ax2.plot(freq_ranging[:N_ranging//2], ranging_spectrum[:N_ranging//2], 'r-', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Frequency (MHz)', fontsize=12)
        ax2.set_ylabel('Power (dB)', fontsize=12)
        ax2.set_title('Spread Spectrum Signal (100 MHz Bandwidth)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        max_power = np.max(ranging_spectrum[:N_ranging//2])
        ax2.axhline(y=max_power-3, color='g', linestyle='--', alpha=0.5, label='-3dB Bandwidth')
        ax2.legend(fontsize=11)
        
        # Power spectral density
        ax3 = plt.subplot(3, 1, 3)
        f_psd, psd = signal.welch(ranging, config.sample_rate_hz, nperseg=1024, return_onesided=True)
        ax3.semilogy(f_psd/1e6, psd, 'g-', alpha=0.8, linewidth=1.5)
        ax3.set_xlabel('Frequency (MHz)', fontsize=12)
        ax3.set_ylabel('PSD (V²/Hz)', fontsize=12)
        ax3.set_title('Power Spectral Density', fontsize=12)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_xlim([0, 100])
        
        plt.suptitle('Page 3: Frequency Domain Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ==================== PAGE 4: CORRELATION & RANGING ====================
        fig = plt.figure(figsize=(11, 8.5))
        
        # Simulate received signal
        delay_samples = 1000
        noise_power = 0.1
        received = np.concatenate([
            np.zeros(delay_samples, dtype=complex),
            ranging + noise_power * (np.random.randn(len(ranging)) + 1j * np.random.randn(len(ranging)))
        ])
        result = correlator.correlate(received)
        correlation = result['correlation']
        peak_idx = int(result['fine_peak_idx'])
        
        # Correlation peak
        ax1 = plt.subplot(3, 1, 1)
        window = 500
        start = max(0, peak_idx - window)
        end = min(len(correlation), peak_idx + window)
        x_samples = np.arange(start, end)
        x_meters = (x_samples - peak_idx) * (3e8 / config.sample_rate_hz)
        ax1.plot(x_meters, correlation[start:end], 'b-', alpha=0.7, linewidth=1.5)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Peak')
        ax1.set_xlabel('Distance Error (meters)', fontsize=12)
        ax1.set_ylabel('Correlation Magnitude', fontsize=12)
        ax1.set_title(f'Ranging Correlation Peak (SNR = {result["snr_db"]:.1f} dB)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Sub-sample interpolation detail
        ax2 = plt.subplot(3, 1, 2)
        fine_window = 5
        peak_start = max(0, peak_idx - fine_window)
        peak_end = min(len(correlation), peak_idx + fine_window)
        x_coarse = np.arange(peak_start, peak_end)
        y_coarse = correlation[peak_start:peak_end]
        
        ax2.plot(x_coarse, y_coarse, 'bo-', markersize=10, linewidth=2, label='Samples')
        
        if peak_idx > 0 and peak_idx < len(correlation) - 1:
            y1, y2, y3 = correlation[peak_idx-1:peak_idx+2]
            a = (y1 - 2*y2 + y3) / 2
            b = (y3 - y1) / 2
            c = y2
            x_offset = np.linspace(-1, 1, 100)
            y_fine = a * x_offset**2 + b * x_offset + c
            x_fine = peak_idx + x_offset
            ax2.plot(x_fine, y_fine, 'r-', linewidth=2, label='Parabolic Fit')
        
        ax2.axvline(x=result['fine_peak_idx'], color='g', linestyle='--', alpha=0.5, linewidth=2, label='Sub-sample Peak')
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Correlation Magnitude', fontsize=12)
        ax2.set_title('Sub-sample Interpolation (10x Resolution Enhancement)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Range resolution analysis
        ax3 = plt.subplot(3, 1, 3)
        distances = np.linspace(0, 100, 1000)
        theoretical_resolution = 3e8 / (2 * config.bandwidth_hz)
        effective_resolution = theoretical_resolution / 10  # With sub-sampling
        
        ax3.axhline(y=theoretical_resolution, color='r', linestyle='--', linewidth=2, 
                   label=f'Theoretical: {theoretical_resolution:.2f}m')
        ax3.axhline(y=effective_resolution, color='g', linestyle='-', linewidth=2,
                   label=f'With Sub-sampling: {effective_resolution:.2f}m')
        ax3.fill_between([0, 100], 0, effective_resolution, alpha=0.3, color='green', label='Achievable')
        ax3.set_xlabel('Range (meters)', fontsize=12)
        ax3.set_ylabel('Resolution (meters)', fontsize=12)
        ax3.set_title('Range Resolution vs Distance', fontsize=12)
        ax3.set_xlim([0, 100])
        ax3.set_ylim([0, 3])
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Page 4: Correlation & Ranging Performance', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ==================== PAGE 5: SYSTEM PARAMETERS ====================
        fig = plt.figure(figsize=(11, 8.5))
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        
        params_text = f"""
SPREAD SPECTRUM SIGNAL PARAMETERS
{'='*60}

RF PARAMETERS
-------------
• Carrier Frequency:     2.4 GHz (ISM Band)
• Bandwidth:            {config.bandwidth_hz/1e6:.0f} MHz
• Sample Rate:          {config.sample_rate_hz/1e6:.0f} Msps (2x oversampling)
• Chip Rate:            {config.chip_rate_hz/1e6:.0f} Mcps

SPREADING PARAMETERS
--------------------
• Gold Code Length:     {config.ranging_pn_length} chips
• Spreading Factor:     {config.ranging_pn_length}
• Processing Gain:      {10*np.log10(config.ranging_pn_length):.1f} dB
• Pulse Shaping:        Root Raised Cosine (β = 0.35)

PILOT CONFIGURATION
-------------------
• Pilot Frequencies:    {config.pilot_frequencies_mhz} MHz
• Pilot Power:          {config.pilot_power_fraction*100:.0f}% of total
• Ranging Power:        {config.ranging_power*100:.0f}% of total
• Data Power:           {config.data_power*100:.0f}% of total

FRAME STRUCTURE
---------------
• Preamble Duration:    {config.preamble_duration_ms} ms (pilots)
• Ranging Duration:     {config.ranging_duration_ms} ms (PN sequence)
• Payload Duration:     {config.payload_duration_ms} ms (data)
• Total Frame:          {config.preamble_duration_ms + config.ranging_duration_ms + config.payload_duration_ms} ms

RANGING PERFORMANCE
-------------------
• Theoretical Resolution:   {3e8/(2*config.bandwidth_hz):.2f} m (c/2B)
• Sub-sample Factor:        ~10x improvement
• Effective Resolution:     ~0.15 m
• Maximum Unambiguous Range: {3e8 * config.ranging_pn_length / config.chip_rate_hz:.1f} m
• Update Rate:              10-100 Hz (depends on TDMA)

SYNCHRONIZATION
---------------
• Frequency Sync:       Via pilot tones (PLL)
• Time Sync:           PTP 4-timestamp exchange
• CFO Tolerance:       ±100 Hz after lock
• Time Sync Accuracy:  ±10 ns (3m equivalent)
"""
        
        ax.text(0.5, 0.5, params_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.2))
        
        plt.suptitle('Page 5: System Parameters Summary', fontsize=16, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Spread Spectrum Signal Analysis'
        d['Author'] = 'FTL Localization System'
        d['Subject'] = 'RF Waveform Analysis for Distributed Localization'
        d['Keywords'] = 'Spread Spectrum, Gold Codes, Ranging, Localization'
        
    print(f"\nPDF generated successfully: {pdf_filename}")
    print(f"Total pages: 5")
    print("\nPage contents:")
    print("  1. Waveform Time Domain Structure")
    print("  2. Gold Code Properties & Pulse Shaping")
    print("  3. Frequency Domain Analysis")
    print("  4. Correlation & Ranging Performance")
    print("  5. System Parameters Summary")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING SPREAD SPECTRUM ANALYSIS PDF")
    print("="*70)
    
    create_spread_spectrum_pdf()
    
    print("\n" + "="*70)
    print("PDF GENERATION COMPLETE")
    print("="*70)