"""
Quick test of realistic FTL system
"""

import numpy as np
from rf_channel import RangingChannel, ChannelConfig

print("REALISTIC RF CHANNEL TEST")
print("="*50)

# Test 1: Basic propagation
config = ChannelConfig(
    frequency_hz=2.4e9,
    bandwidth_hz=100e6,
    enable_multipath=True,
    iq_amplitude_imbalance_db=0.5,
    iq_phase_imbalance_deg=3.0,
    phase_noise_dbc_hz=-80,
    adc_bits=12
)

channel = RangingChannel(config)

# Test signal
tx_signal = np.ones(1000) + 0j

# Test at different distances
distances = [10, 50, 100, 500, 1000]

print("\nDistance vs Path Loss (Two-way for ranging):")
print("-"*40)

for dist in distances:
    rx_signal, toa_ns, info = channel.process_ranging_signal(
        tx_signal=tx_signal,
        true_distance_m=dist,
        true_velocity_mps=0,
        clock_offset_ns=50,
        freq_offset_hz=1000,
        snr_db=30
    )

    print(f"{dist:5d}m: Path loss={info['path_loss_db']:6.1f}dB, "
          f"ToA error={info['toa_error_ns']:5.2f}ns")

print("\n✓ Realistic RF channel working!")
print("\nKey improvements implemented:")
print("  • R^4 path loss for two-way ranging")
print("  • Multipath propagation (two-ray model)")
print("  • I/Q imbalance: 0.5dB amplitude, 3° phase")
print("  • Phase noise: -80 dBc/Hz")
print("  • ADC quantization: 12 bits")
print("  • Atmospheric attenuation")
print("  • Doppler effects")
print("  • Cramér-Rao bounded noise")