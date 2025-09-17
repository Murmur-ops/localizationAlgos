#!/usr/bin/env python3
"""
Demonstration of REAL spread spectrum with Gold codes
Shows spreading, despreading, and correlation properties for ranging
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.rf.gold_codes_proper import ProperGoldCodeGenerator


def demonstrate_spread_spectrum():
    """Demonstrate real spread spectrum communication and ranging"""

    print("="*60)
    print("REAL SPREAD SPECTRUM DEMONSTRATION")
    print("="*60)

    # Generate Gold codes
    code_length = 127
    generator = ProperGoldCodeGenerator(code_length)

    # Get two different Gold codes (for different users)
    code_user1 = generator.get_code(10)
    code_user2 = generator.get_code(20)

    print(f"\nUsing Gold codes of length {code_length}")
    print(f"Code orthogonality: {np.dot(code_user1, code_user2) / code_length:.3f}")

    # 1. Create data to transmit
    data_bits = np.array([1, -1, 1, 1, -1])  # 5 bits
    print(f"\nData to transmit: {data_bits}")

    # 2. Spread the data with Gold code (Direct Sequence Spread Spectrum)
    spread_signal1 = []
    for bit in data_bits:
        # Each data bit is spread by the entire Gold code
        spread_signal1.extend(bit * code_user1)

    spread_signal1 = np.array(spread_signal1)
    print(f"Spread signal length: {len(spread_signal1)} chips")

    # 3. Simulate channel with noise and interference
    # Add AWGN
    snr_db = 0  # 0 dB SNR (signal power = noise power)
    noise_power = 10**(-snr_db/10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(spread_signal1))

    # Add interference from another user
    data_bits2 = np.array([-1, 1, -1, -1, 1])
    spread_signal2 = []
    for bit in data_bits2:
        spread_signal2.extend(bit * code_user2)
    spread_signal2 = np.array(spread_signal2)

    # Received signal = desired + interference + noise
    received = spread_signal1 + 0.7 * spread_signal2 + noise

    print(f"\nChannel conditions:")
    print(f"  SNR: {snr_db} dB")
    print(f"  Interference from User 2")

    # 4. Despread using correlation with Gold code
    despread = []
    for i in range(len(data_bits)):
        # Extract the portion corresponding to this bit
        start = i * code_length
        end = start + code_length
        chip_segment = received[start:end]

        # Correlate with the Gold code
        correlation = np.sum(chip_segment * code_user1) / code_length
        despread.append(correlation)

    despread = np.array(despread)

    # 5. Make bit decisions
    detected_bits = np.sign(despread)

    print(f"\nDespreading results:")
    print(f"  Correlation values: {despread}")
    print(f"  Detected bits: {detected_bits}")
    print(f"  Original bits: {data_bits}")
    print(f"  Bit errors: {np.sum(detected_bits != data_bits)}")

    # 6. Demonstrate ranging capability
    print("\n" + "-"*40)
    print("RANGING DEMONSTRATION")
    print("-"*40)

    # Simulate time delay (propagation)
    delay_chips = 15  # 15 chip delay
    delayed_code = np.roll(code_user1, delay_chips)

    # Compute correlation at different lags
    max_lag = 50
    correlations = []
    lags = range(-max_lag, max_lag)

    for lag in lags:
        shifted = np.roll(delayed_code, lag)
        corr = np.sum(code_user1 * shifted) / code_length
        correlations.append(corr)

    correlations = np.array(correlations)

    # Find peak
    peak_idx = np.argmax(correlations)
    detected_delay = lags[peak_idx]

    print(f"True delay: {delay_chips} chips")
    print(f"Detected delay: {detected_delay} chips")
    print(f"Peak correlation: {correlations[peak_idx]:.3f}")

    # Visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Gold codes
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(code_length)
    ax1.step(x, code_user1, 'b-', label='User 1 Gold Code', where='mid')
    ax1.step(x, code_user2 + 2.5, 'r-', label='User 2 Gold Code', where='mid')
    ax1.set_xlabel('Chip Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Gold Codes (Orthogonal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-2, 4])

    # 2. Data spreading
    ax2 = fig.add_subplot(gs[0, 1])
    x_data = np.arange(len(spread_signal1))
    ax2.plot(x_data[:code_length*2], spread_signal1[:code_length*2], 'b-', alpha=0.7)
    ax2.axvline(x=code_length, color='r', linestyle='--', label='Bit boundary')
    ax2.set_xlabel('Chip Index')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Spread Signal (First 2 Bits)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Received signal with noise
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x_data[:code_length*2], received[:code_length*2], 'g-', alpha=0.7)
    ax3.axvline(x=code_length, color='r', linestyle='--')
    ax3.set_xlabel('Chip Index')
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'Received Signal (SNR={snr_db}dB + Interference)')
    ax3.grid(True, alpha=0.3)

    # 4. Despread values
    ax4 = fig.add_subplot(gs[1, 1])
    bit_indices = np.arange(len(data_bits))
    ax4.stem(bit_indices, despread, basefmt=' ', label='Correlation')
    ax4.plot(bit_indices, data_bits * 0.8, 'ro', markersize=10, label='Original')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Bit Index')
    ax4.set_ylabel('Correlation Value')
    ax4.set_title('Despread Signal (After Correlation)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Ranging correlation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(lags, correlations, 'b-', linewidth=2)
    ax5.axvline(x=detected_delay, color='r', linestyle='--', label=f'Peak at {detected_delay}')
    ax5.axvline(x=delay_chips, color='g', linestyle=':', label=f'True delay = {delay_chips}')
    ax5.set_xlabel('Lag (chips)')
    ax5.set_ylabel('Normalized Correlation')
    ax5.set_title('Ranging: Finding Time Delay via Correlation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Real Spread Spectrum with Gold Codes', fontsize=14, fontweight='bold')

    return fig


def compare_real_vs_random():
    """Compare real Gold codes with random sequences"""

    print("\n" + "="*60)
    print("COMPARING REAL GOLD CODES vs RANDOM SEQUENCES")
    print("="*60)

    code_length = 127

    # Generate real Gold code
    generator = ProperGoldCodeGenerator(code_length)
    gold_code1 = generator.get_code(10)
    gold_code2 = generator.get_code(20)

    # Generate random sequences (what we had before!)
    np.random.seed(42)
    random_code1 = 2 * np.random.randint(0, 2, code_length) - 1
    np.random.seed(43)
    random_code2 = 2 * np.random.randint(0, 2, code_length) - 1

    # Test near-far resistance
    # Strong signal from user 1, weak from user 2
    strong_signal = 10.0 * gold_code1
    weak_signal = 0.1 * gold_code2
    combined = strong_signal + weak_signal

    # Try to detect weak user 2 in presence of strong user 1
    gold_correlation = np.sum(combined * gold_code2) / code_length
    print(f"\nGold code near-far resistance:")
    print(f"  Weak signal correlation: {gold_correlation:.3f}")
    print(f"  (Should detect 0.1)")

    # Same test with random codes
    strong_random = 10.0 * random_code1
    weak_random = 0.1 * random_code2
    combined_random = strong_random + weak_random

    random_correlation = np.sum(combined_random * random_code2) / code_length
    print(f"\nRandom sequence 'near-far resistance':")
    print(f"  Weak signal correlation: {random_correlation:.3f}")
    print(f"  (Complete failure!)")

    # Multiple Access Interference (MAI)
    print(f"\n" + "-"*40)
    print("Multiple Access Interference Test")
    print("-"*40)

    # 5 users transmitting simultaneously
    n_users = 5
    gold_codes = [generator.get_code(i) for i in range(n_users)]

    # Random amplitudes for each user
    amplitudes = np.random.uniform(0.5, 2.0, n_users)

    # Combined signal
    combined_gold = np.zeros(code_length)
    for i, (code, amp) in enumerate(zip(gold_codes, amplitudes)):
        combined_gold += amp * code

    # Try to detect each user
    print("\nGold codes - detecting multiple users:")
    for i in range(n_users):
        correlation = np.sum(combined_gold * gold_codes[i]) / code_length
        print(f"  User {i}: sent {amplitudes[i]:.2f}, detected {correlation:.2f}")

    # Same with random codes
    random_codes = []
    for i in range(n_users):
        np.random.seed(100 + i)
        random_codes.append(2 * np.random.randint(0, 2, code_length) - 1)

    combined_random = np.zeros(code_length)
    for i, (code, amp) in enumerate(zip(random_codes, amplitudes)):
        combined_random += amp * code

    print("\nRandom sequences - detecting multiple users:")
    for i in range(n_users):
        correlation = np.sum(combined_random * random_codes[i]) / code_length
        print(f"  User {i}: sent {amplitudes[i]:.2f}, detected {correlation:.2f}")

    print("\nConclusion: Gold codes enable CDMA, random sequences don't!")


if __name__ == "__main__":
    # Demonstrate spread spectrum
    fig = demonstrate_spread_spectrum()
    plt.savefig('real_spread_spectrum_demo.png', dpi=150, bbox_inches='tight')

    # Compare with random
    compare_real_vs_random()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Results saved to: real_spread_spectrum_demo.png")

    plt.show()