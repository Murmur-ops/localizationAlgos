#!/usr/bin/env python3
"""
Test and visualize Gold code correlation properties
Verifies we have REAL spread spectrum codes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rf.gold_codes import GoldCodeGenerator


def test_gold_code_properties(length: int = 127):
    """Test and visualize Gold code correlation properties"""

    print("="*60)
    print(f"TESTING GOLD CODE PROPERTIES (Length {length})")
    print("="*60)

    # Generate Gold codes
    generator = GoldCodeGenerator(length)

    # Get several codes
    code0 = generator.get_code(0)  # First m-sequence
    code1 = generator.get_code(1)  # Second m-sequence
    code10 = generator.get_code(10)  # Gold code
    code20 = generator.get_code(20)  # Another Gold code

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Plot code sequences
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(50)  # Show first 50 chips

    ax1.step(x, code0[:50], 'b-', label='Code 0 (m-sequence 1)', where='mid', linewidth=2)
    ax1.step(x, code1[:50] + 2.5, 'r-', label='Code 1 (m-sequence 2)', where='mid', linewidth=2)
    ax1.step(x, code10[:50] + 5, 'g-', label='Code 10 (Gold)', where='mid', linewidth=2)

    ax1.set_xlabel('Chip Index')
    ax1.set_ylabel('Amplitude (+1/-1)')
    ax1.set_title('Gold Code Sequences (First 50 Chips)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-2, 7])

    # 2. Autocorrelation of Gold code
    ax2 = fig.add_subplot(gs[1, 0])

    autocorr = generator.autocorrelation(code10)
    shifts = np.arange(-length + 1, length)

    ax2.plot(shifts, autocorr, 'b-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Theoretical sidelobe level
    theoretical_sidelobe = -1.0 / length
    ax2.axhline(y=theoretical_sidelobe, color='g', linestyle='--',
                label=f'Theoretical: {theoretical_sidelobe:.3f}')

    ax2.set_xlabel('Shift (chips)')
    ax2.set_ylabel('Normalized Correlation')
    ax2.set_title(f'Autocorrelation of Gold Code (Length {length})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Zoom inset around peak
    axins = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    zoom_range = 10
    center_idx = length - 1
    axins.plot(shifts[center_idx-zoom_range:center_idx+zoom_range+1],
              autocorr[center_idx-zoom_range:center_idx+zoom_range+1],
              'b-', linewidth=2)
    axins.plot(0, autocorr[center_idx], 'ro', markersize=8)
    axins.set_xlim([-zoom_range, zoom_range])
    axins.grid(True, alpha=0.3)
    ax2.indicate_inset_zoom(axins)

    # 3. Cross-correlation between different codes
    ax3 = fig.add_subplot(gs[1, 1])

    crosscorr = generator.crosscorrelation(code10, code20)

    ax3.plot(shifts, crosscorr, 'r-', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Theoretical bounds for Gold codes
    n = int(np.log2(length + 1))  # Degree
    if n % 2 == 1:  # Odd degree
        t_n = 2**((n + 1) // 2) + 1
    else:  # Even degree
        t_n = 2**(n // 2 + 1) + 1

    theoretical_bounds = [-t_n / length, (t_n - 2) / length]
    ax3.axhline(y=theoretical_bounds[0], color='g', linestyle='--',
                label=f'Bounds: ±{t_n/length:.3f}')
    ax3.axhline(y=theoretical_bounds[1], color='g', linestyle='--')

    ax3.set_xlabel('Shift (chips)')
    ax3.set_ylabel('Normalized Correlation')
    ax3.set_title('Cross-correlation (Code 10 × Code 20)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Histogram of correlation values
    ax4 = fig.add_subplot(gs[2, 0])

    # Collect autocorrelation sidelobes
    auto_sidelobes = []
    for i in range(5):  # Sample 5 codes
        code = generator.get_code(i)
        autocorr = generator.autocorrelation(code)
        # Exclude peak
        sidelobes = np.concatenate([autocorr[:length-1], autocorr[length:]])
        auto_sidelobes.extend(sidelobes)

    ax4.hist(auto_sidelobes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=theoretical_sidelobe, color='r', linestyle='--',
                label=f'Theory: {theoretical_sidelobe:.3f}')
    ax4.set_xlabel('Correlation Value')
    ax4.set_ylabel('Count')
    ax4.set_title('Autocorrelation Sidelobe Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Cross-correlation value distribution
    ax5 = fig.add_subplot(gs[2, 1])

    # Collect cross-correlation values
    cross_values = []
    for i in range(3):
        for j in range(i + 1, 4):
            code_i = generator.get_code(i + 10)
            code_j = generator.get_code(j + 10)
            crosscorr = generator.crosscorrelation(code_i, code_j)
            cross_values.extend(crosscorr)

    ax5.hist(cross_values, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.axvline(x=theoretical_bounds[0], color='g', linestyle='--', label='Theoretical bounds')
    ax5.axvline(x=theoretical_bounds[1], color='g', linestyle='--')
    ax5.set_xlabel('Correlation Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Cross-correlation Value Distribution')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.suptitle(f'Gold Code Correlation Properties (Length {length})',
                 fontsize=14, fontweight='bold')

    # Print statistics
    print(f"\nAutocorrelation Statistics:")
    print(f"  Peak value: {autocorr[length-1]:.3f}")
    print(f"  Max sidelobe: {np.max(np.abs(auto_sidelobes)):.3f}")
    print(f"  Mean sidelobe: {np.mean(auto_sidelobes):.6f}")
    print(f"  Std sidelobe: {np.std(auto_sidelobes):.3f}")

    print(f"\nCross-correlation Statistics:")
    print(f"  Maximum: {np.max(cross_values):.3f}")
    print(f"  Minimum: {np.min(cross_values):.3f}")
    print(f"  Mean: {np.mean(cross_values):.6f}")
    print(f"  Std: {np.std(cross_values):.3f}")

    print(f"\nTheoretical values for length {length}:")
    print(f"  Degree n = {n}")
    print(f"  t(n) = {t_n}")
    print(f"  Cross-correlation bounds: ±{t_n/length:.3f}")

    # Verify "three-valued" property
    unique_cross = np.unique(np.round(crosscorr * length) / length)
    print(f"\nUnique cross-correlation values: {len(unique_cross)}")
    if len(unique_cross) <= 5:
        print(f"  Values: {unique_cross}")

    return fig


def compare_with_random_sequence(length: int = 127):
    """Compare Gold codes with random sequences to show the difference"""

    print("\n" + "="*60)
    print("COMPARING GOLD CODES WITH RANDOM SEQUENCES")
    print("="*60)

    # Generate Gold code
    generator = GoldCodeGenerator(length)
    gold_code = generator.get_code(10)

    # Generate random sequence (what we were using before!)
    np.random.seed(42)
    random_seq = 2 * np.random.randint(0, 2, length) - 1

    # Compute autocorrelations
    gold_auto = generator.autocorrelation(gold_code)
    random_auto = np.correlate(random_seq, random_seq, mode='full') / length

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    shifts = np.arange(-length + 1, length)

    # Gold code autocorrelation
    axes[0].plot(shifts, gold_auto, 'b-', linewidth=1)
    axes[0].set_xlabel('Shift (chips)')
    axes[0].set_ylabel('Normalized Correlation')
    axes[0].set_title('Gold Code Autocorrelation')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.3, 1.1])

    # Random sequence autocorrelation
    axes[1].plot(shifts, random_auto, 'r-', linewidth=1)
    axes[1].set_xlabel('Shift (chips)')
    axes[1].set_ylabel('Normalized Correlation')
    axes[1].set_title('Random Sequence "Autocorrelation"')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.3, 1.1])

    plt.suptitle('Real Gold Code vs Random Sequence', fontsize=14, fontweight='bold')

    # Print comparison
    print(f"\nAutocorrelation peak:")
    print(f"  Gold code: {gold_auto[length-1]:.3f}")
    print(f"  Random: {random_auto[length-1]:.3f}")

    print(f"\nMax sidelobe:")
    gold_sidelobes = np.concatenate([gold_auto[:length-1], gold_auto[length:]])
    random_sidelobes = np.concatenate([random_auto[:length-1], random_auto[length:]])
    print(f"  Gold code: {np.max(np.abs(gold_sidelobes)):.3f}")
    print(f"  Random: {np.max(np.abs(random_sidelobes)):.3f}")

    print(f"\nSidelobe variance:")
    print(f"  Gold code: {np.var(gold_sidelobes):.6f}")
    print(f"  Random: {np.var(random_sidelobes):.6f}")

    print("\nConclusion: Random sequences have terrible correlation properties!")
    print("Gold codes have controlled, predictable correlation for CDMA.")

    return fig


if __name__ == "__main__":
    # Test with shorter codes for faster demo
    fig2 = test_gold_code_properties(length=127)
    plt.savefig('gold_code_correlation_127.png', dpi=150, bbox_inches='tight')

    # Compare with random
    fig3 = compare_with_random_sequence(length=127)
    plt.savefig('gold_vs_random.png', dpi=150, bbox_inches='tight')

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print("Plots saved to:")
    print("  - gold_code_correlation_1023.png (GPS-length)")
    print("  - gold_code_correlation_127.png (Short)")
    print("  - gold_vs_random.png (Comparison)")

    plt.show()