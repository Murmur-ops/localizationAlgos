#!/usr/bin/env python3
"""
Verify LFSR implementation and m-sequence properties
"""

import numpy as np
from src.rf.gold_codes import LFSR, GoldCodeGenerator


def verify_msequence_properties():
    """Verify that our LFSRs generate proper m-sequences"""

    print("="*60)
    print("VERIFYING M-SEQUENCE PROPERTIES")
    print("="*60)

    # Test length-127 m-sequence (degree 7)
    # Primitive polynomial: x^7 + x^3 + 1 (taps [7,3])
    lfsr1 = LFSR(taps=[7, 3])

    print(f"\nLFSR with taps [7,3]:")
    print(f"  Expected period: {2**7 - 1} = 127")

    # Generate sequence and check period
    sequence = []
    states = set()
    state = lfsr1.state

    for i in range(150):  # Generate more than expected period
        bit = lfsr1.shift()
        sequence.append(2*bit - 1)  # Convert to +1/-1

        if lfsr1.state in states:
            print(f"  Period found: {i}")
            break
        states.add(lfsr1.state)

    sequence = np.array(sequence[:127])

    # Check balance property (should have one more -1 than +1)
    n_plus = np.sum(sequence == 1)
    n_minus = np.sum(sequence == -1)
    print(f"  Balance: +1s={n_plus}, -1s={n_minus} (should be 63, 64)")

    # Check run properties
    runs = []
    current_val = sequence[0]
    current_length = 1

    for val in sequence[1:]:
        if val == current_val:
            current_length += 1
        else:
            runs.append((current_val, current_length))
            current_val = val
            current_length = 1
    runs.append((current_val, current_length))

    run_lengths = {}
    for val, length in runs:
        if length not in run_lengths:
            run_lengths[length] = 0
        run_lengths[length] += 1

    print(f"  Run distribution: {run_lengths}")
    print(f"  (Should have ~32 runs of length 1, ~16 of length 2, etc.)")

    # Check autocorrelation
    # For m-sequences, autocorrelation should be N for shift=0, -1 for all other shifts
    autocorr = []
    for shift in range(127):
        shifted = np.roll(sequence, shift)
        corr = np.sum(sequence * shifted)
        autocorr.append(corr)

    print(f"\n  Autocorrelation:")
    print(f"    Peak (shift=0): {autocorr[0]} (should be 127)")
    print(f"    Other values: min={min(autocorr[1:])}, max={max(autocorr[1:])} (should all be -1)")

    # Check if all non-zero shifts have correlation = -1
    unique_values = set(autocorr[1:])
    if unique_values == {-1}:
        print(f"    ✓ Perfect m-sequence autocorrelation!")
    else:
        print(f"    ✗ Not a proper m-sequence! Unique values: {unique_values}")

    return sequence


def test_gold_generation():
    """Test that Gold codes are generated correctly from m-sequences"""

    print("\n" + "="*60)
    print("TESTING GOLD CODE GENERATION")
    print("="*60)

    generator = GoldCodeGenerator(127)

    # Get the base m-sequences
    m1 = generator.codes[0]
    m2 = generator.codes[1]

    # Check that they're different
    if np.array_equal(m1, m2):
        print("ERROR: Both m-sequences are the same!")
        return

    print(f"\nBase m-sequences:")
    print(f"  m1 first 20: {m1[:20]}")
    print(f"  m2 first 20: {m2[:20]}")

    # Gold code should be XOR of m1 and shifted m2
    # In +1/-1 domain, XOR is multiplication
    gold_test = generator.codes[2]  # Should be m1 * m2 (no shift)
    expected = m1 * m2

    if np.array_equal(gold_test, expected):
        print(f"  ✓ Gold code generation correct (m1 * m2)")
    else:
        print(f"  ✗ Gold code generation ERROR")
        print(f"    Difference: {np.sum(gold_test != expected)} positions")

    # Test a shifted version
    shift = 10
    gold_shifted = generator.codes[shift + 2]
    m2_shifted = np.roll(m2, shift)
    expected_shifted = m1 * m2_shifted

    if np.array_equal(gold_shifted, expected_shifted):
        print(f"  ✓ Shifted Gold code correct (shift={shift})")
    else:
        print(f"  ✗ Shifted Gold code ERROR")


def test_length_1023():
    """Test GPS-length Gold codes"""

    print("\n" + "="*60)
    print("TESTING GPS-LENGTH (1023) GOLD CODES")
    print("="*60)

    # GPS uses degree-10 LFSRs
    # Preferred pairs: [10,3] and [10,8,3,2]
    lfsr1 = LFSR(taps=[10, 3])
    lfsr2 = LFSR(taps=[10, 8, 3, 2])

    seq1 = lfsr1.generate_sequence(1023)
    seq2 = lfsr2.generate_sequence(1023)

    # Check autocorrelation of each m-sequence
    for i, seq in enumerate([seq1, seq2], 1):
        autocorr = []
        for shift in range(min(50, len(seq))):  # Check first 50 shifts
            shifted = np.roll(seq, shift)
            corr = np.sum(seq * shifted)
            autocorr.append(corr)

        print(f"\nm-sequence {i} autocorrelation:")
        print(f"  Peak: {autocorr[0]} (should be 1023)")
        print(f"  Sidelobes: {set(autocorr[1:])}")

        if set(autocorr[1:]) == {-1}:
            print(f"  ✓ Perfect m-sequence!")
        else:
            print(f"  ✗ Not ideal m-sequence")

    # Generate a Gold code
    gold = seq1 * seq2

    # Check Gold code autocorrelation
    gold_auto = []
    for shift in range(min(50, len(gold))):
        shifted = np.roll(gold, shift)
        corr = np.sum(gold * shifted)
        gold_auto.append(corr)

    print(f"\nGold code autocorrelation:")
    print(f"  Peak: {gold_auto[0]}")
    print(f"  Sidelobe values: {set(gold_auto[1:])}")
    print(f"  Max sidelobe: {max(abs(v) for v in gold_auto[1:])}")

    # For length-1023 Gold codes:
    # Three correlation values: {-1, -65, 63}
    theoretical = {-1, -65, 63}
    print(f"  Theoretical values: {theoretical}")


if __name__ == "__main__":
    # Verify basic m-sequence properties
    seq = verify_msequence_properties()

    # Test Gold code generation
    test_gold_generation()

    # Test GPS-length codes
    test_length_1023()

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)