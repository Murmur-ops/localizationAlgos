#!/usr/bin/env python3
"""
Working Gold Code Generator
Using reference implementation patterns from GPS C/A code generation
"""

import numpy as np
from typing import List, Dict


def generate_msequence_from_taps(taps: List[int], initial_fill: List[int] = None) -> np.ndarray:
    """
    Generate m-sequence using linear feedback shift register

    Args:
        taps: Feedback tap positions (1-indexed from left)
        initial_fill: Initial register state (defaults to all 1s)

    Returns:
        m-sequence as +1/-1 array
    """
    degree = max(taps)
    length = 2**degree - 1

    if initial_fill is None:
        register = [1] * degree
    else:
        register = initial_fill.copy()

    output = []

    # Generate sequence
    for _ in range(length):
        # Output from rightmost position
        output.append(register[-1])

        # Calculate feedback as XOR of tapped positions
        feedback = 0
        for tap in taps:
            feedback ^= register[tap - 1]  # Convert to 0-indexed

        # Shift register right and insert feedback at left
        register = [feedback] + register[:-1]

    # Convert to +1/-1
    return np.array([2*bit - 1 for bit in output], dtype=np.int8)


class WorkingGoldCodeGenerator:
    """Generate Gold codes using proven tap configurations"""

    # These tap sets are from published literature and standards
    # They generate m-sequences with ideal autocorrelation
    PROVEN_TAPS = {
        127: {
            'g1': [7, 3],      # x^7 + x^3 + 1
            'g2': [7, 1]       # x^7 + x + 1
        },
        1023: {
            'g1': [10, 3],     # x^10 + x^3 + 1 (GPS G1)
            'g2': [10, 9, 8, 6, 3, 2]  # x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1 (GPS G2)
        }
    }

    def __init__(self, length: int = 127):
        """Initialize with desired code length"""
        if length not in self.PROVEN_TAPS:
            raise ValueError(f"Length {length} not supported. Use 127 or 1023.")

        self.length = length
        taps = self.PROVEN_TAPS[length]

        print(f"Generating m-sequences of length {length}...")

        # Generate the two m-sequences
        self.m1 = generate_msequence_from_taps(taps['g1'])
        self.m2 = generate_msequence_from_taps(taps['g2'])

        # Verify they're proper m-sequences
        self._verify_and_report()

        # Generate Gold code family
        self.codes = self._generate_gold_codes()

    def _verify_and_report(self):
        """Verify m-sequence properties"""
        for seq, name in [(self.m1, "m1"), (self.m2, "m2")]:
            # Check period (should use all non-zero states)
            # Check autocorrelation
            auto_peak = np.sum(seq * seq)
            auto_shift = np.sum(seq * np.roll(seq, 1))

            if auto_peak == self.length and auto_shift == -1:
                print(f"  ✓ {name}: Perfect m-sequence!")
            else:
                print(f"  {name}: peak={auto_peak}, shift1={auto_shift}")
                # Check a few more shifts
                for shift in [2, 3, 5, 10]:
                    auto = np.sum(seq * np.roll(seq, shift))
                    if auto != -1:
                        print(f"    shift {shift}: {auto} (should be -1)")

    def _generate_gold_codes(self) -> Dict[int, np.ndarray]:
        """Generate Gold code family"""
        codes = {}

        # The two m-sequences are valid codes
        codes[0] = self.m1.copy()
        codes[1] = self.m2.copy()

        # Generate Gold codes from all relative phases
        for phase in range(self.length):
            gold = self.m1 * np.roll(self.m2, phase)
            codes[phase + 2] = gold

        print(f"Generated {len(codes)} Gold codes")
        return codes

    def get_code(self, index: int) -> np.ndarray:
        """Get a specific Gold code"""
        return self.codes[index % len(self.codes)].copy()

    def test_properties(self):
        """Test correlation properties of generated codes"""
        print("\n" + "="*50)
        print("CORRELATION PROPERTIES TEST")
        print("="*50)

        # Test m-sequence autocorrelation in detail
        print("\nm-sequence autocorrelation (first 10 lags):")
        for seq, name in [(self.m1, "m1"), (self.m2, "m2")]:
            corr_vals = []
            for lag in range(10):
                corr = np.sum(seq * np.roll(seq, lag)) / self.length
                corr_vals.append(corr)
            print(f"  {name}: {[f'{v:.3f}' for v in corr_vals]}")

        # Test Gold code properties
        print("\nGold code cross-correlation samples:")
        test_pairs = [(2, 3), (2, 10), (10, 20), (5, 15)]

        for i, j in test_pairs:
            code_i = self.get_code(i)
            code_j = self.get_code(j)

            # Check correlation at various lags
            max_corr = 0
            for lag in range(0, self.length, self.length // 10):
                corr = np.sum(code_i * np.roll(code_j, lag)) / self.length
                max_corr = max(max_corr, abs(corr))

            print(f"  Code {i} × Code {j}: max |correlation| = {max_corr:.3f}")

        # Theoretical bounds
        degree = int(np.log2(self.length + 1))
        if degree % 2 == 1:  # odd
            t_val = 2**((degree + 1)//2) + 1
        else:  # even
            t_val = 2**((degree + 2)//2) + 1

        bound = t_val / self.length
        print(f"\nTheoretical correlation bound: ±{bound:.3f}")
        print(f"(Three values: ±{t_val}/{self.length} and -1/{self.length})")


def demonstrate_ranging():
    """Demonstrate ranging with Gold codes"""

    print("\n" + "="*60)
    print("RANGING DEMONSTRATION WITH GOLD CODES")
    print("="*60)

    gen = WorkingGoldCodeGenerator(127)

    # Get a Gold code for ranging
    ranging_code = gen.get_code(5)

    # Simulate received signal with delay
    true_delay = 23  # chips
    received = np.roll(ranging_code, true_delay)

    # Add noise
    snr_db = 0
    noise_power = 10**(-snr_db/10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(received))
    received = received + noise

    # Correlate to find delay
    correlations = []
    for lag in range(len(ranging_code)):
        corr = np.sum(received * np.roll(ranging_code, -lag))
        correlations.append(corr)

    correlations = np.array(correlations)
    detected_delay = np.argmax(correlations)

    print(f"\nRanging test (SNR = {snr_db} dB):")
    print(f"  True delay: {true_delay} chips")
    print(f"  Detected delay: {detected_delay} chips")
    print(f"  Error: {abs(detected_delay - true_delay)} chips")
    print(f"  Peak correlation: {correlations[detected_delay]/len(ranging_code):.3f}")
    print(f"  Mean sidelobe: {np.mean(np.abs(correlations[correlations != correlations[detected_delay]])/len(ranging_code)):.3f}")


if __name__ == "__main__":
    # Test with both supported lengths
    for length in [127, 1023]:
        print("\n" + "="*60)
        print(f"TESTING LENGTH-{length} GOLD CODES")
        print("="*60)

        gen = WorkingGoldCodeGenerator(length)
        gen.test_properties()

    # Demonstrate ranging
    demonstrate_ranging()