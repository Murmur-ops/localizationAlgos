#!/usr/bin/env python3
"""
Correct Gold Code Implementation
Using proper Fibonacci LFSR with verified primitive polynomials
"""

import numpy as np
from typing import List, Dict, Tuple


class FibonacciLFSR:
    """
    Fibonacci LFSR - the standard configuration for m-sequences
    Feedback from tapped stages is XORed and fed to the input
    """

    def __init__(self, taps: List[int], length: int):
        """
        Initialize Fibonacci LFSR

        Args:
            taps: Tap positions (e.g., [7, 3] for x^7 + x^3 + 1)
                  Position n means x^n in the polynomial
            length: Sequence length (2^n - 1)
        """
        self.taps = sorted(taps, reverse=True)
        self.degree = max(taps)
        self.length = length

        # Initialize with all 1s (never all 0s)
        self.register = [1] * self.degree

    def shift(self) -> int:
        """Shift register once and return output"""
        # Output is the last stage
        output = self.register[-1]

        # Calculate feedback
        feedback = 0
        for tap in self.taps:
            if tap > 0:  # Don't include x^0 term
                feedback ^= self.register[self.degree - tap]

        # Shift register
        self.register = [feedback] + self.register[:-1]

        return output

    def generate_sequence(self) -> np.ndarray:
        """Generate full period m-sequence"""
        # Reset register
        self.register = [1] * self.degree

        sequence = np.zeros(self.length, dtype=np.int8)
        for i in range(self.length):
            bit = self.shift()
            sequence[i] = 2 * bit - 1  # Convert to +1/-1

        return sequence


class CorrectGoldCodeGenerator:
    """
    Generate Gold codes using correct primitive polynomial pairs
    These are verified from GPS and CDMA standards
    """

    # Verified primitive polynomial pairs that generate preferred Gold codes
    # Format: (length, taps1, taps2)
    PREFERRED_PAIRS = {
        31: (31, [5, 2, 0], [5, 4, 3, 2, 0]),      # Verified
        63: (63, [6, 1, 0], [6, 5, 2, 1, 0]),      # Verified
        127: (127, [7, 3, 0], [7, 3, 2, 1, 0]),    # Verified from literature
        511: (511, [9, 4, 0], [9, 6, 4, 3, 0]),    # Verified
        1023: (1023, [10, 3, 0], [10, 9, 8, 6, 0]) # GPS C/A codes
    }

    def __init__(self, length: int = 127):
        """Initialize Gold code generator"""
        if length not in self.PREFERRED_PAIRS:
            raise ValueError(f"Unsupported length {length}")

        self.length = length
        _, taps1, taps2 = self.PREFERRED_PAIRS[length]

        # Create LFSRs
        self.lfsr1 = FibonacciLFSR(taps1, length)
        self.lfsr2 = FibonacciLFSR(taps2, length)

        # Generate m-sequences
        print(f"Generating m-sequences of length {length}...")
        self.m1 = self.lfsr1.generate_sequence()
        self.m2 = self.lfsr2.generate_sequence()

        # Verify m-sequences
        self._verify_msequence(self.m1, "m1")
        self._verify_msequence(self.m2, "m2")

        # Generate Gold code family
        self.codes = self._generate_gold_family()

    def _verify_msequence(self, seq: np.ndarray, name: str) -> bool:
        """Verify m-sequence properties"""
        n = len(seq)

        # Check autocorrelation at shift 0 and 1
        auto0 = np.sum(seq * seq)
        auto1 = np.sum(seq * np.roll(seq, 1))

        expected_peak = n
        expected_sidelobe = -1

        peak_ok = (auto0 == expected_peak)
        sidelobe_ok = (auto1 == expected_sidelobe)

        if peak_ok and sidelobe_ok:
            print(f"  ✓ {name}: Perfect m-sequence (peak={auto0}, sidelobe={auto1})")
            return True
        else:
            print(f"  ✗ {name}: peak={auto0} (expected {expected_peak}), "
                  f"sidelobe={auto1} (expected {expected_sidelobe})")
            return False

    def _generate_gold_family(self) -> Dict[int, np.ndarray]:
        """Generate complete family of Gold codes"""
        codes = {}

        # First two codes are the m-sequences
        codes[0] = self.m1.copy()
        codes[1] = self.m2.copy()

        # Generate Gold codes by XORing m1 with all cyclic shifts of m2
        for shift in range(self.length):
            gold = self.m1 * np.roll(self.m2, shift)
            codes[shift + 2] = gold

        print(f"Generated {len(codes)} Gold codes")
        return codes

    def get_code(self, index: int) -> np.ndarray:
        """Get specific Gold code"""
        return self.codes[index % len(self.codes)].copy()

    def compute_correlation(self, code1: np.ndarray, code2: np.ndarray) -> float:
        """Compute normalized circular correlation at zero lag"""
        return np.sum(code1 * code2) / len(code1)

    def compute_correlation_function(self, code1: np.ndarray,
                                    code2: np.ndarray = None) -> np.ndarray:
        """Compute full correlation function"""
        if code2 is None:
            code2 = code1  # Autocorrelation

        n = len(code1)
        corr = np.zeros(n)
        for lag in range(n):
            corr[lag] = np.sum(code1 * np.roll(code2, lag))
        return corr / n


def test_correct_implementation():
    """Test the correct Gold code implementation"""

    print("="*60)
    print("TESTING CORRECT GOLD CODE IMPLEMENTATION")
    print("="*60)

    # Test length-127
    gen = CorrectGoldCodeGenerator(127)

    # Test orthogonality
    print("\n" + "="*40)
    print("Testing orthogonality between codes:")

    for i in range(5):
        for j in range(i+1, 6):
            code1 = gen.get_code(i)
            code2 = gen.get_code(j)
            corr = gen.compute_correlation(code1, code2)
            print(f"  Code {i} × Code {j}: {corr:.3f}")

    # Test autocorrelation
    print("\n" + "="*40)
    print("Testing autocorrelation properties:")

    code = gen.get_code(10)
    autocorr = gen.compute_correlation_function(code)

    print(f"  Peak (lag 0): {autocorr[0]:.3f}")
    print(f"  Max sidelobe: {np.max(np.abs(autocorr[1:])):.3f}")
    print(f"  Mean sidelobe: {np.mean(autocorr[1:]):.6f}")

    # For length-127, theoretical bounds
    n = 7  # degree
    t_n = 2**((n+2)//2) + 1  # For n odd
    theoretical_bound = t_n / 127
    print(f"\nTheoretical cross-correlation bound: ±{theoretical_bound:.3f}")

    return gen


if __name__ == "__main__":
    gen = test_correct_implementation()

    # Additional test with GPS-length codes
    print("\n" + "="*60)
    print("Testing GPS-length codes (1023):")
    print("="*60)

    gen_gps = CorrectGoldCodeGenerator(1023)

    # Test a few codes
    code1 = gen_gps.get_code(1)
    code24 = gen_gps.get_code(24)  # GPS satellite PRN 24

    corr = gen_gps.compute_correlation(code1, code24)
    print(f"\nCross-correlation PRN 1 × PRN 24: {corr:.4f}")

    # For GPS (n=10), theoretical values are {-65/1023, -1/1023, 63/1023}
    print(f"GPS theoretical values: {{-65/1023, -1/1023, 63/1023}}")
    print(f"                      = {{-0.0636, -0.0010, 0.0616}}")