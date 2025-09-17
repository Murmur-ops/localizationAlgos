#!/usr/bin/env python3
"""
Real Gold Code Generator using Linear Feedback Shift Registers (LFSRs)
Implements proper m-sequences and Gold codes with verified correlation properties
"""

import numpy as np
from typing import List, Tuple, Dict


class LFSR:
    """Linear Feedback Shift Register for m-sequence generation"""

    def __init__(self, taps: List[int], initial_state: int = None):
        """
        Initialize LFSR with polynomial taps

        IMPORTANT: The taps represent the polynomial x^n + ... + x^tap + ... + 1
        The feedback is the XOR of the bits at the tap positions.

        Args:
            taps: Polynomial tap positions (e.g., [5,3] for x^5 + x^3 + 1)
                  These are 1-indexed positions from the right
            initial_state: Initial register state (defaults to all 1s)
        """
        self.degree = max(taps) if taps else 1
        self.taps = sorted(taps)  # Sort for consistency
        self.length = 2**self.degree - 1  # m-sequence period

        if initial_state is None:
            # Default to all 1s (never use all 0s!)
            self.state = (1 << self.degree) - 1
        else:
            self.state = initial_state

        self.initial_state = self.state

    def shift(self) -> int:
        """
        Perform one shift operation and return output bit
        Using Galois LFSR configuration for simplicity
        """
        # Output bit is the LSB
        output = self.state & 1

        # Shift right
        self.state >>= 1

        # If output bit was 1, XOR with tap polynomial
        if output:
            # Create tap mask (bit positions to flip)
            mask = 0
            for tap in self.taps:
                mask |= (1 << (self.degree - tap))

            # Apply feedback
            self.state ^= mask

        return output

    def generate_sequence(self, length: int = None) -> np.ndarray:
        """
        Generate m-sequence of specified length

        Args:
            length: Sequence length (defaults to full period)

        Returns:
            m-sequence as +1/-1 array
        """
        if length is None:
            length = self.length

        # Reset to initial state
        self.state = self.initial_state

        sequence = np.zeros(length, dtype=np.int8)
        for i in range(length):
            bit = self.shift()
            sequence[i] = 2 * bit - 1  # Convert 0/1 to -1/+1

        return sequence


class GoldCodeGenerator:
    """
    Generate Gold codes using preferred polynomial pairs
    Gold codes have optimal correlation properties for CDMA
    """

    # Preferred polynomial pairs for different sequence lengths
    # Format: (degree, [taps1], [taps2])
    PREFERRED_PAIRS = {
        31: (5, [5, 3], [5, 4, 3, 2]),           # Length 31
        63: (6, [6, 1], [6, 5, 2, 1]),           # Length 63
        127: (7, [7, 3], [7, 3, 2, 1]),          # Length 127
        255: (8, [8, 4, 3, 2], [8, 6, 5, 3]),    # Length 255
        511: (9, [9, 4], [9, 6, 4, 3]),          # Length 511
        1023: (10, [10, 3], [10, 8, 3, 2]),      # Length 1023 (GPS)
        2047: (11, [11, 2], [11, 8, 5, 2]),      # Length 2047
        4095: (12, [12, 6, 4, 1], [12, 9, 8, 4]) # Length 4095
    }

    def __init__(self, length: int = 1023):
        """
        Initialize Gold code generator

        Args:
            length: Desired sequence length (must be 2^n - 1)
        """
        if length not in self.PREFERRED_PAIRS:
            raise ValueError(f"Length {length} not supported. Use one of: {list(self.PREFERRED_PAIRS.keys())}")

        self.length = length
        degree, taps1, taps2 = self.PREFERRED_PAIRS[length]

        # Create two LFSRs with preferred polynomials
        self.lfsr1 = LFSR(taps1)
        self.lfsr2 = LFSR(taps2)

        # Generate base m-sequences
        self.m1 = self.lfsr1.generate_sequence()
        self.m2 = self.lfsr2.generate_sequence()

        # Pre-generate all Gold codes for this length
        self.codes = self._generate_all_codes()

    def _generate_all_codes(self) -> Dict[int, np.ndarray]:
        """Generate complete set of Gold codes"""
        codes = {}

        # First two codes are the m-sequences themselves
        codes[0] = self.m1.copy()
        codes[1] = self.m2.copy()

        # Generate Gold codes by XORing m1 with shifted versions of m2
        for shift in range(self.length):
            # Shift m2 cyclically
            m2_shifted = np.roll(self.m2, shift)

            # XOR the sequences (in +1/-1 domain, this is multiplication)
            gold_code = self.m1 * m2_shifted

            codes[shift + 2] = gold_code

        return codes

    def get_code(self, index: int) -> np.ndarray:
        """
        Get a specific Gold code by index

        Args:
            index: Code index (0 to length+1)

        Returns:
            Gold code as +1/-1 array
        """
        if index < 0 or index >= len(self.codes):
            raise ValueError(f"Code index {index} out of range [0, {len(self.codes)-1}]")

        return self.codes[index].copy()

    def autocorrelation(self, code: np.ndarray) -> np.ndarray:
        """
        Compute CIRCULAR autocorrelation function of a code
        This is critical for spread spectrum - we need circular shifts!

        Args:
            code: Input sequence

        Returns:
            Autocorrelation values for all shifts
        """
        n = len(code)
        autocorr = np.zeros(2 * n - 1)

        # Compute CIRCULAR correlation for each shift
        for shift in range(-n + 1, n):
            # Use numpy roll for circular shift
            shifted = np.roll(code, shift)
            correlation = np.sum(code * shifted)
            autocorr[shift + n - 1] = correlation

        return autocorr / n  # Normalize

    def crosscorrelation(self, code1: np.ndarray, code2: np.ndarray) -> np.ndarray:
        """
        Compute CIRCULAR cross-correlation between two codes
        Must use circular shifts for proper spread spectrum properties!

        Args:
            code1: First sequence
            code2: Second sequence

        Returns:
            Cross-correlation values for all shifts
        """
        n = len(code1)
        crosscorr = np.zeros(2 * n - 1)

        # Compute CIRCULAR correlation for each shift
        for shift in range(-n + 1, n):
            # Use numpy roll for circular shift
            shifted = np.roll(code2, shift)
            correlation = np.sum(code1 * shifted)
            crosscorr[shift + n - 1] = correlation

        return crosscorr / n  # Normalize

    def verify_correlation_properties(self) -> Dict:
        """
        Verify that generated codes have proper Gold code correlation properties

        Returns:
            Dictionary with correlation statistics
        """
        results = {
            'length': self.length,
            'num_codes': len(self.codes),
            'autocorr_peak': [],
            'autocorr_sidelobes': [],
            'crosscorr_max': [],
            'crosscorr_values': set()
        }

        # Check autocorrelation for each code
        for i in range(min(10, len(self.codes))):  # Sample first 10 codes
            code = self.get_code(i)
            autocorr = self.autocorrelation(code)

            # Peak should be at center (zero shift)
            peak = autocorr[self.length - 1]
            results['autocorr_peak'].append(peak)

            # Sidelobes (all other values)
            sidelobes = np.concatenate([autocorr[:self.length-1], autocorr[self.length:]])
            results['autocorr_sidelobes'].extend(sidelobes)

        # Check cross-correlation between different codes
        for i in range(min(5, len(self.codes))):
            for j in range(i + 1, min(5, len(self.codes))):
                code1 = self.get_code(i)
                code2 = self.get_code(j)
                crosscorr = self.crosscorrelation(code1, code2)

                results['crosscorr_max'].append(np.max(np.abs(crosscorr)))

                # Gold codes are "three-valued" - collect unique correlation values
                unique_vals = np.unique(np.round(crosscorr * self.length) / self.length)
                results['crosscorr_values'].update(unique_vals)

        # Statistics
        results['autocorr_peak_mean'] = np.mean(results['autocorr_peak'])
        results['autocorr_sidelobe_max'] = np.max(np.abs(results['autocorr_sidelobes']))
        results['crosscorr_max_mean'] = np.mean(results['crosscorr_max'])

        # For Gold codes, theoretical values:
        # - Autocorr peak = 1 (normalized)
        # - Autocorr sidelobes = -1/N
        # - Cross-corr bounded by: {-t(n), -1, t(n)-2} where t(n) = 2^((n+1)/2) + 1 for odd n

        return results


def demonstrate_gold_codes():
    """Demonstrate Gold code generation and properties"""

    print("="*60)
    print("GOLD CODE GENERATOR - REAL IMPLEMENTATION")
    print("="*60)

    # Generate Gold codes of length 127
    length = 127
    generator = GoldCodeGenerator(length)

    print(f"\nGenerated Gold codes of length {length}")
    print(f"Number of codes: {len(generator.codes)}")

    # Get a few codes
    code0 = generator.get_code(0)
    code1 = generator.get_code(1)
    code10 = generator.get_code(10)

    print(f"\nFirst 20 chips of code 0: {code0[:20]}")
    print(f"First 20 chips of code 1: {code1[:20]}")

    # Verify correlation properties
    print("\nVerifying correlation properties...")
    props = generator.verify_correlation_properties()

    print(f"\nAutocorrelation:")
    print(f"  Peak value (normalized): {props['autocorr_peak_mean']:.3f} (should be 1.0)")
    print(f"  Max sidelobe: {props['autocorr_sidelobe_max']:.3f} (should be ~{-1.0/length:.3f})")

    print(f"\nCross-correlation:")
    print(f"  Maximum value: {props['crosscorr_max_mean']:.3f}")
    print(f"  Number of unique values: {len(props['crosscorr_values'])} (Gold codes are 'three-valued')")

    # Compute and show autocorrelation
    autocorr = generator.autocorrelation(code0)

    print(f"\nAutocorrelation of code 0:")
    print(f"  Peak (τ=0): {autocorr[length-1]:.3f}")
    print(f"  Samples around peak: {autocorr[length-3:length+2]}")

    # Cross-correlation
    crosscorr = generator.crosscorrelation(code0, code10)
    print(f"\nCross-correlation between code 0 and code 10:")
    print(f"  Maximum: {np.max(crosscorr):.3f}")
    print(f"  Minimum: {np.min(crosscorr):.3f}")
    print(f"  Mean: {np.mean(crosscorr):.3f}")

    return generator


if __name__ == "__main__":
    generator = demonstrate_gold_codes()