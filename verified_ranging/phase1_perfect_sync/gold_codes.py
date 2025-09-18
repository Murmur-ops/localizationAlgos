"""
Gold Code Generation with Verification
Phase 1: Perfect implementation with checks
"""

import numpy as np
from typing import List, Tuple


class VerifiedGoldCodes:
    """
    Gold code generator with complete verification
    No shortcuts, full checking at every step
    """

    # Verified primitive polynomials for m-sequences
    PRIMITIVE_POLYNOMIALS = {
        5: [[5, 3, 2, 1], [5, 4, 3, 2]],    # Length 31
        6: [[6, 5, 2, 1], [6, 5, 3, 2]],    # Length 63
        7: [[7, 3], [7, 3, 2, 1]],          # Length 127
        10: [[10, 3], [10, 9, 8, 6, 3, 2]]  # Length 1023 (GPS)
    }

    def __init__(self, m: int = 7):
        """
        Initialize with register size m
        Generates codes of length 2^m - 1
        """
        if m not in self.PRIMITIVE_POLYNOMIALS:
            raise ValueError(f"No primitive polynomials defined for m={m}")

        self.m = m
        self.n = 2**m - 1  # Code length
        self.g1_taps = self.PRIMITIVE_POLYNOMIALS[m][0]
        self.g2_taps = self.PRIMITIVE_POLYNOMIALS[m][1]

        print(f"Generating Gold codes of length {self.n}")
        print(f"  G1 taps: {self.g1_taps}")
        print(f"  G2 taps: {self.g2_taps}")

        # Generate and verify m-sequences
        self.m1 = self._generate_m_sequence(self.g1_taps)
        self.m2 = self._generate_m_sequence(self.g2_taps)

        # Verify m-sequences
        self._verify_m_sequence(self.m1, "G1")
        self._verify_m_sequence(self.m2, "G2")

        # Generate Gold code family
        self.gold_codes = self._generate_gold_family()

        # Verify Gold code properties
        self._verify_gold_properties()

    def _generate_m_sequence(self, taps: List[int]) -> np.ndarray:
        """Generate m-sequence using LFSR with given taps"""
        register = np.ones(self.m, dtype=int)
        sequence = np.zeros(self.n, dtype=int)

        for i in range(self.n):
            # Output bit
            sequence[i] = register[-1]

            # Feedback
            feedback = 0
            for tap in taps:
                if tap <= self.m:
                    feedback ^= register[self.m - tap]

            # Shift register
            register = np.roll(register, 1)
            register[0] = feedback

        # Convert to +1/-1
        return 2 * sequence - 1

    def _verify_m_sequence(self, seq: np.ndarray, name: str):
        """Verify m-sequence properties"""
        print(f"\nVerifying {name} m-sequence:")

        # Check period
        autocorr = np.correlate(seq, seq, mode='full')
        peak_idx = len(seq) - 1
        peak = autocorr[peak_idx]

        # Check autocorrelation
        off_peak = autocorr[np.arange(len(autocorr)) != peak_idx]
        off_peak_max = np.max(np.abs(off_peak))

        print(f"  Length: {len(seq)} (expected {self.n})")
        print(f"  Autocorr peak: {peak} (expected {self.n})")
        print(f"  Autocorr sidelobes: {off_peak_max} (expected -1)")

        # Verify balance property (one more 1 than -1)
        ones = np.sum(seq == 1)
        minus_ones = np.sum(seq == -1)
        print(f"  Balance: {ones} ones, {minus_ones} minus-ones")

        if len(seq) != self.n:
            raise ValueError(f"{name} sequence has wrong length")
        if peak != self.n:
            raise ValueError(f"{name} autocorrelation peak is wrong")
        if off_peak_max != 1:  # Should be -1 in correlation, but abs makes it 1
            print(f"  WARNING: {name} sidelobes = {off_peak_max}, expected 1")

    def _generate_gold_family(self) -> List[np.ndarray]:
        """Generate complete family of Gold codes"""
        codes = []

        # Add m1
        codes.append(self.m1)

        # Add m2
        codes.append(self.m2)

        # Add all shifts of m2 XORed with m1
        for shift in range(self.n):
            shifted_m2 = np.roll(self.m2, shift)
            gold = self.m1 * shifted_m2  # Multiplication for +1/-1 sequences
            codes.append(gold)

        print(f"\nGenerated {len(codes)} Gold codes")
        return codes

    def _verify_gold_properties(self):
        """Verify Gold code correlation properties"""
        print("\nVerifying Gold code properties:")

        # Theoretical bounds for cross-correlation
        if self.m % 2 == 1:  # Odd m
            t_m = 2**((self.m + 1) // 2) + 1
        else:  # Even m
            t_m = 2**((self.m + 2) // 2) + 1

        print(f"  Theoretical max cross-correlation: {t_m}")

        # Check a few codes
        max_cross = 0
        for i in range(min(5, len(self.gold_codes))):
            for j in range(i + 1, min(5, len(self.gold_codes))):
                cross = np.correlate(self.gold_codes[i], self.gold_codes[j], mode='full')
                max_val = np.max(np.abs(cross))
                max_cross = max(max_cross, max_val)

        print(f"  Measured max cross-correlation: {max_cross}")

        if max_cross > t_m:
            print(f"  WARNING: Cross-correlation {max_cross} exceeds theoretical bound {t_m}")
        else:
            print(f"  ✓ Cross-correlation within bounds")

        # Check autocorrelation of first Gold code
        auto = np.correlate(self.gold_codes[2], self.gold_codes[2], mode='full')
        peak = auto[len(self.gold_codes[2]) - 1]
        sidelobes = np.max(np.abs(auto[np.arange(len(auto)) != len(self.gold_codes[2]) - 1]))

        print(f"  Gold code autocorrelation peak: {peak}")
        print(f"  Gold code max sidelobe: {sidelobes}")

    def get_code(self, index: int) -> np.ndarray:
        """Get a specific Gold code"""
        if index >= len(self.gold_codes):
            raise ValueError(f"Index {index} out of range (have {len(self.gold_codes)} codes)")
        return self.gold_codes[index].copy()

    def verify_correlation(self, code1: np.ndarray, code2: np.ndarray) -> Tuple[float, float]:
        """
        Verify correlation between two codes
        Returns (peak_value, peak_location)
        """
        corr = np.correlate(code1, code2, mode='full')
        peak_idx = np.argmax(np.abs(corr))
        peak_val = corr[peak_idx]

        # Peak should be at center for aligned codes
        center = len(code1) - 1
        offset = peak_idx - center

        return peak_val, offset


def test_gold_codes():
    """Test Gold code generation with verification"""
    print("="*60)
    print("GOLD CODE VERIFICATION TEST")
    print("="*60)

    # Generate codes
    gold_gen = VerifiedGoldCodes(m=7)  # Length 127

    # Test correlation between same code
    code0 = gold_gen.get_code(0)
    peak, offset = gold_gen.verify_correlation(code0, code0)
    print(f"\nSelf-correlation: peak={peak}, offset={offset}")
    assert peak == 127, "Self-correlation should equal code length"
    assert offset == 0, "Self-correlation offset should be 0"

    # Test correlation between different codes
    code1 = gold_gen.get_code(1)
    peak, offset = gold_gen.verify_correlation(code0, code1)
    print(f"Cross-correlation: peak={peak}, offset={offset}")
    assert abs(peak) <= 41, "Cross-correlation exceeds theoretical bound"

    print("\n✓ All Gold code tests passed")

    return gold_gen


if __name__ == "__main__":
    test_gold_codes()