#!/usr/bin/env python3
"""
Check if we're at the precision limit of the measurements.
"""

import numpy as np

# Our measurement noise
measurement_std = 0.01  # 1 cm standard deviation
measurement_var = measurement_std**2

# With ToA measurements at 1cm precision
# Position uncertainty follows from range uncertainty
# For trilateration, position error ≈ range_error * GDOP
# Typical GDOP is 1-3 for good geometry

print("=" * 70)
print("PRECISION LIMIT ANALYSIS")
print("=" * 70)

print(f"\nMeasurement noise: {measurement_std*100:.1f} cm std dev")
print(f"Expected position error (GDOP=1): {measurement_std*100:.1f} cm")
print(f"Expected position error (GDOP=2): {measurement_std*200:.1f} cm")
print(f"Expected position error (GDOP=3): {measurement_std*300:.1f} cm")

print(f"\nOur achieved RMS: 2.66 cm")
print(f"This corresponds to GDOP ≈ {2.66/1.0:.1f}")

print("\nConclusion:")
print("We're achieving 2.66 cm RMS with 1 cm measurement noise.")
print("This is reasonable and close to the theoretical limit.")
print("The system may be stuck near the noise floor.")

# Check if step size is the issue
print("\n" + "=" * 70)
print("STEP SIZE ANALYSIS")
print("=" * 70)

print("\nCurrent step size: 0.5")
print("If gradient is 17 and we move 0.5 * delta:")
print("  We're making conservative updates")
print("  This helps stability but slows convergence")

print("\nThe flat lines in convergence could be because:")
print("1. Step size too small (0.5) for the gradient magnitude")
print("2. Regularization too strong (1e-3) damping updates")
print("3. Already near the measurement noise floor (~2-3 cm)")