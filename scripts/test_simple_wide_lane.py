#!/usr/bin/env python3
"""
Simple test to verify wide-lane ambiguity resolution works correctly
"""

import numpy as np

def test_wide_lane_simple():
    """Test wide-lane with known values"""
    
    # Configuration
    c = 299792458  # Speed of light
    f1 = 2.4e9  # L1 frequency
    f2 = 1.9e9  # L2 frequency
    
    lambda1 = c / f1  # 12.5cm
    lambda2 = c / f2  # 15.8cm
    lambda_wl = c / (f1 - f2)  # 60cm
    
    print(f"L1 wavelength: {lambda1*100:.1f}cm")
    print(f"L2 wavelength: {lambda2*100:.1f}cm")
    print(f"Wide-lane wavelength: {lambda_wl*100:.1f}cm")
    print()
    
    # Test case: true distance = 5.0m
    true_distance = 5.0
    
    # True integer ambiguities
    n1_true = int(true_distance / lambda1)  # 40 cycles at 12.5cm
    n2_true = int(true_distance / lambda2)  # 31 cycles at 15.8cm
    n_wl_true = n1_true - n2_true  # 9 cycles
    
    print(f"True distance: {true_distance:.3f}m")
    print(f"True N1: {n1_true}")
    print(f"True N2: {n2_true}")
    print(f"True N_WL: {n_wl_true}")
    print()
    
    # Fractional parts
    frac1 = (true_distance / lambda1) - n1_true
    frac2 = (true_distance / lambda2) - n2_true
    
    # Phase measurements (fractional part only)
    phi1 = frac1 * 2 * np.pi
    phi2 = frac2 * 2 * np.pi
    
    print(f"Phase L1: {phi1:.3f} rad ({frac1:.3f} cycles)")
    print(f"Phase L2: {phi2:.3f} rad ({frac2:.3f} cycles)")
    print()
    
    # Coarse measurement with error
    coarse_error = 0.20  # 20cm error
    coarse_distance = true_distance + coarse_error
    print(f"Coarse distance: {coarse_distance:.3f}m (error: {coarse_error*100:.0f}cm)")
    print()
    
    # Step 1: Resolve wide-lane using Melbourne-Wübbena
    # MW = (f1*φ1 - f2*φ2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)
    # For simplicity, assume P1 ≈ P2 ≈ coarse_distance
    
    # Wide-lane phase combination (in cycles)
    phi_wl_cycles = (f1 * frac1 - f2 * frac2) / (f1 - f2)
    
    # Narrow-lane code combination (in meters)
    p_nl = coarse_distance  # Simplified
    p_nl_cycles = p_nl / lambda_wl
    
    # MW observable (should be close to integer)
    mw = n_wl_true + phi_wl_cycles - p_nl_cycles
    n_wl_resolved = round(mw)
    
    print("Wide-lane resolution:")
    print(f"  MW observable: {mw:.3f}")
    print(f"  Resolved N_WL: {n_wl_resolved}")
    print(f"  Correct: {n_wl_resolved == n_wl_true}")
    print()
    
    # Step 2: Resolve L1 using wide-lane constraint
    # We know N1 - N2 = N_WL
    # From coarse: N1 ≈ coarse_distance/lambda1 - phi1_cycles
    
    n1_estimate = round(coarse_distance / lambda1 - frac1)
    
    # Search for best N1 that satisfies wide-lane constraint
    best_n1 = None
    best_error = float('inf')
    
    for n1 in range(n1_estimate - 5, n1_estimate + 6):
        n2 = n1 - n_wl_resolved
        
        # Calculate distances
        d1 = (n1 + frac1) * lambda1
        d2 = (n2 + frac2) * lambda2
        
        # Check consistency
        error = abs(d1 - d2) + abs(d1 - coarse_distance)
        
        if error < best_error:
            best_error = error
            best_n1 = n1
    
    best_n2 = best_n1 - n_wl_resolved
    
    print("L1/L2 resolution:")
    print(f"  Resolved N1: {best_n1} (true: {n1_true})")
    print(f"  Resolved N2: {best_n2} (true: {n2_true})")
    print()
    
    # Final distance
    final_distance = (best_n1 + frac1) * lambda1
    error_mm = abs(final_distance - true_distance) * 1000
    
    print(f"Final distance: {final_distance:.6f}m")
    print(f"Error: {error_mm:.2f}mm")
    print()
    
    # Check if we would have failed with single frequency
    n1_single = round(coarse_distance / lambda1 - frac1)
    single_distance = (n1_single + frac1) * lambda1
    single_error_mm = abs(single_distance - true_distance) * 1000
    
    print(f"Single-frequency would give:")
    print(f"  N1: {n1_single} (true: {n1_true})")
    print(f"  Distance: {single_distance:.6f}m")
    print(f"  Error: {single_error_mm:.2f}mm")
    print()
    
    if error_mm < 15 and single_error_mm > 100:
        print("✓ Wide-lane succeeds where single-frequency fails!")
    elif error_mm < 15:
        print("✓ Wide-lane achieves millimeter accuracy!")
    else:
        print("✗ Need to debug further")

if __name__ == "__main__":
    test_wide_lane_simple()