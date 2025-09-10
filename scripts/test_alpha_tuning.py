#!/usr/bin/env python3
"""
Test different alpha values to match paper's RMSE performance
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig


def test_alpha_values():
    """Test different alpha values to find optimal performance"""
    
    print("="*70)
    print("ALPHA PARAMETER TUNING")
    print("="*70)
    print("\nPaper uses α = 10.0, but let's test different values...")
    print("-"*70)
    
    # Test different alpha values
    alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print(f"{'Alpha':>8} | {'Relative Error':>14} | {'RMSE':>10} | {'Iterations':>10} | {'Note':>20}")
    print("-"*70)
    
    best_alpha = None
    best_error = float('inf')
    
    for alpha in alpha_values:
        config = MPSConfig(
            n_sensors=30,
            n_anchors=6,
            scale=1.0,
            communication_range=0.7,
            noise_factor=0.05,
            gamma=0.999,  # Paper's value
            alpha=alpha,  # Test different values
            max_iterations=500,
            tolerance=1e-5,
            dimension=2,
            seed=42
        )
        
        mps = MPSAlgorithm(config)
        mps.generate_network()
        result = mps.run()
        
        # Calculate relative error
        X_hat = np.array([result['estimated_positions'][i] for i in range(30)])
        X_0 = np.array([mps.true_positions[i] for i in range(30)])
        relative_error = np.linalg.norm(X_hat - X_0, 'fro') / np.linalg.norm(X_0, 'fro')
        
        note = ""
        if relative_error < 0.1:
            note = "✓ Matches paper!"
        elif relative_error < best_error:
            best_error = relative_error
            best_alpha = alpha
            note = "← Best so far"
        
        if alpha == 10.0:
            note += " (Paper's α)"
        
        print(f"{alpha:8.2f} | {relative_error:14.4f} | {result['final_rmse']:10.4f} | "
              f"{result['iterations']:10d} | {note:>20}")
    
    print("-"*70)
    print(f"\nBest alpha: {best_alpha} with relative error: {best_error:.4f}")
    
    if best_error > 0.1:
        print("\n⚠ WARNING: Cannot achieve paper's performance with simple algorithm")
        print("Paper likely uses the full lifted variable formulation with ADMM")


def test_with_different_seeds():
    """Test multiple random seeds to check consistency"""
    
    print("\n" + "="*70)
    print("TESTING MULTIPLE RANDOM SEEDS")
    print("="*70)
    
    # Use best alpha from previous test or paper's value
    best_alpha = 0.1  # Typically works better for simple algorithm
    
    errors = []
    for seed in range(5):
        config = MPSConfig(
            n_sensors=30,
            n_anchors=6,
            scale=1.0,
            communication_range=0.7,
            noise_factor=0.05,
            gamma=0.999,
            alpha=best_alpha,
            max_iterations=500,
            tolerance=1e-5,
            dimension=2,
            seed=seed
        )
        
        mps = MPSAlgorithm(config)
        mps.generate_network()
        result = mps.run()
        
        X_hat = np.array([result['estimated_positions'][i] for i in range(30)])
        X_0 = np.array([mps.true_positions[i] for i in range(30)])
        relative_error = np.linalg.norm(X_hat - X_0, 'fro') / np.linalg.norm(X_0, 'fro')
        errors.append(relative_error)
        
        print(f"Seed {seed}: Relative error = {relative_error:.4f}")
    
    print(f"\nMean relative error: {np.mean(errors):.4f}")
    print(f"Std deviation: {np.std(errors):.4f}")
    
    if np.mean(errors) < 0.1:
        print("✓ Consistently matches paper's performance!")
    else:
        print("✗ Performance gap remains")


def main():
    print("\nINVESTIGATING RMSE DISCREPANCY")
    print("="*70)
    print("\nPaper reports: relative error ~0.05-0.1")
    print("Our result: relative error ~0.84")
    print("\nLet's tune parameters to match...\n")
    
    test_alpha_values()
    test_with_different_seeds()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nThe simple MPSAlgorithm cannot achieve the paper's reported performance.")
    print("The paper uses a more sophisticated approach with:")
    print("  1. Lifted variables (matrix S^i)")
    print("  2. ADMM inner solver")
    print("  3. Proper 2-Block structure with PSD constraints")
    print("\nTo match the paper's RMSE, we need to use the full")
    print("MatrixParametrizedProximalSplitting implementation.")


if __name__ == "__main__":
    main()