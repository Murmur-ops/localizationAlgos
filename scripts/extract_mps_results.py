#!/usr/bin/env python3
"""
Extract and display MPS vs ADMM accuracy results
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from section3_numerical_experiments import Section3Experiments, ExperimentConfig

def run_comparison_trials(n_trials=10):
    """Run a comparison with fewer trials to get quick results"""
    
    config = ExperimentConfig(
        n_sensors=30,
        n_anchors=6,
        n_trials=n_trials,
        max_iterations=500  # Full iterations to see convergence
    )
    
    print("="*80)
    print("MPS vs ADMM ACCURACY COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - {config.n_sensors} sensors, {config.n_anchors} anchors")
    print(f"  - Communication range: {config.communication_range}")
    print(f"  - Noise factor: {config.noise_factor} (5%)")
    print(f"  - MPS: γ={config.mps_gamma}, α={config.mps_alpha}")
    print(f"  - ADMM: α={config.admm_alpha}")
    print(f"  - Running {n_trials} trials")
    print()
    
    experiments = Section3Experiments(config)
    
    # Store results
    mps_cold_errors = []
    admm_cold_errors = []
    mps_warm_errors = []
    admm_warm_errors = []
    
    print("Trial Results:")
    print("-"*80)
    print("Trial | MPS Cold | ADMM Cold | MPS Warm | ADMM Warm | MPS Better?")
    print("-"*80)
    
    for trial in range(n_trials):
        # Generate network
        network = experiments.generate_network(seed=trial)
        
        # Cold start comparison
        mps_cold = experiments.run_mps_algorithm(network, warm_start=False)
        admm_cold = experiments.run_admm_baseline(network, warm_start=False)
        
        # Warm start comparison
        mps_warm = experiments.run_mps_algorithm(network, warm_start=True)
        admm_warm = experiments.run_admm_baseline(network, warm_start=True)
        
        # Extract final errors
        if len(mps_cold.get('relative_error', [])) > 0:
            mps_cold_err = mps_cold['relative_error'][-1]
            mps_cold_errors.append(mps_cold_err)
        else:
            mps_cold_err = np.nan
            
        if len(admm_cold.get('relative_error', [])) > 0:
            admm_cold_err = admm_cold['relative_error'][-1]
            admm_cold_errors.append(admm_cold_err)
        else:
            admm_cold_err = np.nan
            
        if len(mps_warm.get('relative_error', [])) > 0:
            mps_warm_err = mps_warm['relative_error'][-1]
            mps_warm_errors.append(mps_warm_err)
        else:
            mps_warm_err = np.nan
            
        if len(admm_warm.get('relative_error', [])) > 0:
            admm_warm_err = admm_warm['relative_error'][-1]
            admm_warm_errors.append(admm_warm_err)
        else:
            admm_warm_err = np.nan
        
        # Check if MPS is better
        mps_better = "✓" if mps_cold_err < admm_cold_err else "✗"
        
        print(f"{trial+1:5} | {mps_cold_err:8.4f} | {admm_cold_err:9.4f} | "
              f"{mps_warm_err:8.4f} | {admm_warm_err:9.4f} | {mps_better:^11}")
        
        # Show convergence details for first trial
        if trial == 0 and len(mps_cold['relative_error']) > 1:
            print("\n  Convergence details (Trial 1):")
            iterations = mps_cold['iterations']
            mps_errors = mps_cold['relative_error']
            admm_errors = admm_cold['relative_error']
            
            # Show errors at key iterations
            key_iters = [0, 10, 20, 50, 100, 200, -1]
            print("  Iteration | MPS Error | ADMM Error")
            print("  ----------|-----------|------------")
            for idx in key_iters:
                if idx == -1:
                    iter_num = iterations[-1] if len(iterations) > 0 else 500
                    mps_e = mps_errors[-1] if len(mps_errors) > 0 else np.nan
                    admm_e = admm_errors[-1] if len(admm_errors) > 0 else np.nan
                elif idx < len(iterations):
                    iter_num = iterations[idx]
                    mps_e = mps_errors[idx] if idx < len(mps_errors) else np.nan
                    admm_e = admm_errors[idx] if idx < len(admm_errors) else np.nan
                else:
                    continue
                print(f"  {iter_num:9} | {mps_e:9.4f} | {admm_e:10.4f}")
            print()
    
    print("-"*80)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if len(mps_cold_errors) > 0 and len(admm_cold_errors) > 0:
        print("\nCold Start:")
        print(f"  MPS  - Mean: {np.mean(mps_cold_errors):.4f}, "
              f"Median: {np.median(mps_cold_errors):.4f}, "
              f"Std: {np.std(mps_cold_errors):.4f}")
        print(f"  ADMM - Mean: {np.mean(admm_cold_errors):.4f}, "
              f"Median: {np.median(admm_cold_errors):.4f}, "
              f"Std: {np.std(admm_cold_errors):.4f}")
        
        # Performance comparison
        mps_better_count = sum(1 for m, a in zip(mps_cold_errors, admm_cold_errors) if m < a)
        pct_better = (mps_better_count / len(mps_cold_errors)) * 100
        avg_improvement = (1 - np.mean(mps_cold_errors) / np.mean(admm_cold_errors)) * 100
        
        print(f"\n  MPS outperforms ADMM in {mps_better_count}/{len(mps_cold_errors)} trials ({pct_better:.1f}%)")
        if avg_improvement > 0:
            print(f"  Average improvement: {avg_improvement:.1f}% lower error")
        else:
            print(f"  ADMM performs {-avg_improvement:.1f}% better on average")
    
    if len(mps_warm_errors) > 0 and len(admm_warm_errors) > 0:
        print("\nWarm Start:")
        print(f"  MPS  - Mean: {np.mean(mps_warm_errors):.4f}, "
              f"Median: {np.median(mps_warm_errors):.4f}, "
              f"Std: {np.std(mps_warm_errors):.4f}")
        print(f"  ADMM - Mean: {np.mean(admm_warm_errors):.4f}, "
              f"Median: {np.median(admm_warm_errors):.4f}, "
              f"Std: {np.std(admm_warm_errors):.4f}")
        
        mps_better_warm = sum(1 for m, a in zip(mps_warm_errors, admm_warm_errors) if m < a)
        pct_better_warm = (mps_better_warm / len(mps_warm_errors)) * 100
        avg_improvement_warm = (1 - np.mean(mps_warm_errors) / np.mean(admm_warm_errors)) * 100
        
        print(f"\n  MPS outperforms ADMM in {mps_better_warm}/{len(mps_warm_errors)} trials ({pct_better_warm:.1f}%)")
        if avg_improvement_warm > 0:
            print(f"  Average improvement: {avg_improvement_warm:.1f}% lower error")
        else:
            print(f"  ADMM performs {-avg_improvement_warm:.1f}% better on average")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if len(mps_cold_errors) > 0 and len(admm_cold_errors) > 0:
        mps_final = np.mean(mps_cold_errors)
        admm_final = np.mean(admm_cold_errors)
        
        print(f"\n1. Final Error Comparison (Cold Start):")
        print(f"   - MPS:  {mps_final:.1%} relative error")
        print(f"   - ADMM: {admm_final:.1%} relative error")
        
        if mps_final < 0.1:
            print(f"\n✓ MPS achieves sub-10% error as expected from the paper")
        elif mps_final < 0.2:
            print(f"\n⚠ MPS error higher than paper's claim but still reasonable")
        else:
            print(f"\n✗ MPS error significantly higher than expected")
        
        if admm_final > 0.35:
            print(f"✓ ADMM shows ~40% error as reported in the paper")
        
        print(f"\n2. Paper Claims vs Our Results:")
        print(f"   Paper: MPS achieves errors less than half of ADMM")
        ratio = mps_final / admm_final
        if ratio < 0.5:
            print(f"   ✓ Confirmed: MPS error is {ratio:.1%} of ADMM error")
        else:
            print(f"   ✗ Not achieved: MPS error is {ratio:.1%} of ADMM error")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Run with 10 trials for quick results
    run_comparison_trials(n_trials=10)