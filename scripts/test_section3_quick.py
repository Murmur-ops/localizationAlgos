#!/usr/bin/env python3
"""
Quick test of Section 3 implementation with fewer trials
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from section3_numerical_experiments import Section3Experiments, ExperimentConfig

def main():
    """Run a quick test with reduced parameters"""
    
    # Create config with fewer trials for testing
    config = ExperimentConfig(
        n_sensors=30,
        n_anchors=6,
        n_trials=2,  # Just 2 trials instead of 50
        max_iterations=100  # Fewer iterations
    )
    
    print("="*70)
    print("QUICK TEST OF SECTION 3 IMPLEMENTATION")
    print("="*70)
    print(f"Running with {config.n_trials} trials, {config.max_iterations} iterations")
    print()
    
    experiments = Section3Experiments(config)
    
    # Test network generation
    print("Testing network generation...")
    network = experiments.generate_network(seed=42)
    print(f"✓ Generated network with {config.n_sensors} sensors, {config.n_anchors} anchors")
    print(f"  Adjacency shape: {network['adjacency'].shape}")
    print(f"  Distance measurements: {len(network['distance_measurements'])} pairs")
    print(f"  Anchor connections: {sum(len(v) for v in network['anchor_distances'].values())} total")
    
    # Test MPS algorithm
    print("\nTesting MPS algorithm...")
    mps_result = experiments.run_mps_algorithm(network, warm_start=False)
    if 'relative_error' in mps_result and len(mps_result['relative_error']) > 0:
        print(f"✓ MPS completed: final error = {mps_result['relative_error'][-1]:.4f}")
    else:
        print("✓ MPS completed")
    
    # Test ADMM baseline
    print("\nTesting ADMM baseline...")
    admm_result = experiments.run_admm_baseline(network, warm_start=False)
    if 'relative_error' in admm_result and len(admm_result['relative_error']) > 0:
        print(f"✓ ADMM completed: final error = {admm_result['relative_error'][-1]:.4f}")
    else:
        print("✓ ADMM completed")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if (len(mps_result.get('relative_error', [])) > 0 and 
        len(admm_result.get('relative_error', [])) > 0):
        mps_error = mps_result['relative_error'][-1]
        admm_error = admm_result['relative_error'][-1]
        
        print(f"Final MPS error:  {mps_error:.4f}")
        print(f"Final ADMM error: {admm_error:.4f}")
        
        if mps_error < admm_error:
            improvement = (1 - mps_error/admm_error) * 100
            print(f"\n✓ MPS outperforms ADMM by {improvement:.1f}%")
        else:
            print("\n⚠ ADMM performed better in this trial")
    
    print("\nQuick test completed successfully!")
    print("Full experiments are running in the background...")

if __name__ == "__main__":
    main()