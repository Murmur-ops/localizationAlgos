#!/usr/bin/env python3
"""Quick parameter test to verify algorithm behavior."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    create_network_data
)

def quick_test():
    """Quick test with different parameter combinations."""
    
    print("Quick Parameter Test")
    print("=" * 50)
    
    # Create small test network
    network_data = create_network_data(
        n_sensors=5,
        n_anchors=2,
        dimension=2,
        communication_range=0.7,
        measurement_noise=0.01
    )
    
    # Test configurations
    test_configs = [
        {'gamma': 0.9, 'alpha': 10.0, 'name': 'Conservative'},
        {'gamma': 0.95, 'alpha': 10.0, 'name': 'Moderate'},
        {'gamma': 0.99, 'alpha': 10.0, 'name': 'Aggressive'},
        {'gamma': 0.999, 'alpha': 10.0, 'name': 'Paper'},
    ]
    
    for test in test_configs:
        print(f"\nTesting {test['name']}: γ={test['gamma']}, α={test['alpha']}")
        
        config = MPSConfig(
            n_sensors=5,
            n_anchors=2,
            dimension=2,
            gamma=test['gamma'],
            alpha=test['alpha'],
            max_iterations=50,
            tolerance=1e-6,
            communication_range=0.7,
            verbose=False,
            early_stopping=False,
            admm_iterations=50,
            admm_tolerance=1e-6,
            admm_rho=1.0,
            warm_start=True,
            use_2block=True,
            parallel_proximal=False,
            adaptive_alpha=False
        )
        
        try:
            mps = MatrixParametrizedProximalSplitting(config, network_data)
            
            # Run a few iterations
            errors = []
            for k in range(10):
                stats = mps.run_iteration(k)
                errors.append(stats['position_error'])
            
            print(f"  Errors: {[f'{e:.3f}' for e in errors]}")
            
            # Check if improving
            if errors[-1] < errors[0]:
                print(f"  ✓ Improving (start: {errors[0]:.3f}, end: {errors[-1]:.3f})")
            else:
                print(f"  ✗ Not improving (start: {errors[0]:.3f}, end: {errors[-1]:.3f})")
                
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete")

if __name__ == "__main__":
    quick_test()