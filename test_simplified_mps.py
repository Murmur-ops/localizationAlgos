#!/usr/bin/env python3
"""
Test the simplified MPS algorithm performance
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig

def test_simplified_mps():
    """Test simplified MPS with paper configuration"""
    
    print("="*60)
    print("TESTING SIMPLIFIED MPS ALGORITHM")
    print("="*60)
    
    # Paper configuration (Section 3)
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        scale=50.0,  # 50m x 50m network
        communication_range=0.3,  # 30% of scale = 15m
        noise_factor=0.05,  # 5% noise as in paper
        gamma=0.999,  # Paper value
        alpha=0.01,   # Very small step size for 50m scale
        max_iterations=500,
        tolerance=1e-5,
        dimension=2,
        seed=42
    )
    
    print(f"\nConfiguration (from paper):")
    print(f"  Network size: {config.scale}m × {config.scale}m")
    print(f"  Sensors: {config.n_sensors}")
    print(f"  Anchors: {config.n_anchors}")
    print(f"  Communication range: {config.communication_range * config.scale:.1f}m")
    print(f"  Noise: {config.noise_factor*100}%")
    print(f"  γ (gamma): {config.gamma}")
    print(f"  α (alpha): {config.alpha}")
    
    # Create algorithm instance
    mps = MPSAlgorithm(config)
    
    # Generate network
    print(f"\nGenerating network...")
    mps.generate_network()
    
    print(f"  True positions: {len(mps.true_positions)} nodes")
    print(f"  Distance measurements: {len(mps.distance_measurements)} pairs")
    
    # Get max distance from measurements
    max_distance = max(mps.distance_measurements.values()) if mps.distance_measurements else 0
    print(f"  Max measured distance: {max_distance:.2f}m")
    
    # Convert positions dict to array for analysis
    true_pos_array = np.array([mps.true_positions[i] for i in range(config.n_sensors)])
    network_diameter = np.max([np.linalg.norm(true_pos_array[i] - true_pos_array[j]) 
                               for i in range(len(true_pos_array)) 
                               for j in range(i+1, len(true_pos_array))])
    print(f"  Network diameter: {network_diameter:.2f}m")
    
    # Run algorithm
    print(f"\nRunning simplified MPS algorithm...")
    result = mps.run()
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final objective: {result['final_objective']:.6f}")
    print(f"RMSE: {result['final_rmse']:.4f}m")
    
    # Calculate relative error (as percentage of network diameter)
    relative_error = result['final_rmse'] / network_diameter
    print(f"Relative error: {relative_error:.2%} of network diameter")
    
    # Compare to paper's reported performance
    print(f"\nPaper reports: 5-10% relative error")
    print(f"Our result: {relative_error:.2%}")
    
    if relative_error > 0.10:
        print(f"⚠️  Performance is worse than paper!")
    else:
        print(f"✓ Performance matches paper!")
    
    # Show per-node errors
    if 'final_positions' in result and result['final_positions']:
        # Convert positions dict to array
        estimated_pos_array = np.array([result['final_positions'][i] for i in range(config.n_sensors)])
        errors = np.linalg.norm(estimated_pos_array - true_pos_array, axis=1)
        print(f"\nPer-node error statistics:")
        print(f"  Mean: {np.mean(errors):.4f}m")
        print(f"  Std: {np.std(errors):.4f}m")
        print(f"  Min: {np.min(errors):.4f}m")
        print(f"  Max: {np.max(errors):.4f}m")
        print(f"  Median: {np.median(errors):.4f}m")
    
    return result

if __name__ == "__main__":
    result = test_simplified_mps()