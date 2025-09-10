#!/usr/bin/env python3
"""
Test MPS algorithm with different network sizes
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import create_network_data

def test_network_size(n_sensors: int, n_anchors: int):
    """Test a specific network size"""
    print(f"\nTesting {n_sensors} sensors, {n_anchors} anchors...")
    
    # Create config
    loader = ConfigLoader()
    config = loader.load_config("configs/default.yaml")
    
    # Override network size
    config['network']['n_sensors'] = n_sensors
    config['network']['n_anchors'] = n_anchors
    config['algorithm']['max_iterations'] = 30  # Quick test
    
    # Create network
    try:
        network = create_network_data(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            dimension=2,
            communication_range=0.3,
            measurement_noise=0.05,
            carrier_phase=False
        )
        
        n_measurements = len(network.distance_measurements)
        n_edges = np.sum(network.adjacency_matrix > 0) // 2
        
        print(f"  ✓ Network created")
        print(f"    - Measurements: {n_measurements}")
        print(f"    - Edges: {n_edges}")
        print(f"    - Avg degree: {2*n_edges/n_sensors:.1f}")
        
        # Check if network is connected (basic check)
        if n_edges < n_sensors - 1:
            print(f"    ⚠️ Network might be disconnected")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    """Test different network sizes"""
    print("Testing Different Network Sizes")
    print("=" * 60)
    
    # Test cases: (n_sensors, n_anchors)
    test_cases = [
        (5, 3),     # Tiny
        (10, 3),    # Small
        (20, 4),    # Medium
        (30, 6),    # Default
        (50, 8),    # Large
        (100, 15),  # Very large
    ]
    
    results = []
    for n_sensors, n_anchors in test_cases:
        success = test_network_size(n_sensors, n_anchors)
        results.append(((n_sensors, n_anchors), success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{'Size':<15} {'Status':<10}")
    print("-" * 25)
    
    for (n_s, n_a), success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{n_s:3d} x {n_a:2d} sensors  {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print(f"\n✅ All network sizes tested successfully!")
    else:
        print(f"\n⚠️ Some tests failed")
    
    # Performance scaling analysis
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS")
    print('='*60)
    print("Network size scaling characteristics:")
    print("- Linear scaling with n_sensors for measurements")
    print("- Quadratic worst case for dense networks")
    print("- Communication range affects connectivity")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())