#!/usr/bin/env python3
"""
Test single-process MPS execution with YAML config
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import (
    MPSAlgorithm, create_network_data
)

def test_single_process():
    """Test single-process MPS execution"""
    print("Testing Single-Process MPS Execution")
    print("=" * 50)
    
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config("configs/default.yaml")
    
    # Override for quick test
    config['algorithm']['max_iterations'] = 50
    config['algorithm']['verbose'] = True
    
    # Convert to MPSConfig
    mps_config = loader.to_mps_config(config, distributed=False)
    
    print(f"\nConfiguration:")
    print(f"  Network: {mps_config.n_sensors} sensors, {mps_config.n_anchors} anchors")
    print(f"  Algorithm: gamma={mps_config.gamma}, alpha={mps_config.alpha}")
    print(f"  Max iterations: {mps_config.max_iterations}")
    
    # Create network
    print(f"\nGenerating network...")
    network = create_network_data(
        n_sensors=mps_config.n_sensors,
        n_anchors=mps_config.n_anchors,
        dimension=mps_config.dimension,
        communication_range=mps_config.communication_range,
        measurement_noise=config['measurements']['noise_factor'],
        carrier_phase=mps_config.carrier_phase_mode
    )
    print(f"  ✓ Network created with {len(network.distance_measurements)} measurements")
    
    # Run MPS algorithm
    print(f"\nRunning MPS algorithm...")
    try:
        mps = MPSAlgorithm(mps_config, network)
        results = mps.run()
        
        print(f"\n✓ Algorithm completed!")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Converged: {results['converged']}")
        print(f"  Final objective: {results['objectives'][-1]:.6f}")
        print(f"  Best error: {results['best_error']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_process()
    sys.exit(0 if success else 1)