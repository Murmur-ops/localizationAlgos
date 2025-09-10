#!/usr/bin/env python3
"""
Test MPS algorithm with different YAML configurations
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import create_network_data
from src.core.mps_core.algorithm_sdp import matrix_parametrized_algorithm_sdp

def test_configuration(config_path: str, description: str):
    """Test a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Config: {config_path}")
    print('='*60)
    
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config(config_path)
    
    # Override for faster testing
    if 'high_accuracy' not in config_path and 'benchmark' not in config_path:
        config['algorithm']['max_iterations'] = 50
    
    print(f"Network: {config['network']['n_sensors']} sensors, {config['network']['n_anchors']} anchors")
    print(f"Noise: {config['measurements']['noise_factor']*100:.1f}%")
    print(f"Max iterations: {config['algorithm']['max_iterations']}")
    
    # Create network
    network = create_network_data(
        n_sensors=config['network']['n_sensors'],
        n_anchors=config['network']['n_anchors'],
        dimension=config['network']['dimension'],
        communication_range=config['network']['communication_range'],
        measurement_noise=config['measurements']['noise_factor'],
        carrier_phase=config['measurements'].get('carrier_phase', False)
    )
    
    # Run algorithm
    start_time = time.time()
    try:
        results = matrix_parametrized_algorithm_sdp(
            network.distance_measurements,
            network.anchor_positions,
            network.adjacency_matrix,
            n_sensors=config['network']['n_sensors'],
            d=config['network']['dimension'],
            gamma=config['algorithm']['gamma'],
            alpha=config['algorithm']['alpha'],
            max_iterations=config['algorithm']['max_iterations'],
            epsilon_convergence=config['algorithm']['tolerance'],
            admm_max_iterations=config['admm']['iterations'],
            admm_epsilon=config['admm']['tolerance'],
            rho=config['admm']['rho'],
            verbose=False,
            use_admm_warm_start=config['admm']['warm_start'],
            use_sinkhorn=True,
            use_2block=config['algorithm'].get('use_2block', True),
            true_positions=network.true_positions
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Completed in {elapsed:.2f}s")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Converged: {results['converged']}")
        print(f"  Relative error: {results['relative_error']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def main():
    """Run tests on different configurations"""
    print("Testing MPS Algorithm with YAML Configurations")
    print("=" * 60)
    
    configs_to_test = [
        ("configs/default.yaml", "Default configuration"),
        ("configs/mpi/mpi_small.yaml", "Small MPI configuration"),
        ("configs/fast_convergence.yaml", "Fast convergence"),
        ("configs/noisy_measurements.yaml", "Noisy measurements"),
    ]
    
    results = []
    for config_path, description in configs_to_test:
        success = test_configuration(config_path, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {desc}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n⚠️ Some tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())