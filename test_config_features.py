#!/usr/bin/env python3
"""
Test YAML configuration inheritance and overrides
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader

def test_inheritance():
    """Test configuration inheritance"""
    print("Testing Configuration Inheritance")
    print("=" * 50)
    
    loader = ConfigLoader()
    
    # Test 1: Base config
    print("\n1. Loading base config (default.yaml)...")
    base = loader.load_config("configs/default.yaml")
    print(f"   n_sensors: {base['network']['n_sensors']}")
    print(f"   MPI enabled: {base['mpi']['enable']}")
    print(f"   Output dir: {base['output']['output_dir']}")
    
    # Test 2: Child config with inheritance
    print("\n2. Loading child config (mpi_medium.yaml)...")
    child = loader.load_config("configs/mpi/mpi_medium.yaml")
    print(f"   n_sensors: {child['network']['n_sensors']} (overridden)")
    print(f"   MPI enabled: {child['mpi']['enable']} (overridden)")
    print(f"   gamma: {child['algorithm']['gamma']} (inherited)")
    print(f"   Output dir: {child['output']['output_dir']} (overridden)")
    
    return True

def test_overrides():
    """Test parameter overrides"""
    print("\nTesting Parameter Overrides")
    print("=" * 50)
    
    loader = ConfigLoader()
    
    # Test command-line style overrides
    print("\n1. Testing dot-notation overrides...")
    overrides = {
        'network.n_sensors': 100,
        'algorithm.max_iterations': 200,
        'mpi.enable': True,
        'algorithm.gamma': 0.95
    }
    
    config = loader.load_config("configs/default.yaml", overrides=overrides)
    
    print(f"   n_sensors: {config['network']['n_sensors']} (expected: 100)")
    print(f"   max_iterations: {config['algorithm']['max_iterations']} (expected: 200)")
    print(f"   MPI enabled: {config['mpi']['enable']} (expected: True)")
    print(f"   gamma: {config['algorithm']['gamma']} (expected: 0.95)")
    
    # Verify overrides worked
    assert config['network']['n_sensors'] == 100
    assert config['algorithm']['max_iterations'] == 200
    assert config['mpi']['enable'] == True
    assert config['algorithm']['gamma'] == 0.95
    
    print("   ✓ All overrides applied correctly")
    
    return True

def test_env_variables():
    """Test environment variable substitution"""
    print("\nTesting Environment Variables")
    print("=" * 50)
    
    # Set test environment variables
    os.environ['MPS_TEST_SENSORS'] = '75'
    os.environ['MPS_TEST_OUTPUT'] = '/tmp/test_output/'
    
    # Create test config with env vars
    test_config = """
network:
  n_sensors: ${MPS_TEST_SENSORS:30}
  n_anchors: ${MPS_TEST_ANCHORS:5}  # Will use default
  
output:
  output_dir: ${MPS_TEST_OUTPUT:results/}
"""
    
    # Write temporary config
    test_path = Path("test_env_config.yaml")
    test_path.write_text(test_config)
    
    try:
        loader = ConfigLoader()
        config = loader.load_config(test_path)
        
        print(f"   MPS_TEST_SENSORS set to: 75")
        print(f"   n_sensors: {config['network']['n_sensors']} (from env)")
        print(f"   n_anchors: {config['network']['n_anchors']} (default)")
        print(f"   output_dir: {config['output']['output_dir']} (from env)")
        
        assert config['network']['n_sensors'] == '75'
        assert config['network']['n_anchors'] == 5
        assert config['output']['output_dir'] == '/tmp/test_output/'
        
        print("   ✓ Environment variables processed correctly")
        return True
        
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()
        del os.environ['MPS_TEST_SENSORS']
        del os.environ['MPS_TEST_OUTPUT']

def test_multiple_configs():
    """Test loading and merging multiple configs"""
    print("\nTesting Multiple Config Merging")
    print("=" * 50)
    
    loader = ConfigLoader()
    
    # Load multiple configs
    configs = [
        "configs/default.yaml",
        "configs/mpi/mpi_small.yaml"
    ]
    
    print(f"\n1. Loading and merging: {configs}")
    merged = loader.load_multiple_configs(configs)
    
    print(f"   n_sensors: {merged['network']['n_sensors']} (from mpi_small)")
    print(f"   MPI enabled: {merged['mpi']['enable']} (from mpi_small)")
    print(f"   scale: {merged['network']['scale']} (from default)")
    
    # Second config should override first
    assert merged['network']['n_sensors'] == 20  # From mpi_small
    assert merged['mpi']['enable'] == True  # From mpi_small
    
    print("   ✓ Configs merged correctly (later overrides earlier)")
    
    return True

def main():
    """Run all configuration tests"""
    print("Configuration Feature Tests")
    print("=" * 60)
    
    tests = [
        test_inheritance,
        test_overrides,
        test_env_variables,
        test_multiple_configs
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append((test.__name__, success))
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print(f"\n✅ All configuration tests passed!")
    else:
        print(f"\n⚠️ Some tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())