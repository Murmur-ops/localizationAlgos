#!/usr/bin/env python3
"""
Test YAML configuration system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader

def test_config_loading():
    """Test basic configuration loading"""
    print("Testing YAML Configuration System")
    print("=" * 50)
    
    loader = ConfigLoader()
    
    # Test 1: Load default config
    print("\n1. Loading default.yaml...")
    try:
        config = loader.load_config("configs/default.yaml")
        print(f"   ✓ Loaded successfully")
        print(f"   - Network: {config['network']['n_sensors']} sensors, {config['network']['n_anchors']} anchors")
        print(f"   - Algorithm: gamma={config['algorithm']['gamma']}, alpha={config['algorithm']['alpha']}")
        print(f"   - MPI enabled: {config['mpi']['enable']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: Load MPI config with inheritance
    print("\n2. Loading mpi_small.yaml (with inheritance)...")
    try:
        config = loader.load_config("configs/mpi/mpi_small.yaml")
        print(f"   ✓ Loaded successfully")
        print(f"   - Network: {config['network']['n_sensors']} sensors (overridden)")
        print(f"   - MPI enabled: {config['mpi']['enable']} (overridden)")
        print(f"   - Gamma: {config['algorithm']['gamma']} (inherited)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 3: Test configuration validation
    print("\n3. Testing configuration validation...")
    try:
        loader.validate_schema(config)
        print(f"   ✓ Configuration valid")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
        return False
    
    # Test 4: Test parameter overrides
    print("\n4. Testing parameter overrides...")
    try:
        overrides = {'network.n_sensors': 100, 'algorithm.max_iterations': 500}
        config = loader.load_config("configs/default.yaml", overrides=overrides)
        print(f"   ✓ Overrides applied")
        print(f"   - n_sensors: {config['network']['n_sensors']} (overridden to 100)")
        print(f"   - max_iterations: {config['algorithm']['max_iterations']} (overridden to 500)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 5: Convert to MPSConfig
    print("\n5. Testing conversion to MPSConfig...")
    try:
        mps_config = loader.to_mps_config(config, distributed=False)
        print(f"   ✓ Converted to MPSConfig")
        print(f"   - Type: {type(mps_config).__name__}")
        print(f"   - n_sensors: {mps_config.n_sensors}")
        print(f"   - gamma: {mps_config.gamma}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 6: Test distributed config
    print("\n6. Testing distributed MPSConfig...")
    try:
        dist_config = loader.to_mps_config(config, distributed=True)
        print(f"   ✓ Converted to DistributedMPSConfig")
        print(f"   - Type: {type(dist_config).__name__}")
        print(f"   - async_communication: {dist_config.async_communication}")
        print(f"   - buffer_size_kb: {dist_config.buffer_size_kb}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All YAML configuration tests passed!")
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)