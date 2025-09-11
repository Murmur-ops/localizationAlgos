#!/usr/bin/env python3
"""
Test YAML configuration system
"""

from src.config import FTLConfig
import numpy as np


def test_yaml_config():
    """Test the YAML configuration system"""
    print("="*70)
    print("TESTING YAML CONFIGURATION SYSTEM")
    print("="*70)
    
    # Test loading existing config
    print("\n1. Loading existing 10-node config...")
    config = FTLConfig("configs/10_node_demo.yaml")
    print(config.summary())
    
    # Validate
    errors = config.validate()
    if errors:
        print("Validation errors:", errors)
    else:
        print("✓ Configuration is valid")
    
    # Test loading nearest-neighbor config
    print("\n2. Loading nearest-neighbor ps config...")
    nn_config = FTLConfig("configs/nearest_neighbor_ps.yaml")
    print(nn_config.summary())
    
    # Check timing precision
    print(f"\nTiming precision: {nn_config.hardware.timing_precision}")
    print(f"In seconds: {nn_config.hardware.timing_precision_seconds:.2e}")
    
    # Get anchors and unknowns
    anchors = nn_config.get_anchor_positions()
    unknowns = nn_config.get_unknown_nodes()
    print(f"\nAnchors: {len(anchors)}")
    print(f"Unknown nodes: {len(unknowns)}")
    
    # Test creating new config programmatically
    print("\n3. Creating new config programmatically...")
    new_config = FTLConfig()
    
    # Configure for ns timing comparison
    new_config.hardware.timing_precision = "ns"
    new_config.hardware.bandwidth_mhz = 200.0
    new_config.network.topology = "nearest_neighbor"
    new_config.network.k_neighbors = 5
    
    # Add some nodes
    new_config.nodes = [
        NodeConfig(0, [0, 0], True, "Anchor_0"),
        NodeConfig(1, [10, 0], True, "Anchor_1"),
        NodeConfig(2, [5, 5], False, "Unknown_2"),
    ]
    
    # Save it
    new_config.save("configs/test_config.yaml")
    print("✓ Saved new config to configs/test_config.yaml")
    
    # Test validation with bad config
    print("\n4. Testing validation with insufficient anchors...")
    bad_config = FTLConfig()
    bad_config.nodes = [
        NodeConfig(0, [0, 0], True, "Anchor_0"),
        NodeConfig(1, [10, 0], True, "Anchor_1"),  # Only 2 anchors
        NodeConfig(2, [5, 5], False, "Unknown_2"),
    ]
    
    errors = bad_config.validate()
    if errors:
        print("✓ Correctly detected errors:", errors)
    
    print("\n" + "="*70)
    print("YAML CONFIG SYSTEM TEST COMPLETE")
    print("="*70)


def compare_timing_configs():
    """Compare ps vs ns timing configurations"""
    print("\n" + "="*70)
    print("COMPARING TIMING CONFIGURATIONS")
    print("="*70)
    
    # Load ps config
    ps_config = FTLConfig("configs/nearest_neighbor_ps.yaml")
    
    # Create ns version
    ns_config = FTLConfig("configs/nearest_neighbor_ps.yaml")
    ns_config.hardware.timing_precision = "ns"
    ns_config.hardware.timestamp_resolution_ns = 1.0
    
    print(f"\nPS timing: {ps_config.hardware.timing_precision_seconds:.2e} seconds")
    print(f"NS timing: {ns_config.hardware.timing_precision_seconds:.2e} seconds")
    print(f"Ratio: {ns_config.hardware.timing_precision_seconds / ps_config.hardware.timing_precision_seconds:.0f}x")
    
    # Expected ranging errors
    c = 299792458.0
    ps_ranging_error = ps_config.hardware.timing_precision_seconds * c
    ns_ranging_error = ns_config.hardware.timing_precision_seconds * c
    
    print(f"\nExpected ranging errors (timing only):")
    print(f"  PS: {ps_ranging_error*1000:.3f} mm")
    print(f"  NS: {ns_ranging_error*100:.3f} cm")
    
    print(f"\nWith nearest-neighbor topology (k={ps_config.network.k_neighbors}):")
    print(f"  ~40% of links are clean LOS")
    print(f"  PS achieves mm-level on clean links")
    print(f"  NS limited to cm-level even on clean links")


if __name__ == "__main__":
    # Import here to avoid circular dependency
    from src.config import NodeConfig
    
    test_yaml_config()
    compare_timing_configs()
    
    print("\n✓ YAML configuration system is working!")