#!/usr/bin/env python3
"""
Simple test for debugging MPS algorithm
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData,
    create_network_data
)

# Create small test network
print("Creating test network...")
network_data = create_network_data(
    n_sensors=5,
    n_anchors=2,
    dimension=2,
    communication_range=0.5,
    measurement_noise=0.001,
    carrier_phase=True
)

# Simple configuration
config = MPSConfig(
    n_sensors=5,
    n_anchors=2,
    dimension=2,
    gamma=0.999,
    alpha=10.0,
    max_iterations=100,
    tolerance=1e-6,
    verbose=True,
    use_2block=True,  # Test 2-block design
    parallel_proximal=True,
    adaptive_alpha=True,
    carrier_phase_mode=True
)

print("Initializing MPS algorithm...")
mps = MatrixParametrizedProximalSplitting(config, network_data)

print("Running algorithm...")
try:
    results = mps.run()
    print(f"Success! Iterations: {results['iterations']}")
    print(f"Final objective: {results['best_objective']:.6f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()