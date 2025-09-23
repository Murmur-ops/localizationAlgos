#!/usr/bin/env python3
"""Direct test of simulate_rf_measurement"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ftl.clocks import ClockState
from run_unified_ftl import simulate_rf_measurement

# Simple test case
true_positions = np.array([
    [0, 0],
    [10, 0]
])

clock_states = {
    0: ClockState(bias=0, drift=0, cfo=0),
    1: ClockState(bias=0, drift=0, cfo=0)
}

rf_config = {
    'signal': {
        'carrier_freq': '6.5e9',
        'bandwidth': '499.2e6',
        'sample_rate': '1e9',
        'burst_duration': '1e-6',
        'prf': '124.8e6',
        'snr_db': '50.0'
    },
    'channel': {
        'environment': 'indoor',
        'path_loss_exponent': '2.0',
        'shadowing_std_db': '0.1'
    },
    'simulation': {
        'max_range': '50.0',
        'los_probability': '1.0',
        'enable_multipath': False
    }
}

print("Testing simulate_rf_measurement directly")
print("=" * 50)

# Enable debug in the function by modifying it temporarily
import run_unified_ftl
original_func = run_unified_ftl.simulate_rf_measurement

def debug_simulate_rf_measurement(i, j, true_positions, clock_states, rf_config):
    """Wrapper with debug enabled"""
    result = original_func(i, j, true_positions, clock_states, rf_config)
    return result

result = debug_simulate_rf_measurement(0, 1, true_positions, clock_states, rf_config)

print(f"\nResults:")
print(f"  True distance: {np.linalg.norm(true_positions[1] - true_positions[0]):.3f} m")
print(f"  Measured range: {result['range_m']:.3f} m")
print(f"  Error: {result['range_m'] - 10.0:.3f} m")

# Now test with clock bias
clock_states[1] = ClockState(bias=1e-9, drift=0, cfo=0)  # 1ns = 0.3m

result2 = debug_simulate_rf_measurement(0, 1, true_positions, clock_states, rf_config)

print(f"\nWith 1ns clock bias:")
print(f"  Expected range: 10.3 m")
print(f"  Measured range: {result2['range_m']:.3f} m")
print(f"  Error: {result2['range_m'] - 10.3:.3f} m")