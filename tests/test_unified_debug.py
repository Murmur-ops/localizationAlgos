#!/usr/bin/env python3
"""Debug unified system components"""

import numpy as np
from algorithms.bp_simple import SimpleBeliefPropagation
from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer

# Test parameters
n_sensors = 20
n_anchors = 4
noise = 0.05

print("Testing individual components...")
np.random.seed(42)
true_positions = {i: np.random.uniform(0, 1, 2) for i in range(n_sensors)}
anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])

# Test BP
bp = SimpleBeliefPropagation(
    n_sensors=n_sensors,
    n_anchors=n_anchors,
    communication_range=0.4,
    noise_factor=noise,
    max_iter=100
)
bp.generate_network(true_positions, anchor_positions)
bp_result = bp.run()

# Test MPS  
mps = ProperMPSAlgorithm(
    n_sensors=n_sensors,
    n_anchors=n_anchors,
    communication_range=0.4,
    noise_factor=noise,
    max_iter=500
)
mps.generate_network(true_positions, anchor_positions)
mps_result = mps.run()

# Calculate CRLB
analyzer = CRLBAnalyzer(n_sensors=n_sensors, n_anchors=n_anchors,
                       communication_range=0.4)
crlb = analyzer.compute_crlb(noise)

bp_eff = (crlb / bp_result['final_error']) * 100
mps_eff = (crlb / mps_result['final_error']) * 100

print(f"\nBaseline Results:")
print(f"  BP:  RMSE={bp_result['final_error']:.4f}, Efficiency={bp_eff:.1f}%")
print(f"  MPS: RMSE={mps_result['final_error']:.4f}, Efficiency={mps_eff:.1f}%")

# Now test hierarchical alone
from algorithms.hierarchical_processing import HierarchicalProcessor

hier = HierarchicalProcessor(
    n_sensors=n_sensors,
    n_anchors=n_anchors,
    communication_range=0.4,
    noise_factor=noise
)
hier.generate_network(true_positions, anchor_positions)
hier_result = hier.run(max_iter=50)
hier_eff = (crlb / hier_result['final_error']) * 100

print(f"  Hierarchical: RMSE={hier_result['final_error']:.4f}, Efficiency={hier_eff:.1f}%")

# The issue seems to be that hierarchical processing alone performs worse
# Let's implement a better integrated version
print("\nThe unified system needs better integration between components.")