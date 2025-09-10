import sys
sys.path.append('.')
from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig, CarrierPhaseConfig
import numpy as np

config = MPSConfig(
    n_sensors=5,
    n_anchors=4,
    scale=10.0,
    communication_range=0.5,
    noise_factor=0.001,
    gamma=0.999,
    alpha=0.5,
    max_iterations=10,
    tolerance=1e-6,
    seed=42,
    carrier_phase=CarrierPhaseConfig(
        enable=True,
        frequency_ghz=2.4,
        phase_noise_milliradians=1.0,
        coarse_time_accuracy_ns=0.05
    )
)

mps = MPSAlgorithm(config)
mps.generate_network()

print(f"Scale: {mps.scale}m")
print(f"Working in normalized 0-1 space")
print(f"\nTrue positions (normalized):")
for i in range(3):
    print(f"  Sensor {i}: {mps.true_positions[i]}")

print(f"\nMeasurement errors (normalized):")
errors = []
for (i,j), measured in mps.distance_measurements.items():
    if i < j and i < 3:
        true_dist = np.linalg.norm(mps.true_positions[i] - mps.true_positions[j])
        error = abs(measured - true_dist)
        print(f"  Pair ({i},{j}): true={true_dist:.6f}, measured={measured:.6f}, error={error:.6f}")
        errors.append(error)

print(f"\nMean error (normalized): {np.mean(errors):.6f}")
print(f"Mean error (physical): {np.mean(errors) * mps.scale * 1000:.3f}mm")
