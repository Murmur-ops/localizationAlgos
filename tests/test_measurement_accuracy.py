import sys
sys.path.append('.')
from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig, CarrierPhaseConfig
import numpy as np

# Configure for mm accuracy
config = MPSConfig(
    n_sensors=5,
    n_anchors=4,
    scale=10.0,
    communication_range=0.5,
    noise_factor=0.001,
    gamma=0.999,
    alpha=0.01,
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

# Check measurement accuracy
errors = []
for (i,j), measured in mps.distance_measurements.items():
    if i < j:
        true_dist = np.linalg.norm(mps.true_positions[i] - mps.true_positions[j])
        error = abs(measured - true_dist)
        errors.append(error)
        print(f"Pair ({i},{j}): true={true_dist:.6f}m, measured={measured:.6f}m, error={error*1000:.3f}mm")

print(f"\nMean error: {np.mean(errors)*1000:.3f}mm")
print(f"Max error: {np.max(errors)*1000:.3f}mm")
print(f"RMSE: {np.sqrt(np.mean(np.square(errors)))*1000:.3f}mm")
