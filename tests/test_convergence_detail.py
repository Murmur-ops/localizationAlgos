import sys
sys.path.append('.')
from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig, CarrierPhaseConfig
import numpy as np

config = MPSConfig(
    n_sensors=3,
    n_anchors=3,
    scale=10.0,
    communication_range=0.8,
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

# Check what alpha is being used
print(f"Config alpha: {config.alpha}")
print(f"Sensor alpha: {config.alpha / 10} = {config.alpha/10}")
print(f"Anchor alpha: {config.alpha / 5} = {config.alpha/5}")

# Initial state
state = mps.initialize_state()
init_rmse = mps.compute_rmse(state)
print(f"\nInitial RMSE (normalized): {init_rmse:.6f}")
print(f"Initial RMSE (physical): {init_rmse * mps.scale * 1000:.3f}mm")

# Run one iteration
X_old = state.X.copy()
state.X = mps.prox_f(state)
state.Y = mps.Z_matrix @ state.X
state.U = state.U + config.alpha * (state.X - state.Y)

for i in range(config.n_sensors):
    state.positions[i] = (state.Y[i] + state.Y[i + config.n_sensors]) / 2

rmse = mps.compute_rmse(state)
print(f"\nAfter 1 iteration:")
print(f"  RMSE (normalized): {rmse:.6f}")
print(f"  RMSE (physical): {rmse * mps.scale * 1000:.3f}mm")
print(f"  Position change: {np.linalg.norm(state.X - X_old):.6f}")
