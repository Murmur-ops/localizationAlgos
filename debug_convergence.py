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
    alpha=0.1,
    max_iterations=100,
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

# Initialize near-perfect
state = mps.initialize_state()
for i in range(config.n_sensors):
    # Start very close to true positions
    state.positions[i] = mps.true_positions[i] + np.random.randn(2) * 0.001
    state.X[i] = state.positions[i]
    state.X[i + config.n_sensors] = state.positions[i]
    state.Y[i] = state.positions[i]
    state.Y[i + config.n_sensors] = state.positions[i]

print("Initial RMSE:", mps.compute_rmse(state), "m")

# Run a few iterations manually
for iter in range(10):
    X_old = state.X.copy()
    
    # Apply proximal operator
    state.X = mps.prox_f(state)
    
    # Consensus
    state.Y = mps.Z_matrix @ state.X
    
    # Dual update
    state.U = state.U + config.alpha * (state.X - state.Y)
    
    # Extract positions
    for i in range(config.n_sensors):
        state.positions[i] = (state.Y[i] + state.Y[i + config.n_sensors]) / 2
    
    rmse = mps.compute_rmse(state)
    print(f"Iter {iter+1}: RMSE = {rmse:.6f}m = {rmse*1000:.3f}mm")
    
    if rmse > 10:
        print("DIVERGED!")
        break
