#!/usr/bin/env python3
"""
Fixed consensus algorithm with proper scaling for zero-noise convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class FixedConsensusConfig:
    """Configuration with proper scaling"""
    max_iterations: int = 200
    gradient_tol: float = 1e-10

    # Step size and damping scaled for whitening
    measurement_std: float = 1e-2  # 1cm measurement noise (realistic)
    base_step_size: float = 0.1    # Base step size (unwhitened)
    base_damping: float = 1e-4     # Base LM damping (unwhitened)

    # Consensus parameters
    consensus_gain: float = 0.1    # Reduced for stability
    use_metropolis: bool = True    # Use Metropolis weights for doubly-stochastic

    @property
    def whitening_scale(self):
        """Whitening scales by 1/σ"""
        return 1.0 / self.measurement_std

    @property
    def step_size(self):
        """Step size scaled by whitening factor squared"""
        return self.base_step_size / (self.whitening_scale ** 2)

    @property
    def damping(self):
        """LM damping scaled by whitening factor squared"""
        return self.base_damping * (self.whitening_scale ** 2)


class FixedConsensusNode:
    """Node with fixed numerical scaling"""

    def __init__(self, node_id: int, initial_state: np.ndarray, is_anchor: bool, config: FixedConsensusConfig):
        self.id = node_id
        self.state = initial_state.copy()
        self.is_anchor = is_anchor
        self.config = config
        self.neighbors = {}
        self.measurements = []

    def compute_local_gradient(self, all_states: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient and Hessian from measurements"""
        n = 5  # State dimension
        H = np.zeros((n, n))
        g = np.zeros(n)

        for meas in self.measurements:
            i, j = meas['i'], meas['j']
            range_m = meas['range']

            # Get states
            if i == self.id:
                xi = self.state
                xj = all_states.get(j, None)
            else:
                xi = all_states.get(i, None)
                xj = self.state

            if xi is None or xj is None:
                continue

            # Compute residual and Jacobian (unwhitened first)
            pi, pj = xi[:2], xj[:2]
            dist = np.linalg.norm(pi - pj)

            if dist < 1e-10:
                continue

            # Residual (measured - predicted)
            predicted = dist
            residual = range_m - predicted

            # Unit vector
            u = (pj - pi) / dist

            # Jacobian (unwhitened)
            if i == self.id:
                J = np.zeros(n)
                J[0] = u[0]
                J[1] = u[1]
            else:
                J = np.zeros(n)
                J[0] = -u[0]
                J[1] = -u[1]

            # Apply whitening
            std = self.config.measurement_std
            r_wh = residual / std
            J_wh = J / std

            # Add to normal equations
            H += np.outer(J_wh, J_wh)
            g += J_wh * r_wh

        return H, g

    def add_consensus_penalty(self, H: np.ndarray, g: np.ndarray, neighbor_states: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Add consensus term with proper Metropolis weights"""
        if self.is_anchor:
            return H, g

        n_neighbors = len(neighbor_states)
        if n_neighbors == 0:
            return H, g

        mu = self.config.consensus_gain

        if self.config.use_metropolis:
            # Metropolis weights for doubly-stochastic matrix
            degree_i = len(self.neighbors)
            for neighbor_id, neighbor_state in neighbor_states.items():
                degree_j = len(self.neighbors)  # Simplified: assume similar degrees
                w_ij = 1.0 / (1 + max(degree_i, degree_j))

                # Consensus gradient: μ * w_ij * (x_i - x_j)
                g += mu * w_ij * (self.state - neighbor_state)
                H += mu * w_ij * np.eye(5)
        else:
            # Simple averaging
            g += mu * n_neighbors * self.state
            for neighbor_state in neighbor_states.values():
                g -= mu * neighbor_state
            H += mu * n_neighbors * np.eye(5)

        return H, g

    def update_state(self, H: np.ndarray, g: np.ndarray):
        """Update state with damped Gauss-Newton"""
        if self.is_anchor:
            return

        # Add Levenberg-Marquardt damping
        lambda_lm = self.config.damping
        H_damped = H + lambda_lm * np.diag(np.maximum(np.diag(H), 1e-6))

        try:
            # Solve for step
            delta = np.linalg.solve(H_damped, g)

            # Apply step with scaled step size
            self.state = self.state - self.config.step_size * delta
        except np.linalg.LinAlgError:
            # Fallback to gradient descent
            self.state = self.state - self.config.step_size * g / (np.linalg.norm(g) + 1e-10)


def run_fixed_consensus(n_nodes=10, n_anchors=4):
    """Run consensus with fixed numerical issues"""

    # Generate network
    area_size = 20.0
    positions = np.zeros((n_nodes, 2))

    # Place anchors at corners
    positions[0] = [0, 0]
    positions[1] = [area_size, 0]
    positions[2] = [area_size, area_size]
    positions[3] = [0, area_size]

    # Place unknowns in grid
    idx = n_anchors
    grid_size = int(np.sqrt(n_nodes - n_anchors))
    spacing = area_size / (grid_size + 1)
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n_nodes:
                break
            positions[idx] = [spacing * (i+1), spacing * (j+1)]
            idx += 1

    # Create configuration
    config = FixedConsensusConfig(
        measurement_std=0.01,  # 1cm (realistic)
        base_step_size=0.5,
        base_damping=1e-3,
        consensus_gain=0.05,
        use_metropolis=True
    )

    print(f"Configuration:")
    print(f"  Measurement std: {config.measurement_std} m")
    print(f"  Whitening scale: {config.whitening_scale}")
    print(f"  Adjusted step size: {config.step_size:.2e}")
    print(f"  Adjusted damping: {config.damping:.2e}")

    # Create nodes
    nodes = []
    for i in range(n_nodes):
        is_anchor = i < n_anchors

        # Initialize with small error
        initial_state = np.zeros(5)
        if is_anchor:
            initial_state[:2] = positions[i]
        else:
            initial_state[:2] = positions[i] + np.random.normal(0, 1.0, 2)

        node = FixedConsensusNode(i, initial_state, is_anchor, config)
        nodes.append(node)

    # Create measurements (perfect distances)
    measurements = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 30.0:  # Within range
                meas = {'i': i, 'j': j, 'range': dist}
                measurements.append(meas)
                nodes[i].measurements.append(meas)
                nodes[j].measurements.append(meas)
                nodes[i].neighbors[j] = True
                nodes[j].neighbors[i] = True

    print(f"\nNetwork: {n_nodes} nodes ({n_anchors} anchors), {len(measurements)} measurements")

    # Track convergence
    position_rmse_history = []

    for iter in range(config.max_iterations):
        # Get all current states
        all_states = {node.id: node.state for node in nodes}

        # Each node computes gradient
        for node in nodes:
            if node.is_anchor:
                continue

            # Local gradient from measurements
            H, g = node.compute_local_gradient(all_states)

            # Get neighbor states
            neighbor_states = {nid: all_states[nid] for nid in node.neighbors if nid in all_states}

            # Add consensus penalty
            H, g = node.add_consensus_penalty(H, g, neighbor_states)

            # Update state
            node.update_state(H, g)

        # Compute RMSE
        errors = []
        for i in range(n_anchors, n_nodes):
            est_pos = nodes[i].state[:2]
            true_pos = positions[i]
            errors.append(np.linalg.norm(est_pos - true_pos))

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        position_rmse_history.append(rmse)

        if iter % 20 == 0:
            print(f"Iter {iter}: RMSE = {rmse:.3e} m")

        if rmse < 1e-10:
            print(f"Converged at iteration {iter}")
            break

    return position_rmse_history


# Run test
np.random.seed(42)
rmse_history = run_fixed_consensus()

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(rmse_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Position RMSE (m)')
plt.title('Fixed Consensus Convergence (Zero Noise)')
plt.grid(True, alpha=0.3)
plt.axhline(y=1e-3, color='r', linestyle='--', label='1mm')
plt.axhline(y=1e-6, color='g', linestyle='--', label='1μm')
plt.axhline(y=1e-9, color='m', linestyle='--', label='1nm')
plt.legend()
plt.savefig('fixed_consensus_convergence.png')
plt.show()

# Analysis
if len(rmse_history) > 10:
    print(f"\n=== Results ===")
    print(f"Final RMSE: {rmse_history[-1]:.3e} m")
    print(f"Iterations to 1mm: {np.argmax(np.array(rmse_history) < 1e-3)}")
    print(f"Iterations to 1μm: {np.argmax(np.array(rmse_history) < 1e-6)}")

    # Check if exponential
    mid = len(rmse_history) // 2
    if rmse_history[mid] > 0 and rmse_history[0] > 0:
        rate = np.log(rmse_history[mid] / rmse_history[0]) / mid
        print(f"Convergence rate: {rate:.3f} (should be negative for exponential decay)")