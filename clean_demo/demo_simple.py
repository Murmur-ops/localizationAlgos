#!/usr/bin/env python3
"""
Simple FTL Localization Demo
A minimal example showing distributed localization with realistic RF physics
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Measurement:
    """Ranging measurement between two nodes"""
    node_i: int
    node_j: int
    distance: float
    snr_db: float


class SimpleLocalizer:
    """Simplified robust localization solver"""

    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.max_iterations = 50
        self.convergence_threshold = 1e-4

    def solve(self, initial_positions: np.ndarray,
              measurements: List[Measurement],
              anchors: Dict[int, np.ndarray]) -> Tuple[np.ndarray, dict]:
        """
        Solve localization using Levenberg-Marquardt optimization

        Args:
            initial_positions: Initial guess for unknown positions [x1,y1,x2,y2,...]
            measurements: List of ranging measurements
            anchors: Dictionary of anchor positions {id: [x, y]}

        Returns:
            Optimized positions and convergence info
        """
        positions = initial_positions.copy()
        n_unknowns = len(positions) // self.dimension

        # Map unknown node IDs (assumes they start after anchors)
        unknown_ids = list(range(len(anchors), len(anchors) + n_unknowns))

        for iteration in range(self.max_iterations):
            old_positions = positions.copy()

            # Build Jacobian and residual
            J = []
            residuals = []
            weights = []

            for m in measurements:
                # Check if measurement involves an unknown
                if m.node_i in unknown_ids:
                    ui_idx = unknown_ids.index(m.node_i)
                    ui_pos = positions[ui_idx*2:(ui_idx+1)*2]
                else:
                    ui_pos = anchors.get(m.node_i)
                    ui_idx = None

                if m.node_j in unknown_ids:
                    uj_idx = unknown_ids.index(m.node_j)
                    uj_pos = positions[uj_idx*2:(uj_idx+1)*2]
                else:
                    uj_pos = anchors.get(m.node_j)
                    uj_idx = None

                if ui_pos is None or uj_pos is None:
                    continue

                # Compute estimated distance
                est_dist = np.linalg.norm(ui_pos - uj_pos)
                if est_dist < 1e-6:
                    est_dist = 1e-6

                # Residual: measured - estimated
                residual = m.distance - est_dist
                residuals.append(residual)

                # Weight based on SNR
                weight = min(1.0, m.snr_db / 20.0) if m.snr_db > 0 else 0.1
                weights.append(weight)

                # Jacobian row
                j_row = np.zeros(len(positions))

                if ui_idx is not None:
                    # Derivative w.r.t unknown i position
                    grad_i = -(ui_pos - uj_pos) / est_dist
                    j_row[ui_idx*2:(ui_idx+1)*2] = grad_i

                if uj_idx is not None:
                    # Derivative w.r.t unknown j position
                    grad_j = (ui_pos - uj_pos) / est_dist
                    j_row[uj_idx*2:(uj_idx+1)*2] = grad_j

                J.append(j_row)

            if not J:
                break

            # Convert to arrays
            J = np.array(J)
            r = np.array(residuals)
            W = np.diag(weights)

            # Levenberg-Marquardt update
            lambda_lm = 0.01
            JTW = J.T @ W
            H = JTW @ J + lambda_lm * np.eye(len(positions))
            g = -JTW @ r

            try:
                delta = np.linalg.solve(H, g)
                positions += delta
            except np.linalg.LinAlgError:
                break

            # Check convergence
            if np.linalg.norm(delta) < self.convergence_threshold:
                break

        info = {
            'iterations': iteration + 1,
            'final_error': np.linalg.norm(residuals) if residuals else 0
        }

        return positions, info


def generate_ranging_measurements(
    nodes: Dict[int, np.ndarray],
    config: dict
) -> List[Measurement]:
    """
    Generate realistic ranging measurements with noise

    Args:
        nodes: All node positions {id: [x, y]}
        config: Configuration dictionary

    Returns:
        List of measurements between nodes
    """
    measurements = []

    # Channel parameters
    bandwidth_hz = float(config['channel']['bandwidth_hz'])
    snr_nominal_db = float(config['channel']['snr_db'])

    # Resolution floor from bandwidth
    resolution_m = 3e8 / (2 * bandwidth_hz)

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            # True distance
            true_dist = np.linalg.norm(nodes[i] - nodes[j])

            # Skip if too far (out of range)
            if true_dist > config['channel']['max_range_m']:
                continue

            # Path loss (simplified)
            if true_dist > 0.1:
                path_loss_db = 40 + 20 * np.log10(max(1.0, true_dist))
            else:
                path_loss_db = 40
            snr_db = snr_nominal_db - path_loss_db + 40  # Add 40dB to make it more realistic

            # Skip if SNR too low
            if snr_db < float(config['channel']['min_snr_db']):
                continue

            # Add noise based on SNR and bandwidth
            noise_std = resolution_m / np.sqrt(10**(snr_db/10))
            measured_dist = true_dist + np.random.normal(0, noise_std)

            # Ensure positive
            measured_dist = max(0.1, measured_dist)

            measurements.append(Measurement(
                node_i=i,
                node_j=j,
                distance=measured_dist,
                snr_db=snr_db
            ))

    return measurements


def visualize_results(
    true_positions: Dict[int, np.ndarray],
    estimated_positions: Dict[int, np.ndarray],
    anchors: Dict[int, np.ndarray],
    measurements: List[Measurement],
    config: dict
):
    """Create visualization of localization results"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Network topology and measurements
    ax = axes[0]

    # Draw measurements as edges
    for m in measurements:
        if m.node_i in true_positions and m.node_j in true_positions:
            pos_i = true_positions[m.node_i]
            pos_j = true_positions[m.node_j]

            # Color by SNR
            color = 'g' if m.snr_db > 10 else ('y' if m.snr_db > 0 else 'r')
            alpha = min(1.0, max(0.1, m.snr_db / 20))

            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                   color=color, alpha=alpha, linewidth=0.5)

    # Plot anchors
    for aid, pos in anchors.items():
        ax.scatter(pos[0], pos[1], s=200, c='blue', marker='^',
                  edgecolors='black', linewidth=2, zorder=5)
        ax.text(pos[0], pos[1]-1, f'A{aid}', ha='center', fontsize=10)

    # Plot true unknown positions
    for uid, pos in true_positions.items():
        if uid not in anchors:
            ax.scatter(pos[0], pos[1], s=100, c='green', marker='o',
                      edgecolors='black', linewidth=1, zorder=4)

    ax.set_xlim(-2, config['area']['size_m']+2)
    ax.set_ylim(-2, config['area']['size_m']+2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Network Topology')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Right: Localization results
    ax = axes[1]

    # Plot anchors
    for aid, pos in anchors.items():
        ax.scatter(pos[0], pos[1], s=200, c='blue', marker='^',
                  edgecolors='black', linewidth=2, zorder=5, label='Anchor' if aid == 0 else '')

    # Plot true vs estimated positions
    errors = []
    for uid, true_pos in true_positions.items():
        if uid not in anchors:
            # True position
            ax.scatter(true_pos[0], true_pos[1], s=100, c='green', marker='o',
                      alpha=0.5, label='True' if uid == len(anchors) else '')

            # Estimated position
            if uid in estimated_positions:
                est_pos = estimated_positions[uid]
                ax.scatter(est_pos[0], est_pos[1], s=100, c='red', marker='x',
                          linewidth=2, label='Estimated' if uid == len(anchors) else '')

                # Error line
                ax.plot([true_pos[0], est_pos[0]], [true_pos[1], est_pos[1]],
                       'k--', alpha=0.3)

                # Calculate error
                error = np.linalg.norm(true_pos - est_pos)
                errors.append(error)

                # Annotate with error
                mid_x = (true_pos[0] + est_pos[0]) / 2
                mid_y = (true_pos[1] + est_pos[1]) / 2
                ax.text(mid_x, mid_y, f'{error:.2f}m', fontsize=8, alpha=0.7)

    # Add RMSE to title
    if errors:
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        ax.set_title(f'Localization Results (RMSE: {rmse:.2f}m)')
    else:
        ax.set_title('Localization Results')

    ax.set_xlim(-2, config['area']['size_m']+2)
    ax.set_ylim(-2, config['area']['size_m']+2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    plt.suptitle('FTL Localization Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def run_demo(config_path: str):
    """
    Run the localization demo

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("FTL LOCALIZATION DEMO")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Area: {config['area']['size_m']}×{config['area']['size_m']}m")
    print(f"Nodes: {config['nodes']['total']} ({config['nodes']['anchors']} anchors)")
    print(f"Channel: {float(config['channel']['bandwidth_hz'])/1e6:.0f}MHz, {config['channel']['snr_db']}dB SNR")
    print()

    # Generate node positions
    np.random.seed(config['seed'])
    area_size = config['area']['size_m']
    n_anchors = config['nodes']['anchors']
    n_unknowns = config['nodes']['total'] - n_anchors

    nodes = {}
    anchors = {}

    # Place anchors at corners (optimal for small areas)
    if n_anchors >= 4:
        anchor_positions = [
            [0, 0], [area_size, 0],
            [area_size, area_size], [0, area_size]
        ]
        for i in range(min(4, n_anchors)):
            nodes[i] = np.array(anchor_positions[i])
            anchors[i] = nodes[i]

        # Additional anchors randomly placed
        for i in range(4, n_anchors):
            nodes[i] = np.random.uniform(0, area_size, 2)
            anchors[i] = nodes[i]
    else:
        # Random anchor placement if less than 4
        for i in range(n_anchors):
            nodes[i] = np.random.uniform(0, area_size, 2)
            anchors[i] = nodes[i]

    # Place unknown nodes randomly
    for i in range(n_unknowns):
        nodes[n_anchors + i] = np.random.uniform(0, area_size, 2)

    # Generate measurements
    print("Generating measurements...")
    measurements = generate_ranging_measurements(nodes, config)
    print(f"Generated {len(measurements)} measurements")

    # Filter measurements to only those involving unknowns
    unknown_ids = list(range(n_anchors, n_anchors + n_unknowns))
    filtered_measurements = [
        m for m in measurements
        if m.node_i in unknown_ids or m.node_j in unknown_ids
    ]
    print(f"Using {len(filtered_measurements)} measurements for localization")

    # Initialize solver
    solver = SimpleLocalizer(dimension=2)

    # Initial guess: center of area
    center = area_size / 2
    initial_positions = np.ones(n_unknowns * 2) * center

    # Solve
    print("\nRunning localization...")
    estimated_positions, info = solver.solve(
        initial_positions,
        filtered_measurements,
        anchors
    )

    print(f"Converged in {info['iterations']} iterations")

    # Reconstruct estimated positions dictionary
    estimated_dict = {}
    for i in range(n_unknowns):
        uid = n_anchors + i
        estimated_dict[uid] = estimated_positions[i*2:(i+1)*2]

    # Calculate errors
    errors = []
    for uid in unknown_ids:
        true_pos = nodes[uid]
        est_pos = estimated_dict[uid]
        error = np.linalg.norm(true_pos - est_pos)
        errors.append(error)
        print(f"Node {uid}: error = {error:.3f}m")

    rmse = np.sqrt(np.mean(np.array(errors)**2))
    print(f"\nRMSE: {rmse:.3f}m")
    print(f"Mean error: {np.mean(errors):.3f}m")
    print(f"Max error: {np.max(errors):.3f}m")

    # Visualize
    fig = visualize_results(nodes, estimated_dict, anchors, measurements, config)
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to demo_results.png")
    plt.show()

    return rmse


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "configs/simple.yaml"

    run_demo(config_file)