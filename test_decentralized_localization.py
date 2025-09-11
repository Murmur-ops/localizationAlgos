#!/usr/bin/env python3
"""
Test Decentralized vs Centralized Localization
Demonstrates that decentralized achieves similar accuracy to centralized
"""

import numpy as np
import matplotlib.pyplot as plt
from src.localization.decentralized_solver import (
    DecentralizedLocalizationSystem, 
    RangingMeasurement,
    create_measurements_from_distances
)
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_scenario(n_anchors: int = 3, n_unknowns: int = 2, room_size: float = 10.0):
    """
    Create a test scenario with anchors and unknown nodes
    
    Args:
        n_anchors: Number of anchor nodes
        n_unknowns: Number of unknown nodes
        room_size: Size of the room (meters)
        
    Returns:
        positions, anchor_indices, unknown_indices
    """
    np.random.seed(42)  # For reproducibility
    
    # Place anchors at corners/edges
    if n_anchors == 3:
        anchor_positions = np.array([
            [0, 0],
            [room_size, 0],
            [room_size/2, room_size]
        ])
    elif n_anchors == 4:
        anchor_positions = np.array([
            [0, 0],
            [room_size, 0],
            [room_size, room_size],
            [0, room_size]
        ])
    else:
        # Random anchor placement
        anchor_positions = np.random.uniform(0, room_size, (n_anchors, 2))
    
    # Place unknown nodes randomly
    unknown_positions = np.random.uniform(
        room_size * 0.2, 
        room_size * 0.8, 
        (n_unknowns, 2)
    )
    
    # Combine all positions
    all_positions = np.vstack([anchor_positions, unknown_positions])
    anchor_indices = list(range(n_anchors))
    unknown_indices = list(range(n_anchors, n_anchors + n_unknowns))
    
    return all_positions, anchor_indices, unknown_indices


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix"""
    n = len(positions)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances


def test_centralized(positions, anchor_indices, unknown_indices, measurements):
    """Test centralized localization"""
    logger.info("\n=== Testing Centralized Localization ===")
    
    # Convert to centralized format
    anchor_positions = {i: positions[i] for i in anchor_indices}
    
    # Initial guess for unknowns (center of room with noise)
    n_unknowns = len(unknown_indices)
    initial_guess = np.random.uniform(3, 7, n_unknowns * 2)
    
    # Convert measurements to MeasurementEdge format
    edges = []
    for m in measurements:
        edge = MeasurementEdge(
            node_i=m.from_node,
            node_j=m.to_node,
            distance=m.distance,
            quality=m.quality,
            variance=m.variance
        )
        edges.append(edge)
    
    # Solve
    localizer = RobustLocalizer(dimension=2)
    optimized_positions, info = localizer.solve(
        initial_guess, 
        edges, 
        anchor_positions
    )
    
    # Compute errors
    errors = []
    for idx, unknown_idx in enumerate(unknown_indices):
        est_pos = optimized_positions[idx*2:(idx+1)*2]
        true_pos = positions[unknown_idx]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        logger.info(f"Node {unknown_idx}: Error = {error:.3f}m")
    
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    logger.info(f"Centralized RMSE: {rmse:.3f}m")
    logger.info(f"Converged in {info['iterations']} iterations")
    
    return optimized_positions, rmse, info


def test_decentralized(positions, anchor_indices, unknown_indices, measurements):
    """Test decentralized localization"""
    logger.info("\n=== Testing Decentralized Localization ===")
    
    # Create decentralized system
    system = DecentralizedLocalizationSystem(dimension=2)
    
    # Add anchor nodes
    for idx in anchor_indices:
        system.add_node(idx, positions[idx], is_anchor=True)
    
    # Add unknown nodes with initial guess
    for idx in unknown_indices:
        initial_pos = np.random.uniform(3, 7, 2)  # Random initial guess
        system.add_node(idx, initial_pos, is_anchor=False)
    
    # Add measurements
    for m in measurements:
        system.add_measurement(m)
    
    # Run distributed algorithm
    final_positions, info = system.run(max_iterations=100, convergence_threshold=1e-4)
    
    # Compute errors
    errors = []
    for idx in unknown_indices:
        est_pos = final_positions[idx]
        true_pos = positions[idx]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        logger.info(f"Node {idx}: Error = {error:.3f}m")
    
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    logger.info(f"Decentralized RMSE: {rmse:.3f}m")
    logger.info(f"Converged in {info['iterations']} iterations")
    
    return final_positions, rmse, info


def visualize_results(positions, anchor_indices, unknown_indices,
                      centralized_positions, decentralized_positions,
                      centralized_rmse, decentralized_rmse):
    """Visualize localization results"""
    plt.figure(figsize=(12, 5))
    
    # Centralized results
    plt.subplot(1, 2, 1)
    
    # Plot anchors
    for idx in anchor_indices:
        plt.scatter(positions[idx, 0], positions[idx, 1], 
                   marker='^', s=200, c='red', label='Anchor' if idx == anchor_indices[0] else '')
    
    # Plot true positions of unknowns
    for idx in unknown_indices:
        plt.scatter(positions[idx, 0], positions[idx, 1],
                   marker='o', s=100, c='blue', label='True' if idx == unknown_indices[0] else '')
    
    # Plot estimated positions (centralized)
    for i, idx in enumerate(unknown_indices):
        est_pos = centralized_positions[i*2:(i+1)*2]
        plt.scatter(est_pos[0], est_pos[1],
                   marker='x', s=100, c='green', label='Estimated' if i == 0 else '')
        # Draw error line
        plt.plot([positions[idx, 0], est_pos[0]], 
                [positions[idx, 1], est_pos[1]], 
                'k--', alpha=0.3)
    
    plt.title(f'Centralized Localization\nRMSE: {centralized_rmse:.3f}m')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Decentralized results
    plt.subplot(1, 2, 2)
    
    # Plot anchors
    for idx in anchor_indices:
        plt.scatter(positions[idx, 0], positions[idx, 1],
                   marker='^', s=200, c='red', label='Anchor' if idx == anchor_indices[0] else '')
    
    # Plot true positions of unknowns
    for idx in unknown_indices:
        plt.scatter(positions[idx, 0], positions[idx, 1],
                   marker='o', s=100, c='blue', label='True' if idx == unknown_indices[0] else '')
    
    # Plot estimated positions (decentralized)
    for idx in unknown_indices:
        est_pos = decentralized_positions[idx]
        plt.scatter(est_pos[0], est_pos[1],
                   marker='x', s=100, c='green', 
                   label='Estimated' if idx == unknown_indices[0] else '')
        # Draw error line
        plt.plot([positions[idx, 0], est_pos[0]],
                [positions[idx, 1], est_pos[1]],
                'k--', alpha=0.3)
    
    plt.title(f'Decentralized Localization\nRMSE: {decentralized_rmse:.3f}m')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('decentralized_vs_centralized.png', dpi=150)
    plt.show()


def plot_convergence(centralized_info, decentralized_info):
    """Plot convergence curves"""
    plt.figure(figsize=(10, 5))
    
    # Cost convergence
    plt.subplot(1, 2, 1)
    if 'convergence_history' in centralized_info:
        plt.semilogy(centralized_info['convergence_history'], 
                    'b-', label='Centralized', linewidth=2)
    if 'cost_history' in decentralized_info:
        plt.semilogy(decentralized_info['cost_history'],
                    'r--', label='Decentralized', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost (log scale)')
    plt.title('Cost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Position convergence (for decentralized)
    plt.subplot(1, 2, 2)
    if 'position_history' in decentralized_info:
        position_changes = []
        for i in range(1, len(decentralized_info['position_history'])):
            max_change = 0
            curr = decentralized_info['position_history'][i]
            prev = decentralized_info['position_history'][i-1]
            for node_id in curr:
                if node_id >= 3:  # Unknown nodes only
                    change = np.linalg.norm(curr[node_id] - prev[node_id])
                    max_change = max(max_change, change)
            position_changes.append(max_change)
        
        plt.semilogy(position_changes, 'g-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Max Position Change (m, log scale)')
        plt.title('Decentralized Position Convergence')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_curves.png', dpi=150)
    plt.show()


def main():
    """Run complete comparison test"""
    logger.info("=" * 60)
    logger.info("Testing Decentralized vs Centralized Localization")
    logger.info("=" * 60)
    
    # Create scenario
    n_anchors = 3
    n_unknowns = 2
    room_size = 10.0
    noise_std = 0.05  # 5cm measurement noise
    
    positions, anchor_indices, unknown_indices = create_scenario(
        n_anchors, n_unknowns, room_size
    )
    
    logger.info(f"\nScenario: {n_anchors} anchors, {n_unknowns} unknown nodes")
    logger.info(f"Room size: {room_size}x{room_size} meters")
    logger.info(f"Measurement noise: {noise_std*100:.0f}% ({noise_std}m std)")
    
    # Generate measurements
    distance_matrix = compute_distance_matrix(positions)
    measurements = create_measurements_from_distances(
        distance_matrix, 
        noise_std=noise_std,
        quality_model='snr'
    )
    
    # Test centralized
    centralized_positions, centralized_rmse, centralized_info = test_centralized(
        positions, anchor_indices, unknown_indices, measurements
    )
    
    # Test decentralized
    decentralized_positions, decentralized_rmse, decentralized_info = test_decentralized(
        positions, anchor_indices, unknown_indices, measurements
    )
    
    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Centralized RMSE:   {centralized_rmse:.3f}m")
    logger.info(f"Decentralized RMSE: {decentralized_rmse:.3f}m")
    logger.info(f"Difference:         {abs(centralized_rmse - decentralized_rmse):.3f}m")
    
    improvement = (centralized_rmse - decentralized_rmse) / centralized_rmse * 100
    if improvement > 0:
        logger.info(f"Decentralized is {improvement:.1f}% better!")
    else:
        logger.info(f"Centralized is {-improvement:.1f}% better")
    
    # Visualize
    visualize_results(
        positions, anchor_indices, unknown_indices,
        centralized_positions, decentralized_positions,
        centralized_rmse, decentralized_rmse
    )
    
    plot_convergence(centralized_info, decentralized_info)
    
    logger.info("\n✅ Test complete! Results saved to PNG files.")


if __name__ == "__main__":
    main()