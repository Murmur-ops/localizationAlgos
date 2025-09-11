#!/usr/bin/env python3
"""
Test TRUE Decentralized Localization
Nodes only use local information and communicate with direct neighbors
"""

import numpy as np
import matplotlib.pyplot as plt
from src.localization.true_decentralized import TrueDecentralizedSystem
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_network_scenario(n_anchors: int = 3, n_unknowns: int = 7, room_size: float = 10.0):
    """
    Create a network scenario with realistic connectivity
    
    Args:
        n_anchors: Number of anchor nodes
        n_unknowns: Number of unknown nodes  
        room_size: Size of deployment area
        
    Returns:
        positions, anchor_ids, unknown_ids, edges
    """
    np.random.seed(42)
    
    # Place anchors at strategic positions
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
        angles = np.linspace(0, 2*np.pi, n_anchors, endpoint=False)
        anchor_positions = room_size/2 * np.column_stack([
            1 + 0.8*np.cos(angles),
            1 + 0.8*np.sin(angles)
        ])
    
    # Place unknown nodes
    unknown_positions = np.random.uniform(
        room_size * 0.1,
        room_size * 0.9,
        (n_unknowns, 2)
    )
    
    # Combine all positions
    all_positions = np.vstack([anchor_positions, unknown_positions])
    
    # Create edges based on communication range
    max_range = room_size * 0.6  # Communication range
    edges = []
    
    for i in range(len(all_positions)):
        for j in range(i + 1, len(all_positions)):
            dist = np.linalg.norm(all_positions[i] - all_positions[j])
            if dist <= max_range:
                # Add noise to measurement
                noise = np.random.normal(0, 0.05)  # 5cm std
                measured_dist = dist + noise
                
                # Quality based on distance (path loss)
                quality = max(0.1, 1.0 - (dist / max_range) ** 2)
                variance = (0.05 * (1 + dist/10)) ** 2
                
                edges.append({
                    'node_i': i,
                    'node_j': j, 
                    'distance': measured_dist,
                    'true_distance': dist,
                    'variance': variance,
                    'quality': quality
                })
    
    return all_positions, list(range(n_anchors)), list(range(n_anchors, len(all_positions))), edges


def test_centralized(positions, anchor_ids, unknown_ids, edges):
    """Test centralized localization for comparison"""
    logger.info("\n=== Centralized Localization (Baseline) ===")
    
    # Setup
    anchor_positions = {i: positions[i] for i in anchor_ids}
    n_unknowns = len(unknown_ids)
    initial_guess = np.random.uniform(3, 7, n_unknowns * 2)
    
    # Convert edges to measurements
    measurements = []
    for edge in edges:
        measurements.append(MeasurementEdge(
            node_i=edge['node_i'],
            node_j=edge['node_j'],
            distance=edge['distance'],
            quality=edge['quality'],
            variance=edge['variance']
        ))
    
    # Solve
    localizer = RobustLocalizer(dimension=2)
    optimized_positions, info = localizer.solve(
        initial_guess,
        measurements,
        anchor_positions
    )
    
    # Compute errors
    errors = []
    results = {}
    for idx, unknown_id in enumerate(unknown_ids):
        est_pos = optimized_positions[idx*2:(idx+1)*2]
        true_pos = positions[unknown_id]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        results[unknown_id] = est_pos
        logger.info(f"  Node {unknown_id}: error = {error:.3f}m")
    
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    logger.info(f"Centralized RMSE: {rmse:.3f}m")
    logger.info(f"Iterations: {info['iterations']}")
    
    return results, rmse, info


def test_true_decentralized(positions, anchor_ids, unknown_ids, edges):
    """Test truly decentralized localization"""
    logger.info("\n=== True Decentralized Localization ===")
    logger.info("(Nodes only use LOCAL information)")
    
    # Create system
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add nodes
    for anchor_id in anchor_ids:
        system.add_node(anchor_id, positions[anchor_id], is_anchor=True)
    
    for unknown_id in unknown_ids:
        # Random initial guess
        initial_pos = np.random.uniform(3, 7, 2)
        system.add_node(unknown_id, initial_pos, is_anchor=False)
    
    # Add edges (measurements)
    for edge in edges:
        system.add_edge(
            edge['node_i'],
            edge['node_j'],
            edge['distance'],
            edge['variance'],
            edge['quality']
        )
    
    # Check connectivity
    logger.info("\nNetwork Topology:")
    for node_id in sorted(system.topology.keys()):
        neighbors = sorted(list(system.topology[node_id]))
        node_type = "Anchor" if node_id in anchor_ids else "Unknown"
        logger.info(f"  Node {node_id} ({node_type}): neighbors = {neighbors}")
    
    # Run distributed algorithm
    final_positions, info = system.run(max_iterations=200, convergence_threshold=1e-3)
    
    # Compute errors
    errors = []
    for unknown_id in unknown_ids:
        est_pos = final_positions[unknown_id]
        true_pos = positions[unknown_id]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        logger.info(f"  Node {unknown_id}: error = {error:.3f}m")
    
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    logger.info(f"Decentralized RMSE: {rmse:.3f}m")
    logger.info(f"Iterations: {info['iterations']}")
    
    return final_positions, rmse, info


def visualize_network_and_results(positions, anchor_ids, unknown_ids, edges,
                                  centralized_results, decentralized_results,
                                  cent_rmse, decent_rmse):
    """Visualize network topology and localization results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Network Topology
    ax = axes[0]
    
    # Draw edges
    for edge in edges:
        i, j = edge['node_i'], edge['node_j']
        ax.plot([positions[i, 0], positions[j, 0]],
               [positions[i, 1], positions[j, 1]],
               'k-', alpha=0.2, linewidth=0.5)
    
    # Draw nodes
    for anchor_id in anchor_ids:
        ax.scatter(positions[anchor_id, 0], positions[anchor_id, 1],
                  marker='^', s=200, c='red', label='Anchor' if anchor_id == anchor_ids[0] else '')
    
    for unknown_id in unknown_ids:
        ax.scatter(positions[unknown_id, 0], positions[unknown_id, 1],
                  marker='o', s=100, c='blue', label='Unknown' if unknown_id == unknown_ids[0] else '')
        ax.text(positions[unknown_id, 0], positions[unknown_id, 1] + 0.3,
               str(unknown_id), ha='center', fontsize=8)
    
    ax.set_title('Network Topology\n(edges = measurements)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Centralized Results
    ax = axes[1]
    
    # Anchors
    for anchor_id in anchor_ids:
        ax.scatter(positions[anchor_id, 0], positions[anchor_id, 1],
                  marker='^', s=200, c='red')
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        # True position
        ax.scatter(positions[unknown_id, 0], positions[unknown_id, 1],
                  marker='o', s=100, c='blue', alpha=0.5)
        
        # Estimated position
        if unknown_id in centralized_results:
            est_pos = centralized_results[unknown_id]
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=100, c='green')
            
            # Error line
            ax.plot([positions[unknown_id, 0], est_pos[0]],
                   [positions[unknown_id, 1], est_pos[1]],
                   'r--', alpha=0.5)
    
    ax.set_title(f'Centralized\nRMSE: {cent_rmse:.3f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 3: Decentralized Results
    ax = axes[2]
    
    # Anchors
    for anchor_id in anchor_ids:
        ax.scatter(positions[anchor_id, 0], positions[anchor_id, 1],
                  marker='^', s=200, c='red', label='Anchor' if anchor_id == anchor_ids[0] else '')
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        # True position
        ax.scatter(positions[unknown_id, 0], positions[unknown_id, 1],
                  marker='o', s=100, c='blue', alpha=0.5,
                  label='True' if unknown_id == unknown_ids[0] else '')
        
        # Estimated position
        if unknown_id in decentralized_results:
            est_pos = decentralized_results[unknown_id]
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=100, c='green',
                      label='Estimated' if unknown_id == unknown_ids[0] else '')
            
            # Error line
            ax.plot([positions[unknown_id, 0], est_pos[0]],
                   [positions[unknown_id, 1], est_pos[1]],
                   'r--', alpha=0.5)
    
    ax.set_title(f'True Decentralized\nRMSE: {decent_rmse:.3f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('true_decentralized_results.png', dpi=150)
    plt.show()


def plot_convergence_comparison(cent_info, decent_info):
    """Plot convergence curves"""
    plt.figure(figsize=(12, 5))
    
    # Cost convergence
    plt.subplot(1, 2, 1)
    
    if 'convergence_history' in cent_info:
        plt.semilogy(cent_info['convergence_history'],
                    'b-', label='Centralized', linewidth=2)
    
    if 'cost_history' in decent_info:
        plt.semilogy(decent_info['cost_history'],
                    'r--', label='Decentralized', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost (log scale)')
    plt.title('Cost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Position changes
    plt.subplot(1, 2, 2)
    
    if 'position_history' in decent_info:
        changes = []
        for i in range(1, len(decent_info['position_history'])):
            max_change = 0
            for node_id in decent_info['position_history'][i]:
                if node_id >= 3:  # Unknown nodes
                    prev = decent_info['position_history'][i-1][node_id]
                    curr = decent_info['position_history'][i][node_id]
                    change = np.linalg.norm(curr - prev)
                    max_change = max(max_change, change)
            changes.append(max_change)
        
        if changes:
            plt.semilogy(changes, 'g-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Max Position Change (m, log scale)')
            plt.title('Decentralized Convergence')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=150)
    plt.show()


def main():
    """Run complete test of true decentralized localization"""
    logger.info("=" * 70)
    logger.info("TRUE DECENTRALIZED LOCALIZATION TEST")
    logger.info("Nodes only use LOCAL information and communicate with NEIGHBORS")
    logger.info("=" * 70)
    
    # Create scenario
    n_anchors = 4
    n_unknowns = 6
    room_size = 10.0
    
    positions, anchor_ids, unknown_ids, edges = create_network_scenario(
        n_anchors, n_unknowns, room_size
    )
    
    logger.info(f"\nScenario Configuration:")
    logger.info(f"  Anchors: {n_anchors}")
    logger.info(f"  Unknown nodes: {n_unknowns}")
    logger.info(f"  Room size: {room_size}x{room_size}m")
    logger.info(f"  Measurements: {len(edges)}")
    logger.info(f"  Average connectivity: {2*len(edges)/(n_anchors+n_unknowns):.1f} neighbors/node")
    
    # Test centralized (baseline)
    cent_results, cent_rmse, cent_info = test_centralized(
        positions, anchor_ids, unknown_ids, edges
    )
    
    # Test true decentralized
    decent_results, decent_rmse, decent_info = test_true_decentralized(
        positions, anchor_ids, unknown_ids, edges
    )
    
    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Centralized RMSE:    {cent_rmse:.3f}m (baseline)")
    logger.info(f"Decentralized RMSE:  {decent_rmse:.3f}m")
    logger.info(f"Difference:          {abs(cent_rmse - decent_rmse):.3f}m")
    
    ratio = decent_rmse / cent_rmse if cent_rmse > 0 else float('inf')
    logger.info(f"Decentralized/Centralized ratio: {ratio:.2f}x")
    
    if ratio < 1.5:
        logger.info("✅ Excellent! Decentralized achieves near-optimal performance!")
    elif ratio < 2.0:
        logger.info("✅ Good! Decentralized is within 2x of centralized.")
    else:
        logger.info("⚠️  Decentralized needs tuning or better connectivity.")
    
    # Visualize
    visualize_network_and_results(
        positions, anchor_ids, unknown_ids, edges,
        cent_results, decent_results,
        cent_rmse, decent_rmse
    )
    
    plot_convergence_comparison(cent_info, decent_info)
    
    logger.info("\n✅ Test complete! Results saved to PNG files.")


if __name__ == "__main__":
    main()