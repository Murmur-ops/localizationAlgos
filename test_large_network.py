#!/usr/bin/env python3
"""
Large Network Test: 30 nodes (8 anchors, 22 unknowns) in 50x50m area
Tests scalability and performance of decentralized localization
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from src.localization.true_decentralized import TrueDecentralizedSystem
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge


def create_large_network(n_total=30, n_anchors=8, area_size=50.0):
    """
    Create a large network scenario
    
    Args:
        n_total: Total number of nodes
        n_anchors: Number of anchor nodes
        area_size: Size of deployment area (meters)
        
    Returns:
        positions, anchor_ids, unknown_ids, connectivity
    """
    np.random.seed(42)
    
    # Place anchors around perimeter for good coverage
    anchor_positions = []
    
    # Place 4 anchors at corners
    anchor_positions.extend([
        [0, 0],
        [area_size, 0],
        [area_size, area_size],
        [0, area_size]
    ])
    
    # Place remaining anchors at edges
    if n_anchors > 4:
        anchor_positions.append([area_size/2, 0])  # Bottom center
    if n_anchors > 5:
        anchor_positions.append([area_size, area_size/2])  # Right center
    if n_anchors > 6:
        anchor_positions.append([area_size/2, area_size])  # Top center
    if n_anchors > 7:
        anchor_positions.append([0, area_size/2])  # Left center
        
    anchor_positions = np.array(anchor_positions[:n_anchors])
    
    # Place unknown nodes randomly in the area
    n_unknowns = n_total - n_anchors
    unknown_positions = np.random.uniform(
        area_size * 0.1,  # Keep away from edges
        area_size * 0.9,
        (n_unknowns, 2)
    )
    
    # Combine all positions
    all_positions = np.vstack([anchor_positions, unknown_positions])
    
    print(f"\nNetwork Configuration:")
    print(f"  Total nodes: {n_total}")
    print(f"  Anchors: {n_anchors}")
    print(f"  Unknown nodes: {n_unknowns}")
    print(f"  Area: {area_size}x{area_size} meters")
    
    return all_positions, list(range(n_anchors)), list(range(n_anchors, n_total))


def compute_connectivity(positions, comm_range=20.0):
    """
    Compute network connectivity based on communication range
    
    Args:
        positions: Node positions
        comm_range: Maximum communication range (meters)
        
    Returns:
        edges: List of connected node pairs with measurements
    """
    n_nodes = len(positions)
    edges = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(positions[i] - positions[j])
            
            if true_dist <= comm_range:
                # Add realistic noise based on distance
                noise_std = 0.01 + 0.02 * (true_dist / comm_range)  # 1-3cm noise
                measured_dist = true_dist + np.random.normal(0, noise_std)
                
                # Quality decreases with distance
                quality = max(0.2, 1.0 - (true_dist / comm_range) ** 2)
                
                edges.append({
                    'i': i,
                    'j': j,
                    'true_dist': true_dist,
                    'measured_dist': measured_dist,
                    'noise_std': noise_std,
                    'quality': quality
                })
    
    return edges


def test_centralized_large(positions, anchor_ids, unknown_ids, edges):
    """Test centralized localization on large network"""
    print("\n" + "="*60)
    print("CENTRALIZED LOCALIZATION")
    print("="*60)
    
    start_time = time.time()
    
    # Setup anchor dictionary
    anchor_dict = {i: positions[i] for i in anchor_ids}
    
    # Create measurements for unknowns to anchors
    measurements = []
    for edge in edges:
        # Only use measurements involving at least one anchor for centralized
        if edge['i'] in anchor_ids or edge['j'] in anchor_ids:
            measurements.append(MeasurementEdge(
                node_i=edge['i'],
                node_j=edge['j'],
                distance=edge['measured_dist'],
                quality=edge['quality'],
                variance=edge['noise_std']**2
            ))
    
    print(f"Measurements used: {len(measurements)}")
    
    # Initial guess for all unknowns
    n_unknowns = len(unknown_ids)
    initial_guess = np.random.uniform(10, 40, n_unknowns * 2)
    
    # Solve
    localizer = RobustLocalizer(dimension=2)
    optimized_positions, info = localizer.solve(
        initial_guess,
        measurements,
        anchor_dict
    )
    
    elapsed_time = time.time() - start_time
    
    # Compute errors
    errors = []
    for idx, unknown_id in enumerate(unknown_ids):
        est_pos = optimized_positions[idx*2:(idx+1)*2]
        true_pos = positions[unknown_id]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        if idx < 5:  # Show first 5 for brevity
            print(f"  Node {unknown_id}: error = {error:.3f}m")
    
    if len(unknown_ids) > 5:
        print(f"  ... ({len(unknown_ids)-5} more nodes)")
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.3f}m")
    print(f"  Max error: {max_error:.3f}m")
    print(f"  Min error: {min_error:.3f}m")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    # Create result dictionary for unknowns
    cent_results = {}
    for idx, unknown_id in enumerate(unknown_ids):
        cent_results[unknown_id] = optimized_positions[idx*2:(idx+1)*2]
    
    return cent_results, rmse, errors


def test_decentralized_large(positions, anchor_ids, unknown_ids, edges):
    """Test decentralized localization on large network"""
    print("\n" + "="*60)
    print("DECENTRALIZED LOCALIZATION")
    print("="*60)
    
    start_time = time.time()
    
    # Create system
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add all nodes
    for anchor_id in anchor_ids:
        system.add_node(anchor_id, positions[anchor_id], is_anchor=True)
    
    for unknown_id in unknown_ids:
        # Random initial guess
        initial_pos = np.random.uniform(10, 40, 2)
        system.add_node(unknown_id, initial_pos, is_anchor=False)
    
    # Add all edges (measurements)
    for edge in edges:
        system.add_edge(
            edge['i'],
            edge['j'],
            edge['measured_dist'],
            variance=edge['noise_std']**2,
            quality=edge['quality']
        )
    
    print(f"Total edges: {len(edges)}")
    
    # Analyze connectivity
    connectivity = []
    for node_id in range(len(positions)):
        n_neighbors = len(system.topology[node_id])
        connectivity.append(n_neighbors)
    
    avg_connectivity = np.mean(connectivity)
    min_connectivity = np.min(connectivity)
    max_connectivity = np.max(connectivity)
    
    print(f"Connectivity:")
    print(f"  Average: {avg_connectivity:.1f} neighbors/node")
    print(f"  Min: {min_connectivity} neighbors")
    print(f"  Max: {max_connectivity} neighbors")
    
    # Check if network is well-connected
    isolated_nodes = [i for i, c in enumerate(connectivity) if c == 0]
    if isolated_nodes:
        print(f"  ⚠️ WARNING: {len(isolated_nodes)} isolated nodes: {isolated_nodes}")
    
    poorly_connected = [i for i in unknown_ids if connectivity[i] < 3]
    if poorly_connected:
        print(f"  ⚠️ WARNING: {len(poorly_connected)} unknown nodes with < 3 neighbors")
    
    # Run distributed algorithm
    print("\nRunning distributed consensus...")
    final_positions, info = system.run(
        max_iterations=100,
        convergence_threshold=0.01  # Slightly relaxed for large network
    )
    
    elapsed_time = time.time() - start_time
    
    # Compute errors
    errors = []
    for unknown_id in unknown_ids:
        est_pos = final_positions[unknown_id]
        true_pos = positions[unknown_id]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        if unknown_id - anchor_ids[-1] <= 5:  # Show first 5 unknowns
            print(f"  Node {unknown_id}: error = {error:.3f}m")
    
    if len(unknown_ids) > 5:
        print(f"  ... ({len(unknown_ids)-5} more nodes)")
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.3f}m")
    print(f"  Max error: {max_error:.3f}m")
    print(f"  Min error: {min_error:.3f}m")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Converged: {info['converged']}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return final_positions, rmse, errors


def visualize_large_network(positions, anchor_ids, unknown_ids, edges,
                           cent_results, decent_results, cent_errors, decent_errors):
    """Visualize large network and results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Network topology
    ax = axes[0, 0]
    
    # Draw edges
    for edge in edges:
        i, j = edge['i'], edge['j']
        ax.plot([positions[i, 0], positions[j, 0]],
               [positions[i, 1], positions[j, 1]],
               'k-', alpha=0.1, linewidth=0.5)
    
    # Draw nodes
    ax.scatter(positions[anchor_ids, 0], positions[anchor_ids, 1],
              marker='^', s=200, c='red', label='Anchors', zorder=5)
    ax.scatter(positions[unknown_ids, 0], positions[unknown_ids, 1],
              marker='o', s=50, c='blue', label='Unknown', alpha=0.6, zorder=4)
    
    ax.set_title('Network Topology (30 nodes, 50x50m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    
    # Plot 2: Centralized results
    ax = axes[0, 1]
    
    # Anchors
    ax.scatter(positions[anchor_ids, 0], positions[anchor_ids, 1],
              marker='^', s=200, c='red', zorder=5)
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        true_pos = positions[unknown_id]
        if unknown_id in cent_results:
            est_pos = cent_results[unknown_id]
            # Error line
            ax.plot([true_pos[0], est_pos[0]],
                   [true_pos[1], est_pos[1]],
                   'r-', alpha=0.3, linewidth=1)
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=30, c='green', alpha=0.6)
    
    ax.scatter(positions[unknown_ids, 0], positions[unknown_ids, 1],
              marker='o', s=30, c='blue', alpha=0.4)
    
    cent_rmse = np.sqrt(np.mean(np.array(cent_errors)**2))
    ax.set_title(f'Centralized Results\nRMSE: {cent_rmse:.3f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    
    # Plot 3: Decentralized results
    ax = axes[1, 0]
    
    # Anchors
    ax.scatter(positions[anchor_ids, 0], positions[anchor_ids, 1],
              marker='^', s=200, c='red', zorder=5)
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        true_pos = positions[unknown_id]
        if unknown_id in decent_results:
            est_pos = decent_results[unknown_id]
            # Error line
            ax.plot([true_pos[0], est_pos[0]],
                   [true_pos[1], est_pos[1]],
                   'r-', alpha=0.3, linewidth=1)
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=30, c='orange', alpha=0.6)
    
    ax.scatter(positions[unknown_ids, 0], positions[unknown_ids, 1],
              marker='o', s=30, c='blue', alpha=0.4)
    
    decent_rmse = np.sqrt(np.mean(np.array(decent_errors)**2))
    ax.set_title(f'Decentralized Results\nRMSE: {decent_rmse:.3f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    
    # Plot 4: Error comparison
    ax = axes[1, 1]
    
    node_indices = range(len(cent_errors))
    width = 0.35
    
    ax.bar([i - width/2 for i in node_indices], cent_errors,
           width, label='Centralized', color='green', alpha=0.7)
    ax.bar([i + width/2 for i in node_indices], decent_errors,
           width, label='Decentralized', color='orange', alpha=0.7)
    
    ax.axhline(y=cent_rmse, color='green', linestyle='--', alpha=0.5, label=f'Cent RMSE: {cent_rmse:.3f}m')
    ax.axhline(y=decent_rmse, color='orange', linestyle='--', alpha=0.5, label=f'Decent RMSE: {decent_rmse:.3f}m')
    
    ax.set_title('Per-Node Error Comparison')
    ax.set_xlabel('Unknown Node Index')
    ax.set_ylabel('Position Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('large_network_30nodes.png', dpi=150)
    plt.show()


def main():
    """Run large network test"""
    print("="*60)
    print("LARGE NETWORK TEST: 30 NODES IN 50x50m AREA")
    print("="*60)
    
    # Create network
    positions, anchor_ids, unknown_ids = create_large_network(
        n_total=30,
        n_anchors=8,
        area_size=50.0
    )
    
    # Compute connectivity (20m communication range for 50x50m area)
    comm_range = 20.0
    edges = compute_connectivity(positions, comm_range)
    
    print(f"\nConnectivity with {comm_range}m range:")
    print(f"  Total edges: {len(edges)}")
    print(f"  Average edges per node: {2*len(edges)/30:.1f}")
    
    # Test centralized
    cent_results, cent_rmse, cent_errors = test_centralized_large(
        positions, anchor_ids, unknown_ids, edges
    )
    
    # Test decentralized
    decent_results, decent_rmse, decent_errors = test_decentralized_large(
        positions, anchor_ids, unknown_ids, edges
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Centralized RMSE:    {cent_rmse:.3f}m")
    print(f"Decentralized RMSE:  {decent_rmse:.3f}m")
    print(f"Ratio (Decent/Cent): {decent_rmse/cent_rmse:.2f}x")
    
    if decent_rmse < cent_rmse * 3:
        print("✅ SUCCESS: Decentralized performs well at scale!")
    else:
        print("⚠️ WARNING: Large performance gap")
    
    # Statistics
    print(f"\nError Statistics:")
    print(f"  Centralized:   Mean={np.mean(cent_errors):.3f}m, "
          f"Std={np.std(cent_errors):.3f}m, "
          f"Max={np.max(cent_errors):.3f}m")
    print(f"  Decentralized: Mean={np.mean(decent_errors):.3f}m, "
          f"Std={np.std(decent_errors):.3f}m, "
          f"Max={np.max(decent_errors):.3f}m")
    
    # Visualize
    visualize_large_network(
        positions, anchor_ids, unknown_ids, edges,
        cent_results, decent_results, cent_errors, decent_errors
    )
    
    print("\n✅ Test complete! Results saved to large_network_30nodes.png")


if __name__ == "__main__":
    main()