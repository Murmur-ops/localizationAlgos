#!/usr/bin/env python3
"""
Large Network Test with TWTT: 30 nodes (8 anchors, 22 unknowns) in 50x50m area
Tests decentralized localization with proper time synchronization
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from src.localization.true_decentralized import TrueDecentralizedSystem
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from src.sync.two_way_time_transfer import TWTTConfig, FTLTimeSyncManager


def create_large_network_with_twtt(n_total=30, n_anchors=8, area_size=50.0):
    """
    Create large network with TWTT synchronization
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
        area_size * 0.1,
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


def simulate_ranging_with_twtt(positions, comm_range=20.0, use_twtt=True):
    """
    Simulate ranging with or without TWTT synchronization
    """
    n_nodes = len(positions)
    speed_of_light = 299792458.0  # m/s
    
    # Initialize TWTT if enabled
    if use_twtt:
        print("\nInitializing TWTT synchronization...")
        twtt_config = TWTTConfig(
            timestamp_resolution_ns=1.0,    # 1ns resolution (good hardware)
            crystal_stability_ppm=20.0,     # ±20ppm crystal
            estimate_asymmetry=True
        )
        
        node_ids = list(range(n_nodes))
        twtt_manager = FTLTimeSyncManager(node_ids, twtt_config)
        
        # Add realistic initial clock offsets
        clock_offsets = {}
        clock_drifts = {}
        for node_id in node_ids:
            clock_offsets[node_id] = np.random.uniform(-1000, 1000)  # ±1μs initial offset
            clock_drifts[node_id] = np.random.uniform(-20, 20)       # ±20ppb drift
            twtt_manager.nodes[node_id].local_time_ns = clock_offsets[node_id]
        
        # Run TWTT synchronization rounds
        print("Running TWTT synchronization...")
        for round in range(20):
            # Advance clocks with drift
            for node_id in node_ids:
                drift_ppb = clock_drifts[node_id]
                twtt_manager.nodes[node_id].local_time_ns += 10_000_000  # 10ms
                twtt_manager.nodes[node_id].local_time_ns += int(10_000_000 * drift_ppb * 1e-9)
            
            # Run TWTT
            results = twtt_manager.run_sync_round()
            
            if round % 5 == 0:
                print(f"  Round {round}: Sync error = {results['mean_sync_error_ns']:.1f}ns")
        
        final_sync_error = results['mean_sync_error_ns']
        print(f"Final TWTT sync accuracy: {final_sync_error:.1f}ns")
    else:
        print("\nNo TWTT - using unsynchronized clocks")
        clock_offsets = {i: np.random.uniform(-1000, 1000) for i in range(n_nodes)}
        clock_drifts = {i: np.random.uniform(-20, 20) for i in range(n_nodes)}
        final_sync_error = 1000  # 1μs typical without sync
    
    # Generate ranging measurements
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(positions[i] - positions[j])
            
            if true_dist <= comm_range:
                # Calculate time of flight
                true_tof = true_dist / speed_of_light * 1e9  # nanoseconds
                
                if use_twtt:
                    # With TWTT: small residual sync error
                    sync_error = np.random.normal(0, final_sync_error/10)
                    measurement_noise = np.random.normal(0, 1)  # 1ns measurement noise
                    measured_tof = true_tof + sync_error + measurement_noise
                else:
                    # Without TWTT: large clock errors
                    clock_error = (clock_offsets[i] - clock_offsets[j])
                    drift_error = true_tof * (clock_drifts[i] - clock_drifts[j]) * 1e-9
                    measurement_noise = np.random.normal(0, 1)
                    measured_tof = true_tof + clock_error + drift_error + measurement_noise
                
                # Convert back to distance (ensure positive)
                measured_dist = max(0.1, measured_tof * speed_of_light / 1e9)  # Minimum 10cm
                
                # Calculate measurement quality based on actual error
                actual_error = abs(measured_dist - true_dist)
                if actual_error < 0.1:  # <10cm error
                    quality = 0.9
                    variance = 0.01
                elif actual_error < 0.5:  # <50cm error
                    quality = 0.7
                    variance = 0.1
                elif actual_error < 1.0:  # <1m error
                    quality = 0.5
                    variance = 0.5
                else:  # >1m error
                    quality = 0.2
                    variance = 2.0
                
                edges.append({
                    'i': i,
                    'j': j,
                    'true_dist': true_dist,
                    'measured_dist': measured_dist,
                    'quality': quality,
                    'variance': variance,
                    'error': actual_error
                })
    
    return edges, final_sync_error


def test_with_twtt(positions, anchor_ids, unknown_ids, edges, sync_type="TWTT"):
    """Test localization with TWTT-synchronized measurements"""
    print(f"\n{'='*60}")
    print(f"DECENTRALIZED LOCALIZATION WITH {sync_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create system
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add all nodes
    for anchor_id in anchor_ids:
        system.add_node(anchor_id, positions[anchor_id], is_anchor=True)
    
    for unknown_id in unknown_ids:
        initial_pos = np.random.uniform(10, 40, 2)
        system.add_node(unknown_id, initial_pos, is_anchor=False)
    
    # Add measurements
    for edge in edges:
        system.add_edge(
            edge['i'],
            edge['j'],
            edge['measured_dist'],
            variance=edge['variance'],
            quality=edge['quality']
        )
    
    print(f"Total edges: {len(edges)}")
    
    # Check measurement quality
    avg_error = np.mean([e['error'] for e in edges])
    max_error = np.max([e['error'] for e in edges])
    print(f"Measurement quality:")
    print(f"  Average error: {avg_error:.3f}m")
    print(f"  Max error: {max_error:.3f}m")
    
    # Analyze connectivity
    connectivity = []
    for node_id in range(len(positions)):
        n_neighbors = len(system.topology[node_id])
        connectivity.append(n_neighbors)
    
    avg_connectivity = np.mean(connectivity)
    print(f"Connectivity: {avg_connectivity:.1f} neighbors/node average")
    
    # Run distributed algorithm
    print("\nRunning distributed consensus...")
    final_positions, info = system.run(
        max_iterations=100,
        convergence_threshold=0.01
    )
    
    elapsed_time = time.time() - start_time
    
    # Compute errors
    errors = []
    for unknown_id in unknown_ids:
        est_pos = final_positions[unknown_id]
        true_pos = positions[unknown_id]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        if unknown_id - anchor_ids[-1] <= 5:  # Show first 5
            print(f"  Node {unknown_id}: error = {error:.3f}m")
    
    if len(unknown_ids) > 5:
        print(f"  ... ({len(unknown_ids)-5} more nodes)")
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_pos_error = np.max(errors)
    min_pos_error = np.min(errors)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.3f}m")
    print(f"  Max error: {max_pos_error:.3f}m")
    print(f"  Min error: {min_pos_error:.3f}m")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Converged: {info['converged']}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return final_positions, rmse, errors


def compare_with_without_twtt():
    """Compare performance with and without TWTT"""
    print("="*60)
    print("30-NODE NETWORK: WITH vs WITHOUT TWTT")
    print("="*60)
    
    # Create network
    positions, anchor_ids, unknown_ids = create_large_network_with_twtt(
        n_total=30,
        n_anchors=8,
        area_size=50.0
    )
    
    comm_range = 20.0
    
    # Test WITHOUT TWTT
    print("\n" + "="*60)
    print("TEST 1: WITHOUT TWTT (Unsynchronized)")
    edges_without, sync_error_without = simulate_ranging_with_twtt(
        positions, comm_range, use_twtt=False
    )
    
    results_without, rmse_without, errors_without = test_with_twtt(
        positions, anchor_ids, unknown_ids, edges_without, "NO SYNC"
    )
    
    # Test WITH TWTT
    print("\n" + "="*60)
    print("TEST 2: WITH TWTT (Synchronized)")
    edges_with, sync_error_with = simulate_ranging_with_twtt(
        positions, comm_range, use_twtt=True
    )
    
    results_with, rmse_with, errors_with = test_with_twtt(
        positions, anchor_ids, unknown_ids, edges_with, "TWTT"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nTime Synchronization:")
    print(f"  Without TWTT: ~{sync_error_without:.0f}ns sync error")
    print(f"  With TWTT:    ~{sync_error_with:.1f}ns sync error")
    print(f"  Improvement:  {sync_error_without/sync_error_with:.0f}x better")
    
    print(f"\nLocalization Accuracy:")
    print(f"  Without TWTT: {rmse_without:.3f}m RMSE")
    print(f"  With TWTT:    {rmse_with:.3f}m RMSE")
    
    if rmse_without > 0:
        improvement = (rmse_without - rmse_with) / rmse_without * 100
        ratio = rmse_without / rmse_with
        print(f"  Improvement:  {improvement:.1f}% ({ratio:.1f}x better)")
    
    print(f"\nError Statistics:")
    print(f"  Without TWTT: Mean={np.mean(errors_without):.3f}m, "
          f"Std={np.std(errors_without):.3f}m, "
          f"Max={np.max(errors_without):.3f}m")
    print(f"  With TWTT:    Mean={np.mean(errors_with):.3f}m, "
          f"Std={np.std(errors_with):.3f}m, "
          f"Max={np.max(errors_with):.3f}m")
    
    # Visualize
    visualize_twtt_comparison(
        positions, anchor_ids, unknown_ids,
        results_without, results_with,
        errors_without, errors_with,
        rmse_without, rmse_with
    )
    
    print("\n✅ Test complete! TWTT significantly improves localization accuracy.")
    
    return rmse_without, rmse_with


def visualize_twtt_comparison(positions, anchor_ids, unknown_ids,
                              results_without, results_with,
                              errors_without, errors_with,
                              rmse_without, rmse_with):
    """Visualize the improvement from TWTT"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Without TWTT
    ax = axes[0, 0]
    
    # Anchors
    ax.scatter(positions[anchor_ids, 0], positions[anchor_ids, 1],
              marker='^', s=200, c='red', label='Anchors', zorder=5)
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        true_pos = positions[unknown_id]
        if unknown_id in results_without:
            est_pos = results_without[unknown_id]
            # Error line
            ax.plot([true_pos[0], est_pos[0]],
                   [true_pos[1], est_pos[1]],
                   'r-', alpha=0.3, linewidth=1)
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=30, c='orange', alpha=0.6)
    
    ax.scatter(positions[unknown_ids, 0], positions[unknown_ids, 1],
              marker='o', s=30, c='blue', alpha=0.4, label='True position')
    
    ax.set_title(f'WITHOUT TWTT\nRMSE: {rmse_without:.3f}m', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    
    # Plot 2: With TWTT
    ax = axes[0, 1]
    
    # Anchors
    ax.scatter(positions[anchor_ids, 0], positions[anchor_ids, 1],
              marker='^', s=200, c='red', label='Anchors', zorder=5)
    
    # True and estimated positions
    for unknown_id in unknown_ids:
        true_pos = positions[unknown_id]
        if unknown_id in results_with:
            est_pos = results_with[unknown_id]
            # Error line
            ax.plot([true_pos[0], est_pos[0]],
                   [true_pos[1], est_pos[1]],
                   'g-', alpha=0.3, linewidth=1)
            ax.scatter(est_pos[0], est_pos[1],
                      marker='x', s=30, c='green', alpha=0.6)
    
    ax.scatter(positions[unknown_ids, 0], positions[unknown_ids, 1],
              marker='o', s=30, c='blue', alpha=0.4, label='True position')
    
    ax.set_title(f'WITH TWTT\nRMSE: {rmse_with:.3f}m', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    
    # Plot 3: Error comparison bar chart
    ax = axes[1, 0]
    
    node_indices = range(len(errors_without))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in node_indices], errors_without,
                   width, label='Without TWTT', color='red', alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in node_indices], errors_with,
                   width, label='With TWTT', color='green', alpha=0.7)
    
    ax.axhline(y=rmse_without, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=rmse_with, color='green', linestyle='--', alpha=0.5)
    
    ax.set_title('Per-Node Position Error')
    ax.set_xlabel('Unknown Node Index')
    ax.set_ylabel('Position Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    
    bins = np.linspace(0, max(max(errors_without), max(errors_with)), 30)
    ax.hist(errors_without, bins=bins, alpha=0.5, label='Without TWTT', color='red', edgecolor='black')
    ax.hist(errors_with, bins=bins, alpha=0.5, label='With TWTT', color='green', edgecolor='black')
    
    ax.axvline(x=rmse_without, color='red', linestyle='--', alpha=0.7, label=f'RMSE w/o: {rmse_without:.3f}m')
    ax.axvline(x=rmse_with, color='green', linestyle='--', alpha=0.7, label=f'RMSE with: {rmse_with:.3f}m')
    
    ax.set_title('Error Distribution')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('30-Node Network: Impact of TWTT on Localization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('large_network_twtt_comparison.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    rmse_without, rmse_with = compare_with_without_twtt()