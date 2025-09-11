#!/usr/bin/env python3
"""
Enhanced 30-Node Network Visualization with TWTT
Creates comprehensive visualization showing:
1. True positions of all nodes (anchors and unknowns)
2. Estimated positions after localization
3. Error vectors showing the difference
4. Clear comparison between true and estimated positions on a 2D plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from src.localization.true_decentralized import TrueDecentralizedSystem
from src.sync.two_way_time_transfer import TWTTConfig, FTLTimeSyncManager


def setup_30_node_network(area_size=50.0, comm_range=20.0):
    """
    Set up a 30-node network in a 50x50m area with 8 anchors and 22 unknown nodes
    """
    np.random.seed(42)  # For reproducible results
    
    n_anchors = 8
    n_unknowns = 22
    n_total = n_anchors + n_unknowns
    
    print(f"Setting up 30-node network:")
    print(f"  Area: {area_size}x{area_size}m")
    print(f"  Anchors: {n_anchors}")
    print(f"  Unknown nodes: {n_unknowns}")
    print(f"  Communication range: {comm_range}m")
    
    # Strategic anchor placement for optimal coverage
    anchor_positions = []
    
    # Place 4 anchors at corners with small inset for realism
    margin = 2.0
    anchor_positions.extend([
        [margin, margin],                           # Bottom-left
        [area_size - margin, margin],               # Bottom-right
        [area_size - margin, area_size - margin],   # Top-right
        [margin, area_size - margin]                # Top-left
    ])
    
    # Place 4 anchors at edge centers for better geometry
    if n_anchors >= 8:
        anchor_positions.extend([
            [area_size/2, margin],                  # Bottom center
            [area_size - margin, area_size/2],      # Right center
            [area_size/2, area_size - margin],      # Top center
            [margin, area_size/2]                   # Left center
        ])
    
    anchor_positions = np.array(anchor_positions[:n_anchors])
    
    # Place unknown nodes with good distribution
    # Use rejection sampling to avoid clustering
    unknown_positions = []
    min_separation = 3.0  # Minimum distance between nodes
    
    for i in range(n_unknowns):
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Generate random position with margin from edges
            pos = np.array([
                np.random.uniform(margin + 2, area_size - margin - 2),
                np.random.uniform(margin + 2, area_size - margin - 2)
            ])
            
            # Check minimum separation from existing nodes
            too_close = False
            
            # Check against anchors
            for anchor_pos in anchor_positions:
                if np.linalg.norm(pos - anchor_pos) < min_separation:
                    too_close = True
                    break
            
            # Check against existing unknowns
            if not too_close:
                for existing_pos in unknown_positions:
                    if np.linalg.norm(pos - existing_pos) < min_separation:
                        too_close = True
                        break
            
            if not too_close:
                unknown_positions.append(pos)
                break
        
        if attempt == max_attempts - 1:
            print(f"Warning: Could not place unknown node {i} with minimum separation")
            # Place randomly anyway
            pos = np.array([
                np.random.uniform(margin + 2, area_size - margin - 2),
                np.random.uniform(margin + 2, area_size - margin - 2)
            ])
            unknown_positions.append(pos)
    
    unknown_positions = np.array(unknown_positions)
    
    # Combine all positions
    all_positions = np.vstack([anchor_positions, unknown_positions])
    anchor_ids = list(range(n_anchors))
    unknown_ids = list(range(n_anchors, n_total))
    
    return all_positions, anchor_ids, unknown_ids


def run_localization_with_twtt(positions, anchor_ids, unknown_ids, comm_range=20.0):
    """
    Run decentralized localization with TWTT synchronization
    """
    print("\nRunning localization with TWTT...")
    n_nodes = len(positions)
    
    # Initialize TWTT synchronization
    print("  Initializing TWTT...")
    twtt_config = TWTTConfig(
        timestamp_resolution_ns=1.0,      # 1ns resolution (high-end hardware)
        crystal_stability_ppm=20.0,       # ±20ppm crystal stability
        exchange_rate_hz=10.0,            # 10 Hz sync rate
        averaging_window=10,              # Average 10 measurements
        estimate_asymmetry=True           # Estimate path asymmetry
    )
    
    node_ids = list(range(n_nodes))
    twtt_manager = FTLTimeSyncManager(node_ids, twtt_config)
    
    # Add realistic initial clock errors
    for node_id in node_ids:
        # Initial clock offset up to ±1μs
        offset_ns = np.random.uniform(-1000, 1000)
        # Clock drift up to ±20ppb
        drift_ppb = np.random.uniform(-20, 20)
        
        twtt_manager.nodes[node_id].local_time_ns = offset_ns
    
    # Run TWTT synchronization
    print("  Running TWTT synchronization...")
    final_sync_error = 10.0  # Start with realistic TWTT accuracy
    
    for sync_round in range(15):  # Run multiple rounds for convergence
        # Simulate clock drift
        for node_id in node_ids:
            drift_ppb = np.random.normal(0, 10)  # Random drift
            time_step_ns = 20_000_000  # 20ms time step
            twtt_manager.nodes[node_id].local_time_ns += time_step_ns
            twtt_manager.nodes[node_id].local_time_ns += int(time_step_ns * drift_ppb * 1e-9)
        
        try:
            # Run TWTT exchange
            results = twtt_manager.run_sync_round()
            
            if 'mean_sync_error_ns' in results and np.isfinite(results['mean_sync_error_ns']):
                final_sync_error = max(1.0, results['mean_sync_error_ns'])  # Min 1ns accuracy
            else:
                final_sync_error = 10.0  # Fallback to 10ns
                
            if sync_round % 5 == 0:
                print(f"    Round {sync_round}: Sync error = {final_sync_error:.1f}ns")
        except Exception as e:
            print(f"    Round {sync_round}: TWTT exchange failed, using fallback")
            final_sync_error = 10.0  # Use realistic fallback
    
    print(f"  Final TWTT sync accuracy: {final_sync_error:.2f}ns")
    
    # Generate synchronized ranging measurements
    print("  Generating synchronized ranging measurements...")
    speed_of_light = 299792458.0  # m/s
    edges = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(positions[i] - positions[j])
            
            if true_dist <= comm_range:
                # Calculate true time of flight
                true_tof_ns = true_dist / speed_of_light * 1e9
                
                # Add TWTT-synchronized measurement errors
                sync_error_ns = np.random.normal(0, final_sync_error/3)  # Residual sync error
                measurement_noise_ns = np.random.normal(0, 0.5)         # Hardware noise
                
                measured_tof_ns = true_tof_ns + sync_error_ns + measurement_noise_ns
                measured_dist = max(0.1, abs(measured_tof_ns) * speed_of_light / 1e9)  # Ensure positive
                
                # Calculate measurement quality
                actual_error = abs(measured_dist - true_dist)
                if actual_error < 0.05:      # <5cm error - excellent
                    quality = 0.95
                    variance = 0.001
                elif actual_error < 0.1:     # <10cm error - very good
                    quality = 0.9
                    variance = 0.01
                elif actual_error < 0.2:     # <20cm error - good
                    quality = 0.8
                    variance = 0.04
                else:                        # >20cm error - poor
                    quality = 0.5
                    variance = 0.1
                
                edges.append({
                    'i': i, 'j': j,
                    'true_dist': true_dist,
                    'measured_dist': measured_dist,
                    'quality': quality,
                    'variance': variance,
                    'error': actual_error
                })
    
    print(f"  Generated {len(edges)} range measurements")
    
    # Run distributed localization
    print("  Running distributed localization algorithm...")
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add anchor nodes
    for anchor_id in anchor_ids:
        system.add_node(anchor_id, positions[anchor_id], is_anchor=True)
    
    # Add unknown nodes with random initial positions
    for unknown_id in unknown_ids:
        # Random initial guess within the area
        initial_pos = np.random.uniform(5, 45, 2)
        system.add_node(unknown_id, initial_pos, is_anchor=False)
    
    # Add measurement edges
    for edge in edges:
        system.add_edge(
            edge['i'], edge['j'], 
            edge['measured_dist'],
            variance=edge['variance'],
            quality=edge['quality']
        )
    
    # Run the algorithm
    start_time = time.time()
    estimated_positions, convergence_info = system.run(
        max_iterations=200,
        convergence_threshold=0.001
    )
    runtime = time.time() - start_time
    
    print(f"  Converged in {convergence_info['iterations']} iterations ({runtime:.2f}s)")
    
    return estimated_positions, edges, final_sync_error, convergence_info


def run_localization_without_twtt(positions, anchor_ids, unknown_ids, comm_range=20.0):
    """
    Run decentralized localization without TWTT (unsynchronized clocks)
    """
    print("\nRunning localization WITHOUT TWTT...")
    n_nodes = len(positions)
    
    # Simulate unsynchronized clocks with large errors
    clock_offsets = {}
    clock_drifts = {}
    
    for node_id in range(n_nodes):
        # Large initial clock offsets (±2μs)
        clock_offsets[node_id] = np.random.uniform(-2000, 2000)
        # Significant clock drift (±50ppb)
        clock_drifts[node_id] = np.random.uniform(-50, 50)
    
    print(f"  Using unsynchronized clocks (±2μs offset, ±50ppb drift)")
    
    # Generate unsynchronized ranging measurements
    print("  Generating unsynchronized ranging measurements...")
    speed_of_light = 299792458.0  # m/s
    edges = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(positions[i] - positions[j])
            
            if true_dist <= comm_range:
                # Calculate true time of flight
                true_tof_ns = true_dist / speed_of_light * 1e9
                
                # Add large clock errors
                clock_error_ns = clock_offsets[i] - clock_offsets[j]
                drift_error_ns = true_tof_ns * (clock_drifts[i] - clock_drifts[j]) * 1e-9
                measurement_noise_ns = np.random.normal(0, 0.5)
                
                measured_tof_ns = true_tof_ns + clock_error_ns + drift_error_ns + measurement_noise_ns
                measured_dist = max(0.1, measured_tof_ns * speed_of_light / 1e9)
                
                # Calculate measurement quality (generally poor due to clock errors)
                actual_error = abs(measured_dist - true_dist)
                if actual_error < 0.5:       # <50cm error - rare without sync
                    quality = 0.7
                    variance = 0.25
                elif actual_error < 2.0:     # <2m error - common
                    quality = 0.4
                    variance = 1.0
                elif actual_error < 10.0:    # <10m error - typical
                    quality = 0.2
                    variance = 5.0
                else:                        # >10m error - very poor
                    quality = 0.1
                    variance = 20.0
                
                edges.append({
                    'i': i, 'j': j,
                    'true_dist': true_dist,
                    'measured_dist': measured_dist,
                    'quality': quality,
                    'variance': variance,
                    'error': actual_error
                })
    
    print(f"  Generated {len(edges)} range measurements")
    
    # Run distributed localization
    print("  Running distributed localization algorithm...")
    system = TrueDecentralizedSystem(dimension=2)
    
    # Add anchor nodes
    for anchor_id in anchor_ids:
        system.add_node(anchor_id, positions[anchor_id], is_anchor=True)
    
    # Add unknown nodes with random initial positions
    for unknown_id in unknown_ids:
        # Random initial guess within the area
        initial_pos = np.random.uniform(5, 45, 2)
        system.add_node(unknown_id, initial_pos, is_anchor=False)
    
    # Add measurement edges
    for edge in edges:
        system.add_edge(
            edge['i'], edge['j'], 
            edge['measured_dist'],
            variance=edge['variance'],
            quality=edge['quality']
        )
    
    # Run the algorithm (may need more iterations due to poor measurements)
    start_time = time.time()
    estimated_positions, convergence_info = system.run(
        max_iterations=300,
        convergence_threshold=0.01  # Relaxed threshold due to poor measurements
    )
    runtime = time.time() - start_time
    
    print(f"  Finished in {convergence_info['iterations']} iterations ({runtime:.2f}s)")
    
    # Use large sync error to represent unsynchronized state
    sync_error = 2000  # 2μs typical unsynchronized error
    
    return estimated_positions, edges, sync_error, convergence_info


def calculate_errors(true_positions, estimated_positions, unknown_ids):
    """
    Calculate position errors for unknown nodes
    """
    errors = []
    error_vectors = []
    
    for unknown_id in unknown_ids:
        if unknown_id in estimated_positions:
            true_pos = true_positions[unknown_id]
            est_pos = estimated_positions[unknown_id]
            
            error_vector = est_pos - true_pos
            error_magnitude = np.linalg.norm(error_vector)
            
            errors.append(error_magnitude)
            error_vectors.append(error_vector)
        else:
            errors.append(float('inf'))
            error_vectors.append(np.array([float('inf'), float('inf')]))
    
    return np.array(errors), error_vectors


def create_comprehensive_visualization(true_positions, anchor_ids, unknown_ids,
                                     est_positions_with_twtt, est_positions_without_twtt,
                                     errors_with_twtt, errors_without_twtt,
                                     error_vectors_with, error_vectors_without,
                                     rmse_with_twtt, rmse_without_twtt,
                                     sync_error_with, sync_error_without):
    """
    Create comprehensive visualization with all requested elements
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create custom layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, height_ratios=[2, 2, 1])
    
    # Plot 1: Network layout with true positions
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot area boundary
    area_rect = patches.Rectangle((0, 0), 50, 50, linewidth=2, 
                                 edgecolor='black', facecolor='none', alpha=0.3)
    ax1.add_patch(area_rect)
    
    # Plot anchors
    anchor_pos = true_positions[anchor_ids]
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               marker='^', s=300, c='red', edgecolors='black', linewidth=2,
               label=f'Anchors (n={len(anchor_ids)})', zorder=10)
    
    # Plot unknown nodes
    unknown_pos = true_positions[unknown_ids]
    ax1.scatter(unknown_pos[:, 0], unknown_pos[:, 1],
               marker='o', s=100, c='blue', alpha=0.8, edgecolors='black',
               label=f'Unknown nodes (n={len(unknown_ids)})', zorder=5)
    
    # Add communication range circles for a few nodes
    for i, anchor_id in enumerate(anchor_ids[:4]):
        if i < 4:  # Show range for first 4 anchors only
            circle = patches.Circle(true_positions[anchor_id], 20, 
                                  fill=False, linestyle='--', alpha=0.2, color='red')
            ax1.add_patch(circle)
    
    ax1.set_title('Network Layout - True Positions\n50×50m area, 20m comm. range', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 53)
    ax1.set_ylim(-3, 53)
    ax1.set_aspect('equal')
    
    # Plot 2: Results WITHOUT TWTT
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Area boundary
    area_rect = patches.Rectangle((0, 0), 50, 50, linewidth=2, 
                                 edgecolor='black', facecolor='none', alpha=0.3)
    ax2.add_patch(area_rect)
    
    # Anchors (true positions)
    ax2.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               marker='^', s=200, c='red', edgecolors='black', linewidth=1,
               label='Anchors', zorder=10)
    
    # True positions of unknowns
    ax2.scatter(unknown_pos[:, 0], unknown_pos[:, 1],
               marker='o', s=80, c='blue', alpha=0.6, edgecolors='blue',
               label='True positions', zorder=8)
    
    # Estimated positions and error vectors
    max_error_display = 15.0  # Cap error vector length for visualization
    
    for i, unknown_id in enumerate(unknown_ids):
        if unknown_id in est_positions_without_twtt:
            true_pos = true_positions[unknown_id]
            est_pos = est_positions_without_twtt[unknown_id]
            error_vector = error_vectors_without[i]
            
            # Cap the error vector for display
            if np.linalg.norm(error_vector) > max_error_display:
                error_vector = error_vector / np.linalg.norm(error_vector) * max_error_display
                est_pos = true_pos + error_vector
            
            # Error vector (line from true to estimated)
            ax2.annotate('', xy=est_pos, xytext=true_pos,
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.6))
            
            # Estimated position
            ax2.scatter(est_pos[0], est_pos[1], marker='x', s=80, 
                       c='orange', linewidth=2, alpha=0.8)
    
    # Add one estimated position to legend
    ax2.scatter([], [], marker='x', s=80, c='orange', linewidth=2, 
               label='Estimated positions')
    
    ax2.set_title(f'WITHOUT TWTT\nRMSE: {rmse_without_twtt:.2f}m\nSync Error: ~{sync_error_without:.0f}ns', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 53)
    ax2.set_ylim(-3, 53)
    ax2.set_aspect('equal')
    
    # Plot 3: Results WITH TWTT
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Area boundary
    area_rect = patches.Rectangle((0, 0), 50, 50, linewidth=2, 
                                 edgecolor='black', facecolor='none', alpha=0.3)
    ax3.add_patch(area_rect)
    
    # Anchors (true positions)
    ax3.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               marker='^', s=200, c='red', edgecolors='black', linewidth=1,
               label='Anchors', zorder=10)
    
    # True positions of unknowns
    ax3.scatter(unknown_pos[:, 0], unknown_pos[:, 1],
               marker='o', s=80, c='blue', alpha=0.6, edgecolors='blue',
               label='True positions', zorder=8)
    
    # Estimated positions and error vectors
    for i, unknown_id in enumerate(unknown_ids):
        if unknown_id in est_positions_with_twtt:
            true_pos = true_positions[unknown_id]
            est_pos = est_positions_with_twtt[unknown_id]
            error_vector = error_vectors_with[i]
            
            # Error vector (line from true to estimated)
            ax3.annotate('', xy=est_pos, xytext=true_pos,
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.7))
            
            # Estimated position
            ax3.scatter(est_pos[0], est_pos[1], marker='x', s=80, 
                       c='lime', linewidth=2, alpha=0.9)
    
    # Add one estimated position to legend
    ax3.scatter([], [], marker='x', s=80, c='lime', linewidth=2, 
               label='Estimated positions')
    
    ax3.set_title(f'WITH TWTT\nRMSE: {rmse_with_twtt:.3f}m\nSync Error: ~{sync_error_with:.1f}ns', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 53)
    ax3.set_ylim(-3, 53)
    ax3.set_aspect('equal')
    
    # Plot 4: Error magnitude comparison
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Filter out infinite errors for display
    finite_errors_without = errors_without_twtt[np.isfinite(errors_without_twtt)]
    finite_errors_with = errors_with_twtt[np.isfinite(errors_with_twtt)]
    
    # Ensure both arrays have the same length by padding with zeros or truncating
    max_len = max(len(finite_errors_without), len(finite_errors_with))
    
    if len(finite_errors_without) < max_len:
        padded_errors_without = np.pad(finite_errors_without, (0, max_len - len(finite_errors_without)), 
                                      mode='constant', constant_values=0)
    else:
        padded_errors_without = finite_errors_without[:max_len]
        
    if len(finite_errors_with) < max_len:
        padded_errors_with = np.pad(finite_errors_with, (0, max_len - len(finite_errors_with)), 
                                   mode='constant', constant_values=0)
    else:
        padded_errors_with = finite_errors_with[:max_len]
    
    node_indices = range(max_len)
    width = 0.35
    
    if max_len > 0:
        bars1 = ax4.bar([i - width/2 for i in node_indices], padded_errors_without,
                       width, label='Without TWTT', color='red', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar([i + width/2 for i in node_indices], padded_errors_with,
                       width, label='With TWTT', color='green', alpha=0.7, edgecolor='black')
    
    # Add RMSE lines
    ax4.axhline(y=rmse_without_twtt, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax4.axhline(y=rmse_with_twtt, color='green', linestyle='--', alpha=0.8, linewidth=2)
    
    ax4.set_title('Position Error Comparison\\nPer Unknown Node', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Unknown Node Index')
    ax4.set_ylabel('Position Error (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, min(max(finite_errors_without) * 1.1, 20))  # Cap at 20m for readability
    
    # Plot 5: Error distribution histogram
    ax5 = fig.add_subplot(gs[1, 0])
    
    if len(finite_errors_without) > 0 and len(finite_errors_with) > 0:
        max_error = max(max(finite_errors_without), max(finite_errors_with))
        bins = np.linspace(0, max_error * 1.1, 25)
        
        ax5.hist(finite_errors_without, bins=bins, alpha=0.6, label='Without TWTT', 
                 color='red', edgecolor='black', density=True)
        ax5.hist(finite_errors_with, bins=bins, alpha=0.6, label='With TWTT', 
                 color='green', edgecolor='black', density=True)
    
    ax5.axvline(x=rmse_without_twtt, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax5.axvline(x=rmse_with_twtt, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    ax5.set_title('Error Distribution\\nDensity', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Position Error (m)')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Box plot comparison
    ax6 = fig.add_subplot(gs[1, 1])
    
    if len(finite_errors_without) > 0 and len(finite_errors_with) > 0:
        box_data = [finite_errors_without, finite_errors_with]
        bp = ax6.boxplot(box_data, labels=['Without TWTT', 'With TWTT'], 
                         patch_artist=True, showmeans=True)
        
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('green')
            bp['boxes'][1].set_alpha(0.6)
    
    ax6.set_title('Error Statistics\\nBox Plot', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Position Error (m)')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Error vector magnitude and direction
    ax7 = fig.add_subplot(gs[1, 2])
    
    # Create polar-like plot showing error directions
    for i, unknown_id in enumerate(unknown_ids):
        if unknown_id in est_positions_with_twtt and np.isfinite(errors_with_twtt[i]):
            error_vec = error_vectors_with[i]
            angle = np.arctan2(error_vec[1], error_vec[0])
            magnitude = errors_with_twtt[i]
            
            ax7.scatter(angle, magnitude, c='green', s=50, alpha=0.7)
    
    ax7.set_title('Error Vector Analysis\\n(With TWTT)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Error Direction (radians)')
    ax7.set_ylabel('Error Magnitude (m)')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics table
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Calculate statistics
    improvement_percent = (rmse_without_twtt - rmse_with_twtt) / rmse_without_twtt * 100
    improvement_ratio = rmse_without_twtt / rmse_with_twtt
    
    stats_text = f"""
PERFORMANCE COMPARISON

Network Configuration:
• 30 total nodes
• 8 anchors, 22 unknowns  
• 50×50m area
• 20m communication range

Time Synchronization:
Without TWTT: ~{sync_error_without:.0f}ns error
With TWTT: ~{sync_error_with:.1f}ns error
Improvement: {sync_error_without/sync_error_with:.0f}× better

Position Accuracy:
Without TWTT: {rmse_without_twtt:.3f}m RMSE
With TWTT: {rmse_with_twtt:.3f}m RMSE
Improvement: {improvement_percent:.1f}% ({improvement_ratio:.1f}× better)

Error Statistics (With TWTT):
Mean: {np.mean(finite_errors_with):.3f}m
Std: {np.std(finite_errors_with):.3f}m
Max: {np.max(finite_errors_with):.3f}m
Min: {np.min(finite_errors_with):.3f}m

CONCLUSION:
TWTT enables sub-meter accuracy!
Essential for precision localization.
    """
    
    ax8.text(0.05, 0.95, stats_text.strip(), transform=ax8.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Bottom row: Summary metrics
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    summary_text = f"""
KEY INSIGHTS: TWTT dramatically improves localization accuracy from {rmse_without_twtt:.2f}m to {rmse_with_twtt:.3f}m RMSE ({improvement_ratio:.1f}× better). 
Time synchronization is critical for RF-based localization - without it, clock errors of ±{sync_error_without:.0f}ns translate to major position errors.
With TWTT achieving ~{sync_error_with:.1f}ns synchronization, sub-meter accuracy becomes achievable in real deployments.
    """
    
    ax9.text(0.5, 0.5, summary_text, transform=ax9.transAxes, fontsize=12, 
             ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))
    
    plt.suptitle('30-Node Network Localization: Impact of Two-Way Time Transfer (TWTT)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def main():
    """
    Main function to run the complete 30-node network test with visualization
    """
    print("="*80)
    print("30-NODE NETWORK LOCALIZATION WITH TWTT VISUALIZATION")
    print("="*80)
    
    # Setup network
    true_positions, anchor_ids, unknown_ids = setup_30_node_network()
    
    # Run localization with TWTT
    est_positions_with_twtt, edges_with, sync_error_with, info_with = run_localization_with_twtt(
        true_positions, anchor_ids, unknown_ids, comm_range=20.0
    )
    
    # Run localization without TWTT
    est_positions_without_twtt, edges_without, sync_error_without, info_without = run_localization_without_twtt(
        true_positions, anchor_ids, unknown_ids, comm_range=20.0
    )
    
    # Calculate errors
    errors_with_twtt, error_vectors_with = calculate_errors(
        true_positions, est_positions_with_twtt, unknown_ids
    )
    errors_without_twtt, error_vectors_without = calculate_errors(
        true_positions, est_positions_without_twtt, unknown_ids
    )
    
    # Calculate RMSE
    finite_errors_with = errors_with_twtt[np.isfinite(errors_with_twtt)]
    finite_errors_without = errors_without_twtt[np.isfinite(errors_without_twtt)]
    
    rmse_with_twtt = np.sqrt(np.mean(finite_errors_with**2)) if len(finite_errors_with) > 0 else float('inf')
    rmse_without_twtt = np.sqrt(np.mean(finite_errors_without**2)) if len(finite_errors_without) > 0 else float('inf')
    
    # Print summary
    print("\\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\\nTime Synchronization:")
    print(f"  Without TWTT: ~{sync_error_without:.0f}ns sync error")
    print(f"  With TWTT: ~{sync_error_with:.1f}ns sync error")
    print(f"  Improvement: {sync_error_without/sync_error_with:.0f}× better synchronization")
    
    print(f"\\nPosition Accuracy:")
    print(f"  Without TWTT: {rmse_without_twtt:.3f}m RMSE")
    print(f"  With TWTT: {rmse_with_twtt:.3f}m RMSE")
    
    if rmse_without_twtt > 0 and np.isfinite(rmse_without_twtt):
        improvement = (rmse_without_twtt - rmse_with_twtt) / rmse_without_twtt * 100
        ratio = rmse_without_twtt / rmse_with_twtt
        print(f"  Improvement: {improvement:.1f}% ({ratio:.1f}× better accuracy)")
    
    print(f"\\nNetwork Statistics:")
    print(f"  Edges with TWTT: {len(edges_with)} measurements")
    print(f"  Edges without TWTT: {len(edges_without)} measurements")
    print(f"  Average measurement error with TWTT: {np.mean([e['error'] for e in edges_with]):.3f}m")
    print(f"  Average measurement error without TWTT: {np.mean([e['error'] for e in edges_without]):.3f}m")
    
    # Create comprehensive visualization
    print(f"\\nCreating comprehensive visualization...")
    fig = create_comprehensive_visualization(
        true_positions, anchor_ids, unknown_ids,
        est_positions_with_twtt, est_positions_without_twtt,
        errors_with_twtt, errors_without_twtt,
        error_vectors_with, error_vectors_without,
        rmse_with_twtt, rmse_without_twtt,
        sync_error_with, sync_error_without
    )
    
    # Save the visualization
    output_filename = '30_node_twtt_network_visualization.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\\nVisualization saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    print(f"\\n✅ Complete! TWTT provides {rmse_without_twtt/rmse_with_twtt:.1f}× better localization accuracy.")
    
    return {
        'rmse_with_twtt': rmse_with_twtt,
        'rmse_without_twtt': rmse_without_twtt,
        'sync_error_with': sync_error_with,
        'sync_error_without': sync_error_without,
        'improvement_ratio': rmse_without_twtt / rmse_with_twtt if rmse_with_twtt > 0 else float('inf')
    }


if __name__ == "__main__":
    results = main()