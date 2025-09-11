#!/usr/bin/env python3
"""
Analyze and visualize the 10-node localization results
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_10_nodes import load_config, generate_measurements, run_localization

def analyze_results():
    """Run demo and analyze results in detail"""
    
    # Load configuration
    config = load_config("configs/10_node_demo.yaml")
    np.random.seed(config['system']['seed'])
    
    # Generate measurements
    measurements, anchors, unknowns, measurement_details = generate_measurements(config)
    
    # Run localization
    results, solver_info = run_localization(config, measurements, anchors, unknowns)
    
    # Calculate detailed metrics
    print("\n" + "="*70)
    print("DETAILED POSITION ANALYSIS")
    print("="*70)
    
    print("\n%-8s %-20s %-20s %-10s" % ("Node ID", "True Position", "Estimated Position", "Error (m)"))
    print("-"*70)
    
    errors = []
    distances_from_origin = []
    
    for uid in sorted(results.keys()):
        true_pos = results[uid]['true']
        est_pos = results[uid]['estimated']
        error = results[uid]['error']
        errors.append(error)
        
        # Distance from origin for normalization
        dist_from_origin = np.linalg.norm(true_pos)
        distances_from_origin.append(dist_from_origin)
        
        print("%-8d (%.2f, %.2f)%8s (%.2f, %.2f)%8s %.2f" % 
              (uid, true_pos[0], true_pos[1], "", est_pos[0], est_pos[1], "", error))
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    min_error = np.min(errors)
    mean_error = np.mean(errors)
    
    # Calculate normalized RMSE (normalized by diagonal of area)
    area_diagonal = np.sqrt(10**2 + 10**2)  # 10x10 meter area
    normalized_rmse = rmse / area_diagonal
    
    # Alternative normalization by average distance from origin
    avg_distance = np.mean(distances_from_origin)
    normalized_rmse_dist = rmse / avg_distance if avg_distance > 0 else 0
    
    print("\n" + "="*70)
    print("ERROR METRICS")
    print("="*70)
    print(f"\nAbsolute Metrics:")
    print(f"  RMSE:           {rmse:.2f} meters")
    print(f"  Mean Error:     {mean_error:.2f} meters")
    print(f"  Max Error:      {max_error:.2f} meters")
    print(f"  Min Error:      {min_error:.2f} meters")
    
    print(f"\nNormalized Metrics:")
    print(f"  RMSE/Diagonal:  {normalized_rmse:.1%} (normalized by {area_diagonal:.1f}m diagonal)")
    print(f"  RMSE/Distance:  {normalized_rmse_dist:.1%} (normalized by {avg_distance:.1f}m avg distance)")
    
    # Categorize errors
    excellent = sum(1 for e in errors if e < 0.5)
    good = sum(1 for e in errors if 0.5 <= e < 2.0)
    moderate = sum(1 for e in errors if 2.0 <= e < 5.0)
    poor = sum(1 for e in errors if e >= 5.0)
    
    print(f"\nError Distribution:")
    print(f"  Excellent (<0.5m):  {excellent}/{len(errors)} nodes ({excellent/len(errors)*100:.0f}%)")
    print(f"  Good (0.5-2m):      {good}/{len(errors)} nodes ({good/len(errors)*100:.0f}%)")
    print(f"  Moderate (2-5m):    {moderate}/{len(errors)} nodes ({moderate/len(errors)*100:.0f}%)")
    print(f"  Poor (>5m):         {poor}/{len(errors)} nodes ({poor/len(errors)*100:.0f}%)")
    
    # Create visualization
    create_visualization(anchors, results, measurement_details, rmse, normalized_rmse)
    
    return results, rmse, normalized_rmse

def create_visualization(anchors, results, measurements, rmse, normalized_rmse):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main plot: Positions
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Plot measurement links with quality coloring
    for meas in measurements:
        if meas['pair'][0] in anchors or meas['pair'][1] in anchors:
            continue  # Skip anchor-anchor links for clarity
            
        # Get positions
        pos1 = results.get(meas['pair'][0], {}).get('true')
        pos2 = results.get(meas['pair'][1], {}).get('true')
        
        if pos1 is None:
            pos1 = anchors.get(meas['pair'][0])
        if pos2 is None:
            pos2 = anchors.get(meas['pair'][1])
            
        if pos1 is not None and pos2 is not None:
            alpha = 0.2 if meas['propagation'] == 'line_of_sight' else 0.1
            color = 'green' if meas['propagation'] == 'line_of_sight' else 'red'
            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                    color=color, alpha=alpha, linewidth=0.5)
    
    # Plot anchors
    anchor_pos = np.array(list(anchors.values()))
    ax1.scatter(anchor_pos[:, 0], anchor_pos[:, 1], 
               s=300, c='red', marker='^', label='Anchors', 
               edgecolors='black', linewidth=2, zorder=5)
    
    # Annotate anchors
    for aid, apos in anchors.items():
        ax1.annotate(f'A{aid}', xy=apos, xytext=(0, -15), 
                    textcoords='offset points', ha='center', fontweight='bold')
    
    # Plot true and estimated positions with error vectors
    for uid, result in results.items():
        true = result['true']
        est = result['estimated']
        error = result['error']
        
        # True position (green circle)
        ax1.scatter(true[0], true[1], s=150, c='green', marker='o', 
                   alpha=0.6, edgecolors='darkgreen', linewidth=2)
        
        # Estimated position (blue X)
        ax1.scatter(est[0], est[1], s=150, c='blue', marker='x', linewidth=3)
        
        # Error vector
        ax1.arrow(true[0], true[1], est[0]-true[0], est[1]-true[1],
                 head_width=0.2, head_length=0.1, fc='orange', ec='orange', 
                 alpha=0.5, linewidth=2)
        
        # Label with node ID and error
        ax1.annotate(f'{uid}\n{error:.1f}m', 
                    xy=est, xytext=(10, 10), textcoords='offset points', 
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='yellow', alpha=0.7))
    
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.set_xlabel('X Position (meters)', fontsize=12)
    ax1.set_ylabel('Y Position (meters)', fontsize=12)
    ax1.set_title(f'10-Node Localization Results\nRMSE: {rmse:.2f}m ({normalized_rmse:.1%} normalized)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='r', 
               markersize=12, label='Anchors'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
               markersize=10, label='True Position', alpha=0.6),
        Line2D([0], [0], marker='x', color='b', markersize=10, 
               label='Estimated Position', linewidth=3),
        Line2D([0], [0], color='orange', linewidth=2, 
               label='Error Vector'),
        Line2D([0], [0], color='green', alpha=0.3, 
               label='LOS Link'),
        Line2D([0], [0], color='red', alpha=0.3, 
               label='NLOS Link')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Error histogram
    ax2 = plt.subplot(2, 3, 2)
    errors = [r['error'] for r in results.values()]
    ax2.hist(errors, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(rmse, color='red', linestyle='--', linewidth=2, label=f'RMSE: {rmse:.2f}m')
    ax2.set_xlabel('Localization Error (meters)')
    ax2.set_ylabel('Number of Nodes')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error vs Node ID
    ax3 = plt.subplot(2, 3, 3)
    node_ids = sorted(results.keys())
    node_errors = [results[nid]['error'] for nid in node_ids]
    colors = ['green' if e < 2 else 'orange' if e < 5 else 'red' for e in node_errors]
    ax3.bar(range(len(node_ids)), node_errors, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(node_ids)))
    ax3.set_xticklabels(node_ids)
    ax3.set_xlabel('Node ID')
    ax3.set_ylabel('Error (meters)')
    ax3.set_title('Per-Node Localization Error')
    ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1m threshold')
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5m threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Measurement quality vs error
    ax4 = plt.subplot(2, 3, 5)
    meas_errors = [m['error'] for m in measurements]
    meas_qualities = [m['quality'] for m in measurements]
    meas_colors = ['green' if m['propagation'] == 'line_of_sight' else 'red' 
                   for m in measurements]
    ax4.scatter(meas_qualities, meas_errors, c=meas_colors, alpha=0.6, s=30)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Measurement Quality Score')
    ax4.set_ylabel('Measurement Error (meters)')
    ax4.set_title('Measurement Quality Analysis')
    ax4.grid(True, alpha=0.3)
    
    # X-Y error components
    ax5 = plt.subplot(2, 3, 6)
    x_errors = [r['estimated'][0] - r['true'][0] for r in results.values()]
    y_errors = [r['estimated'][1] - r['true'][1] for r in results.values()]
    ax5.scatter(x_errors, y_errors, s=100, alpha=0.6, c=node_errors, cmap='RdYlGn_r')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='green', label='1m radius')
    ax5.add_patch(circle)
    circle2 = plt.Circle((0, 0), 5, fill=False, linestyle='--', color='orange', label='5m radius')
    ax5.add_patch(circle2)
    ax5.set_xlabel('X Error (meters)')
    ax5.set_ylabel('Y Error (meters)')
    ax5.set_title('Error Vector Components')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    cbar = plt.colorbar(ax5.scatter(x_errors, y_errors, s=100, alpha=0.6, 
                                    c=node_errors, cmap='RdYlGn_r'), ax=ax5)
    cbar.set_label('Total Error (m)')
    
    plt.suptitle('10-Node Indoor Localization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results, rmse, normalized_rmse = analyze_results()
    
    print("\n" + "="*70)
    print("HARDWARE ASSUMPTIONS IMPACT")
    print("="*70)
    print("""
    The spread spectrum signal generator implements:
    
    1. WAVEFORM STRUCTURE:
       - Gold codes (1023 chips) for ranging
       - 100 MHz chip rate = 100 MHz bandwidth
       - Root-raised cosine pulse shaping (β=0.35)
       - 7 pilot tones for frequency synchronization
       
    2. RANGING ACCURACY FACTORS:
       - Bandwidth: 100 MHz → 1.5m theoretical resolution (c/2B)
       - Sub-sample interpolation: ~10x improvement → 0.15m
       - SNR impact: σ² = c²/(2β²ρ) where ρ is SNR
       - Allan variance: Additional clock drift contribution
       
    3. ACTUAL PERFORMANCE:
       - RMSE: {:.2f}m absolute
       - Normalized: {:.1%} of area diagonal
       - Best case: <0.5m (high SNR, LOS)
       - Worst case: >5m (NLOS with bias)
    """.format(rmse, normalized_rmse))