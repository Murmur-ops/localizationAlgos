#!/usr/bin/env python3
"""
Visualization tool for sensor network and localization results
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import yaml

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict_to_config(config_dict: dict) -> MPSConfig:
    """Convert configuration dictionary to MPSConfig object"""
    return MPSConfig(
        n_sensors=config_dict['network']['n_sensors'],
        n_anchors=config_dict['network']['n_anchors'],
        communication_range=config_dict['network']['communication_range'],
        dimension=config_dict['network']['dimension'],
        noise_factor=config_dict['measurements']['noise_factor'],
        seed=config_dict['measurements'].get('seed', None),
        gamma=config_dict['algorithm']['gamma'],
        alpha=config_dict['algorithm']['alpha'],
        max_iterations=config_dict['algorithm']['max_iterations'],
        tolerance=config_dict['algorithm']['tolerance']
    )


def visualize_network(config_path: str, results_path: str = None):
    """
    Visualize the sensor network topology and localization results
    
    Args:
        config_path: Path to configuration YAML file
        results_path: Optional path to results JSON file
    """
    # Load configuration
    config_dict = load_config(config_path)
    config = dict_to_config(config_dict)
    
    # Create algorithm instance and generate network
    mps = MPSAlgorithm(config)
    mps.generate_network()
    
    # Load results if provided
    estimated_positions = None
    if results_path and Path(results_path).exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
            if 'results' in results and 'final_positions' in results['results']:
                estimated_positions = {
                    int(k): np.array(v) 
                    for k, v in results['results']['final_positions'].items()
                }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Network Topology
    ax1 = plt.subplot(131)
    
    # Plot communication links
    n = config.n_sensors
    for i in range(n):
        for j in range(i+1, n):
            if mps.adjacency[i, j] > 0:
                x = [mps.true_positions[i][0], mps.true_positions[j][0]]
                y = [mps.true_positions[i][1], mps.true_positions[j][1]]
                ax1.plot(x, y, 'gray', alpha=0.3, linewidth=0.5)
    
    # Plot sensors
    sensor_positions = np.array([mps.true_positions[i] for i in range(n)])
    ax1.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
               c='blue', s=100, alpha=0.7, label='Sensors', zorder=5)
    
    # Plot anchors
    if config.n_anchors > 0:
        ax1.scatter(mps.anchor_positions[:, 0], mps.anchor_positions[:, 1],
                   c='red', s=150, marker='^', alpha=0.9, label='Anchors', zorder=6)
        
        # Show anchor-sensor connections
        for i in range(n):
            if i in mps.anchor_distances:
                for k in mps.anchor_distances[i]:
                    x = [mps.true_positions[i][0], mps.anchor_positions[k][0]]
                    y = [mps.true_positions[i][1], mps.anchor_positions[k][1]]
                    ax1.plot(x, y, 'r--', alpha=0.2, linewidth=0.5)
    
    # Add sensor labels
    for i in range(n):
        ax1.text(mps.true_positions[i][0], mps.true_positions[i][1], 
                str(i), fontsize=8, ha='center', va='center')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Network Topology\n({n} sensors, {config.n_anchors} anchors)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Subplot 2: True vs Estimated Positions
    ax2 = plt.subplot(132)
    
    # Plot true positions
    ax2.scatter(sensor_positions[:, 0], sensor_positions[:, 1],
               c='blue', s=100, alpha=0.5, label='True Positions', marker='o')
    
    # Plot estimated positions if available
    if estimated_positions:
        est_positions = np.array([estimated_positions[i] for i in range(n)])
        ax2.scatter(est_positions[:, 0], est_positions[:, 1],
                   c='green', s=100, alpha=0.7, label='Estimated', marker='x')
        
        # Draw error lines
        for i in range(n):
            x = [mps.true_positions[i][0], estimated_positions[i][0]]
            y = [mps.true_positions[i][1], estimated_positions[i][1]]
            ax2.plot(x, y, 'r-', alpha=0.3, linewidth=1)
    
    # Plot anchors
    if config.n_anchors > 0:
        ax2.scatter(mps.anchor_positions[:, 0], mps.anchor_positions[:, 1],
                   c='red', s=150, marker='^', alpha=0.9, label='Anchors')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('True vs Estimated Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Subplot 3: Error Distribution
    ax3 = plt.subplot(133)
    
    if estimated_positions:
        errors = []
        for i in range(n):
            error = np.linalg.norm(mps.true_positions[i] - estimated_positions[i])
            errors.append(error)
        
        # Create bar chart of errors
        indices = np.arange(n)
        colors = ['green' if e < 0.1 else 'orange' if e < 0.2 else 'red' for e in errors]
        bars = ax3.bar(indices, errors, color=colors, alpha=0.7)
        
        # Add average line
        avg_error = np.mean(errors)
        ax3.axhline(y=avg_error, color='blue', linestyle='--', 
                   label=f'Avg: {avg_error:.3f}', linewidth=2)
        
        # Add RMSE
        rmse = np.sqrt(np.mean(np.square(errors)))
        ax3.axhline(y=rmse, color='red', linestyle='--', 
                   label=f'RMSE: {rmse:.3f}', linewidth=2)
        
        ax3.set_xlabel('Sensor ID')
        ax3.set_ylabel('Position Error')
        ax3.set_title('Localization Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No results available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Localization Error Distribution')
    
    plt.suptitle(f'Sensor Network Localization Analysis\n'
                 f'Config: {Path(config_path).name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_dir = config_dict['output']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig_path = Path(output_dir) / 'network_visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    
    plt.show()
    
    # Print statistics
    if estimated_positions:
        print("\nLocalization Statistics:")
        print(f"  Average Error: {avg_error:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Max Error: {max(errors):.4f}")
        print(f"  Min Error: {min(errors):.4f}")
        print(f"  Sensors within 0.1: {sum(1 for e in errors if e < 0.1)}/{n}")
        print(f"  Sensors within 0.2: {sum(1 for e in errors if e < 0.2)}/{n}")


def main():
    parser = argparse.ArgumentParser(description='Visualize sensor network and results')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results JSON file (optional)')
    
    args = parser.parse_args()
    
    visualize_network(args.config, args.results)


if __name__ == "__main__":
    main()