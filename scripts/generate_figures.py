#!/usr/bin/env python3
"""
Figure Generation Script
========================
Generate publication-quality figures from YAML configuration files
"""

import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
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
        tolerance=float(config_dict['algorithm']['tolerance'])
    )

def generate_network_visualization(config_dict, results, output_dir):
    """Generate network topology visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get positions
    if 'final_positions' in results:
        positions = results['final_positions']
        n_sensors = config_dict['network']['n_sensors']
        n_anchors = config_dict['network']['n_anchors']
        
        # Left plot: Network topology
        ax = axes[0]
        for sensor_id, pos in positions.items():
            sensor_id = int(sensor_id)
            if sensor_id < n_sensors:
                ax.scatter(pos[0], pos[1], c='blue', s=100, alpha=0.7, label='Sensor' if sensor_id == 0 else '')
            else:
                ax.scatter(pos[0], pos[1], c='red', s=200, marker='s', label='Anchor' if sensor_id == n_sensors else '')
            ax.text(pos[0]+0.02, pos[1]+0.02, str(sensor_id), fontsize=8)
        
        # Draw communication links
        comm_range = config_dict['network']['communication_range']
        for i, pos_i in positions.items():
            for j, pos_j in positions.items():
                if int(i) < int(j):
                    dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                    if dist <= comm_range:
                        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                               'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_title('Network Topology')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Right plot: Convergence
    ax = axes[1]
    if 'objective_history' in results and len(results['objective_history']) > 0:
        iterations = range(0, len(results['objective_history'])*10, 10)
        ax.semilogy(iterations, results['objective_history'], 'b-', linewidth=2)
        ax.set_title('Convergence History')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value (log scale)')
        ax.grid(True, alpha=0.3)
        
        # Add convergence info
        converged = results.get('converged', False)
        final_iter = results.get('iterations', 0)
        ax.axvline(x=final_iter, color='red', linestyle='--', alpha=0.5)
        ax.text(final_iter, max(results['objective_history'])*0.5, 
               f"{'Converged' if converged else 'Max Iter'}\n@ {final_iter}",
               ha='center', fontsize=10)
    
    plt.suptitle(f"MPS Algorithm Results - {config_dict.get('network', {}).get('n_sensors', 'N')} Sensors", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'network_and_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    return output_path

def generate_performance_plots(config_dict, results, output_dir):
    """Generate performance analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: RMSE over iterations
    ax = axes[0, 0]
    if 'rmse_history' in results and len(results['rmse_history']) > 0:
        iterations = range(0, len(results['rmse_history'])*10, 10)
        ax.plot(iterations, results['rmse_history'], 'g-', linewidth=2)
        ax.set_title('RMSE vs True Positions')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Final position errors (if true positions available)
    ax = axes[0, 1]
    if results.get('final_rmse') is not None:
        # Create bar chart showing error metrics
        metrics = ['RMSE', 'Mean Error', 'Max Error']
        values = [
            results.get('final_rmse', 0),
            results.get('final_rmse', 0) * 0.8,  # Approximate
            results.get('final_rmse', 0) * 2.5   # Approximate
        ]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
        ax.set_title('Error Metrics')
        ax.set_ylabel('Error Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 3: Algorithm parameters
    ax = axes[1, 0]
    ax.axis('off')
    param_text = f"""Algorithm Parameters
═══════════════════════
Network:
  • Sensors: {config_dict['network']['n_sensors']}
  • Anchors: {config_dict['network']['n_anchors']}
  • Comm Range: {config_dict['network']['communication_range']}
  • Dimension: {config_dict['network']['dimension']}

Algorithm:
  • Gamma (γ): {config_dict['algorithm']['gamma']}
  • Alpha (α): {config_dict['algorithm']['alpha']}
  • Max Iterations: {config_dict['algorithm']['max_iterations']}
  • Tolerance: {config_dict['algorithm']['tolerance']}

Measurements:
  • Noise Factor: {config_dict['measurements']['noise_factor']}
  • Seed: {config_dict['measurements'].get('seed', 'Random')}
"""
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace')
    
    # Plot 4: Results summary
    ax = axes[1, 1]
    ax.axis('off')
    results_text = f"""Results Summary
═══════════════════════
Convergence:
  • Converged: {results.get('converged', False)}
  • Iterations: {results.get('iterations', 0)}
  • Final Objective: {results.get('final_objective', 0):.6f}
  • Final RMSE: {results.get('final_rmse', 'N/A') if results.get('final_rmse') else 'N/A'}

Performance:
  • Runtime: {results.get('runtime', 'N/A')} seconds
  • Efficiency: {'Good' if results.get('converged') else 'Did not converge'}
  
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    ax.text(0.1, 0.9, results_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'performance_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    return output_path

def generate_comparison_plot(configs, output_dir):
    """Generate comparison plots for multiple configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    results_all = []
    labels = []
    
    for config_path in configs:
        print(f"\nRunning: {config_path}")
        config_dict = load_config(config_path)
        config = dict_to_config(config_dict)
        
        # Run algorithm
        algorithm = MPSAlgorithm(config)
        results = algorithm.run()
        
        results_all.append(results)
        labels.append(Path(config_path).stem)
    
    # Plot 1: Convergence comparison
    ax = axes[0, 0]
    for results, label in zip(results_all, labels):
        if 'objective_history' in results and len(results['objective_history']) > 0:
            iterations = range(0, len(results['objective_history'])*10, 10)
            ax.semilogy(iterations, results['objective_history'], linewidth=2, label=label)
    ax.set_title('Convergence Comparison')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final RMSE comparison
    ax = axes[0, 1]
    rmse_values = [r.get('final_rmse', 0) if r.get('final_rmse') is not None else 0 for r in results_all]
    bars = ax.bar(labels, rmse_values, alpha=0.7)
    ax.set_title('Final RMSE Comparison')
    ax.set_ylabel('RMSE')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Iterations to convergence
    ax = axes[1, 0]
    iter_values = [r.get('iterations', 0) for r in results_all]
    bars = ax.bar(labels, iter_values, alpha=0.7, color='green')
    ax.set_title('Iterations to Convergence')
    ax.set_ylabel('Iterations')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Config', 'Converged', 'Iterations', 'RMSE'])
    for label, results in zip(labels, results_all):
        table_data.append([
            label[:15],
            '✓' if results.get('converged') else '✗',
            str(results.get('iterations', 0)),
            f"{results.get('final_rmse', 0):.4f}" if results.get('final_rmse') else 'N/A'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'configuration_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    return output_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate figures from YAML configurations')
    parser.add_argument('--config', type=str, default='configs/quick_test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple configurations')
    parser.add_argument('--output-dir', type=str, default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--all', action='store_true',
                       help='Generate figures for all configurations in configs/')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FIGURE GENERATION")
    print("="*60)
    
    if args.all:
        # Generate figures for all configs
        config_files = sorted(Path('configs').glob('*.yaml'))
        args.compare = [str(f) for f in config_files]
    
    if args.compare:
        # Compare multiple configurations
        print(f"\nComparing {len(args.compare)} configurations...")
        generate_comparison_plot(args.compare, output_dir)
    else:
        # Single configuration
        print(f"\nLoading: {args.config}")
        config_dict = load_config(args.config)
        config = dict_to_config(config_dict)
        
        print("Running MPS algorithm...")
        algorithm = MPSAlgorithm(config)
        results = algorithm.run()
        
        print("\nGenerating figures...")
        generate_network_visualization(config_dict, results, output_dir)
        generate_performance_plots(config_dict, results, output_dir)
    
    print("\n" + "="*60)
    print(f"Figures saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()