#!/usr/bin/env python3
"""
Visualization of True vs Estimated Positions
=============================================
Generate detailed comparison plots showing localization accuracy
"""

import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig, CarrierPhaseConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict_to_config(config_dict: dict, enable_carrier_phase: bool = False) -> MPSConfig:
    """Convert configuration dictionary to MPSConfig object"""
    # Create carrier phase config if present in YAML or enabled via command line
    carrier_phase_config = None
    if enable_carrier_phase or 'carrier_phase' in config_dict:
        cp_dict = config_dict.get('carrier_phase', {})
        carrier_phase_config = CarrierPhaseConfig(
            enable=True,
            frequency_ghz=cp_dict.get('frequency_ghz', 2.4),
            phase_noise_milliradians=cp_dict.get('phase_noise_milliradians', 1.0),
            frequency_stability_ppb=cp_dict.get('frequency_stability_ppb', 0.1),
            coarse_time_accuracy_ns=cp_dict.get('coarse_time_accuracy_ns', 1.0)
        )
    
    return MPSConfig(
        n_sensors=config_dict['network']['n_sensors'],
        n_anchors=config_dict['network']['n_anchors'],
        communication_range=config_dict['network'].get('communication_range', 0.3),
        dimension=config_dict['network'].get('dimension', 2),
        noise_factor=config_dict.get('measurements', {}).get('noise_factor', 0.001 if carrier_phase_config else 0.05),
        seed=config_dict.get('measurements', {}).get('seed', 42),
        gamma=config_dict['algorithm']['gamma'],
        alpha=config_dict['algorithm']['alpha'],
        max_iterations=config_dict['algorithm']['max_iterations'],
        tolerance=float(config_dict['algorithm']['tolerance']),
        carrier_phase=carrier_phase_config
    )


def visualize_true_vs_estimated(config_path: str, output_dir: str = 'figures/', 
                                enable_carrier_phase: bool = False, save_data: bool = False):
    """
    Generate comprehensive visualization of true vs estimated positions
    """
    # Load configuration
    config_dict = load_config(config_path)
    config = dict_to_config(config_dict, enable_carrier_phase)
    
    print(f"\nRunning MPS algorithm with config: {config_path}")
    if config.carrier_phase and config.carrier_phase.enable:
        print(f"  Carrier phase enabled: {config.carrier_phase.frequency_ghz} GHz")
        print(f"  Expected accuracy: {config.carrier_phase.ranging_accuracy_m*1000:.3f} mm")
    
    # Run algorithm
    algorithm = MPSAlgorithm(config)
    algorithm.generate_network()  # This creates true positions
    results = algorithm.run()
    
    # Extract positions
    true_positions = algorithm.true_positions
    estimated_positions = results['final_positions']
    anchor_positions = algorithm.anchor_positions
    anchor_indices = set(range(config.n_sensors, config.n_sensors + config.n_anchors))
    
    # Calculate errors
    errors = {}
    errors_mm = []
    for i in range(config.n_sensors):
        if i in estimated_positions and i in true_positions:
            error = np.linalg.norm(estimated_positions[i] - true_positions[i])
            errors[i] = error
            errors_mm.append(error * 1000)  # Convert to mm
    
    # Statistics
    rmse = np.sqrt(np.mean([e**2 for e in errors.values()]))
    rmse_mm = rmse * 1000
    max_error_mm = max(errors_mm) if errors_mm else 0
    mean_error_mm = np.mean(errors_mm) if errors_mm else 0
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main comparison plot (large, left side)
    ax1 = fig.add_subplot(gs[:, 0:2])
    
    # Plot true positions
    for i in range(config.n_sensors):
        if i in true_positions:
            ax1.scatter(true_positions[i][0], true_positions[i][1], 
                       c='blue', s=100, alpha=0.5, marker='o', label='True' if i == 0 else '')
            
    # Plot estimated positions
    for i in range(config.n_sensors):
        if i in estimated_positions:
            ax1.scatter(estimated_positions[i][0], estimated_positions[i][1], 
                       c='green', s=80, alpha=0.7, marker='^', label='Estimated' if i == 0 else '')
    
    # Plot anchors
    for k in range(config.n_anchors):
        ax1.scatter(anchor_positions[k][0], anchor_positions[k][1], 
                   c='red', s=200, marker='s', label='Anchor' if k == 0 else '')
        ax1.text(anchor_positions[k][0], anchor_positions[k][1] - 0.05, 
                f'A{k}', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Draw error vectors
    for i in range(config.n_sensors):
        if i in true_positions and i in estimated_positions:
            # Error vector
            ax1.plot([true_positions[i][0], estimated_positions[i][0]], 
                    [true_positions[i][1], estimated_positions[i][1]], 
                    'r-', alpha=0.3, linewidth=1)
            
            # Add node labels
            ax1.text(true_positions[i][0], true_positions[i][1] + 0.03, 
                    f'{i}', ha='center', va='bottom', fontsize=7, color='blue')
    
    # Draw communication range circles for anchors
    if config.communication_range < 1.0:  # Only if reasonable size
        for k in range(config.n_anchors):
            circle = Circle((anchor_positions[k][0], anchor_positions[k][1]), 
                          config.communication_range, fill=False, 
                          edgecolor='red', alpha=0.1, linestyle='--')
            ax1.add_patch(circle)
    
    ax1.set_title('True vs Estimated Positions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add RMSE annotation
    ax1.text(0.02, 0.98, f'RMSE: {rmse_mm:.2f} mm', 
            transform=ax1.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    # 2. Error distribution histogram
    ax2 = fig.add_subplot(gs[0, 2])
    if errors_mm:
        ax2.hist(errors_mm, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(mean_error_mm, color='red', linestyle='--', label=f'Mean: {mean_error_mm:.2f} mm')
        ax2.axvline(rmse_mm, color='green', linestyle='-', label=f'RMSE: {rmse_mm:.2f} mm')
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Localization Error (mm)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    # Determine if using carrier phase
    using_carrier = config.carrier_phase and config.carrier_phase.enable
    
    metrics_text = f"""PERFORMANCE METRICS
{'='*25}
Algorithm: MPS
Mode: {'Carrier Phase' if using_carrier else 'Standard'}

Network:
  Sensors: {config.n_sensors}
  Anchors: {config.n_anchors}
  Comm Range: {config.communication_range:.2f} m

Accuracy:
  RMSE: {rmse_mm:.3f} mm
  Mean: {mean_error_mm:.3f} mm
  Max: {max_error_mm:.3f} mm
  Min: {min(errors_mm) if errors_mm else 0:.3f} mm
  Std Dev: {np.std(errors_mm) if errors_mm else 0:.3f} mm

Algorithm:
  Converged: {results.get('converged', False)}
  Iterations: {results.get('iterations', 0)}
  Tolerance: {config.tolerance}
"""
    
    if using_carrier:
        metrics_text += f"""
Carrier Phase:
  Freq: {config.carrier_phase.frequency_ghz} GHz
  λ: {config.carrier_phase.wavelength*100:.1f} cm
  Phase Noise: {config.carrier_phase.phase_noise_milliradians} mrad
  Expected: {config.carrier_phase.ranging_accuracy_m*1000:.3f} mm
"""
    
    ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Add color coding for performance
    if rmse_mm < 1.0:
        status = "EXCELLENT (<1mm)"
        color = 'green'
    elif rmse_mm < 10.0:
        status = "VERY GOOD (<10mm)"
        color = 'blue'
    elif rmse_mm < 100.0:
        status = "GOOD (<100mm)"
        color = 'orange'
    else:
        status = "NEEDS IMPROVEMENT"
        color = 'red'
    
    ax3.text(0.5, 0.1, status, transform=ax3.transAxes, fontsize=14,
            fontweight='bold', color=color, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    # Overall title
    config_name = Path(config_path).stem
    plt.suptitle(f'Localization Results - {config_name}', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = Path(output_dir) / f'true_vs_estimated_{config_name}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save data if requested
    if save_data:
        data_output = {
            'config': config_path,
            'timestamp': datetime.now().isoformat(),
            'carrier_phase_enabled': using_carrier,
            'network': {
                'n_sensors': config.n_sensors,
                'n_anchors': config.n_anchors,
                'communication_range': config.communication_range
            },
            'results': {
                'rmse_mm': rmse_mm,
                'mean_error_mm': mean_error_mm,
                'max_error_mm': max_error_mm,
                'min_error_mm': min(errors_mm) if errors_mm else 0,
                'std_error_mm': np.std(errors_mm) if errors_mm else 0,
                'converged': results.get('converged', False),
                'iterations': results.get('iterations', 0)
            },
            'positions': {
                'true': {str(k): v.tolist() for k, v in true_positions.items()},
                'estimated': {str(k): v.tolist() for k, v in estimated_positions.items()},
                'anchors': anchor_positions.tolist()
            },
            'errors': {str(k): v for k, v in errors.items()}
        }
        
        data_path = Path(output_dir) / f'true_vs_estimated_{config_name}.json'
        with open(data_path, 'w') as f:
            json.dump(data_output, f, indent=2)
        print(f"Data saved to: {data_path}")
    
    plt.show()
    
    return rmse_mm


def compare_multiple_configs(config_paths: list, output_dir: str = 'figures/', 
                            enable_carrier_phase: bool = False):
    """Compare true vs estimated for multiple configurations"""
    
    fig, axes = plt.subplots(2, len(config_paths), figsize=(6*len(config_paths), 10))
    if len(config_paths) == 1:
        axes = axes.reshape(-1, 1)
    
    rmse_results = []
    
    for idx, config_path in enumerate(config_paths):
        config_name = Path(config_path).stem
        print(f"\nProcessing {config_name}...")
        
        # Load and run
        config_dict = load_config(config_path)
        config = dict_to_config(config_dict, enable_carrier_phase)
        
        algorithm = MPSAlgorithm(config)
        algorithm.generate_network()
        results = algorithm.run()
        
        true_positions = algorithm.true_positions
        estimated_positions = results['final_positions']
        anchor_positions = algorithm.anchor_positions
        
        # Calculate errors
        errors_mm = []
        for i in range(config.n_sensors):
            if i in estimated_positions and i in true_positions:
                error = np.linalg.norm(estimated_positions[i] - true_positions[i])
                errors_mm.append(error * 1000)
        
        rmse_mm = np.sqrt(np.mean(np.square(errors_mm))) if errors_mm else 0
        rmse_results.append((config_name, rmse_mm))
        
        # Top plot: positions
        ax = axes[0, idx]
        
        # Plot positions
        for i in range(config.n_sensors):
            if i in true_positions:
                ax.scatter(true_positions[i][0], true_positions[i][1], 
                          c='blue', s=50, alpha=0.5, marker='o')
            if i in estimated_positions:
                ax.scatter(estimated_positions[i][0], estimated_positions[i][1], 
                          c='green', s=40, alpha=0.7, marker='^')
                # Error line
                if i in true_positions:
                    ax.plot([true_positions[i][0], estimated_positions[i][0]], 
                           [true_positions[i][1], estimated_positions[i][1]], 
                           'r-', alpha=0.2, linewidth=0.5)
        
        # Anchors
        for k in range(config.n_anchors):
            ax.scatter(anchor_positions[k][0], anchor_positions[k][1], 
                      c='red', s=100, marker='s')
        
        ax.set_title(f'{config_name}\nRMSE: {rmse_mm:.2f} mm')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Bottom plot: error distribution
        ax = axes[1, idx]
        if errors_mm:
            ax.hist(errors_mm, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(errors_mm), color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Configuration Comparison - True vs Estimated Positions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'comparison_true_vs_estimated.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("RMSE SUMMARY")
    print("="*50)
    for name, rmse in rmse_results:
        print(f"{name:30} {rmse:8.3f} mm")
    print("="*50)
    
    plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Visualize True vs Estimated Positions')
    parser.add_argument('--config', type=str, default='configs/quick_test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple configurations')
    parser.add_argument('--output-dir', type=str, default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--carrier-phase', action='store_true',
                       help='Enable carrier phase synchronization')
    parser.add_argument('--save-data', action='store_true',
                       help='Save position data to JSON')
    parser.add_argument('--all', action='store_true',
                       help='Run all configurations')
    
    args = parser.parse_args()
    
    if args.all:
        config_files = sorted(Path('configs').glob('*.yaml'))
        args.compare = [str(f) for f in config_files]
    
    if args.compare:
        compare_multiple_configs(args.compare, args.output_dir, args.carrier_phase)
    else:
        rmse = visualize_true_vs_estimated(args.config, args.output_dir, 
                                          args.carrier_phase, args.save_data)
        print(f"\nFinal RMSE: {rmse:.3f} mm")


if __name__ == "__main__":
    main()