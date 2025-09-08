#!/usr/bin/env python3
"""
Main entry point for running MPS algorithm
Supports both single-process and MPI distributed execution
"""

import argparse
import yaml
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict_to_config(config_dict: dict, enable_carrier_phase: bool = False) -> MPSConfig:
    """Convert configuration dictionary to MPSConfig object"""
    from src.core.mps_core.algorithm import CarrierPhaseConfig
    
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


def save_results(results: dict, config: dict, output_dir: str):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mps_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Combine results and configuration
    output = {
        'configuration': config,
        'results': {
            'converged': results['converged'],
            'iterations': results['iterations'],
            'final_objective': results['final_objective'],
            'final_rmse': results['final_rmse'],
            'objective_history': results['objective_history'],
            'rmse_history': results['rmse_history']
        },
        'timestamp': timestamp
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if 'final_positions' in results:
        output['results']['final_positions'] = {
            str(k): v.tolist() for k, v in results['final_positions'].items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def print_results(results: dict, verbose: bool = True):
    """Print results to console"""
    print("\n" + "="*60)
    print("MPS ALGORITHM RESULTS")
    print("="*60)
    
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Final Objective (Distance Error): {results['final_objective']:.6f}")
    
    if results['final_rmse'] is not None:
        print(f"Final RMSE (vs True Positions): {results['final_rmse']:.6f}")
    
    if verbose and len(results['objective_history']) > 0:
        print(f"\nObjective History:")
        for i, obj in enumerate(results['objective_history'][-5:]):
            print(f"  Iteration {(i+1)*10}: {obj:.6f}")
    
    print("="*60 + "\n")


def visualize_results(results: dict, config: dict):
    """Simple visualization of final positions"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot sensors
        positions = results['final_positions']
        for sensor_id, pos in positions.items():
            ax.scatter(pos[0], pos[1], c='blue', s=50, alpha=0.7)
            ax.text(pos[0], pos[1], str(sensor_id), fontsize=8)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Final Sensor Positions (MPS Algorithm)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save figure
        output_dir = config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'final_positions.png')
        plt.savefig(fig_path, dpi=150)
        print(f"Visualization saved to: {fig_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run MPS Algorithm for Sensor Network Localization')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file (YAML)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize final positions')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--carrier-phase', action='store_true',
                       help='Enable carrier phase synchronization (Nanzer approach for mm-level accuracy)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_dict = load_config(args.config)
    config = dict_to_config(config_dict, enable_carrier_phase=args.carrier_phase)
    
    # Print configuration summary
    if not args.quiet:
        print(f"\nNetwork Configuration:")
        print(f"  Sensors: {config.n_sensors}")
        print(f"  Anchors: {config.n_anchors}")
        print(f"  Communication Range: {config.communication_range}")
        print(f"  Noise Factor: {config.noise_factor}")
        print(f"\nAlgorithm Parameters:")
        print(f"  Gamma: {config.gamma}")
        print(f"  Alpha: {config.alpha}")
        print(f"  Max Iterations: {config.max_iterations}")
        print(f"  Tolerance: {config.tolerance}")
    
    # Create and run algorithm
    print(f"\nInitializing MPS algorithm...")
    mps = MPSAlgorithm(config)
    
    print("Generating network...")
    mps.generate_network()
    
    print("Running MPS algorithm...")
    start_time = time.time()
    results = mps.run()
    elapsed_time = time.time() - start_time
    
    print(f"Algorithm completed in {elapsed_time:.2f} seconds")
    
    # Print results
    print_results(results, verbose=not args.quiet)
    
    # Save results
    if not args.no_save and config_dict['output']['save_results']:
        save_results(results, config_dict, config_dict['output']['output_dir'])
    
    # Visualize if requested
    if args.visualize:
        visualize_results(results, config_dict)
    
    return results


if __name__ == "__main__":
    main()