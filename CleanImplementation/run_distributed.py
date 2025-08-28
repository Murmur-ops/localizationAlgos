#!/usr/bin/env python3
"""
MPI launcher script for distributed MPS algorithm
Usage: mpirun -n <num_processes> python run_distributed.py --config <config_file>
"""

import argparse
import yaml
import json
import os
import sys
import time
from pathlib import Path
from mpi4py import MPI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from mps_core import DistributedMPS, MPSConfig


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


def save_results_distributed(results: dict, config: dict, output_dir: str, rank: int):
    """Save results from rank 0"""
    if rank != 0 or results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"mps_distributed_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Combine results and configuration
    output = {
        'configuration': config,
        'mpi': {
            'n_processes': results['n_processes']
        },
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
    
    # Convert numpy arrays to lists
    if 'final_positions' in results:
        output['results']['final_positions'] = {
            str(k): v.tolist() for k, v in results['final_positions'].items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[Rank 0] Results saved to: {filepath}")
    return filepath


def print_results_distributed(results: dict, rank: int, size: int, elapsed_time: float):
    """Print results from rank 0"""
    if rank != 0 or results is None:
        return
    
    print("\n" + "="*60)
    print("DISTRIBUTED MPS ALGORITHM RESULTS")
    print("="*60)
    
    print(f"MPI Processes: {size}")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Final Objective (Distance Error): {results['final_objective']:.6f}")
    
    if results['final_rmse'] is not None:
        print(f"Final RMSE (vs True Positions): {results['final_rmse']:.6f}")
    
    print(f"Total Runtime: {elapsed_time:.2f} seconds")
    print(f"Time per iteration: {elapsed_time / results['iterations'] * 1000:.2f} ms")
    
    if len(results['objective_history']) > 0:
        print(f"\nObjective History (last 5):")
        for i, obj in enumerate(results['objective_history'][-5:]):
            print(f"  Step {len(results['objective_history'])-5+i+1}: {obj:.6f}")
    
    print("="*60 + "\n")


def main():
    """Main execution function for distributed MPS"""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # Parse arguments (same on all ranks)
    parser = argparse.ArgumentParser(description='Run Distributed MPS Algorithm')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file (YAML)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Load configuration
    if rank == 0:
        print(f"[Rank 0] Loading configuration from: {args.config}")
        print(f"[Rank 0] Running with {size} MPI processes")
    
    config_dict = load_config(args.config)
    config = dict_to_config(config_dict)
    
    # Print configuration summary (rank 0 only)
    if rank == 0 and not args.quiet:
        print(f"\n[Rank 0] Network Configuration:")
        print(f"  Sensors: {config.n_sensors}")
        print(f"  Anchors: {config.n_anchors}")
        print(f"  Communication Range: {config.communication_range}")
        print(f"  Noise Factor: {config.noise_factor}")
        print(f"\n[Rank 0] Algorithm Parameters:")
        print(f"  Gamma: {config.gamma}")
        print(f"  Alpha: {config.alpha}")
        print(f"  Max Iterations: {config.max_iterations}")
        print(f"  Tolerance: {config.tolerance}")
    
    # Create distributed MPS instance
    if rank == 0:
        print(f"\n[Rank 0] Initializing Distributed MPS algorithm...")
    
    distributed_mps = DistributedMPS(config, comm)
    
    # Synchronize before starting
    comm.Barrier()
    
    if rank == 0:
        print(f"[Rank 0] Running distributed MPS algorithm...")
    
    start_time = time.time()
    results = distributed_mps.run_distributed()
    elapsed_time = time.time() - start_time
    
    # Synchronize after completion
    comm.Barrier()
    
    if rank == 0:
        print(f"[Rank 0] Algorithm completed in {elapsed_time:.2f} seconds")
    
    # Print and save results (rank 0 only)
    print_results_distributed(results, rank, size, elapsed_time)
    
    if not args.no_save and config_dict['output']['save_results']:
        save_results_distributed(results, config_dict, 
                                config_dict['output']['output_dir'], rank)
    
    # Finalize MPI
    if rank == 0:
        print("[Rank 0] MPI execution completed successfully")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Ensure proper MPI finalization
    MPI.Finalize()