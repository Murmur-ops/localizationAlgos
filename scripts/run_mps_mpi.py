#!/usr/bin/env python3
"""
MPI Launcher Script for Distributed MPS Algorithm
Usage: mpirun -n <num_processes> python run_mps_mpi.py --config <config.yaml>

This script provides a command-line interface for running the distributed
MPS algorithm with YAML configuration support.
"""

import argparse
import sys
import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from mpi4py import MPI

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_distributed import DistributedMPS, DistributedMPSConfig
from src.core.mps_core.mps_full_algorithm import NetworkData, create_network_data


def setup_logging(rank: int, verbose: bool = False):
    """Setup logging for MPI processes"""
    import logging
    
    # Configure logging level based on rank and verbosity
    if rank == 0:
        level = logging.DEBUG if verbose else logging.INFO
    else:
        level = logging.WARNING if verbose else logging.ERROR
    
    # Format with rank information
    format_str = f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Distributed MPS Algorithm with MPI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 4 MPI processes using default config
  mpirun -n 4 python run_mps_mpi.py --config configs/mpi/mpi_medium.yaml
  
  # Run with parameter overrides
  mpirun -n 8 python run_mps_mpi.py --config configs/base.yaml \\
    --override network.n_sensors=100 algorithm.max_iterations=500
  
  # Run in benchmark mode with profiling
  mpirun -n 16 python run_mps_mpi.py --config configs/mpi/mpi_benchmark.yaml \\
    --profile --output-dir results/benchmark/
  
  # Run with multiple config files (merged in order)
  mpirun -n 4 python run_mps_mpi.py \\
    --config configs/base.yaml configs/mpi/mpi_settings.yaml
        """
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        nargs='+',
        default=['configs/default.yaml'],
        help='Path(s) to YAML configuration file(s). Multiple files are merged.'
    )
    
    parser.add_argument(
        '--override', '-o',
        action='append',
        default=[],
        help='Override configuration parameters (e.g., network.n_sensors=50)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-d',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    # Execution modes
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run in benchmark mode (minimal output, timing focus)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running algorithm'
    )
    
    # Display options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots after completion (rank 0 only)'
    )
    
    return parser.parse_args()


def process_overrides(override_list):
    """Process command-line overrides into dictionary"""
    overrides = {}
    
    for override in override_list:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}")
        
        key, value = override.split('=', 1)
        
        # Try to parse value as appropriate type
        try:
            # Try as number
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Try as boolean
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            # Otherwise keep as string
        
        overrides[key] = value
    
    return overrides


def generate_network_from_config(config: Dict[str, Any], rank: int) -> NetworkData:
    """Generate network data from configuration"""
    network_config = config['network']
    measurement_config = config['measurements']
    
    # Create network on rank 0 and broadcast
    if rank == 0:
        network = create_network_data(
            n_sensors=network_config['n_sensors'],
            n_anchors=network_config['n_anchors'],
            dimension=network_config['dimension'],
            communication_range=network_config['communication_range'],
            measurement_noise=measurement_config['noise_factor'],
            carrier_phase=measurement_config.get('carrier_phase', False)
        )
    else:
        network = None
    
    # Broadcast network data to all processes
    comm = MPI.COMM_WORLD
    network = comm.bcast(network, root=0)
    
    return network


def save_results(results: Dict[str, Any], config: Dict[str, Any], 
                output_dir: str, rank: int):
    """Save results to file (rank 0 only)"""
    if rank != 0:
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save main results as JSON
    results_file = output_path / f"mps_mpi_results_{timestamp}.json"
    
    # Prepare JSON-serializable results
    json_results = {
        'timestamp': timestamp,
        'configuration': config,
        'converged': results['converged'],
        'iterations': results['iterations'],
        'best_error': float(results['best_error']),
        'final_error': float(results['final_error']) if results['final_error'] else None,
        'n_processes': results['n_processes'],
        'timing': results['timing_stats'],
        'objectives': [float(x) for x in results['objectives']],
        'errors': [float(x) for x in results['errors']]
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save positions if available
    if results.get('best_positions') is not None:
        positions_file = output_path / f"mps_mpi_positions_{timestamp}.npy"
        np.save(positions_file, results['best_positions'])
        print(f"Positions saved to: {positions_file}")
    
    # Save full results as pickle for detailed analysis
    pickle_file = output_path / f"mps_mpi_full_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to: {pickle_file}")


def plot_results(results: Dict[str, Any], config: Dict[str, Any], rank: int):
    """Generate plots from results (rank 0 only)"""
    if rank != 0:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Objective convergence
        ax = axes[0, 0]
        iterations = range(0, len(results['objectives']) * 10, 10)
        ax.semilogy(iterations, results['objectives'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Function Convergence')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error convergence
        ax = axes[0, 1]
        ax.semilogy(iterations, results['errors'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('Position Error Convergence')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Timing breakdown
        ax = axes[1, 0]
        timing = results['timing_stats']
        categories = [k for k in timing.keys() if k != 'total']
        values = [timing[k] for k in categories]
        colors = plt.cm.Set3(range(len(categories)))
        
        wedges, texts, autotexts = ax.pie(
            values, labels=categories, colors=colors,
            autopct='%1.1f%%', startangle=90
        )
        ax.set_title('Timing Breakdown')
        
        # Plot 4: Final positions (if 2D)
        ax = axes[1, 1]
        if results.get('best_positions') is not None and config['network']['dimension'] == 2:
            positions = results['best_positions']
            n_sensors = config['network']['n_sensors']
            
            # Plot sensors
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c='blue', s=50, alpha=0.7, label='Sensors')
            
            # Add sensor IDs
            for i in range(n_sensors):
                ax.text(positions[i, 0], positions[i, 1], str(i),
                       fontsize=8, ha='center', va='bottom')
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Final Sensor Positions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'Positions not available\nor not 2D',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Overall title
        fig.suptitle(
            f'Distributed MPS Results (n={config["network"]["n_sensors"]}, '
            f'MPI processes={results["n_processes"]})',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - skipping plots")


def run_benchmark(config: Dict[str, Any], network: NetworkData, comm: MPI.Comm):
    """Run algorithm in benchmark mode with multiple repetitions"""
    rank = comm.rank
    
    # Benchmark configuration
    n_runs = 5
    warmup_runs = 2
    
    if rank == 0:
        print("\n" + "="*60)
        print("BENCHMARK MODE")
        print("="*60)
        print(f"Warmup runs: {warmup_runs}")
        print(f"Benchmark runs: {n_runs}")
        print(f"MPI processes: {comm.size}")
        print(f"Network size: {config['network']['n_sensors']} sensors")
        print("="*60)
    
    # Create distributed MPS instance
    mps_config = ConfigLoader().to_mps_config(config, distributed=True)
    mps = DistributedMPS(mps_config, network, comm)
    
    # Warmup runs
    if rank == 0:
        print("\nRunning warmup...")
    
    for i in range(warmup_runs):
        mps.run_distributed(max_iterations=10)
        comm.Barrier()
    
    # Benchmark runs
    if rank == 0:
        print("Running benchmark...")
    
    timings = []
    for i in range(n_runs):
        comm.Barrier()
        start = time.time()
        
        results = mps.run_distributed()
        
        comm.Barrier()
        elapsed = time.time() - start
        
        timings.append(elapsed)
        
        if rank == 0:
            print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Compute statistics
    if rank == 0:
        timings = np.array(timings)
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Mean time: {np.mean(timings):.3f}s ± {np.std(timings):.3f}s")
        print(f"Min time: {np.min(timings):.3f}s")
        print(f"Max time: {np.max(timings):.3f}s")
        print(f"Median time: {np.median(timings):.3f}s")
        
        # Compute throughput
        iterations = results['iterations']
        throughput = iterations / np.mean(timings)
        print(f"Throughput: {throughput:.1f} iterations/second")
        
        # Compute parallel efficiency
        # (Would need single-process baseline for true efficiency)
        print(f"Parallel speedup: ~{comm.size:.1f}x (theoretical max)")
        print("="*60)
    
    return results


def main():
    """Main execution function"""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(rank, args.verbose and not args.quiet)
    
    # Load configuration
    if rank == 0:
        logger.info(f"Loading configuration from: {args.config}")
    
    loader = ConfigLoader()
    
    # Load and merge configs
    if len(args.config) == 1:
        config = loader.load_config(args.config[0])
    else:
        config = loader.load_multiple_configs(args.config)
    
    # Process overrides
    if args.override:
        overrides = process_overrides(args.override)
        config = loader._apply_overrides(config, overrides)
    
    # Override output directory if specified
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # Validate configuration
    if rank == 0:
        logger.info("Validating configuration...")
        try:
            loader.validate_schema(config)
            logger.info("Configuration valid")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
    
    # Exit if validation only
    if args.validate_only:
        if rank == 0:
            print("Configuration validation successful")
        sys.exit(0)
    
    # Print configuration summary (rank 0 only)
    if rank == 0 and not args.quiet:
        print("\n" + "="*60)
        print("MPS DISTRIBUTED ALGORITHM")
        print("="*60)
        print(f"MPI Processes: {size}")
        print(f"Network: {config['network']['n_sensors']} sensors, "
              f"{config['network']['n_anchors']} anchors")
        print(f"Algorithm: gamma={config['algorithm']['gamma']}, "
              f"alpha={config['algorithm']['alpha']}")
        print(f"Max iterations: {config['algorithm']['max_iterations']}")
        print("="*60 + "\n")
    
    # Generate network
    if rank == 0:
        logger.info("Generating network...")
    network = generate_network_from_config(config, rank)
    
    # Run algorithm
    if args.benchmark:
        # Benchmark mode
        results = run_benchmark(config, network, comm)
    else:
        # Normal execution
        if rank == 0:
            logger.info("Starting distributed MPS algorithm...")
        
        # Create distributed MPS instance
        mps_config = loader.to_mps_config(config, distributed=True)
        mps = DistributedMPS(mps_config, network, comm)
        
        # Enable profiling if requested
        if args.profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Run algorithm
        start_time = time.time()
        results = mps.run_distributed()
        elapsed_time = time.time() - start_time
        
        # Disable profiling
        if args.profile:
            profiler.disable()
            if rank == 0:
                profiler.dump_stats(f"profile_rank{rank}.stats")
                print(f"Profile saved to profile_rank{rank}.stats")
    
    # Print results summary (rank 0 only)
    if rank == 0 and not args.quiet:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Best Error: {results['best_error']:.6f}")
        print(f"Final Error: {results['final_error']:.6f}")
        print(f"Total Time: {results['timing_stats']['total']:.2f}s")
        print("="*60)
    
    # Save results
    if not args.no_save:
        output_dir = config['output'].get('output_dir', 'results/')
        save_results(results, config, output_dir, rank)
    
    # Generate plots if requested
    if args.plot:
        # Use new visualization module for separate windows
        if rank == 0:
            try:
                from src.core.mps_core.visualization import generate_figures
                logger.info("Generating visualization figures...")
                
                # Add network data to results if not present
                results['network_data'] = network
                
                # Generate separate figure windows
                output_prefix = config['output'].get('output_dir', 'results/') + 'mps_figures'
                generate_figures(results, network, config, save_path=output_prefix)
                logger.info("Figures generated successfully")
            except ImportError:
                logger.warning("Visualization module not available, using basic plotting")
                plot_results(results, config, rank)
            except Exception as e:
                logger.error(f"Error generating figures: {e}")
                plot_results(results, config, rank)
    
    # Finalize MPI
    if rank == 0:
        logger.info("MPI execution completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())