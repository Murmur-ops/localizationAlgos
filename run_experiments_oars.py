"""
Enhanced experiment runner with OARS integration
Compares different matrix parameter generation methods
"""

import numpy as np
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from mpi4py import MPI

from snl_main import SNLProblem
from snl_main_oars import EnhancedDistributedSNL, OARS_AVAILABLE
from run_experiments import ExperimentResult, ExperimentConfig, ExperimentRunner

logger = logging.getLogger(__name__)


@dataclass
class OARSExperimentResult(ExperimentResult):
    """Extended results including OARS matrix methods"""
    matrix_method: str = "Sinkhorn-Knopp"
    matrix_generation_time: float = 0.0
    matrix_verification_passed: bool = True


class OARSExperimentRunner(ExperimentRunner):
    """Enhanced experiment runner with OARS matrix method comparisons"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.matrix_methods = ["Sinkhorn-Knopp"]
        
        if OARS_AVAILABLE:
            self.matrix_methods.extend(["MinSLEM", "MaxConnectivity", "MinResist"])
            if self.rank == 0:
                logger.info(f"OARS available - testing {len(self.matrix_methods)} matrix methods")
        else:
            if self.rank == 0:
                logger.warning("OARS not available - using Sinkhorn-Knopp only")
    
    def run_matrix_comparison(self, problem: SNLProblem) -> Dict[str, OARSExperimentResult]:
        """Run experiments comparing different matrix generation methods"""
        results = {}
        
        # Initialize solver
        snl = EnhancedDistributedSNL(problem, use_oars=True)
        snl.generate_network()
        
        for method in self.matrix_methods:
            if self.rank == 0:
                logger.info(f"Testing matrix method: {method}")
            
            # Reset variables
            for sensor_id in snl.sensor_ids:
                snl.sensor_data[sensor_id].X = np.zeros(problem.d)
                snl.sensor_data[sensor_id].Y = np.zeros_like(snl.sensor_data[sensor_id].Y)
                snl.sensor_data[sensor_id].v_gi = (
                    np.zeros(problem.d), 
                    np.zeros_like(snl.sensor_data[sensor_id].Y)
                )
                snl.sensor_data[sensor_id].v_delta = (
                    np.zeros(problem.d), 
                    np.zeros_like(snl.sensor_data[sensor_id].Y)
                )
            
            # Time matrix generation
            matrix_start = time.time()
            if method == "Sinkhorn-Knopp":
                Z_blocks, W_blocks = snl.compute_matrix_parameters()
            else:
                Z_blocks, W_blocks = snl.compute_matrix_parameters_oars(method)
            matrix_time = time.time() - matrix_start
            
            # Run MPS
            start_time = time.time()
            if method == "Sinkhorn-Knopp":
                mps_results = snl.matrix_parametrized_splitting()
            else:
                mps_results = snl.matrix_parametrized_splitting_oars(method)
            mps_time = time.time() - start_time
            mps_error = snl.compute_error(mps_results)
            
            if self.rank == 0 and mps_error is not None:
                result = OARSExperimentResult(
                    experiment_id=0,
                    seed=problem.seed,
                    n_sensors=problem.n_sensors,
                    n_anchors=problem.n_anchors,
                    noise_factor=problem.noise_factor,
                    communication_range=problem.communication_range,
                    mps_error=mps_error,
                    mps_time=mps_time,
                    mps_iterations=200,  # Placeholder
                    mps_objective_history=[],
                    admm_error=0,  # Will be filled later
                    admm_time=0,
                    admm_iterations=0,
                    admm_objective_history=[],
                    error_ratio=0,
                    speedup=0,
                    matrix_method=method,
                    matrix_generation_time=matrix_time
                )
                results[method] = result
        
        # Run ADMM for comparison
        if self.rank == 0:
            logger.info("Running ADMM for comparison...")
        
        # Reset for ADMM
        for sensor_id in snl.sensor_ids:
            snl.sensor_data[sensor_id].X = np.zeros(problem.d)
            snl.sensor_data[sensor_id].Y = np.zeros_like(snl.sensor_data[sensor_id].Y)
            snl.sensor_data[sensor_id].U = (np.zeros(problem.d), np.zeros_like(snl.sensor_data[sensor_id].Y))
            snl.sensor_data[sensor_id].V = (np.zeros(problem.d), np.zeros_like(snl.sensor_data[sensor_id].Y))
            snl.sensor_data[sensor_id].R = (np.zeros(problem.d), np.zeros_like(snl.sensor_data[sensor_id].Y))
        
        start_time = time.time()
        admm_results = snl.admm_decentralized()
        admm_time = time.time() - start_time
        admm_error = snl.compute_error(admm_results)
        
        # Update results with ADMM comparison
        if self.rank == 0:
            for method, result in results.items():
                result.admm_error = admm_error
                result.admm_time = admm_time
                result.admm_iterations = 400  # Placeholder
                result.error_ratio = admm_error / result.mps_error if result.mps_error > 0 else float('inf')
                result.speedup = admm_time / result.mps_time if result.mps_time > 0 else float('inf')
        
        return results if self.rank == 0 else {}
    
    def run_oars_study(self, n_experiments: int = 10) -> Dict[str, List[OARSExperimentResult]]:
        """Run study comparing OARS matrix methods across multiple experiments"""
        all_results = {method: [] for method in self.matrix_methods}
        
        for exp_id in range(n_experiments):
            if self.rank == 0:
                logger.info(f"\nRunning experiment {exp_id + 1}/{n_experiments}")
            
            problem = SNLProblem(
                n_sensors=self.config.n_sensors_list[0],
                n_anchors=self.config.n_anchors_list[0],
                noise_factor=self.config.noise_factors[0],
                communication_range=self.config.communication_ranges[0],
                seed=self.config.base_seed + exp_id
            )
            
            method_results = self.run_matrix_comparison(problem)
            
            if self.rank == 0:
                for method, result in method_results.items():
                    result.experiment_id = exp_id
                    all_results[method].append(result)
        
        if self.rank == 0:
            # Save results
            self.save_results(all_results, "oars_matrix_comparison")
            
            # Analyze results
            self.analyze_oars_results(all_results)
        
        return all_results if self.rank == 0 else {}
    
    def analyze_oars_results(self, results: Dict[str, List[OARSExperimentResult]]):
        """Analyze and summarize OARS comparison results"""
        summary = {}
        
        for method, method_results in results.items():
            if not method_results:
                continue
            
            df = pd.DataFrame([asdict(r) for r in method_results])
            
            summary[method] = {
                'mps_error': {
                    'mean': df['mps_error'].mean(),
                    'std': df['mps_error'].std(),
                    'median': df['mps_error'].median()
                },
                'mps_time': {
                    'mean': df['mps_time'].mean(),
                    'std': df['mps_time'].std()
                },
                'matrix_time': {
                    'mean': df['matrix_generation_time'].mean(),
                    'std': df['matrix_generation_time'].std()
                },
                'error_ratio': {
                    'mean': df['error_ratio'].mean(),
                    'std': df['error_ratio'].std()
                }
            }
        
        # Print summary table
        print("\n" + "="*80)
        print("OARS MATRIX METHOD COMPARISON")
        print("="*80)
        print(f"{'Method':20s} | {'MPS Error':>12s} | {'MPS Time (s)':>12s} | {'Matrix Gen (s)':>14s} | {'vs ADMM':>8s}")
        print("-"*80)
        
        for method, stats in summary.items():
            print(f"{method:20s} | "
                  f"{stats['mps_error']['mean']:>6.4f} ± {stats['mps_error']['std']:>5.4f} | "
                  f"{stats['mps_time']['mean']:>6.2f} ± {stats['mps_time']['std']:>5.2f} | "
                  f"{stats['matrix_time']['mean']:>7.3f} ± {stats['matrix_time']['std']:>6.3f} | "
                  f"{stats['error_ratio']['mean']:>6.2f}x")
        
        print("="*80)
        
        # Find best method
        best_method = min(summary.keys(), key=lambda m: summary[m]['mps_error']['mean'])
        print(f"\nBest method by error: {best_method}")
        
        # Save analysis
        with open(os.path.join(self.config.save_dir, "oars_analysis.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run_scaling_study(self, sensor_counts: List[int] = None) -> Dict[int, Dict[str, float]]:
        """Study how matrix generation time scales with problem size"""
        if sensor_counts is None:
            sensor_counts = [10, 20, 30, 50, 100, 200]
        
        scaling_results = {}
        
        for n_sensors in sensor_counts:
            if self.rank == 0:
                logger.info(f"\nTesting with {n_sensors} sensors...")
            
            # Skip if too many sensors for available processes
            if n_sensors > self.size * 5:
                if self.rank == 0:
                    logger.warning(f"Skipping {n_sensors} sensors (too many for {self.size} processes)")
                continue
            
            problem = SNLProblem(
                n_sensors=n_sensors,
                n_anchors=max(3, n_sensors // 5),
                communication_range=0.7,
                noise_factor=0.05,
                seed=42
            )
            
            snl = EnhancedDistributedSNL(problem, use_oars=True)
            snl.generate_network()
            
            timing_results = {}
            
            for method in self.matrix_methods:
                if self.rank == 0:
                    logger.info(f"  Testing {method}...")
                
                start_time = time.time()
                
                try:
                    if method == "Sinkhorn-Knopp":
                        Z_blocks, W_blocks = snl.compute_matrix_parameters()
                    else:
                        Z_blocks, W_blocks = snl.compute_matrix_parameters_oars(method)
                    
                    elapsed = time.time() - start_time
                    
                    if self.rank == 0:
                        timing_results[method] = elapsed
                        logger.info(f"    Time: {elapsed:.3f}s")
                        
                except Exception as e:
                    if self.rank == 0:
                        logger.error(f"    Failed: {e}")
                        timing_results[method] = float('inf')
            
            if self.rank == 0:
                scaling_results[n_sensors] = timing_results
        
        if self.rank == 0:
            # Save and visualize scaling results
            self.save_results(scaling_results, "matrix_generation_scaling")
            self._plot_scaling_results(scaling_results)
        
        return scaling_results if self.rank == 0 else {}
    
    def _plot_scaling_results(self, scaling_results: Dict[int, Dict[str, float]]):
        """Create scaling plot for matrix generation times"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data for plotting
            sensor_counts = sorted(scaling_results.keys())
            
            for method in self.matrix_methods:
                times = []
                valid_counts = []
                
                for n in sensor_counts:
                    if method in scaling_results[n] and scaling_results[n][method] < float('inf'):
                        times.append(scaling_results[n][method])
                        valid_counts.append(n)
                
                if times:
                    ax.plot(valid_counts, times, 'o-', label=method, linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Sensors', fontsize=12)
            ax.set_ylabel('Matrix Generation Time (s)', fontsize=12)
            ax.set_title('Scaling of Matrix Parameter Generation Methods', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log scale if needed
            if max(sensor_counts) / min(sensor_counts) > 10:
                ax.set_xscale('log')
                ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.save_dir, "matrix_scaling.png"), dpi=300)
            plt.close()
            
            logger.info("Saved scaling plot to matrix_scaling.png")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def main():
    """Run OARS comparison experiments"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Configure experiments
    config = ExperimentConfig(
        n_experiments=10,
        save_dir="results_oars"
    )
    
    # Create runner
    runner = OARSExperimentRunner(config)
    
    # Run matrix method comparison
    if rank == 0:
        logger.info("=== Running OARS Matrix Method Comparison ===")
    results = runner.run_oars_study(n_experiments=10)
    
    # Run scaling study
    if rank == 0:
        logger.info("\n=== Running Scaling Study ===")
    scaling_results = runner.run_scaling_study()
    
    if rank == 0:
        logger.info("\nAll OARS experiments complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()