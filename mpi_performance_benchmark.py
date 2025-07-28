"""
MPI Performance Benchmarks for Decentralized SNL
Tests scalability across different network sizes and process counts
"""

import numpy as np
from mpi4py import MPI
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
import matplotlib.pyplot as plt
import os

from snl_mpi_optimized import OptimizedMPISNL
from mpi_distributed_operations import DistributedMatrixOps, DistributedSinkhornKnopp


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    n_sensors: int
    n_anchors: int
    n_processes: int
    communication_range: float
    noise_factor: float
    
    # Timing results
    total_time: float
    setup_time: float
    matrix_generation_time: float
    algorithm_time: float
    iterations: int
    
    # Performance metrics
    time_per_iteration: float
    communication_time: float
    computation_time: float
    
    # Algorithm metrics
    final_objective: float
    final_error: float
    converged: bool
    
    # Scalability metrics
    speedup: float = 1.0
    efficiency: float = 1.0


class MPIBenchmarkSuite:
    """Comprehensive benchmark suite for MPI implementation"""
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.results = []
        
    def run_single_benchmark(self, n_sensors: int, n_anchors: int, 
                           communication_range: float = 0.3,
                           noise_factor: float = 0.05) -> BenchmarkResult:
        """Run a single benchmark configuration"""
        
        if self.rank == 0:
            print(f"\nBenchmarking: {n_sensors} sensors, {n_anchors} anchors, "
                  f"{self.size} processes")
        
        # Problem parameters
        problem_params = {
            'n_sensors': n_sensors,
            'n_anchors': n_anchors,
            'd': 2,
            'communication_range': communication_range,
            'noise_factor': noise_factor,
            'gamma': 0.999,
            'alpha_mps': 10.0,
            'max_iter': 500,
            'tol': 1e-4
        }
        
        # Setup phase
        setup_start = time.time()
        solver = OptimizedMPISNL(problem_params)
        solver.generate_network()
        setup_time = time.time() - setup_start
        
        # Matrix generation phase
        matrix_start = time.time()
        solver.compute_matrix_parameters_optimized()
        matrix_time = time.time() - matrix_start
        
        # Algorithm phase
        algo_start = time.time()
        results = solver.run_mps_optimized()
        algo_time = time.time() - algo_start
        
        # Gather timing statistics
        timing_stats = results['timing_stats']
        
        # Create benchmark result
        benchmark = BenchmarkResult(
            n_sensors=n_sensors,
            n_anchors=n_anchors,
            n_processes=self.size,
            communication_range=communication_range,
            noise_factor=noise_factor,
            total_time=setup_time + matrix_time + algo_time,
            setup_time=setup_time,
            matrix_generation_time=matrix_time,
            algorithm_time=algo_time,
            iterations=results['iterations'],
            time_per_iteration=algo_time / results['iterations'],
            communication_time=timing_stats['communication'],
            computation_time=timing_stats['computation'],
            final_objective=results['objectives'][-1] if results['objectives'] else 0,
            final_error=results['errors'][-1] if results['errors'] else 0,
            converged=results['converged']
        )
        
        return benchmark
    
    def run_scalability_study(self):
        """Run comprehensive scalability study"""
        
        # Test configurations
        sensor_counts = [50, 100, 200, 500, 1000]
        
        if self.rank == 0:
            print("="*60)
            print("MPI Scalability Study")
            print(f"Process count: {self.size}")
            print("="*60)
        
        results = []
        
        for n_sensors in sensor_counts:
            # Skip if too few sensors per process
            if n_sensors < self.size * 2:
                continue
                
            n_anchors = max(4, n_sensors // 10)
            
            try:
                result = self.run_single_benchmark(n_sensors, n_anchors)
                results.append(result)
                
                if self.rank == 0:
                    print(f"✓ Completed: {n_sensors} sensors in {result.total_time:.2f}s")
                    
            except Exception as e:
                if self.rank == 0:
                    print(f"✗ Failed: {n_sensors} sensors - {str(e)}")
        
        return results
    
    def run_strong_scaling_study(self, n_sensors: int = 500):
        """
        Strong scaling: fixed problem size, varying process count
        This requires multiple runs with different process counts
        """
        
        if self.rank == 0:
            print(f"\nStrong Scaling Study: {n_sensors} sensors with {self.size} processes")
        
        n_anchors = max(4, n_sensors // 10)
        result = self.run_single_benchmark(n_sensors, n_anchors)
        
        # Calculate speedup and efficiency (requires baseline)
        # In practice, you'd run with -np 1, -np 2, -np 4, etc.
        result.speedup = 1.0  # Would be calculated from baseline
        result.efficiency = result.speedup / self.size
        
        return result
    
    def run_weak_scaling_study(self, sensors_per_process: int = 50):
        """
        Weak scaling: problem size scales with process count
        """
        
        n_sensors = sensors_per_process * self.size
        n_anchors = max(4, n_sensors // 10)
        
        if self.rank == 0:
            print(f"\nWeak Scaling Study: {sensors_per_process} sensors/process "
                  f"({n_sensors} total)")
        
        result = self.run_single_benchmark(n_sensors, n_anchors)
        
        return result
    
    def analyze_communication_patterns(self, n_sensors: int = 100):
        """Analyze communication patterns and overhead"""
        
        if self.rank == 0:
            print(f"\nCommunication Pattern Analysis: {n_sensors} sensors")
        
        # Create problem
        problem_params = {
            'n_sensors': n_sensors,
            'n_anchors': 10,
            'd': 2,
            'communication_range': 0.3,
            'noise_factor': 0.05
        }
        
        solver = OptimizedMPISNL(problem_params)
        solver.generate_network()
        
        # Analyze neighbor distribution
        local_neighbors = 0
        remote_neighbors = 0
        
        for sensor_id in solver.local_sensors:
            sensor = solver.sensor_data[sensor_id]
            for neighbor in sensor.neighbors:
                if neighbor in solver.local_sensors:
                    local_neighbors += 1
                else:
                    remote_neighbors += 1
        
        # Gather statistics
        total_local = self.comm.allreduce(local_neighbors, op=MPI.SUM)
        total_remote = self.comm.allreduce(remote_neighbors, op=MPI.SUM)
        
        if self.rank == 0:
            total = total_local + total_remote
            print(f"Local edges: {total_local} ({100*total_local/total:.1f}%)")
            print(f"Remote edges: {total_remote} ({100*total_remote/total:.1f}%)")
            print(f"Communication ratio: {total_remote/total:.3f}")
        
        return {
            'local_edges': total_local,
            'remote_edges': total_remote,
            'communication_ratio': total_remote / (total_local + total_remote)
        }
    
    def save_results(self, results: List[BenchmarkResult], filename: str = None):
        """Save benchmark results to file"""
        
        if self.rank != 0:
            return
            
        if filename is None:
            filename = f"mpi_benchmark_p{self.size}_{int(time.time())}.json"
        
        # Convert to dictionary
        data = {
            'metadata': {
                'n_processes': self.size,
                'timestamp': time.time(),
                'mpi_version': MPI.Get_version(),
            },
            'results': [asdict(r) for r in results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nResults saved to {filename}")
    
    def plot_results(self, results: List[BenchmarkResult]):
        """Create performance plots"""
        
        if self.rank != 0:
            return
            
        # Extract data
        n_sensors = [r.n_sensors for r in results]
        total_times = [r.total_time for r in results]
        algo_times = [r.algorithm_time for r in results]
        time_per_iter = [r.time_per_iteration for r in results]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total time vs problem size
        ax = axes[0, 0]
        ax.plot(n_sensors, total_times, 'o-', label='Total')
        ax.plot(n_sensors, algo_times, 's-', label='Algorithm')
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Execution Time vs Problem Size')
        ax.legend()
        ax.grid(True)
        
        # Time per iteration
        ax = axes[0, 1]
        ax.plot(n_sensors, time_per_iter, 'o-')
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('Time per Iteration (seconds)')
        ax.set_title('Iteration Time Scaling')
        ax.grid(True)
        
        # Communication vs computation
        ax = axes[1, 0]
        comm_times = [r.communication_time for r in results]
        comp_times = [r.computation_time for r in results]
        x = range(len(results))
        width = 0.35
        ax.bar([i - width/2 for i in x], comm_times, width, label='Communication')
        ax.bar([i + width/2 for i in x], comp_times, width, label='Computation')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Communication vs Computation Time')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.n_sensors}" for r in results])
        ax.legend()
        
        # Convergence quality
        ax = axes[1, 1]
        errors = [r.final_error for r in results]
        iterations = [r.iterations for r in results]
        ax2 = ax.twinx()
        
        p1 = ax.plot(n_sensors, errors, 'o-', color='blue', label='Final Error')
        p2 = ax2.plot(n_sensors, iterations, 's-', color='red', label='Iterations')
        
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('Final Error', color='blue')
        ax2.set_ylabel('Iterations', color='red')
        ax.set_title('Solution Quality')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lns = p1 + p2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'mpi_performance_p{self.size}.png', dpi=150)
        if self.rank == 0:
            print(f"\nPlots saved to mpi_performance_p{self.size}.png")


def run_comprehensive_benchmark():
    """Run comprehensive MPI benchmark suite"""
    
    benchmark = MPIBenchmarkSuite()
    
    # 1. Scalability study
    scalability_results = benchmark.run_scalability_study()
    
    # 2. Communication pattern analysis
    comm_analysis = benchmark.analyze_communication_patterns(n_sensors=200)
    
    # 3. Weak scaling (if enough processes)
    if benchmark.size >= 2:
        weak_result = benchmark.run_weak_scaling_study(sensors_per_process=50)
        scalability_results.append(weak_result)
    
    # Save and plot results
    benchmark.save_results(scalability_results)
    benchmark.plot_results(scalability_results)
    
    # Print summary
    if benchmark.rank == 0:
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        print(f"Process count: {benchmark.size}")
        print(f"Configurations tested: {len(scalability_results)}")
        
        if scalability_results:
            avg_efficiency = np.mean([r.converged for r in scalability_results])
            print(f"Convergence rate: {100*avg_efficiency:.1f}%")
            
            largest = max(scalability_results, key=lambda r: r.n_sensors)
            print(f"Largest problem solved: {largest.n_sensors} sensors")
            print(f"  Time: {largest.total_time:.2f}s")
            print(f"  Error: {largest.final_error:.6f}")


if __name__ == "__main__":
    run_comprehensive_benchmark()