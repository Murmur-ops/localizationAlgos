#!/usr/bin/env python3
"""
Run proper CRLB assessment using actual MPI simulation data
"""

import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
import json
import pickle
import subprocess
import os
import sys
from typing import List, Dict, Tuple

def compute_crlb_for_network(true_positions: np.ndarray, 
                            anchor_positions: np.ndarray,
                            communication_range: float,
                            noise_factor: float) -> float:
    """
    Compute CRLB for a given network configuration
    """
    n_sensors = len(true_positions)
    n_anchors = len(anchor_positions)
    d = true_positions.shape[1]
    
    # Build adjacency matrix
    adjacency = np.zeros((n_sensors, n_sensors))
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            dist = norm(true_positions[i] - true_positions[j])
            if dist <= communication_range:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    
    # Fisher Information Matrix
    FIM = np.zeros((n_sensors * d, n_sensors * d))
    sigma_squared = (noise_factor * communication_range) ** 2
    
    for i in range(n_sensors):
        i_idx = i * d
        
        # Sensor-to-sensor measurements
        for j in range(n_sensors):
            if i != j and adjacency[i, j] > 0:
                diff = true_positions[i] - true_positions[j]
                true_dist = norm(diff)
                
                if true_dist > 0:
                    u_ij = diff / true_dist
                    info_contrib = np.outer(u_ij, u_ij) / sigma_squared
                    
                    FIM[i_idx:i_idx+d, i_idx:i_idx+d] += info_contrib
                    j_idx = j * d
                    FIM[i_idx:i_idx+d, j_idx:j_idx+d] -= info_contrib
        
        # Anchor measurements
        for a in range(n_anchors):
            diff = true_positions[i] - anchor_positions[a]
            anchor_dist = norm(diff)
            
            if anchor_dist <= communication_range and anchor_dist > 0:
                u_ia = diff / anchor_dist
                info_contrib = np.outer(u_ia, u_ia) / sigma_squared
                FIM[i_idx:i_idx+d, i_idx:i_idx+d] += info_contrib
    
    # Compute CRLB
    try:
        FIM_reg = FIM + np.eye(FIM.shape[0]) * 1e-10
        crlb_matrix = inv(FIM_reg)
        
        position_variances = []
        for i in range(n_sensors):
            i_idx = i * d
            var = np.trace(crlb_matrix[i_idx:i_idx+d, i_idx:i_idx+d])
            position_variances.append(var)
        
        avg_crlb = np.sqrt(np.mean(position_variances))
        
    except np.linalg.LinAlgError:
        # Fallback approximation
        avg_degree = np.mean(np.sum(adjacency, axis=1))
        avg_crlb = noise_factor * communication_range / np.sqrt(avg_degree + 2)
    
    return avg_crlb

def run_mpi_simulation_with_noise(noise_factor: float) -> Dict:
    """
    Modify the MPI simulation to run with specific noise level
    and return results
    """
    
    # Create a temporary modified version of the MPI script
    with open('snl_mpi_optimized.py', 'r') as f:
        original_code = f.read()
    
    # Modify the noise factor in problem_params
    modified_code = original_code.replace(
        "'noise_factor': 0.05,",
        f"'noise_factor': {noise_factor},"
    )
    
    # Save temporary version
    with open('snl_mpi_temp.py', 'w') as f:
        f.write(modified_code)
    
    # Run MPI simulation
    try:
        # Run with single process for simplicity and consistency
        result = subprocess.run(
            ['python', 'snl_mpi_temp.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Load the saved results
            with open('mpi_simulation_results.pkl', 'rb') as f:
                sim_data = pickle.load(f)
            
            # Clean up temp file
            os.remove('snl_mpi_temp.py')
            
            return sim_data
        else:
            print(f"Simulation failed for noise factor {noise_factor}")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out for noise factor {noise_factor}")
        return None
    finally:
        # Clean up
        if os.path.exists('snl_mpi_temp.py'):
            os.remove('snl_mpi_temp.py')

def analyze_crlb_performance():
    """
    Run comprehensive CRLB analysis with actual MPI simulation
    """
    
    noise_factors = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    results = []
    
    print("="*60)
    print("CRLB Analysis with Actual MPI Simulation")
    print("="*60)
    
    for noise_factor in noise_factors:
        print(f"\nTesting noise factor: {noise_factor}")
        
        # Run MPI simulation
        sim_data = run_mpi_simulation_with_noise(noise_factor)
        
        if sim_data:
            # Extract data
            true_positions = np.array([sim_data['true_positions'][i] 
                                      for i in range(len(sim_data['true_positions']))])
            final_positions = np.array([sim_data['final_positions'][i] 
                                       for i in range(len(sim_data['final_positions']))])
            anchor_positions = np.array(sim_data['anchor_positions'])
            
            # Calculate actual error
            errors = [norm(true_positions[i] - final_positions[i]) 
                     for i in range(len(true_positions))]
            actual_rmse = np.sqrt(np.mean(np.square(errors)))
            
            # Compute CRLB
            crlb = compute_crlb_for_network(
                true_positions,
                anchor_positions,
                sim_data['problem_params']['communication_range'],
                noise_factor
            )
            
            # Calculate efficiency
            efficiency = (crlb / actual_rmse) * 100 if actual_rmse > 0 else 0
            
            results.append({
                'noise_factor': noise_factor,
                'crlb': crlb,
                'actual_rmse': actual_rmse,
                'efficiency': efficiency,
                'converged': sim_data['results']['converged'],
                'iterations': sim_data['results']['iterations'],
                'final_objective': sim_data['results']['objectives'][-1]
            })
            
            print(f"  CRLB: {crlb:.4f}")
            print(f"  Actual RMSE: {actual_rmse:.4f}")
            print(f"  Efficiency: {efficiency:.1f}%")
            print(f"  Converged: {sim_data['results']['converged']}")
            print(f"  Iterations: {sim_data['results']['iterations']}")
    
    return results

def visualize_results(results: List[Dict]):
    """
    Create comprehensive visualization of CRLB comparison
    """
    
    if not results:
        print("No results to visualize")
        return
    
    # Extract data
    noise_factors = [r['noise_factor'] for r in results]
    crlb_values = [r['crlb'] for r in results]
    actual_errors = [r['actual_rmse'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. CRLB vs Actual Performance
    ax1.plot(noise_factors, crlb_values, 'k-', linewidth=3,
            label='CRLB (Theoretical Limit)', marker='o', markersize=8)
    ax1.plot(noise_factors, actual_errors, 'b-', linewidth=2,
            label='MPI-MPS Algorithm', marker='s', markersize=8)
    
    ax1.fill_between(noise_factors, crlb_values, actual_errors,
                     alpha=0.2, color='blue', label='Performance Gap')
    
    ax1.set_xlabel('Noise Factor', fontsize=12)
    ax1.set_ylabel('Localization Error (RMSE)', fontsize=12)
    ax1.set_title('MPI Algorithm Performance vs CRLB', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency Analysis
    ax2.plot(noise_factors, efficiencies, 'g-', linewidth=2, marker='^', markersize=8)
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax2.axhline(y=80, color='orange', linestyle=':', linewidth=2, label='Target (80%)')
    
    ax2.set_xlabel('Noise Factor', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Algorithm Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Log-scale comparison
    ax3.semilogy(noise_factors, crlb_values, 'k-', linewidth=3,
                label='CRLB', marker='o', markersize=8)
    ax3.semilogy(noise_factors, actual_errors, 'b-', linewidth=2,
                label='MPI-MPS', marker='s', markersize=8)
    
    ax3.set_xlabel('Noise Factor', fontsize=12)
    ax3.set_ylabel('Error (log scale)', fontsize=12)
    ax3.set_title('Log-Scale Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Performance ratio
    ratios = [a/c if c > 0 else 0 for a, c in zip(actual_errors, crlb_values)]
    ax4.plot(noise_factors, ratios, 'r-', linewidth=2, marker='d', markersize=8)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Optimal (ratio=1)')
    
    ax4.set_xlabel('Noise Factor', fontsize=12)
    ax4.set_ylabel('Error Ratio (Actual/CRLB)', fontsize=12)
    ax4.set_title('Performance Ratio', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('CRLB Assessment: MPI-MPS Algorithm Performance', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('crlb_mpi_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to crlb_mpi_assessment.png")

def main():
    """
    Main execution
    """
    
    # Run analysis
    results = analyze_crlb_performance()
    
    # Save results
    if results:
        with open('crlb_mpi_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to crlb_mpi_results.json")
        
        # Create visualization
        visualize_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        avg_efficiency = np.mean([r['efficiency'] for r in results])
        print(f"Average Efficiency: {avg_efficiency:.1f}%")
        
        print("\nDetailed Results:")
        print(f"{'Noise':<10} {'CRLB':<10} {'Actual':<10} {'Efficiency':<12}")
        print("-"*50)
        
        for r in results:
            print(f"{r['noise_factor']:<10.3f} {r['crlb']:<10.4f} "
                  f"{r['actual_rmse']:<10.4f} {r['efficiency']:<12.1f}%")
    else:
        print("No valid results obtained")

if __name__ == "__main__":
    main()