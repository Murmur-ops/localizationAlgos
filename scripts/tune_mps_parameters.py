#!/usr/bin/env python3
"""
Systematic parameter tuning for MPS algorithm following priority order:
1. γ and α (most sensitive)
2. ADMM iterations (accuracy)
3. ADMM ρ (least sensitive)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mps_core.mps_full_algorithm import (
    MatrixParametrizedProximalSplitting,
    MPSConfig,
    NetworkData
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import json
from typing import Dict, List, Tuple, Any


def create_test_network(n_sensors: int = 10, n_anchors: int = 3, 
                        noise_factor: float = 0.05, seed: int = 42):
    """Create a test network for parameter tuning."""
    np.random.seed(seed)
    
    # Generate positions
    sensor_positions = np.random.uniform(0, 1, (n_sensors, 2))
    anchor_positions = np.random.uniform(0, 1, (n_anchors, 2))
    
    # Build adjacency and measurements
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    # Communication range
    comm_range = 0.7
    
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
            if true_dist < comm_range:
                adjacency[i, j] = adjacency[j, i] = 1
                # Add noise
                epsilon = np.random.randn()
                noisy_dist = true_dist * (1 + noise_factor * epsilon)
                distance_measurements[(i, j)] = noisy_dist
    
    # Anchor connections
    anchor_connections = {i: [] for i in range(n_sensors)}
    for i in range(n_sensors):
        for k in range(n_anchors):
            true_dist = np.linalg.norm(sensor_positions[i] - anchor_positions[k])
            if true_dist < comm_range:
                anchor_connections[i].append(k)
                epsilon = np.random.randn()
                noisy_dist = true_dist * (1 + noise_factor * epsilon)
                distance_measurements[(i, k)] = noisy_dist
    
    return NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=sensor_positions,
        measurement_variance=noise_factor**2
    )


def run_single_test(network_data: NetworkData, gamma: float, alpha: float,
                   admm_iterations: int = 100, admm_rho: float = 1.0,
                   max_iterations: int = 200, verbose: bool = False) -> Dict:
    """Run a single test with given parameters."""
    
    config = MPSConfig(
        n_sensors=network_data.adjacency_matrix.shape[0],
        n_anchors=len(network_data.anchor_positions),
        dimension=2,
        gamma=gamma,
        alpha=alpha,
        max_iterations=max_iterations,
        tolerance=1e-6,
        communication_range=0.7,
        scale=1.0,
        verbose=verbose,
        early_stopping=True,
        early_stopping_window=50,
        admm_iterations=admm_iterations,
        admm_tolerance=1e-6,
        admm_rho=admm_rho,
        warm_start=True,  # Enable warm-starting
        parallel_proximal=False,
        use_2block=True,
        adaptive_alpha=False,
        carrier_phase_mode=False
    )
    
    # Run algorithm
    mps = MatrixParametrizedProximalSplitting(config, network_data)
    start_time = time.time()
    result = mps.run(max_iterations=max_iterations)
    elapsed = time.time() - start_time
    
    # Calculate metrics
    final_positions = result['final_positions']
    true_positions = network_data.true_positions
    
    relative_error = np.linalg.norm(final_positions - true_positions, 'fro') / \
                    np.linalg.norm(true_positions, 'fro')
    
    return {
        'gamma': gamma,
        'alpha': alpha,
        'admm_iterations': admm_iterations,
        'admm_rho': admm_rho,
        'relative_error': relative_error,
        'final_rmse': result['final_rmse'],
        'iterations': result['iterations'],
        'converged': result['converged'],
        'elapsed_time': elapsed,
        'final_objective': result['history']['objective'][-1] if result['history']['objective'] else np.inf,
        'final_consensus': result['history']['consensus_error'][-1] if result['history']['consensus_error'] else np.inf,
        'final_psd_violation': result['history']['psd_violation'][-1] if result['history']['psd_violation'] else np.inf,
        'history': result['history']  # Store full history for analysis
    }


def stage1_tune_gamma_alpha(network_data: NetworkData, 
                           gammas: List[float] = [0.9, 0.95, 0.99, 0.999],
                           alphas: List[float] = [5.0, 10.0, 15.0, 20.0]) -> Dict:
    """Stage 1: Tune γ and α (most critical parameters)."""
    
    print("\n" + "="*70)
    print("STAGE 1: TUNING γ AND α")
    print("="*70)
    print(f"Testing {len(gammas)} γ values: {gammas}")
    print(f"Testing {len(alphas)} α values: {alphas}")
    print(f"Total combinations: {len(gammas) * len(alphas)}")
    
    results = []
    best_result = None
    best_error = float('inf')
    
    for gamma in gammas:
        for alpha in alphas:
            print(f"\nTesting γ={gamma:.3f}, α={alpha:.1f}...")
            
            try:
                result = run_single_test(
                    network_data, gamma, alpha, 
                    admm_iterations=100, admm_rho=1.0,
                    max_iterations=200
                )
                results.append(result)
                
                print(f"  Relative error: {result['relative_error']:.4f}")
                print(f"  Iterations: {result['iterations']}")
                print(f"  Time: {result['elapsed_time']:.1f}s")
                
                if result['relative_error'] < best_error:
                    best_error = result['relative_error']
                    best_result = result
                    
            except Exception as e:
                print(f"  Failed: {e}")
                results.append({
                    'gamma': gamma, 'alpha': alpha,
                    'relative_error': np.inf, 'failed': True
                })
    
    print("\n" + "-"*70)
    print("STAGE 1 BEST RESULT:")
    if best_result:
        print(f"  γ = {best_result['gamma']:.3f}")
        print(f"  α = {best_result['alpha']:.1f}")
        print(f"  Relative error = {best_result['relative_error']:.4f}")
    
    return {'results': results, 'best': best_result}


def stage2_tune_admm_iterations(network_data: NetworkData, 
                               best_gamma: float, best_alpha: float,
                               iterations_list: List[int] = [50, 100, 150, 200]) -> Dict:
    """Stage 2: Tune ADMM iterations for accuracy."""
    
    print("\n" + "="*70)
    print("STAGE 2: TUNING ADMM ITERATIONS")
    print("="*70)
    print(f"Using best γ={best_gamma:.3f}, α={best_alpha:.1f}")
    print(f"Testing iterations: {iterations_list}")
    
    results = []
    best_result = None
    best_score = float('inf')
    
    for admm_iters in iterations_list:
        print(f"\nTesting ADMM iterations={admm_iters}...")
        
        result = run_single_test(
            network_data, best_gamma, best_alpha,
            admm_iterations=admm_iters, admm_rho=1.0,
            max_iterations=200
        )
        results.append(result)
        
        # Score combines error and time (prefer accurate but fast)
        score = result['relative_error'] + 0.001 * result['elapsed_time']
        
        print(f"  Relative error: {result['relative_error']:.4f}")
        print(f"  Time: {result['elapsed_time']:.1f}s")
        print(f"  Score: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_result = result
    
    print("\n" + "-"*70)
    print("STAGE 2 BEST RESULT:")
    if best_result:
        print(f"  ADMM iterations = {best_result['admm_iterations']}")
        print(f"  Relative error = {best_result['relative_error']:.4f}")
        print(f"  Time = {best_result['elapsed_time']:.1f}s")
    
    return {'results': results, 'best': best_result}


def stage3_tune_admm_rho(network_data: NetworkData,
                        best_gamma: float, best_alpha: float, 
                        best_admm_iters: int,
                        rhos: List[float] = [0.5, 1.0, 2.0, 5.0]) -> Dict:
    """Stage 3: Fine-tune ADMM ρ (least sensitive)."""
    
    print("\n" + "="*70)
    print("STAGE 3: TUNING ADMM ρ")
    print("="*70)
    print(f"Using γ={best_gamma:.3f}, α={best_alpha:.1f}, ADMM iters={best_admm_iters}")
    print(f"Testing ρ values: {rhos}")
    
    results = []
    best_result = None
    best_error = float('inf')
    
    for rho in rhos:
        print(f"\nTesting ρ={rho:.1f}...")
        
        result = run_single_test(
            network_data, best_gamma, best_alpha,
            admm_iterations=best_admm_iters, admm_rho=rho,
            max_iterations=200
        )
        results.append(result)
        
        print(f"  Relative error: {result['relative_error']:.4f}")
        print(f"  Time: {result['elapsed_time']:.1f}s")
        
        if result['relative_error'] < best_error:
            best_error = result['relative_error']
            best_result = result
    
    print("\n" + "-"*70)
    print("STAGE 3 BEST RESULT:")
    if best_result:
        print(f"  ρ = {best_result['admm_rho']:.1f}")
        print(f"  Relative error = {best_result['relative_error']:.4f}")
    
    return {'results': results, 'best': best_result}


def plot_tuning_results(stage1_results: Dict, stage2_results: Dict, 
                        stage3_results: Dict):
    """Create visualization of tuning results."""
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Stage 1: γ-α heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    if stage1_results and 'results' in stage1_results:
        results = stage1_results['results']
        gammas = sorted(set(r['gamma'] for r in results if 'gamma' in r))
        alphas = sorted(set(r['alpha'] for r in results if 'alpha' in r))
        
        error_matrix = np.zeros((len(alphas), len(gammas)))
        for r in results:
            if 'gamma' in r and 'alpha' in r and 'relative_error' in r:
                i = alphas.index(r['alpha'])
                j = gammas.index(r['gamma'])
                error_matrix[i, j] = min(r['relative_error'], 1.0)  # Cap at 1.0
        
        im = ax1.imshow(error_matrix, aspect='auto', cmap='viridis_r')
        ax1.set_xticks(range(len(gammas)))
        ax1.set_xticklabels([f'{g:.3f}' for g in gammas])
        ax1.set_yticks(range(len(alphas)))
        ax1.set_yticklabels([f'{a:.1f}' for a in alphas])
        ax1.set_xlabel('γ')
        ax1.set_ylabel('α')
        ax1.set_title('Stage 1: γ-α Parameter Space (Relative Error)')
        plt.colorbar(im, ax=ax1)
    
    # Stage 2: ADMM iterations
    ax2 = fig.add_subplot(gs[1, 0])
    if stage2_results and 'results' in stage2_results:
        results = stage2_results['results']
        iters = [r['admm_iterations'] for r in results]
        errors = [r['relative_error'] for r in results]
        times = [r['elapsed_time'] for r in results]
        
        ax2.plot(iters, errors, 'o-', label='Error')
        ax2.set_xlabel('ADMM Iterations')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Stage 2: ADMM Iterations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Stage 2: Time vs iterations
    ax3 = fig.add_subplot(gs[1, 1])
    if stage2_results and 'results' in stage2_results:
        ax3.plot(iters, times, 'o-', color='orange')
        ax3.set_xlabel('ADMM Iterations')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Computation Time')
        ax3.grid(True, alpha=0.3)
    
    # Stage 3: ADMM ρ
    ax4 = fig.add_subplot(gs[2, 0])
    if stage3_results and 'results' in stage3_results:
        results = stage3_results['results']
        rhos = [r['admm_rho'] for r in results]
        errors = [r['relative_error'] for r in results]
        
        ax4.semilogx(rhos, errors, 'o-', color='green')
        ax4.set_xlabel('ADMM ρ')
        ax4.set_ylabel('Relative Error')
        ax4.set_title('Stage 3: ADMM ρ')
        ax4.grid(True, alpha=0.3)
    
    # Best parameters summary
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis('off')
    
    summary_text = "OPTIMAL PARAMETERS\n" + "="*25 + "\n\n"
    
    if stage1_results and 'best' in stage1_results and stage1_results['best']:
        best = stage1_results['best']
        summary_text += f"Stage 1 (γ, α):\n"
        summary_text += f"  γ = {best['gamma']:.3f}\n"
        summary_text += f"  α = {best['alpha']:.1f}\n"
        summary_text += f"  Error = {best['relative_error']:.4f}\n\n"
    
    if stage2_results and 'best' in stage2_results and stage2_results['best']:
        best = stage2_results['best']
        summary_text += f"Stage 2 (ADMM iters):\n"
        summary_text += f"  Iterations = {best['admm_iterations']}\n"
        summary_text += f"  Error = {best['relative_error']:.4f}\n\n"
    
    if stage3_results and 'best' in stage3_results and stage3_results['best']:
        best = stage3_results['best']
        summary_text += f"Stage 3 (ADMM ρ):\n"
        summary_text += f"  ρ = {best['admm_rho']:.1f}\n"
        summary_text += f"  Error = {best['relative_error']:.4f}\n\n"
        
        summary_text += "="*25 + "\n"
        summary_text += f"FINAL ERROR: {best['relative_error']:.4f}\n"
        
        if best['relative_error'] < 0.10:
            summary_text += "\n✓ MATCHES PAPER!"
        elif best['relative_error'] < 0.15:
            summary_text += "\n✓ Close to paper"
        else:
            summary_text += "\n⚠ Needs more tuning"
    
    ax5.text(0.1, 0.9, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax5.transAxes)
    
    plt.suptitle('MPS Algorithm Parameter Tuning Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('parameter_tuning_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_results(results: Dict, filename: str = 'tuning_results.json'):
    """Save tuning results to JSON file."""
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        else:
            return obj
    
    with open(filename, 'w') as f:
        json.dump(convert_arrays(results), f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    """Main tuning pipeline."""
    
    print("\n" + "="*70)
    print("MPS ALGORITHM PARAMETER TUNING")
    print("="*70)
    print("\nFollowing priority order:")
    print("  1. γ and α (most sensitive)")
    print("  2. ADMM iterations (accuracy)")
    print("  3. ADMM ρ (least sensitive)")
    
    # Create test networks
    print("\nCreating test networks...")
    
    # Small network for quick tuning
    small_network = create_test_network(n_sensors=10, n_anchors=3, seed=42)
    print(f"Small network: {10} sensors, {3} anchors")
    
    # Medium network for validation
    medium_network = create_test_network(n_sensors=20, n_anchors=4, seed=43)
    print(f"Medium network: {20} sensors, {4} anchors")
    
    # Run tuning stages on small network
    print("\n" + "="*70)
    print("RUNNING PARAMETER TUNING ON SMALL NETWORK")
    print("="*70)
    
    # Stage 1: γ and α
    stage1_results = stage1_tune_gamma_alpha(
        small_network,
        gammas=[0.9, 0.95, 0.99, 0.999],
        alphas=[5.0, 10.0, 15.0, 20.0]
    )
    
    if stage1_results['best']:
        best_gamma = stage1_results['best']['gamma']
        best_alpha = stage1_results['best']['alpha']
        
        # Stage 2: ADMM iterations
        stage2_results = stage2_tune_admm_iterations(
            small_network, best_gamma, best_alpha,
            iterations_list=[50, 100, 150, 200]
        )
        
        if stage2_results['best']:
            best_admm_iters = stage2_results['best']['admm_iterations']
            
            # Stage 3: ADMM ρ
            stage3_results = stage3_tune_admm_rho(
                small_network, best_gamma, best_alpha, best_admm_iters,
                rhos=[0.5, 1.0, 2.0, 5.0]
            )
        else:
            stage3_results = {}
    else:
        stage2_results = {}
        stage3_results = {}
    
    # Validate on medium network
    print("\n" + "="*70)
    print("VALIDATION ON MEDIUM NETWORK")
    print("="*70)
    
    if stage3_results and 'best' in stage3_results and stage3_results['best']:
        best = stage3_results['best']
        print(f"\nUsing optimal parameters:")
        print(f"  γ = {best['gamma']:.3f}")
        print(f"  α = {best['alpha']:.1f}")
        print(f"  ADMM iterations = {best['admm_iterations']}")
        print(f"  ADMM ρ = {best['admm_rho']:.1f}")
        
        print("\nRunning on medium network...")
        validation_result = run_single_test(
            medium_network, 
            best['gamma'], best['alpha'],
            best['admm_iterations'], best['admm_rho'],
            max_iterations=300
        )
        
        print(f"\nValidation Results:")
        print(f"  Relative error: {validation_result['relative_error']:.4f}")
        print(f"  Iterations: {validation_result['iterations']}")
        print(f"  Time: {validation_result['elapsed_time']:.1f}s")
        
        # Add validation to results
        stage3_results['validation'] = validation_result
    
    # Plot results
    plot_tuning_results(stage1_results, stage2_results, stage3_results)
    
    # Save results
    all_results = {
        'stage1': stage1_results,
        'stage2': stage2_results,
        'stage3': stage3_results
    }
    save_results(all_results)
    
    # Final summary
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    
    if stage3_results and 'best' in stage3_results and stage3_results['best']:
        best = stage3_results['best']
        print(f"\nOptimal parameters found:")
        print(f"  γ = {best['gamma']:.3f}")
        print(f"  α = {best['alpha']:.1f}")
        print(f"  ADMM iterations = {best['admm_iterations']}")
        print(f"  ADMM ρ = {best['admm_rho']:.1f}")
        print(f"  Relative error = {best['relative_error']:.4f}")
        
        if best['relative_error'] < 0.10:
            print("\n✓✓✓ SUCCESS! Matches paper's performance (<0.10)")
        elif best['relative_error'] < 0.15:
            print("\n✓✓ Good - Close to paper's performance")
        else:
            print("\n✓ Reasonable - May need further tuning")


if __name__ == "__main__":
    main()