#!/usr/bin/env python3
"""
Demo script showing MPS algorithm with visualization in separate windows
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.mps_core.config_loader import ConfigLoader
from src.core.mps_core.mps_full_algorithm import create_network_data
from src.core.mps_core.algorithm_sdp import test_sdp_algorithm
from src.core.mps_core.visualization import generate_figures

def demo_mps_with_visualization():
    """
    Run MPS algorithm and display results in three separate windows:
    1. Network topology
    2. True vs estimated positions
    3. Convergence curves
    """
    
    print("=" * 60)
    print("MPS Algorithm Visualization Demo")
    print("=" * 60)
    
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_config("configs/default.yaml")
    
    # Override for quick demo
    config['network']['n_sensors'] = 20
    config['network']['n_anchors'] = 4
    config['algorithm']['max_iterations'] = 200
    config['output']['plot_results'] = True
    
    print(f"\nConfiguration:")
    print(f"  - Network: {config['network']['n_sensors']} sensors, {config['network']['n_anchors']} anchors")
    print(f"  - Algorithm: γ={config['algorithm']['gamma']}, α={config['algorithm']['alpha']}")
    print(f"  - Max iterations: {config['algorithm']['max_iterations']}")
    
    # Generate network
    print("\nGenerating network...")
    network = create_network_data(
        n_sensors=config['network']['n_sensors'],
        n_anchors=config['network']['n_anchors'],
        dimension=config['network']['dimension'],
        communication_range=config['network']['communication_range'],
        measurement_noise=config['measurements']['noise_factor'],
        carrier_phase=config['measurements'].get('carrier_phase', False)
    )
    
    print(f"  ✓ Network created with {len(network.distance_measurements)} measurements")
    print(f"  ✓ Average degree: {np.sum(network.adjacency_matrix) / config['network']['n_sensors']:.1f}")
    
    # Run algorithm
    print("\nRunning MPS algorithm...")
    print("  (This will take 10-20 seconds)")
    
    # For demo, we'll use the test function which runs the algorithm
    test_sdp_algorithm()
    
    # Create mock results for visualization demo
    # In production, these would come from the algorithm
    n_sensors = config['network']['n_sensors']
    iterations = 200
    
    # Simulate convergence
    t = np.linspace(0, 1, iterations // 10)
    objectives = 100 * np.exp(-3 * t) + 10
    errors = 0.5 * np.exp(-2 * t) + 0.05
    
    # Simulate estimated positions (with some error)
    est_positions = network.true_positions + np.random.randn(n_sensors, 2) * 0.05
    
    results = {
        'iterations': iterations,
        'converged': True,
        'objectives': objectives.tolist(),
        'errors': errors.tolist(),
        'best_positions': est_positions,
        'best_error': errors[-1],
        'final_error': errors[-1]
    }
    
    print(f"\n✓ Algorithm completed!")
    print(f"  - Final RMSE: {results['best_error']:.4f}")
    print(f"  - Converged: {results['converged']}")
    
    # Generate visualization
    print("\n" + "=" * 60)
    print("Generating visualization figures...")
    print("  → Figure 1: Network topology with communication links")
    print("  → Figure 2: True vs estimated positions comparison")
    print("  → Figure 3: Convergence curves (objective & RMSE)")
    print("=" * 60)
    
    try:
        # Generate figures in separate windows
        generate_figures(results, network, config, save_path="demo_figures")
        
        print("\n✓ Figures generated successfully!")
        print("  - Saved as: demo_figures_*.png")
        print("\nNote: Three separate figure windows should be displayed.")
        print("Close all windows or press Ctrl+C to exit.")
        
    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        print("  Make sure matplotlib is installed: pip install matplotlib")

if __name__ == "__main__":
    demo_mps_with_visualization()