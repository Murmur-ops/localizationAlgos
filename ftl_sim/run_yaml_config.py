#!/usr/bin/env python3
"""
Run FTL consensus simulation from YAML configuration file
"""

import yaml
import numpy as np
import sys
import time
from pathlib import Path
from ftl.consensus.consensus_gn import ConsensusGaussNewton, ConsensusGNConfig
from ftl.factors_scaled import ToAFactorMeters

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_network(config):
    """Setup network from configuration"""
    scene = config['scene']

    # Get dimensions
    n_nodes = scene['n_nodes']
    n_anchors = scene['n_anchors']
    area_width = scene['area']['width']
    area_height = scene['area']['height']
    comm_range = scene['comm_range']

    # Set up anchor positions
    anchor_positions = np.array(scene['placement']['anchors']['positions'])

    # Set up unknown positions (grid or random)
    n_unknowns = n_nodes - n_anchors
    if scene['placement']['unknowns']['type'] == 'grid':
        grid_size = scene['placement']['unknowns'].get('grid_size',
                                                        int(np.ceil(np.sqrt(n_unknowns))))
        x_range = scene['placement']['unknowns']['x_range']
        y_range = scene['placement']['unknowns']['y_range']

        x_pos = np.linspace(x_range[0], x_range[1], grid_size)
        y_pos = np.linspace(y_range[0], y_range[1], grid_size)

        unknown_positions = []
        for x in x_pos:
            for y in y_pos:
                unknown_positions.append([x, y])
                if len(unknown_positions) >= n_unknowns:
                    break
            if len(unknown_positions) >= n_unknowns:
                break
        unknown_positions = np.array(unknown_positions[:n_unknowns])
    else:  # random
        x_range = scene['placement']['unknowns'].get('x_range', [5, area_width-5])
        y_range = scene['placement']['unknowns'].get('y_range', [5, area_height-5])
        unknown_positions = np.random.uniform(
            [x_range[0], y_range[0]],
            [x_range[1], y_range[1]],
            (n_unknowns, 2)
        )

    true_positions = np.vstack([anchor_positions, unknown_positions])

    return true_positions, n_anchors, comm_range

def setup_consensus(config):
    """Create consensus solver from configuration"""
    consensus_cfg = config['consensus']

    # Get parameter set
    param_key = consensus_cfg.get('use_parameters', 'parameters')
    if param_key == 'accurate':
        params = consensus_cfg.get('parameters_accurate', consensus_cfg.get('parameters'))
    elif param_key == 'fast':
        params = consensus_cfg.get('parameters_fast', consensus_cfg.get('parameters'))
    else:
        params = consensus_cfg.get('parameters', consensus_cfg.get('parameters_accurate'))

    # Create configuration
    cgn_config = ConsensusGNConfig(
        max_iterations=params.get('max_iterations', 200),
        consensus_gain=params.get('consensus_gain', 0.1),
        step_size=params.get('step_size', 0.3),
        gradient_tol=params.get('gradient_tol', 1e-5),
        step_tol=params.get('step_tol', 1e-6),
        verbose=config.get('simulation', {}).get('verbose', False)
    )

    return ConsensusGaussNewton(cgn_config)

def run_consensus_simulation(config_path):
    """Run consensus simulation from YAML config"""

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Set random seed
    seed = config.get('simulation', {}).get('random_seed', 42)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print(f"Running: {config['scene']['name']}")
    print(f"{'='*70}")
    print(f"{config['scene']['description']}")

    # Setup network
    true_positions, n_anchors, comm_range = setup_network(config)
    n_nodes = len(true_positions)

    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {n_nodes} ({n_anchors} anchors)")
    print(f"  Area: {config['scene']['area']['width']}×{config['scene']['area']['height']}m")
    print(f"  Communication range: {comm_range}m")

    # Create consensus solver
    cgn = setup_consensus(config)

    # Get measurement noise
    meas_noise = config['measurements']['range_noise_std']
    print(f"  Measurement noise: {meas_noise*100:.1f}cm")

    # Get initialization noise
    init_noise = config['initialization']['position_noise_std']
    print(f"  Initial position noise: {init_noise*100:.0f}cm")

    # Add nodes
    initial_positions = []
    for i in range(n_nodes):
        state = np.zeros(5)
        if i < n_anchors:
            state[:2] = true_positions[i]
            initial_positions.append(true_positions[i].copy())
            cgn.add_node(i, state, is_anchor=True)
        else:
            # Initialize with noise
            if config['initialization']['method'] == 'gaussian':
                initial_pos = true_positions[i] + np.random.normal(0, init_noise, 2)
            elif config['initialization']['method'] == 'center':
                initial_pos = np.array([
                    config['scene']['area']['width']/2,
                    config['scene']['area']['height']/2
                ])
            else:  # random
                initial_pos = np.random.uniform(
                    [5, 5],
                    [config['scene']['area']['width']-5,
                     config['scene']['area']['height']-5]
                )

            state[:2] = initial_pos
            initial_positions.append(initial_pos.copy())
            cgn.add_node(i, state, is_anchor=False)

    # Add edges and measurements
    n_edges = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if dist <= comm_range:
                cgn.add_edge(i, j)
                n_edges += 1

                # Add measurement with noise
                meas_range = dist + np.random.normal(0, meas_noise)
                cgn.add_measurement(ToAFactorMeters(i, j, meas_range, meas_noise**2))

    print(f"\nNetwork Topology:")
    print(f"  Edges: {n_edges}")
    print(f"  Average degree: {n_edges * 2 / n_nodes:.1f}")

    # Check connectivity
    direct_anchor = sum(1 for i in range(n_anchors, n_nodes)
                       if any((i, j) in cgn.edges or (j, i) in cgn.edges
                             for j in range(n_anchors)))
    print(f"  Direct anchor connections: {direct_anchor}/{n_nodes-n_anchors}")

    # Set true positions for evaluation
    cgn.set_true_positions({i: true_positions[i] for i in range(n_anchors, n_nodes)})

    # Run optimization
    print(f"\nRunning Consensus Optimization...")
    print(f"  Algorithm: {config['consensus']['algorithm']}")
    print(f"  Consensus gain (μ): {cgn.config.consensus_gain}")
    print(f"  Step size (α): {cgn.config.step_size}")
    print(f"  Max iterations: {cgn.config.max_iterations}")

    start_time = time.time()
    results = cgn.optimize()
    elapsed = time.time() - start_time

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"Convergence:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Time: {elapsed:.2f}s")

    if 'position_errors' in results:
        errors = results['position_errors']
        print(f"\nAccuracy:")
        print(f"  RMSE: {errors['rmse']*100:.2f}cm")
        print(f"  Mean error: {errors['mean']*100:.2f}cm")
        print(f"  Max error: {errors['max']*100:.2f}cm")

        # Check against benchmarks
        if 'benchmarks' in config:
            benchmarks = config['benchmarks']
            if 'ideal' in benchmarks:
                expected_rmse = benchmarks['ideal']['rmse_cm']
                print(f"\nBenchmark Comparison:")
                print(f"  Expected RMSE: {expected_rmse:.1f}cm")
                print(f"  Achieved RMSE: {errors['rmse']*100:.2f}cm")

                if errors['rmse']*100 <= expected_rmse * 1.5:
                    print(f"  ✓ Performance meets expectations")
                else:
                    print(f"  ✗ Performance below expectations")

            if 'thresholds' in benchmarks:
                thresholds = benchmarks['thresholds']
                max_rmse = thresholds.get('max_rmse_cm', 5.0)
                if errors['rmse']*100 <= max_rmse:
                    print(f"  ✓ Within threshold ({max_rmse}cm)")
                else:
                    print(f"  ✗ Exceeds threshold ({max_rmse}cm)")

    # Save results if configured
    if config.get('output', {}).get('save_positions', False):
        output_dir = Path(config['output'].get('output_dir', 'results/'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save positions
        np.save(output_dir / 'true_positions.npy', true_positions)

        estimated_positions = np.zeros((n_nodes, 2))
        for i in range(n_nodes):
            estimated_positions[i] = cgn.nodes[i].state[:2]
        np.save(output_dir / 'estimated_positions.npy', estimated_positions)

        print(f"\nResults saved to {output_dir}")

    return results, cgn

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python run_yaml_config.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)

    try:
        results, cgn = run_consensus_simulation(config_path)

        # Return success/failure based on convergence
        if results.get('converged', False):
            sys.exit(0)
        else:
            # Check if we met accuracy target even without convergence
            if 'position_errors' in results:
                if results['position_errors']['rmse'] < 0.02:  # 2cm
                    sys.exit(0)
            sys.exit(1)

    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()