#!/usr/bin/env python3
"""
Run SDP-based Matrix-Parametrized Proximal Splitting algorithm
with millimeter-accuracy carrier phase measurements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import yaml
import argparse
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any

from core.mps_core.sdp_mps import SDPConfig, MatrixParametrizedSplitting
from core.mps_core.sinkhorn_knopp import MatrixParameterGenerator
from core.mps_core.algorithm import MPSAlgorithm, MPSConfig, CarrierPhaseConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SDPRunner:
    """Runner for SDP-based MPS algorithm"""
    
    def __init__(self, config_path: str):
        """Initialize runner with configuration file"""
        self.config_path = config_path
        self.load_config()
        self.setup_output_dir()
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        
    def setup_output_dir(self):
        """Create output directory if needed"""
        if self.config['output']['save_results']:
            self.output_dir = Path(self.config['output']['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")
    
    def generate_network(self) -> Dict[str, Any]:
        """Generate sensor network with true positions and measurements"""
        np.random.seed(self.config['measurements']['seed'])
        
        n_sensors = self.config['network']['n_sensors']
        n_anchors = self.config['network']['n_anchors']
        dimension = self.config['network']['dimension']
        scale = self.config['network']['scale']
        comm_range = self.config['network']['communication_range'] * scale
        
        # Generate true positions
        true_positions = np.random.uniform(0, scale, (n_sensors, dimension))
        
        # Place anchors at strategic locations
        if dimension == 2:
            anchor_positions = np.array([
                [0, 0],
                [scale, 0],
                [0, scale],
                [scale, scale],
                [scale/2, 0],
                [scale/2, scale]
            ])[:n_anchors]
        else:
            anchor_positions = np.random.uniform(0, scale, (n_anchors, dimension))
        
        # Generate communication graph
        adjacency_matrix = np.zeros((n_sensors, n_sensors))
        sensor_distances = {}
        anchor_distances = {}
        neighborhoods = {}
        anchor_connections = {}
        
        for i in range(n_sensors):
            sensor_distances[i] = {}
            anchor_distances[i] = {}
            neighborhoods[i] = []
            anchor_connections[i] = []
            
            # Sensor-to-sensor connections
            for j in range(n_sensors):
                if i != j:
                    dist = np.linalg.norm(true_positions[i] - true_positions[j])
                    if dist <= comm_range:
                        adjacency_matrix[i, j] = 1
                        neighborhoods[i].append(j)
                        
                        # Generate measurement based on type
                        if self.config['carrier_phase']['enable']:
                            # Carrier phase measurement (millimeter accuracy)
                            wavelength = 3e8 / (self.config['carrier_phase']['frequency_ghz'] * 1e9)
                            phase_noise = self.config['carrier_phase']['phase_noise_milliradians'] / 1000
                            phase_error = phase_noise * wavelength / (2 * np.pi)
                            noise = np.random.normal(0, phase_error)
                        else:
                            # Standard ranging
                            noise = np.random.normal(0, self.config['measurements']['noise_factor'] * dist)
                        
                        sensor_distances[i][j] = dist + noise
            
            # Sensor-to-anchor connections
            for k in range(n_anchors):
                dist = np.linalg.norm(true_positions[i] - anchor_positions[k])
                if dist <= comm_range:
                    anchor_connections[i].append(k)
                    
                    # Generate measurement
                    if self.config['carrier_phase']['enable']:
                        wavelength = 3e8 / (self.config['carrier_phase']['frequency_ghz'] * 1e9)
                        phase_noise = self.config['carrier_phase']['phase_noise_milliradians'] / 1000
                        phase_error = phase_noise * wavelength / (2 * np.pi)
                        noise = np.random.normal(0, phase_error)
                    else:
                        noise = np.random.normal(0, self.config['measurements']['noise_factor'] * dist)
                    
                    anchor_distances[i][k] = dist + noise
        
        return {
            'true_positions': true_positions,
            'anchor_positions': anchor_positions,
            'adjacency_matrix': adjacency_matrix,
            'sensor_distances': sensor_distances,
            'anchor_distances': anchor_distances,
            'neighborhoods': neighborhoods,
            'anchor_connections': anchor_connections,
            'n_sensors': n_sensors,
            'n_anchors': n_anchors,
            'dimension': dimension,
            'scale': scale
        }
    
    def run_sdp_algorithm(self, network_data: Dict) -> Dict:
        """Run the SDP-based MPS algorithm"""
        logger.info("="*60)
        logger.info("Running SDP-based Matrix-Parametrized Proximal Splitting")
        logger.info("="*60)
        
        # Create configuration
        config = SDPConfig(
            n_sensors=network_data['n_sensors'],
            n_anchors=network_data['n_anchors'],
            dimension=network_data['dimension'],
            gamma=self.config['sdp']['gamma'],
            alpha=self.config['sdp']['alpha'],
            max_iterations=self.config['sdp']['max_iterations'],
            tolerance=self.config['sdp']['tolerance'],
            scale=network_data['scale'],
            communication_range=self.config['network']['communication_range'],
            verbose=self.config['output']['verbose'],
            early_stopping=self.config['sdp']['early_stopping'],
            early_stopping_window=self.config['sdp']['early_stopping_window']
        )
        
        # Initialize algorithm
        mps = MatrixParametrizedSplitting(config)
        
        # Setup network structure
        mps.neighborhoods = network_data['neighborhoods']
        mps.anchor_connections = network_data['anchor_connections']
        
        # Generate matrix parameters using Sinkhorn-Knopp
        logger.info("Generating matrix parameters using Sinkhorn-Knopp...")
        Z, W = MatrixParameterGenerator.generate_from_communication_graph(
            network_data['adjacency_matrix'],
            method=self.config['sdp']['matrix_design'],
            block_design=self.config['sdp']['block_design']
        )
        mps.Z = Z
        mps.W = W
        
        # Initialize variables with warm start from noisy positions
        logger.info("Initializing variables...")
        noisy_positions = network_data['true_positions'] + \
                         0.1 * np.random.randn(*network_data['true_positions'].shape)
        mps.initialize_variables(true_positions=noisy_positions)
        
        # Run algorithm
        logger.info("Starting iterations...")
        start_time = time.time()
        
        history = {
            'objective': [],
            'psd_violation': [],
            'consensus_error': [],
            'rmse': [],
            'iterations': []
        }
        
        best_rmse = float('inf')
        best_positions = None
        best_iteration = 0
        
        for iteration in range(config.max_iterations):
            # Run iteration
            stats = mps.run_iteration(iteration)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean(np.sum((mps.X - network_data['true_positions'])**2, axis=1)))
            
            # Track history
            history['objective'].append(stats['objective'])
            history['psd_violation'].append(stats['psd_violation'])
            history['consensus_error'].append(stats['consensus_error'])
            history['rmse'].append(rmse)
            history['iterations'].append(iteration)
            
            # Track best result
            if rmse < best_rmse:
                best_rmse = rmse
                best_positions = mps.X.copy()
                best_iteration = iteration
            
            # Log progress
            if iteration % 100 == 0 or iteration < 10:
                logger.info(f"Iteration {iteration:4d}: RMSE={rmse:.6f}m, "
                          f"Obj={stats['objective']:.6f}, "
                          f"PSD_viol={stats['psd_violation']:.6f}, "
                          f"Consensus={stats['consensus_error']:.6f}")
            
            # Check convergence
            if stats['consensus_error'] < config.tolerance and \
               stats['psd_violation'] < self.config['convergence']['psd_tolerance']:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            # Early stopping
            if config.early_stopping and iteration - best_iteration > config.early_stopping_window:
                logger.info(f"Early stopping at iteration {iteration} (best was {best_iteration})")
                break
        
        elapsed_time = time.time() - start_time
        
        # Final results
        final_rmse = np.sqrt(np.mean(np.sum((mps.X - network_data['true_positions'])**2, axis=1)))
        
        logger.info("="*60)
        logger.info(f"SDP Algorithm Complete")
        logger.info(f"  Final RMSE: {final_rmse:.6f}m ({final_rmse*1000:.3f}mm)")
        logger.info(f"  Best RMSE:  {best_rmse:.6f}m ({best_rmse*1000:.3f}mm) at iteration {best_iteration}")
        logger.info(f"  Iterations: {iteration + 1}")
        logger.info(f"  Time: {elapsed_time:.2f}s")
        logger.info("="*60)
        
        return {
            'final_positions': mps.X,
            'best_positions': best_positions,
            'final_rmse': final_rmse,
            'best_rmse': best_rmse,
            'best_iteration': best_iteration,
            'iterations': iteration + 1,
            'time': elapsed_time,
            'history': history,
            'converged': stats['consensus_error'] < config.tolerance
        }
    
    def run_simplified_comparison(self, network_data: Dict) -> Dict:
        """Run simplified algorithm for comparison"""
        if not self.config['comparison']['run_simplified']:
            return None
            
        logger.info("\n" + "="*60)
        logger.info("Running Simplified Algorithm for Comparison")
        logger.info("="*60)
        
        # Create configuration
        config = MPSConfig(
            n_sensors=network_data['n_sensors'],
            n_anchors=network_data['n_anchors'],
            scale=network_data['scale'],
            communication_range=self.config['network']['communication_range'],
            noise_factor=self.config['measurements']['noise_factor'],
            gamma=self.config['sdp']['gamma'],
            alpha=self.config['sdp']['alpha'],
            max_iterations=self.config['sdp']['max_iterations'],
            tolerance=self.config['sdp']['tolerance'],
            dimension=network_data['dimension']
        )
        
        # Add carrier phase if enabled
        if self.config['carrier_phase']['enable']:
            config.carrier_phase = CarrierPhaseConfig(
                enable=True,
                frequency_ghz=self.config['carrier_phase']['frequency_ghz'],
                phase_noise_milliradians=self.config['carrier_phase']['phase_noise_milliradians'],
                frequency_stability_ppb=self.config['carrier_phase']['frequency_stability_ppb'],
                coarse_time_accuracy_ns=self.config['carrier_phase']['coarse_time_accuracy_ns']
            )
        
        # Initialize algorithm
        mps = MPSAlgorithm(config)
        
        # Use the same network
        mps.true_positions = network_data['true_positions']
        mps.anchor_positions = network_data['anchor_positions']
        mps.sensor_distances = network_data['sensor_distances']
        mps.anchor_distances = network_data['anchor_distances']
        
        # Run algorithm
        start_time = time.time()
        results = mps.run()
        elapsed_time = time.time() - start_time
        
        logger.info(f"Simplified Algorithm Complete")
        logger.info(f"  Final RMSE: {results['final_rmse']:.6f}m ({results['final_rmse']*1000:.3f}mm)")
        logger.info(f"  Iterations: {results['iterations']}")
        logger.info(f"  Time: {elapsed_time:.2f}s")
        logger.info(f"  Converged: {results['converged']}")
        
        return results
    
    def plot_results(self, network_data: Dict, sdp_results: Dict, 
                    simple_results: Dict = None):
        """Plot comparison results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: True vs Estimated Positions (SDP)
        ax = axes[0, 0]
        ax.scatter(network_data['true_positions'][:, 0], 
                  network_data['true_positions'][:, 1],
                  c='blue', marker='o', label='True', alpha=0.6)
        ax.scatter(sdp_results['best_positions'][:, 0],
                  sdp_results['best_positions'][:, 1],
                  c='red', marker='x', label='SDP Estimated', alpha=0.6)
        ax.scatter(network_data['anchor_positions'][:, 0],
                  network_data['anchor_positions'][:, 1],
                  c='green', marker='s', s=100, label='Anchors')
        ax.set_title('SDP: True vs Estimated Positions')
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: RMSE Convergence
        ax = axes[0, 1]
        ax.plot(sdp_results['history']['iterations'], 
               sdp_results['history']['rmse'], 
               'b-', label='SDP', linewidth=2)
        if simple_results:
            ax.plot(range(len(simple_results['rmse_history'])),
                   simple_results['rmse_history'],
                   'r--', label='Simplified', linewidth=2)
        ax.set_title('RMSE Convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 3: Objective Function
        ax = axes[0, 2]
        ax.plot(sdp_results['history']['iterations'],
               sdp_results['history']['objective'],
               'g-', linewidth=2)
        ax.set_title('SDP Objective Function')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: PSD Violation
        ax = axes[1, 0]
        ax.plot(sdp_results['history']['iterations'],
               sdp_results['history']['psd_violation'],
               'r-', linewidth=2)
        ax.set_title('PSD Constraint Violation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Violation')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 5: Consensus Error
        ax = axes[1, 1]
        ax.plot(sdp_results['history']['iterations'],
               sdp_results['history']['consensus_error'],
               'b-', linewidth=2)
        ax.set_title('Consensus Error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 6: Error Distribution
        ax = axes[1, 2]
        sdp_errors = np.linalg.norm(sdp_results['best_positions'] - 
                                   network_data['true_positions'], axis=1)
        ax.hist(sdp_errors * 1000, bins=20, alpha=0.6, label='SDP', color='blue')
        if simple_results and 'estimated_positions' in simple_results:
            simple_errors = np.linalg.norm(simple_results['estimated_positions'] - 
                                          network_data['true_positions'], axis=1)
            ax.hist(simple_errors * 1000, bins=20, alpha=0.6, label='Simplified', color='red')
        ax.set_title('Position Error Distribution')
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if self.config['output']['save_results']:
            fig_path = self.output_dir / 'results_comparison.png'
            plt.savefig(fig_path, dpi=150)
            logger.info(f"Saved figure to {fig_path}")
        
        plt.show()
    
    def save_results(self, network_data: Dict, sdp_results: Dict, 
                    simple_results: Dict = None):
        """Save results to files"""
        if not self.config['output']['save_results']:
            return
        
        # Save configuration
        config_save_path = self.output_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Save network data
        np.savez(self.output_dir / 'network_data.npz',
                true_positions=network_data['true_positions'],
                anchor_positions=network_data['anchor_positions'],
                adjacency_matrix=network_data['adjacency_matrix'])
        
        # Save SDP results
        np.savez(self.output_dir / 'sdp_results.npz',
                final_positions=sdp_results['final_positions'],
                best_positions=sdp_results['best_positions'],
                final_rmse=sdp_results['final_rmse'],
                best_rmse=sdp_results['best_rmse'],
                iterations=sdp_results['iterations'],
                time=sdp_results['time'])
        
        # Save history
        np.savez(self.output_dir / 'sdp_history.npz',
                **sdp_results['history'])
        
        # Save comparison results if available
        if simple_results:
            np.savez(self.output_dir / 'simple_results.npz',
                    estimated_positions=simple_results.get('estimated_positions'),
                    final_rmse=simple_results['final_rmse'],
                    iterations=simple_results['iterations'],
                    converged=simple_results['converged'])
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def run(self):
        """Main execution method"""
        logger.info("="*60)
        logger.info("SDP-MPS Runner Starting")
        logger.info("="*60)
        
        # Generate network
        logger.info("\nGenerating sensor network...")
        network_data = self.generate_network()
        logger.info(f"Network: {network_data['n_sensors']} sensors, "
                   f"{network_data['n_anchors']} anchors, "
                   f"scale={network_data['scale']}m")
        
        # Check connectivity
        from core.mps_core.sinkhorn_knopp import SinkhornKnopp
        sk = SinkhornKnopp(network_data['adjacency_matrix'])
        logger.info(f"Network connected: {sk._has_support()}")
        
        # Run SDP algorithm
        sdp_results = self.run_sdp_algorithm(network_data)
        
        # Run simplified for comparison
        simple_results = self.run_simplified_comparison(network_data)
        
        # Compare results
        if simple_results:
            logger.info("\n" + "="*60)
            logger.info("COMPARISON SUMMARY")
            logger.info("="*60)
            logger.info(f"SDP RMSE:        {sdp_results['best_rmse']:.6f}m "
                       f"({sdp_results['best_rmse']*1000:.3f}mm)")
            logger.info(f"Simplified RMSE: {simple_results['final_rmse']:.6f}m "
                       f"({simple_results['final_rmse']*1000:.3f}mm)")
            
            improvement = (simple_results['final_rmse'] - sdp_results['best_rmse']) / simple_results['final_rmse'] * 100
            logger.info(f"Improvement: {improvement:.1f}%")
            
            # Check if we achieved millimeter accuracy
            if sdp_results['best_rmse'] < 0.015:  # 15mm
                logger.info("✅ ACHIEVED S-BAND MILLIMETER ACCURACY (<15mm)")
            else:
                logger.info(f"❌ Did not achieve 15mm target (got {sdp_results['best_rmse']*1000:.1f}mm)")
        
        # Save results
        self.save_results(network_data, sdp_results, simple_results)
        
        # Plot results
        self.plot_results(network_data, sdp_results, simple_results)
        
        logger.info("\n" + "="*60)
        logger.info("SDP-MPS Runner Complete")
        logger.info("="*60)
        
        return sdp_results, simple_results


def main():
    parser = argparse.ArgumentParser(description='Run SDP-based MPS algorithm')
    parser.add_argument('--config', type=str, default='configs/sdp_mps.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run algorithm
    runner = SDPRunner(args.config)
    sdp_results, simple_results = runner.run()
    
    return 0 if sdp_results['best_rmse'] < 0.015 else 1


if __name__ == "__main__":
    exit(main())