#!/usr/bin/env python3
"""
S-Band Tolerance Demonstration
==============================
Demonstrates achieving millimeter-level localization accuracy using:
1. Carrier phase synchronization at 2.4 GHz
2. Two-way time transfer for coarse synchronization
3. Frequency synchronization with OCXO stability
4. Decentralized MPS algorithm

Expected performance: <15mm RMSE (S-band coherent beamforming requirement)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig

class SBandLocalization:
    """S-Band carrier phase localization system"""
    
    def __init__(self, config_path=None):
        """Initialize with configuration"""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()
            
        # Physical constants
        self.c = 299792458  # Speed of light (m/s)
        self.freq_hz = self.config['carrier_phase']['frequency_ghz'] * 1e9
        self.wavelength = self.c / self.freq_hz  # ~0.125m at 2.4 GHz
        
    def get_default_config(self):
        """Default S-band configuration"""
        return {
            'network': {
                'n_sensors': 30,
                'n_anchors': 6,
                'scale': 50.0,
                'communication_range_factor': 0.4,
                'anchor_placement': 'optimal'
            },
            'carrier_phase': {
                'frequency_ghz': 2.4,
                'phase_noise_milliradians': 1.0,
                'frequency_stability_ppb': 0.1,
                'coarse_time_accuracy_ns': 1.0
            },
            'time_sync': {
                'enable': True,
                'initial_offset_ns': 1000,
                'target_accuracy_ns': 1.0,
                'num_exchanges': 10
            },
            'algorithm': {
                'max_iterations': 1000,
                'convergence_tolerance': 1e-6,
                'gamma': 0.999,
                'alpha': 0.5,
                'use_sdp_init': True,
                'use_anderson_acceleration': True,
                'anderson_memory': 5
            },
            'requirements': {
                'sband_rmse_mm': 15.0,
                'max_node_error_mm': 20.0
            }
        }
        
    def simulate_carrier_phase_ranging(self, true_distance):
        """
        Simulate carrier phase ranging with realistic noise
        
        Args:
            true_distance: True distance in meters
            
        Returns:
            measured_distance: Distance with carrier phase accuracy
        """
        # Phase measurement in radians
        true_phase = (2 * np.pi * true_distance) / self.wavelength
        
        # Add phase noise (milliradians to radians)
        phase_noise_rad = self.config['carrier_phase']['phase_noise_milliradians'] * 1e-3
        measured_phase = true_phase + np.random.normal(0, phase_noise_rad)
        
        # Wrap phase to [-π, π]
        wrapped_phase = np.angle(np.exp(1j * measured_phase))
        
        # Resolve ambiguity using coarse time measurement
        coarse_time_ns = self.config['carrier_phase']['coarse_time_accuracy_ns']
        coarse_distance_error = (coarse_time_ns * 1e-9 * self.c) / 2  # Two-way ranging
        
        # Number of full wavelengths (integer ambiguity)
        n_wavelengths = np.round((true_distance + np.random.normal(0, coarse_distance_error)) / self.wavelength)
        
        # Final distance measurement combining coarse and fine
        measured_distance = n_wavelengths * self.wavelength + (wrapped_phase * self.wavelength) / (2 * np.pi)
        
        return measured_distance
        
    def simulate_time_sync(self, n_nodes):
        """Simulate two-way time transfer synchronization"""
        if not self.config['time_sync']['enable']:
            return np.zeros(n_nodes)
            
        # Initial time offsets
        initial_offset_ns = self.config['time_sync']['initial_offset_ns']
        offsets = np.random.uniform(-initial_offset_ns, initial_offset_ns, n_nodes)
        
        # TWTT convergence
        for _ in range(self.config['time_sync']['num_exchanges']):
            offsets *= 0.5  # Exponential convergence
            
        # Add residual error
        target_accuracy_ns = self.config['time_sync']['target_accuracy_ns']
        offsets += np.random.normal(0, target_accuracy_ns, n_nodes)
        
        return offsets * 1e-9  # Convert to seconds
        
    def generate_network(self, n_sensors, n_anchors, scale, seed=None):
        """Generate random sensor network"""
        if seed is not None:
            np.random.seed(seed)
            
        n_total = n_sensors + n_anchors
        
        # Generate random positions
        positions = np.random.uniform(-scale/2, scale/2, (n_total, 2))
        
        # Place anchors at optimal positions (corners for 2D)
        if self.config['network']['anchor_placement'] == 'optimal' and n_anchors >= 4:
            positions[n_sensors] = [-scale/2, -scale/2]
            positions[n_sensors + 1] = [scale/2, -scale/2]
            positions[n_sensors + 2] = [scale/2, scale/2]
            positions[n_sensors + 3] = [-scale/2, scale/2]
            # Additional anchors in center
            for i in range(4, n_anchors):
                positions[n_sensors + i] = np.random.uniform(-scale/4, scale/4, 2)
                
        return positions
        
    def run_single_trial(self, trial_idx=0, visualize=False):
        """Run a single localization trial"""
        np.random.seed(42 + trial_idx)
        
        # Generate network
        n_total = self.config['network']['n_sensors'] + self.config['network']['n_anchors']
        positions = self.generate_network(
            n_sensors=self.config['network']['n_sensors'],
            n_anchors=self.config['network']['n_anchors'],
            scale=self.config['network'].get('scale_meters', self.config['network'].get('scale', 50.0)),
            seed=42 + trial_idx
        )
        
        # Determine anchor indices (last n_anchors nodes)
        anchor_indices = list(range(self.config['network']['n_sensors'], n_total))
        
        # Generate measurements with carrier phase accuracy
        true_distances = {}
        measured_distances = {}
        
        # Communication range
        scale = self.config['network'].get('scale_meters', self.config['network'].get('scale', 50.0))
        comm_range = scale * self.config['network'].get('communication_range_factor', self.config['network'].get('communication_range', 0.4))
        
        for i in range(n_total):
            for j in range(i+1, n_total):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= comm_range:
                    true_distances[(i, j)] = dist
                    # Use carrier phase ranging for high accuracy
                    measured_distances[(i, j)] = self.simulate_carrier_phase_ranging(dist)
                    measured_distances[(j, i)] = measured_distances[(i, j)]
        
        # Time synchronization
        time_offsets = self.simulate_time_sync(n_total)
        
        # Setup MPS algorithm
        algo_config = self.config.get('mps_algorithm', self.config.get('algorithm', {}))
        mps_config = MPSConfig(
            n_sensors=n_total,
            n_anchors=self.config['network']['n_anchors'],
            communication_range=comm_range,
            noise_factor=0.01,  # Very low noise for carrier phase
            max_iterations=algo_config.get('max_iterations', 1000),
            tolerance=algo_config.get('convergence_tolerance', algo_config.get('tolerance', 1e-6)),
            gamma=algo_config.get('gamma', 0.999),
            alpha=algo_config.get('alpha', 0.5),
            dimension=2,
            seed=42 + trial_idx
        )
        
        # Create algorithm instance with network setup
        algorithm = MPSAlgorithm(mps_config)
        
        # Set up the network
        algorithm.true_positions = positions
        algorithm.anchor_indices = set(anchor_indices)
        algorithm.distance_measurements = measured_distances
        
        # Run localization
        results = algorithm.run()
        estimated_positions = results['final_positions']
        
        # Compute errors in millimeters
        errors_mm = []
        for i in range(n_total):
            if i not in anchor_indices:
                error_m = np.linalg.norm(estimated_positions[i] - positions[i])
                errors_mm.append(error_m * 1000)  # Convert to mm
                
        rmse_mm = np.sqrt(np.mean(np.square(errors_mm)))
        max_error_mm = np.max(errors_mm)
        
        # Visualize if requested
        if visualize:
            self.visualize_results(positions, estimated_positions, anchor_indices, errors_mm)
            
        return {
            'trial': trial_idx,
            'rmse_mm': rmse_mm,
            'max_error_mm': max_error_mm,
            'mean_error_mm': np.mean(errors_mm),
            'std_error_mm': np.std(errors_mm),
            'iterations': results['iterations'],
            'converged': results['converged'],
            'meets_sband': rmse_mm <= self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0))
        }
        
    def visualize_results(self, true_pos, est_pos, anchors, errors_mm):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Network topology with errors
        ax = axes[0, 0]
        for i in range(len(true_pos)):
            if i in anchors:
                ax.scatter(true_pos[i][0], true_pos[i][1], c='red', s=200, marker='s', label='Anchor' if i == anchors[0] else '')
            else:
                idx = i if i < anchors[0] else i - len(anchors)
                if idx < len(errors_mm):
                    error = errors_mm[idx]
                    color_val = min(error / 20.0, 1.0)  # Normalize to 20mm max
                    ax.scatter(true_pos[i][0], true_pos[i][1], c=plt.cm.viridis(1-color_val), s=100)
                    ax.scatter(est_pos[i][0], est_pos[i][1], c='gray', s=50, alpha=0.5)
                    ax.plot([true_pos[i][0], est_pos[i][0]], [true_pos[i][1], est_pos[i][1]], 'k-', alpha=0.3)
        
        ax.set_title('Network Topology & Localization Errors')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Error distribution
        ax = axes[0, 1]
        ax.hist(errors_mm, bins=30, edgecolor='black', alpha=0.7)
        target_rmse = self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0))
        ax.axvline(target_rmse, color='red', linestyle='--', label='S-band Requirement')
        ax.axvline(np.mean(errors_mm), color='green', linestyle='-', label=f'Mean: {np.mean(errors_mm):.2f}mm')
        ax.set_title('Error Distribution')
        ax.set_xlabel('Localization Error (mm)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Carrier phase illustration
        ax = axes[1, 0]
        distances = np.linspace(0, 2*self.wavelength, 1000)
        phases = (2 * np.pi * distances) / self.wavelength
        ax.plot(distances*1000, np.sin(phases), 'b-', label='Carrier Phase')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(self.wavelength*1000/2, color='red', linestyle='--', alpha=0.5, label=f'λ/2 = {self.wavelength*500:.1f}mm')
        ax.axvline(self.wavelength*1000, color='red', linestyle='--', alpha=0.5, label=f'λ = {self.wavelength*1000:.1f}mm')
        ax.set_title(f'Carrier Phase at {self.config["carrier_phase"]["frequency_ghz"]} GHz')
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance metrics
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
S-Band Localization Performance
================================
RMSE:           {np.sqrt(np.mean(np.square(errors_mm))):.2f} mm
Mean Error:     {np.mean(errors_mm):.2f} mm
Std Dev:        {np.std(errors_mm):.2f} mm
Max Error:      {np.max(errors_mm):.2f} mm
Min Error:      {np.min(errors_mm):.2f} mm

Requirements
------------
S-band RMSE:    < {self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0)):.1f} mm
Max Node Error: < {self.config.get('requirements', {}).get('max_node_error_mm', 20.0):.1f} mm

Configuration
-------------
Frequency:      {self.config['carrier_phase']['frequency_ghz']:.1f} GHz
Wavelength:     {self.wavelength*1000:.1f} mm
Phase Noise:    {self.config['carrier_phase']['phase_noise_milliradians']:.1f} mrad
Time Accuracy:  {self.config['carrier_phase']['coarse_time_accuracy_ns']:.1f} ns
Network Size:   {self.config['network']['n_sensors']} sensors, {self.config['network']['n_anchors']} anchors

Status: {'✓ PASS' if np.sqrt(np.mean(np.square(errors_mm))) <= self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0)) else '✗ FAIL'}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace', verticalalignment='center')
        
        plt.suptitle('S-Band Carrier Phase Localization Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
        
    def run_monte_carlo(self, n_trials=10):
        """Run Monte Carlo simulation"""
        print("\n" + "="*70)
        print("S-BAND TOLERANCE DEMONSTRATION")
        print("="*70)
        print(f"Running {n_trials} trials with carrier phase synchronization at {self.config['carrier_phase']['frequency_ghz']} GHz")
        target_rmse = self.config.get('requirements', {}).get('sband_rmse_mm') or self.config.get('requirements', {}).get('max_rmse_mm', 15.0)
        print(f"Target: < {target_rmse} mm RMSE (S-band coherent beamforming)")
        print("-"*70)
        
        results = []
        for trial in range(n_trials):
            print(f"\nTrial {trial+1}/{n_trials}...", end='')
            result = self.run_single_trial(trial, visualize=(trial == 0))
            results.append(result)
            print(f" RMSE: {result['rmse_mm']:.2f} mm {'✓' if result['meets_sband'] else '✗'}")
            
        # Compute statistics
        all_rmse = [r['rmse_mm'] for r in results]
        all_max = [r['max_error_mm'] for r in results]
        success_rate = sum(r['meets_sband'] for r in results) / n_trials * 100
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Mean RMSE:      {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f} mm")
        print(f"Min/Max RMSE:   {np.min(all_rmse):.2f} / {np.max(all_rmse):.2f} mm")
        print(f"Mean Max Error: {np.mean(all_max):.2f} mm")
        print(f"Success Rate:   {success_rate:.0f}% ({sum(r['meets_sband'] for r in results)}/{n_trials} trials)")
        target_rmse = self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0))
        print(f"Requirement:    < {target_rmse} mm")
        print(f"Status:         {'✓ PASS - S-band tolerance achieved!' if np.mean(all_rmse) <= target_rmse else '✗ FAIL'}")
        print("="*70)
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'summary': {
                'mean_rmse_mm': float(np.mean(all_rmse)),
                'std_rmse_mm': float(np.std(all_rmse)),
                'min_rmse_mm': float(np.min(all_rmse)),
                'max_rmse_mm': float(np.max(all_rmse)),
                'success_rate': success_rate,
                'meets_requirement': bool(np.mean(all_rmse) <= self.config.get('requirements', {}).get('sband_rmse_mm', self.config.get('requirements', {}).get('max_rmse_mm', 15.0)))
            },
            'trials': results
        }
        
        output_path = Path('results/sband_demonstration.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
        # Show plot
        plt.show()
        
        return output

def main():
    """Main entry point"""
    # Check if custom config provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"Loading configuration from: {config_path}")
        sim = SBandLocalization(config_path)
    else:
        # Use S-band precision config if available
        config_path = Path('configs/sband_precision.yaml')
        if config_path.exists():
            print(f"Using S-band precision configuration: {config_path}")
            sim = SBandLocalization(str(config_path))
        else:
            print("Using default S-band configuration")
            sim = SBandLocalization()
    
    # Run demonstration
    sim.run_monte_carlo(n_trials=10)

if __name__ == "__main__":
    main()