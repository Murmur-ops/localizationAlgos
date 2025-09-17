#!/usr/bin/env python3
"""
FTL Sensitivity Analysis Tests
Systematically varies SNR, Bandwidth, Clock Sync, and Geometry to understand performance limits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.channel.propagation import RangingChannel, ChannelConfig
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from ftl_sensitivity_utils import (
    compute_cramer_rao_bound,
    calculate_gdop,
    inject_clock_error,
    generate_test_network,
    run_localization_trial
)


@dataclass
class SensitivityResults:
    """Results from sensitivity analysis"""
    parameter_values: np.ndarray
    rmse_values: np.ndarray
    std_values: np.ndarray
    cramer_rao_bounds: np.ndarray
    convergence_iterations: np.ndarray


def test_snr_sensitivity(config: dict, n_trials: int = 10) -> SensitivityResults:
    """Test localization sensitivity to SNR variations"""

    print("\n" + "="*60)
    print("SNR SENSITIVITY ANALYSIS")
    print("="*60)

    # SNR values to test (in dB)
    snr_values_db = np.arange(-10, 45, 5)

    rmse_results = []
    std_results = []
    crb_results = []
    convergence_results = []

    for snr_db in snr_values_db:
        print(f"\nTesting SNR = {snr_db} dB")

        trial_rmse = []
        trial_iters = []

        for trial in range(n_trials):
            # Generate network
            anchors, unknowns = generate_test_network(config)

            # Configure channel with specific SNR
            channel_config = ChannelConfig()
            channel_config.bandwidth_hz = config['channel']['bandwidth_hz']

            # Convert SNR to noise power
            signal_power = 1.0  # Normalized
            noise_power = signal_power / (10 ** (snr_db / 10))

            channel = RangingChannel(channel_config)

            # Generate measurements with noise
            measurements = []
            for aid in anchors:
                for uid in unknowns:
                    true_dist = np.linalg.norm(anchors[aid] - unknowns[uid])

                    # Add noise based on SNR
                    noise_std = np.sqrt(noise_power) * true_dist * 0.01  # 1% of distance
                    measured_dist = true_dist + np.random.normal(0, noise_std)

                    # Quality based on SNR
                    quality = max(0.1, min(1.0, (snr_db + 20) / 40))

                    measurements.append(MeasurementEdge(
                        node_i=aid,
                        node_j=uid,
                        distance=measured_dist,
                        quality=quality,
                        variance=noise_std**2
                    ))

            # Run localization
            rmse, iterations = run_localization_trial(
                measurements, anchors, unknowns, config
            )

            trial_rmse.append(rmse)
            trial_iters.append(iterations)

        # Store results
        rmse_results.append(np.mean(trial_rmse))
        std_results.append(np.std(trial_rmse))
        convergence_results.append(np.mean(trial_iters))

        # Compute Cramér-Rao bound
        crb = compute_cramer_rao_bound(
            snr_linear=10**(snr_db/10),
            bandwidth_hz=float(config['channel']['bandwidth_hz']),
            n_measurements=len(measurements)
        )
        crb_results.append(crb)

        print(f"  RMSE: {rmse_results[-1]:.3f}m ± {std_results[-1]:.3f}m")
        print(f"  CRB: {crb:.3f}m")
        print(f"  Convergence: {convergence_results[-1]:.1f} iterations")

    return SensitivityResults(
        parameter_values=snr_values_db,
        rmse_values=np.array(rmse_results),
        std_values=np.array(std_results),
        cramer_rao_bounds=np.array(crb_results),
        convergence_iterations=np.array(convergence_results)
    )


def test_bandwidth_sensitivity(config: dict, n_trials: int = 10) -> SensitivityResults:
    """Test localization sensitivity to bandwidth variations"""

    print("\n" + "="*60)
    print("BANDWIDTH SENSITIVITY ANALYSIS")
    print("="*60)

    # Bandwidth values to test (in MHz)
    bandwidth_values_mhz = np.array([1, 5, 10, 20, 40, 80, 100, 200])

    rmse_results = []
    std_results = []
    crb_results = []
    convergence_results = []

    for bw_mhz in bandwidth_values_mhz:
        print(f"\nTesting Bandwidth = {bw_mhz} MHz")

        trial_rmse = []
        trial_iters = []

        for trial in range(n_trials):
            # Generate network
            anchors, unknowns = generate_test_network(config)

            # Configure channel with specific bandwidth
            channel_config = ChannelConfig()
            channel_config.bandwidth_hz = bw_mhz * 1e6

            channel = RangingChannel(channel_config)

            # Resolution floor based on bandwidth
            resolution_m = 3e8 / (2 * channel_config.bandwidth_hz)

            # Generate measurements
            measurements = []
            for aid in anchors:
                for uid in unknowns:
                    true_dist = np.linalg.norm(anchors[aid] - unknowns[uid])

                    # Quantization to resolution floor
                    quantized_dist = np.round(true_dist / resolution_m) * resolution_m

                    # Add thermal noise
                    snr_db = config['channel']['nominal_snr_db']
                    noise_std = resolution_m / np.sqrt(10**(snr_db/10))
                    measured_dist = quantized_dist + np.random.normal(0, noise_std)

                    measurements.append(MeasurementEdge(
                        node_i=aid,
                        node_j=uid,
                        distance=measured_dist,
                        quality=0.8,
                        variance=noise_std**2
                    ))

            # Run localization
            rmse, iterations = run_localization_trial(
                measurements, anchors, unknowns, config
            )

            trial_rmse.append(rmse)
            trial_iters.append(iterations)

        # Store results
        rmse_results.append(np.mean(trial_rmse))
        std_results.append(np.std(trial_rmse))
        convergence_results.append(np.mean(trial_iters))

        # Compute Cramér-Rao bound
        crb = compute_cramer_rao_bound(
            snr_linear=10**(float(config['channel']['nominal_snr_db'])/10),
            bandwidth_hz=float(bw_mhz * 1e6),
            n_measurements=len(measurements)
        )
        crb_results.append(crb)

        print(f"  Resolution floor: {resolution_m:.3f}m")
        print(f"  RMSE: {rmse_results[-1]:.3f}m ± {std_results[-1]:.3f}m")
        print(f"  CRB: {crb:.3f}m")

    return SensitivityResults(
        parameter_values=bandwidth_values_mhz,
        rmse_values=np.array(rmse_results),
        std_values=np.array(std_results),
        cramer_rao_bounds=np.array(crb_results),
        convergence_iterations=np.array(convergence_results)
    )


def test_clock_sync_sensitivity(config: dict, n_trials: int = 10) -> SensitivityResults:
    """Test localization sensitivity to clock synchronization errors"""

    print("\n" + "="*60)
    print("CLOCK SYNC SENSITIVITY ANALYSIS")
    print("="*60)

    # Clock offset values to test (in nanoseconds)
    clock_offset_ns = np.array([0, 1, 5, 10, 50, 100, 500, 1000])

    rmse_results = []
    std_results = []
    crb_results = []
    convergence_results = []

    for offset_ns in clock_offset_ns:
        print(f"\nTesting Clock Offset = {offset_ns} ns")

        trial_rmse = []
        trial_iters = []

        for trial in range(n_trials):
            # Generate network
            anchors, unknowns = generate_test_network(config)

            # Generate measurements
            measurements = []
            for aid in anchors:
                for uid in unknowns:
                    true_dist = np.linalg.norm(anchors[aid] - unknowns[uid])

                    # Inject clock error (1 ns = 0.3m ranging error)
                    clock_bias_m = inject_clock_error(offset_ns, drift_ppb=0)
                    measured_dist = true_dist + clock_bias_m

                    # Add thermal noise
                    noise_std = 0.1  # 10cm thermal noise
                    measured_dist += np.random.normal(0, noise_std)

                    measurements.append(MeasurementEdge(
                        node_i=aid,
                        node_j=uid,
                        distance=measured_dist,
                        quality=0.8,
                        variance=noise_std**2 + (clock_bias_m * 0.1)**2
                    ))

            # Run localization
            rmse, iterations = run_localization_trial(
                measurements, anchors, unknowns, config
            )

            trial_rmse.append(rmse)
            trial_iters.append(iterations)

        # Store results
        rmse_results.append(np.mean(trial_rmse))
        std_results.append(np.std(trial_rmse))
        convergence_results.append(np.mean(trial_iters))

        # CRB doesn't directly account for clock bias
        crb_results.append(offset_ns * 0.3e-9 * 3e8)  # Convert ns to meters

        print(f"  Clock bias: {offset_ns * 0.3}m")
        print(f"  RMSE: {rmse_results[-1]:.3f}m ± {std_results[-1]:.3f}m")

    return SensitivityResults(
        parameter_values=clock_offset_ns,
        rmse_values=np.array(rmse_results),
        std_values=np.array(std_results),
        cramer_rao_bounds=np.array(crb_results),
        convergence_iterations=np.array(convergence_results)
    )


def test_geometry_sensitivity(config: dict, n_trials: int = 10) -> SensitivityResults:
    """Test localization sensitivity to anchor geometry"""

    print("\n" + "="*60)
    print("GEOMETRY SENSITIVITY ANALYSIS")
    print("="*60)

    # Anchor density values to test (percentage of total nodes)
    anchor_percentages = np.array([10, 15, 20, 30, 40, 50, 60, 70])

    rmse_results = []
    std_results = []
    gdop_results = []
    convergence_results = []

    for anchor_pct in anchor_percentages:
        print(f"\nTesting Anchor Density = {anchor_pct}%")

        trial_rmse = []
        trial_gdop = []
        trial_iters = []

        for trial in range(n_trials):
            # Generate network with varying anchor density
            n_total = config['system']['num_nodes']
            n_anchors = max(4, int(n_total * anchor_pct / 100))
            n_unknowns = n_total - n_anchors

            # Create custom network
            area_size = config['system']['area_size_m']

            # Place anchors optimally (corners first, then edges, then interior)
            anchors = {}
            anchor_positions = []

            # Corner positions
            corners = [
                [0, 0], [area_size, 0],
                [area_size, area_size], [0, area_size]
            ]

            for i in range(min(4, n_anchors)):
                anchors[i] = np.array(corners[i])
                anchor_positions.append(corners[i])

            # Additional anchors on edges
            if n_anchors > 4:
                for i in range(4, n_anchors):
                    if i < 8:  # Edge anchors
                        edge = (i - 4) % 4
                        t = np.random.random()
                        if edge == 0:  # Bottom edge
                            pos = [t * area_size, 0]
                        elif edge == 1:  # Right edge
                            pos = [area_size, t * area_size]
                        elif edge == 2:  # Top edge
                            pos = [t * area_size, area_size]
                        else:  # Left edge
                            pos = [0, t * area_size]
                    else:  # Interior anchors
                        pos = np.random.uniform(0, area_size, 2)

                    anchors[i] = np.array(pos)
                    anchor_positions.append(pos)

            # Place unknowns randomly
            unknowns = {}
            for i in range(n_unknowns):
                unknowns[n_anchors + i] = np.random.uniform(0, area_size, 2)

            # Calculate GDOP for this geometry
            gdop = calculate_gdop(anchors, unknowns)
            trial_gdop.append(gdop)

            # Generate measurements
            measurements = []
            for aid in anchors:
                for uid in unknowns:
                    true_dist = np.linalg.norm(anchors[aid] - unknowns[uid])

                    # Add noise
                    noise_std = 0.1  # 10cm
                    measured_dist = true_dist + np.random.normal(0, noise_std)

                    measurements.append(MeasurementEdge(
                        node_i=aid,
                        node_j=uid,
                        distance=measured_dist,
                        quality=0.8,
                        variance=noise_std**2
                    ))

            # Run localization
            rmse, iterations = run_localization_trial(
                measurements, anchors, unknowns, config
            )

            trial_rmse.append(rmse)
            trial_iters.append(iterations)

        # Store results
        rmse_results.append(np.mean(trial_rmse))
        std_results.append(np.std(trial_rmse))
        gdop_results.append(np.mean(trial_gdop))
        convergence_results.append(np.mean(trial_iters))

        print(f"  GDOP: {gdop_results[-1]:.2f}")
        print(f"  RMSE: {rmse_results[-1]:.3f}m ± {std_results[-1]:.3f}m")
        print(f"  Convergence: {convergence_results[-1]:.1f} iterations")

    return SensitivityResults(
        parameter_values=anchor_percentages,
        rmse_values=np.array(rmse_results),
        std_values=np.array(std_results),
        cramer_rao_bounds=np.array(gdop_results),  # Using GDOP instead of CRB
        convergence_iterations=np.array(convergence_results)
    )


def visualize_sensitivity_results(
    snr_results: SensitivityResults,
    bw_results: SensitivityResults,
    clock_results: SensitivityResults,
    geom_results: SensitivityResults
):
    """Create comprehensive visualization of sensitivity analysis"""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # SNR Sensitivity
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.errorbar(snr_results.parameter_values, snr_results.rmse_values,
                 yerr=snr_results.std_values, fmt='o-', label='Measured RMSE', capsize=5)
    ax1.plot(snr_results.parameter_values, snr_results.cramer_rao_bounds,
             'r--', label='Cramér-Rao Bound')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_title('SNR Sensitivity (Precision Floor)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Bandwidth Sensitivity
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.errorbar(bw_results.parameter_values, bw_results.rmse_values,
                 yerr=bw_results.std_values, fmt='o-', label='Measured RMSE', capsize=5)
    ax2.plot(bw_results.parameter_values, bw_results.cramer_rao_bounds,
             'r--', label='Cramér-Rao Bound')
    ax2.plot(bw_results.parameter_values,
             3e8 / (2 * bw_results.parameter_values * 1e6),
             'g:', label='Resolution Floor')
    ax2.set_xlabel('Bandwidth (MHz)')
    ax2.set_ylabel('RMSE (m)')
    ax2.set_title('Bandwidth Sensitivity (Resolution Floor)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Clock Sync Sensitivity
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.errorbar(clock_results.parameter_values, clock_results.rmse_values,
                 yerr=clock_results.std_values, fmt='o-', label='Measured RMSE', capsize=5)
    ax3.plot(clock_results.parameter_values,
             clock_results.parameter_values * 0.3,
             'r--', label='Expected Bias (0.3m/ns)')
    ax3.set_xlabel('Clock Offset (ns)')
    ax3.set_ylabel('RMSE (m)')
    ax3.set_title('Clock Sync Sensitivity (Bias Floor)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Geometry Sensitivity
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.errorbar(geom_results.parameter_values, geom_results.rmse_values,
                 yerr=geom_results.std_values, fmt='o-', label='Measured RMSE', capsize=5)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(geom_results.parameter_values, geom_results.cramer_rao_bounds,
                  'g--', label='GDOP')
    ax4.set_xlabel('Anchor Density (%)')
    ax4.set_ylabel('RMSE (m)', color='b')
    ax4_twin.set_ylabel('GDOP', color='g')
    ax4.set_title('Geometry Sensitivity (Error Amplification)')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True, alpha=0.3)

    # Convergence Analysis
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(snr_results.parameter_values, snr_results.convergence_iterations,
             'o-', label='SNR variation')
    ax5.set_xlabel('SNR (dB)')
    ax5.set_ylabel('Iterations to Converge')
    ax5.set_title('Convergence vs SNR')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Combined Effects Heatmap
    ax6 = fig.add_subplot(gs[2, 2:])
    # Create synthetic interaction data
    snr_range = np.linspace(0, 40, 20)
    bw_range = np.linspace(10, 100, 20)
    SNR, BW = np.meshgrid(snr_range, bw_range)
    # Simple model: RMSE = base / (sqrt(SNR_linear) * sqrt(BW))
    RMSE = 100 / (np.sqrt(10**(SNR/10)) * np.sqrt(BW/10))

    im = ax6.contourf(SNR, BW, RMSE, levels=20, cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax6, label='RMSE (m)')
    ax6.set_xlabel('SNR (dB)')
    ax6.set_ylabel('Bandwidth (MHz)')
    ax6.set_title('SNR-Bandwidth Interaction Effect')

    plt.suptitle('FTL Sensitivity Analysis: The Four Fundamental Knobs',
                 fontsize=14, fontweight='bold')

    return fig


def main():
    """Run comprehensive FTL sensitivity analysis"""

    print("="*70)
    print("FTL SENSITIVITY ANALYSIS")
    print("Testing: SNR, Bandwidth, Clock Sync, and Geometry")
    print("="*70)

    # Load configuration
    with open('configs/sensitivity_test.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run sensitivity tests
    snr_results = test_snr_sensitivity(config, n_trials=5)
    bw_results = test_bandwidth_sensitivity(config, n_trials=5)
    clock_results = test_clock_sync_sensitivity(config, n_trials=5)
    geom_results = test_geometry_sensitivity(config, n_trials=5)

    # Visualize results
    fig = visualize_sensitivity_results(
        snr_results, bw_results, clock_results, geom_results
    )

    plt.savefig('ftl_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*70)
    print("Analysis complete! Results saved to ftl_sensitivity_analysis.png")
    print("="*70)

    # Print summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find critical thresholds
    snr_threshold = snr_results.parameter_values[
        np.where(snr_results.rmse_values < 1.0)[0][0] if np.any(snr_results.rmse_values < 1.0) else -1
    ]
    print(f"\n1. SNR Critical Threshold: {snr_threshold} dB for sub-meter accuracy")

    bw_threshold = bw_results.parameter_values[
        np.where(bw_results.rmse_values < 1.0)[0][0] if np.any(bw_results.rmse_values < 1.0) else -1
    ]
    print(f"2. Bandwidth Requirement: {bw_threshold} MHz minimum for sub-meter accuracy")

    clock_threshold = clock_results.parameter_values[
        np.where(clock_results.rmse_values > 1.0)[0][0] if np.any(clock_results.rmse_values > 1.0) else -1
    ]
    print(f"3. Clock Sync Requirement: <{clock_threshold} ns offset for sub-meter accuracy")

    anchor_threshold = geom_results.parameter_values[
        np.where(geom_results.rmse_values < 1.0)[0][0] if np.any(geom_results.rmse_values < 1.0) else -1
    ]
    print(f"4. Anchor Density: >{anchor_threshold}% nodes as anchors recommended")

    print("\n" + "="*70)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*70)
    print("\n1. Indoor (10×10m):")
    print("   - SNR: >20 dB (easily achievable)")
    print("   - Bandwidth: 100 MHz (UWB)")
    print("   - Clock: <10 ns (hardware timestamps)")
    print("   - Anchors: 4 corners sufficient")

    print("\n2. Urban (100×100m):")
    print("   - SNR: >15 dB (challenging with NLOS)")
    print("   - Bandwidth: 20-40 MHz (WiFi ToF)")
    print("   - Clock: <50 ns (PTP sync)")
    print("   - Anchors: 20-30% recommended")

    print("\n3. Rural (1000×1000m):")
    print("   - SNR: >10 dB (power limited)")
    print("   - Bandwidth: 10 MHz (LoRa)")
    print("   - Clock: <100 ns (GPS disciplined)")
    print("   - Anchors: Dense grid required")

    plt.show()


if __name__ == "__main__":
    main()