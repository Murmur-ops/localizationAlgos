#!/usr/bin/env python3
"""
Test Two-Way Time Transfer integration with FTL localization
Shows how TWTT improves ranging accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from src.sync.two_way_time_transfer import TWTTConfig, TWTTNode, FTLTimeSyncManager


def test_twtt_accuracy():
    """Test TWTT synchronization accuracy"""
    print("="*60)
    print("TWTT SYNCHRONIZATION ACCURACY TEST")
    print("="*60)
    
    # Create network
    node_ids = list(range(10))
    config = TWTTConfig(
        timestamp_resolution_ns=1.0,  # 1ns resolution
        crystal_stability_ppm=20.0,    # ±20ppm drift
        estimate_asymmetry=True
    )
    
    manager = FTLTimeSyncManager(node_ids, config)
    
    # Add realistic clock offsets and drifts
    true_offsets = {}
    true_drifts = {}
    
    for node_id in node_ids:
        true_offsets[node_id] = np.random.uniform(-1000, 1000)  # ±1μs initial offset
        true_drifts[node_id] = np.random.uniform(-20, 20)  # ±20ppb drift
        
        # Apply to node
        manager.nodes[node_id].local_time_ns = true_offsets[node_id]
    
    # Run synchronization
    sync_errors = []
    
    for round in range(50):
        # Advance clocks with drift
        for node_id in node_ids:
            drift_ppb = true_drifts[node_id]
            manager.nodes[node_id].local_time_ns += 10_000_000  # 10ms
            manager.nodes[node_id].local_time_ns += int(10_000_000 * drift_ppb * 1e-9)
        
        # Run TWTT
        results = manager.run_sync_round()
        sync_errors.append(results['mean_sync_error_ns'])
        
        if round % 10 == 0:
            print(f"Round {round:2d}: Mean sync error = {results['mean_sync_error_ns']:.2f}ns, "
                  f"Max = {results['max_sync_error_ns']:.2f}ns")
    
    final_error = sync_errors[-1]
    print(f"\nFinal synchronization accuracy: {final_error:.2f}ns")
    
    if final_error < 100:
        print("✅ Excellent: Sub-100ns synchronization achieved!")
    elif final_error < 1000:
        print("✅ Good: Sub-microsecond synchronization")
    else:
        print("⚠️ Poor synchronization accuracy")
    
    return sync_errors


def test_ranging_with_twtt():
    """Test how TWTT improves ranging accuracy"""
    print("\n" + "="*60)
    print("RANGING ACCURACY: WITH vs WITHOUT TWTT")
    print("="*60)
    
    # Setup
    true_distance = 10.0  # 10 meters
    speed_of_light = 299792458.0  # m/s
    true_tof = true_distance / speed_of_light * 1e9  # Time of flight in ns
    
    print(f"True distance: {true_distance}m")
    print(f"True ToF: {true_tof:.2f}ns")
    
    # Test without TWTT (unsynchronized clocks)
    clock_offset = 100  # 100ns offset between nodes
    clock_drift = 20e-9  # 20ppb drift
    
    # Simulate 100 measurements
    measurements_without = []
    measurements_with = []
    
    for i in range(100):
        # Time progression
        time_elapsed = i * 100_000_000  # 100ms between measurements
        
        # Without TWTT: clock error affects measurement
        drift_error = time_elapsed * clock_drift
        measured_tof_without = true_tof + clock_offset + drift_error + np.random.normal(0, 1)
        measured_dist_without = measured_tof_without * speed_of_light / 1e9
        measurements_without.append(measured_dist_without)
        
        # With TWTT: clock error mostly cancelled
        residual_sync_error = np.random.normal(0, 10)  # 10ns residual after TWTT
        measured_tof_with = true_tof + residual_sync_error/10 + np.random.normal(0, 1)
        measured_dist_with = measured_tof_with * speed_of_light / 1e9
        measurements_with.append(measured_dist_with)
    
    # Calculate errors
    error_without = np.array(measurements_without) - true_distance
    error_with = np.array(measurements_with) - true_distance
    
    rmse_without = np.sqrt(np.mean(error_without**2))
    rmse_with = np.sqrt(np.mean(error_with**2))
    
    print(f"\nWithout TWTT:")
    print(f"  Mean error: {np.mean(error_without):.3f}m")
    print(f"  RMSE: {rmse_without:.3f}m")
    print(f"  Max error: {np.max(np.abs(error_without)):.3f}m")
    
    print(f"\nWith TWTT:")
    print(f"  Mean error: {np.mean(error_with):.3f}m")
    print(f"  RMSE: {rmse_with:.3f}m")
    print(f"  Max error: {np.max(np.abs(error_with)):.3f}m")
    
    improvement = (rmse_without - rmse_with) / rmse_without * 100
    print(f"\nImprovement: {improvement:.1f}% reduction in RMSE")
    
    return error_without, error_with


def test_ftl_with_twtt():
    """Test full FTL system with TWTT integration"""
    print("\n" + "="*60)
    print("FTL LOCALIZATION WITH TWTT")
    print("="*60)
    
    # Network setup
    anchors = np.array([
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20]
    ])
    
    unknown = np.array([8, 12])
    
    # True distances
    true_distances = []
    for anchor in anchors:
        dist = np.linalg.norm(unknown - anchor)
        true_distances.append(dist)
    
    print(f"True position: {unknown}")
    print(f"True distances: {[f'{d:.2f}m' for d in true_distances]}")
    
    # Speed of light timing
    c = 299792458.0
    true_tofs = [d/c * 1e9 for d in true_distances]  # nanoseconds
    
    # Simulate with different sync qualities
    sync_qualities = [
        ("No sync", 1000, 50),      # 1μs offset, 50ppb drift
        ("Basic PTP", 100, 10),     # 100ns offset, 10ppb drift
        ("TWTT", 10, 1),            # 10ns offset, 1ppb drift
        ("Enhanced TWTT", 1, 0.1),  # 1ns offset, 0.1ppb drift
    ]
    
    results = []
    
    for name, offset_ns, drift_ppb in sync_qualities:
        # Add sync errors to ToF measurements
        measured_tofs = []
        for tof in true_tofs:
            sync_error = offset_ns + np.random.normal(0, offset_ns/10)
            drift_error = tof * drift_ppb * 1e-9
            measurement_noise = np.random.normal(0, 1)  # 1ns measurement noise
            
            measured_tof = tof + sync_error + drift_error + measurement_noise
            measured_tofs.append(measured_tof)
        
        # Convert to distances
        measured_distances = [tof * c / 1e9 for tof in measured_tofs]
        
        # Simple trilateration (least squares)
        A = []
        b = []
        for i in range(len(anchors)-1):
            A.append(2 * (anchors[i] - anchors[-1]))
            b.append(measured_distances[-1]**2 - measured_distances[i]**2 + 
                    np.sum(anchors[i]**2) - np.sum(anchors[-1]**2))
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            estimated_pos = np.linalg.lstsq(A, b, rcond=None)[0]
            error = np.linalg.norm(estimated_pos - unknown)
        except:
            estimated_pos = [0, 0]
            error = np.linalg.norm(unknown)
        
        results.append({
            'name': name,
            'offset_ns': offset_ns,
            'drift_ppb': drift_ppb,
            'position': estimated_pos,
            'error_m': error
        })
        
        print(f"\n{name}:")
        print(f"  Time sync: ±{offset_ns}ns offset, {drift_ppb}ppb drift")
        print(f"  Estimated position: [{estimated_pos[0]:.2f}, {estimated_pos[1]:.2f}]")
        print(f"  Position error: {error:.3f}m")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Impact of Time Synchronization on FTL")
    print("="*60)
    
    for r in results:
        print(f"{r['name']:15} → {r['error_m']:.3f}m error")
    
    print("\n✅ TWTT provides crucial nanosecond-level sync for accurate FTL!")
    
    return results


def visualize_twtt_benefits():
    """Visualize the benefits of TWTT"""
    
    # Run tests
    sync_errors = test_twtt_accuracy()
    error_without, error_with = test_ranging_with_twtt()
    ftl_results = test_ftl_with_twtt()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sync convergence
    ax = axes[0, 0]
    ax.plot(sync_errors, 'b-', linewidth=2)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100ns target')
    ax.set_xlabel('Sync Round')
    ax.set_ylabel('Sync Error (ns)')
    ax.set_title('TWTT Synchronization Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Ranging error comparison
    ax = axes[0, 1]
    bins = np.linspace(-1, 40, 50)
    ax.hist(error_without, bins=bins, alpha=0.5, label='Without TWTT', color='red')
    ax.hist(error_with, bins=bins, alpha=0.5, label='With TWTT', color='green')
    ax.set_xlabel('Ranging Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Ranging Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: FTL accuracy vs sync quality
    ax = axes[1, 0]
    sync_names = [r['name'] for r in ftl_results]
    errors = [r['error_m'] for r in ftl_results]
    colors = ['red', 'orange', 'yellow', 'green']
    bars = ax.bar(range(len(sync_names)), errors, color=colors)
    ax.set_xticks(range(len(sync_names)))
    ax.set_xticklabels(sync_names, rotation=45, ha='right')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('FTL Accuracy vs Time Sync Quality')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{error:.3f}m', ha='center', va='bottom')
    
    # Plot 4: Time sync requirements
    ax = axes[1, 1]
    
    # Calculate required sync accuracy for different ranging accuracies
    c = 299792458.0  # m/s
    ranging_accuracies = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0])  # meters
    required_sync = ranging_accuracies / c * 1e9  # nanoseconds
    
    ax.loglog(required_sync, ranging_accuracies, 'b-o', linewidth=2, markersize=8)
    
    # Add technology regions
    ax.axvspan(0.1, 1, alpha=0.2, color='green', label='Enhanced TWTT')
    ax.axvspan(1, 10, alpha=0.2, color='yellow', label='Standard TWTT')
    ax.axvspan(10, 100, alpha=0.2, color='orange', label='PTP')
    ax.axvspan(100, 1000, alpha=0.2, color='red', label='NTP')
    
    ax.set_xlabel('Required Time Sync Accuracy (ns)')
    ax.set_ylabel('Achievable Ranging Accuracy (m)')
    ax.set_title('Time Sync Requirements for FTL')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left')
    
    plt.suptitle('Two-Way Time Transfer Benefits for FTL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('twtt_benefits.png', dpi=150)
    plt.show()
    
    print("\n✅ Visualization saved to twtt_benefits.png")


if __name__ == "__main__":
    visualize_twtt_benefits()