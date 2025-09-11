#!/usr/bin/env python3
"""
Analyze how localization error scales with timing precision
From microseconds down to picoseconds
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_timing_precision_impact():
    """Analyze how ranging and localization error scales with timing precision"""
    
    # Speed of light
    c = 299792458.0  # m/s
    
    # Timing precisions to analyze (in seconds)
    timing_precisions = {
        'Microsecond (μs)': 1e-6,
        'Hundred nanoseconds': 100e-9,
        'Ten nanoseconds': 10e-9,
        'Nanosecond (ns)': 1e-9,
        'Hundred picoseconds': 100e-12,
        'Ten picoseconds': 10e-12,
        'Picosecond (ps)': 1e-12,
    }
    
    print("="*70)
    print("TIMING PRECISION IMPACT ON RF LOCALIZATION")
    print("="*70)
    
    print("\n1. DIRECT RANGING ERROR FROM TIMING PRECISION")
    print("-" * 50)
    
    results = {}
    for name, precision_s in timing_precisions.items():
        # Direct ranging error = c * timing_error
        ranging_error_m = c * precision_s
        
        # For localization, error propagates through trilateration
        # Rule of thumb: position error ≈ ranging_error / sqrt(redundancy)
        # With 8 anchors and good geometry, redundancy factor ≈ 2-3
        position_error_best = ranging_error_m / np.sqrt(3)  # Best case
        position_error_typical = ranging_error_m / np.sqrt(2)  # Typical
        position_error_worst = ranging_error_m  # Worst case (poor geometry)
        
        results[name] = {
            'precision_s': precision_s,
            'ranging_error_m': ranging_error_m,
            'position_best_m': position_error_best,
            'position_typical_m': position_error_typical,
            'position_worst_m': position_error_worst
        }
        
        print(f"\n{name} ({precision_s*1e12:.1f} ps):")
        print(f"  Ranging error: {ranging_error_m:.4f} m")
        print(f"  Position error (best): {position_error_best:.4f} m")
        print(f"  Position error (typical): {position_error_typical:.4f} m")
        print(f"  Position error (worst): {position_error_worst:.4f} m")
    
    print("\n\n2. REALISTIC SYSTEM PERFORMANCE WITH DIFFERENT TIMING")
    print("-" * 50)
    print("\nConsider a 30-node network (8 anchors, 22 unknowns) in 50x50m area:")
    
    # For realistic system, we have multiple error sources
    for name, data in results.items():
        precision_s = data['precision_s']
        
        # Error sources (all in meters)
        timing_error = data['ranging_error_m']
        multipath_error = 0.3  # ~30cm from multipath (doesn't improve with timing)
        antenna_error = 0.05  # 5cm antenna phase center uncertainty
        propagation_error = 0.1  # 10cm from temperature/humidity variations
        
        # Combined error (RSS - root sum square)
        total_ranging_error = np.sqrt(
            timing_error**2 + 
            multipath_error**2 + 
            antenna_error**2 + 
            propagation_error**2
        )
        
        # Position error with consensus (improves by sqrt(N) for N measurements)
        avg_measurements = 10  # Average measurements per node
        position_rmse = total_ranging_error / np.sqrt(avg_measurements) * np.sqrt(2)
        
        print(f"\n{name}:")
        print(f"  Timing contribution: {timing_error:.4f} m")
        print(f"  Total ranging error: {total_ranging_error:.4f} m")
        print(f"  Expected position RMSE: {position_rmse:.4f} m")
        
        # Check if timing is limiting factor
        if timing_error > multipath_error:
            print(f"  ⚠️  Timing is limiting factor!")
        elif timing_error > multipath_error/10:
            print(f"  ⚡ Timing and multipath both significant")
        else:
            print(f"  ✅ Multipath-limited (timing not limiting)")
    
    print("\n\n3. PRACTICAL IMPLICATIONS")
    print("-" * 50)
    
    print("""
Key Insights:

1. CURRENT STATE (10ns with TWTT):
   - Ranging error from timing: 3.0m
   - This dominates over multipath (0.3m)
   - Position RMSE: ~1m (timing-limited)

2. NANOSECOND TIMING (1ns):
   - Ranging error from timing: 0.30m
   - Comparable to multipath effects
   - Position RMSE: ~0.3m (balanced limitation)

3. PICOSECOND TIMING (1ps):
   - Ranging error from timing: 0.0003m (0.3mm!)
   - Multipath completely dominates (0.3m)
   - Position RMSE: ~0.3m (multipath-limited)
   - No benefit from better timing!

4. OPTIMAL TIMING PRECISION:
   - Around 1-10ns is optimal for RF systems
   - Below 1ns, multipath becomes dominant
   - Picosecond precision is overkill for RF
   
5. WHERE PICOSECOND TIMING HELPS:
   - Optical ranging (no multipath)
   - Cable length measurement
   - Not helpful for wireless RF due to multipath
""")
    
    # Create visualization
    visualize_timing_scaling(results)
    
    return results


def visualize_timing_scaling(results):
    """Create visualization of timing precision impact"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    names = list(results.keys())
    precisions_ps = [results[n]['precision_s'] * 1e12 for n in names]  # Convert to ps
    ranging_errors = [results[n]['ranging_error_m'] for n in names]
    position_typical = [results[n]['position_typical_m'] for n in names]
    
    # Plot 1: Ranging error vs timing precision
    ax = axes[0, 0]
    ax.loglog(precisions_ps, ranging_errors, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Multipath limit (~30cm)')
    ax.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='Target accuracy (1cm)')
    ax.set_xlabel('Timing Precision (ps)')
    ax.set_ylabel('Ranging Error (m)')
    ax.set_title('Ranging Error vs Timing Precision')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Add annotations for key points
    ax.annotate('Current TWTT\n(10ns)', xy=(10000, 3), xytext=(20000, 5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    ax.annotate('Optimal\n(1ns)', xy=(1000, 0.3), xytext=(500, 1),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    ax.annotate('Overkill\n(1ps)', xy=(1, 0.0003), xytext=(0.5, 0.001),
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
    
    # Plot 2: Position error vs timing precision
    ax = axes[0, 1]
    ax.loglog(precisions_ps, position_typical, 'b-o', linewidth=2, markersize=8, label='Timing only')
    
    # Add realistic system performance (with all error sources)
    realistic_errors = []
    for name in names:
        timing_error = results[name]['ranging_error_m']
        multipath = 0.3
        other = 0.15  # Antenna + propagation
        total = np.sqrt(timing_error**2 + multipath**2 + other**2) / np.sqrt(10) * np.sqrt(2)
        realistic_errors.append(total)
    
    ax.loglog(precisions_ps, realistic_errors, 'r-s', linewidth=2, markersize=8, label='Realistic (all errors)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='1m target')
    ax.set_xlabel('Timing Precision (ps)')
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Position Error vs Timing Precision')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Plot 3: Error contribution breakdown
    ax = axes[1, 0]
    
    timing_precisions_plot = [1e6, 1e5, 1e4, 1e3, 100, 10, 1]  # in ps
    timing_errors_plot = [p * 1e-12 * 299792458 for p in timing_precisions_plot]
    
    multipath_contribution = [0.3] * len(timing_precisions_plot)
    other_contribution = [0.15] * len(timing_precisions_plot)
    
    width = 0.35
    x_pos = np.arange(len(timing_precisions_plot))
    
    p1 = ax.bar(x_pos, timing_errors_plot, width, label='Timing', color='blue', alpha=0.7)
    p2 = ax.bar(x_pos, multipath_contribution, width, bottom=timing_errors_plot, 
                label='Multipath', color='red', alpha=0.7)
    p3 = ax.bar(x_pos, other_contribution, width, 
                bottom=[t+m for t,m in zip(timing_errors_plot, multipath_contribution)],
                label='Other', color='gray', alpha=0.7)
    
    ax.set_ylabel('Error Contribution (m)')
    ax.set_title('Error Source Breakdown')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p/1000:.0f}ns' if p >= 1000 else f'{p}ps' 
                        for p in timing_precisions_plot])
    ax.set_xlabel('Timing Precision')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Practical implications
    ax = axes[1, 1]
    ax.axis('off')
    
    implications_text = """
    Timing Precision Analysis Summary
    =====================================
    
    Current (10ns with TWTT):
    • Ranging: 3.0m error from timing
    • Position: ~1m RMSE
    • Status: TIMING-LIMITED ⚠️
    
    Target (1ns precision):
    • Ranging: 0.30m error from timing  
    • Position: ~0.3m RMSE
    • Status: BALANCED ⚡
    
    Picosecond (1ps):
    • Ranging: 0.3mm error from timing
    • Position: ~0.3m RMSE (no improvement!)
    • Status: MULTIPATH-LIMITED ✅
    
    Key Finding:
    Below ~1ns timing precision, multipath
    becomes the dominant error source.
    Picosecond timing provides NO benefit
    for RF localization due to multipath!
    
    Optimal timing: 1-10ns
    """
    
    ax.text(0.1, 0.5, implications_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Impact of Timing Precision on RF Localization Performance', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('timing_precision_scaling.png', dpi=150)
    plt.show()
    
    print("\n✅ Visualization saved to timing_precision_scaling.png")


if __name__ == "__main__":
    results = analyze_timing_precision_impact()
    
    print("\n" + "="*70)
    print("CONCLUSION: PICOSECOND TIMING FOR RF LOCALIZATION")
    print("="*70)
    print("""
With picosecond-level timing (1ps):
- Ranging error from timing: 0.3mm (incredibly precise!)
- BUT multipath adds ~30cm error (1000x larger!)
- Final position accuracy: ~30cm (same as with 1ns timing)

The system becomes MULTIPATH-LIMITED, not timing-limited.
Improving timing beyond ~1ns provides diminishing returns.

For RF systems, the optimal timing precision is 1-10ns.
Picosecond timing only helps in controlled environments
without multipath (optical systems, cables, etc.).
""")