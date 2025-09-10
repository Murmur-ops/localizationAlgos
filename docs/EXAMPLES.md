# Comprehensive Examples

## Table of Contents
1. [Basic Examples](#basic-examples)
2. [Synchronization Examples](#synchronization-examples)
3. [GPS-Disciplined Networks](#gps-disciplined-networks)
4. [Distributed Processing](#distributed-processing)
5. [Analysis and Visualization](#analysis-and-visualization)
6. [Real-World Scenarios](#real-world-scenarios)

## Basic Examples

### Example 1: Simple Localization with Known Anchors

```python
#!/usr/bin/env python3
"""
Simple localization example with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.mps_advanced import AdvancedMPSAlgorithm

def simple_localization_demo():
    # Initialize algorithm
    mps = AdvancedMPSAlgorithm(n_sensors=10, n_anchors=4, noise_std=0.05)
    
    # Generate test network
    positions, measurements, anchor_positions = mps.generate_test_network()
    
    # Run localization
    estimated_positions = mps.localize(measurements, anchor_positions)
    
    # Calculate error
    rmse = mps.calculate_rmse(positions, estimated_positions)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    
    # True positions
    plt.scatter(positions[:, 0], positions[:, 1], 
                c='blue', label='True Positions', alpha=0.6)
    
    # Estimated positions
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], 
                c='red', marker='x', label='Estimated', s=100)
    
    # Anchors
    plt.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
                c='green', marker='^', s=200, label='Anchors')
    
    # Draw error lines
    for i in range(len(positions)):
        plt.plot([positions[i, 0], estimated_positions[i, 0]], 
                [positions[i, 1], estimated_positions[i, 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Localization Results (RMSE: {rmse:.3f}m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    return rmse

if __name__ == "__main__":
    rmse = simple_localization_demo()
    print(f"Final RMSE: {rmse:.3f}m")
```

### Example 2: Impact of Noise Levels

```python
#!/usr/bin/env python3
"""
Analyze how measurement noise affects localization accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.mps_advanced import AdvancedMPSAlgorithm

def noise_impact_analysis():
    noise_levels = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    rmse_results = []
    
    for noise in noise_levels:
        # Run multiple trials
        trials_rmse = []
        for trial in range(10):
            mps = AdvancedMPSAlgorithm(n_sensors=20, n_anchors=4, noise_std=noise)
            positions, measurements, anchors = mps.generate_test_network()
            estimated = mps.localize(measurements, anchors)
            rmse = mps.calculate_rmse(positions, estimated)
            trials_rmse.append(rmse)
        
        avg_rmse = np.mean(trials_rmse)
        rmse_results.append(avg_rmse)
        print(f"Noise {noise*100:.0f}%: RMSE = {avg_rmse:.3f}m")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(noise_levels)*100, rmse_results, 'b-o', linewidth=2)
    plt.xlabel('Measurement Noise (%)')
    plt.ylabel('RMSE (m)')
    plt.title('Impact of Measurement Noise on Localization Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    noise_impact_analysis()
```

## Synchronization Examples

### Example 3: Complete Time Synchronization Pipeline

```python
#!/usr/bin/env python3
"""
Demonstrate full synchronization pipeline with real timestamps
"""

import numpy as np
import time
from algorithms.time_sync.twtt import RealTWTT
from algorithms.time_sync.frequency_sync import RealFrequencySync
from algorithms.time_sync.consensus_clock import RealClockConsensus

def full_sync_demo():
    num_nodes = 10
    
    print("="*60)
    print("FULL TIME SYNCHRONIZATION DEMO")
    print("="*60)
    
    # Step 1: Initial time sync with TWTT
    print("\n1. Two-Way Time Transfer (TWTT)")
    print("-"*40)
    twtt = RealTWTT(num_nodes=num_nodes)
    offsets = twtt.synchronize_network(reference_node=0)
    
    print(f"Initial offsets (ns):")
    for node, offset in offsets.items():
        print(f"  Node {node}: {offset:.1f} ns")
    print(f"Achieved accuracy: {twtt.achieved_accuracy_ns:.1f} ns")
    
    # Step 2: Track frequency drift
    print("\n2. Frequency Synchronization")
    print("-"*40)
    freq_sync = RealFrequencySync(num_nodes=num_nodes)
    
    # Simulate tracking over time
    for round in range(5):
        time.sleep(0.1)  # Wait 100ms
        freq_sync.track_frequency_drift(duration_seconds=0.1)
    
    drift_estimates = freq_sync.get_drift_estimates()
    print(f"Frequency drift estimates (ppb):")
    for node, drift_ppb in drift_estimates.items():
        print(f"  Node {node}: {drift_ppb:.2f} ppb")
    
    # Step 3: Consensus clock
    print("\n3. Consensus Clock Synchronization")
    print("-"*40)
    consensus = RealClockConsensus(num_nodes=num_nodes)
    consensus.set_initial_offsets(offsets)
    
    # Run consensus
    final_offsets = consensus.achieve_consensus(rounds=20)
    
    print(f"Final synchronized offsets (ns):")
    for node, offset in final_offsets.items():
        print(f"  Node {node}: {offset:.1f} ns")
    
    # Calculate improvement
    initial_spread = max(offsets.values()) - min(offsets.values())
    final_spread = max(final_offsets.values()) - min(final_offsets.values())
    
    print(f"\nSummary:")
    print(f"  Initial spread: {initial_spread:.1f} ns")
    print(f"  Final spread: {final_spread:.1f} ns")
    print(f"  Improvement: {initial_spread/final_spread:.1f}x")
    
    # Convert to distance
    c = 299792458  # m/s
    distance_error = (final_spread / 1e9) * c
    print(f"  Distance error: {distance_error*100:.1f} cm")

if __name__ == "__main__":
    full_sync_demo()
```

### Example 4: Measuring Synchronization Limits

```python
#!/usr/bin/env python3
"""
Measure actual achievable synchronization accuracy
"""

import numpy as np
import time
from algorithms.time_sync.twtt import RealTWTT

def measure_sync_limits():
    print("MEASURING SYNCHRONIZATION LIMITS")
    print("="*60)
    
    node_counts = [2, 5, 10, 20, 50]
    results = {}
    
    for n_nodes in node_counts:
        print(f"\nTesting with {n_nodes} nodes...")
        
        # Run multiple trials
        accuracies = []
        for trial in range(5):
            twtt = RealTWTT(num_nodes=n_nodes)
            offsets = twtt.synchronize_network(reference_node=0)
            accuracies.append(twtt.achieved_accuracy_ns)
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        results[n_nodes] = {
            'mean': avg_accuracy,
            'std': std_accuracy,
            'distance_cm': (avg_accuracy / 1e9) * 299792458 * 100
        }
        
        print(f"  Accuracy: {avg_accuracy:.1f} ± {std_accuracy:.1f} ns")
        print(f"  Distance: {results[n_nodes]['distance_cm']:.1f} cm")
    
    print("\nSummary Table:")
    print(f"{'Nodes':<10} {'Sync (ns)':<15} {'Distance (cm)':<15}")
    print("-"*40)
    for n_nodes, res in results.items():
        print(f"{n_nodes:<10} {res['mean']:.1f} ± {res['std']:.1f}    "
              f"{res['distance_cm']:.1f}")
    
    return results

if __name__ == "__main__":
    results = measure_sync_limits()
```

## GPS-Disciplined Networks

### Example 5: GPS Anchor Comparison

```python
#!/usr/bin/env python3
"""
Compare localization with and without GPS-disciplined anchors
"""

from test_gps_anchors_fixed import GPSDisciplinedNetwork
import matplotlib.pyplot as plt
import numpy as np

def gps_comparison_demo():
    scales = [1, 10, 100]  # Network scales in meters
    gps_accuracies = [5, 10, 15, 20, 30]  # GPS sync accuracy in ns
    
    results_matrix = np.zeros((len(scales), len(gps_accuracies)))
    
    for i, scale in enumerate(scales):
        for j, gps_ns in enumerate(gps_accuracies):
            network = GPSDisciplinedNetwork(
                n_sensors=20,
                n_anchors=6,
                network_scale=scale,
                gps_time_sync_ns=gps_ns
            )
            
            results = network.run_localization_comparison()
            results_matrix[i, j] = results['improvement']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(gps_accuracies)))
    ax.set_xticklabels([f"{x}ns" for x in gps_accuracies])
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s}m" for s in scales])
    
    # Annotations
    for i in range(len(scales)):
        for j in range(len(gps_accuracies)):
            text = ax.text(j, i, f"{results_matrix[i, j]:.1f}x",
                          ha="center", va="center", color="black")
    
    ax.set_xlabel("GPS Time Sync Accuracy")
    ax.set_ylabel("Network Scale")
    ax.set_title("Improvement Factor with GPS-Disciplined Anchors")
    
    plt.colorbar(im, ax=ax, label="Improvement Factor")
    plt.tight_layout()
    plt.show()
    
    return results_matrix

if __name__ == "__main__":
    results = gps_comparison_demo()
```

## Distributed Processing

### Example 6: MPI-Based Distributed Localization

```python
#!/usr/bin/env python3
"""
Distributed MPS using MPI
Run with: mpirun -n 4 python distributed_example.py
"""

from mpi4py import MPI
import numpy as np
from algorithms.distributed_mps import DistributedMPS
import yaml

def distributed_mps_demo():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Load configuration
    with open('mps_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize distributed algorithm
    dmps = DistributedMPS(comm, config)
    
    if rank == 0:
        print(f"Running distributed MPS on {size} processes")
        print(f"Network: {config['algorithm']['n_sensors']} sensors, "
              f"{config['algorithm']['n_anchors']} anchors")
    
    # Generate or load network data
    if rank == 0:
        # Master generates network
        positions, measurements, anchors = dmps.generate_network()
        data = {'positions': positions, 'measurements': measurements, 
                'anchors': anchors}
    else:
        data = None
    
    # Broadcast network data
    data = comm.bcast(data, root=0)
    
    # Run distributed localization
    local_estimates = dmps.localize_distributed(
        data['measurements'], 
        data['anchors']
    )
    
    # Gather results
    all_estimates = comm.gather(local_estimates, root=0)
    
    if rank == 0:
        # Combine estimates
        final_estimates = dmps.combine_estimates(all_estimates)
        rmse = dmps.calculate_rmse(data['positions'], final_estimates)
        print(f"Distributed MPS RMSE: {rmse:.3f}m")
        return rmse
    
    return None

if __name__ == "__main__":
    rmse = distributed_mps_demo()
```

## Analysis and Visualization

### Example 7: Comprehensive Performance Analysis

```python
#!/usr/bin/env python3
"""
Analyze performance across multiple dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.mps_advanced import AdvancedMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer

def performance_analysis():
    # Parameters to vary
    n_sensors_list = [10, 20, 30, 40, 50]
    n_anchors_list = [3, 4, 5, 6]
    noise_levels = [0.01, 0.05, 0.10]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. RMSE vs Number of Sensors
    ax = axes[0, 0]
    for n_anchors in n_anchors_list:
        rmse_values = []
        for n_sensors in n_sensors_list:
            mps = AdvancedMPSAlgorithm(n_sensors, n_anchors, noise_std=0.05)
            positions, measurements, anchors = mps.generate_test_network()
            estimated = mps.localize(measurements, anchors)
            rmse = mps.calculate_rmse(positions, estimated)
            rmse_values.append(rmse)
        ax.plot(n_sensors_list, rmse_values, marker='o', 
                label=f'{n_anchors} anchors')
    ax.set_xlabel('Number of Sensors')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE vs Network Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. RMSE vs Noise Level
    ax = axes[0, 1]
    for n_anchors in [3, 4, 6]:
        rmse_values = []
        for noise in noise_levels:
            mps = AdvancedMPSAlgorithm(20, n_anchors, noise_std=noise)
            positions, measurements, anchors = mps.generate_test_network()
            estimated = mps.localize(measurements, anchors)
            rmse = mps.calculate_rmse(positions, estimated)
            rmse_values.append(rmse)
        ax.plot(np.array(noise_levels)*100, rmse_values, marker='s',
                label=f'{n_anchors} anchors')
    ax.set_xlabel('Measurement Noise (%)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Connectivity Impact
    ax = axes[1, 0]
    connectivity_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rmse_values = []
    for conn in connectivity_levels:
        mps = AdvancedMPSAlgorithm(20, 4, noise_std=0.05)
        # Generate network with specific connectivity
        positions, measurements, anchors = mps.generate_test_network(
            connectivity=conn
        )
        if measurements:  # Check if network is connected
            estimated = mps.localize(measurements, anchors)
            rmse = mps.calculate_rmse(positions, estimated)
            rmse_values.append(rmse)
        else:
            rmse_values.append(np.nan)
    ax.plot(connectivity_levels, rmse_values, 'g-o', linewidth=2)
    ax.set_xlabel('Network Connectivity')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Impact of Network Connectivity')
    ax.grid(True, alpha=0.3)
    
    # 4. CRLB Comparison
    ax = axes[1, 1]
    mps = AdvancedMPSAlgorithm(20, 4, noise_std=0.05)
    positions, measurements, anchors = mps.generate_test_network()
    
    # Calculate CRLB
    crlb_analyzer = CRLBAnalyzer()
    crlb = crlb_analyzer.calculate_crlb(positions, anchors, noise_std=0.05)
    
    # Run localization
    estimated = mps.localize(measurements, anchors)
    actual_errors = np.linalg.norm(positions - estimated, axis=1)
    
    # Plot comparison
    x = range(len(positions))
    ax.bar(x, actual_errors, alpha=0.7, label='Actual Error')
    ax.plot(x, crlb, 'r-', linewidth=2, label='CRLB (Lower Bound)')
    ax.set_xlabel('Sensor ID')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Actual Error vs CRLB')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Performance Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    performance_analysis()
```

## Real-World Scenarios

### Example 8: Indoor Positioning System

```python
#!/usr/bin/env python3
"""
Simulate indoor positioning with realistic constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.mps_advanced import AdvancedMPSAlgorithm

def indoor_positioning_demo():
    """
    Simulate positioning in a 20m x 15m indoor space
    with WiFi-based ranging (high noise)
    """
    
    # Indoor environment parameters
    room_width = 20.0  # meters
    room_height = 15.0  # meters
    
    # WiFi ranging has high noise (~20-30%)
    noise_std = 0.25  
    
    # Place anchors at room corners and center
    anchor_positions = np.array([
        [0, 0],                    # Corner 1
        [room_width, 0],           # Corner 2
        [room_width, room_height], # Corner 3
        [0, room_height],          # Corner 4
        [room_width/2, room_height/2]  # Center
    ])
    
    # Generate random sensor positions (people/devices)
    n_sensors = 15
    positions = np.random.uniform(
        low=[1, 1], 
        high=[room_width-1, room_height-1], 
        size=(n_sensors, 2)
    )
    
    # Simulate distance measurements with WiFi-level noise
    measurements = {}
    for i in range(n_sensors):
        # Sensor to sensor measurements (if in range)
        for j in range(i+1, n_sensors):
            true_dist = np.linalg.norm(positions[i] - positions[j])
            if true_dist < 10:  # WiFi range limit
                noise = np.random.normal(0, noise_std)
                measured = true_dist * (1 + noise)
                measurements[(i, j)] = measured
                measurements[(j, i)] = measured
        
        # Sensor to anchor measurements
        for a in range(len(anchor_positions)):
            true_dist = np.linalg.norm(positions[i] - anchor_positions[a])
            noise = np.random.normal(0, noise_std)
            measured = true_dist * (1 + noise)
            measurements[(i, f"anchor_{a}")] = measured
    
    # Run localization
    mps = AdvancedMPSAlgorithm(n_sensors, len(anchor_positions), noise_std)
    estimated = mps.localize_with_measurements(measurements, anchor_positions)
    
    # Calculate errors
    errors = np.linalg.norm(positions - estimated, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Positions
    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', 
                label='True Position', s=100, alpha=0.6)
    ax1.scatter(estimated[:, 0], estimated[:, 1], c='red', 
                marker='x', label='Estimated', s=100)
    ax1.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
                c='green', marker='^', label='WiFi Anchors', s=200)
    
    # Draw room boundaries
    room = plt.Rectangle((0, 0), room_width, room_height, 
                         fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(room)
    
    # Error lines
    for i in range(n_sensors):
        ax1.plot([positions[i, 0], estimated[i, 0]], 
                [positions[i, 1], estimated[i, 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlim(-1, room_width+1)
    ax1.set_ylim(-1, room_height+1)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Indoor Positioning (RMSE: {rmse:.2f}m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: Error distribution
    ax2.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(rmse, color='red', linestyle='--', linewidth=2, 
                label=f'RMSE: {rmse:.2f}m')
    ax2.set_xlabel('Position Error (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Indoor Positioning Results:")
    print(f"  RMSE: {rmse:.2f}m")
    print(f"  Mean Error: {np.mean(errors):.2f}m")
    print(f"  Max Error: {np.max(errors):.2f}m")
    print(f"  Min Error: {np.min(errors):.2f}m")
    print(f"  90th Percentile: {np.percentile(errors, 90):.2f}m")

if __name__ == "__main__":
    indoor_positioning_demo()
```

### Example 9: Testing Hardware vs Software Timing

```python
#!/usr/bin/env python3
"""
Compare software timing limitations with simulated hardware timing
"""

import numpy as np
from test_python_timing_limits import PythonTimingLimits

def hardware_vs_software_comparison():
    print("="*70)
    print("HARDWARE vs SOFTWARE TIMING COMPARISON")
    print("="*70)
    
    # Test actual Python capabilities
    python_test = PythonTimingLimits()
    capabilities = python_test.assess_python_capabilities()
    
    # Define timing scenarios
    scenarios = {
        "Python (Actual)": {
            "resolution_ns": capabilities['median_resolution_ns'],
            "sync_accuracy_ns": capabilities['achievable_sync_ns'],
            "distance_error_m": capabilities['distance_error_cm'] / 100
        },
        "GPS Disciplined": {
            "resolution_ns": 10,
            "sync_accuracy_ns": 15,
            "distance_error_m": 0.045  # 4.5cm
        },
        "FPGA": {
            "resolution_ns": 0.1,
            "sync_accuracy_ns": 0.2,
            "distance_error_m": 0.00006  # 0.06mm
        },
        "RF Phase Measurement": {
            "resolution_ns": 0.01,
            "sync_accuracy_ns": 0.01,
            "distance_error_m": 0.000003  # 3mm
        }
    }
    
    print("\nTiming Capabilities Comparison:")
    print(f"{'Platform':<25} {'Resolution':<12} {'Sync':<12} {'Distance':<12}")
    print("-"*61)
    
    for platform, specs in scenarios.items():
        res_str = f"{specs['resolution_ns']:.2f}ns"
        sync_str = f"{specs['sync_accuracy_ns']:.2f}ns"
        
        if specs['distance_error_m'] < 0.01:
            dist_str = f"{specs['distance_error_m']*1000:.2f}mm"
        elif specs['distance_error_m'] < 1:
            dist_str = f"{specs['distance_error_m']*100:.1f}cm"
        else:
            dist_str = f"{specs['distance_error_m']:.2f}m"
        
        print(f"{platform:<25} {res_str:<12} {sync_str:<12} {dist_str:<12}")
    
    # Calculate localization performance
    print("\nExpected Localization Performance:")
    print(f"{'Platform':<25} {'Expected RMSE':<20} {'S-band Ready?':<15}")
    print("-"*60)
    
    for platform, specs in scenarios.items():
        # Simple model: RMSE ≈ distance_error * sqrt(n_measurements)
        expected_rmse = specs['distance_error_m'] * np.sqrt(10)  # 10 measurements
        
        if expected_rmse < 0.01:
            rmse_str = f"{expected_rmse*1000:.2f}mm"
        elif expected_rmse < 1:
            rmse_str = f"{expected_rmse*100:.1f}cm"
        else:
            rmse_str = f"{expected_rmse:.2f}m"
        
        # S-band requires <1.5cm RMSE
        s_band_ready = "YES ✓" if expected_rmse < 0.015 else "NO ✗"
        
        print(f"{platform:<25} {rmse_str:<20} {s_band_ready:<15}")
    
    print("\n" + "="*70)
    print("CONCLUSION: Hardware timing is essential for S-band coherent")
    print("="*70)

if __name__ == "__main__":
    hardware_vs_software_comparison()
```

## Running the Examples

### Basic Usage
```bash
# Run any example directly
python example_1_simple_localization.py

# With visualization
python example_7_performance_analysis.py
```

### MPI Examples
```bash
# Run distributed examples with MPI
mpirun -n 4 python example_6_distributed_mps.py
```

### Batch Testing
```bash
# Run all examples
for i in {1..9}; do
    echo "Running Example $i..."
    python example_${i}_*.py
done
```

## Tips for Using Examples

1. **Start Simple**: Begin with Example 1 to understand basic localization
2. **Check Timing**: Run Example 4 to see your system's timing limitations
3. **Test Scale**: Try different network scales to see when sync helps
4. **Visualize**: Use matplotlib examples to understand spatial relationships
5. **Document Results**: Save outputs to compare different approaches

## Common Modifications

### Changing Network Size
```python
# Modify these parameters
n_sensors = 50  # Increase sensors
n_anchors = 8   # More anchors for better coverage
network_scale = 100.0  # Larger physical area
```

### Adjusting Noise Models
```python
# Different noise scenarios
noise_std = 0.01  # 1% - Very accurate sensors
noise_std = 0.50  # 50% - Very noisy (e.g., acoustic)
```

### Custom Anchor Placement
```python
# Strategic anchor positions
anchor_positions = np.array([
    [0, 0],           # Corner
    [10, 0],          # Edge
    [5, 5],           # Center
    [0, 10],          # Corner
    [10, 10]          # Corner
])
```

## Troubleshooting Examples

If examples fail to run:

1. **Check Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Python Version**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Test Timer Resolution**
   ```bash
   python test_python_timing_limits.py
   ```

4. **For MPI Issues**
   ```bash
   mpirun --version
   which mpirun
   ```