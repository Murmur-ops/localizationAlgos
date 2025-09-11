#!/usr/bin/env python3
"""
Minimal 4-Node Localization Demo
Shows the complete pipeline working on a simple diamond configuration
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType

def run_minimal_demo():
    """Run a minimal 4-node localization demo"""
    print("\n" + "="*60)
    print("MINIMAL 4-NODE LOCALIZATION DEMO")
    print("="*60)
    
    # Define diamond configuration
    # 3 anchors in triangle, 1 unknown in center
    anchors = {
        0: np.array([0, 0]),      # Bottom left
        1: np.array([10, 0]),     # Bottom right  
        2: np.array([5, 8.66])    # Top (equilateral triangle)
    }
    
    # True position of unknown node
    true_position = np.array([5, 3])  # Center-ish
    
    print("\nNode Configuration:")
    print("  Anchors:")
    for aid, pos in anchors.items():
        print(f"    Node {aid}: ({pos[0]:.1f}, {pos[1]:.1f})")
    print(f"  Unknown:")
    print(f"    Node 3: ({true_position[0]:.1f}, {true_position[1]:.1f}) [TRUE]")
    
    # Generate measurements with realistic channel
    channel = RangingChannel(ChannelConfig())
    measurements = []
    
    print("\nGenerating Measurements:")
    
    # Measurements from unknown to each anchor
    for anchor_id, anchor_pos in anchors.items():
        true_dist = np.linalg.norm(true_position - anchor_pos)
        
        # Determine propagation type based on distance
        if true_dist < 5:
            prop_type = PropagationType.LOS
        else:
            prop_type = PropagationType.LOS if np.random.rand() > 0.2 else PropagationType.NLOS
        
        # Generate measurement
        meas = channel.generate_measurement(true_dist, prop_type, "indoor_los")
        
        # Create measurement edge
        edge = MeasurementEdge(
            node_i=3,  # Unknown node
            node_j=anchor_id,
            distance=meas['measured_distance_m'],
            quality=meas['quality_score'],
            variance=meas['measurement_std_m']**2
        )
        measurements.append(edge)
        
        error = meas['measured_distance_m'] - true_dist
        print(f"  3↔{anchor_id}: True={true_dist:.2f}m, Meas={meas['measured_distance_m']:.2f}m, "
              f"Err={error:+.2f}m, {prop_type.value}, Q={meas['quality_score']:.2f}")
    
    # Also add inter-anchor measurements (they know each other's positions but helps with validation)
    for i in range(3):
        for j in range(i+1, 3):
            true_dist = np.linalg.norm(anchors[i] - anchors[j])
            meas = channel.generate_measurement(true_dist, PropagationType.LOS, "indoor_los")
            edge = MeasurementEdge(
                node_i=i,
                node_j=j,
                distance=meas['measured_distance_m'],
                quality=meas['quality_score'],
                variance=meas['measurement_std_m']**2
            )
            measurements.append(edge)
    
    # Initialize solver
    print("\nRunning Robust Localization:")
    solver = RobustLocalizer(dimension=2, huber_delta=1.0)
    
    # Initial guess (random near center)
    initial_pos = np.array([5.0, 5.0]) + np.random.randn(2) * 2
    print(f"  Initial guess: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")
    
    # Solve
    optimized_pos, info = solver.solve(initial_pos, measurements, anchors)
    
    # Extract final position
    final_position = optimized_pos  # Already a 2D position vector
    
    # Calculate error
    final_error = np.linalg.norm(final_position - true_position)
    
    print(f"\nResults:")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Converged: {info['converged']}")
    print(f"  Final cost: {info['final_cost']:.4f}")
    print(f"\n  True position:      ({true_position[0]:.2f}, {true_position[1]:.2f})")
    print(f"  Estimated position: ({final_position[0]:.2f}, {final_position[1]:.2f})")
    print(f"  Final error: {final_error:.2f}m")
    
    # Show convergence
    if len(info['convergence_history']) > 0:
        print(f"\n  Convergence (first 10 iterations):")
        for i, cost in enumerate(info['convergence_history'][:10]):
            print(f"    Iter {i+1}: Cost = {cost:.4f}")
    
    # Success criteria
    print("\n" + "="*60)
    if final_error < 1.0:
        print("✅ SUCCESS: Sub-meter accuracy achieved!")
    elif final_error < 2.0:
        print("⚠️  PARTIAL: Meter-level accuracy achieved")
    else:
        print("❌ NEEDS TUNING: Error > 2m")
    
    print("="*60)
    
    return final_error

if __name__ == "__main__":
    np.random.seed(42)
    error = run_minimal_demo()
    
    # Run multiple trials to show consistency
    print("\nRunning 5 more trials to test consistency:")
    errors = []
    for i in range(5):
        np.random.seed(100 + i)
        error = run_minimal_demo()
        errors.append(error)
    
    print(f"\nAverage error over 6 trials: {np.mean([error] + errors):.2f}m")
    print(f"Std deviation: {np.std([error] + errors):.2f}m")