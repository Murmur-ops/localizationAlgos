#!/usr/bin/env python3
"""
Simple trilateration test to verify solver works correctly
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_trilateration(anchors, distances):
    """
    Simple least-squares trilateration
    """
    n_anchors = len(anchors)
    anchor_positions = np.array(list(anchors.values()))
    
    # Build linear system: Ax = b
    # Using linearization around a rough initial guess
    x_init = np.mean(anchor_positions, axis=0)
    
    max_iterations = 10
    position = x_init.copy()
    
    for iteration in range(max_iterations):
        A = []
        b = []
        
        for i, (aid, anchor_pos) in enumerate(anchors.items()):
            # Current estimated distance
            diff = position - anchor_pos
            est_dist = np.linalg.norm(diff)
            
            if est_dist > 0:
                # Gradient
                gradient = diff / est_dist
                
                # Add to system
                A.append(gradient)
                b.append(distances[aid] - est_dist)
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve least squares
        try:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
            position += delta * 0.5  # Damping
            
            # Check convergence
            if np.linalg.norm(delta) < 1e-6:
                break
        except:
            break
    
    return position

def test_trilateration():
    """Test simple trilateration with perfect measurements"""
    
    print("\n" + "="*60)
    print("SIMPLE TRILATERATION TEST")
    print("="*60)
    
    # Define anchors (4 corners of 10x10)
    anchors = {
        0: np.array([0.0, 0.0]),
        1: np.array([10.0, 0.0]),
        2: np.array([10.0, 10.0]),
        3: np.array([0.0, 10.0])
    }
    
    # True position
    true_pos = np.array([5.0, 3.0])
    
    # Perfect distance measurements
    distances = {}
    for aid, anchor_pos in anchors.items():
        distances[aid] = np.linalg.norm(true_pos - anchor_pos)
    
    print("\nSetup:")
    print(f"  True position: ({true_pos[0]:.1f}, {true_pos[1]:.1f})")
    print(f"  Anchors: 4 corners of 10x10m square")
    print(f"\nTrue distances:")
    for aid, dist in distances.items():
        print(f"  Anchor {aid}: {dist:.2f}m")
    
    # Add small noise
    noisy_distances = {}
    np.random.seed(42)
    for aid, dist in distances.items():
        noisy_distances[aid] = dist + np.random.normal(0, 0.05)  # 5cm std
    
    print(f"\nNoisy distances (5cm std):")
    for aid, dist in noisy_distances.items():
        error = dist - distances[aid]
        print(f"  Anchor {aid}: {dist:.2f}m (error: {error:+.3f}m)")
    
    # Solve
    estimated_pos = simple_trilateration(anchors, noisy_distances)
    error = np.linalg.norm(estimated_pos - true_pos)
    
    print(f"\nResults:")
    print(f"  Estimated position: ({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f})")
    print(f"  True position:      ({true_pos[0]:.2f}, {true_pos[1]:.2f})")
    print(f"  Error: {error:.3f}m")
    
    if error < 0.1:
        print("\n✅ SUCCESS: Sub-10cm accuracy achieved!")
    else:
        print(f"\n⚠️  Error is {error:.3f}m (expected <0.1m)")
    
    # Now test with more realistic multipath scenario
    print("\n" + "-"*60)
    print("TEST WITH MULTIPATH (one measurement has positive bias)")
    
    biased_distances = noisy_distances.copy()
    biased_distances[2] += 0.5  # Add 50cm positive bias to one measurement
    
    print(f"\nBiased distances:")
    for aid, dist in biased_distances.items():
        error = dist - distances[aid]
        print(f"  Anchor {aid}: {dist:.2f}m (error: {error:+.3f}m)")
    
    estimated_pos_biased = simple_trilateration(anchors, biased_distances)
    error_biased = np.linalg.norm(estimated_pos_biased - true_pos)
    
    print(f"\nResults with bias:")
    print(f"  Estimated position: ({estimated_pos_biased[0]:.2f}, {estimated_pos_biased[1]:.2f})")
    print(f"  True position:      ({true_pos[0]:.2f}, {true_pos[1]:.2f})")
    print(f"  Error: {error_biased:.3f}m")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"\nWith good measurements: {error:.3f}m error")
    print(f"With one biased measurement: {error_biased:.3f}m error")
    print(f"Degradation factor: {error_biased/error:.1f}x")
    print("\nThis shows why robust optimization (Huber loss) is important!")

if __name__ == "__main__":
    test_trilateration()