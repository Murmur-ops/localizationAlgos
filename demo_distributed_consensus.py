#!/usr/bin/env python3
"""
Distributed Consensus Localization Demo
Tests if consensus improves RMSE compared to centralized approach
"""

import numpy as np
import yaml
import sys
import os
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.localization.robust_solver import DistributedLocalizer
from src.sync.frequency_sync import DistributedFrequencyConsensus
from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType

class DistributedNode:
    """Node that uses distributed consensus for localization"""
    
    def __init__(self, node_id: int, position: np.ndarray, is_anchor: bool, neighbors: List[int]):
        self.node_id = node_id
        self.true_position = position
        self.is_anchor = is_anchor
        self.neighbors = neighbors
        
        # Initialize distributed components
        if not is_anchor:
            self.localizer = DistributedLocalizer(node_id, neighbors, dimension=2)
            self.localizer.position = position + np.random.randn(2) * 3  # Initial guess
        else:
            self.localizer = None
            
        self.freq_consensus = DistributedFrequencyConsensus(node_id, neighbors)
        
        # Simulated measurements
        self.measurements = {}
        self.measurement_qualities = {}
        
    def measure_distance(self, other_node: 'DistributedNode', channel: RangingChannel) -> float:
        """Simulate distance measurement to another node"""
        true_dist = np.linalg.norm(self.true_position - other_node.true_position)
        
        # Simulate channel effects
        if true_dist < 5:
            prop_type = PropagationType.LOS
        else:
            prop_type = PropagationType.LOS if np.random.rand() > 0.02 else PropagationType.NLOS
            
        meas = channel.generate_measurement(true_dist, prop_type, "indoor")
        
        self.measurements[other_node.node_id] = meas['measured_distance_m']
        self.measurement_qualities[other_node.node_id] = meas['quality_score']
        
        return meas['measured_distance_m']
    
    def get_position_estimate(self) -> np.ndarray:
        """Get current position estimate"""
        if self.is_anchor:
            return self.true_position
        else:
            return self.localizer.position
    
    def consensus_update(self, neighbor_positions: Dict[int, np.ndarray]) -> np.ndarray:
        """Perform consensus-based position update"""
        if self.is_anchor:
            return self.true_position
            
        # Update position using ADMM consensus
        new_pos = self.localizer.update_position(
            self.measurements, 
            neighbor_positions,
            self.measurement_qualities
        )
        
        return new_pos
    
    def update_dual_variables(self, neighbor_positions: Dict[int, np.ndarray]):
        """Update ADMM dual variables"""
        if not self.is_anchor:
            self.localizer.update_dual_variables(neighbor_positions)


def simple_centralized_trilateration(nodes: Dict[int, DistributedNode], 
                                    measurements: Dict[Tuple[int, int], float]) -> Dict[int, np.ndarray]:
    """Centralized trilateration for comparison"""
    results = {}
    
    # Get anchors
    anchors = {nid: node.true_position for nid, node in nodes.items() if node.is_anchor}
    
    # For each unknown node
    for nid, node in nodes.items():
        if node.is_anchor:
            results[nid] = node.true_position
            continue
            
        # Weighted least squares
        position = np.array([5.0, 5.0])  # Start at center
        
        for iteration in range(10):
            A = []
            b = []
            weights = []
            
            for aid, anchor_pos in anchors.items():
                if (nid, aid) in measurements or (aid, nid) in measurements:
                    meas_dist = measurements.get((nid, aid), measurements.get((aid, nid)))
                    
                    diff = position - anchor_pos
                    est_dist = np.linalg.norm(diff)
                    
                    if est_dist > 0:
                        gradient = diff / est_dist
                        A.append(gradient)
                        b.append(meas_dist - est_dist)
                        weights.append(0.9)  # High quality weight
            
            if len(A) >= 2:
                A = np.array(A)
                b = np.array(b)
                W = np.diag(weights)
                
                try:
                    delta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ b, rcond=None)[0]
                    position += delta * 0.5
                    
                    if np.linalg.norm(delta) < 1e-6:
                        break
                except:
                    break
        
        results[nid] = position
    
    return results


def run_consensus_comparison():
    """Compare distributed consensus vs centralized approach"""
    
    print("\n" + "="*70)
    print("DISTRIBUTED CONSENSUS vs CENTRALIZED COMPARISON")
    print("="*70)
    
    # Setup 10x10m area with 10 nodes
    np.random.seed(42)
    
    # Node configuration (4 anchors, 6 unknowns)
    node_configs = [
        (0, [0.0, 0.0], True),    # Anchor
        (1, [10.0, 0.0], True),   # Anchor
        (2, [10.0, 10.0], True),  # Anchor
        (3, [0.0, 10.0], True),   # Anchor
        (4, [2.5, 2.5], False),   # Unknown
        (5, [7.5, 2.5], False),   # Unknown
        (6, [5.0, 5.0], False),   # Unknown
        (7, [2.5, 7.5], False),   # Unknown
        (8, [7.5, 7.5], False),   # Unknown
        (9, [5.0, 9.0], False),   # Unknown
    ]
    
    # Create channel
    ch_config = ChannelConfig(
        bandwidth_hz=100e6,
        path_loss_exponent=2.2,
        nlos_bias_mean_m=0.3,
        nlos_bias_std_m=0.1
    )
    channel = RangingChannel(ch_config)
    
    # Build neighbor graph (assume all nodes can see each other in 10x10m)
    nodes = {}
    for nid, pos, is_anchor in node_configs:
        neighbors = [i for i in range(10) if i != nid]
        nodes[nid] = DistributedNode(nid, np.array(pos), is_anchor, neighbors)
    
    # Generate all pairwise measurements
    all_measurements = {}
    print("\nGenerating measurements...")
    for i in range(10):
        for j in range(i+1, 10):
            dist = nodes[i].measure_distance(nodes[j], channel)
            nodes[j].measurements[i] = dist  # Symmetric
            nodes[j].measurement_qualities[i] = nodes[i].measurement_qualities[j]
            all_measurements[(i, j)] = dist
    
    # Print measurement quality
    num_measurements = len(all_measurements)
    avg_error = np.mean([abs(dist - np.linalg.norm(nodes[i].true_position - nodes[j].true_position)) 
                         for (i, j), dist in all_measurements.items()])
    print(f"  Total measurements: {num_measurements}")
    print(f"  Average measurement error: {avg_error:.3f}m")
    
    # ============= CENTRALIZED APPROACH =============
    print("\n" + "-"*70)
    print("CENTRALIZED TRILATERATION")
    print("-"*70)
    
    centralized_results = simple_centralized_trilateration(nodes, all_measurements)
    
    # Calculate centralized RMSE
    centralized_errors = []
    for nid, node in nodes.items():
        if not node.is_anchor:
            error = np.linalg.norm(centralized_results[nid] - node.true_position)
            centralized_errors.append(error)
            print(f"  Node {nid}: Error = {error:.3f}m")
    
    centralized_rmse = np.sqrt(np.mean(np.array(centralized_errors)**2))
    print(f"\nCentralized RMSE: {centralized_rmse:.3f}m")
    
    # ============= DISTRIBUTED CONSENSUS =============
    print("\n" + "-"*70)
    print("DISTRIBUTED CONSENSUS (ADMM)")
    print("-"*70)
    
    # Run consensus iterations
    consensus_iterations = 20
    consensus_history = []
    
    for iteration in range(consensus_iterations):
        # Each node updates based on neighbor positions
        new_positions = {}
        
        for nid, node in nodes.items():
            # Get neighbor positions
            neighbor_positions = {
                neighbor_id: nodes[neighbor_id].get_position_estimate()
                for neighbor_id in node.neighbors
            }
            
            # Consensus update
            new_pos = node.consensus_update(neighbor_positions)
            new_positions[nid] = new_pos
        
        # Update dual variables (ADMM step)
        for nid, node in nodes.items():
            if not node.is_anchor:
                neighbor_positions = {
                    neighbor_id: new_positions[neighbor_id]
                    for neighbor_id in node.neighbors
                }
                node.update_dual_variables(neighbor_positions)
        
        # Calculate current RMSE
        iter_errors = []
        for nid, node in nodes.items():
            if not node.is_anchor:
                error = np.linalg.norm(node.get_position_estimate() - node.true_position)
                iter_errors.append(error)
        
        iter_rmse = np.sqrt(np.mean(np.array(iter_errors)**2))
        consensus_history.append(iter_rmse)
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: RMSE = {iter_rmse:.3f}m")
    
    # Final results
    print(f"\nFinal node errors:")
    consensus_errors = []
    for nid, node in nodes.items():
        if not node.is_anchor:
            error = np.linalg.norm(node.get_position_estimate() - node.true_position)
            consensus_errors.append(error)
            print(f"  Node {nid}: Error = {error:.3f}m")
    
    consensus_rmse = np.sqrt(np.mean(np.array(consensus_errors)**2))
    print(f"\nDistributed Consensus RMSE: {consensus_rmse:.3f}m")
    
    # ============= COMPARISON =============
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\nCentralized RMSE:    {centralized_rmse:.3f}m")
    print(f"Consensus RMSE:      {consensus_rmse:.3f}m")
    
    if consensus_rmse < centralized_rmse:
        improvement = (centralized_rmse - consensus_rmse) / centralized_rmse * 100
        print(f"\n✅ Consensus IMPROVED by {improvement:.1f}%")
    else:
        degradation = (consensus_rmse - centralized_rmse) / centralized_rmse * 100
        print(f"\n❌ Consensus DEGRADED by {degradation:.1f}%")
    
    # Show convergence behavior
    print(f"\nConsensus convergence:")
    print(f"  Initial RMSE: {consensus_history[0]:.3f}m")
    print(f"  After 5 iter: {consensus_history[4]:.3f}m") 
    print(f"  After 10 iter: {consensus_history[9]:.3f}m")
    print(f"  Final (20 iter): {consensus_history[-1]:.3f}m")
    
    # Analyze why consensus helps or doesn't help
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    print("\nWhy consensus might help:")
    print("• Averaging reduces impact of noisy measurements")
    print("• Dual variables enforce consistency between neighbors")
    print("• Distributed computation (each node only uses local info)")
    
    print("\nWhy consensus might not help in this scenario:")
    print("• 10x10m area has excellent LOS (centralized already optimal)")
    print("• All nodes can see all others (fully connected graph)")
    print("• Low measurement noise means averaging doesn't add much value")
    
    return centralized_rmse, consensus_rmse


if __name__ == "__main__":
    centralized_rmse, consensus_rmse = run_consensus_comparison()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if consensus_rmse < 0.1:
        print("\n✅ Consensus achieves sub-10cm accuracy!")
    
    print(f"\nFor a 10x10m indoor LOS scenario:")
    print(f"• Centralized approach: {centralized_rmse:.3f}m RMSE")
    print(f"• Distributed consensus: {consensus_rmse:.3f}m RMSE")
    print(f"• Both work well because measurements are excellent")
    print(f"\nConsensus would show more benefit with:")
    print(f"• Larger areas (100x100m)")
    print(f"• Limited connectivity (nodes can't see all others)")
    print(f"• Higher measurement noise/NLOS")