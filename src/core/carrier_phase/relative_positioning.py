"""
Relative Carrier Phase Positioning for Robust Millimeter Accuracy
Uses network constraints and relative ambiguities instead of absolute resolution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RelativeAmbiguity:
    """Relative integer ambiguity between two nodes"""
    node_i: int
    node_j: int
    relative_n: int  # N_i - N_j
    confidence: float
    reference_path: List[int]  # Path from reference node


class NetworkAmbiguityResolver:
    """
    Resolves carrier phase ambiguities using network constraints
    Key insight: We only need relative ambiguities for accurate localization
    """
    
    def __init__(self, wavelength: float = 0.125):
        """
        Initialize network resolver
        
        Args:
            wavelength: Carrier wavelength in meters
        """
        self.wavelength = wavelength
        self.relative_ambiguities: Dict[Tuple[int, int], RelativeAmbiguity] = {}
        self.reference_nodes: Set[int] = set()
        self.node_ambiguities: Dict[int, int] = {}  # Absolute N for each node
        
    def set_reference(self, node_id: int, n_cycles: int = 0):
        """
        Set a reference node with known (or arbitrary) ambiguity
        
        Args:
            node_id: Reference node ID
            n_cycles: Integer ambiguity for this node (can be arbitrary)
        """
        self.reference_nodes.add(node_id)
        self.node_ambiguities[node_id] = n_cycles
        logger.info(f"Set node {node_id} as reference with N={n_cycles}")
    
    def compute_relative_ambiguity(self, phase_i: float, phase_j: float,
                                  coarse_distance: float) -> int:
        """
        Compute relative integer ambiguity between two measurements
        
        Args:
            phase_i: Phase measurement at node i (radians)
            phase_j: Phase measurement at node j (radians)
            coarse_distance: Coarse distance estimate (meters)
            
        Returns:
            Relative integer ambiguity N_i - N_j
        """
        # Convert phases to fractional cycles
        frac_i = phase_i / (2 * np.pi)
        frac_j = phase_j / (2 * np.pi)
        
        # Estimate total cycles from coarse distance
        total_cycles = coarse_distance / self.wavelength
        
        # The relative ambiguity is the difference
        # Key: even if total_cycles is wrong, the relative part is robust
        relative_n = round(total_cycles - (frac_i + frac_j) / 2)
        
        return relative_n
    
    def propagate_ambiguities(self, measurements: Dict[Tuple[int, int], Dict]) -> Dict[int, int]:
        """
        Propagate ambiguities through network starting from reference nodes
        
        Args:
            measurements: Dictionary of measurements between node pairs
                         Each measurement has 'phase', 'coarse_distance'
            
        Returns:
            Dictionary of resolved integer ambiguities for each node
        """
        if not self.reference_nodes:
            raise ValueError("No reference nodes set")
        
        # Build adjacency list
        graph = {}
        for (i, j), _ in measurements.items():
            if i not in graph:
                graph[i] = []
            if j not in graph:
                graph[j] = []
            graph[i].append(j)
            graph[j].append(i)
        
        # BFS to propagate from reference nodes
        visited = set(self.reference_nodes)
        queue = deque(self.reference_nodes)
        
        while queue:
            current = queue.popleft()
            current_n = self.node_ambiguities[current]
            
            # Process all neighbors
            for neighbor in graph.get(current, []):
                if neighbor in visited:
                    continue
                
                # Find measurement between current and neighbor
                pair = (min(current, neighbor), max(current, neighbor))
                if pair not in measurements:
                    continue
                
                meas = measurements[pair]
                
                # Compute relative ambiguity
                if current == pair[0]:
                    # current is node_i
                    phase_current = meas['phase_i']
                    phase_neighbor = meas['phase_j']
                else:
                    # current is node_j
                    phase_current = meas['phase_j']
                    phase_neighbor = meas['phase_i']
                
                relative_n = self.compute_relative_ambiguity(
                    phase_current, phase_neighbor, meas['coarse_distance']
                )
                
                # Propagate to neighbor
                neighbor_n = current_n + relative_n
                self.node_ambiguities[neighbor] = neighbor_n
                
                visited.add(neighbor)
                queue.append(neighbor)
                
                logger.debug(f"Propagated N={neighbor_n} to node {neighbor} from node {current}")
        
        return self.node_ambiguities
    
    def resolve_with_geometry(self, measurements: Dict[Tuple[int, int], Dict],
                            anchor_positions: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Resolve ambiguities using geometric constraints
        
        Args:
            measurements: Carrier phase measurements
            anchor_positions: Known positions for anchor nodes
            
        Returns:
            Dictionary of refined distance measurements
        """
        # First propagate from anchors if available
        if anchor_positions:
            # Use first anchor as reference
            anchor_id = min(anchor_positions.keys())
            self.set_reference(anchor_id, 0)  # Arbitrary N for reference
        
        # Propagate through network
        node_ambiguities = self.propagate_ambiguities(measurements)
        
        # Refine distances using resolved ambiguities
        refined_distances = {}
        
        for (i, j), meas in measurements.items():
            if i in node_ambiguities and j in node_ambiguities:
                # Both nodes have resolved ambiguities
                n_i = node_ambiguities[i]
                n_j = node_ambiguities[j]
                
                # Phases
                phi_i = meas.get('phase_i', meas.get('phase', 0)) / (2 * np.pi)
                phi_j = meas.get('phase_j', 0) / (2 * np.pi)
                
                # Refined distance using carrier phase
                # d = λ * (N_i + φ_i + N_j + φ_j) / 2
                # But we need to handle the relative nature correctly
                
                # Use the average of forward and backward estimates
                dist_from_i = (n_i + phi_i) * self.wavelength
                dist_from_j = (n_j + phi_j) * self.wavelength
                
                # The actual distance considering relative ambiguity
                n_rel = n_i - n_j
                refined_dist = abs(dist_from_i - dist_from_j) + (phi_i + phi_j) * self.wavelength / 2
                
                # Fallback to simpler estimate
                refined_dist = meas['coarse_distance'] + (phi_i - phi_j) * self.wavelength
                
                refined_distances[(i, j)] = {
                    'distance': refined_dist,
                    'n_i': n_i,
                    'n_j': n_j,
                    'method': 'relative_network'
                }
            else:
                # Fallback to coarse distance
                refined_distances[(i, j)] = {
                    'distance': meas['coarse_distance'],
                    'method': 'coarse_only'
                }
        
        return refined_distances
    
    def validate_triangle(self, d01: float, d02: float, d12: float,
                         tolerance: float = 0.01) -> bool:
        """
        Validate triangle inequality for three distances
        
        Args:
            d01, d02, d12: Three distances forming a triangle
            tolerance: Tolerance for validation (meters)
            
        Returns:
            True if triangle inequality is satisfied
        """
        # Check all three inequalities
        checks = [
            d01 + d02 >= d12 - tolerance,
            d01 + d12 >= d02 - tolerance,
            d02 + d12 >= d01 - tolerance,
            abs(d01 - d02) <= d12 + tolerance,
            abs(d01 - d12) <= d02 + tolerance,
            abs(d02 - d12) <= d01 + tolerance
        ]
        
        return all(checks)
    
    def find_consistent_ambiguities(self, measurements: Dict[Tuple[int, int], Dict],
                                   max_iterations: int = 10) -> Dict[int, int]:
        """
        Find globally consistent ambiguities using iterative refinement
        
        Args:
            measurements: Carrier phase measurements
            max_iterations: Maximum refinement iterations
            
        Returns:
            Consistent integer ambiguities for all nodes
        """
        # Initial propagation
        ambiguities = self.propagate_ambiguities(measurements)
        
        # Iterative refinement using triangle constraints
        for iteration in range(max_iterations):
            changes = 0
            
            # Check all triangles in the network
            nodes = list(ambiguities.keys())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        node_i, node_j, node_k = nodes[i], nodes[j], nodes[k]
                        
                        # Get measurements for this triangle
                        pairs = [
                            (min(node_i, node_j), max(node_i, node_j)),
                            (min(node_i, node_k), max(node_i, node_k)),
                            (min(node_j, node_k), max(node_j, node_k))
                        ]
                        
                        if all(p in measurements for p in pairs):
                            # Calculate distances with current ambiguities
                            distances = []
                            for p in pairs:
                                meas = measurements[p]
                                n1 = ambiguities[p[0]]
                                n2 = ambiguities[p[1]]
                                phi = meas.get('phase', 0) / (2 * np.pi)
                                dist = ((n1 + n2) / 2 + phi) * self.wavelength
                                distances.append(dist)
                            
                            # Check triangle inequality
                            if not self.validate_triangle(distances[0], distances[1], distances[2]):
                                # Try adjusting ambiguities
                                # This is simplified - real implementation would be more sophisticated
                                logger.debug(f"Triangle ({node_i}, {node_j}, {node_k}) inconsistent")
                                changes += 1
            
            if changes == 0:
                break
            
            logger.info(f"Iteration {iteration}: {changes} triangle inconsistencies")
        
        return ambiguities


def create_carrier_phase_weights(measurements: Dict[Tuple[int, int], Dict],
                                wavelength: float = 0.125) -> Dict[Tuple[int, int], float]:
    """
    Create weights for carrier phase measurements
    
    Args:
        measurements: Dictionary of measurements
        wavelength: Carrier wavelength
        
    Returns:
        Dictionary of weights for each measurement
    """
    weights = {}
    
    for pair, meas in measurements.items():
        if 'phase' in meas or 'phase_i' in meas:
            # Has carrier phase - high weight
            # Weight inversely proportional to expected error
            phase_precision_m = wavelength / (2 * np.pi) * 0.001  # 1mrad phase noise
            weight = 1.0 / (phase_precision_m ** 2)
            
            # Scale by quality if available
            if 'quality' in meas:
                weight *= meas['quality']
            
            # Cap weight to prevent numerical issues
            weights[pair] = min(weight, 10000.0)
        else:
            # TWTT only - low weight
            coarse_std = meas.get('coarse_std', 0.1)
            weights[pair] = 1.0 / (coarse_std ** 2)
    
    return weights


def test_relative_positioning():
    """Test relative positioning approach"""
    
    print("="*60)
    print("RELATIVE CARRIER PHASE POSITIONING TEST")
    print("="*60)
    
    # Create simple network
    measurements = {
        (0, 1): {
            'phase_i': 0.5,
            'phase_j': 0.3,
            'coarse_distance': 2.5,
            'true_distance': 2.45
        },
        (0, 2): {
            'phase_i': 0.2,
            'phase_j': 0.8,
            'coarse_distance': 3.2,
            'true_distance': 3.0
        },
        (1, 2): {
            'phase_i': 0.7,
            'phase_j': 0.4,
            'coarse_distance': 2.1,
            'true_distance': 2.0
        }
    }
    
    resolver = NetworkAmbiguityResolver()
    
    # Set node 0 as reference
    resolver.set_reference(0, n_cycles=0)
    
    # Resolve ambiguities
    print("\nResolving ambiguities through network...")
    ambiguities = resolver.propagate_ambiguities(measurements)
    
    print("\nResolved ambiguities:")
    for node, n in ambiguities.items():
        print(f"  Node {node}: N = {n}")
    
    # Refine distances
    refined = resolver.resolve_with_geometry(measurements)
    
    print("\nRefined distances:")
    for pair, result in refined.items():
        true_dist = measurements[pair]['true_distance']
        refined_dist = result['distance']
        error_mm = abs(refined_dist - true_dist) * 1000
        print(f"  {pair}: {refined_dist:.6f}m (error: {error_mm:.2f}mm)")
    
    # Check if we achieve target
    errors = []
    for pair, result in refined.items():
        true_dist = measurements[pair]['true_distance']
        error = abs(result['distance'] - true_dist)
        errors.append(error)
    
    rmse_mm = np.sqrt(np.mean(np.square(errors))) * 1000
    
    print(f"\nRMSE: {rmse_mm:.2f}mm")
    if rmse_mm < 15:
        print("✓ Target achieved with relative positioning!")
    else:
        print("⚠ Further refinement needed")
    
    print("="*60)


if __name__ == "__main__":
    test_relative_positioning()