"""
Stable MPS Algorithm Implementation
Simplified version focusing on convergence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations


@dataclass  
class MPSConfig:
    """Configuration for MPS algorithm"""
    n_sensors: int = 30
    n_anchors: int = 6
    scale: float = 1.0  # Network scale in meters  
    communication_range: float = 0.7  # As fraction of scale
    noise_factor: float = 0.05  # 5% noise
    gamma: float = 0.999  # Consensus mixing parameter
    alpha: float = 10.0   # Proximal step size
    max_iterations: int = 1000
    tolerance: float = 1e-5
    dimension: int = 2
    seed: Optional[int] = 42
    verbose: bool = False


class StableMPSAlgorithm:
    """
    Stable MPS implementation with simplified structure
    """
    
    def __init__(self, config: MPSConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Network data
        self.true_positions = None
        self.anchor_positions = None
        self.distance_measurements = {}
        self.anchor_distances = {}
        self.adjacency = None
        self.consensus_matrix = None
        
    def generate_network(self):
        """Generate synthetic network"""
        n = self.config.n_sensors
        d = self.config.dimension
        scale = self.config.scale
        
        # Generate random positions
        self.true_positions = {}
        for i in range(n):
            self.true_positions[i] = np.random.uniform(0, scale, d)
        
        # Generate anchor positions (corners + random)
        if self.config.n_anchors > 0:
            self.anchor_positions = np.zeros((self.config.n_anchors, d))
            if d == 2 and self.config.n_anchors >= 4:
                # Corners
                corners = np.array([
                    [0.1, 0.1], [0.9, 0.1], 
                    [0.9, 0.9], [0.1, 0.9]
                ]) * scale
                self.anchor_positions[:min(4, self.config.n_anchors)] = corners[:min(4, self.config.n_anchors)]
                # Additional random anchors
                for i in range(4, self.config.n_anchors):
                    self.anchor_positions[i] = np.random.uniform(0.2*scale, 0.8*scale, d)
            else:
                self.anchor_positions = np.random.uniform(0, scale, (self.config.n_anchors, d))
        
        # Build adjacency
        comm_range = self.config.communication_range * scale
        self.adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                if dist <= comm_range:
                    self.adjacency[i, j] = 1
                    self.adjacency[j, i] = 1
        
        # Generate measurements
        self._generate_measurements()
        
        # Create consensus matrix
        self._create_consensus_matrix()
    
    def _generate_measurements(self):
        """Generate noisy distance measurements"""
        n = self.config.n_sensors
        noise = self.config.noise_factor
        
        # Sensor-to-sensor
        for i in range(n):
            for j in range(i+1, n):
                if self.adjacency[i, j] > 0:
                    true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.distance_measurements[(i, j)] = max(0.01, noisy_dist)
                    self.distance_measurements[(j, i)] = self.distance_measurements[(i, j)]
        
        # Sensor-to-anchor
        for i in range(n):
            self.anchor_distances[i] = {}
            for k in range(self.config.n_anchors):
                true_dist = np.linalg.norm(self.true_positions[i] - self.anchor_positions[k])
                if true_dist <= self.config.communication_range * self.config.scale:
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.anchor_distances[i][k] = max(0.01, noisy_dist)
    
    def _create_consensus_matrix(self):
        """Create consensus averaging matrix"""
        n = self.config.n_sensors
        
        # Start with adjacency + self-loops
        W = self.adjacency + np.eye(n)
        
        # Make doubly stochastic
        W = MatrixOperations.sinkhorn_knopp(W)
        
        self.consensus_matrix = W
    
    def initialize_positions(self) -> Dict[int, np.ndarray]:
        """Initialize positions near anchors"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        positions = {}
        for i in range(n):
            if i in self.anchor_distances and len(self.anchor_distances[i]) > 0:
                # Initialize near anchors
                anchor_ids = list(self.anchor_distances[i].keys())
                positions[i] = np.mean(self.anchor_positions[anchor_ids], axis=0)
                positions[i] += 0.01 * self.config.scale * np.random.randn(d)
            else:
                # Random initialization
                positions[i] = np.random.uniform(0, self.config.scale, d)
        
        return positions
    
    def apply_distance_constraints(self, positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Apply proximal operator for distance constraints"""
        n = self.config.n_sensors
        new_positions = {}
        
        for i in range(n):
            pos = positions[i].copy()
            
            # Sensor-to-sensor constraints
            n_constraints = 0
            for j in range(n):
                if (i, j) in self.distance_measurements:
                    measured_dist = self.distance_measurements[(i, j)]
                    n_constraints += 1
                    
                    # Adaptive weight
                    weight = self.config.alpha / max(1, n_constraints)
                    
                    pos = ProximalOperators.prox_distance(
                        pos, positions[j], measured_dist, alpha=weight
                    )
            
            # Anchor constraints (stronger weight)
            if i in self.anchor_distances:
                for k, measured_dist in self.anchor_distances[i].items():
                    weight = self.config.alpha * 2 / max(1, len(self.anchor_distances[i]))
                    pos = ProximalOperators.prox_distance(
                        pos, self.anchor_positions[k], measured_dist, alpha=weight
                    )
            
            # Box constraint
            pos = ProximalOperators.prox_box_constraint(
                pos, -0.1 * self.config.scale, 1.1 * self.config.scale
            )
            
            new_positions[i] = pos
        
        return new_positions
    
    def apply_consensus(self, positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Apply consensus averaging"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Convert to matrix form
        X = np.array([positions[i] for i in range(n)])
        
        # Apply consensus matrix
        X_new = self.consensus_matrix @ X
        
        # Convert back to dictionary
        new_positions = {}
        for i in range(n):
            new_positions[i] = X_new[i]
        
        return new_positions
    
    def compute_relative_error(self, positions: Dict[int, np.ndarray]) -> float:
        """Compute relative error"""
        if self.true_positions is None:
            return 0.0
        
        n = self.config.n_sensors
        X_hat = np.array([positions[i] for i in range(n)])
        X_0 = np.array([self.true_positions[i] for i in range(n)])
        
        return np.linalg.norm(X_hat - X_0, 'fro') / (np.linalg.norm(X_0, 'fro') + 1e-10)
    
    def run(self) -> Dict:
        """
        Run simplified MPS algorithm
        Alternates between:
        1. Applying distance constraints (proximal step)
        2. Applying consensus (averaging step)
        """
        # Initialize
        positions = self.initialize_positions()
        
        # Track metrics
        relative_errors = []
        iterations_tracked = []
        
        # Adaptive parameters
        gamma = self.config.gamma
        best_error = float('inf')
        patience = 0
        max_patience = 50
        
        # Main loop
        for k in range(self.config.max_iterations):
            # Store old positions
            old_positions = {i: pos.copy() for i, pos in positions.items()}
            
            # Step 1: Apply distance constraints (proximal step)
            positions_prox = self.apply_distance_constraints(positions)
            
            # Step 2: Mix with consensus
            positions_consensus = self.apply_consensus(positions)
            
            # Step 3: Combine with mixing parameter gamma
            for i in range(self.config.n_sensors):
                positions[i] = gamma * positions_prox[i] + (1 - gamma) * positions_consensus[i]
            
            # Track metrics
            if k % 10 == 0:
                rel_error = self.compute_relative_error(positions)
                relative_errors.append(rel_error)
                iterations_tracked.append(k)
                
                # Adaptive gamma based on progress
                if rel_error < best_error:
                    best_error = rel_error
                    patience = 0
                else:
                    patience += 1
                    if patience > max_patience:
                        gamma *= 0.95  # Reduce step size
                        patience = 0
                
                if self.config.verbose and k % 100 == 0:
                    print(f"  Iter {k}: relative_error={rel_error:.4f}, γ={gamma:.4f}")
            
            # Check convergence
            max_change = max([np.linalg.norm(positions[i] - old_positions[i]) 
                             for i in range(self.config.n_sensors)])
            if max_change < self.config.tolerance:
                if self.config.verbose:
                    print(f"  Converged at iteration {k}")
                break
        
        # Final metrics
        final_rel_error = self.compute_relative_error(positions)
        
        return {
            'converged': k < self.config.max_iterations - 1,
            'iterations': k,
            'final_relative_error': final_rel_error,
            'relative_errors': relative_errors,
            'iterations_tracked': iterations_tracked,
            'final_positions': dict(positions),
            'true_positions': dict(self.true_positions) if self.true_positions else {}
        }


def test_stable_algorithm():
    """Test the stable MPS algorithm"""
    print("Testing Stable MPS Algorithm")
    print("="*50)
    
    # Test with multiple runs to check consistency
    errors = []
    for run in range(5):
        config = MPSConfig(
            n_sensors=30,
            n_anchors=6,
            verbose=(run == 0),  # Only verbose on first run
            seed=42 + run
        )
        
        algo = StableMPSAlgorithm(config)
        algo.generate_network()
        
        if run == 0:
            print(f"Network: {config.n_sensors} sensors, {config.n_anchors} anchors")
            print(f"Parameters: γ={config.gamma}, α={config.alpha}")
            print("\nRunning algorithm...")
        
        results = algo.run()
        errors.append(results['final_relative_error'])
        
        if run == 0:
            print(f"\nRun {run+1} Results:")
            print(f"  Converged: {results['converged']} at iteration {results['iterations']}")
            print(f"  Final relative error: {results['final_relative_error']:.4f}")
    
    # Summary
    print(f"\nSummary over 5 runs:")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std error: {np.std(errors):.4f}")
    print(f"  Min error: {np.min(errors):.4f}")
    print(f"  Max error: {np.max(errors):.4f}")
    
    if np.mean(errors) < 0.15:
        print("  ✓ Achieving expected performance range!")
    else:
        print("  ⚠ Performance needs further tuning")
    
    return errors


if __name__ == "__main__":
    test_stable_algorithm()