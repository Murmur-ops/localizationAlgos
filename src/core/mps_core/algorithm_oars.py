"""
MPS Algorithm Implementation based on OARS framework
Proper implementation of Algorithm 1 from arxiv:2503.13403v1
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
    gamma: float = 0.999  # Step size for v update
    alpha: float = 10.0   # Proximal step size
    max_iterations: int = 1000
    tolerance: float = 1e-5
    dimension: int = 2
    seed: Optional[int] = 42
    verbose: bool = False


class OARSBasedMPSAlgorithm:
    """
    MPS implementation following OARS serial algorithm structure
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
        
        # Algorithm matrices
        self.Z_matrix = None  # Lower triangular implicit equation matrix
        self.W_matrix = None  # Update matrix
        self.n_blocks = 2  # Using 2-block structure
        
    def generate_network(self):
        """Generate synthetic network"""
        n = self.config.n_sensors
        d = self.config.dimension
        scale = self.config.scale
        
        # Generate random positions
        self.true_positions = {}
        for i in range(n):
            self.true_positions[i] = np.random.uniform(0, scale, d)
        
        # Generate anchor positions (well-distributed)
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
        
        # Create Z and W matrices
        self._create_algorithm_matrices()
    
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
    
    def _create_algorithm_matrices(self):
        """
        Create Z and W matrices following OARS structure
        Z is used for the implicit equation: y = v - Z*x
        W is used for the update: v^(k+1) = v^k - gamma*W*x^k
        """
        n = self.config.n_sensors
        p = self.n_blocks * n  # Total blocks (2n for 2-block structure)
        
        # Create weight matrix from adjacency
        weight_matrix = self.adjacency + np.eye(n)  # Add self-loops
        # Make doubly stochastic
        weight_matrix = MatrixOperations.sinkhorn_knopp(weight_matrix)
        
        # Create Z matrix (lower triangular structure for implicit equation)
        # Following OARS: Z[i,j] only for j < i (strictly lower triangular)
        self.Z_matrix = np.zeros((p, p))
        
        # For 2-block structure:
        # Block 1 (sensors 0 to n-1): consensus within block
        for i in range(n):
            for j in range(i):  # j < i (lower triangular)
                self.Z_matrix[i, j] = weight_matrix[i, j] * (1 - self.config.gamma)
        
        # Block 2 (sensors n to 2n-1): coupled with block 1
        for i in range(n, p):
            sensor_i = i - n
            # Coupling with first block
            for j in range(n):
                if j < sensor_i:  # Maintain lower triangular structure
                    self.Z_matrix[i, j] = weight_matrix[sensor_i, j] * (1 - self.config.gamma) * 0.5
            # Within second block
            for j in range(n, i):
                sensor_j = j - n
                if sensor_j < sensor_i:
                    self.Z_matrix[i, j] = weight_matrix[sensor_i, sensor_j] * (1 - self.config.gamma)
        
        # Create W matrix for updates
        self.W_matrix = np.zeros((p, p))
        
        # W should be such that consensus is achieved
        # Using block structure similar to Z but full (not just lower triangular)
        for i in range(n):
            for j in range(n):
                self.W_matrix[i, j] = weight_matrix[i, j]
                self.W_matrix[i+n, j+n] = weight_matrix[i, j]
            # Cross-block coupling
            self.W_matrix[i, i+n] = 0.5
            self.W_matrix[i+n, i] = 0.5
        
        # Normalize W
        self.W_matrix = self.W_matrix / np.max(np.sum(np.abs(self.W_matrix), axis=1))
    
    def create_resolvents(self):
        """
        Create resolvent operators for each block
        Following OARS structure where each block has its own resolvent
        """
        n = self.config.n_sensors
        resolvents = []
        
        # Create resolvent for each block
        for block in range(self.n_blocks):
            # Each block represents the sensors with different constraints
            block_resolvent = SensorBlockResolvent(
                block_id=block,
                n_sensors=n,
                distance_measurements=self.distance_measurements,
                anchor_distances=self.anchor_distances,
                anchor_positions=self.anchor_positions,
                scale=self.config.scale,
                alpha=self.config.alpha
            )
            resolvents.append(block_resolvent)
        
        return resolvents
    
    def run(self) -> Dict:
        """
        Run MPS algorithm following OARS serial algorithm structure
        """
        n = self.config.n_sensors
        p = self.n_blocks * n
        d = self.config.dimension
        
        # Initialize resolvents
        resolvents = self.create_resolvents()
        
        # Initialize variables
        all_x = [np.zeros((n, d)) for _ in range(self.n_blocks)]
        all_v = [np.zeros((n, d)) for _ in range(self.n_blocks)]
        
        # Initialize v with positions near anchors
        for i in range(n):
            if i in self.anchor_distances and len(self.anchor_distances[i]) > 0:
                anchor_ids = list(self.anchor_distances[i].keys())
                init_pos = np.mean(self.anchor_positions[anchor_ids], axis=0)
            else:
                init_pos = np.random.uniform(0, self.config.scale, d)
            
            # Set initial v for both blocks
            all_v[0][i] = init_pos
            all_v[1][i] = init_pos
        
        # Track metrics
        relative_errors = []
        iterations_tracked = []
        
        # Main OARS-style iteration loop
        for itr in range(self.config.max_iterations):
            # Store old x for convergence check
            old_x = [x.copy() for x in all_x]
            
            # Step 1: Sequential block updates (following OARS serial.py structure)
            for block_idx in range(self.n_blocks):
                # Compute y = v - sum(Z[i,j]*x[j] for j < i)
                # This is the implicit equation part
                y = all_v[block_idx].copy()
                
                # Apply Z matrix coupling (lower triangular)
                for i in range(n):
                    global_i = block_idx * n + i
                    for j in range(global_i):  # j < i (lower triangular)
                        if j < n:
                            # Coupling with block 0
                            y[i] -= self.Z_matrix[global_i, j] * all_x[0][j]
                        else:
                            # Coupling with block 1
                            y[i] -= self.Z_matrix[global_i, j] * all_x[1][j - n]
                
                # Apply proximal operator (resolvent)
                all_x[block_idx] = resolvents[block_idx].prox(y, self.config.alpha)
            
            # Step 2: Update v using W matrix (following OARS line 87-88)
            wx = [np.zeros((n, d)) for _ in range(self.n_blocks)]
            
            for block_idx in range(self.n_blocks):
                for i in range(n):
                    global_i = block_idx * n + i
                    # Compute W*x for this component
                    for j in range(p):
                        if j < n:
                            wx[block_idx][i] += self.W_matrix[global_i, j] * all_x[0][j]
                        else:
                            wx[block_idx][i] += self.W_matrix[global_i, j] * all_x[1][j - n]
                
                # Update v: v^(k+1) = v^k - gamma*W*x^k
                all_v[block_idx] = all_v[block_idx] - self.config.gamma * wx[block_idx]
            
            # Extract positions (average across blocks)
            positions = {}
            for i in range(n):
                positions[i] = (all_x[0][i] + all_x[1][i]) / 2
            
            # Track metrics
            if itr % 10 == 0:
                rel_error = self.compute_relative_error(positions)
                relative_errors.append(rel_error)
                iterations_tracked.append(itr)
                
                if self.config.verbose and itr % 100 == 0:
                    delta_v = self.config.gamma * np.linalg.norm([np.linalg.norm(wx[b]) for b in range(self.n_blocks)])
                    xbar = sum([all_x[b].sum(axis=0) for b in range(self.n_blocks)]) / p
                    sum_diff = sum([np.linalg.norm(all_x[b][i] - xbar) for b in range(self.n_blocks) for i in range(n)])
                    print(f"  Iter {itr}: rel_error={rel_error:.4f}, delta_v={delta_v:.4f}, sum_diff={sum_diff:.4f}")
            
            # Check convergence
            max_change = max([np.linalg.norm(all_x[b] - old_x[b]) for b in range(self.n_blocks)])
            if max_change < self.config.tolerance:
                if self.config.verbose:
                    print(f"  Converged at iteration {itr}")
                break
        
        # Final metrics
        final_rel_error = self.compute_relative_error(positions)
        
        return {
            'converged': itr < self.config.max_iterations - 1,
            'iterations': itr,
            'final_relative_error': final_rel_error,
            'relative_errors': relative_errors,
            'iterations_tracked': iterations_tracked,
            'final_positions': dict(positions),
            'true_positions': dict(self.true_positions) if self.true_positions else {}
        }
    
    def compute_relative_error(self, positions: Dict[int, np.ndarray]) -> float:
        """Compute relative error"""
        if self.true_positions is None:
            return 0.0
        
        n = self.config.n_sensors
        X_hat = np.array([positions[i] for i in range(n)])
        X_0 = np.array([self.true_positions[i] for i in range(n)])
        
        return np.linalg.norm(X_hat - X_0, 'fro') / (np.linalg.norm(X_0, 'fro') + 1e-10)


class SensorBlockResolvent:
    """
    Resolvent operator for a block of sensors
    Implements the proximal operator for distance constraints
    """
    
    def __init__(self, block_id, n_sensors, distance_measurements, anchor_distances, anchor_positions, scale, alpha):
        self.block_id = block_id
        self.n_sensors = n_sensors
        self.distance_measurements = distance_measurements
        self.anchor_distances = anchor_distances
        self.anchor_positions = anchor_positions
        self.scale = scale
        self.alpha = alpha
        self.shape = (n_sensors, 2)  # Assuming 2D
    
    def prox(self, y, alpha):
        """
        Apply proximal operator for distance constraints
        y: input positions (n_sensors x d)
        alpha: step size
        """
        result = y.copy()
        
        # Apply distance constraints for each sensor
        for i in range(self.n_sensors):
            pos = result[i]
            
            # Sensor-to-sensor constraints
            n_constraints = 0
            for j in range(self.n_sensors):
                if (i, j) in self.distance_measurements:
                    measured_dist = self.distance_measurements[(i, j)]
                    n_constraints += 1
                    
                    # Adaptive weight based on block and constraints
                    weight = alpha / max(1, n_constraints)
                    if self.block_id == 1:  # Second block gets slightly different weight
                        weight *= 0.9
                    
                    pos = ProximalOperators.prox_distance(
                        pos, result[j], measured_dist, alpha=weight
                    )
            
            # Anchor constraints (stronger weight)
            if i in self.anchor_distances:
                for k, measured_dist in self.anchor_distances[i].items():
                    weight = alpha * 2 / max(1, len(self.anchor_distances[i]))
                    pos = ProximalOperators.prox_distance(
                        pos, self.anchor_positions[k], measured_dist, alpha=weight
                    )
            
            # Box constraint
            pos = ProximalOperators.prox_box_constraint(
                pos, -0.1 * self.scale, 1.1 * self.scale
            )
            
            result[i] = pos
        
        return result


def test_oars_algorithm():
    """Test the OARS-based MPS algorithm"""
    print("Testing OARS-based MPS Algorithm")
    print("="*50)
    
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        verbose=True,
        max_iterations=1000
    )
    
    algo = OARSBasedMPSAlgorithm(config)
    algo.generate_network()
    
    print(f"Network: {config.n_sensors} sensors, {config.n_anchors} anchors")
    print(f"Parameters: γ={config.gamma}, α={config.alpha}")
    print("\nRunning OARS-based algorithm...")
    
    results = algo.run()
    
    print(f"\nResults:")
    print(f"  Converged: {results['converged']} at iteration {results['iterations']}")
    print(f"  Final relative error: {results['final_relative_error']:.4f}")
    
    if results['final_relative_error'] < 0.15:
        print("  ✓ Achieving expected performance range!")
    else:
        print("  ⚠ Performance needs further tuning")
    
    return results


if __name__ == "__main__":
    test_oars_algorithm()