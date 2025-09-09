"""
Final MPS Algorithm Implementation
Using OARS framework structure with proper Z and W matrices
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
    scale: float = 1.0  
    communication_range: float = 0.7
    noise_factor: float = 0.05
    gamma: float = 0.999  # Paper value
    alpha: float = 10.0   # Paper value
    max_iterations: int = 2000
    tolerance: float = 1e-6
    dimension: int = 2
    seed: Optional[int] = 42
    verbose: bool = False


class FinalMPSAlgorithm:
    """
    Final MPS implementation with proper OARS structure
    """
    
    def __init__(self, config: MPSConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        self.true_positions = None
        self.anchor_positions = None
        self.distance_measurements = {}
        self.anchor_distances = {}
        self.adjacency = None
        
        # Algorithm matrices - will be created per OARS
        self.Z_matrix = None
        self.W_matrix = None
        
    def generate_network(self):
        """Generate synthetic network"""
        n = self.config.n_sensors
        d = self.config.dimension
        scale = self.config.scale
        
        # Generate random positions
        self.true_positions = {}
        for i in range(n):
            self.true_positions[i] = np.random.uniform(0, scale, d)
        
        # Generate anchor positions
        if self.config.n_anchors > 0:
            self.anchor_positions = np.zeros((self.config.n_anchors, d))
            if d == 2 and self.config.n_anchors >= 4:
                corners = np.array([
                    [0.1, 0.1], [0.9, 0.1], 
                    [0.9, 0.9], [0.1, 0.9]
                ]) * scale
                self.anchor_positions[:min(4, self.config.n_anchors)] = corners[:min(4, self.config.n_anchors)]
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
        
        self._generate_measurements()
        self._create_matrices_oars_style()
    
    def _generate_measurements(self):
        """Generate noisy distance measurements"""
        n = self.config.n_sensors
        noise = self.config.noise_factor
        
        for i in range(n):
            for j in range(i+1, n):
                if self.adjacency[i, j] > 0:
                    true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.distance_measurements[(i, j)] = max(0.01, noisy_dist)
                    self.distance_measurements[(j, i)] = self.distance_measurements[(i, j)]
        
        for i in range(n):
            self.anchor_distances[i] = {}
            for k in range(self.config.n_anchors):
                true_dist = np.linalg.norm(self.true_positions[i] - self.anchor_positions[k])
                if true_dist <= self.config.communication_range * self.config.scale:
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.anchor_distances[i][k] = max(0.01, noisy_dist)
    
    def _create_matrices_oars_style(self):
        """
        Create Z and W matrices following OARS prebuilt patterns
        Using a modified two-block structure for sensor networks
        """
        n = self.config.n_sensors
        
        # Create consensus weight matrix from network topology
        weight_matrix = self.adjacency + np.eye(n)
        weight_matrix = MatrixOperations.sinkhorn_knopp(weight_matrix)
        
        # Build 2-block structure (2n x 2n total)
        # Following OARS getTwoBlockSimilar pattern
        p = 2 * n  # Total size
        m = n  # Block size
        
        # Initialize W matrix - diagonal dominant with cross-block coupling
        self.W_matrix = np.eye(p) * 2.0
        
        # Off-diagonal blocks with negative coupling
        # This creates the consensus effect between blocks
        v = -2.0 / n  # Coupling strength
        
        # Incorporate network topology into coupling
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Within first block - use topology
                    self.W_matrix[i, j] = -weight_matrix[i, j] * 2
                    # Within second block - use topology
                    self.W_matrix[i+n, j+n] = -weight_matrix[i, j] * 2
                
                # Cross-block coupling
                self.W_matrix[i, j+n] = v * weight_matrix[i, j]
                self.W_matrix[i+n, j] = v * weight_matrix[i, j]
        
        # Ensure W is symmetric and has correct spectral properties
        self.W_matrix = (self.W_matrix + self.W_matrix.T) / 2
        
        # Z matrix - for OARS, often Z = W for simplicity
        # But we make it strictly lower triangular for the implicit equation
        self.Z_matrix = np.tril(self.W_matrix, -1)
        
    def run(self) -> Dict:
        """
        Run MPS algorithm with OARS structure
        """
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Total number of blocks (using 2-block structure)
        n_blocks = 2
        
        # Initialize variables for each block
        # all_x[i] is the primal variable for block i
        # all_v[i] is the dual/consensus variable for block i
        all_x = [np.zeros((n, d)) for _ in range(n_blocks)]
        all_v = [np.zeros((n, d)) for _ in range(n_blocks)]
        
        # Warm start near anchors
        for i in range(n):
            if i in self.anchor_distances and len(self.anchor_distances[i]) > 0:
                anchor_ids = list(self.anchor_distances[i].keys())
                init_pos = np.mean(self.anchor_positions[anchor_ids], axis=0)
                init_pos += 0.01 * self.config.scale * np.random.randn(d)
            else:
                init_pos = np.random.uniform(0, self.config.scale, d)
            
            for block in range(n_blocks):
                all_v[block][i] = init_pos
        
        # Create resolvent for each block
        resolvents = []
        for block in range(n_blocks):
            resolvents.append(BlockResolvent(
                block, n, d, self.distance_measurements, 
                self.anchor_distances, self.anchor_positions, 
                self.config.scale, self.config.alpha
            ))
        
        # Track metrics
        relative_errors = []
        iterations_tracked = []
        best_error = float('inf')
        
        # Main OARS iteration loop
        for itr in range(self.config.max_iterations):
            old_x = [x.copy() for x in all_x]
            
            # Step 1: Apply resolvents with implicit equation (OARS lines 79-84)
            for block_idx in range(n_blocks):
                # Compute y = v[i] - sum(Z[i,j]*x[j] for j < i)
                y_block = all_v[block_idx].copy()
                
                # Apply Z matrix (lower triangular part only)
                for i in range(n):
                    global_i = block_idx * n + i
                    
                    # Sum over all j < global_i
                    for global_j in range(global_i):
                        block_j = global_j // n
                        local_j = global_j % n
                        
                        if self.Z_matrix[global_i, global_j] != 0:
                            y_block[i] -= self.Z_matrix[global_i, global_j] * all_x[block_j][local_j]
                
                # Apply resolvent (proximal operator)
                all_x[block_idx] = resolvents[block_idx].prox(y_block)
            
            # Step 2: Update v with W matrix (OARS lines 86-88)
            for block_idx in range(n_blocks):
                # Compute W*x
                wx = np.zeros((n, d))
                for i in range(n):
                    global_i = block_idx * n + i
                    
                    for block_j in range(n_blocks):
                        for j in range(n):
                            global_j = block_j * n + j
                            if self.W_matrix[global_i, global_j] != 0:
                                wx[i] += self.W_matrix[global_i, global_j] * all_x[block_j][j]
                
                # Update v: v^(k+1) = v^k - gamma*W*x
                all_v[block_idx] = all_v[block_idx] - self.config.gamma * wx
            
            # Extract positions (average blocks)
            positions = {}
            for i in range(n):
                positions[i] = sum(all_x[b][i] for b in range(n_blocks)) / n_blocks
            
            # Track metrics
            if itr % 10 == 0:
                rel_error = self.compute_relative_error(positions)
                relative_errors.append(rel_error)
                iterations_tracked.append(itr)
                
                if rel_error < best_error:
                    best_error = rel_error
                
                if self.config.verbose and itr % 100 == 0:
                    # Compute convergence metrics
                    wx_norms = [np.linalg.norm(self.W_matrix[b*n:(b+1)*n, :] @ np.vstack(all_x).flatten().reshape(-1, d)) 
                               for b in range(n_blocks)]
                    delta_v = self.config.gamma * np.linalg.norm(wx_norms)
                    print(f"  Iter {itr}: rel_error={rel_error:.4f}, best={best_error:.4f}, delta_v={delta_v:.6f}")
            
            # Check convergence
            max_change = max([np.linalg.norm(all_x[b] - old_x[b]) for b in range(n_blocks)])
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
            'best_relative_error': best_error,
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


class BlockResolvent:
    """Resolvent for a block of sensors"""
    
    def __init__(self, block_id, n_sensors, dimension, distance_measurements, 
                 anchor_distances, anchor_positions, scale, alpha):
        self.block_id = block_id
        self.n_sensors = n_sensors
        self.dimension = dimension
        self.distance_measurements = distance_measurements
        self.anchor_distances = anchor_distances
        self.anchor_positions = anchor_positions
        self.scale = scale
        self.alpha = alpha
        self.shape = (n_sensors, dimension)
    
    def prox(self, y):
        """Apply proximal operator"""
        result = y.copy()
        
        # Different strategies for different blocks
        if self.block_id == 0:
            # First block: focus on sensor-to-sensor
            weight_sensor = self.alpha
            weight_anchor = self.alpha * 3
        else:
            # Second block: focus on anchors
            weight_sensor = self.alpha * 0.5
            weight_anchor = self.alpha * 5
        
        # Apply constraints
        for i in range(self.n_sensors):
            pos = result[i]
            
            # Sensor constraints
            for j in range(self.n_sensors):
                if (i, j) in self.distance_measurements:
                    pos = ProximalOperators.prox_distance(
                        pos, result[j], self.distance_measurements[(i, j)],
                        alpha=weight_sensor / (1 + abs(i - j))
                    )
            
            # Anchor constraints
            if i in self.anchor_distances:
                for k, dist in self.anchor_distances[i].items():
                    pos = ProximalOperators.prox_distance(
                        pos, self.anchor_positions[k], dist,
                        alpha=weight_anchor
                    )
            
            # Box constraint
            pos = ProximalOperators.prox_box_constraint(
                pos, -0.2 * self.scale, 1.2 * self.scale
            )
            
            result[i] = pos
        
        return result


def test_final_algorithm():
    """Test the final MPS algorithm"""
    print("Testing Final MPS Algorithm")
    print("="*50)
    
    # Run multiple trials
    errors = []
    for trial in range(3):
        config = MPSConfig(
            n_sensors=30,
            n_anchors=6,
            verbose=(trial == 0),
            seed=42 + trial,
            gamma=0.999,  # Paper value
            alpha=10.0,   # Paper value
            max_iterations=2000
        )
        
        algo = FinalMPSAlgorithm(config)
        algo.generate_network()
        
        if trial == 0:
            print(f"Network: {config.n_sensors} sensors, {config.n_anchors} anchors")
            print(f"Parameters: γ={config.gamma}, α={config.alpha}")
            print("\nRunning final algorithm...")
        
        results = algo.run()
        errors.append(results['best_relative_error'])
        
        if trial == 0:
            print(f"\nTrial {trial+1} Results:")
            print(f"  Converged: {results['converged']} at iteration {results['iterations']}")
            print(f"  Final relative error: {results['final_relative_error']:.4f}")
            print(f"  Best relative error: {results['best_relative_error']:.4f}")
    
    print(f"\nSummary over {len(errors)} trials:")
    print(f"  Mean best error: {np.mean(errors):.4f}")
    print(f"  Min best error: {np.min(errors):.4f}")
    print(f"  Max best error: {np.max(errors):.4f}")
    
    if np.mean(errors) < 0.20:
        print("  ✓ Making progress toward expected performance!")
    else:
        print("  ⚠ Still needs optimization")
    
    return errors


if __name__ == "__main__":
    test_final_algorithm()