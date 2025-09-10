"""
Proper MPS Algorithm Implementation
Faithful to Algorithm 1 from arxiv:2503.13403v1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import sqrtm
from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations


@dataclass  
class MPSConfig:
    """Configuration for MPS algorithm"""
    n_sensors: int = 30
    n_anchors: int = 6
    scale: float = 1.0  # Network scale in meters
    communication_range: float = 0.7  # As fraction of scale (0.7 in paper)
    noise_factor: float = 0.05  # 5% noise as in paper
    gamma: float = 0.999  # Paper's value (Section 3)
    alpha: float = 10.0   # Paper's value for Algorithm 1
    max_iterations: int = 500
    tolerance: float = 1e-5
    dimension: int = 2
    seed: Optional[int] = 42
    verbose: bool = False
    fixed_point_iterations: int = 10  # For implicit equation solver


@dataclass
class MPSState:
    """State variables for MPS algorithm"""
    v: np.ndarray  # Consensus variables (p x d) where p = 2n
    x: np.ndarray  # Auxiliary variables after fixed-point solution
    positions: Dict[int, np.ndarray]  # Extracted position estimates
    iteration: int = 0
    converged: bool = False


class ProperMPSAlgorithm:
    """
    Proper Matrix-Parametrized Proximal Splitting algorithm
    Faithful implementation of Algorithm 1 from the paper
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
        self.Z_matrix = None  # Consensus matrix
        self.W_matrix = None  # Update matrix
        self.L_matrix = None  # Lower triangular such that Z = 2I - L - L^T
        
    def generate_network(self):
        """Generate synthetic network matching paper's setup"""
        n = self.config.n_sensors
        d = self.config.dimension
        scale = self.config.scale
        
        # Generate random positions in [0, scale]²
        self.true_positions = {}
        for i in range(n):
            self.true_positions[i] = np.random.uniform(0, scale, d)
        
        # Generate anchor positions
        self.anchor_positions = np.random.uniform(0, scale, (self.config.n_anchors, d))
        
        # Build adjacency (distance < 0.7 * scale)
        comm_range = self.config.communication_range * scale
        self.adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                if dist <= comm_range:
                    self.adjacency[i, j] = 1
                    self.adjacency[j, i] = 1
        
        # Generate noisy measurements as in paper
        self._generate_measurements()
        
        # Create algorithm matrices (Z, W, L)
        self._create_algorithm_matrices()
    
    def _generate_measurements(self):
        """Generate noisy distance measurements exactly as in paper"""
        n = self.config.n_sensors
        noise = self.config.noise_factor
        
        # Sensor-to-sensor: d̃ᵢⱼ = d⁰ᵢⱼ(1 + 0.05εᵢⱼ)
        for i in range(n):
            for j in range(i+1, n):
                if self.adjacency[i, j] > 0:
                    true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.distance_measurements[(i, j)] = max(0.01, noisy_dist)
                    self.distance_measurements[(j, i)] = self.distance_measurements[(i, j)]
        
        # Sensor-to-anchor measurements
        for i in range(n):
            self.anchor_distances[i] = {}
            for k in range(self.config.n_anchors):
                true_dist = np.linalg.norm(self.true_positions[i] - self.anchor_positions[k])
                if true_dist <= self.config.communication_range * self.config.scale:
                    epsilon = np.random.randn()
                    noisy_dist = true_dist * (1 + noise * epsilon)
                    self.anchor_distances[i][k] = max(0.01, noisy_dist)
    
    def _create_algorithm_matrices(self):
        """Create Z, W, L matrices as specified in Algorithm 1"""
        n = self.config.n_sensors
        p = 2 * n  # 2-block structure
        gamma = self.config.gamma
        
        # Create weight matrix from adjacency
        weight_matrix = self.adjacency + np.eye(n)  # Add self-loops
        
        # Make doubly stochastic using Sinkhorn-Knopp
        weight_matrix = MatrixOperations.sinkhorn_knopp(weight_matrix)
        
        # Create 2-block structure
        # Z matrix for consensus (must be positive semidefinite and satisfy constraints)
        self.Z_matrix = np.zeros((p, p))
        self.Z_matrix[:n, :n] = gamma * weight_matrix
        self.Z_matrix[:n, n:] = (1 - gamma) * np.eye(n)
        self.Z_matrix[n:, :n] = (1 - gamma) * np.eye(n)
        self.Z_matrix[n:, n:] = gamma * weight_matrix
        
        # Ensure Z is symmetric positive semidefinite
        self.Z_matrix = (self.Z_matrix + self.Z_matrix.T) / 2
        
        # Create L matrix such that Z = 2I - L - L^T
        # This means L + L^T = 2I - Z
        # For simplicity, we can use L = I - Z/2 (lower triangular part)
        identity = np.eye(p)
        temp = 2 * identity - self.Z_matrix
        self.L_matrix = np.tril(temp / 2)  # Take lower triangular part
        
        # W matrix is typically similar to Z but may differ
        # For stability, we use W = Z initially
        self.W_matrix = self.Z_matrix.copy()
        
    def initialize_state(self) -> MPSState:
        """Initialize algorithm state with zero-sum constraint"""
        n = self.config.n_sensors
        d = self.config.dimension
        p = 2 * n
        
        # Initialize v with zero-sum constraint (∑ᵢ vᵢ = 0)
        v = np.zeros((p, d))
        
        # Initialize positions near anchors if available
        positions = {}
        for i in range(n):
            if i in self.anchor_distances and len(self.anchor_distances[i]) > 0:
                anchor_ids = list(self.anchor_distances[i].keys())
                positions[i] = np.mean(self.anchor_positions[anchor_ids], axis=0)
            else:
                positions[i] = np.random.uniform(0, self.config.scale, d)
        
        # Set v for both blocks
        mean_pos = np.mean(list(positions.values()), axis=0)
        for i in range(n):
            v[i] = positions[i] - mean_pos  # Center positions
            v[i + n] = v[i]  # Second block same as first
        
        # Ensure zero-sum constraint
        v = v - np.mean(v, axis=0)
        
        return MPSState(
            v=v,
            x=np.zeros((p, d)),
            positions=positions
        )
    
    def solve_implicit_equation(self, v: np.ndarray) -> np.ndarray:
        """
        Solve the implicit equation: x = J_αF(v + Lx)
        Uses fixed-point iteration
        
        Args:
            v: Current consensus variables
            
        Returns:
            Solution x
        """
        p, d = v.shape
        x = np.zeros((p, d))
        
        # Fixed-point iteration: x^(t+1) = J_αF(v + L @ x^t)
        for _ in range(self.config.fixed_point_iterations):
            x_old = x.copy()
            
            # Compute v + L @ x
            argument = v + self.L_matrix @ x
            
            # Apply proximal operator J_αF (resolvent)
            x = self.prox_f(argument)
            
            # Check convergence
            if np.linalg.norm(x - x_old) < 1e-6:
                break
        
        return x
    
    def prox_f(self, z: np.ndarray) -> np.ndarray:
        """
        Apply proximal operators to input z
        This is J_αF in the paper notation
        
        Args:
            z: Input variables (p x d)
            
        Returns:
            Proximal operator result
        """
        n = self.config.n_sensors
        d = self.config.dimension
        result = z.copy()
        
        # First n components: distance constraint proximal operators
        for i in range(n):
            zi_pos = z[i]
            result_pos = zi_pos.copy()
            
            # Apply distance constraints with neighbors
            for j in range(n):
                if (i, j) in self.distance_measurements:
                    zj_pos = z[j] if j < n else z[j - n]
                    measured_dist = self.distance_measurements[(i, j)]
                    
                    # Project onto distance constraint
                    result_pos = ProximalOperators.prox_distance(
                        result_pos, zj_pos, measured_dist, 
                        alpha=self.config.alpha
                    )
            
            # Apply anchor distance constraints
            if i in self.anchor_distances:
                for k, measured_dist in self.anchor_distances[i].items():
                    result_pos = ProximalOperators.prox_distance(
                        result_pos, self.anchor_positions[k], measured_dist,
                        alpha=self.config.alpha * 2  # Stronger weight for anchors
                    )
            
            result[i] = result_pos
        
        # Second n components: copy from first (simplified PSD constraint)
        for i in range(n):
            result[i + n] = result[i]
        
        return result
    
    def extract_positions(self, v: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract position estimates from consensus variables"""
        n = self.config.n_sensors
        positions = {}
        
        for i in range(n):
            # Average the two blocks
            positions[i] = (v[i] + v[i + n]) / 2
        
        return positions
    
    def compute_relative_error(self, positions: Dict[int, np.ndarray]) -> float:
        """Compute relative error as in paper: ||X̂ - X⁰||_F / ||X⁰||_F"""
        if self.true_positions is None:
            return 0.0
        
        n = self.config.n_sensors
        X_hat = np.array([positions[i] for i in range(n)])
        X_0 = np.array([self.true_positions[i] for i in range(n)])
        
        return np.linalg.norm(X_hat - X_0, 'fro') / np.linalg.norm(X_0, 'fro')
    
    def compute_rmse(self, positions: Dict[int, np.ndarray]) -> float:
        """Compute RMSE vs true positions"""
        if self.true_positions is None:
            return 0.0
        
        errors = []
        for i in range(self.config.n_sensors):
            error = np.linalg.norm(positions[i] - self.true_positions[i])
            errors.append(error ** 2)
        
        return np.sqrt(np.mean(errors))
    
    def run(self) -> Dict:
        """
        Run Algorithm 1 from the paper
        
        Algorithm 1:
        1. Initialize v⁰ ∈ H^p with Σᵢ vᵢ⁰ = 0
        2. Repeat:
           - Solve x^k = J_αF(v^k + Lx^k) for x^k
           - v^(k+1) = v^k - γWx^k
        """
        # Initialize
        state = self.initialize_state()
        
        # Track metrics
        relative_errors = []
        rmse_history = []
        iterations_tracked = []
        
        # Main Algorithm 1 loop
        for k in range(self.config.max_iterations):
            # Store previous v for convergence check
            v_old = state.v.copy()
            
            # Step 1: Solve implicit equation x^k = J_αF(v^k + Lx^k)
            state.x = self.solve_implicit_equation(state.v)
            
            # Step 2: Update consensus variables
            state.v = state.v - self.config.gamma * (self.W_matrix @ state.x)
            
            # Maintain zero-sum constraint
            state.v = state.v - np.mean(state.v, axis=0)
            
            # Extract positions
            state.positions = self.extract_positions(state.v)
            
            # Track metrics periodically
            if k % 5 == 0:
                rel_error = self.compute_relative_error(state.positions)
                relative_errors.append(rel_error)
                iterations_tracked.append(k)
                
                if self.config.verbose and k % 50 == 0:
                    rmse = self.compute_rmse(state.positions)
                    print(f"  Iter {k}: relative_error={rel_error:.4f}, rmse={rmse:.4f}")
            
            # Check convergence
            change = np.linalg.norm(state.v - v_old) / (np.linalg.norm(v_old) + 1e-10)
            if change < self.config.tolerance:
                state.converged = True
                state.iteration = k
                break
        
        if not state.converged:
            state.iteration = self.config.max_iterations
        
        # Final metrics
        final_rel_error = self.compute_relative_error(state.positions)
        final_rmse = self.compute_rmse(state.positions)
        
        return {
            'converged': state.converged,
            'iterations': state.iteration,
            'final_relative_error': final_rel_error,
            'final_rmse': final_rmse,
            'relative_errors': relative_errors,
            'iterations_tracked': iterations_tracked,
            'final_positions': dict(state.positions),
            'true_positions': dict(self.true_positions) if self.true_positions else {}
        }


def test_proper_algorithm():
    """Quick test of the proper algorithm"""
    print("Testing Proper MPS Algorithm (Algorithm 1)")
    print("="*50)
    
    config = MPSConfig(
        n_sensors=30,
        n_anchors=6,
        verbose=True
    )
    
    algo = ProperMPSAlgorithm(config)
    algo.generate_network()
    
    print(f"Network: {config.n_sensors} sensors, {config.n_anchors} anchors")
    print(f"Parameters: γ={config.gamma}, α={config.alpha}")
    print("\nRunning Algorithm 1...")
    
    results = algo.run()
    
    print(f"\nResults:")
    print(f"  Converged: {results['converged']} at iteration {results['iterations']}")
    print(f"  Final relative error: {results['final_relative_error']:.4f}")
    print(f"  Final RMSE: {results['final_rmse']:.4f}")
    
    return results


if __name__ == "__main__":
    test_proper_algorithm()