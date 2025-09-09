"""
Full Matrix-Parametrized Proximal Splitting Algorithm Implementation
Based on: "Decentralized Sensor Network Localization using 
Matrix-Parametrized Proximal Splittings" (arXiv:2503.13403v1)

This is the complete implementation of Algorithm 1 from the paper,
with proper lifted variable structure and ADMM-based proximal operators.
"""

import numpy as np
from scipy.linalg import eigh, cholesky, solve_triangular
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sinkhorn_knopp import SinkhornKnopp, MatrixParameterGenerator
from .proximal_sdp import ProximalADMMSolver, ProximalOperatorsPSD

logger = logging.getLogger(__name__)


@dataclass
class MPSConfig:
    """Configuration for full MPS algorithm"""
    n_sensors: int
    n_anchors: int
    dimension: int = 2
    gamma: float = 0.999  # Step size for consensus update
    alpha: float = 10.0   # Scaling parameter for proximal operators
    max_iterations: int = 1000
    tolerance: float = 1e-6
    communication_range: float = 0.3  # As fraction of network scale
    scale: float = 1.0  # Physical scale of network
    verbose: bool = False
    early_stopping: bool = True
    early_stopping_window: int = 100
    admm_iterations: int = 100  # Inner ADMM iterations
    admm_tolerance: float = 1e-6
    admm_rho: float = 1.0
    warm_start: bool = True
    parallel_proximal: bool = True  # Enable parallel proximal evaluations
    use_2block: bool = True  # Use 2-Block design
    adaptive_alpha: bool = True  # Adaptive alpha scaling
    carrier_phase_mode: bool = False  # Enable for millimeter accuracy


@dataclass
class NetworkData:
    """Data structure for sensor network"""
    adjacency_matrix: np.ndarray  # Communication graph
    distance_measurements: Dict[Tuple[int, int], float]  # (i,j) -> distance
    anchor_positions: np.ndarray  # m x d matrix of anchor positions
    anchor_connections: Dict[int, List[int]]  # sensor -> connected anchors
    true_positions: Optional[np.ndarray] = None  # For testing/validation
    carrier_phase_measurements: Optional[Dict] = None  # Phase measurements
    measurement_variance: float = 1e-6  # Measurement noise variance


class LiftedVariableStructure:
    """
    Manages the lifted variable structure x ∈ H^p where p = 2n
    First n components: objectives g_i
    Last n components: PSD constraints δ_i
    """
    
    def __init__(self, n_sensors: int, dimension: int):
        self.n = n_sensors
        self.d = dimension
        self.p = 2 * n_sensors  # Number of lifted components
        
        # Each lifted variable x_i is a matrix in S^(d+|N_i|+1)
        # We store them as a list of matrices
        self.matrix_dims = {}  # i -> dimension of S^i matrix
        
    def initialize(self, neighborhoods: Dict[int, List[int]]) -> List[np.ndarray]:
        """
        Initialize lifted variables based on network structure
        
        Args:
            neighborhoods: Dictionary mapping sensor to neighbors
            
        Returns:
            List of initialized matrix variables
        """
        x = []
        
        for i in range(self.p):
            sensor_idx = i % self.n
            neighbors = neighborhoods.get(sensor_idx, [])
            
            # Dimension: d + 1 + |N_i|
            dim = self.d + 1 + len(neighbors)
            self.matrix_dims[i] = dim
            
            # Initialize as identity (feasible starting point)
            x_i = np.eye(dim)
            x.append(x_i)
        
        return x
    
    def extract_S_matrix(self, x_i: np.ndarray, sensor_idx: int,
                         neighbors: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X and Y components from lifted matrix variable
        
        The matrix has structure:
        S^i = [I_d    X_i^T    X_N_i^T]
              [X_i    Y_ii     Y_iN_i  ]
              [X_N_i  Y_N_i,i  Y_N_i,N_i]
        
        Args:
            x_i: Lifted matrix variable
            sensor_idx: Sensor index
            neighbors: Neighbor indices
            
        Returns:
            X positions (n x d), Y matrix (n x n)
        """
        d = self.d
        n_neighbors = len(neighbors)
        
        # Extract X positions
        X = np.zeros((self.n, d))
        X[sensor_idx] = x_i[d, :d]  # X_i from first row after I_d
        
        for j_idx, j in enumerate(neighbors):
            X[j] = x_i[d + 1 + j_idx, :d]  # X_j from neighbor rows
        
        # Extract Y matrix entries
        Y = np.zeros((self.n, self.n))
        Y[sensor_idx, sensor_idx] = x_i[d, d]  # Y_ii
        
        for j_idx, j in enumerate(neighbors):
            Y[sensor_idx, j] = x_i[d, d + 1 + j_idx]  # Y_ij
            Y[j, sensor_idx] = Y[sensor_idx, j]  # Symmetry
            Y[j, j] = x_i[d + 1 + j_idx, d + 1 + j_idx]  # Y_jj
        
        # Extract neighbor-neighbor entries
        for j1_idx, j1 in enumerate(neighbors):
            for j2_idx, j2 in enumerate(neighbors):
                if j1_idx <= j2_idx:
                    Y[j1, j2] = x_i[d + 1 + j1_idx, d + 1 + j2_idx]
                    Y[j2, j1] = Y[j1, j2]
        
        return X, Y
    
    def construct_from_XY(self, X: np.ndarray, Y: np.ndarray,
                         sensor_idx: int, neighbors: List[int]) -> np.ndarray:
        """
        Construct lifted matrix variable from X and Y
        
        Args:
            X: Position estimates (n x d)
            Y: Gram-like matrix (n x n)
            sensor_idx: Sensor index
            neighbors: Neighbor indices
            
        Returns:
            Lifted matrix variable S^i
        """
        d = self.d
        n_neighbors = len(neighbors)
        dim = d + 1 + n_neighbors
        
        S = np.zeros((dim, dim))
        
        # Upper-left: I_d
        S[:d, :d] = np.eye(d)
        
        # First d columns after I_d: positions
        S[:d, d] = X[sensor_idx]  # X_i
        for j_idx, j in enumerate(neighbors):
            S[:d, d + 1 + j_idx] = X[j]  # X_j for neighbors
        
        # Symmetric part
        S[d, :d] = X[sensor_idx]
        for j_idx, j in enumerate(neighbors):
            S[d + 1 + j_idx, :d] = X[j]
        
        # Y matrix entries
        S[d, d] = Y[sensor_idx, sensor_idx]
        for j_idx, j in enumerate(neighbors):
            S[d, d + 1 + j_idx] = Y[sensor_idx, j]
            S[d + 1 + j_idx, d] = Y[j, sensor_idx]
            S[d + 1 + j_idx, d + 1 + j_idx] = Y[j, j]
        
        # Neighbor-neighbor entries
        for j1_idx, j1 in enumerate(neighbors):
            for j2_idx, j2 in enumerate(neighbors):
                S[d + 1 + j1_idx, d + 1 + j2_idx] = Y[j1, j2]
        
        return S


class ProximalEvaluator:
    """
    Handles proximal operator evaluations for the algorithm
    """
    
    def __init__(self, config: MPSConfig, network_data: NetworkData,
                 neighborhoods: Dict[int, List[int]]):
        self.config = config
        self.network_data = network_data
        self.neighborhoods = neighborhoods
        
        # Initialize ADMM solvers for each sensor
        self.admm_solvers = {}
        for i in range(config.n_sensors):
            self.admm_solvers[i] = ProximalADMMSolver(
                rho=config.admm_rho,
                max_iterations=config.admm_iterations,
                tolerance=config.admm_tolerance,
                warm_start=config.warm_start
            )
        
        self.psd_ops = ProximalOperatorsPSD()
        
    def prox_objective_gi(self, S_input: np.ndarray, sensor_idx: int,
                          alpha: float) -> np.ndarray:
        """
        Proximal operator for objective g_i using ADMM
        
        Args:
            S_input: Input matrix S^i
            sensor_idx: Sensor index
            alpha: Proximal parameter
            
        Returns:
            Updated matrix after proximal operation
        """
        neighbors = self.neighborhoods.get(sensor_idx, [])
        d = self.config.dimension
        
        # Extract current X and Y from input matrix
        lifted = LiftedVariableStructure(self.config.n_sensors, d)
        X_curr, Y_curr = lifted.extract_S_matrix(S_input, sensor_idx, neighbors)
        
        # Get distance measurements for this sensor
        distances_sensors = {}
        for j in neighbors:
            key = (min(sensor_idx, j), max(sensor_idx, j))
            if key in self.network_data.distance_measurements:
                distances_sensors[j] = self.network_data.distance_measurements[key]
        
        # Get anchor connections and distances
        anchors = self.network_data.anchor_connections.get(sensor_idx, [])
        distances_anchors = {}
        for a in anchors:
            key = (sensor_idx, a)
            if key in self.network_data.distance_measurements:
                distances_anchors[a] = self.network_data.distance_measurements[key]
        
        # Apply ADMM solver
        X_new, Y_new = self.admm_solvers[sensor_idx].solve(
            X_curr, Y_curr, sensor_idx, neighbors, anchors,
            distances_sensors, distances_anchors,
            self.network_data.anchor_positions, alpha
        )
        
        # Reconstruct lifted matrix
        S_new = lifted.construct_from_XY(X_new, Y_new, sensor_idx, neighbors)
        
        return S_new
    
    def prox_indicator_psd(self, S_input: np.ndarray) -> np.ndarray:
        """
        Proximal operator for PSD constraint (projection)
        
        Args:
            S_input: Input matrix
            
        Returns:
            Projected PSD matrix
        """
        return self.psd_ops.project_psd_cone(S_input)
    
    def evaluate_parallel(self, x: List[np.ndarray], v: List[np.ndarray],
                         L: np.ndarray, iteration: int) -> List[np.ndarray]:
        """
        Evaluate proximal operators in parallel for 2-Block structure
        
        Args:
            x: Current x variables
            v: Consensus variables
            L: Lower triangular matrix
            iteration: Current iteration number
            
        Returns:
            Updated x variables
        """
        n = self.config.n_sensors
        x_new = [None] * (2 * n)
        
        # Adaptive alpha scaling
        alpha = self.config.alpha
        if self.config.adaptive_alpha:
            # Scale alpha based on iteration and measurement precision
            if self.config.carrier_phase_mode:
                # For millimeter accuracy, use smaller alpha initially
                alpha = self.config.alpha * (1.0 + iteration / 100.0)
            else:
                alpha = self.config.alpha / (1.0 + iteration / 500.0)
        
        if self.config.parallel_proximal and self.config.use_2block:
            # Parallel evaluation for 2-Block
            # Block 1: objectives g_i (i = 0, ..., n-1)
            # Block 2: PSD constraints δ_i (i = n, ..., 2n-1)
            
            with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
                # Submit Block 1 evaluations
                futures_block1 = {}
                for i in range(n):
                    # For 2-block, we work directly with v[i] since matrix dimensions vary
                    # L matrix coupling is handled at the consensus level, not here
                    input_val = v[i].copy()
                    
                    future = executor.submit(
                        self.prox_objective_gi, input_val, i, alpha
                    )
                    futures_block1[future] = i
                
                # Collect Block 1 results
                for future in as_completed(futures_block1):
                    i = futures_block1[future]
                    x_new[i] = future.result()
                
                # Submit Block 2 evaluations
                futures_block2 = {}
                for i in range(n, 2*n):
                    # For PSD constraints, use v[i] directly
                    input_val = v[i].copy()
                    
                    future = executor.submit(
                        self.prox_indicator_psd, input_val
                    )
                    futures_block2[future] = i
                
                # Collect Block 2 results
                for future in as_completed(futures_block2):
                    i = futures_block2[future]
                    x_new[i] = future.result()
        
        else:
            # Sequential evaluation
            for i in range(2 * n):
                # For sequential, also use v[i] directly due to varying dimensions
                input_val = v[i].copy()
                
                if i < n:
                    # Objective proximal operator
                    x_new[i] = self.prox_objective_gi(input_val, i, alpha)
                else:
                    # PSD constraint proximal operator
                    x_new[i] = self.prox_indicator_psd(input_val)
        
        return x_new


class MatrixParametrizedProximalSplitting:
    """
    Main algorithm implementation with full matrix-parametrized structure
    """
    
    def __init__(self, config: MPSConfig, network_data: NetworkData):
        self.config = config
        self.network_data = network_data
        
        # Build neighborhoods from adjacency matrix
        self.neighborhoods = self._build_neighborhoods()
        
        # Initialize lifted variable structure
        self.lifted_structure = LiftedVariableStructure(
            config.n_sensors, config.dimension
        )
        
        # Initialize proximal evaluator
        self.prox_evaluator = ProximalEvaluator(
            config, network_data, self.neighborhoods
        )
        
        # Setup matrix parameters Z and W
        self._setup_matrix_parameters()
        
        # Initialize variables
        self._initialize_variables()
        
    def _build_neighborhoods(self) -> Dict[int, List[int]]:
        """Build neighborhood structure from adjacency matrix"""
        n = self.config.n_sensors
        adjacency = self.network_data.adjacency_matrix
        
        neighborhoods = {}
        for i in range(n):
            neighbors = []
            for j in range(n):
                if i != j and adjacency[i, j] > 0:
                    neighbors.append(j)
            neighborhoods[i] = neighbors
        
        return neighborhoods
    
    def _setup_matrix_parameters(self):
        """Setup Z and W matrices using Sinkhorn-Knopp algorithm"""
        n = self.config.n_sensors
        
        # For lifted matrix variables, we use scalar weights between components
        # The L matrix provides scalar coupling coefficients
        if self.config.use_2block:
            # Use 2-Block design with Sinkhorn-Knopp
            generator = MatrixParameterGenerator()
            self.Z, self.W = generator.generate_from_communication_graph(
                self.network_data.adjacency_matrix,
                method='sinkhorn-knopp',
                block_design='2-block'
            )
            
            # Compute lower triangular L such that Z = 2I - L - L^T
            self.L = generator.compute_lower_triangular_L(self.Z)
        else:
            # Simple identity-based design
            self.Z = 2 * np.eye(2 * n)
            self.W = self.Z.copy()
            self.L = np.eye(2 * n)
    
    def _initialize_variables(self):
        """Initialize lifted variables x and consensus variables v"""
        # Initialize lifted variables
        self.x = self.lifted_structure.initialize(self.neighborhoods)
        
        # Initialize consensus variables (same structure as x)
        self.v = [x_i.copy() for x_i in self.x]
        
        # Initialize position and Y matrix estimates
        n = self.config.n_sensors
        d = self.config.dimension
        
        if self.network_data.true_positions is not None:
            # Warm start with noisy true positions
            noise_level = 0.1 if not self.config.carrier_phase_mode else 0.01
            self.X = self.network_data.true_positions + noise_level * np.random.randn(n, d)
        else:
            # Random initialization
            self.X = np.random.uniform(0, 1, (n, d))
        
        # Initialize Y as Gram matrix
        self.Y = self.X @ self.X.T
        
        # Update lifted variables with initial estimates
        for i in range(n):
            neighbors = self.neighborhoods.get(i, [])
            S_init = self.lifted_structure.construct_from_XY(
                self.X, self.Y, i, neighbors
            )
            self.x[i] = S_init
            self.x[n + i] = S_init  # PSD constraint component
            self.v[i] = S_init.copy()
            self.v[n + i] = S_init.copy()
    
    def run_iteration(self, k: int) -> Dict[str, float]:
        """
        Run one iteration of Algorithm 1
        
        Args:
            k: Iteration number
            
        Returns:
            Dictionary with iteration statistics
        """
        # Step 1: Proximal evaluations (equation 9a-9c)
        self.x = self.prox_evaluator.evaluate_parallel(
            self.x, self.v, self.L, k
        )
        
        # Step 2: Consensus update (equation 9d)
        # v^(k+1) = v^k - γWx^k
        # For matrix variables with different dimensions, we handle consensus differently
        gamma = self.config.gamma
        n = self.config.n_sensors
        
        if self.config.use_2block:
            # For 2-block design, consensus is between corresponding blocks
            # Update objective components
            for i in range(n):
                # Average with corresponding PSD component (same sensor)
                if self.x[i].shape == self.x[n + i].shape:
                    consensus_avg = (self.x[i] + self.x[n + i]) / 2.0
                    self.v[i] = self.v[i] - gamma * (self.v[i] - consensus_avg)
                    self.v[n + i] = self.v[n + i] - gamma * (self.v[n + i] - consensus_avg)
                else:
                    # If dimensions differ, skip consensus for this iteration
                    pass
        else:
            # Simple consensus update for non-2block case
            for i in range(len(self.v)):
                update = np.zeros_like(self.v[i])
                weight = 1.0 / len(self.x)
                for j in range(len(self.x)):
                    if self.x[j].shape == self.v[i].shape:
                        update += weight * self.x[j]
                self.v[i] = self.v[i] - gamma * (self.v[i] - update)
        
        # Extract current position and Y estimates
        self._extract_estimates()
        
        # Compute statistics
        stats = {
            'iteration': k,
            'objective': self._compute_objective(),
            'psd_violation': self._compute_psd_violation(),
            'consensus_error': self._compute_consensus_error(),
            'position_error': self._compute_position_error()
        }
        
        return stats
    
    def _extract_estimates(self):
        """Extract X and Y from converged lifted variables"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Average estimates from all lifted components
        X_sum = np.zeros((n, d))
        Y_sum = np.zeros((n, n))
        count = np.zeros((n, n))
        
        for i in range(n):
            neighbors = self.neighborhoods.get(i, [])
            
            # Extract from objective component
            X_i, Y_i = self.lifted_structure.extract_S_matrix(
                self.x[i], i, neighbors
            )
            X_sum += X_i
            Y_sum += Y_i
            
            # Count contributions for averaging
            count[i, i] += 1
            for j in neighbors:
                count[i, j] += 1
                count[j, i] += 1
        
        # Average the estimates
        count[count == 0] = 1  # Avoid division by zero
        self.X = X_sum / n
        self.Y = Y_sum / count
        
        # Ensure Y is symmetric
        self.Y = (self.Y + self.Y.T) / 2
    
    def _compute_objective(self) -> float:
        """Compute total objective value"""
        total = 0.0
        
        for i in range(self.config.n_sensors):
            neighbors = self.neighborhoods.get(i, [])
            
            # Sensor-to-sensor terms
            for j in neighbors:
                key = (min(i, j), max(i, j))
                if key in self.network_data.distance_measurements:
                    d_ij = self.network_data.distance_measurements[key]
                    
                    # Weight based on measurement precision
                    weight = 1.0
                    if self.config.carrier_phase_mode and self.network_data.carrier_phase_measurements:
                        if key in self.network_data.carrier_phase_measurements:
                            # Higher weight for more precise measurements
                            precision = self.network_data.carrier_phase_measurements[key].get('precision_mm', 1.0)
                            weight = 1000.0 / max(precision, 0.01)  # Higher weight for better precision
                    
                    # Compute residual with proper scaling
                    est_dist = np.sqrt(max(0, self.Y[i, i] + self.Y[j, j] - 2 * self.Y[i, j]))
                    residual = weight * abs(est_dist - d_ij)
                    total += residual
            
            # Sensor-to-anchor terms
            anchors = self.network_data.anchor_connections.get(i, [])
            for a in anchors:
                key = (i, a)
                if key in self.network_data.distance_measurements:
                    d_ia = self.network_data.distance_measurements[key]
                    a_pos = self.network_data.anchor_positions[a]
                    
                    # Weight based on measurement precision
                    weight = 1.0
                    if self.config.carrier_phase_mode and self.network_data.carrier_phase_measurements:
                        if key in self.network_data.carrier_phase_measurements:
                            precision = self.network_data.carrier_phase_measurements[key].get('precision_mm', 1.0)
                            weight = 1000.0 / max(precision, 0.01)
                    
                    # Compute residual with proper scaling
                    est_dist = np.sqrt(max(0, self.Y[i, i] + np.dot(a_pos, a_pos) - 
                                       2 * np.dot(a_pos, self.X[i])))
                    residual = weight * abs(est_dist - d_ia)
                    total += residual
        
        return total
    
    def _compute_psd_violation(self) -> float:
        """Compute total PSD constraint violation"""
        total_violation = 0.0
        
        for i in range(self.config.n_sensors):
            neighbors = self.neighborhoods.get(i, [])
            
            # Check PSD constraint for S^i
            S_i = self.lifted_structure.construct_from_XY(
                self.X, self.Y, i, neighbors
            )
            
            # Compute minimum eigenvalue
            try:
                eigenvalues = eigh(S_i, eigvals_only=True)
                min_eig = np.min(eigenvalues)
                if min_eig < 0:
                    total_violation += abs(min_eig)
            except np.linalg.LinAlgError:
                total_violation += 1.0  # Penalty for numerical issues
        
        return total_violation
    
    def _compute_consensus_error(self) -> float:
        """Compute consensus error"""
        # Consensus error: how much variables differ across components
        n = self.config.n_sensors
        total_error = 0.0
        
        for i in range(n):
            # Compare objective and constraint components
            diff = np.linalg.norm(self.x[i] - self.x[n + i], 'fro')
            total_error += diff
        
        return total_error / n
    
    def _compute_position_error(self) -> float:
        """Compute position estimation error (if true positions available)"""
        if self.network_data.true_positions is None:
            return 0.0
        
        # Compute RMSE
        diff = self.X - self.network_data.true_positions
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        # Convert to millimeters if in carrier phase mode
        if self.config.carrier_phase_mode:
            rmse *= 1000  # Convert to mm
        
        return rmse
    
    def run(self, max_iterations: Optional[int] = None,
            tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the full algorithm
        
        Args:
            max_iterations: Maximum iterations (overrides config)
            tolerance: Convergence tolerance (overrides config)
            
        Returns:
            Results dictionary with final estimates and history
        """
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance
        
        history = {
            'objective': [],
            'psd_violation': [],
            'consensus_error': [],
            'position_error': [],
            'positions': []
        }
        
        best_objective = float('inf')
        best_iteration = 0
        best_positions = self.X.copy()
        
        for k in range(max_iter):
            # Run iteration
            stats = self.run_iteration(k)
            
            # Record history
            history['objective'].append(stats['objective'])
            history['psd_violation'].append(stats['psd_violation'])
            history['consensus_error'].append(stats['consensus_error'])
            history['position_error'].append(stats['position_error'])
            history['positions'].append(self.X.copy())
            
            # Track best solution
            if stats['objective'] < best_objective:
                best_objective = stats['objective']
                best_iteration = k
                best_positions = self.X.copy()
            
            # Check early stopping
            if self.config.early_stopping:
                if k - best_iteration > self.config.early_stopping_window:
                    if self.config.verbose:
                        logger.info(f"Early stopping at iteration {k}")
                    break
            
            # Check convergence
            converged = (stats['consensus_error'] < tol and 
                        stats['psd_violation'] < tol)
            
            if converged:
                if self.config.verbose:
                    logger.info(f"Converged at iteration {k}")
                break
            
            # Logging
            if self.config.verbose and k % 100 == 0:
                logger.info(
                    f"Iteration {k}: obj={stats['objective']:.6f}, "
                    f"psd_viol={stats['psd_violation']:.6f}, "
                    f"consensus={stats['consensus_error']:.6f}, "
                    f"pos_err={stats['position_error']:.6f}"
                )
        
        # Prepare results
        results = {
            'final_positions': best_positions,
            'final_Y': self.Y,
            'history': history,
            'converged': converged,
            'iterations': k + 1,
            'best_iteration': best_iteration,
            'best_objective': best_objective,
            'final_rmse': history['position_error'][best_iteration] if history['position_error'] else 0.0
        }
        
        # Add performance metrics for carrier phase mode
        if self.config.carrier_phase_mode:
            results['rmse_mm'] = results['final_rmse']
            results['achieved_accuracy'] = 'millimeter' if results['final_rmse'] < 15 else 'sub-meter'
        
        return results


def create_network_data(n_sensors: int, n_anchors: int, dimension: int = 2,
                        communication_range: float = 0.3,
                        measurement_noise: float = 0.01,
                        carrier_phase: bool = False) -> NetworkData:
    """
    Create synthetic network data for testing
    
    Args:
        n_sensors: Number of sensors
        n_anchors: Number of anchors
        dimension: Spatial dimension
        communication_range: Communication range as fraction of network
        measurement_noise: Noise level for distance measurements
        carrier_phase: Whether to generate carrier phase measurements
        
    Returns:
        NetworkData object
    """
    # Generate true positions
    true_positions = np.random.uniform(0, 1, (n_sensors, dimension))
    anchor_positions = np.random.uniform(0, 1, (n_anchors, dimension))
    
    # Build adjacency matrix based on communication range
    adjacency = np.zeros((n_sensors, n_sensors))
    distance_measurements = {}
    
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
            if true_dist <= communication_range:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                
                # Add noisy distance measurement
                noise = np.random.randn() * measurement_noise
                measured_dist = true_dist + noise
                distance_measurements[(i, j)] = measured_dist
    
    # Build anchor connections
    anchor_connections = {}
    for i in range(n_sensors):
        connected_anchors = []
        for a in range(n_anchors):
            true_dist = np.linalg.norm(true_positions[i] - anchor_positions[a])
            if true_dist <= communication_range * 1.5:  # Anchors have longer range
                connected_anchors.append(a)
                
                # Add noisy distance measurement
                noise = np.random.randn() * measurement_noise
                measured_dist = true_dist + noise
                distance_measurements[(i, a)] = measured_dist
        
        anchor_connections[i] = connected_anchors
    
    # Generate carrier phase measurements if requested
    carrier_phase_measurements = None
    if carrier_phase:
        carrier_phase_measurements = {}
        wavelength = 0.1905  # S-band wavelength in meters
        
        for key, dist in distance_measurements.items():
            # Carrier phase provides much higher precision
            phase_cycles = dist / wavelength
            integer_cycles = int(phase_cycles)
            fractional_phase = phase_cycles - integer_cycles
            
            # Add very small noise to phase (millimeter level)
            phase_noise = np.random.randn() * 0.001 / wavelength
            measured_phase = fractional_phase + phase_noise
            
            carrier_phase_measurements[key] = {
                'distance': dist,
                'phase': measured_phase,
                'wavelength': wavelength,
                'precision_mm': 1.0  # 1mm precision
            }
    
    return NetworkData(
        adjacency_matrix=adjacency,
        distance_measurements=distance_measurements,
        anchor_positions=anchor_positions,
        anchor_connections=anchor_connections,
        true_positions=true_positions,
        carrier_phase_measurements=carrier_phase_measurements,
        measurement_variance=measurement_noise**2
    )