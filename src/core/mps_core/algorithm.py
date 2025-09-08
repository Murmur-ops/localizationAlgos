"""
Simplified MPS Algorithm Implementation
Core algorithm from the paper without unnecessary complexity
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations


@dataclass  
class CarrierPhaseConfig:
    """Configuration for carrier phase synchronization (Nanzer approach)"""
    enable: bool = False
    frequency_ghz: float = 2.4  # S-band frequency
    phase_noise_milliradians: float = 1.0  # Achievable with RF hardware
    frequency_stability_ppb: float = 0.1  # OCXO stability
    coarse_time_accuracy_ns: float = 1.0  # For ambiguity resolution
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength in meters"""
        c = 299792458  # Speed of light m/s
        return c / (self.frequency_ghz * 1e9)
    
    @property
    def ranging_accuracy_m(self) -> float:
        """Expected ranging accuracy in meters"""
        phase_noise_rad = self.phase_noise_milliradians / 1000
        return (phase_noise_rad / (2 * np.pi)) * self.wavelength


@dataclass
class MPSConfig:
    """Configuration for MPS algorithm"""
    n_sensors: int = 30
    n_anchors: int = 6
    communication_range: float = 0.3
    noise_factor: float = 0.05
    gamma: float = 0.99
    alpha: float = 1.0
    max_iterations: int = 500
    tolerance: float = 1e-5
    dimension: int = 2
    seed: Optional[int] = 42
    carrier_phase: Optional[CarrierPhaseConfig] = None
    

@dataclass
class MPSState:
    """State variables for MPS algorithm"""
    positions: Dict[int, np.ndarray]
    X: np.ndarray  # Primal variable (2n x d)
    Y: np.ndarray  # Consensus variable (2n x d)
    U: np.ndarray  # Dual variable (2n x d)
    iteration: int = 0
    converged: bool = False
    

class MPSAlgorithm:
    """
    Simplified Matrix-Parametrized Proximal Splitting algorithm
    Implements the core algorithm from the paper
    """
    
    def __init__(self, config: MPSConfig):
        """
        Initialize MPS algorithm with configuration
        
        Args:
            config: Algorithm configuration
        """
        self.config = config
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Network data
        self.true_positions = None
        self.anchor_positions = None
        self.distance_measurements = {}
        self.anchor_distances = {}
        self.adjacency = None
        self.Z_matrix = None
        
    def generate_network(self):
        """Generate synthetic network for testing"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Generate random true positions
        self.true_positions = {}
        for i in range(n):
            self.true_positions[i] = np.random.uniform(0, 1, d)
        
        # Generate anchor positions (well-distributed)
        if self.config.n_anchors > 0:
            if d == 2 and self.config.n_anchors >= 4:
                # Place anchors at corners for 2D
                self.anchor_positions = np.array([
                    [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
                ])
                # Add more anchors if needed
                for i in range(4, self.config.n_anchors):
                    self.anchor_positions = np.vstack([
                        self.anchor_positions,
                        np.random.uniform(0.2, 0.8, d)
                    ])
                self.anchor_positions = self.anchor_positions[:self.config.n_anchors]
            else:
                self.anchor_positions = np.random.uniform(0, 1, (self.config.n_anchors, d))
        
        # Build adjacency matrix
        self.adjacency = MatrixOperations.build_adjacency(
            self.true_positions, 
            self.config.communication_range
        )
        
        # Generate noisy distance measurements
        self._generate_measurements()
        
        # Create consensus matrix
        self.Z_matrix = MatrixOperations.create_consensus_matrix(
            self.adjacency, 
            self.config.gamma
        )
    
    def _generate_measurements(self):
        """Generate noisy distance measurements (with optional carrier phase)"""
        n = self.config.n_sensors
        noise = self.config.noise_factor
        
        # Use carrier phase ranging if enabled
        if self.config.carrier_phase and self.config.carrier_phase.enable:
            self._generate_carrier_phase_measurements()
        else:
            # Standard noisy distance measurements
            for i in range(n):
                for j in range(i+1, n):
                    if self.adjacency[i, j] > 0:
                        true_dist = np.linalg.norm(
                            self.true_positions[i] - self.true_positions[j]
                        )
                        noisy_dist = true_dist * (1 + noise * np.random.randn())
                        self.distance_measurements[(i, j)] = max(0.01, noisy_dist)
                        self.distance_measurements[(j, i)] = self.distance_measurements[(i, j)]
            
            # Sensor-to-anchor measurements
            if self.config.n_anchors > 0:
                for i in range(n):
                    self.anchor_distances[i] = {}
                    for k in range(self.config.n_anchors):
                        true_dist = np.linalg.norm(
                            self.true_positions[i] - self.anchor_positions[k]
                        )
                        if true_dist <= self.config.communication_range:
                            noisy_dist = true_dist * (1 + noise * np.random.randn())
                            self.anchor_distances[i][k] = max(0.01, noisy_dist)
    
    def _generate_carrier_phase_measurements(self):
        """Generate carrier phase-based distance measurements (Nanzer approach)"""
        n = self.config.n_sensors
        cp_config = self.config.carrier_phase
        
        # Physical parameters
        wavelength = cp_config.wavelength
        phase_noise_rad = cp_config.phase_noise_milliradians / 1000
        coarse_accuracy_m = cp_config.coarse_time_accuracy_ns * 1e-9 * 299792458 / 2
        
        # Sensor-to-sensor measurements with carrier phase
        for i in range(n):
            for j in range(i+1, n):
                if self.adjacency[i, j] > 0:
                    true_dist = np.linalg.norm(
                        self.true_positions[i] - self.true_positions[j]
                    )
                    
                    # Carrier phase measurement (fine but ambiguous)
                    true_phase = (2 * np.pi * true_dist) / wavelength
                    measured_phase = true_phase + np.random.normal(0, phase_noise_rad)
                    wrapped_phase = np.angle(np.exp(1j * measured_phase))
                    
                    # Coarse distance for ambiguity resolution
                    # With picosecond timing (e.g., 50ps = 15mm), we can resolve ambiguity
                    coarse_dist = true_dist + np.random.normal(0, coarse_accuracy_m)
                    
                    # Integer wavelength count - with ps timing, this is unambiguous
                    n_wavelengths = np.round(coarse_dist / wavelength)
                    
                    # Fine phase measurement gives sub-mm precision
                    fine_offset = (wrapped_phase * wavelength) / (2 * np.pi)
                    
                    # Combined measurement: coarse (unambiguous) + fine (precise)
                    measured_dist = n_wavelengths * wavelength + fine_offset
                    
                    self.distance_measurements[(i, j)] = max(0.001, measured_dist)
                    self.distance_measurements[(j, i)] = self.distance_measurements[(i, j)]
        
        # Sensor-to-anchor measurements with carrier phase
        if self.config.n_anchors > 0:
            for i in range(n):
                self.anchor_distances[i] = {}
                for k in range(self.config.n_anchors):
                    true_dist = np.linalg.norm(
                        self.true_positions[i] - self.anchor_positions[k]
                    )
                    if true_dist <= self.config.communication_range:
                        # Same carrier phase approach for anchors
                        true_phase = (2 * np.pi * true_dist) / wavelength
                        measured_phase = true_phase + np.random.normal(0, phase_noise_rad)
                        wrapped_phase = np.angle(np.exp(1j * measured_phase))
                        
                        # With picosecond timing, no ambiguity
                        coarse_dist = true_dist + np.random.normal(0, coarse_accuracy_m)
                        n_wavelengths = np.round(coarse_dist / wavelength)
                        
                        # Fine phase for sub-mm precision
                        fine_offset = (wrapped_phase * wavelength) / (2 * np.pi)
                        measured_dist = n_wavelengths * wavelength + fine_offset
                        self.anchor_distances[i][k] = max(0.001, measured_dist)
    
    def initialize_state(self) -> MPSState:
        """Initialize algorithm state"""
        n = self.config.n_sensors
        d = self.config.dimension
        
        # Initialize positions
        positions = {}
        for i in range(n):
            if i in self.anchor_distances and len(self.anchor_distances[i]) > 0:
                # Initialize near anchors if available
                anchor_ids = list(self.anchor_distances[i].keys())
                positions[i] = np.mean(self.anchor_positions[anchor_ids], axis=0)
                positions[i] += 0.1 * np.random.randn(d)  # Small perturbation
            else:
                # Random initialization
                positions[i] = np.random.uniform(0, 1, d)
        
        # Initialize algorithm variables (2-block structure)
        X = np.zeros((2*n, d))
        Y = np.zeros((2*n, d))
        
        for i in range(n):
            X[i] = positions[i]
            X[i + n] = positions[i]
            Y[i] = positions[i]
            Y[i + n] = positions[i]
        
        return MPSState(
            positions=positions,
            X=X,
            Y=Y,
            U=np.zeros((2*n, d))
        )
    
    def prox_f(self, state: MPSState) -> np.ndarray:
        """
        Apply proximal operator for distance constraints
        
        Args:
            state: Current algorithm state
            
        Returns:
            Updated X variable
        """
        n = self.config.n_sensors
        X_new = state.X.copy()
        
        for i in range(n):
            # Apply distance constraints
            position = X_new[i]
            
            # Sensor-to-sensor constraints
            for j in range(n):
                if (i, j) in self.distance_measurements:
                    measured_dist = self.distance_measurements[(i, j)]
                    position = ProximalOperators.prox_distance(
                        position, X_new[j], measured_dist, 
                        alpha=self.config.alpha / 10
                    )
            
            # Sensor-to-anchor constraints
            if i in self.anchor_distances:
                for k, measured_dist in self.anchor_distances[i].items():
                    position = ProximalOperators.prox_distance(
                        position, self.anchor_positions[k], measured_dist,
                        alpha=self.config.alpha / 5
                    )
            
            # Update both blocks
            X_new[i] = position
            X_new[i + n] = position
            
            # Box constraint to keep positions bounded
            X_new[i] = ProximalOperators.prox_box_constraint(X_new[i], -0.5, 1.5)
            X_new[i + n] = X_new[i]
        
        return X_new
    
    def compute_objective(self, state: MPSState) -> float:
        """Compute objective function (distance error)"""
        total_error = 0.0
        count = 0
        
        # Sensor-to-sensor errors
        for (i, j), measured_dist in self.distance_measurements.items():
            if i < j:
                actual_dist = np.linalg.norm(state.positions[i] - state.positions[j])
                total_error += (actual_dist - measured_dist) ** 2
                count += 1
        
        # Sensor-to-anchor errors
        for i, anchor_dists in self.anchor_distances.items():
            for k, measured_dist in anchor_dists.items():
                actual_dist = np.linalg.norm(
                    state.positions[i] - self.anchor_positions[k]
                )
                total_error += (actual_dist - measured_dist) ** 2
                count += 1
        
        return np.sqrt(total_error / max(count, 1))
    
    def compute_rmse(self, state: MPSState) -> float:
        """Compute RMSE vs true positions"""
        if self.true_positions is None:
            return 0.0
        
        errors = []
        for i in range(self.config.n_sensors):
            error = np.linalg.norm(state.positions[i] - self.true_positions[i])
            errors.append(error ** 2)
        
        return np.sqrt(np.mean(errors))
    
    def run(self) -> Dict:
        """
        Run MPS algorithm
        
        Returns:
            Dictionary with results
        """
        # Ensure Z_matrix is initialized
        if self.Z_matrix is None:
            # Build a simple consensus matrix if not already done
            n = self.config.n_sensors
            if not hasattr(self, 'adjacency') or self.adjacency is None:
                # Create fully connected graph if no adjacency specified
                self.adjacency = np.ones((n, n)) - np.eye(n)
            self.Z_matrix = MatrixOperations.create_consensus_matrix(
                self.adjacency, 
                self.config.gamma
            )
        
        # Initialize
        state = self.initialize_state()
        
        # Track metrics
        objective_history = []
        rmse_history = []
        
        # Main iteration loop
        for iteration in range(self.config.max_iterations):
            # Store previous X for convergence check
            X_old = state.X.copy()
            
            # Step 1: Proximal step (distance constraints)
            state.X = self.prox_f(state)
            
            # Step 2: Consensus step via matrix multiplication
            state.Y = self.Z_matrix @ state.X
            
            # Step 3: Dual update
            state.U = state.U + self.config.alpha * (state.X - state.Y)
            
            # Step 4: Extract position estimates
            n = self.config.n_sensors
            for i in range(n):
                # Average the two blocks
                state.positions[i] = (state.Y[i] + state.Y[i + n]) / 2
            
            # Track metrics periodically
            if iteration % 10 == 0:
                obj = self.compute_objective(state)
                objective_history.append(obj)
                
                if self.true_positions is not None:
                    rmse = self.compute_rmse(state)
                    rmse_history.append(rmse)
                
                # Check convergence
                change = np.linalg.norm(state.X - X_old) / (np.linalg.norm(X_old) + 1e-10)
                if change < self.config.tolerance:
                    state.converged = True
                    state.iteration = iteration
                    break
        
        if not state.converged:
            state.iteration = self.config.max_iterations
        
        # Final metrics
        final_objective = self.compute_objective(state)
        final_rmse = self.compute_rmse(state) if self.true_positions else None
        
        return {
            'converged': state.converged,
            'iterations': state.iteration,
            'final_objective': final_objective,
            'final_rmse': final_rmse,
            'objective_history': objective_history,
            'rmse_history': rmse_history,
            'final_positions': dict(state.positions),
            'config': self.config
        }