"""
FrequencyFTL: Extended FTL with frequency synchronization

Extends state vector from 3D [x, y, tau] to 4D [x, y, tau, delta_f]
for long-term clock drift compensation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from ftl.frequency_factors.frequency_factors import (
    RangeFrequencyFactor,
    FrequencyPrior,
    MultiEpochFactor,
    FrequencyConfig
)
from ftl.optimization.adaptive_lm import AdaptiveLM, AdaptiveLMConfig


class FrequencyFTLConfig(EnhancedFTLConfig):
    """Configuration for frequency-enhanced FTL"""

    def __init__(self):
        super().__init__()
        # Frequency-specific parameters
        self.enable_frequency = True
        self.frequency_prior_ppb = 10.0  # Expected frequency stability
        self.frequency_damping = 0.1  # Damping for frequency updates
        self.multi_epoch_window = 10  # Number of epochs for batch processing

        # State dimension
        self.state_dim = 4  # [x, y, tau, delta_f]


class FrequencyFTL(EnhancedFTL):
    """
    FTL with frequency synchronization capability

    Extends state vector to 4D to track clock frequency offset
    """

    def __init__(self, config: Optional[FrequencyFTLConfig] = None):
        """Initialize FrequencyFTL with extended state"""
        if config is None:
            config = FrequencyFTLConfig()

        # Set state dimension before parent init
        self.state_dim = 4

        # Call parent constructor
        super().__init__(config)

        # Frequency-specific attributes
        self.frequency_config = FrequencyConfig()
        self.measurement_history = {}  # Store past measurements
        self.timestamps = []  # Measurement timestamps

    def setup_network(self, n_nodes: int, n_anchors: int, area_size: float):
        """
        Set up the network topology

        Args:
            n_nodes: Total number of nodes
            n_anchors: Number of anchor nodes (known positions)
            area_size: Size of deployment area (square)
        """
        # Update config
        self.config.n_nodes = n_nodes
        self.config.n_anchors = n_anchors
        self.n_nodes = n_nodes
        self.n_anchors = n_anchors
        self.area_size = area_size

        # Call parent setup
        self._setup_network()

        # Create adjacency matrix (fully connected for now)
        self.adjacency_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)

        # Initialize 4D states
        self._initialize_states()

    def _initialize_states(self):
        """Initialize 4D state vectors"""
        # Get current states from parent (3D)
        if hasattr(self, 'states') and self.states is not None:
            old_states = self.states.copy()
            # Extend to 4D
            self.states = np.zeros((self.n_nodes, self.state_dim))
            # Copy position and time if they exist
            if old_states.shape[1] >= 3:
                self.states[:, :3] = old_states[:, :3]
        else:
            # Initialize fresh
            self.states = np.zeros((self.n_nodes, self.state_dim))
            # Initialize positions for anchors
            for i in range(self.n_anchors):
                if hasattr(self, 'true_positions'):
                    self.states[i, :2] = self.true_positions[i]
            # Initialize unknown positions randomly
            for i in range(self.n_anchors, self.n_nodes):
                self.states[i, :2] = np.random.rand(2) * self.area_size

        # Initialize frequency offsets to zero
        self.states[:, 3] = 0.0

    def _compute_measurement_error(self, i: int, j: int,
                                    measured_range: float,
                                    timestamp: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute error and Jacobian for frequency-aware measurement

        Returns:
            error: Scalar residual
            J_i: Jacobian w.r.t. state i (4D)
            J_j: Jacobian w.r.t. state j (4D)
        """
        # Create frequency-aware range factor
        factor = RangeFrequencyFactor(
            measured_range=measured_range,
            timestamp=timestamp,
            sigma=self.config.measurement_std
        )

        # Get states
        state_i = self.states[i]
        state_j = self.states[j]

        # Compute error
        error = factor.error(state_i, state_j)

        # Compute Jacobian
        J_i, J_j = factor.jacobian(state_i, state_j)

        return error, J_i, J_j

    def _add_frequency_prior(self, node_id: int,
                              H: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add frequency prior constraint to normal equations"""
        prior = FrequencyPrior(
            nominal_freq_ppb=0.0,
            sigma_ppb=self.config.frequency_prior_ppb
        )

        # Get state
        state = self.states[node_id]

        # Compute prior error and Jacobian
        error = prior.error(state)
        J = prior.jacobian(state)

        # Add to normal equations - H and g are for single node only
        # H is state_dim x state_dim for this node
        H += np.outer(J, J)
        g -= J * error

        return H, g

    def local_optimization_step(self, node_id: int) -> np.ndarray:
        """Extended optimization step with frequency"""
        # Build local measurement system
        H = np.zeros((self.state_dim, self.state_dim))
        g = np.zeros(self.state_dim)

        # Current timestamp (could be from system clock)
        timestamp = len(self.timestamps) * 0.1  # Example: 100ms intervals

        # Collect measurements from neighbors
        for neighbor_id in range(self.n_nodes):
            if neighbor_id == node_id:
                continue

            # Check if we have a measurement
            if self.adjacency_matrix[node_id, neighbor_id] > 0:
                # Generate measurement (in real system, would come from hardware)
                measured_range = self._simulate_range_measurement(node_id, neighbor_id, timestamp)

                # Compute error and Jacobian
                error, J_i, J_j = self._compute_measurement_error(
                    node_id, neighbor_id, measured_range, timestamp
                )

                # Accumulate in normal equations (only local part)
                H += np.outer(J_i, J_i)
                g -= J_i * error

        # Add frequency prior
        H_prior = np.zeros_like(H)
        g_prior = np.zeros_like(g)
        H_prior, g_prior = self._add_frequency_prior(node_id, H_prior, g_prior)

        # Combine with weight
        prior_weight = 0.1
        H += prior_weight * H_prior
        g += prior_weight * g_prior

        # Add anchor constraint if applicable
        if node_id < self.n_anchors:
            # Strong prior on position for anchors
            H[:2, :2] += np.eye(2) * 1e6
            g[:2] -= 1e6 * (self.states[node_id, :2] - self.true_positions[node_id])

        # Solve using adaptive LM
        if not hasattr(self, 'lm_solver'):
            lm_config = AdaptiveLMConfig()
            self.lm_solver = AdaptiveLM(lm_config)

        # Compute update
        try:
            # Add regularization for conditioning
            H_reg = H + np.eye(self.state_dim) * 1e-6
            delta = np.linalg.solve(H_reg, -g)

            # Apply damping to frequency component
            delta[3] *= self.config.frequency_damping

        except np.linalg.LinAlgError:
            # If singular, return zero update
            delta = np.zeros(self.state_dim)

        return delta

    def _simulate_range_measurement(self, i: int, j: int, timestamp: float) -> float:
        """
        Simulate range measurement with frequency drift

        In real system, this would come from actual hardware measurements
        """
        # True geometric range
        true_range = np.linalg.norm(self.true_positions[j] - self.true_positions[i])

        # Clock contribution (including frequency drift)
        tau_i = self.states[i, 2] if len(self.states[i]) > 2 else 0.0
        tau_j = self.states[j, 2] if len(self.states[j]) > 2 else 0.0
        df_i = self.states[i, 3] if len(self.states[i]) > 3 else 0.0
        df_j = self.states[j, 3] if len(self.states[j]) > 3 else 0.0

        # Time offset with drift
        time_offset = (tau_j - tau_i) + (df_j - df_i) * timestamp
        clock_contrib = time_offset * 299792458.0 * 1e-9

        # Add measurement noise
        noise = np.random.normal(0, self.config.measurement_std)

        return true_range + clock_contrib + noise

    def compute_position_rmse(self) -> float:
        """Compute RMSE for unknown node positions"""
        errors = []
        for i in range(self.n_anchors, self.n_nodes):
            pos_est = self.states[i, :2]
            pos_true = self.true_positions[i]
            error = np.linalg.norm(pos_est - pos_true)
            errors.append(error)

        return np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0

    def run(self, n_iterations: int = 100,
            verbose: bool = True,
            save_history: bool = True) -> Dict:
        """
        Run frequency-enhanced FTL

        Args:
            n_iterations: Number of iterations
            verbose: Print progress
            save_history: Store iteration history

        Returns:
            Results dictionary with performance metrics
        """
        if save_history:
            self.position_history = []
            self.time_history = []
            self.frequency_history = []
            self.error_history = []

        for iteration in range(n_iterations):
            # Store current state
            if save_history:
                self.position_history.append(self.states[:, :2].copy())
                self.time_history.append(self.states[:, 2].copy())
                self.frequency_history.append(self.states[:, 3].copy())

            # Update timestamp
            current_time = iteration * 0.1  # 100ms per iteration
            self.timestamps.append(current_time)

            # Perform local updates for all nodes
            updates = np.zeros_like(self.states)

            for node_id in range(self.n_nodes):
                if node_id < self.n_anchors:
                    continue  # Skip anchors (known positions)

                # Get local update
                delta = self.local_optimization_step(node_id)
                updates[node_id] = delta

            # Apply updates with step size
            step_size = self.config.basic_step_size / (1 + iteration * 0.01)  # Decreasing step
            self.states += step_size * updates

            # Compute and store error
            if save_history:
                pos_error = self.compute_position_rmse()
                self.error_history.append(pos_error)

                if verbose and iteration % 10 == 0:
                    freq_std = np.std(self.states[self.n_anchors:, 3])
                    print(f"Iter {iteration:3d}: RMSE = {pos_error:.4f}m, "
                          f"Freq STD = {freq_std:.3f} ppb")

        # Compute final metrics
        results = {
            'final_rmse': self.compute_position_rmse(),
            'final_time_error': np.std(self.states[self.n_anchors:, 2]),
            'final_freq_error': np.std(self.states[self.n_anchors:, 3]),
            'position_history': self.position_history if save_history else None,
            'time_history': self.time_history if save_history else None,
            'frequency_history': self.frequency_history if save_history else None,
            'error_history': self.error_history if save_history else None
        }

        return results

    def plot_frequency_convergence(self, results: Dict):
        """Plot frequency synchronization convergence"""
        if results['frequency_history'] is None:
            print("No history available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Position RMSE over time
        ax = axes[0, 0]
        ax.semilogy(results['error_history'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Position RMSE (m)')
        ax.set_title('Position Convergence')
        ax.grid(True, alpha=0.3)

        # Time synchronization
        ax = axes[0, 1]
        time_history = np.array(results['time_history'])
        time_std = [np.std(t[self.n_anchors:]) for t in time_history]
        ax.semilogy(time_std)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time Sync Error (ns)')
        ax.set_title('Time Synchronization')
        ax.grid(True, alpha=0.3)

        # Frequency convergence
        ax = axes[1, 0]
        freq_history = np.array(results['frequency_history'])
        freq_std = [np.std(f[self.n_anchors:]) for f in freq_history]
        ax.plot(freq_std)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Frequency STD (ppb)')
        ax.set_title('Frequency Synchronization')
        ax.grid(True, alpha=0.3)

        # Final frequency distribution
        ax = axes[1, 1]
        final_freqs = freq_history[-1][self.n_anchors:]
        ax.hist(final_freqs, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Frequency Offset (ppb)')
        ax.set_ylabel('Count')
        ax.set_title(f'Final Frequency Distribution (std={np.std(final_freqs):.2f} ppb)')
        ax.grid(True, alpha=0.3)

        plt.suptitle('FrequencyFTL Convergence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig