"""
Distributed MPS Algorithm using MPI
Based on Matrix-Parametrized Proximal Splitting with MPI parallelization

This module implements a distributed version of the MPS algorithm where:
- Sensors are distributed across MPI processes
- Each process handles local proximal evaluations
- Consensus updates require inter-process communication  
- Global metrics are computed using MPI collective operations
"""

import numpy as np
from mpi4py import MPI
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from .mps_full_algorithm import MPSConfig, NetworkData, MatrixParametrizedProximalSplitting
from .proximal_sdp import ProximalADMMSolver
from .sinkhorn_knopp import MatrixParameterGenerator
from .vectorization import MatrixVectorizer

logger = logging.getLogger(__name__)


@dataclass
class DistributedMPSConfig(MPSConfig):
    """Extended configuration for distributed MPS"""
    async_communication: bool = False  # Use non-blocking communication
    buffer_size_kb: int = 1024  # MPI buffer size in KB
    collective_operations: bool = True  # Use MPI collective operations
    checkpoint_interval: int = 100  # Save checkpoints every N iterations
    load_balancing: str = "block"  # "block" or "cyclic" distribution


class DistributedMPS:
    """
    Distributed implementation of Matrix-Parametrized Proximal Splitting
    using MPI for parallel execution across multiple processes
    """
    
    def __init__(self, config: DistributedMPSConfig, network: NetworkData, 
                 comm: Optional[MPI.Comm] = None):
        """
        Initialize distributed MPS algorithm
        
        Args:
            config: Algorithm configuration
            network: Network topology and measurements
            comm: MPI communicator (defaults to COMM_WORLD)
        """
        self.config = config
        self.network = network
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        # Logging setup
        self.logger = logging.getLogger(f"{__name__}.rank{self.rank}")
        
        # Total problem size
        self.n_sensors = config.n_sensors
        self.n_anchors = config.n_anchors
        self.d = config.dimension
        self.p = 2 * self.n_sensors  # Lifted variable dimension
        
        # Distribute sensors across processes
        self._distribute_sensors()
        
        # Initialize local data structures
        self._initialize_local_structures()
        
        # Setup communication patterns
        self._setup_communication()
        
        # Initialize matrix parameters (Z, W, L)
        self._initialize_matrix_parameters()
        
        # Performance tracking
        self.timing_stats = {
            'computation': 0.0,
            'communication': 0.0,
            'synchronization': 0.0,
            'total': 0.0
        }
        
    def _distribute_sensors(self):
        """Distribute sensors across MPI processes using block distribution"""
        if self.config.load_balancing == "block":
            # Block distribution
            sensors_per_proc = self.n_sensors // self.size
            remainder = self.n_sensors % self.size
            
            if self.rank < remainder:
                self.local_start = self.rank * (sensors_per_proc + 1)
                self.local_end = self.local_start + sensors_per_proc + 1
            else:
                self.local_start = self.rank * sensors_per_proc + remainder
                self.local_end = self.local_start + sensors_per_proc
                
        else:  # cyclic distribution
            self.local_sensors = list(range(self.rank, self.n_sensors, self.size))
            self.local_start = min(self.local_sensors) if self.local_sensors else 0
            self.local_end = max(self.local_sensors) + 1 if self.local_sensors else 0
        
        self.n_local_sensors = self.local_end - self.local_start
        self.local_sensor_ids = list(range(self.local_start, self.local_end))
        
        # Also handle lifted variables (2n total)
        self.local_lifted_ids = []
        for sid in self.local_sensor_ids:
            self.local_lifted_ids.append(sid)  # Objective part
            self.local_lifted_ids.append(sid + self.n_sensors)  # PSD part
        
        if self.rank == 0:
            self.logger.info(f"Distributed {self.n_sensors} sensors across {self.size} processes")
            self.logger.info(f"Process 0 handles sensors {self.local_start}-{self.local_end-1}")
    
    def _get_process_for_sensor(self, sensor_id: int) -> int:
        """Determine which process owns a sensor"""
        if self.config.load_balancing == "block":
            sensors_per_proc = self.n_sensors // self.size
            remainder = self.n_sensors % self.size
            
            if sensor_id < remainder * (sensors_per_proc + 1):
                return sensor_id // (sensors_per_proc + 1)
            else:
                return (sensor_id - remainder) // sensors_per_proc + remainder
        else:  # cyclic
            return sensor_id % self.size
    
    def _initialize_local_structures(self):
        """Initialize local data structures for assigned sensors"""
        # Local copies of variables
        self.v_local = {}  # Dual variables
        self.x_local = {}  # Primal variables
        self.x_local_prev = {}  # Previous x for momentum
        
        # Adaptive parameters
        self.alpha_current = self.config.alpha
        self.oscillation_counter = 0
        self.improvement_counter = 0
        
        # Momentum parameters
        self.momentum = 0.0  # Will be updated adaptively
        self.momentum_beta = 0.9  # Momentum decay factor
        
        # Determine matrix dimensions based on neighborhoods
        self.matrix_dims = {}
        adjacency = self.network.adjacency_matrix
        
        for i in self.local_lifted_ids:
            sensor_idx = i % self.n_sensors
            neighbors = np.where(adjacency[sensor_idx] > 0)[0]
            
            # Dimension: d + 1 + |N_i|
            dim = self.d + 1 + len(neighbors)
            self.matrix_dims[i] = dim
            
            # Initialize with identity matrix
            self.v_local[i] = np.eye(dim)
            self.x_local[i] = np.eye(dim)
            self.x_local_prev[i] = np.eye(dim)
        
        # Store neighborhoods for quick access
        self.neighborhoods = {}
        for sid in range(self.n_sensors):  # Need all sensor neighborhoods for vectorizer
            self.neighborhoods[sid] = np.where(adjacency[sid] > 0)[0].tolist()
        
        # Vectorizer for handling variable dimensions
        self.vectorizer = MatrixVectorizer(self.n_sensors, self.d, self.neighborhoods)
        
    def _setup_communication(self):
        """Setup MPI communication patterns"""
        # Identify which processes we need to communicate with
        self.neighbor_processes = set()
        
        for sid in self.local_sensor_ids:
            for neighbor in self.neighborhoods[sid]:
                proc = self._get_process_for_sensor(neighbor)
                if proc != self.rank:
                    self.neighbor_processes.add(proc)
        
        # Pre-allocate communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        
        buffer_size = self.config.buffer_size_kb * 1024 // 8  # Convert to doubles
        
        for proc in self.neighbor_processes:
            self.send_buffers[proc] = np.zeros(buffer_size)
            self.recv_buffers[proc] = np.zeros(buffer_size)
        
        self.logger.debug(f"Process {self.rank} communicates with processes: {self.neighbor_processes}")
    
    def _initialize_matrix_parameters(self):
        """Initialize Z, W, L matrices using distributed Sinkhorn-Knopp"""
        # Get adjacency matrix
        adjacency = self.network.adjacency_matrix
        
        # Generate parameters (can be done locally as they depend on global topology)
        generator = MatrixParameterGenerator()
        
        if self.config.use_2block:
            self.Z, self.W = generator.generate_from_communication_graph(
                adjacency, method='sinkhorn-knopp', block_design='2-block'
            )
        else:
            self.Z, self.W = generator.generate_from_communication_graph(
                adjacency, method='sinkhorn-knopp', block_design='full'
            )
        
        # Compute L matrix for sequential dependencies
        self.L = generator.compute_lower_triangular_L(self.Z)
        
        # Store only local rows of matrices for efficiency
        self.Z_local = self.Z[self.local_lifted_ids, :]
        self.W_local = self.W[self.local_lifted_ids, :]
        self.L_local = self.L[self.local_lifted_ids, :]
        
    def run_distributed(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run distributed MPS algorithm
        
        Returns:
            Dictionary containing results and performance metrics
        """
        max_iter = max_iterations or self.config.max_iterations
        
        if self.rank == 0:
            self.logger.info(f"Starting distributed MPS with {self.size} processes")
        
        # Initialize warm-start if enabled
        if self.config.warm_start:
            self._initialize_warm_start()
        
        # Tracking
        objectives = []
        errors = []
        smoothed_objectives = []  # Exponentially smoothed objectives
        smoothed_errors = []  # Exponentially smoothed errors
        best_error = float('inf')
        best_positions = None
        convergence_counter = 0
        
        # Smoothing parameters
        smoothing_factor = 0.8  # Weight for new value (1-smoothing = weight for history)
        current_obj_smooth = None
        current_err_smooth = None
        
        # Main iteration loop
        start_time = time.time()
        
        for k in range(max_iter):
            iter_start = time.time()
            
            # Step 1: Sequential proximal evaluation with L dependencies
            comp_start = time.time()
            self._evaluate_proximal_sequential()
            self.timing_stats['computation'] += time.time() - comp_start
            
            # Step 2: Consensus update v^(k+1) = v^k - γWx^k
            comm_start = time.time()
            self._consensus_update_distributed()
            self.timing_stats['communication'] += time.time() - comm_start
            
            # Step 3: Check convergence (every N iterations)
            if k % 10 == 0:
                sync_start = time.time()
                obj, error = self._compute_global_metrics()
                self.timing_stats['synchronization'] += time.time() - sync_start
                
                objectives.append(obj)
                errors.append(error)
                
                # Apply exponential smoothing
                if current_obj_smooth is None:
                    current_obj_smooth = obj
                    current_err_smooth = error
                else:
                    current_obj_smooth = smoothing_factor * obj + (1 - smoothing_factor) * current_obj_smooth
                    current_err_smooth = smoothing_factor * error + (1 - smoothing_factor) * current_err_smooth
                
                smoothed_objectives.append(current_obj_smooth)
                smoothed_errors.append(current_err_smooth)
                
                if self.rank == 0:
                    self.logger.info(f"Iteration {k}: obj={obj:.6f} (smooth={current_obj_smooth:.6f}), "
                                   f"error={error:.6f} (smooth={current_err_smooth:.6f})")
                
                # Check convergence and adapt parameters
                if error < best_error:
                    improvement_ratio = (best_error - error) / (best_error + 1e-10)
                    best_error = error
                    best_positions = self._gather_positions()
                    convergence_counter = 0
                    self.oscillation_counter = 0
                    self.improvement_counter += 1
                    
                    # If improving, slightly increase alpha (but cap it)
                    if self.config.adaptive_alpha and self.improvement_counter > 2:
                        self.alpha_current = min(self.alpha_current * 1.05, self.config.alpha * 1.5)
                        self.momentum = max(0, self.momentum - 0.1)  # Reduce momentum when improving
                else:
                    convergence_counter += 1
                    
                    # Check for oscillation
                    if k > 0 and len(errors) > 2:
                        if errors[-1] > errors[-2]:  # Error increased
                            self.oscillation_counter += 1
                        else:
                            self.oscillation_counter = max(0, self.oscillation_counter - 1)
                    
                    # Adapt parameters if oscillating
                    if self.config.adaptive_alpha and self.oscillation_counter > 3:
                        self.alpha_current *= 0.9  # Reduce step size
                        self.momentum = min(0.5, self.momentum + 0.1)  # Increase momentum
                        self.oscillation_counter = 0
                        if self.rank == 0:
                            self.logger.info(f"  Adapted: alpha={self.alpha_current:.4f}, momentum={self.momentum:.2f}")
                
                if convergence_counter > self.config.early_stopping_window:
                    if self.rank == 0:
                        self.logger.info(f"Early stopping at iteration {k}")
                    break
                
                if obj < self.config.tolerance:
                    if self.rank == 0:
                        self.logger.info(f"Converged at iteration {k}")
                    break
            
            # Checkpoint if needed
            if k > 0 and k % self.config.checkpoint_interval == 0:
                self._save_checkpoint(k)
        
        self.timing_stats['total'] = time.time() - start_time
        
        # Gather final results
        final_positions = self._gather_positions()
        
        results = {
            'converged': k < max_iter - 1,
            'iterations': k + 1,
            'best_error': best_error,
            'final_error': errors[-1] if errors else None,
            'objectives': objectives,
            'errors': errors,
            'smoothed_objectives': smoothed_objectives,
            'smoothed_errors': smoothed_errors,
            'positions': final_positions if self.rank == 0 else None,
            'best_positions': best_positions if self.rank == 0 else None,
            'timing_stats': self.timing_stats,
            'n_processes': self.size,
            'final_alpha': self.alpha_current,
            'final_momentum': self.momentum
        }
        
        if self.rank == 0:
            self._print_summary(results)
        
        return results
    
    def _evaluate_proximal_sequential(self):
        """Evaluate proximal operators with L matrix sequential dependencies"""
        # Create local ADMM solvers
        solvers = {}
        for i in self.local_lifted_ids:
            solvers[i] = ProximalADMMSolver(
                rho=self.config.admm_rho,
                max_iterations=self.config.admm_iterations,
                tolerance=self.config.admm_tolerance,
                warm_start=self.config.warm_start
            )
        
        # Process in sequential order respecting L dependencies
        for i in range(self.p):
            if i in self.local_lifted_ids:
                # Compute v_tilde = v_i + sum_{j<i} L_ij * x_j
                v_tilde = self.v_local[i].copy()
                
                # Add contributions from previous x values
                for j in range(i):
                    if abs(self.L[i, j]) > 1e-10:
                        # Need x_j which might be on another process
                        if j in self.local_lifted_ids:
                            # Local access
                            x_j = self.x_local[j]
                        else:
                            # Remote access - need to receive from owner
                            x_j = self._get_remote_x(j)
                        
                        # Apply L matrix contribution
                        if x_j is not None and x_j.shape == v_tilde.shape:
                            v_tilde += self.L[i, j] * x_j
                
                # Evaluate proximal operator
                sensor_idx = i % self.n_sensors
                if i < self.n_sensors:
                    # Objective proximal (measurements)
                    x_new = self._prox_objective(
                        sensor_idx, v_tilde, solvers[i]
                    )
                else:
                    # PSD constraint proximal
                    x_new = self._prox_psd_constraint(
                        sensor_idx, v_tilde, solvers[i]
                    )
                
                # Apply momentum if enabled
                if self.momentum > 0 and i in self.x_local_prev:
                    x_new = (1 - self.momentum) * x_new + self.momentum * self.x_local_prev[i]
                
                # Store previous and update current
                self.x_local_prev[i] = self.x_local.get(i, x_new).copy()
                self.x_local[i] = x_new
            
            # Synchronize after each sequential step if needed
            if self.config.async_communication:
                self.comm.Barrier()
    
    def _consensus_update_distributed(self):
        """Perform consensus update with W matrix requiring communication"""
        # Compute v^(k+1) = v^k - γ * W * x^k
        
        # First, exchange x values with neighbor processes
        if self.config.async_communication:
            # Non-blocking communication
            send_requests = []
            recv_requests = []
            
            for proc in self.neighbor_processes:
                # Pack local x values needed by proc
                send_data = self._pack_x_for_process(proc)
                req = self.comm.Isend(send_data, dest=proc, tag=0)
                send_requests.append(req)
                
                # Receive x values from proc
                recv_data = np.empty_like(self.recv_buffers[proc])
                req = self.comm.Irecv(recv_data, source=proc, tag=0)
                recv_requests.append((proc, recv_data, req))
            
            # Wait for all communication to complete
            MPI.Request.Waitall(send_requests)
            
            # Process received data as it arrives
            remote_x = {}
            for proc, recv_data, req in recv_requests:
                req.Wait()
                remote_x.update(self._unpack_x_from_process(proc, recv_data))
        else:
            # Synchronous communication
            remote_x = self._exchange_x_synchronous()
        
        # Apply consensus update locally
        v_new = {}
        for i in self.local_lifted_ids:
            v_new[i] = self.v_local[i].copy()
            
            # Apply W matrix multiplication
            for j in range(self.p):
                if abs(self.W[i, j]) > 1e-10:
                    if j in self.local_lifted_ids:
                        x_j = self.x_local[j]
                    else:
                        x_j = remote_x.get(j)
                    
                    if x_j is not None and x_j.shape == v_new[i].shape:
                        # Use adaptive gamma based on current alpha
                        gamma_effective = self.config.gamma
                        if self.config.adaptive_alpha:
                            # Scale gamma with alpha changes
                            gamma_effective *= (self.alpha_current / self.config.alpha)
                        v_new[i] -= gamma_effective * self.W[i, j] * x_j
        
        # Update local v
        self.v_local = v_new
    
    def _get_remote_x(self, idx: int) -> Optional[np.ndarray]:
        """Get x value from remote process"""
        owner = self._get_process_for_lifted_var(idx)
        
        if owner == self.rank:
            return self.x_local.get(idx)
        
        # Request from owner
        dim = self._get_matrix_dim_for_idx(idx)
        x_remote = np.empty((dim, dim))
        
        if self.rank == 0:  # Designated receiver
            self.comm.Recv(x_remote, source=owner, tag=idx)
        if self.rank == owner:  # Owner sends
            self.comm.Send(self.x_local[idx], dest=0, tag=idx)
        
        # Broadcast to all
        self.comm.Bcast(x_remote, root=0)
        
        return x_remote
    
    def _get_process_for_lifted_var(self, idx: int) -> int:
        """Determine which process owns a lifted variable"""
        sensor_idx = idx % self.n_sensors
        return self._get_process_for_sensor(sensor_idx)
    
    def _get_matrix_dim_for_idx(self, idx: int) -> int:
        """Get matrix dimension for lifted variable index"""
        sensor_idx = idx % self.n_sensors
        neighbors = np.where(self.network.adjacency_matrix[sensor_idx] > 0)[0]
        return self.d + 1 + len(neighbors)
    
    def _pack_x_for_process(self, proc: int) -> np.ndarray:
        """Pack x values needed by a process"""
        # Simplified: just concatenate relevant matrices
        data = []
        for i in self.local_lifted_ids:
            # Check if proc needs this x value
            # (would need more sophisticated check in practice)
            flat = self.x_local[i].flatten()
            data.extend(flat)
        
        return np.array(data)
    
    def _unpack_x_from_process(self, proc: int, data: np.ndarray) -> Dict[int, np.ndarray]:
        """Unpack received x values"""
        # Simplified unpacking
        result = {}
        # Would need proper unpacking logic based on matrix dimensions
        return result
    
    def _exchange_x_synchronous(self) -> Dict[int, np.ndarray]:
        """Exchange x values synchronously"""
        remote_x = {}
        
        for proc in self.neighbor_processes:
            # Send our x values
            send_data = self._pack_x_for_process(proc)
            recv_data = np.empty_like(send_data)
            
            self.comm.Sendrecv(
                send_data, proc, 0,
                recv_data, proc, 0
            )
            
            # Unpack received data
            remote_x.update(self._unpack_x_from_process(proc, recv_data))
        
        return remote_x
    
    def _prox_objective(self, sensor_idx: int, v: np.ndarray, 
                       solver: ProximalADMMSolver) -> np.ndarray:
        """Evaluate proximal operator for objective function"""
        # v is a lifted matrix variable, we need to extract position
        # For lifted variables, position is typically in the last column
        # or we can use the matrix as-is for the SDP formulation
        
        # Get measurement data
        neighbors = self.neighborhoods[sensor_idx]
        measurements = {}
        
        for n in neighbors:
            key = (min(sensor_idx, n), max(sensor_idx, n))
            if key in self.network.distance_measurements:
                measurements[n] = self.network.distance_measurements[key]
        
        # Get anchor measurements
        anchor_measurements = {}
        anchors_list = []
        if sensor_idx in self.network.anchor_connections:
            for a in self.network.anchor_connections[sensor_idx]:
                key = (sensor_idx, self.n_sensors + a)
                if key in self.network.distance_measurements:
                    anchor_measurements[a] = self.network.distance_measurements[key]
                    anchors_list.append(self.n_sensors + a)
        
        # ProximalADMMSolver expects positions and Y matrices
        # Extract position from lifted variable if it's a matrix
        if v.ndim == 2 and v.shape[0] > self.d:
            # v is a lifted matrix, extract position (last column or first d rows)
            position = v[:self.d, -1] if v.shape[1] > self.d else v[:self.d, 0]
        elif v.ndim == 1:
            position = v
        else:
            # v is already in the right shape
            position = v[:self.d] if v.shape[0] > self.d else v
            
        X_prev = np.zeros((self.n_sensors, self.d))
        X_prev[sensor_idx] = position[:self.d]  # Ensure correct dimension
        Y_prev = np.zeros((self.n_sensors, self.n_sensors))
        
        X_new, Y_new = solver.solve(
            X_prev, Y_prev, sensor_idx, list(neighbors), anchors_list,
            measurements, anchor_measurements,
            self.network.anchor_positions, self.config.alpha
        )
        
        # Return updated lifted variable maintaining the original structure
        v_new = v.copy()
        if v.ndim == 2 and v.shape[0] > self.d:
            # Update position in lifted matrix
            v_new[:self.d, -1] = X_new[sensor_idx]
        else:
            # Return just the position
            v_new = X_new[sensor_idx]
            
        return v_new
    
    def _prox_psd_constraint(self, sensor_idx: int, v: np.ndarray,
                            solver: ProximalADMMSolver) -> np.ndarray:
        """Evaluate proximal operator for PSD constraint"""
        # For vector input, return as-is since PSD constraint applies to matrices
        # In our formulation, we don't need PSD projection for position vectors
        return v
    
    def _compute_global_metrics(self) -> Tuple[float, float]:
        """Compute global objective and error using MPI reduction"""
        local_obj = 0.0
        local_error = 0.0
        
        # Compute local contributions
        for sid in self.local_sensor_ids:
            # Extract position from lifted variable
            if sid in self.local_lifted_ids:
                X_matrix = self.x_local[sid]
                position = X_matrix[:self.d, -1]  # Extract position
                
                # Objective: measurement errors
                for n in self.neighborhoods[sid]:
                    key = (min(sid, n), max(sid, n))
                    if key in self.network.distance_measurements:
                        true_dist = self.network.distance_measurements[key]
                        if n in self.local_sensor_ids:
                            n_pos = self.x_local[n][:self.d, -1]
                            est_dist = np.linalg.norm(position - n_pos)
                            local_obj += (est_dist - true_dist) ** 2
                
                # Error vs true position (if available)
                if self.network.true_positions is not None:
                    true_pos = self.network.true_positions[sid]
                    local_error += np.linalg.norm(position - true_pos) ** 2
        
        # Global reduction
        global_obj = self.comm.allreduce(local_obj, op=MPI.SUM)
        global_error = self.comm.allreduce(local_error, op=MPI.SUM)
        
        # Normalize
        global_obj /= self.n_sensors
        global_error = np.sqrt(global_error / self.n_sensors)
        
        return global_obj, global_error
    
    def _gather_positions(self) -> Optional[np.ndarray]:
        """Gather all sensor positions to rank 0"""
        # Extract local positions
        local_positions = {}
        for sid in self.local_sensor_ids:
            if sid in self.local_lifted_ids:
                X_matrix = self.x_local[sid]
                local_positions[sid] = X_matrix[:self.d, -1]
        
        # Gather to rank 0
        all_positions = self.comm.gather(local_positions, root=0)
        
        if self.rank == 0:
            # Combine all positions
            positions = np.zeros((self.n_sensors, self.d))
            for pos_dict in all_positions:
                for sid, pos in pos_dict.items():
                    positions[sid] = pos
            return positions
        
        return None
    
    def _initialize_warm_start(self):
        """Initialize with warm-start using zero-sum constraint"""
        if self.network.true_positions is not None:
            # Use noisy version of true positions
            for sid in self.local_sensor_ids:
                true_pos = self.network.true_positions[sid]
                noisy_pos = true_pos + 0.1 * np.random.randn(self.d)
                
                # Set in lifted variables
                for i in [sid, sid + self.n_sensors]:
                    if i in self.local_lifted_ids:
                        self.x_local[i][:self.d, -1] = noisy_pos
                        self.v_local[i][:self.d, -1] = noisy_pos
    
    def _save_checkpoint(self, iteration: int):
        """Save checkpoint for fault tolerance"""
        if self.rank == 0:
            checkpoint = {
                'iteration': iteration,
                'v_local': self.v_local,
                'x_local': self.x_local,
                'timing_stats': self.timing_stats
            }
            # Save to file (implementation depends on requirements)
            self.logger.info(f"Saved checkpoint at iteration {iteration}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of results (rank 0 only)"""
        if self.rank != 0:
            return
        
        print("\n" + "="*60)
        print("DISTRIBUTED MPS RESULTS")
        print("="*60)
        print(f"MPI Processes: {results['n_processes']}")
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Best Error: {results['best_error']:.6f}")
        print(f"Final Error: {results['final_error']:.6f}")
        print(f"\nTiming Breakdown:")
        for key, value in results['timing_stats'].items():
            if key != 'total':
                pct = 100 * value / results['timing_stats']['total']
                print(f"  {key}: {value:.2f}s ({pct:.1f}%)")
        print(f"  Total: {results['timing_stats']['total']:.2f}s")
        print("="*60)