"""
Enhanced implementation of Decentralized Sensor Network Localization
with optional OARS integration for matrix parameter generation

Based on the paper by Barkley and Bassett (2025)
"""

import numpy as np
from mpi4py import MPI
import scipy.linalg as la
import scipy.sparse as sp
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging
import json

# Try to import OARS
try:
    import oars
    import oars.matrices
    import oars.algorithms
    OARS_AVAILABLE = True
    logging.info("OARS library available for enhanced matrix parameter generation")
except ImportError:
    OARS_AVAILABLE = False
    logging.warning("OARS not available, using built-in Sinkhorn-Knopp only")

# Import our original implementation
from snl_main import SNLProblem, SensorData, DistributedSNL

logger = logging.getLogger(__name__)


class EnhancedDistributedSNL(DistributedSNL):
    """Enhanced SNL solver with OARS integration"""
    
    def __init__(self, problem: SNLProblem, use_oars: bool = True):
        super().__init__(problem)
        self.use_oars = use_oars and OARS_AVAILABLE
        
        if self.use_oars and self.rank == 0:
            logger.info("Using OARS for matrix parameter generation")
    
    def compute_matrix_parameters_oars(self, method: str = "MinSLEM") -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Compute matrix parameters using OARS library
        
        Methods:
        - MinSLEM: Minimize second-largest eigenvalue magnitude
        - MinResist: Minimize total effective resistance  
        - MaxConnectivity: Maximize algebraic connectivity (Fiedler value)
        - MinSpectralDifference: Minimize spectral difference
        """
        if not self.use_oars:
            logger.warning("OARS not available, falling back to Sinkhorn-Knopp")
            return self.compute_matrix_parameters()
        
        # Build adjacency matrix
        adjacency = np.zeros((self.problem.n_sensors, self.problem.n_sensors))
        
        for sensor_id in range(self.problem.n_sensors):
            sensor_data = self.sensor_data.get(sensor_id)
            if sensor_data:
                for neighbor in sensor_data.neighbors:
                    adjacency[sensor_id, neighbor] = 1
        
        # Ensure symmetric
        adjacency = np.maximum(adjacency, adjacency.T)
        
        # Add self-loops for Sinkhorn-Knopp compatibility
        adjacency_plus_I = adjacency + np.eye(self.problem.n_sensors)
        
        if self.rank == 0:
            # Generate matrices using OARS
            try:
                if method == "MinSLEM":
                    Z_global, W_global = oars.matrices.getMinSLEM(
                        self.problem.n_sensors,
                        sparsity_pattern=adjacency_plus_I
                    )
                elif method == "MinResist":
                    Z_global, W_global = oars.matrices.getMinResist(
                        self.problem.n_sensors,
                        sparsity_pattern=adjacency_plus_I
                    )
                elif method == "MaxConnectivity":
                    Z_global, W_global = oars.matrices.getMaxConnectivity(
                        self.problem.n_sensors,
                        sparsity_pattern=adjacency_plus_I
                    )
                elif method == "MinSpectralDifference":
                    Z_global, W_global = oars.matrices.getMinSpectralDifference(
                        self.problem.n_sensors,
                        sparsity_pattern=adjacency_plus_I
                    )
                else:
                    # For 2-Block design
                    block_size = self.problem.n_sensors // 2
                    Z_global, W_global = oars.matrices.getBlockMin(
                        self.problem.n_sensors,
                        block_size,
                        sparsity_pattern=adjacency_plus_I
                    )
                
                logger.info(f"Generated matrix parameters using OARS {method} method")
                
            except Exception as e:
                logger.warning(f"OARS matrix generation failed: {e}, falling back to Sinkhorn-Knopp")
                return self.compute_matrix_parameters()
        else:
            Z_global = None
            W_global = None
        
        # Broadcast matrices
        Z_global = self.comm.bcast(Z_global, root=0)
        W_global = self.comm.bcast(W_global, root=0)
        
        # Extract local blocks
        Z_blocks = {}
        W_blocks = {}
        
        for sensor_id in self.sensor_ids:
            # Get indices for this sensor and its neighbors
            indices = [sensor_id] + self.sensor_data[sensor_id].neighbors
            
            # Extract submatrices
            Z_blocks[sensor_id] = Z_global[np.ix_(indices, indices)]
            W_blocks[sensor_id] = W_global[np.ix_(indices, indices)]
        
        return Z_blocks, W_blocks
    
    def matrix_parametrized_splitting_oars(self, matrix_method: str = "MinSLEM") -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Enhanced MPS algorithm using OARS for matrix parameters
        """
        from proximal_operators import ProximalOperators
        prox_ops = ProximalOperators(self.problem)
        
        # Compute matrix parameters using OARS
        if self.use_oars:
            Z_blocks, W_blocks = self.compute_matrix_parameters_oars(matrix_method)
        else:
            Z_blocks, W_blocks = self.compute_matrix_parameters()
        
        # Verify matrix parameters (on rank 0)
        if self.rank == 0 and self.use_oars:
            from proximal_operators import MatrixParameterVerification
            # Verify first sensor's block as a sample
            sensor_id = self.sensor_ids[0]
            verification = MatrixParameterVerification.verify_equation_8(
                Z_blocks[sensor_id], W_blocks[sensor_id]
            )
            if verification['all_satisfied']:
                logger.info("Matrix parameter verification passed ✓")
            else:
                logger.warning(f"Matrix parameter verification failed: {verification}")
        
        # Initialize variables for OARS-style algorithm if available
        if self.use_oars and hasattr(oars, 'algorithms'):
            # Use OARS algorithm framework
            return self._run_oars_algorithm(Z_blocks, W_blocks, prox_ops)
        else:
            # Fall back to our implementation
            return self._run_mps_algorithm(Z_blocks, W_blocks, prox_ops)
    
    def _run_oars_algorithm(self, Z_blocks, W_blocks, prox_ops):
        """Run algorithm using OARS framework"""
        # Prepare resolvents for each sensor
        resolvents = []
        
        for sensor_id in self.sensor_ids:
            # Create resolvent functions for gi and indicator
            def resolvent_gi(v, alpha):
                v_X = v[:self.problem.d]
                v_Y = v[self.problem.d:].reshape(len(self.sensor_data[sensor_id].neighbors) + 1, -1)
                X_new, Y_new = prox_ops.prox_gi_admm(
                    sensor_id, self.sensor_data[sensor_id],
                    v_X, v_Y, self.anchor_positions, alpha
                )
                return np.concatenate([X_new, Y_new.flatten()])
            
            def resolvent_indicator(v, alpha):
                v_X = v[:self.problem.d]
                v_Y = v[self.problem.d:].reshape(len(self.sensor_data[sensor_id].neighbors) + 1, -1)
                S_i = prox_ops.construct_Si(v_X, v_Y, self.problem.d)
                S_i_proj = prox_ops.prox_indicator_psd(S_i)
                X_new, Y_new = prox_ops.extract_from_Si(S_i_proj, sensor_id, self.sensor_data[sensor_id])
                return np.concatenate([X_new, Y_new.flatten()])
            
            resolvents.extend([resolvent_gi, resolvent_indicator])
        
        # Initial data
        n_total = 2 * len(self.sensor_ids)
        data = np.zeros(n_total * (self.problem.d + (max(len(s.neighbors) for s in self.sensor_data.values()) + 1)**2))
        
        # Run OARS solver
        try:
            x_opt, results = oars.solve(
                n_total, data, resolvents,
                W=W_blocks, Z=Z_blocks,
                gamma=self.problem.gamma,
                alpha=self.problem.alpha_mps,
                max_iter=self.problem.max_iter,
                tol=self.problem.tol
            )
            
            # Extract results
            results_dict = {}
            idx = 0
            for sensor_id in self.sensor_ids:
                n_neighbors = len(self.sensor_data[sensor_id].neighbors)
                size = self.problem.d + (n_neighbors + 1)**2
                
                sensor_result = x_opt[idx:idx+size]
                X = sensor_result[:self.problem.d]
                Y = sensor_result[self.problem.d:].reshape(n_neighbors + 1, n_neighbors + 1)
                
                results_dict[sensor_id] = (X, Y)
                idx += size
                
                # Update sensor data
                self.sensor_data[sensor_id].X = X
                self.sensor_data[sensor_id].Y = Y
            
            return results_dict
            
        except Exception as e:
            logger.warning(f"OARS solver failed: {e}, falling back to custom implementation")
            return self._run_mps_algorithm(Z_blocks, W_blocks, prox_ops)
    
    def _run_mps_algorithm(self, Z_blocks, W_blocks, prox_ops):
        """Run our custom MPS implementation"""
        # This is essentially the same as the parent class method
        # but separated for clarity
        return super().matrix_parametrized_splitting()
    
    def compare_algorithms_with_oars(self, matrix_methods: List[str] = None) -> Dict[str, any]:
        """
        Compare algorithms using different OARS matrix generation methods
        """
        if matrix_methods is None:
            matrix_methods = ["Sinkhorn-Knopp", "MinSLEM", "MaxConnectivity", "MinResist"]
        
        all_results = {}
        
        for method in matrix_methods:
            if self.rank == 0:
                logger.info(f"\nTesting with matrix method: {method}")
            
            # Reset variables
            for sensor_id in self.sensor_ids:
                self.sensor_data[sensor_id].X = np.zeros(self.problem.d)
                self.sensor_data[sensor_id].Y = np.zeros_like(self.sensor_data[sensor_id].Y)
            
            # Run MPS with this method
            start_time = time.time()
            
            if method == "Sinkhorn-Knopp":
                mps_results = self.matrix_parametrized_splitting()
            else:
                mps_results = self.matrix_parametrized_splitting_oars(method)
            
            mps_time = time.time() - start_time
            mps_error = self.compute_error(mps_results)
            
            if self.rank == 0:
                all_results[method] = {
                    'error': mps_error,
                    'time': mps_time
                }
                logger.info(f"{method}: Error={mps_error:.6f}, Time={mps_time:.2f}s")
        
        # Also run ADMM for comparison
        for sensor_id in self.sensor_ids:
            self.sensor_data[sensor_id].X = np.zeros(self.problem.d)
            self.sensor_data[sensor_id].Y = np.zeros_like(self.sensor_data[sensor_id].Y)
            self.sensor_data[sensor_id].U = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
            self.sensor_data[sensor_id].V = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
            self.sensor_data[sensor_id].R = (np.zeros(self.problem.d), np.zeros_like(self.sensor_data[sensor_id].Y))
        
        start_time = time.time()
        admm_results = self.admm_decentralized()
        admm_time = time.time() - start_time
        admm_error = self.compute_error(admm_results)
        
        if self.rank == 0:
            all_results['ADMM'] = {
                'error': admm_error,
                'time': admm_time
            }
            
            # Find best MPS method
            mps_methods = [m for m in all_results.keys() if m != 'ADMM']
            best_method = min(mps_methods, key=lambda m: all_results[m]['error'])
            
            logger.info("\n" + "="*50)
            logger.info("COMPARISON SUMMARY")
            logger.info("="*50)
            for method, result in all_results.items():
                logger.info(f"{method:20s}: Error={result['error']:.6f}, Time={result['time']:.2f}s")
            logger.info(f"\nBest MPS method: {best_method}")
            logger.info(f"MPS/ADMM error ratio: {all_results[best_method]['error']/admm_error:.2f}")
            logger.info("="*50)
        
        return all_results if self.rank == 0 else None


def main():
    """Example usage with OARS integration"""
    problem = SNLProblem(
        n_sensors=30,
        n_anchors=6,
        communication_range=0.7,
        noise_factor=0.05,
        seed=42
    )
    
    # Create enhanced solver
    snl = EnhancedDistributedSNL(problem, use_oars=True)
    snl.generate_network()
    
    if OARS_AVAILABLE:
        # Compare different matrix generation methods
        results = snl.compare_algorithms_with_oars()
    else:
        # Fall back to standard comparison
        results = snl.compare_algorithms()
    
    if snl.rank == 0 and results is not None:
        print("\nFinal Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()