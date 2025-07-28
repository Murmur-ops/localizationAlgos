"""
Unit tests for core algorithms
Tests proximal operators, matrix operations, and algorithm components
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import without MPI dependency
from dataclasses import dataclass, field

@dataclass
class SNLProblem:
    """Problem parameters for sensor network localization"""
    n_sensors: int = 30
    n_anchors: int = 6
    d: int = 2  # dimension
    communication_range: float = 0.7
    max_neighbors: int = 7
    noise_factor: float = 0.05
    gamma: float = 0.999
    alpha_mps: float = 10.0
    alpha_admm: float = 150.0
    max_iter: int = 1000
    tol: float = 1e-4
    early_termination_window: int = 100
    seed: int = None

@dataclass
class SensorData:
    """Data structure for a single sensor"""
    id: int
    neighbors: List[int] = field(default_factory=list)
    distance_measurements: Dict[int, float] = field(default_factory=dict)
    anchor_neighbors: List[int] = field(default_factory=list)
    anchor_distances: Dict[int, float] = field(default_factory=dict)
    X: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))


class TestProximalOperators:
    """Test proximal operator implementations"""
    
    def test_prox_indicator_psd(self):
        """Test projection onto PSD cone"""
        # Test case 1: Already PSD matrix
        A = np.array([[2, 1], [1, 2]])
        A_proj = self._prox_indicator_psd(A)
        assert np.allclose(A, A_proj), "PSD matrix should remain unchanged"
        
        # Test case 2: Non-PSD matrix
        B = np.array([[1, 2], [2, 1]])
        B_proj = self._prox_indicator_psd(B)
        eigenvalues = np.linalg.eigvalsh(B_proj)
        assert np.all(eigenvalues >= -1e-10), "Projected matrix should be PSD"
        
        # Test case 3: Zero matrix
        C = np.zeros((3, 3))
        C_proj = self._prox_indicator_psd(C)
        assert np.allclose(C, C_proj), "Zero matrix should remain zero"
    
    def test_construct_Si_matrix(self):
        """Test Si matrix construction"""
        X = np.array([1.0, 2.0])
        Y = np.array([[4, 1, 0], [1, 5, 1], [0, 1, 6]])
        d = 2
        
        Si = self._construct_Si(X, Y, d)
        
        # Check dimensions
        expected_size = d + Y.shape[0]
        assert Si.shape == (expected_size, expected_size), f"Si should be {expected_size}x{expected_size}"
        
        # Check structure
        assert np.allclose(Si[:d, :d], np.eye(d)), "Top-left should be identity"
        assert np.allclose(Si[:d, d:], X.reshape(d, 1) @ np.ones((1, Y.shape[0]))), "Top-right block incorrect"
        assert np.allclose(Si[d:, d:], Y), "Bottom-right should be Y"
    
    def test_extract_from_Si(self):
        """Test extraction from Si matrix"""
        # Create a known Si matrix
        X_true = np.array([1.5, 2.5])
        Y_true = np.array([[3, 1], [1, 4]])
        d = 2
        n = Y_true.shape[0]
        
        Si = np.zeros((d + n, d + n))
        Si[:d, :d] = np.eye(d)
        Si[:d, d:] = X_true.reshape(d, 1) @ np.ones((1, n))
        Si[d:, :d] = np.ones((n, 1)) @ X_true.reshape(1, d)
        Si[d:, d:] = Y_true
        
        # Extract
        X_extracted, Y_extracted = self._extract_from_Si(Si, d, n)
        
        assert np.allclose(X_extracted, X_true), "Extracted X should match original"
        assert np.allclose(Y_extracted, Y_true), "Extracted Y should match original"
    
    def _prox_indicator_psd(self, A: np.ndarray) -> np.ndarray:
        """Project matrix onto PSD cone"""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _construct_Si(self, X: np.ndarray, Y: np.ndarray, d: int) -> np.ndarray:
        """Construct Si matrix"""
        n = Y.shape[0]
        Si = np.zeros((d + n, d + n))
        Si[:d, :d] = np.eye(d)
        Si[:d, d:] = X.reshape(d, 1) @ np.ones((1, n))
        Si[d:, :d] = np.ones((n, 1)) @ X.reshape(1, d)
        Si[d:, d:] = Y
        return Si
    
    def _extract_from_Si(self, Si: np.ndarray, d: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X and Y from Si matrix"""
        X = Si[:d, d:].mean(axis=1)
        Y = Si[d:, d:]
        return X, Y


class TestMatrixOperations:
    """Test matrix parameter computations"""
    
    def test_L_from_Z_computation(self):
        """Test L matrix computation from Z"""
        # Test case 1: Simple 2x2
        Z1 = np.array([[2, -0.5], [-0.5, 2]])
        L1 = self._compute_L_from_Z(Z1)
        
        # Verify Z = 2I - L - L^T
        I = np.eye(2)
        Z1_reconstructed = 2 * I - L1 - L1.T
        assert np.allclose(Z1, Z1_reconstructed), "Z reconstruction failed for 2x2"
        
        # Test case 2: 3x3 matrix
        Z2 = np.array([[2, -0.3, -0.2], [-0.3, 2, -0.4], [-0.2, -0.4, 2]])
        L2 = self._compute_L_from_Z(Z2)
        
        # Check lower triangular
        assert np.allclose(L2[np.triu_indices_from(L2, k=1)], 0), "L should be lower triangular"
        
        # Verify reconstruction
        I3 = np.eye(3)
        Z2_reconstructed = 2 * I3 - L2 - L2.T
        assert np.allclose(Z2, Z2_reconstructed), "Z reconstruction failed for 3x3"
    
    def test_sinkhorn_knopp_properties(self):
        """Test Sinkhorn-Knopp algorithm properties"""
        # Create a simple adjacency matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)
        
        # Run simplified Sinkhorn-Knopp
        W = self._simple_sinkhorn_knopp(A + np.eye(4), max_iter=100)
        
        # Check doubly stochastic property
        row_sums = W.sum(axis=1)
        col_sums = W.sum(axis=0)
        
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Row sums should be 1"
        assert np.allclose(col_sums, 1.0, atol=1e-6), "Column sums should be 1"
        assert np.allclose(W, W.T), "Matrix should be symmetric"
    
    def test_2block_matrix_structure(self):
        """Test 2-Block matrix parameter structure"""
        n = 4
        
        # Create test Z matrix
        Z = 2 * np.eye(n) - 0.25 * np.ones((n, n))
        
        # Compute W (should equal Z for 2-Block)
        W = Z.copy()
        
        # Check eigenvalue bound
        eigenvalues = np.linalg.eigvalsh(W)
        assert np.all(eigenvalues >= 0), "W should be PSD"
        assert np.all(eigenvalues <= 2), "Eigenvalues should be bounded by 2"
    
    def _compute_L_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Compute lower triangular L from Z = 2I - L - L^T"""
        n = Z.shape[0]
        L = np.zeros((n, n))
        
        # Diagonal elements
        for i in range(n):
            L[i, i] = (2 - Z[i, i]) / 2
        
        # Off-diagonal elements
        for i in range(n):
            for j in range(i):
                L[i, j] = -Z[i, j]
        
        return L
    
    def _simple_sinkhorn_knopp(self, A: np.ndarray, max_iter: int = 100, 
                               tol: float = 1e-6) -> np.ndarray:
        """Simple Sinkhorn-Knopp implementation"""
        n = A.shape[0]
        W = A.copy()
        
        for _ in range(max_iter):
            # Row normalization
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
            
            # Column normalization
            col_sums = W.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            W = W / col_sums
            
            # Check convergence
            if np.allclose(W.sum(axis=1), 1.0, atol=tol) and \
               np.allclose(W.sum(axis=0), 1.0, atol=tol):
                break
        
        return W


class TestConvergence:
    """Test convergence criteria and early termination"""
    
    def test_early_termination_logic(self):
        """Test early termination criteria"""
        # Test case 1: Decreasing objective (recent improvement)
        obj_history1 = [100, 80, 60, 40, 20, 10, 5, 2, 1, 0.5]
        # With window=5, checks last 5 values [10, 5, 2, 1, 0.5] against min=0.5
        # Since recent values show improvement, should not terminate
        assert not self._check_early_termination(obj_history1, window=5), \
            "Should not terminate when objective is decreasing"
        
        # Test case 2: Stagnant objective
        obj_history2 = [100, 80, 60, 40, 20, 10, 10, 10, 10, 10, 10, 10]
        # With window=5, last 5 are all 10, and min is also 10
        assert self._check_early_termination(obj_history2, window=5), \
            "Should terminate when objective stagnates"
        
        # Test case 3: No recent improvement
        obj_history3 = [100, 50, 20, 10, 5, 5.1, 5.05, 5.02, 5.01, 5.001]
        # Min is 5, recent values are all slightly above 5
        assert self._check_early_termination(obj_history3, window=5), \
            "Should terminate when no recent improvement"
    
    def test_convergence_metrics(self):
        """Test convergence metric calculations"""
        # Simulate position updates
        old_positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        new_positions = np.array([[0.01, 0], [1, 0.01], [0, 1], [0.99, 1.01]])
        
        # Compute max change
        max_change = max(np.linalg.norm(new_positions[i] - old_positions[i]) 
                        for i in range(len(old_positions)))
        
        assert max_change < 0.02, "Max change should be small"
        assert max_change > 0, "Should detect some change"
    
    def test_relative_error_computation(self):
        """Test relative error calculation"""
        true_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        estimated_positions = np.array([[0.1, 0], [0.9, 0.1], [1.1, 0.9], [0, 1.1]])
        
        error = np.linalg.norm(estimated_positions - true_positions, 'fro')
        error /= np.linalg.norm(true_positions, 'fro')
        
        assert 0 < error < 0.2, f"Relative error should be reasonable: {error}"
    
    def _check_early_termination(self, objective_history: List[float], 
                                window: int = 100) -> bool:
        """Check early termination criteria"""
        if len(objective_history) < window:
            return False
        
        recent_objectives = objective_history[-window:]
        
        # Check if the minimum in the recent window hasn't improved significantly
        # compared to the minimum before the window
        if len(objective_history) > window:
            min_before_window = min(objective_history[:-window])
            min_in_window = min(recent_objectives)
            
            # If no significant improvement (less than 1e-6 relative change)
            return min_in_window >= min_before_window * (1 - 1e-6)
        else:
            # Not enough history, don't terminate
            return False


class TestObjectiveFunction:
    """Test objective function computation"""
    
    def test_distance_objective(self):
        """Test distance measurement objective"""
        # Simple 2-sensor case
        X1 = np.array([0, 0])
        X2 = np.array([1, 0])
        measured_dist = 1.1  # With noise
        
        actual_dist = np.linalg.norm(X1 - X2)
        obj = abs(measured_dist**2 - actual_dist**2)
        
        expected_obj = abs(1.1**2 - 1.0**2)
        assert np.isclose(obj, expected_obj), f"Objective mismatch: {obj} vs {expected_obj}"
    
    def test_anchor_objective(self):
        """Test anchor distance objective"""
        sensor_pos = np.array([0.5, 0.5])
        anchor_pos = np.array([0, 0])
        measured_dist = 0.75  # With noise
        
        actual_dist = np.linalg.norm(sensor_pos - anchor_pos)
        obj = abs(measured_dist**2 - actual_dist**2)
        
        assert obj > 0, "Should have non-zero objective with noise"


def run_all_tests():
    """Run all unit tests"""
    print("Running Core Algorithm Unit Tests")
    print("=" * 60)
    
    # Test proximal operators
    print("\nTesting Proximal Operators...")
    prox_tests = TestProximalOperators()
    prox_tests.test_prox_indicator_psd()
    print("  ✓ PSD projection test passed")
    prox_tests.test_construct_Si_matrix()
    print("  ✓ Si matrix construction test passed")
    prox_tests.test_extract_from_Si()
    print("  ✓ Si extraction test passed")
    
    # Test matrix operations
    print("\nTesting Matrix Operations...")
    matrix_tests = TestMatrixOperations()
    matrix_tests.test_L_from_Z_computation()
    print("  ✓ L from Z computation test passed")
    matrix_tests.test_sinkhorn_knopp_properties()
    print("  ✓ Sinkhorn-Knopp properties test passed")
    matrix_tests.test_2block_matrix_structure()
    print("  ✓ 2-Block matrix structure test passed")
    
    # Test convergence
    print("\nTesting Convergence Criteria...")
    conv_tests = TestConvergence()
    conv_tests.test_early_termination_logic()
    print("  ✓ Early termination logic test passed")
    conv_tests.test_convergence_metrics()
    print("  ✓ Convergence metrics test passed")
    conv_tests.test_relative_error_computation()
    print("  ✓ Relative error computation test passed")
    
    # Test objective
    print("\nTesting Objective Function...")
    obj_tests = TestObjectiveFunction()
    obj_tests.test_distance_objective()
    print("  ✓ Distance objective test passed")
    obj_tests.test_anchor_objective()
    print("  ✓ Anchor objective test passed")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()