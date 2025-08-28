"""
OARS Integration for Decentralized SNL - Enhanced Version
Implements advanced matrix parameter generation methods from OARS
Optimized for CleanImplementation without any mock data
"""

import numpy as np
import sys
import os

# Add OARS to path
oars_path = '/Users/maxburnett/Documents/DecentralizedLocale/oars'
sys.path.insert(0, oars_path)

try:
    import cvxpy as cvx
    CVX_AVAILABLE = True
except ImportError:
    CVX_AVAILABLE = False

# Import OARS components directly to avoid pyomo dependency
try:
    sys.path.insert(0, oars_path + '/oars/matrices')
    import prebuilt
    getMT = prebuilt.getMT
    getFull = prebuilt.getFull
    getRyu = prebuilt.getRyu
    OARS_AVAILABLE = True
    print("OARS matrices successfully loaded (without pyomo dependency)")
except ImportError as e:
    OARS_AVAILABLE = False
    print(f"Warning: OARS not available. Using basic matrix generation. Error: {e}")

# Try to import advanced optimization methods
try:
    if CVX_AVAILABLE:
        import core
        getCore = core.getCore
        OARS_ADVANCED = True
    else:
        OARS_ADVANCED = False
except:
    OARS_ADVANCED = False


class EnhancedOARSMatrixGenerator:
    """Generate optimal matrix parameters using OARS methods"""
    
    def __init__(self, n_sensors, adjacency_matrix=None):
        self.n = n_sensors
        self.adjacency = adjacency_matrix
        self.method_names = [
            'sinkhorn_knopp',  # Basic fallback
            'malitsky_tam',    # From OARS prebuilt - good baseline
            'full_connected',  # From OARS prebuilt - for dense networks
            'ryu_extended',    # From OARS prebuilt - fast convergence
            'optimal_spectral', # OARS optimization - minimize ||Z-W||
            'optimal_connectivity' # OARS optimization - maximize connectivity
        ]
        
        # Analyze network for smart method selection
        self.network_density = self._compute_network_density()
        self.recommended_method = self._recommend_method()
    
    def _compute_network_density(self) -> float:
        """Compute network density (0 to 1)"""
        if self.adjacency is None:
            return 1.0  # Assume fully connected
        n = self.n
        max_edges = n * (n - 1) / 2
        actual_edges = np.sum(self.adjacency) / 2
        return actual_edges / max_edges if max_edges > 0 else 0
    
    def _recommend_method(self) -> str:
        """Recommend best method based on network topology"""
        if not OARS_AVAILABLE:
            return 'sinkhorn_knopp'
        
        if self.network_density > 0.7:
            return 'full_connected'  # Dense network
        elif self.network_density < 0.3:
            return 'ryu_extended'  # Sparse network
        else:
            return 'malitsky_tam'  # Medium density
    
    def generate_matrices(self, method='auto', fixed_edges=None, gamma=0.99):
        """
        Generate Z and W matrices using specified method
        
        Args:
            method: One of the available methods or 'auto' for smart selection
            fixed_edges: Dict of (i,j): value for fixed edge weights
            gamma: Scaling parameter for Z matrix (default 0.99)
            
        Returns:
            Z: Consensus matrix (scaled)
            W: Convergence matrix
        """
        if method == 'auto':
            method = self.recommended_method
            print(f"Auto-selected method: {method} (density={self.network_density:.2f})")
        
        if not OARS_AVAILABLE and method != 'sinkhorn_knopp':
            print(f"OARS not available, falling back to sinkhorn_knopp")
            method = 'sinkhorn_knopp'
        
        if method == 'sinkhorn_knopp':
            return self._generate_sinkhorn_knopp()
        elif method == 'malitsky_tam':
            Z, W = getMT(self.n)
            return gamma * Z, W  # Apply scaling
        elif method == 'full_connected':
            Z, W = getFull(self.n)
            return gamma * Z, W
        elif method == 'ryu_extended':
            Z, W = getRyu(self.n)
            return gamma * Z, W
        elif method == 'optimal_spectral':
            return self._generate_optimal_spectral(fixed_edges, gamma)
        elif method == 'optimal_connectivity':
            return self._generate_optimal_connectivity(fixed_edges, gamma)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_sinkhorn_knopp(self, max_iter=100, tol=1e-6):
        """Our basic Sinkhorn-Knopp implementation"""
        if self.adjacency is None:
            # Create fully connected
            A = np.ones((self.n, self.n)) - np.eye(self.n)
        else:
            A = self.adjacency.copy()
        
        # Add self-loops
        A = A + np.eye(self.n)
        
        # Sinkhorn-Knopp iteration
        for _ in range(max_iter):
            # Row normalize
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            A = A / row_sums
            
            # Column normalize
            col_sums = A.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            A = A / col_sums
            
            # Check convergence
            if np.allclose(A.sum(axis=1), 1.0, atol=tol) and \
               np.allclose(A.sum(axis=0), 1.0, atol=tol):
                break
        
        # Create Z and W (Laplacian-like matrices)
        I = np.eye(self.n)
        Z = 2 * I - A
        W = Z.copy()
        
        return Z, W
    
    def _generate_optimal_spectral(self, fixed_edges=None, gamma=0.99):
        """Generate matrices minimizing spectral norm difference ||Z-W||"""
        if not CVX_AVAILABLE or not OARS_ADVANCED:
            print("CVXPy/OARS optimization not available, falling back to Malitsky-Tam")
            Z, W = getMT(self.n)
            return gamma * Z, W
        
        if fixed_edges is None:
            fixed_edges = {}
        
        # Convert adjacency to fixed edges
        fixed_Z = {}
        fixed_W = {}
        if self.adjacency is not None:
            for i in range(self.n):
                for j in range(i):
                    if self.adjacency[i, j] == 0:
                        fixed_Z[(i, j)] = 0
                        fixed_Z[(j, i)] = 0
                        fixed_W[(i, j)] = 0
                        fixed_W[(j, i)] = 0
        
        # Add user-specified fixed edges
        fixed_Z.update(fixed_edges or {})
        fixed_W.update(fixed_edges or {})
        
        try:
            # Get core constraints from OARS
            Z, W, cons = getCore(self.n, fixed_Z=fixed_Z, fixed_W=fixed_W, 
                                gamma=gamma, eps=0.1)
            
            # Minimize spectral norm difference
            obj = cvx.Minimize(cvx.norm(Z - W, 'fro'))
            prob = cvx.Problem(obj, cons)
            prob.solve(solver='CLARABEL', verbose=False)
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return Z.value, W.value
            else:
                print(f"Optimization failed: {prob.status}, using Malitsky-Tam")
                Z, W = getMT(self.n)
                return gamma * Z, W
                
        except Exception as e:
            print(f"Optimization error: {e}, using Malitsky-Tam")
            Z, W = getMT(self.n)
            return gamma * Z, W
    
    def _generate_optimal_connectivity(self, fixed_edges=None, gamma=0.99):
        """Generate matrices maximizing algebraic connectivity (Fiedler value)"""
        if not CVX_AVAILABLE or not OARS_ADVANCED:
            print("CVXPy/OARS optimization not available, falling back to full connected")
            Z, W = getFull(self.n)
            return gamma * Z, W
        
        if fixed_edges is None:
            fixed_edges = {}
        
        # Convert adjacency to fixed edges
        fixed_Z = {}
        fixed_W = {}
        if self.adjacency is not None:
            for i in range(self.n):
                for j in range(i):
                    if self.adjacency[i, j] == 0:
                        fixed_Z[(i, j)] = 0
                        fixed_Z[(j, i)] = 0
                        fixed_W[(i, j)] = 0
                        fixed_W[(j, i)] = 0
        
        fixed_Z.update(fixed_edges or {})
        fixed_W.update(fixed_edges or {})
        
        try:
            # Get core constraints from OARS
            # We want to maximize connectivity while keeping Z-W small
            Z, W, cons = getCore(self.n, fixed_Z=fixed_Z, fixed_W=fixed_W,
                                gamma=gamma, eps=0.1, c=0.5)  # Higher connectivity
            
            # Objective: minimize difference while maximizing connectivity
            obj = cvx.Minimize(cvx.norm(Z - W, 'fro') - 0.1 * cvx.lambda_sum_smallest(W, 2))
            prob = cvx.Problem(obj, cons)
            prob.solve(solver='CLARABEL', verbose=False)
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return Z.value, W.value
            else:
                print(f"Optimization failed: {prob.status}, using full connected")
                Z, W = getFull(self.n)
                return gamma * Z, W
                
        except Exception as e:
            print(f"Optimization error: {e}, using full connected")
            Z, W = getFull(self.n)
            return gamma * Z, W
    
    def compare_methods(self, verbose=True):
        """Compare different matrix generation methods"""
        results = {}
        
        for method in self.method_names:
            try:
                Z, W = self.generate_matrices(method)
                
                # Compute metrics
                spectral_diff = np.linalg.norm(Z - W, 2)
                eigenvals = np.sort(np.linalg.eigvalsh(W))
                fiedler = eigenvals[1] if len(eigenvals) > 1 else 0  # Second smallest
                max_eigenvalue = eigenvals[-1]
                
                results[method] = {
                    'Z': Z,
                    'W': W,
                    'spectral_diff': spectral_diff,
                    'fiedler_value': fiedler,
                    'max_eigenvalue': max_eigenvalue,
                    'condition_number': max_eigenvalue / fiedler if fiedler > 1e-10 else np.inf,
                    'success': True
                }
                
                if verbose:
                    print(f"\n{method}:")
                    print(f"  Spectral difference ||Z-W||: {spectral_diff:.4f}")
                    print(f"  Fiedler value (connectivity): {fiedler:.4f}")
                    print(f"  Condition number: {results[method]['condition_number']:.2f}")
                    
            except Exception as e:
                if verbose:
                    print(f"\n{method}: Failed - {str(e)}")
                results[method] = {'success': False, 'error': str(e)}
        
        # Find best method
        best_method = None
        best_score = float('inf')
        
        for method, result in results.items():
            if result.get('success', False):
                # Score based on condition number and spectral difference
                score = result['condition_number'] + 10 * result['spectral_diff']
                if score < best_score:
                    best_method = method
                    best_score = score
        
        if verbose and best_method:
            print(f"\n{'='*60}")
            print(f"Best method: {best_method}")
            print(f"Score: {best_score:.2f}")
            print(f"{'='*60}")
        
        return results


def test_oars_integration():
    """Test enhanced OARS integration"""
    print("="*60)
    print("Testing Enhanced OARS Matrix Generation")
    print("="*60)
    
    # Create a simple network
    n = 6
    adjacency = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    
    generator = EnhancedOARSMatrixGenerator(n, adjacency)
    
    print(f"\nNetwork Analysis:")
    print(f"  Sensors: {n}")
    print(f"  Density: {generator.network_density:.2f}")
    print(f"  Recommended method: {generator.recommended_method}")
    
    # Test auto selection
    print("\n1. Auto-selected method:")
    Z_auto, W_auto = generator.generate_matrices('auto')
    print(f"   Z shape: {Z_auto.shape}")
    print(f"   W shape: {W_auto.shape}")
    print(f"   ||Z-W||: {np.linalg.norm(Z_auto - W_auto, 2):.4f}")
    
    # Compare all methods
    print("\n2. Comparing all methods:")
    results = generator.compare_methods(verbose=True)
    
    return results


if __name__ == "__main__":
    results = test_oars_integration()