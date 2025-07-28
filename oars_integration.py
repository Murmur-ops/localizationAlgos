"""
OARS Integration for Decentralized SNL
Implements advanced matrix parameter generation methods from OARS
"""

import numpy as np
import sys
import os

# Add OARS to path
oars_path = os.path.join(os.path.dirname(__file__), 'oars')
sys.path.insert(0, oars_path)

try:
    import cvxpy as cvx
    from oars.matrices.core import getCore, getMinSpectralDifference
    from oars.matrices.prebuilt import getMT, getFull, getRyu
    OARS_AVAILABLE = True
    
    # Try to import advanced methods if available
    try:
        from oars.matrices import getMinSLEM, getMaxConnectivity
        ADVANCED_OARS = True
    except ImportError:
        ADVANCED_OARS = False
        
except ImportError as e:
    OARS_AVAILABLE = False
    print(f"Warning: OARS or cvxpy not available. Using basic matrix generation. Error: {e}")


class OARSMatrixGenerator:
    """Generate matrix parameters using OARS methods"""
    
    def __init__(self, n_sensors, adjacency_matrix=None):
        self.n = n_sensors
        self.adjacency = adjacency_matrix
        self.method_names = [
            'sinkhorn_knopp',  # Our basic implementation
            'malitsky_tam',    # From OARS prebuilt
            'full_connected',  # From OARS prebuilt
            'ryu_extended',    # From OARS prebuilt
            'min_spectral',    # OARS optimization
            'min_slem',        # OARS optimization
            'max_connectivity' # OARS optimization
        ]
    
    def generate_matrices(self, method='sinkhorn_knopp', fixed_edges=None):
        """
        Generate Z and W matrices using specified method
        
        Args:
            method: One of the available methods
            fixed_edges: Dict of (i,j): value for fixed edge weights
            
        Returns:
            Z: Consensus matrix
            W: Convergence matrix
        """
        if not OARS_AVAILABLE and method != 'sinkhorn_knopp':
            print(f"OARS not available, falling back to sinkhorn_knopp")
            method = 'sinkhorn_knopp'
        
        if method == 'sinkhorn_knopp':
            return self._generate_sinkhorn_knopp()
        elif method == 'malitsky_tam':
            return getMT(self.n)
        elif method == 'full_connected':
            return getFull(self.n)
        elif method == 'ryu_extended':
            return getRyu(self.n)
        elif method == 'min_spectral':
            return self._generate_min_spectral(fixed_edges)
        elif method == 'min_slem':
            return self._generate_min_slem(fixed_edges)
        elif method == 'max_connectivity':
            return self._generate_max_connectivity(fixed_edges)
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
        
        # Create Z and W
        I = np.eye(self.n)
        Z = 2 * (I - A)
        W = Z.copy()
        
        return Z, W
    
    def _generate_min_spectral(self, fixed_edges=None):
        """Generate matrices minimizing spectral norm difference"""
        if fixed_edges is None:
            fixed_edges = {}
            
        # Convert adjacency to fixed edges
        if self.adjacency is not None:
            for i in range(self.n):
                for j in range(i):
                    if self.adjacency[i, j] == 0:
                        fixed_edges[(i, j)] = 0
                        fixed_edges[(j, i)] = 0
        
        return getMinSpectralDifference(self.n, fixed_Z=fixed_edges, fixed_W=fixed_edges)
    
    def _generate_min_slem(self, fixed_edges=None):
        """Generate matrices minimizing SLEM (Squared Laplacian Eigenvalue Multiplicity)"""
        if fixed_edges is None:
            fixed_edges = {}
            
        if self.adjacency is not None:
            for i in range(self.n):
                for j in range(i):
                    if self.adjacency[i, j] == 0:
                        fixed_edges[(i, j)] = 0
                        fixed_edges[(j, i)] = 0
        
        return getMinSLEM(self.n, fixed_Z=fixed_edges, fixed_W=fixed_edges)
    
    def _generate_max_connectivity(self, fixed_edges=None):
        """Generate matrices maximizing algebraic connectivity"""
        if fixed_edges is None:
            fixed_edges = {}
            
        if self.adjacency is not None:
            for i in range(self.n):
                for j in range(i):
                    if self.adjacency[i, j] == 0:
                        fixed_edges[(i, j)] = 0
                        fixed_edges[(j, i)] = 0
        
        return getMaxConnectivity(self.n, fixed_Z=fixed_edges, fixed_W=fixed_edges)
    
    def compare_methods(self, verbose=True):
        """Compare different matrix generation methods"""
        results = {}
        
        for method in self.method_names:
            try:
                Z, W = self.generate_matrices(method)
                
                # Compute metrics
                spectral_diff = np.linalg.norm(Z - W, 2)
                fiedler = np.sort(np.linalg.eigvalsh(W))[1]  # Second smallest eigenvalue
                max_eigenvalue = np.max(np.linalg.eigvalsh(W))
                
                results[method] = {
                    'Z': Z,
                    'W': W,
                    'spectral_diff': spectral_diff,
                    'fiedler_value': fiedler,
                    'max_eigenvalue': max_eigenvalue,
                    'condition_number': max_eigenvalue / fiedler if fiedler > 0 else np.inf
                }
                
                if verbose:
                    print(f"\n{method}:")
                    print(f"  Spectral difference ||Z-W||: {spectral_diff:.4f}")
                    print(f"  Fiedler value (connectivity): {fiedler:.4f}")
                    print(f"  Condition number: {results[method]['condition_number']:.2f}")
                    
            except Exception as e:
                if verbose:
                    print(f"\n{method}: Failed - {str(e)}")
                results[method] = None
        
        return results


def test_oars_integration():
    """Test OARS integration with a small example"""
    print("="*60)
    print("Testing OARS Matrix Generation Methods")
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
    
    generator = OARSMatrixGenerator(n, adjacency)
    
    # Test individual methods
    print("\nTesting individual methods:")
    
    # 1. Basic Sinkhorn-Knopp
    print("\n1. Sinkhorn-Knopp (baseline):")
    Z_sk, W_sk = generator.generate_matrices('sinkhorn_knopp')
    print(f"   Z diagonal: {np.diag(Z_sk)}")
    print(f"   Row sums: {Z_sk.sum(axis=1)}")
    
    if OARS_AVAILABLE:
        # 2. Malitsky-Tam
        print("\n2. Malitsky-Tam:")
        Z_mt, W_mt = generator.generate_matrices('malitsky_tam')
        print(f"   Z diagonal: {np.diag(Z_mt)}")
        print(f"   Spectral norm difference: {np.linalg.norm(Z_mt - W_mt, 2):.4f}")
        
        # 3. Min Spectral
        print("\n3. Min Spectral Difference:")
        try:
            Z_ms, W_ms = generator.generate_matrices('min_spectral')
            print(f"   Optimization successful!")
            print(f"   Spectral norm difference: {np.linalg.norm(Z_ms - W_ms, 2):.4f}")
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Compare all methods
    print("\n" + "="*60)
    print("Comparing All Methods:")
    print("="*60)
    
    results = generator.compare_methods()
    
    # Find best method
    best_method = None
    best_fiedler = 0
    
    for method, result in results.items():
        if result is not None and result['fiedler_value'] > best_fiedler:
            best_method = method
            best_fiedler = result['fiedler_value']
    
    print(f"\nBest method (highest connectivity): {best_method}")
    print(f"Fiedler value: {best_fiedler:.4f}")
    
    return results


def integrate_with_snl():
    """Show how to integrate OARS with our SNL implementation"""
    print("\n" + "="*60)
    print("Integration with SNL Implementation")
    print("="*60)
    
    code_example = '''
# In snl_main_full.py or snl_threaded_standalone.py:

from oars_integration import OARSMatrixGenerator

class DistributedSNL:
    def compute_matrix_parameters_oars(self, method='min_slem'):
        """Use OARS to generate optimal matrix parameters"""
        
        # Build global adjacency from local information
        adjacency = self._build_global_adjacency()
        
        # Generate matrices using OARS
        generator = OARSMatrixGenerator(self.problem.n_sensors, adjacency)
        Z_global, W_global = generator.generate_matrices(method)
        
        # Extract local blocks for each sensor
        for sensor_id in self.sensor_ids:
            # Get local indices
            local_indices = self._get_local_indices(sensor_id)
            
            # Extract blocks
            self.Z_blocks[sensor_id] = Z_global[np.ix_(local_indices, local_indices)]
            self.W_blocks[sensor_id] = W_global[np.ix_(local_indices, local_indices)]
            
            # Compute L from Z
            self.L_blocks[sensor_id] = self._compute_L_from_Z(self.Z_blocks[sensor_id])
    '''
    
    print("Example integration:")
    print(code_example)
    
    print("\nBenefits of OARS integration:")
    print("1. Optimal matrix parameters for faster convergence")
    print("2. Better conditioning for numerical stability")
    print("3. Flexibility to choose method based on network topology")
    print("4. Theoretical guarantees on convergence rate")


if __name__ == "__main__":
    # Test OARS integration
    results = test_oars_integration()
    
    # Show integration example
    integrate_with_snl()
    
    print("\n" + "="*60)
    print("OARS integration complete!")
    print("="*60)