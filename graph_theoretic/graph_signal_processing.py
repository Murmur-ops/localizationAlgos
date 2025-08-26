"""
Graph Signal Processing Operations for Sensor Localization

Based on research from:
- "Convolutional neural networks on graphs with fast localized spectral filtering" (NeurIPS 2016)
- "Graph Signal Processing: Overview, Challenges and Applications" (IEEE SPM 2018)

Implements Chebyshev polynomial filters and graph Fourier transform for distributed processing.
"""

import numpy as np
from scipy.special import chebyt
from typing import Optional, List, Callable
from scipy.linalg import expm


class GraphSignalProcessor:
    """
    Graph signal processing operations for distributed sensor networks
    
    Key innovations:
    1. Chebyshev polynomial approximation for localized filters
    2. Graph Fourier transform for spectral analysis
    3. Distributed implementable operations
    """
    
    def __init__(self, laplacian: np.ndarray, 
                 eigenvalues: Optional[np.ndarray] = None,
                 eigenvectors: Optional[np.ndarray] = None):
        """
        Initialize GSP processor
        
        Args:
            laplacian: Graph Laplacian matrix
            eigenvalues: Pre-computed eigenvalues (optional)
            eigenvectors: Pre-computed eigenvectors (optional)
        """
        self.L = laplacian
        self.n = laplacian.shape[0]
        
        # Store or compute spectral decomposition
        if eigenvalues is not None and eigenvectors is not None:
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        else:
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)
        
        # Normalize Laplacian eigenvalues to [-1, 1] for Chebyshev
        self.lambda_max = np.max(np.abs(self.eigenvalues))
        self.L_normalized = 2.0 * self.L / self.lambda_max - np.eye(self.n)
    
    def chebyshev_filter(self, signal: np.ndarray, 
                        coefficients: List[float],
                        K: Optional[int] = None) -> np.ndarray:
        """
        Apply Chebyshev polynomial filter to graph signal
        
        Research: "Fast localized spectral filtering" - K-hop localized
        Distributed implementation possible!
        
        Args:
            signal: Graph signal (node values)
            coefficients: Chebyshev coefficients defining filter
            K: Polynomial order (defaults to len(coefficients))
            
        Returns:
            Filtered signal
        """
        if K is None:
            K = len(coefficients)
        
        # Chebyshev recursion: T_0 = I, T_1 = L_norm, T_k = 2*L_norm*T_{k-1} - T_{k-2}
        T_old = signal.copy()
        T_curr = self.L_normalized @ signal
        
        # Apply filter
        filtered = coefficients[0] * T_old
        if K > 1:
            filtered += coefficients[1] * T_curr
        
        for k in range(2, K):
            T_new = 2 * self.L_normalized @ T_curr - T_old
            filtered += coefficients[k] * T_new
            T_old = T_curr
            T_curr = T_new
        
        return filtered
    
    def heat_diffusion_filter(self, signal: np.ndarray, 
                             t: float = 1.0,
                             K: int = 10) -> np.ndarray:
        """
        Apply heat diffusion filter (smoothing)
        
        Research: Heat kernel h(λ) = exp(-tλ) provides optimal smoothing
        Approximated with Chebyshev polynomials for distributed implementation
        
        Args:
            signal: Input signal
            t: Diffusion time (larger = more smoothing)
            K: Chebyshev approximation order
            
        Returns:
            Smoothed signal
        """
        # Compute Chebyshev coefficients for exp(-tλ)
        coeffs = self._compute_heat_kernel_coeffs(t, K)
        return self.chebyshev_filter(signal, coeffs, K)
    
    def _compute_heat_kernel_coeffs(self, t: float, K: int) -> List[float]:
        """
        Compute Chebyshev coefficients for heat kernel
        
        Using Bessel function approximation
        """
        from scipy.special import iv  # Modified Bessel function
        
        coeffs = []
        for k in range(K):
            if k == 0:
                coeff = np.exp(-t * self.lambda_max) * iv(0, t * self.lambda_max)
            else:
                coeff = 2 * np.exp(-t * self.lambda_max) * iv(k, t * self.lambda_max)
            coeffs.append(coeff)
        
        return coeffs
    
    def graph_fourier_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Graph Fourier Transform (GFT)
        
        Research: GFT projects signal onto graph eigenvectors
        Analogous to classical Fourier transform
        
        Args:
            signal: Graph signal
            
        Returns:
            Spectral coefficients
        """
        return self.eigenvectors.T @ signal
    
    def inverse_graph_fourier_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Compute inverse Graph Fourier Transform
        
        Args:
            coefficients: Spectral coefficients
            
        Returns:
            Graph signal
        """
        return self.eigenvectors @ coefficients
    
    def low_pass_filter(self, signal: np.ndarray, 
                       cutoff: float = 0.5,
                       K: int = 10) -> np.ndarray:
        """
        Apply ideal low-pass filter
        
        Research: Low frequencies = smooth signals on graph
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency (fraction of lambda_max)
            K: Chebyshev order
            
        Returns:
            Low-pass filtered signal
        """
        # Design filter: h(λ) = 1 if λ < cutoff*λ_max, 0 otherwise
        def filter_func(lam):
            return 1.0 if lam < cutoff * self.lambda_max else 0.0
        
        coeffs = self._compute_filter_coeffs(filter_func, K)
        return self.chebyshev_filter(signal, coeffs, K)
    
    def high_pass_filter(self, signal: np.ndarray,
                        cutoff: float = 0.5,
                        K: int = 10) -> np.ndarray:
        """
        Apply ideal high-pass filter
        
        Research: High frequencies = rapid variations on graph
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency (fraction of lambda_max)
            K: Chebyshev order
            
        Returns:
            High-pass filtered signal
        """
        # Design filter: h(λ) = 0 if λ < cutoff*λ_max, 1 otherwise
        def filter_func(lam):
            return 0.0 if lam < cutoff * self.lambda_max else 1.0
        
        coeffs = self._compute_filter_coeffs(filter_func, K)
        return self.chebyshev_filter(signal, coeffs, K)
    
    def _compute_filter_coeffs(self, filter_func: Callable, K: int) -> List[float]:
        """
        Compute Chebyshev coefficients for arbitrary filter function
        
        Using Chebyshev approximation on [-1, 1]
        """
        # Sample filter at Chebyshev nodes
        nodes = np.cos(np.pi * (np.arange(K) + 0.5) / K)
        # Map from [-1, 1] to [0, lambda_max]
        lambda_nodes = (nodes + 1) * self.lambda_max / 2
        samples = [filter_func(lam) for lam in lambda_nodes]
        
        # Compute Chebyshev coefficients via DCT
        from scipy.fft import dct
        coeffs = dct(samples, type=2, norm='ortho') * np.sqrt(2/K)
        coeffs[0] /= np.sqrt(2)
        
        return coeffs.tolist()
    
    def denoise_signal(self, noisy_signal: np.ndarray,
                      regularization: float = 0.1) -> np.ndarray:
        """
        Denoise graph signal using Tikhonov regularization
        
        Research: Minimize ||y - x||^2 + α*x^T*L*x
        Solution: x = (I + αL)^{-1} * y
        
        Args:
            noisy_signal: Noisy measurements
            regularization: Regularization parameter (α)
            
        Returns:
            Denoised signal
        """
        # Solve (I + αL)x = y
        A = np.eye(self.n) + regularization * self.L
        denoised = np.linalg.solve(A, noisy_signal)
        return denoised
    
    def interpolate_signal(self, partial_signal: np.ndarray,
                          known_indices: List[int],
                          regularization: float = 0.1) -> np.ndarray:
        """
        Interpolate missing values using graph structure
        
        Research: Smooth signals have low graph frequency content
        
        Args:
            partial_signal: Signal with known values
            known_indices: Indices of known values
            regularization: Smoothness parameter
            
        Returns:
            Interpolated complete signal
        """
        n = self.n
        unknown_indices = [i for i in range(n) if i not in known_indices]
        
        # Partition Laplacian
        L_uu = self.L[np.ix_(unknown_indices, unknown_indices)]
        L_uk = self.L[np.ix_(unknown_indices, known_indices)]
        
        # Known values
        x_known = partial_signal[known_indices]
        
        # Solve for unknown values: L_uu * x_unknown = -L_uk * x_known
        x_unknown = np.linalg.solve(L_uu + regularization * np.eye(len(unknown_indices)), 
                                   -L_uk @ x_known)
        
        # Combine
        complete_signal = np.zeros(n)
        complete_signal[known_indices] = x_known
        complete_signal[unknown_indices] = x_unknown
        
        return complete_signal
    
    def graph_wavelet_transform(self, signal: np.ndarray,
                               scales: List[float]) -> np.ndarray:
        """
        Compute graph wavelet transform at multiple scales
        
        Research: Wavelets provide multi-resolution analysis on graphs
        
        Args:
            signal: Input signal
            scales: List of wavelet scales
            
        Returns:
            Wavelet coefficients (n_nodes x n_scales)
        """
        coefficients = np.zeros((self.n, len(scales)))
        
        for i, scale in enumerate(scales):
            # Wavelet kernel: g(sλ) = exp(-(sλ)^2)
            wavelet_coeffs = [np.exp(-(scale * lam)**2) for lam in self.eigenvalues]
            # Apply wavelet
            filtered = self.eigenvectors @ np.diag(wavelet_coeffs) @ self.eigenvectors.T @ signal
            coefficients[:, i] = filtered
        
        return coefficients


def test_gsp_operations():
    """Test graph signal processing operations"""
    print("="*60)
    print("Testing Graph Signal Processing Operations")
    print("="*60)
    
    # Create simple test graph (path graph for visualization)
    n = 20
    L = np.zeros((n, n))
    for i in range(n-1):
        L[i, i] += 1
        L[i+1, i+1] += 1
        L[i, i+1] = -1
        L[i+1, i] = -1
    
    # Add some additional edges for more interesting topology
    L[0, n-1] = L[n-1, 0] = -0.5
    L[0, 0] += 0.5
    L[n-1, n-1] += 0.5
    
    # Initialize GSP
    gsp = GraphSignalProcessor(L)
    
    # Create test signal (step function + noise)
    signal = np.zeros(n)
    signal[:n//2] = 1.0
    signal += 0.1 * np.random.randn(n)
    
    print(f"\nOriginal signal stats:")
    print(f"  Mean: {signal.mean():.4f}")
    print(f"  Std: {signal.std():.4f}")
    
    # Test heat diffusion (smoothing)
    smoothed = gsp.heat_diffusion_filter(signal, t=0.5, K=10)
    print(f"\nSmoothed signal stats:")
    print(f"  Mean: {smoothed.mean():.4f}")
    print(f"  Std: {smoothed.std():.4f}")
    
    # Test denoising
    denoised = gsp.denoise_signal(signal, regularization=0.5)
    print(f"\nDenoised signal stats:")
    print(f"  Mean: {denoised.mean():.4f}")
    print(f"  Std: {denoised.std():.4f}")
    
    # Test interpolation
    known_indices = [0, 5, 10, 15, 19]
    partial = signal.copy()
    interpolated = gsp.interpolate_signal(partial, known_indices, regularization=0.1)
    print(f"\nInterpolation from {len(known_indices)} known values:")
    print(f"  MSE: {np.mean((interpolated - signal)**2):.6f}")
    
    # Test Graph Fourier Transform
    gft_coeffs = gsp.graph_fourier_transform(signal)
    print(f"\nGraph Fourier Transform:")
    print(f"  Low frequency energy (first 5): {np.sum(gft_coeffs[:5]**2):.4f}")
    print(f"  High frequency energy (last 5): {np.sum(gft_coeffs[-5:]**2):.4f}")
    
    return gsp


if __name__ == "__main__":
    gsp = test_gsp_operations()