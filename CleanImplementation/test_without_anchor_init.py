"""
Test MPS performance WITHOUT using anchors for initialization
This shows the TRUE distributed performance
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.mps_proper import ProperMPSAlgorithm
from analysis.crlb_analysis import CRLBAnalyzer


def test_without_anchor_init():
    """Test MPS with random initialization (no anchor help)"""
    
    print("="*70)
    print("Testing MPS WITHOUT Anchor-based Initialization")
    print("="*70)
    
    # Setup
    n_sensors = 20
    n_anchors = 4
    noise_factor = 0.05
    
    analyzer = CRLBAnalyzer(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        d=2
    )
    
    crlb = analyzer.compute_crlb(noise_factor)
    print(f"\nCRLB theoretical bound: {crlb:.4f}")
    
    # Test 1: WITH anchor initialization (current approach)
    print("\n1. WITH Anchor Initialization (current):")
    mps_with = ProperMPSAlgorithm(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        noise_factor=noise_factor,
        gamma=0.99,
        alpha=1.0,
        max_iter=500,
        tol=1e-5
    )
    
    mps_with.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    
    # Check initial error BEFORE optimization
    state = mps_with._initialize_variables()
    initial_errors = []
    for i in range(n_sensors):
        error = np.linalg.norm(state.positions[i] - analyzer.true_positions[i])
        initial_errors.append(error)
    initial_rmse_with = np.sqrt(np.mean(np.square(initial_errors)))
    
    result_with = mps_with.run()
    
    print(f"   Initial RMSE: {initial_rmse_with:.4f}")
    print(f"   Final RMSE: {result_with['final_error']:.4f}")
    print(f"   CRLB Efficiency: {(crlb/result_with['final_error']*100):.1f}%")
    print(f"   Iterations: {result_with['iterations']}")
    
    # Test 2: WITHOUT anchor initialization (truly distributed)
    print("\n2. WITHOUT Anchor Initialization (truly distributed):")
    
    class MPSNoAnchorInit(ProperMPSAlgorithm):
        def _initialize_variables(self):
            """Override to use ONLY random initialization"""
            n = self.n_sensors
            from algorithms.mps_proper import MPSState
            state = MPSState(
                positions={},
                Y=np.zeros((2 * n, self.d)),
                X=np.zeros((2 * n, self.d)),
                U=np.zeros((2 * n, self.d))
            )
            
            # RANDOM initialization only - no anchor information!
            for i in range(n):
                state.positions[i] = np.random.uniform(0.2, 0.8, self.d)
            
            # Initialize consensus variables
            for i in range(n):
                state.X[i] = state.positions[i]
                state.X[i + n] = state.positions[i]
                state.Y[i] = state.positions[i]
                state.Y[i + n] = state.positions[i]
            
            return state
    
    mps_without = MPSNoAnchorInit(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        noise_factor=noise_factor,
        gamma=0.99,
        alpha=1.0,
        max_iter=500,
        tol=1e-5
    )
    
    mps_without.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    
    # Check initial error
    state_no_anchor = mps_without._initialize_variables()
    initial_errors_no = []
    for i in range(n_sensors):
        error = np.linalg.norm(state_no_anchor.positions[i] - analyzer.true_positions[i])
        initial_errors_no.append(error)
    initial_rmse_without = np.sqrt(np.mean(np.square(initial_errors_no)))
    
    result_without = mps_without.run()
    
    print(f"   Initial RMSE: {initial_rmse_without:.4f}")
    print(f"   Final RMSE: {result_without['final_error']:.4f}")
    print(f"   CRLB Efficiency: {(crlb/result_without['final_error']*100):.1f}%")
    print(f"   Iterations: {result_without['iterations']}")
    
    # Test 3: What if we also don't use anchors during optimization?
    print("\n3. NO Anchors At All (only sensor-to-sensor):")
    
    class MPSNoAnchorsAtAll(MPSNoAnchorInit):
        def _prox_f(self, state):
            """Override to IGNORE anchor measurements"""
            X_new = state.X.copy()
            n = self.n_sensors
            
            for i in range(n):
                # ONLY sensor-to-sensor constraints
                grad = np.zeros(self.d)
                weight_sum = 0
                
                for j in range(n):
                    if (i, j) in self.distance_measurements:
                        measured_dist = self.distance_measurements[(i, j)]
                        # Simple gradient descent
                        from algorithms.proximal_operators import ProximalOperators
                        X_new[i] = ProximalOperators.prox_distance_constraint(
                            X_new[i], X_new[j], measured_dist, 
                            alpha=self.alpha / (len(self.distance_measurements) + 1)
                        )
                
                # NO ANCHOR CONSTRAINTS!
                
                # Copy to second block
                X_new[i + n] = X_new[i]
            
            return X_new
    
    mps_no_anchors = MPSNoAnchorsAtAll(
        n_sensors=n_sensors,
        n_anchors=n_anchors,
        communication_range=0.4,
        noise_factor=noise_factor,
        gamma=0.99,
        alpha=1.0,
        max_iter=500,
        tol=1e-5
    )
    
    mps_no_anchors.generate_network(analyzer.true_positions, analyzer.anchor_positions)
    result_no_anchors = mps_no_anchors.run()
    
    print(f"   Initial RMSE: {initial_rmse_without:.4f}")
    print(f"   Final RMSE: {result_no_anchors['final_error']:.4f}")
    print(f"   CRLB Efficiency: {(crlb/result_no_anchors['final_error']*100 if result_no_anchors['final_error'] > 0 else 0):.1f}%")
    print(f"   Iterations: {result_no_anchors['iterations']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Impact of Anchor Usage")
    print("="*70)
    
    print(f"\nCRLB Bound: {crlb:.4f}")
    print(f"\n{'Method':<40} {'Init RMSE':<12} {'Final RMSE':<12} {'CRLB Eff':<10}")
    print("-"*70)
    print(f"{'WITH anchor init + anchor constraints':<40} {initial_rmse_with:<12.4f} {result_with['final_error']:<12.4f} {(crlb/result_with['final_error']*100):<10.1f}%")
    print(f"{'Random init + anchor constraints':<40} {initial_rmse_without:<12.4f} {result_without['final_error']:<12.4f} {(crlb/result_without['final_error']*100):<10.1f}%")
    print(f"{'Random init + NO anchor constraints':<40} {initial_rmse_without:<12.4f} {result_no_anchors['final_error']:<12.4f} {(crlb/result_no_anchors['final_error']*100 if result_no_anchors['final_error'] > 0 else 0):<10.1f}%")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    anchor_init_boost = (result_without['final_error'] - result_with['final_error']) / result_without['final_error'] * 100
    anchor_constraint_boost = (result_no_anchors['final_error'] - result_without['final_error']) / result_no_anchors['final_error'] * 100 if result_no_anchors['final_error'] > 0 else 0
    
    print(f"\n1. Anchor initialization improves accuracy by: {anchor_init_boost:.1f}%")
    print(f"2. Anchor constraints improve accuracy by: {anchor_constraint_boost:.1f}%")
    print(f"3. Initial position quality (with anchors): {initial_rmse_with:.4f}")
    print(f"4. Initial position quality (random): {initial_rmse_without:.4f}")
    
    if result_no_anchors['final_error'] > initial_rmse_without:
        print(f"\nWARNING: Without anchors, the algorithm DIVERGES!")
        print(f"Final error ({result_no_anchors['final_error']:.4f}) > Initial error ({initial_rmse_without:.4f})")
    
    return result_with, result_without, result_no_anchors


if __name__ == "__main__":
    results = test_without_anchor_init()