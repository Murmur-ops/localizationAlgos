#!/usr/bin/env python3
"""
Validate solver performance against Cramér-Rao Lower Bound (CRLB)
The CRLB provides the theoretical minimum variance achievable by any unbiased estimator
"""

import numpy as np
from ftl.solver_scaled import SquareRootSolver, OptimizationConfig
import matplotlib.pyplot as plt


def compute_crlb_position(anchors, target, range_std_m):
    """
    Compute CRLB for position estimation with ToA measurements

    Args:
        anchors: Array of anchor positions (N x 2)
        target: True target position (2,)
        range_std_m: Range measurement standard deviation

    Returns:
        crlb_x, crlb_y: Standard deviations from CRLB
    """
    n_anchors = len(anchors)
    H = np.zeros((2, 2))  # Fisher Information Matrix for position

    for anchor in anchors:
        # Direction vector from target to anchor
        diff = anchor - target
        dist = np.linalg.norm(diff)
        if dist > 0:
            # Unit vector
            u = diff / dist
            # Add contribution to FIM
            H += np.outer(u, u) / (range_std_m**2)

    # CRLB is inverse of FIM
    try:
        crlb_cov = np.linalg.inv(H)
        crlb_x = np.sqrt(crlb_cov[0, 0])
        crlb_y = np.sqrt(crlb_cov[1, 1])
        return crlb_x, crlb_y
    except:
        return np.inf, np.inf


def monte_carlo_test(n_trials=100):
    """Run Monte Carlo simulation to validate solver against CRLB"""

    # UWB parameters
    range_std_m = 0.1  # 10cm standard deviation

    # Network layout - 4 anchors in square
    anchors_dict = {
        0: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([10.0, 10.0, 0.0, 0.0, 0.0]),
        3: np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    }
    anchors_array = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    # True position (center of square)
    true_pos = np.array([5.0, 5.0, 0.0, 0.0, 0.0])  # No bias for simplicity

    # Compute theoretical CRLB
    crlb_x, crlb_y = compute_crlb_position(anchors_array, true_pos[:2], range_std_m)

    print("=" * 60)
    print("CRLB VALIDATION TEST")
    print("=" * 60)
    print(f"Range std: {range_std_m*100:.1f} cm")
    print(f"True position: [{true_pos[0]:.1f}, {true_pos[1]:.1f}]")
    print(f"Number of trials: {n_trials}")
    print(f"\nTheoretical CRLB:")
    print(f"  σ_x = {crlb_x*100:.2f} cm")
    print(f"  σ_y = {crlb_y*100:.2f} cm")

    # Run Monte Carlo trials
    errors_x = []
    errors_y = []
    errors_bias = []

    for trial in range(n_trials):
        # Create solver
        config = OptimizationConfig(
            max_iterations=50,
            gradient_tol=1e-10
        )
        solver = SquareRootSolver(config)

        # Add anchors
        for id, state in anchors_dict.items():
            solver.add_node(id, state, is_anchor=True)

        # Add unknown with initial guess (perturbed from true)
        initial_guess = true_pos + np.random.randn(5) * np.array([1.0, 1.0, 0.1, 0, 0])
        solver.add_node(4, initial_guess, is_anchor=False)

        # Generate measurements with noise
        for anchor_id, anchor_pos in anchors_dict.items():
            # True range
            geometric_range = np.linalg.norm(true_pos[:2] - anchor_pos[:2])

            # Add measurement noise
            meas_range = geometric_range + np.random.normal(0, range_std_m)

            # Add to solver
            solver.add_toa_factor(anchor_id, 4, meas_range, range_std_m**2)

        # Optimize
        result = solver.optimize(verbose=False)

        if result.converged:
            est = result.estimates[4]
            errors_x.append(est[0] - true_pos[0])
            errors_y.append(est[1] - true_pos[1])
            errors_bias.append(est[2] - true_pos[2])

    # Compute empirical statistics
    emp_std_x = np.std(errors_x)
    emp_std_y = np.std(errors_y)
    emp_std_bias = np.std(errors_bias)
    emp_mean_x = np.mean(errors_x)
    emp_mean_y = np.mean(errors_y)

    print(f"\nEmpirical Results ({len(errors_x)} converged):")
    print(f"  Mean error: x={emp_mean_x*100:.2f} cm, y={emp_mean_y*100:.2f} cm")
    print(f"  Std dev:    x={emp_std_x*100:.2f} cm, y={emp_std_y*100:.2f} cm")
    print(f"  Bias std:   {emp_std_bias:.3f} ns")

    # Statistical test: empirical std should be close to CRLB
    print(f"\nCRLB Efficiency (empirical/theoretical):")
    eff_x = emp_std_x / crlb_x
    eff_y = emp_std_y / crlb_y
    print(f"  x-direction: {eff_x:.2f}")
    print(f"  y-direction: {eff_y:.2f}")

    # Check if solver is efficient (within 20% of CRLB)
    print("\nValidation:")
    all_good = True

    if abs(emp_mean_x) < 0.01 and abs(emp_mean_y) < 0.01:
        print("  ✓ Estimator is unbiased (mean error < 1cm)")
    else:
        print(f"  ✗ Estimator appears biased")
        all_good = False

    if 0.8 < eff_x < 1.3:
        print(f"  ✓ X-efficiency {eff_x:.2f} is close to optimal")
    else:
        print(f"  ✗ X-efficiency {eff_x:.2f} is not optimal")
        all_good = False

    if 0.8 < eff_y < 1.3:
        print(f"  ✓ Y-efficiency {eff_y:.2f} is close to optimal")
    else:
        print(f"  ✗ Y-efficiency {eff_y:.2f} is not optimal")
        all_good = False

    # Plot histogram if matplotlib available
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # X errors
        axes[0].hist(np.array(errors_x)*100, bins=30, density=True, alpha=0.7, label='Empirical')
        x_range = np.linspace(-3*crlb_x*100, 3*crlb_x*100, 100)
        axes[0].plot(x_range, 1/(crlb_x*100*np.sqrt(2*np.pi)) * np.exp(-0.5*(x_range/(crlb_x*100))**2),
                    'r-', label='CRLB', linewidth=2)
        axes[0].set_xlabel('X Error (cm)')
        axes[0].set_ylabel('Probability Density')
        axes[0].legend()
        axes[0].set_title(f'X-Position Errors (Efficiency={eff_x:.2f})')

        # Y errors
        axes[1].hist(np.array(errors_y)*100, bins=30, density=True, alpha=0.7, label='Empirical')
        y_range = np.linspace(-3*crlb_y*100, 3*crlb_y*100, 100)
        axes[1].plot(y_range, 1/(crlb_y*100*np.sqrt(2*np.pi)) * np.exp(-0.5*(y_range/(crlb_y*100))**2),
                    'r-', label='CRLB', linewidth=2)
        axes[1].set_xlabel('Y Error (cm)')
        axes[1].set_ylabel('Probability Density')
        axes[1].legend()
        axes[1].set_title(f'Y-Position Errors (Efficiency={eff_y:.2f})')

        plt.tight_layout()
        plt.savefig('crlb_validation.png', dpi=150)
        print("\n  Saved error distribution plot to crlb_validation.png")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    if all_good:
        print("SUCCESS: Solver achieves near-optimal CRLB performance!")
        print("The numerical fixes work correctly!")
    else:
        print("Issues detected - see details above")
    print("=" * 60)

    return emp_std_x, emp_std_y, crlb_x, crlb_y


def gdop_analysis():
    """Analyze Geometric Dilution of Precision across workspace"""

    # UWB parameters
    range_std_m = 0.1  # 10cm

    # Anchors in square
    anchors = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    # Grid of test points
    x = np.linspace(1, 9, 20)
    y = np.linspace(1, 9, 20)
    X, Y = np.meshgrid(x, y)

    # Compute CRLB at each point
    crlb_x_grid = np.zeros_like(X)
    crlb_y_grid = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            target = np.array([X[i,j], Y[i,j]])
            crlb_x, crlb_y = compute_crlb_position(anchors, target, range_std_m)
            crlb_x_grid[i,j] = crlb_x
            crlb_y_grid[i,j] = crlb_y

    # GDOP = sqrt(crlb_x^2 + crlb_y^2)
    gdop = np.sqrt(crlb_x_grid**2 + crlb_y_grid**2)

    print("\nGDOP ANALYSIS")
    print("=" * 60)
    print(f"Workspace: [{x.min():.0f}, {x.max():.0f}] x [{y.min():.0f}, {y.max():.0f}]")
    print(f"Best GDOP: {gdop.min()*100:.2f} cm at center")
    print(f"Worst GDOP: {gdop.max()*100:.2f} cm at corners")
    print(f"Mean GDOP: {gdop.mean()*100:.2f} cm")

    try:
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, gdop*100, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Position Error Bound (cm)')
        plt.scatter(anchors[:, 0], anchors[:, 1], c='red', s=200, marker='^', label='Anchors')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Geometric Dilution of Precision (GDOP) Map')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('gdop_map.png', dpi=150)
        print("Saved GDOP map to gdop_map.png")
    except ImportError:
        pass


if __name__ == "__main__":
    # Run CRLB validation
    monte_carlo_test(n_trials=100)

    # Analyze GDOP
    gdop_analysis()