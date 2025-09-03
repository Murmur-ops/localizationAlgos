#!/usr/bin/env python3
"""
Test Installation Script
========================
Verifies that all components are properly installed and working
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    success = True
    
    # Core dependencies
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        success = False
    
    try:
        import scipy
        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy: {e}")
        success = False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        success = False
    
    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML: {e}")
        success = False
    
    # Optional dependencies
    try:
        import mpi4py
        print(f"  ✓ mpi4py {mpi4py.__version__} (optional)")
    except ImportError:
        print("  ⚠ mpi4py not installed (optional, needed for distributed mode)")
    
    try:
        import networkx as nx
        print(f"  ✓ NetworkX {nx.__version__}")
    except ImportError as e:
        print(f"  ✗ NetworkX: {e}")
        success = False
    
    try:
        import cvxpy as cp
        print(f"  ✓ CVXPY {cp.__version__}")
    except ImportError as e:
        print(f"  ✗ CVXPY: {e}")
        success = False
    
    return success

def test_modules():
    """Test project modules"""
    print("\nTesting project modules...")
    success = True
    
    try:
        from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig
        print("  ✓ MPS Core modules")
    except ImportError as e:
        print(f"  ✗ MPS Core: {e}")
        success = False
    
    try:
        from src.core.algorithms.mps_proper import ProperMPSAlgorithm
        print("  ✓ MPS Proper algorithm")
    except ImportError as e:
        print(f"  ✗ MPS Proper: {e}")
        success = False
    
    try:
        from src.core.algorithms.admm import DecentralizedADMM
        print("  ✓ ADMM algorithm")
    except ImportError as e:
        print(f"  ✗ ADMM: {e}")
        success = False
    
    return success

def test_configurations():
    """Test configuration files"""
    print("\nTesting configuration files...")
    success = True
    
    config_files = [
        "configs/quick_test.yaml",
        "configs/research_comparison.yaml",
        "configs/distributed_large.yaml",
        "configs/high_accuracy.yaml",
        "configs/sband_precision.yaml"
    ]
    
    import yaml
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"  ✓ {config_file}")
        except Exception as e:
            print(f"  ✗ {config_file}: {e}")
            success = False
    
    return success

def test_simple_run():
    """Test a simple MPS run"""
    print("\nTesting simple MPS run...")
    
    try:
        from src.core.mps_core.algorithm import MPSAlgorithm, MPSConfig
        
        # Create simple config
        config = MPSConfig(
            n_sensors=10,
            n_anchors=3,
            communication_range=0.5,
            noise_factor=0.01,
            max_iterations=100,
            tolerance=1e-4,
            gamma=0.99,
            alpha=1.0,
            dimension=2,
            seed=42
        )
        
        # Create and run algorithm
        algorithm = MPSAlgorithm(config)
        results = algorithm.run()
        
        if results['converged']:
            print(f"  ✓ Algorithm converged in {results['iterations']} iterations")
        else:
            print(f"  ⚠ Algorithm did not converge (ran {results['iterations']} iterations)")
        
        print(f"  ✓ Final objective: {results['final_objective']:.6f}")
        
        if results['final_rmse'] is not None:
            print(f"  ✓ Final RMSE: {results['final_rmse']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def check_directories():
    """Check that necessary directories exist"""
    print("\nChecking directories...")
    dirs = ['data', 'results', 'results/figures', 'results/data', 'configs', 'scripts', 'src']
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")

def main():
    """Main test function"""
    print("="*60)
    print("DECENTRALIZEDLOCALE INSTALLATION TEST")
    print("="*60)
    
    print(f"\nPython version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    tests_passed = []
    
    tests_passed.append(("Dependencies", test_imports()))
    tests_passed.append(("Modules", test_modules()))
    tests_passed.append(("Configurations", test_configurations()))
    check_directories()
    tests_passed.append(("Simple Run", test_simple_run()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Installation successful!")
        print("\nYou can now run:")
        print("  python scripts/run_mps.py --config configs/quick_test.yaml")
        print("  python experiments/run_comparison.py")
        print("  python src/simulation/src/run_phase_sync_simulation.py")
        print("\nFor MPI (if installed):")
        print("  mpirun -n 2 python scripts/run_distributed.py")
    else:
        print("\n✗ SOME TESTS FAILED - Please check the errors above")
        print("\nTry running:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()