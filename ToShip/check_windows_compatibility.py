#!/usr/bin/env python3
"""
Check Windows compatibility and provide setup instructions
"""

import sys
import platform
import subprocess
import importlib

def check_compatibility():
    """Check if the codebase can run on current platform"""
    
    print("="*60)
    print("Sensor Network Localization - Compatibility Check")
    print("="*60)
    
    # Platform info
    print(f"\nPlatform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check core dependencies
    print("\nCore Dependencies:")
    core_deps = ['numpy', 'scipy', 'matplotlib']
    all_good = True
    
    for dep in core_deps:
        try:
            mod = importlib.import_module(dep)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {dep}: {version}")
        except ImportError:
            print(f"✗ {dep}: NOT INSTALLED")
            all_good = False
    
    # Check MPI (optional)
    print("\nMPI Support (optional for production):")
    try:
        import mpi4py
        print(f"✓ mpi4py: {mpi4py.__version__}")
        
        # Check MPI executable
        if platform.system() == 'Windows':
            mpi_exe = 'mpiexec'
        else:
            mpi_exe = 'mpirun'
            
        try:
            result = subprocess.run([mpi_exe, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {mpi_exe}: Available")
            else:
                print(f"⚠ {mpi_exe}: Not found in PATH")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠ {mpi_exe}: Not available")
            
    except ImportError:
        print("⚠ mpi4py: Not installed (required for MPI version)")
        print("  For Windows: Install MS-MPI, then 'pip install mpi4py'")
        print("  For Linux/Mac: 'pip install mpi4py'")
    
    # Recommendations
    print("\n" + "="*60)
    print("Recommendations:")
    print("="*60)
    
    if platform.system() == 'Windows':
        print("\nFor Windows users:")
        print("1. Basic usage (no MPI):")
        print("   - Use snl_threaded_standalone.py")
        print("   - Limited to small networks (<50 sensors)")
        print("\n2. Production usage (with MPI):")
        print("   Option A: Install Microsoft MPI")
        print("   - Download: https://www.microsoft.com/en-us/download/details.aspx?id=100593")
        print("   - Then: pip install mpi4py")
        print("\n   Option B: Use WSL2")
        print("   - Install Ubuntu in WSL2")
        print("   - Follow Linux instructions")
        
    elif platform.system() == 'Darwin':  # macOS
        print("\nFor macOS users:")
        print("- Install MPI: brew install mpich")
        print("- Install mpi4py: pip install mpi4py")
        
    else:  # Linux
        print("\nFor Linux users:")
        print("- Install MPI: sudo apt-get install mpich")
        print("- Install mpi4py: pip install mpi4py")
    
    # Test files
    print("\n" + "="*60)
    print("Runnable Components:")
    print("="*60)
    
    if all_good:
        print("\n✓ Threading version (works everywhere):")
        print("  python snl_threaded_standalone.py")
        print("\n✓ Visualization scripts:")
        print("  python generate_figures.py")
        print("  python crlb_analysis.py")
        
        try:
            import mpi4py
            print("\n✓ MPI version (requires MPI setup):")
            print("  mpirun -np 4 python snl_mpi_optimized.py")
        except ImportError:
            print("\n⚠ MPI version requires mpi4py installation")
    else:
        print("\n⚠ Please install missing core dependencies first:")
        print("  pip install numpy scipy matplotlib")

if __name__ == "__main__":
    check_compatibility()