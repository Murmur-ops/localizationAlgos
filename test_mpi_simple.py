#!/usr/bin/env python3
"""
Simple MPI test for the MPS algorithm
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Test imports
try:
    from mpi4py import MPI
    print(f"✓ MPI imported successfully")
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    print(f"  Process {rank}/{size}")
    
except ImportError as e:
    print(f"✗ MPI import failed: {e}")
    sys.exit(1)

try:
    from src.core.mps_core.config_loader import ConfigLoader
    print(f"✓ ConfigLoader imported (rank {rank})")
except ImportError as e:
    print(f"✗ ConfigLoader import failed: {e}")
    sys.exit(1)

try:
    from src.core.mps_core.mps_distributed import DistributedMPS
    print(f"✓ DistributedMPS imported (rank {rank})")
except ImportError as e:
    print(f"✗ DistributedMPS import failed: {e}")
    sys.exit(1)

try:
    from src.core.mps_core.mps_full_algorithm import create_network_data
    print(f"✓ Network utilities imported (rank {rank})")
except ImportError as e:
    print(f"✗ Network utilities import failed: {e}")
    sys.exit(1)

# Test basic functionality
if rank == 0:
    print("\nTesting basic MPS distributed functionality...")
    
    # Create small test network
    network = create_network_data(
        n_sensors=10,
        n_anchors=3,
        dimension=2,
        communication_range=0.5,
        measurement_noise=0.01
    )
    print("✓ Network created successfully")
else:
    network = None

# Broadcast network
network = comm.bcast(network, root=0)

if rank == 0:
    print("✓ Network broadcast successful")
    
# Test config loading
if rank == 0:
    loader = ConfigLoader()
    config = loader.load_config("configs/mpi/mpi_small.yaml")
    print("✓ Configuration loaded successfully")
    
    # Quick validation
    assert config['mpi']['enable'] == True
    assert config['network']['n_sensors'] == 20
    print("✓ Configuration validated")
    
comm.Barrier()

if rank == 0:
    print("\n✅ All MPI tests passed!")
    print(f"MPI is working with {size} processes")