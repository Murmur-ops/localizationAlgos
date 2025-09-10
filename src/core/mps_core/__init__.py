"""
Matrix-Parametrized Proximal Splitting (MPS) for Sensor Network Localization
Simplified implementation based on the paper: "Decentralized Sensor Network Localization
using Matrix-Parametrized Proximal Splittings"
"""

from .algorithm import MPSAlgorithm, MPSConfig, MPSState
from .distributed import DistributedMPS
from .distributed_fixed import DistributedMPSFixed
from .proximal import ProximalOperators
from .matrix_ops import MatrixOperations

__all__ = [
    'MPSAlgorithm',
    'MPSConfig',
    'MPSState',
    'DistributedMPS',
    'DistributedMPSFixed',
    'ProximalOperators',
    'MatrixOperations'
]