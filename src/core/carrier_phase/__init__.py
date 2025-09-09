"""
Carrier Phase Measurement System for Millimeter-Accuracy Localization

This module provides carrier phase measurements with integer ambiguity resolution
for achieving millimeter-level ranging accuracy in sensor networks.
"""

from .phase_measurement import (
    CarrierPhaseConfig,
    PhaseMeasurement,
    CarrierPhaseMeasurementSystem
)

from .ambiguity_resolver import (
    AmbiguityResolutionResult,
    IntegerAmbiguityResolver
)

from .phase_unwrapper import (
    PhaseState,
    PhaseUnwrapper
)

__all__ = [
    'CarrierPhaseConfig',
    'PhaseMeasurement',
    'CarrierPhaseMeasurementSystem',
    'AmbiguityResolutionResult',
    'IntegerAmbiguityResolver',
    'PhaseState',
    'PhaseUnwrapper'
]