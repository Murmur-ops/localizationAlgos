"""
Real Time and Frequency Synchronization Module for Distributed Localization

This module implements ACTUAL time synchronization using real measurements.
NO MOCK DATA - all timing measurements are from actual system execution.

Based on Nanzer et al. "Real-Time High-Accuracy Digital Wireless Time, 
Frequency, and Phase Calibration" but adapted for distributed localization.

Key Components:
- RealTWTT: Actual Two-Way Time Transfer implementation
- RealFrequencySync: Genuine frequency drift tracking
- RealClockConsensus: Real distributed clock consensus
"""

from .twtt import RealTWTT
from .frequency_sync import RealFrequencySync
from .consensus_clock import RealClockConsensus

__all__ = ['RealTWTT', 'RealFrequencySync', 'RealClockConsensus']