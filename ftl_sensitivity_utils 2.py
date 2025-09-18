#!/usr/bin/env python3
"""
Utility functions for FTL sensitivity analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.localization.robust_solver import RobustLocalizer, MeasurementEdge


def compute_cramer_rao_bound(
    snr_linear: float,
    bandwidth_hz: float,
    n_measurements: int,
    carrier_freq_hz: float = 2.4e9
) -> float:
    """
    Compute the Cramér-Rao lower bound for ranging accuracy

    The CRB gives the theoretical minimum variance achievable by any unbiased estimator.
    For TOA-based ranging: σ²_d = c² / (8π² * β² * SNR)

    Args:
        snr_linear: Linear SNR (not dB)
        bandwidth_hz: Signal bandwidth in Hz
        n_measurements: Number of independent measurements
        carrier_freq_hz: Carrier frequency in Hz

    Returns:
        Standard deviation lower bound in meters
    """
    c = 3e8  # Speed of light

    # Effective bandwidth (RMS bandwidth for rectangular spectrum)
    beta_rms = bandwidth_hz / np.sqrt(3)

    # Single measurement variance (TOA)
    # σ²_τ = 1 / (8π² * β² * SNR)
    var_tau = 1 / (8 * np.pi**2 * beta_rms**2 * snr_linear)

    # Convert time variance to distance variance
    var_distance = c**2 * var_tau

    # With multiple measurements, variance reduces by sqrt(N)
    var_combined = var_distance / n_measurements

    return np.sqrt(var_combined)


def calculate_gdop(anchors: Dict, unknowns: Dict) -> float:
    """
    Calculate Geometric Dilution of Precision (GDOP)

    GDOP quantifies how anchor geometry amplifies ranging errors.
    GDOP = sqrt(trace((H^T H)^-1)) where H is the geometry matrix.

    Args:
        anchors: Dictionary of anchor positions {id: [x, y]}
        unknowns: Dictionary of unknown positions {id: [x, y]}

    Returns:
        Average GDOP over all unknown nodes
    """
    gdop_values = []

    for uid, u_pos in unknowns.items():
        # Build geometry matrix H for this unknown
        H = []

        for aid, a_pos in anchors.items():
            # Direction vector from unknown to anchor
            diff = a_pos - u_pos
            distance = np.linalg.norm(diff)

            if distance > 0:
                # Normalized direction vector
                h_row = diff / distance
                H.append(h_row)

        if len(H) >= 2:  # Need at least 2 anchors for 2D
            H = np.array(H)

            try:
                # GDOP = sqrt(trace((H^T H)^-1))
                HTH = H.T @ H
                HTH_inv = np.linalg.inv(HTH)
                gdop = np.sqrt(np.trace(HTH_inv))
                gdop_values.append(gdop)
            except np.linalg.LinAlgError:
                # Singular matrix, poor geometry
                gdop_values.append(100.0)  # Large penalty
        else:
            gdop_values.append(100.0)  # Insufficient anchors

    return np.mean(gdop_values) if gdop_values else float('inf')


def inject_clock_error(
    offset_ns: float,
    drift_ppb: float = 0,
    time_elapsed_s: float = 0
) -> float:
    """
    Calculate ranging error due to clock synchronization errors

    Clock errors directly translate to ranging errors:
    - 1 ns clock offset = 0.3m ranging error
    - Clock drift accumulates over time

    Args:
        offset_ns: Static clock offset in nanoseconds
        drift_ppb: Clock drift rate in parts per billion
        time_elapsed_s: Time since synchronization in seconds

    Returns:
        Ranging error in meters
    """
    c = 3e8  # Speed of light

    # Static offset contribution
    offset_error_m = offset_ns * 1e-9 * c

    # Drift contribution (accumulates over time)
    drift_error_ns = drift_ppb * 1e-9 * time_elapsed_s * 1e9
    drift_error_m = drift_error_ns * 1e-9 * c

    return offset_error_m + drift_error_m


def generate_test_network(
    config: dict,
    geometry_type: str = 'uniform'
) -> Tuple[Dict, Dict]:
    """
    Generate test network with specified geometry

    Args:
        config: Configuration dictionary
        geometry_type: Type of node placement
            - 'uniform': Uniformly random
            - 'grid': Regular grid
            - 'clustered': Clustered distribution

    Returns:
        anchors: Dictionary of anchor positions
        unknowns: Dictionary of unknown positions
    """
    n_anchors = config['system']['num_anchors']
    n_unknowns = config['system']['num_unknowns']
    area_size = config['system']['area_size_m']

    anchors = {}
    unknowns = {}

    if geometry_type == 'uniform':
        # Place anchors at corners first (optimal for convex hull)
        if n_anchors >= 4:
            anchors[0] = np.array([0, 0])
            anchors[1] = np.array([area_size, 0])
            anchors[2] = np.array([area_size, area_size])
            anchors[3] = np.array([0, area_size])

            # Additional anchors randomly
            for i in range(4, n_anchors):
                anchors[i] = np.random.uniform(0, area_size, 2)
        else:
            # Random placement if fewer than 4 anchors
            for i in range(n_anchors):
                anchors[i] = np.random.uniform(0, area_size, 2)

        # Place unknowns randomly
        for i in range(n_unknowns):
            unknowns[n_anchors + i] = np.random.uniform(0, area_size, 2)

    elif geometry_type == 'grid':
        # Regular grid placement
        grid_size = int(np.sqrt(n_anchors + n_unknowns))

        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * area_size / (grid_size - 1)
                y = j * area_size / (grid_size - 1)
                positions.append([x, y])

        # First n_anchors positions are anchors
        for i in range(min(n_anchors, len(positions))):
            anchors[i] = np.array(positions[i])

        # Remaining are unknowns
        for i in range(min(n_unknowns, len(positions) - n_anchors)):
            unknowns[n_anchors + i] = np.array(positions[n_anchors + i])

    elif geometry_type == 'clustered':
        # Clustered distribution (challenging for localization)
        n_clusters = 3
        cluster_centers = np.random.uniform(0.2 * area_size, 0.8 * area_size, (n_clusters, 2))
        cluster_std = area_size / 10

        # Distribute anchors across clusters
        for i in range(n_anchors):
            cluster = i % n_clusters
            anchors[i] = np.random.normal(cluster_centers[cluster], cluster_std)
            anchors[i] = np.clip(anchors[i], 0, area_size)

        # Distribute unknowns across clusters
        for i in range(n_unknowns):
            cluster = np.random.randint(n_clusters)
            unknowns[n_anchors + i] = np.random.normal(cluster_centers[cluster], cluster_std)
            unknowns[n_anchors + i] = np.clip(unknowns[n_anchors + i], 0, area_size)

    return anchors, unknowns


def run_localization_trial(
    measurements: List[MeasurementEdge],
    anchors: Dict,
    unknowns: Dict,
    config: dict
) -> Tuple[float, int]:
    """
    Run a single localization trial and compute RMSE

    Args:
        measurements: List of ranging measurements
        anchors: Dictionary of anchor positions
        unknowns: Dictionary of true unknown positions
        config: Configuration dictionary

    Returns:
        rmse: Root mean square error in meters
        iterations: Number of iterations to convergence
    """
    # Initialize solver
    solver = RobustLocalizer(
        dimension=2,
        huber_delta=config['solver']['huber_delta']
    )
    solver.max_iterations = config['solver']['max_iterations']
    solver.convergence_threshold = config['system']['convergence_threshold']

    # Initialize unknowns at center
    n_unknowns = len(unknowns)
    center = config['system']['area_size_m'] / 2
    initial_positions = np.ones(n_unknowns * 2) * center

    # Remap IDs for solver (expects sequential IDs)
    unknown_ids = sorted(unknowns.keys())
    id_mapping = {aid: aid for aid in anchors}
    for i, uid in enumerate(unknown_ids):
        id_mapping[uid] = len(anchors) + i

    # Remap measurements
    remapped_measurements = []
    for m in measurements:
        if m.node_i in id_mapping and m.node_j in id_mapping:
            remapped = MeasurementEdge(
                node_i=id_mapping[m.node_i],
                node_j=id_mapping[m.node_j],
                distance=m.distance,
                quality=m.quality,
                variance=m.variance
            )
            remapped_measurements.append(remapped)

    remapped_anchors = {id_mapping[aid]: anchors[aid] for aid in anchors}

    try:
        # Solve
        optimized_positions, info = solver.solve(
            initial_positions,
            remapped_measurements,
            remapped_anchors
        )

        # Calculate RMSE
        errors = []
        for i, uid in enumerate(unknown_ids):
            est_pos = optimized_positions[i*2:(i+1)*2]
            true_pos = unknowns[uid]
            error = np.linalg.norm(est_pos - true_pos)
            errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        iterations = info.get('iterations', solver.max_iterations)

        return rmse, iterations

    except Exception as e:
        print(f"Localization failed: {e}")
        return float('inf'), solver.max_iterations


def analyze_measurement_quality(
    measurements: List[MeasurementEdge],
    true_distances: Dict[Tuple[int, int], float]
) -> Dict:
    """
    Analyze the quality of ranging measurements

    Args:
        measurements: List of ranging measurements
        true_distances: Dictionary of true distances {(node_i, node_j): distance}

    Returns:
        Dictionary with quality metrics
    """
    errors = []
    biases = []
    variances = []

    for m in measurements:
        key = (m.node_i, m.node_j)
        if key in true_distances:
            true_dist = true_distances[key]
            error = m.distance - true_dist
            errors.append(error)
            biases.append(error)  # For bias calculation
            variances.append(m.variance)

    errors = np.array(errors)
    biases = np.array(biases)
    variances = np.array(variances)

    metrics = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'bias': np.mean(biases),
        'variance': np.mean(variances),
        'max_error': np.max(np.abs(errors)),
        'percentile_95': np.percentile(np.abs(errors), 95)
    }

    return metrics


def compute_fisher_information_matrix(
    anchors: Dict,
    unknown_pos: np.ndarray,
    measurement_variance: float
) -> np.ndarray:
    """
    Compute Fisher Information Matrix for localization

    The FIM quantifies the information content in measurements.
    Its inverse gives the Cramér-Rao bound covariance matrix.

    Args:
        anchors: Dictionary of anchor positions
        unknown_pos: Position of unknown node [x, y]
        measurement_variance: Variance of ranging measurements

    Returns:
        2x2 Fisher Information Matrix
    """
    FIM = np.zeros((2, 2))

    for aid, a_pos in anchors.items():
        # Direction vector from unknown to anchor
        diff = a_pos - unknown_pos
        distance = np.linalg.norm(diff)

        if distance > 0:
            # Normalized direction vector
            h = diff / distance

            # Fisher information contribution
            # FIM += (1/σ²) * h * h^T
            FIM += (1 / measurement_variance) * np.outer(h, h)

    return FIM


def estimate_localization_accuracy(
    anchors: Dict,
    unknown_pos: np.ndarray,
    snr_db: float,
    bandwidth_hz: float
) -> Dict:
    """
    Estimate theoretical localization accuracy using CRB

    Args:
        anchors: Dictionary of anchor positions
        unknown_pos: Position of unknown node
        snr_db: Signal-to-noise ratio in dB
        bandwidth_hz: Signal bandwidth in Hz

    Returns:
        Dictionary with accuracy estimates
    """
    # Convert SNR to linear
    snr_linear = 10 ** (snr_db / 10)

    # Ranging variance from CRB
    c = 3e8
    beta_rms = bandwidth_hz / np.sqrt(3)
    var_ranging = c**2 / (8 * np.pi**2 * beta_rms**2 * snr_linear)

    # Fisher Information Matrix
    FIM = compute_fisher_information_matrix(anchors, unknown_pos, var_ranging)

    try:
        # CRB covariance matrix
        CRB_cov = np.linalg.inv(FIM)

        # Extract accuracy metrics
        var_x = CRB_cov[0, 0]
        var_y = CRB_cov[1, 1]
        cov_xy = CRB_cov[0, 1]

        # Position RMSE
        rmse = np.sqrt(var_x + var_y)

        # Circular error probable (50% of estimates within CEP)
        cep = 0.59 * (np.sqrt(var_x) + np.sqrt(var_y))

        # 95% confidence ellipse
        eigenvalues = np.linalg.eigvalsh(CRB_cov)
        conf_95_major = 2.45 * np.sqrt(max(eigenvalues))
        conf_95_minor = 2.45 * np.sqrt(min(eigenvalues))

        return {
            'rmse': rmse,
            'cep': cep,
            'std_x': np.sqrt(var_x),
            'std_y': np.sqrt(var_y),
            'correlation': cov_xy / (np.sqrt(var_x) * np.sqrt(var_y)),
            'conf_95_major': conf_95_major,
            'conf_95_minor': conf_95_minor
        }

    except np.linalg.LinAlgError:
        # Singular FIM, poor observability
        return {
            'rmse': float('inf'),
            'cep': float('inf'),
            'std_x': float('inf'),
            'std_y': float('inf'),
            'correlation': 0,
            'conf_95_major': float('inf'),
            'conf_95_minor': float('inf')
        }