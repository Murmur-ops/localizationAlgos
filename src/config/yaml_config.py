"""
YAML Configuration Manager for FTL System
Provides centralized configuration loading and validation
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SystemConfig:
    """System-wide configuration"""
    seed: int = 42
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SystemConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    # Timing
    timing_precision: str = "ns"  # "ps", "ns", "us"
    timestamp_resolution_ns: float = 1.0
    clock_stability_ppm: float = 20.0
    
    # RF
    carrier_freq_ghz: float = 2.45
    bandwidth_mhz: float = 200.0
    tx_power_dbm: float = 20.0
    antenna_gain_dbi: float = 6.0
    noise_figure_db: float = 5.0
    
    # Sampling
    sample_rate_msps: float = 250.0
    adc_bits: int = 12
    
    @property
    def timing_precision_seconds(self) -> float:
        """Get timing precision in seconds"""
        if self.timing_precision == "ps":
            return 1e-12
        elif self.timing_precision == "ns":
            return 1e-9
        elif self.timing_precision == "us":
            return 1e-6
        else:
            return float(self.timing_precision)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HardwareConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class NetworkConfig:
    """Network topology configuration"""
    topology: str = "nearest_neighbor"  # "full_mesh", "nearest_neighbor", "star"
    k_neighbors: int = 5
    max_neighbor_distance_m: float = 20.0
    min_rssi_dbm: float = -80.0
    
    @classmethod
    def from_dict(cls, d: dict) -> 'NetworkConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class SolverConfig:
    """Solver configuration"""
    algorithm: str = "admm"  # "admm", "levenberg_marquardt", "gauss_newton"
    huber_delta: float = 1.0
    rho: float = 1.0  # ADMM penalty parameter
    use_quality_weights: bool = True
    outlier_threshold: float = 3.0
    damping: float = 0.5
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SolverConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class NodeConfig:
    """Individual node configuration"""
    id: int
    position: List[float]
    is_anchor: bool
    name: Optional[str] = None
    hardware_override: Optional[Dict] = None
    
    @property
    def position_array(self) -> np.ndarray:
        return np.array(self.position)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'NodeConfig':
        return cls(**d)


class FTLConfig:
    """Main configuration manager for FTL system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.raw_config = {}
        
        # Sub-configurations
        self.system = SystemConfig()
        self.hardware = HardwareConfig()
        self.network = NetworkConfig()
        self.solver = SolverConfig()
        self.nodes: List[NodeConfig] = []
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
        
        # Parse sub-configurations
        if 'system' in self.raw_config:
            self.system = SystemConfig.from_dict(self.raw_config['system'])
        
        if 'hardware' in self.raw_config:
            self.hardware = HardwareConfig.from_dict(self.raw_config['hardware'])
        
        if 'network' in self.raw_config:
            self.network = NetworkConfig.from_dict(self.raw_config['network'])
        
        if 'solver' in self.raw_config:
            self.solver = SolverConfig.from_dict(self.raw_config['solver'])
        
        if 'nodes' in self.raw_config:
            self.nodes = [NodeConfig.from_dict(n) for n in self.raw_config['nodes']]
        
        self.config_path = config_path
    
    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save to (uses original path if not specified)
        """
        if output_path is None and self.config_path is None:
            raise ValueError("No output path specified")
        
        output_path = Path(output_path or self.config_path)
        
        # Build config dict
        config = {
            'system': {
                'seed': self.system.seed,
                'max_iterations': self.system.max_iterations,
                'convergence_threshold': self.system.convergence_threshold,
                'verbose': self.system.verbose
            },
            'hardware': {
                'timing_precision': self.hardware.timing_precision,
                'timestamp_resolution_ns': self.hardware.timestamp_resolution_ns,
                'clock_stability_ppm': self.hardware.clock_stability_ppm,
                'carrier_freq_ghz': self.hardware.carrier_freq_ghz,
                'bandwidth_mhz': self.hardware.bandwidth_mhz,
                'tx_power_dbm': self.hardware.tx_power_dbm,
                'antenna_gain_dbi': self.hardware.antenna_gain_dbi,
                'noise_figure_db': self.hardware.noise_figure_db,
                'sample_rate_msps': self.hardware.sample_rate_msps,
                'adc_bits': self.hardware.adc_bits
            },
            'network': {
                'topology': self.network.topology,
                'k_neighbors': self.network.k_neighbors,
                'max_neighbor_distance_m': self.network.max_neighbor_distance_m,
                'min_rssi_dbm': self.network.min_rssi_dbm
            },
            'solver': {
                'algorithm': self.solver.algorithm,
                'huber_delta': self.solver.huber_delta,
                'rho': self.solver.rho,
                'use_quality_weights': self.solver.use_quality_weights,
                'outlier_threshold': self.solver.outlier_threshold,
                'damping': self.solver.damping
            },
            'nodes': [
                {
                    'id': n.id,
                    'position': n.position,
                    'is_anchor': n.is_anchor,
                    'name': n.name
                } for n in self.nodes
            ]
        }
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get_anchor_positions(self) -> Dict[int, np.ndarray]:
        """Get dictionary of anchor positions"""
        return {
            node.id: node.position_array 
            for node in self.nodes 
            if node.is_anchor
        }
    
    def get_unknown_nodes(self) -> List[NodeConfig]:
        """Get list of unknown (non-anchor) nodes"""
        return [node for node in self.nodes if not node.is_anchor]
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for anchors
        anchors = [n for n in self.nodes if n.is_anchor]
        if len(anchors) < 3:
            errors.append(f"Need at least 3 anchors, got {len(anchors)}")
        
        # Check node IDs are unique
        node_ids = [n.id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        # Check timing precision
        if self.hardware.timing_precision not in ["ps", "ns", "us"]:
            try:
                float(self.hardware.timing_precision)
            except ValueError:
                errors.append(f"Invalid timing precision: {self.hardware.timing_precision}")
        
        # Check network topology
        if self.network.topology == "nearest_neighbor":
            if self.network.k_neighbors < 2:
                errors.append(f"k_neighbors must be >= 2, got {self.network.k_neighbors}")
        
        return errors
    
    def summary(self) -> str:
        """Get configuration summary"""
        num_anchors = sum(1 for n in self.nodes if n.is_anchor)
        num_unknown = len(self.nodes) - num_anchors
        
        return f"""
FTL Configuration Summary
========================
Nodes: {len(self.nodes)} ({num_anchors} anchors, {num_unknown} unknown)
Hardware: {self.hardware.timing_precision} timing, {self.hardware.bandwidth_mhz}MHz BW
Network: {self.network.topology} topology
Solver: {self.solver.algorithm} algorithm
Config file: {self.config_path}
"""


def create_example_config(output_path: str = "configs/example.yaml"):
    """Create an example configuration file"""
    config = FTLConfig()
    
    # System settings
    config.system.max_iterations = 50
    config.system.convergence_threshold = 1e-5
    
    # Hardware (picosecond timing example)
    config.hardware.timing_precision = "ps"
    config.hardware.bandwidth_mhz = 200.0
    config.hardware.carrier_freq_ghz = 2.45
    
    # Network (nearest neighbor)
    config.network.topology = "nearest_neighbor"
    config.network.k_neighbors = 5
    
    # Solver
    config.solver.algorithm = "admm"
    config.solver.rho = 1.0
    
    # Create a simple 4-anchor, 2-unknown network
    config.nodes = [
        NodeConfig(0, [0, 0], True, "Anchor_0"),
        NodeConfig(1, [10, 0], True, "Anchor_1"),
        NodeConfig(2, [10, 10], True, "Anchor_2"),
        NodeConfig(3, [0, 10], True, "Anchor_3"),
        NodeConfig(4, [3, 5], False, "Unknown_4"),
        NodeConfig(5, [7, 5], False, "Unknown_5"),
    ]
    
    config.save(output_path)
    print(f"Example config saved to {output_path}")
    return config


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FTL Configuration Manager")
    parser.add_argument("action", choices=["validate", "summary", "create_example"],
                       help="Action to perform")
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--output", help="Output path for create_example")
    
    args = parser.parse_args()
    
    if args.action == "create_example":
        output = args.output or "configs/example.yaml"
        config = create_example_config(output)
        print(config.summary())
        
    elif args.action in ["validate", "summary"]:
        if not args.config:
            print("Error: --config required")
            exit(1)
        
        config = FTLConfig(args.config)
        
        if args.action == "validate":
            errors = config.validate()
            if errors:
                print("Validation errors:")
                for error in errors:
                    print(f"  - {error}")
                exit(1)
            else:
                print("✓ Configuration is valid")
        
        elif args.action == "summary":
            print(config.summary())