"""
YAML Configuration Loader for MPS Algorithm
Handles loading, validation, and merging of YAML configurations
"""

import yaml
import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path
import re

from .mps_full_algorithm import MPSConfig
from .mps_distributed import DistributedMPSConfig


class ConfigLoader:
    """
    Loads and validates YAML configuration files for MPS algorithm
    Supports inheritance, environment variables, and mathematical expressions
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            base_path: Base directory for resolving relative paths
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config_cache = {}
        
    def load_config(self, config_path: Union[str, Path], 
                   overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with support for inheritance
        
        Args:
            config_path: Path to YAML configuration file
            overrides: Dictionary of parameter overrides
            
        Returns:
            Merged configuration dictionary
        """
        config_path = Path(config_path)
        
        # Check cache
        cache_key = str(config_path.absolute())
        if cache_key in self.config_cache and not overrides:
            return self.config_cache[cache_key].copy()
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if 'extends' in config:
            base_configs = config.pop('extends')
            if not isinstance(base_configs, list):
                base_configs = [base_configs]
            
            # Load base configurations
            merged_config = {}
            for base_config in base_configs:
                base_path = self._resolve_path(base_config, config_path.parent)
                base = self.load_config(base_path)
                merged_config = self._deep_merge(merged_config, base)
            
            # Merge current config on top
            config = self._deep_merge(merged_config, config)
        
        # Process environment variables
        config = self._substitute_env_vars(config)
        
        # Evaluate mathematical expressions
        config = self._evaluate_expressions(config)
        
        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        # Validate configuration
        self._validate_config(config)
        
        # Cache result
        self.config_cache[cache_key] = config.copy()
        
        return config
    
    def load_multiple_configs(self, config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files
        
        Args:
            config_paths: List of configuration file paths
            
        Returns:
            Merged configuration
        """
        merged = {}
        for path in config_paths:
            config = self.load_config(path)
            merged = self._deep_merge(merged, config)
        return merged
    
    def to_mps_config(self, config_dict: Dict[str, Any], 
                      distributed: bool = False) -> Union[MPSConfig, DistributedMPSConfig]:
        """
        Convert configuration dictionary to MPSConfig object
        
        Args:
            config_dict: Configuration dictionary
            distributed: Whether to create distributed config
            
        Returns:
            MPSConfig or DistributedMPSConfig object
        """
        # Extract relevant sections
        network = config_dict.get('network', {})
        algorithm = config_dict.get('algorithm', {})
        measurements = config_dict.get('measurements', {})
        admm = config_dict.get('admm', {})
        mpi = config_dict.get('mpi', {})
        
        # Build config parameters
        params = {
            'n_sensors': network.get('n_sensors', 10),
            'n_anchors': network.get('n_anchors', 3),
            'dimension': network.get('dimension', 2),
            'communication_range': network.get('communication_range', 0.3),
            'scale': network.get('scale', 1.0),
            'gamma': algorithm.get('gamma', 0.999),
            'alpha': algorithm.get('alpha', 10.0),
            'max_iterations': algorithm.get('max_iterations', 1000),
            'tolerance': float(algorithm.get('tolerance', 1e-6)),
            'verbose': algorithm.get('verbose', False),
            'early_stopping': algorithm.get('early_stopping', True),
            'early_stopping_window': algorithm.get('early_stopping_window', 100),
            'admm_iterations': admm.get('iterations', 100),
            'admm_tolerance': float(admm.get('tolerance', 1e-6)),
            'admm_rho': admm.get('rho', 1.0),
            'warm_start': admm.get('warm_start', True),
            'parallel_proximal': algorithm.get('parallel_proximal', False),
            'use_2block': algorithm.get('use_2block', True),
            'adaptive_alpha': algorithm.get('adaptive_alpha', False),
            'carrier_phase_mode': measurements.get('carrier_phase', False)
        }
        
        if distributed:
            # Add MPI-specific parameters
            params.update({
                'async_communication': mpi.get('async_communication', False),
                'buffer_size_kb': mpi.get('buffer_size_kb', 1024),
                'collective_operations': mpi.get('collective_operations', True),
                'checkpoint_interval': mpi.get('checkpoint_interval', 100),
                'load_balancing': mpi.get('load_balancing', 'block')
            })
            return DistributedMPSConfig(**params)
        else:
            return MPSConfig(**params)
    
    def save_config(self, config: Union[Dict[str, Any], MPSConfig], 
                   output_path: Union[str, Path]):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary or MPSConfig object
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        # Convert dataclass to dict if needed
        if isinstance(config, (MPSConfig, DistributedMPSConfig)):
            config = asdict(config)
        
        # Format for readability
        formatted = self._format_config_for_yaml(config)
        
        # Write YAML
        with open(output_path, 'w') as f:
            yaml.dump(formatted, f, default_flow_style=False, sort_keys=False)
    
    def validate_schema(self, config: Dict[str, Any], schema_path: Optional[str] = None):
        """
        Validate configuration against schema
        
        Args:
            config: Configuration dictionary
            schema_path: Optional path to JSON schema file
        """
        # Basic validation rules
        required_sections = ['network', 'algorithm']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Network validation
        network = config['network']
        if network.get('n_sensors', 0) <= 0:
            raise ValueError("n_sensors must be positive")
        if network.get('n_anchors', 0) < 0:
            raise ValueError("n_anchors must be non-negative")
        if network.get('n_anchors', 0) > network.get('n_sensors', 0):
            raise ValueError("n_anchors cannot exceed n_sensors")
        
        # Algorithm validation
        algorithm = config['algorithm']
        gamma = algorithm.get('gamma', 0.999)
        if not (0 < gamma < 1):
            raise ValueError("gamma must be in (0, 1)")
        
        alpha = algorithm.get('alpha', 10.0)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # MPI validation if present
        if 'mpi' in config and config['mpi'].get('enable', False):
            mpi = config['mpi']
            if mpi.get('buffer_size_kb', 0) <= 0:
                raise ValueError("buffer_size_kb must be positive")
    
    def _resolve_path(self, path: str, relative_to: Path) -> Path:
        """Resolve path relative to given directory"""
        path = Path(path)
        if not path.is_absolute():
            # Check relative to config file first
            resolved = relative_to / path
            if resolved.exists():
                return resolved
            # Then check relative to base path
            resolved = self.base_path / path
            if resolved.exists():
                return resolved
            # Default to relative to config file
            return relative_to / path
        return path
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Look for ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default if default else match.group(0))
            
            return re.sub(pattern, replacer, config)
        else:
            return config
    
    def _evaluate_expressions(self, config: Any) -> Any:
        """Evaluate mathematical expressions in configuration"""
        if isinstance(config, dict):
            return {k: self._evaluate_expressions(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._evaluate_expressions(item) for item in config]
        elif isinstance(config, str):
            # Check if it's a mathematical expression
            if config.startswith('eval:'):
                expr = config[5:].strip()
                try:
                    # Safe evaluation with limited scope
                    safe_dict = {
                        'pi': np.pi,
                        'e': np.e,
                        'sqrt': np.sqrt,
                        'log': np.log,
                        'exp': np.exp,
                        'sin': np.sin,
                        'cos': np.cos
                    }
                    return eval(expr, {"__builtins__": {}}, safe_dict)
                except Exception as e:
                    raise ValueError(f"Failed to evaluate expression '{expr}': {e}")
            return config
        else:
            return config
    
    def _apply_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Apply parameter overrides to configuration"""
        result = config.copy()
        
        for key, value in overrides.items():
            # Support nested keys with dot notation
            keys = key.split('.')
            target = result
            
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            target[keys[-1]] = value
        
        return result
    
    def _validate_config(self, config: Dict):
        """Validate configuration for consistency"""
        # This calls the validate_schema method
        self.validate_schema(config)
    
    def _format_config_for_yaml(self, config: Dict) -> Dict:
        """Format configuration dictionary for YAML output"""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        formatted = convert_types(config)
        
        # Organize into logical sections
        organized = {}
        
        # Standard section order
        section_order = ['network', 'measurements', 'algorithm', 'admm', 'mpi', 
                        'performance', 'output']
        
        for section in section_order:
            if section in formatted:
                organized[section] = formatted[section]
            else:
                # Group related parameters
                section_params = {}
                for key, value in formatted.items():
                    if key.startswith(section + '_'):
                        param_name = key[len(section)+1:]
                        section_params[param_name] = value
                
                if section_params:
                    organized[section] = section_params
        
        # Add any remaining parameters
        for key, value in formatted.items():
            if not any(key.startswith(s + '_') for s in section_order):
                if key not in organized:
                    organized[key] = value
        
        return organized


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary"""
    return {
        'network': {
            'n_sensors': 30,
            'n_anchors': 6,
            'dimension': 2,
            'communication_range': 0.3,
            'scale': 1.0,
            'topology': 'random'
        },
        'measurements': {
            'noise_factor': 0.05,
            'measurement_type': 'distance',
            'outlier_probability': 0.0,
            'carrier_phase': False,
            'seed': 42
        },
        'algorithm': {
            'name': 'mps',
            'gamma': 0.999,
            'alpha': 10.0,
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'verbose': False,
            'early_stopping': True,
            'early_stopping_window': 100,
            'parallel_proximal': False,
            'use_2block': True,
            'adaptive_alpha': False
        },
        'admm': {
            'iterations': 100,
            'tolerance': 1e-6,
            'rho': 1.0,
            'warm_start': True
        },
        'mpi': {
            'enable': False,
            'async_communication': False,
            'buffer_size_kb': 1024,
            'collective_operations': True,
            'checkpoint_interval': 100,
            'load_balancing': 'block'
        },
        'performance': {
            'track_metrics': True,
            'log_interval': 10,
            'save_checkpoints': False
        },
        'output': {
            'save_results': True,
            'output_dir': 'results/',
            'save_interval': 50,
            'save_positions': True,
            'save_metrics': True,
            'plot_results': False
        }
    }