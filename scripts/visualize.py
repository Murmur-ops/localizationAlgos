#!/usr/bin/env python3
"""
Unified Visualization Script for Decentralized Localization

A single entry point for all visualization needs. Automatically detects
what can be visualized from the input and generates appropriate plots.

Usage:
    # Auto-generate all possible plots from results
    python visualize.py results/my_experiment/
    
    # Generate specific plot types
    python visualize.py --type network convergence results/data.json
    
    # From a YAML config (runs algorithm and visualizes)
    python visualize.py --config configs/example.yaml --run
    
    # Interactive mode
    python visualize.py results/ --show
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.visualization import VisualizationSuite, quick_visualize


def load_data(path: str) -> Dict:
    """
    Load data from various sources.
    
    Args:
        path: Path to JSON file, directory, or YAML config
        
    Returns:
        Dictionary with visualization data
    """
    path = Path(path)
    
    if path.is_dir():
        # Find most recent JSON file in directory
        json_files = list(path.glob('**/*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {path}")
        
        # Sort by modification time
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent results: {latest}")
        
        with open(latest, 'r') as f:
            return json.load(f)
    
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            # Return config as visualization hints
            return {'config': config, 'source': 'yaml'}
    
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def run_from_config(config_path: str) -> Dict:
    """
    Run algorithm from config and return results for visualization.
    
    Args:
        config_path: Path to YAML configuration
        
    Returns:
        Results dictionary
    """
    from scripts.run_mps import main as run_mps
    
    # Run MPS with config
    print(f"Running algorithm from config: {config_path}")
    results = run_mps(config_path, save_results=False, visualize=False)
    
    return results


def detect_visualization_types(data: Dict) -> List[str]:
    """
    Auto-detect which visualizations are possible.
    
    Args:
        data: Results dictionary
        
    Returns:
        List of possible visualization types
    """
    types = []
    
    # Check if data is nested under 'results' key (common format from run_mps.py)
    if 'results' in data and isinstance(data['results'], dict):
        # Flatten the structure for visualization
        results_data = data['results']
        # Merge with top-level data for backwards compatibility
        check_data = {**data, **results_data}
    else:
        check_data = data
    
    # Check for network/position data
    if any(k in check_data for k in ['final_positions', 'estimated_positions', 
                               'true_positions', 'sensor_positions']):
        types.append('network')
    
    # Check for convergence data
    if any(k in check_data for k in ['rmse_history', 'objective_history', 
                               'iterations', 'convergence']):
        types.append('convergence')
    
    # Check for comparison data
    if any(k in check_data for k in ['mps', 'admm', 'algorithms', 
                               'comparison', 'mps_results', 'admm_results']):
        types.append('comparison')
    
    # Check for error distribution data
    if any(k in check_data for k in ['error_distribution', 'spatial_errors',
                               'localization_errors', 'error_map']):
        types.append('heatmap')
    
    # Check for precision/S-band data
    if any(k in check_data for k in ['millimeter_errors', 'precision_errors',
                               'carrier_phase', 'sband_results']):
        types.append('precision')
    
    return types


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Unified visualization for localization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-visualize latest results in directory
  %(prog)s results/
  
  # Generate specific plots
  %(prog)s --type network convergence results/data.json
  
  # Run from config and visualize
  %(prog)s --config configs/example.yaml --run
  
  # Save to specific directory
  %(prog)s results/ --output-dir figures/experiment1/
  
  # Interactive display
  %(prog)s results/latest.json --show
        """
    )
    
    # Input source
    parser.add_argument(
        'input',
        nargs='?',
        default='results/',
        help='Input source: JSON file, directory, or YAML config'
    )
    
    # Config option
    parser.add_argument(
        '--config',
        help='YAML configuration file'
    )
    
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run algorithm from config before visualizing'
    )
    
    # Visualization options
    parser.add_argument(
        '--type',
        nargs='+',
        choices=['network', 'convergence', 'comparison', 'heatmap', 'precision', 'all'],
        help='Specific plot types to generate (default: auto-detect)'
    )
    
    parser.add_argument(
        '--style',
        choices=['publication', 'presentation', 'default'],
        default='publication',
        help='Visualization style preset'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='figures/',
        help='Directory to save figures'
    )
    
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save figures to disk'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load or generate data
        if args.config and args.run:
            data = run_from_config(args.config)
        elif args.config:
            data = load_data(args.config)
        else:
            data = load_data(args.input)
        
        # Handle nested results structure
        if 'results' in data and isinstance(data['results'], dict):
            # Flatten for visualization while keeping config info
            viz_data = {**data['results']}
            if 'configuration' in data:
                viz_data['configuration'] = data['configuration']
            if 'timestamp' in data:
                viz_data['timestamp'] = data['timestamp']
        else:
            viz_data = data
        
        # Determine plot types
        if args.type and 'all' in args.type:
            plot_types = None  # Will auto-detect all
        elif args.type:
            plot_types = args.type
        else:
            plot_types = detect_visualization_types(data)
            if plot_types:
                print(f"Auto-detected plot types: {', '.join(plot_types)}")
            else:
                print("Warning: No visualizable data detected")
                return
        
        # Create visualization suite
        viz = VisualizationSuite(
            style=args.style,
            output_dir=args.output_dir
        )
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        saved_files = viz.visualize_results(
            viz_data,
            plot_types=plot_types,
            show=args.show,
            save=not args.no_save
        )
        
        # Report saved files
        if saved_files and not args.no_save:
            print(f"\nSaved {len(saved_files)} figures:")
            for plot_type, filepath in saved_files.items():
                print(f"  - {plot_type}: {filepath}")
        
        # Show plots if requested
        if args.show:
            plt.show()
            input("\nPress Enter to close...")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()