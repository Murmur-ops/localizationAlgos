"""
Unified Visualization Module for Decentralized Localization

This module provides a comprehensive visualization toolkit for all localization results.
"""

from .network_plots import NetworkVisualizer
from .convergence_plots import ConvergencePlotter
from .comparison_plots import ComparisonPlotter
from .heatmaps import HeatmapGenerator

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt


class VisualizationSuite:
    """
    Main interface for generating all visualizations.
    Auto-detects appropriate plots based on data structure.
    """
    
    def __init__(self, style: str = 'publication', output_dir: str = 'figures/'):
        """
        Initialize visualization suite.
        
        Args:
            style: 'publication', 'presentation', or 'default'
            output_dir: Directory to save figures
        """
        self.style = style
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all plotters
        self.network_viz = NetworkVisualizer(style)
        self.convergence_viz = ConvergencePlotter(style)
        self.comparison_viz = ComparisonPlotter(style)
        self.heatmap_viz = HeatmapGenerator(style)
    
    def visualize_results(self, 
                          results_path: Union[str, Path, Dict],
                          plot_types: Optional[List[str]] = None,
                          show: bool = False,
                          save: bool = True) -> Dict[str, Path]:
        """
        Generate visualizations from results file or dictionary.
        
        Args:
            results_path: Path to JSON results, directory with results, or results dict
            plot_types: Specific plots to generate (None = auto-detect)
            show: Display plots interactively
            save: Save plots to files
            
        Returns:
            Dictionary mapping plot type to saved file path
        """
        # Load results
        if isinstance(results_path, dict):
            results = results_path
        elif Path(results_path).is_dir():
            # Find latest results file in directory
            results = self._load_latest_results(results_path)
        else:
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        # Auto-detect plot types if not specified
        if plot_types is None:
            plot_types = self._detect_plot_types(results)
        
        saved_files = {}
        
        # Generate each plot type
        for plot_type in plot_types:
            if plot_type == 'network' and self._has_network_data(results):
                fig = self.network_viz.plot_network(results)
                if save:
                    path = self._save_figure(fig, 'network')
                    saved_files['network'] = path
                    
            elif plot_type == 'convergence' and self._has_convergence_data(results):
                fig = self.convergence_viz.plot_convergence(results)
                if save:
                    path = self._save_figure(fig, 'convergence')
                    saved_files['convergence'] = path
                    
            elif plot_type == 'comparison' and self._has_comparison_data(results):
                fig = self.comparison_viz.plot_comparison(results)
                if save:
                    path = self._save_figure(fig, 'comparison')
                    saved_files['comparison'] = path
                    
            elif plot_type == 'heatmap' and self._has_error_data(results):
                fig = self.heatmap_viz.plot_error_heatmap(results)
                if save:
                    path = self._save_figure(fig, 'heatmap')
                    saved_files['heatmap'] = path
            
            if show and plot_type in saved_files:
                plt.show()
        
        return saved_files
    
    def _detect_plot_types(self, results: Dict) -> List[str]:
        """Auto-detect which plots can be generated from results."""
        plot_types = []
        
        if self._has_network_data(results):
            plot_types.append('network')
        if self._has_convergence_data(results):
            plot_types.append('convergence')
        if self._has_comparison_data(results):
            plot_types.append('comparison')
        if self._has_error_data(results):
            plot_types.append('heatmap')
            
        return plot_types
    
    def _has_network_data(self, results: Dict) -> bool:
        """Check if results contain network topology data."""
        return any(key in results for key in [
            'final_positions', 'estimated_positions', 'true_positions',
            'anchor_positions', 'sensor_positions'
        ])
    
    def _has_convergence_data(self, results: Dict) -> bool:
        """Check if results contain convergence data."""
        return any(key in results for key in [
            'rmse_history', 'objective_history', 'iterations',
            'convergence_history'
        ])
    
    def _has_comparison_data(self, results: Dict) -> bool:
        """Check if results contain algorithm comparison data."""
        return any(key in results for key in [
            'mps_results', 'admm_results', 'algorithms',
            'comparison', 'performance_comparison'
        ])
    
    def _has_error_data(self, results: Dict) -> bool:
        """Check if results contain error distribution data."""
        return any(key in results for key in [
            'error_distribution', 'spatial_errors', 'rmse_map',
            'localization_errors'
        ])
    
    def _load_latest_results(self, directory: Union[str, Path]) -> Dict:
        """Load the most recent results file from a directory."""
        directory = Path(directory)
        json_files = list(directory.glob('*.json'))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")
        
        # Get most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def _save_figure(self, fig, plot_type: str) -> Path:
        """Save figure with appropriate naming."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_type}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved {plot_type} plot to {filepath}")
        
        return filepath


def quick_visualize(results_path: Union[str, Path, Dict], 
                   output_dir: str = 'figures/',
                   show: bool = True) -> Dict[str, Path]:
    """
    Convenience function for quick visualization of results.
    
    Args:
        results_path: Path to results or results dictionary
        output_dir: Where to save figures
        show: Whether to display plots
        
    Returns:
        Dictionary of saved figure paths
    """
    viz = VisualizationSuite(output_dir=output_dir)
    return viz.visualize_results(results_path, show=show)