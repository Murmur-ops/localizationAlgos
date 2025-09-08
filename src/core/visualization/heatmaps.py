"""
Heatmap visualization for error distribution and spatial analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import matplotlib.cm as cm


class HeatmapGenerator:
    """Generate heatmap visualizations for localization errors."""
    
    def __init__(self, style: str = 'publication'):
        """Initialize with consistent style."""
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style settings."""
        if self.style == 'publication':
            self.fig_size = (12, 10)
            self.font_size = 11
            self.colorbar_size = '3%'
            self.title_size = 13
        elif self.style == 'presentation':
            self.fig_size = (14, 12)
            self.font_size = 14
            self.colorbar_size = '4%'
            self.title_size = 16
        else:
            self.fig_size = (10, 8)
            self.font_size = 10
            self.colorbar_size = '3%'
            self.title_size = 12
    
    def plot_error_heatmap(self, results: Dict, fig=None) -> plt.Figure:
        """
        Create error distribution heatmap.
        
        Args:
            results: Dictionary containing position and error data
            fig: Existing figure to plot on
            
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=self.fig_size)
        
        # Determine layout based on available data
        has_spatial = self._has_spatial_data(results)
        has_temporal = 'error_over_time' in results
        
        if has_spatial and has_temporal:
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            axes = [fig.add_subplot(gs[0, :]),  # Spatial error (top)
                   fig.add_subplot(gs[1, 0]),    # Error histogram (bottom left)
                   fig.add_subplot(gs[1, 1])]     # Temporal error (bottom right)
        elif has_spatial:
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            axes = [fig.add_subplot(gs[0, :]),  # Spatial error (top)
                   fig.add_subplot(gs[1, :])]     # Error histogram (bottom)
        else:
            axes = [fig.add_subplot(111)]
        
        # Plot spatial error distribution
        if has_spatial:
            self._plot_spatial_errors(axes[0], results)
            if len(axes) > 1:
                self._plot_error_histogram(axes[1], results)
            if len(axes) > 2 and has_temporal:
                self._plot_temporal_errors(axes[2], results)
        
        # Overall title
        fig.suptitle('Localization Error Analysis', 
                    fontsize=self.title_size, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _has_spatial_data(self, results: Dict) -> bool:
        """Check if results contain spatial position data."""
        return any(key in results for key in [
            'true_positions', 'estimated_positions', 'final_positions'
        ])
    
    def _plot_spatial_errors(self, ax, results: Dict):
        """Plot spatial distribution of errors as heatmap."""
        # Extract positions
        true_pos = results.get('true_positions', {})
        est_pos = results.get('estimated_positions', results.get('final_positions', {}))
        
        if not true_pos or not est_pos:
            return
        
        # Calculate errors for each sensor
        errors = []
        positions = []
        
        for sensor_id in true_pos:
            if sensor_id in est_pos:
                true = np.array(true_pos[sensor_id])
                est = np.array(est_pos[sensor_id])
                error = np.linalg.norm(true - est)
                errors.append(error)
                positions.append(true[:2])  # Use true position for spatial reference
        
        if not errors:
            return
        
        positions = np.array(positions)
        errors = np.array(errors)
        
        # Create grid for interpolation
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Add padding
        padding = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Create grid
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        
        # Interpolate errors
        grid_errors = griddata(positions, errors, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # Create heatmap
        im = ax.imshow(grid_errors.T, extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap='RdYlGn_r', aspect='auto')
        
        # Overlay sensor positions
        ax.scatter(positions[:, 0], positions[:, 1], c=errors, 
                  cmap='RdYlGn_r', s=100, edgecolors='black', linewidth=2)
        
        # Add anchors if available
        if 'anchor_positions' in results:
            anchors = np.array(list(results['anchor_positions'].values()))
            ax.scatter(anchors[:, 0], anchors[:, 1], marker='s', s=200,
                      c='blue', edgecolors='black', linewidth=2, label='Anchors')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Localization Error', fontsize=self.font_size)
        
        # Labels and styling
        ax.set_xlabel('X Position', fontsize=self.font_size)
        ax.set_ylabel('Y Position', fontsize=self.font_size)
        ax.set_title('Spatial Error Distribution', fontsize=self.font_size + 1)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean Error: {errors.mean():.3f}\nMax Error: {errors.max():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=self.font_size - 1, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_error_histogram(self, ax, results: Dict):
        """Plot histogram of error distribution."""
        # Extract errors
        true_pos = results.get('true_positions', {})
        est_pos = results.get('estimated_positions', results.get('final_positions', {}))
        
        errors = []
        for sensor_id in true_pos:
            if sensor_id in est_pos:
                true = np.array(true_pos[sensor_id])
                est = np.array(est_pos[sensor_id])
                error = np.linalg.norm(true - est)
                errors.append(error)
        
        if not errors:
            return
        
        errors = np.array(errors)
        
        # Create histogram
        n, bins, patches = ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        
        # Color code bins
        norm = plt.Normalize(vmin=errors.min(), vmax=errors.max())
        colors = cm.RdYlGn_r(norm(bins[:-1]))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        
        # Add statistics lines
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
        ax.axvline(np.median(errors), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}')
        
        # Labels and styling
        ax.set_xlabel('Error Magnitude', fontsize=self.font_size)
        ax.set_ylabel('Number of Sensors', fontsize=self.font_size)
        ax.set_title('Error Distribution', fontsize=self.font_size + 1)
        ax.legend(fontsize=self.font_size - 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_temporal_errors(self, ax, results: Dict):
        """Plot error evolution over time/iterations."""
        if 'error_over_time' in results:
            errors = results['error_over_time']
            iterations = range(len(errors))
            
            ax.plot(iterations, errors, linewidth=2, color='#2E86AB')
            ax.fill_between(iterations, errors, alpha=0.3)
            
            ax.set_xlabel('Iteration', fontsize=self.font_size)
            ax.set_ylabel('Average Error', fontsize=self.font_size)
            ax.set_title('Error Evolution', fontsize=self.font_size + 1)
            ax.grid(True, alpha=0.3)
    
    def plot_connectivity_heatmap(self, results: Dict) -> plt.Figure:
        """
        Create connectivity/communication range heatmap.
        
        Args:
            results: Network connectivity data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'adjacency_matrix' in results or 'connectivity_matrix' in results:
            matrix = results.get('adjacency_matrix', results.get('connectivity_matrix'))
            
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Connection Strength', fontsize=self.font_size)
            
            ax.set_xlabel('Sensor ID', fontsize=self.font_size)
            ax.set_ylabel('Sensor ID', fontsize=self.font_size)
            ax.set_title('Network Connectivity Matrix', fontsize=self.font_size + 1)
            
            # Add grid
            ax.set_xticks(np.arange(0, matrix.shape[0], 5))
            ax.set_yticks(np.arange(0, matrix.shape[1], 5))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_heatmap(self, results: Dict) -> plt.Figure:
        """
        Create precision heatmap for S-band or high-precision results.
        
        Args:
            results: High-precision localization data
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Millimeter-scale precision heatmap
        if 'millimeter_errors' in results or 'precision_errors' in results:
            errors = results.get('millimeter_errors', results.get('precision_errors'))
            positions = results.get('sensor_positions', [])
            
            if len(positions) > 0 and len(errors) > 0:
                positions = np.array(positions)
                errors = np.array(errors) * 1000  # Convert to mm if needed
                
                # Create scatter plot with error as color
                sc = ax1.scatter(positions[:, 0], positions[:, 1], 
                               c=errors, cmap='RdYlGn_r', s=200,
                               vmin=0, vmax=1, edgecolors='black', linewidth=2)
                
                # Add colorbar
                cbar = plt.colorbar(sc, ax=ax1)
                cbar.set_label('Error (mm)', fontsize=self.font_size)
                
                ax1.set_xlabel('X Position (m)', fontsize=self.font_size)
                ax1.set_ylabel('Y Position (m)', fontsize=self.font_size)
                ax1.set_title('Millimeter-Scale Precision Map', fontsize=self.font_size + 1)
                ax1.grid(True, alpha=0.3)
                
                # Add S-band requirement line
                ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                ax2.axvline(15, color='red', linestyle='--', linewidth=2, label='S-band Requirement (15mm)')
                ax2.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                           label=f'Mean: {errors.mean():.2f}mm')
                
                ax2.set_xlabel('Error (mm)', fontsize=self.font_size)
                ax2.set_ylabel('Count', fontsize=self.font_size)
                ax2.set_title('Precision Distribution', fontsize=self.font_size + 1)
                ax2.legend(fontsize=self.font_size - 1)
                ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('High-Precision Localization Analysis', 
                    fontsize=self.title_size, fontweight='bold')
        plt.tight_layout()
        return fig