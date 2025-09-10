"""
Unified Visualization Module for Localization Networks

Provides consistent, publication-quality plots for:
- Network topology with anchors and sensors
- True vs estimated positions
- Convergence curves
- RMSE analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


class NetworkVisualizer:
    """Unified visualization for localization networks"""
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize visualizer with consistent style
        
        Args:
            style: 'publication', 'presentation', or 'default'
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for consistent look"""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            self.fig_size = (12, 8)
            self.dpi = 150
            self.marker_size = 50
            self.anchor_size = 200
            self.line_width = 1.5
            self.font_size = 10
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            self.fig_size = (14, 10)
            self.dpi = 100
            self.marker_size = 100
            self.anchor_size = 300
            self.line_width = 2.0
            self.font_size = 12
        else:
            self.fig_size = (10, 8)
            self.dpi = 100
            self.marker_size = 50
            self.anchor_size = 200
            self.line_width = 1.0
            self.font_size = 10
    
    def plot_network_scenario(self, 
                             true_positions: Dict[int, np.ndarray],
                             estimated_positions: Dict[int, np.ndarray],
                             anchor_positions: np.ndarray,
                             network_scale: float,
                             title: str = "Localization Results",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive network visualization
        
        Args:
            true_positions: Dictionary of true sensor positions
            estimated_positions: Dictionary of estimated positions
            anchor_positions: Array of anchor positions
            network_scale: Scale of network in meters
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main plot: Network topology
        ax1 = fig.add_subplot(gs[:, 0])
        self._plot_topology(ax1, true_positions, estimated_positions, 
                           anchor_positions, network_scale)
        
        # Top right: Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        errors = self._plot_error_distribution(ax2, true_positions, estimated_positions)
        
        # Bottom right: Error heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_error_heatmap(ax3, true_positions, errors, network_scale)
        
        fig.suptitle(title, fontsize=self.font_size + 4, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_network(self, results: Dict, fig=None) -> plt.Figure:
        """
        Create network visualization from results dictionary.
        
        Args:
            results: Dictionary containing position data
            fig: Existing figure to plot on
            
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Extract positions from results
        true_positions = results.get('true_positions', {})
        estimated_positions = results.get('estimated_positions', 
                                        results.get('final_positions', {}))
        anchor_positions = results.get('anchor_positions', {})
        
        # Convert anchor positions to array if it's a dict or list
        if isinstance(anchor_positions, dict):
            anchor_array = np.array(list(anchor_positions.values())) if anchor_positions else np.array([])
        elif isinstance(anchor_positions, list):
            anchor_array = np.array(anchor_positions) if anchor_positions else np.array([])
        else:
            anchor_array = anchor_positions if anchor_positions is not None and len(anchor_positions) > 0 else np.array([])
        
        # Get network scale
        network_scale = results.get('network_scale', 50.0)
        if 'configuration' in results and 'network' in results['configuration']:
            network_scale = results['configuration']['network'].get('scale', 50.0)
        
        # If we have positions data, use full plotting
        if true_positions or estimated_positions:
            return self.plot_network_scenario(
                true_positions=true_positions,
                estimated_positions=estimated_positions,
                anchor_positions=anchor_array,
                network_scale=network_scale,
                title="Network Localization Results"
            )
        else:
            # Simple plot if only final positions available
            ax = fig.add_subplot(111)
            if 'final_positions' in results:
                positions = results['final_positions']
                if isinstance(positions, dict):
                    # Convert position values to numpy arrays
                    pos_list = [np.array(v) if not isinstance(v, np.ndarray) else v 
                               for v in positions.values()]
                    if pos_list and len(pos_list[0]) >= 2:
                        pos_array = np.array(pos_list)
                        ax.scatter(pos_array[:, 0], pos_array[:, 1], 
                                 s=self.marker_size, alpha=0.7, label='Sensors')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title('Final Network Positions')
                ax.grid(True, alpha=0.3)
                ax.legend()
            return fig
    
    def _plot_topology(self, ax, true_pos, est_pos, anchors, scale):
        """Plot network topology with positions"""
        # Plot error lines first (so they're behind markers)
        for idx in true_pos:
            if idx in est_pos:
                ax.plot([true_pos[idx][0], est_pos[idx][0]],
                       [true_pos[idx][1], est_pos[idx][1]],
                       'k-', alpha=0.3, linewidth=self.line_width*0.5,
                       zorder=1)
        
        # Plot true positions
        true_x = [pos[0] for pos in true_pos.values()]
        true_y = [pos[1] for pos in true_pos.values()]
        ax.scatter(true_x, true_y, c='blue', s=self.marker_size,
                  alpha=0.6, label='True Position', zorder=2)
        
        # Plot estimated positions
        if est_pos:
            est_x = [pos[0] for pos in est_pos.values()]
            est_y = [pos[1] for pos in est_pos.values()]
            ax.scatter(est_x, est_y, c='red', marker='x', 
                      s=self.marker_size, label='Estimated', zorder=3)
        
        # Plot anchors if available
        if anchors is not None and len(anchors) > 0 and len(anchors.shape) == 2:
            ax.scatter(anchors[:, 0], anchors[:, 1], c='green', 
                      marker='^', s=self.anchor_size, 
                      label='Anchors', edgecolor='black', 
                  linewidth=1, zorder=4)
        
        # Add anchor labels
        if anchors is not None and len(anchors) > 0 and len(anchors.shape) == 2:
            for i, anchor in enumerate(anchors):
                ax.annotate(f'A{i}', (anchor[0], anchor[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.font_size-2)
        
        ax.set_xlim(-scale*0.1, scale*1.1)
        ax.set_ylim(-scale*0.1, scale*1.1)
        ax.set_xlabel('X Position (m)', fontsize=self.font_size)
        ax.set_ylabel('Y Position (m)', fontsize=self.font_size)
        ax.set_title('Network Topology', fontsize=self.font_size+2)
        ax.legend(loc='upper right', fontsize=self.font_size-1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_error_distribution(self, ax, true_pos, est_pos):
        """Plot error distribution histogram"""
        errors = []
        for idx in true_pos:
            if idx in est_pos:
                true = np.array(true_pos[idx]) if not isinstance(true_pos[idx], np.ndarray) else true_pos[idx]
                est = np.array(est_pos[idx]) if not isinstance(est_pos[idx], np.ndarray) else est_pos[idx]
                error = np.linalg.norm(true - est)
                errors.append(error)
        
        if errors:
            ax.hist(errors, bins=20, alpha=0.7, color='blue', 
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_error = np.mean(errors)
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            
            ax.axvline(mean_error, color='red', linestyle='--',
                      linewidth=self.line_width, label=f'Mean: {mean_error:.3f}m')
            ax.axvline(rmse, color='green', linestyle='-.',
                      linewidth=self.line_width, label=f'RMSE: {rmse:.3f}m')
            
            ax.set_xlabel('Position Error (m)', fontsize=self.font_size)
            ax.set_ylabel('Number of Sensors', fontsize=self.font_size)
            ax.set_title('Error Distribution', fontsize=self.font_size+2)
            ax.legend(fontsize=self.font_size-1)
            ax.grid(True, alpha=0.3)
        
        return errors
    
    def _plot_error_heatmap(self, ax, positions, errors, scale):
        """Create spatial error heatmap"""
        if not errors:
            return
        
        # Create grid for heatmap
        grid_size = 20
        x = np.linspace(0, scale, grid_size)
        y = np.linspace(0, scale, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Interpolate errors onto grid
        pos_list = [np.array(p) if not isinstance(p, np.ndarray) else p 
                    for p in positions.values()]
        pos_array = np.array(pos_list)
        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([X[i, j], Y[i, j]])
                # Find nearest sensor
                distances = np.linalg.norm(pos_array - point, axis=1)
                nearest_idx = np.argmin(distances)
                if nearest_idx < len(errors):
                    Z[i, j] = errors[nearest_idx]
        
        # Create heatmap
        im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Error (m)')
        
        # Overlay sensor positions
        for pos in positions.values():
            ax.plot(pos[0], pos[1], 'ko', markersize=3)
        
        ax.set_xlabel('X Position (m)', fontsize=self.font_size)
        ax.set_ylabel('Y Position (m)', fontsize=self.font_size)
        ax.set_title('Spatial Error Distribution', fontsize=self.font_size+2)
        ax.set_aspect('equal')
    
    def plot_convergence(self, 
                        iterations: List[int],
                        rmse_history: List[float],
                        objective_history: Optional[List[float]] = None,
                        title: str = "Algorithm Convergence",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence curves
        
        Args:
            iterations: List of iteration numbers
            rmse_history: RMSE values over iterations
            objective_history: Optional objective function values
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2 if objective_history else 1, 
                                 figsize=(12 if objective_history else 6, 5),
                                 dpi=self.dpi)
        
        if not objective_history:
            axes = [axes]
        
        # RMSE convergence
        ax = axes[0]
        ax.plot(iterations, rmse_history, 'b-', linewidth=self.line_width,
               marker='o', markersize=3, markevery=max(1, len(iterations)//20))
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('RMSE (m)', fontsize=self.font_size)
        ax.set_title('RMSE Convergence', fontsize=self.font_size+2)
        ax.grid(True, alpha=0.3)
        
        # Add convergence info
        if len(rmse_history) > 1:
            improvement = (rmse_history[0] - rmse_history[-1]) / rmse_history[0] * 100
            ax.text(0.95, 0.95, f'Improvement: {improvement:.1f}%',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=self.font_size-1)
        
        # Objective function convergence
        if objective_history:
            ax = axes[1]
            ax.plot(iterations, objective_history, 'r-', linewidth=self.line_width,
                   marker='s', markersize=3, markevery=max(1, len(iterations)//20))
            ax.set_xlabel('Iteration', fontsize=self.font_size)
            ax.set_ylabel('Objective Value', fontsize=self.font_size)
            ax.set_title('Objective Function', fontsize=self.font_size+2)
            ax.grid(True, alpha=0.3)
            
            # Log scale if values vary greatly
            if max(objective_history) / min(objective_history) > 100:
                ax.set_yscale('log')
        
        fig.suptitle(title, fontsize=self.font_size+4, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_rmse_comparison(self,
                            methods: List[str],
                            rmse_values: List[float],
                            errors: Optional[List[float]] = None,
                            requirement: Optional[float] = None,
                            requirement_label: str = "S-band requirement",
                            title: str = "RMSE Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create RMSE comparison bar chart
        
        Args:
            methods: List of method names
            rmse_values: RMSE values for each method
            errors: Optional error bars
            requirement: Optional requirement line
            requirement_label: Label for requirement
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Create bars
        x = np.arange(len(methods))
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(methods)))
        
        bars = ax.bar(x, rmse_values, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)
        
        # Add error bars if provided
        if errors:
            ax.errorbar(x, rmse_values, yerr=errors, fmt='none',
                       color='black', capsize=5, capthick=1)
        
        # Add requirement line
        if requirement:
            ax.axhline(y=requirement, color='red', linestyle='--',
                      linewidth=self.line_width, label=requirement_label)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, rmse_values)):
            if val < 0.001:
                label = f'{val*1000:.2f}mm'
            elif val < 1:
                label = f'{val*100:.1f}cm'
            else:
                label = f'{val:.2f}m'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   label, ha='center', va='bottom', fontsize=self.font_size-1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('RMSE', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size+4, fontweight='bold')
        
        # Set y-scale based on range
        if max(rmse_values) / min(rmse_values) > 100:
            ax.set_yscale('log')
            ax.set_ylabel('RMSE (log scale)', fontsize=self.font_size)
        
        if requirement:
            ax.legend(fontsize=self.font_size)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig