"""
Convergence visualization for localization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec


class ConvergencePlotter:
    """Generate convergence plots for optimization algorithms."""
    
    def __init__(self, style: str = 'publication'):
        """Initialize with consistent style."""
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style settings."""
        if self.style == 'publication':
            self.fig_size = (12, 5)
            self.line_width = 2
            self.marker_size = 6
            self.font_size = 11
            self.grid_alpha = 0.3
        elif self.style == 'presentation':
            self.fig_size = (14, 6)
            self.line_width = 3
            self.marker_size = 8
            self.font_size = 14
            self.grid_alpha = 0.4
        else:
            self.fig_size = (10, 5)
            self.line_width = 1.5
            self.marker_size = 5
            self.font_size = 10
            self.grid_alpha = 0.3
    
    def plot_convergence(self, results: Dict, fig=None) -> plt.Figure:
        """
        Create convergence plots from results.
        
        Args:
            results: Dictionary containing convergence data
            fig: Existing figure to plot on (creates new if None)
            
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=self.fig_size)
        
        # Determine layout based on available data
        has_rmse = 'rmse_history' in results or 'rmse' in results
        has_objective = 'objective_history' in results or 'objective' in results
        
        if has_rmse and has_objective:
            gs = GridSpec(1, 2, figure=fig, hspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            axes = [ax1, ax2]
        else:
            ax = fig.add_subplot(111)
            axes = [ax]
        
        # Plot RMSE convergence
        if has_rmse:
            ax = axes[0]
            self._plot_rmse_convergence(ax, results)
        
        # Plot objective convergence
        if has_objective and len(axes) > 1:
            ax = axes[1]
            self._plot_objective_convergence(ax, results)
        elif has_objective:
            ax = axes[0]
            self._plot_objective_convergence(ax, results)
        
        # Add overall title if available
        if 'algorithm_name' in results:
            fig.suptitle(f"{results['algorithm_name']} Convergence", 
                        fontsize=self.font_size + 2, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_rmse_convergence(self, ax, results: Dict):
        """Plot RMSE over iterations."""
        # Extract RMSE data
        if 'rmse_history' in results:
            rmse = results['rmse_history']
        elif 'rmse' in results:
            rmse = results['rmse']
        else:
            return
        
        iterations = list(range(len(rmse)))
        
        # Main RMSE line
        ax.plot(iterations, rmse, 
                color='#2E86AB', 
                linewidth=self.line_width,
                marker='o', 
                markevery=max(1, len(iterations)//20),
                markersize=self.marker_size,
                label='RMSE')
        
        # Add final value annotation
        final_rmse = rmse[-1]
        ax.annotate(f'Final: {final_rmse:.4f}',
                   xy=(iterations[-1], final_rmse),
                   xytext=(iterations[-1] - len(iterations)*0.1, final_rmse + max(rmse)*0.05),
                   fontsize=self.font_size - 1,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        # Styling
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('RMSE', fontsize=self.font_size)
        ax.set_title('Localization Error Convergence', fontsize=self.font_size + 1)
        ax.grid(True, alpha=self.grid_alpha)
        ax.set_yscale('log' if min(rmse) > 0 and max(rmse)/min(rmse) > 100 else 'linear')
        
        # Add convergence rate if possible
        if len(rmse) > 10:
            self._add_convergence_rate(ax, iterations, rmse)
    
    def _plot_objective_convergence(self, ax, results: Dict):
        """Plot objective function over iterations."""
        # Extract objective data
        if 'objective_history' in results:
            objective = results['objective_history']
        elif 'objective' in results:
            objective = results['objective']
        else:
            return
        
        iterations = list(range(len(objective)))
        
        # Main objective line
        ax.plot(iterations, objective,
                color='#A23B72',
                linewidth=self.line_width,
                marker='s',
                markevery=max(1, len(iterations)//20),
                markersize=self.marker_size - 1,
                label='Objective')
        
        # Styling
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('Objective Value', fontsize=self.font_size)
        ax.set_title('Objective Function Convergence', fontsize=self.font_size + 1)
        ax.grid(True, alpha=self.grid_alpha)
        ax.set_yscale('log' if min(objective) > 0 and max(objective)/min(objective) > 100 else 'linear')
    
    def _add_convergence_rate(self, ax, iterations: List, values: List):
        """Add convergence rate annotation."""
        # Estimate convergence rate from last 25% of iterations
        n_points = len(iterations) // 4
        if n_points < 5:
            return
        
        recent_iters = iterations[-n_points:]
        recent_vals = values[-n_points:]
        
        # Fit exponential decay
        if min(recent_vals) > 0:
            log_vals = np.log(recent_vals)
            rate = np.polyfit(recent_iters, log_vals, 1)[0]
            
            # Add text annotation
            rate_text = f"Conv. rate: {abs(rate):.3f}/iter"
            ax.text(0.95, 0.95, rate_text,
                   transform=ax.transAxes,
                   fontsize=self.font_size - 1,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_multi_algorithm_convergence(self, results_list: List[Dict]) -> plt.Figure:
        """
        Plot convergence comparison for multiple algorithms.
        
        Args:
            results_list: List of result dictionaries, each with 'algorithm_name'
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0]*1.2, self.fig_size[1]))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for idx, results in enumerate(results_list):
            name = results.get('algorithm_name', f'Algorithm {idx+1}')
            color = colors[idx % len(colors)]
            
            # RMSE plot
            if 'rmse_history' in results:
                rmse = results['rmse_history']
                ax1.plot(range(len(rmse)), rmse,
                        color=color,
                        linewidth=self.line_width,
                        label=name,
                        marker='o' if idx == 0 else 's' if idx == 1 else '^',
                        markevery=max(1, len(rmse)//15),
                        markersize=self.marker_size)
            
            # Objective plot
            if 'objective_history' in results:
                obj = results['objective_history']
                ax2.plot(range(len(obj)), obj,
                        color=color,
                        linewidth=self.line_width,
                        label=name,
                        marker='o' if idx == 0 else 's' if idx == 1 else '^',
                        markevery=max(1, len(obj)//15),
                        markersize=self.marker_size)
        
        # Styling
        ax1.set_xlabel('Iteration', fontsize=self.font_size)
        ax1.set_ylabel('RMSE', fontsize=self.font_size)
        ax1.set_title('RMSE Convergence Comparison', fontsize=self.font_size + 1)
        ax1.grid(True, alpha=self.grid_alpha)
        ax1.legend(fontsize=self.font_size - 1)
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Iteration', fontsize=self.font_size)
        ax2.set_ylabel('Objective Value', fontsize=self.font_size)
        ax2.set_title('Objective Convergence Comparison', fontsize=self.font_size + 1)
        ax2.grid(True, alpha=self.grid_alpha)
        ax2.legend(fontsize=self.font_size - 1)
        ax2.set_yscale('log')
        
        plt.suptitle('Algorithm Convergence Comparison', 
                    fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        return fig