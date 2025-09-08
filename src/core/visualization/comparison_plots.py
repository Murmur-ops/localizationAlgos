"""
Algorithm comparison visualization for localization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


class ComparisonPlotter:
    """Generate comparison plots between different algorithms."""
    
    def __init__(self, style: str = 'publication'):
        """Initialize with consistent style."""
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style settings."""
        if self.style == 'publication':
            self.fig_size = (14, 8)
            self.bar_width = 0.35
            self.font_size = 11
            self.title_size = 13
        elif self.style == 'presentation':
            self.fig_size = (16, 10)
            self.bar_width = 0.4
            self.font_size = 14
            self.title_size = 16
        else:
            self.fig_size = (12, 8)
            self.bar_width = 0.35
            self.font_size = 10
            self.title_size = 12
        
        # Color scheme for different algorithms
        self.colors = {
            'mps': '#2E86AB',
            'admm': '#A23B72',
            'belief_propagation': '#F18F01',
            'gradient_descent': '#C73E1D',
            'distributed_mps': '#6A994E',
            'default': '#7F7F7F'
        }
    
    def plot_comparison(self, results: Dict, fig=None) -> plt.Figure:
        """
        Create comparison plots from results.
        
        Args:
            results: Dictionary containing comparison data
            fig: Existing figure to plot on
            
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=self.fig_size)
        
        # Create grid layout
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Performance metrics comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_bars(ax1, results)
        
        # Convergence speed comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_convergence_speed(ax2, results)
        
        # Accuracy vs iterations (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_accuracy_iterations(ax3, results)
        
        # Summary statistics (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_summary_table(ax4, results)
        
        # Overall title
        fig.suptitle('Algorithm Performance Comparison', 
                    fontsize=self.title_size, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_bars(self, ax, results: Dict):
        """Plot bar chart comparing key performance metrics."""
        algorithms = []
        rmse_values = []
        time_values = []
        
        # Extract data for each algorithm
        for alg_name in ['mps', 'admm', 'belief_propagation', 'distributed_mps']:
            if f'{alg_name}_results' in results or alg_name in results:
                alg_data = results.get(f'{alg_name}_results', results.get(alg_name, {}))
                if 'final_rmse' in alg_data or 'rmse' in alg_data:
                    algorithms.append(alg_name.upper().replace('_', ' '))
                    rmse_values.append(alg_data.get('final_rmse', alg_data.get('rmse', 0)))
                    time_values.append(alg_data.get('time', alg_data.get('computation_time', 0)))
        
        if not algorithms:
            # Fallback to simple comparison structure
            if 'mps' in results and 'admm' in results:
                algorithms = ['MPS', 'ADMM']
                rmse_values = [results['mps'].get('final_error', 0),
                              results['admm'].get('final_error', 0)]
                time_values = [results['mps'].get('time', 0),
                              results['admm'].get('time', 0)]
        
        if algorithms:
            x = np.arange(len(algorithms))
            
            # Create bars
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - self.bar_width/2, rmse_values, self.bar_width,
                          label='RMSE', color=self.colors['mps'], alpha=0.8)
            bars2 = ax2.bar(x + self.bar_width/2, time_values, self.bar_width,
                           label='Time (s)', color=self.colors['admm'], alpha=0.8)
            
            # Labels and styling
            ax.set_xlabel('Algorithm', fontsize=self.font_size)
            ax.set_ylabel('RMSE', fontsize=self.font_size, color=self.colors['mps'])
            ax2.set_ylabel('Computation Time (s)', fontsize=self.font_size, color=self.colors['admm'])
            ax.set_title('Performance Metrics', fontsize=self.font_size + 1)
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms, fontsize=self.font_size - 1)
            
            # Add value labels on bars
            for bar, val in zip(bars1, rmse_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=self.font_size - 2)
            
            for bar, val in zip(bars2, time_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=self.font_size - 2)
            
            # Legends
            ax.legend(loc='upper left', fontsize=self.font_size - 1)
            ax2.legend(loc='upper right', fontsize=self.font_size - 1)
    
    def _plot_convergence_speed(self, ax, results: Dict):
        """Plot convergence speed comparison."""
        for alg_name, color_key in [('mps', 'mps'), ('admm', 'admm'), 
                                     ('belief_propagation', 'belief_propagation')]:
            if f'{alg_name}_results' in results or alg_name in results:
                alg_data = results.get(f'{alg_name}_results', results.get(alg_name, {}))
                if 'rmse_history' in alg_data:
                    history = alg_data['rmse_history']
                    iterations = range(len(history))
                    ax.plot(iterations, history,
                           color=self.colors.get(color_key, self.colors['default']),
                           linewidth=2,
                           label=alg_name.upper().replace('_', ' '),
                           marker='o' if alg_name == 'mps' else 's',
                           markevery=max(1, len(history)//10),
                           markersize=5)
        
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('RMSE', fontsize=self.font_size)
        ax.set_title('Convergence Speed', fontsize=self.font_size + 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=self.font_size - 1)
        ax.set_yscale('log')
    
    def _plot_accuracy_iterations(self, ax, results: Dict):
        """Plot accuracy vs computational cost trade-off."""
        algorithms = []
        iterations = []
        final_accuracy = []
        colors_list = []
        
        for alg_name, color_key in [('mps', 'mps'), ('admm', 'admm')]:
            if f'{alg_name}_results' in results or alg_name in results:
                alg_data = results.get(f'{alg_name}_results', results.get(alg_name, {}))
                if 'iterations' in alg_data and ('final_rmse' in alg_data or 'final_error' in alg_data):
                    algorithms.append(alg_name.upper())
                    iterations.append(alg_data['iterations'])
                    final_accuracy.append(alg_data.get('final_rmse', alg_data.get('final_error', 0)))
                    colors_list.append(self.colors.get(color_key, self.colors['default']))
        
        if algorithms:
            # Create scatter plot
            for i, (alg, iter_count, acc, color) in enumerate(zip(algorithms, iterations, final_accuracy, colors_list)):
                ax.scatter(iter_count, acc, s=200, c=color, alpha=0.7, 
                          edgecolors='black', linewidth=2, label=alg)
                # Add text annotation
                ax.annotate(alg, (iter_count, acc), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.font_size - 1)
            
            ax.set_xlabel('Iterations Required', fontsize=self.font_size)
            ax.set_ylabel('Final RMSE', fontsize=self.font_size)
            ax.set_title('Accuracy vs Computational Cost', fontsize=self.font_size + 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=self.font_size - 1)
    
    def _plot_summary_table(self, ax, results: Dict):
        """Create summary statistics table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Algorithm', 'RMSE', 'Iterations', 'Time (s)', 'Converged']
        table_data = []
        
        for alg_name in ['mps', 'admm']:
            if alg_name in results:
                alg_data = results[alg_name]
                row = [
                    alg_name.upper(),
                    f"{alg_data.get('final_error', 0):.4f}",
                    str(alg_data.get('iterations', 0)),
                    f"{alg_data.get('time', 0):.2f}",
                    '✓' if alg_data.get('converged', False) else '✗'
                ]
                table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=headers,
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.2] * 5)
            
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 1)
            table.scale(1.2, 1.5)
            
            # Style header row
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#E0E0E0')
                table[(0, i)].set_text_props(weight='bold')
            
            # Color code convergence column
            for i in range(1, len(table_data) + 1):
                if table_data[i-1][4] == '✓':
                    table[(i, 4)].set_facecolor('#90EE90')
                else:
                    table[(i, 4)].set_facecolor('#FFB6C1')
        
        ax.set_title('Summary Statistics', fontsize=self.font_size + 1, pad=20)
    
    def plot_performance_ratio(self, results: Dict) -> plt.Figure:
        """
        Create a focused comparison showing performance ratios.
        
        Args:
            results: Comparison results
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'performance_ratio' in results:
            ratio = results['performance_ratio']
            speedup = results.get('speedup', 1.0)
            
            # Performance ratio bar
            ax1.barh(['Accuracy\nImprovement'], [ratio], 
                    color=self.colors['mps'], height=0.5)
            ax1.set_xlim(0, max(2, ratio * 1.2))
            ax1.set_xlabel(f'MPS is {ratio:.2f}x more accurate', fontsize=self.font_size)
            ax1.set_title('Relative Performance', fontsize=self.font_size + 1)
            ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            ax1.text(1, 0, '1x\n(Equal)', ha='center', va='bottom', fontsize=self.font_size - 1)
            
            # Speedup bar
            ax2.barh(['Convergence\nSpeed'], [speedup],
                    color=self.colors['distributed_mps'], height=0.5)
            ax2.set_xlim(0, max(2, speedup * 1.2))
            ax2.set_xlabel(f'MPS converges {speedup:.2f}x faster', fontsize=self.font_size)
            ax2.set_title('Convergence Speed', fontsize=self.font_size + 1)
            ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            ax2.text(1, 0, '1x\n(Equal)', ha='center', va='bottom', fontsize=self.font_size - 1)
        
        plt.suptitle('MPS vs ADMM Performance Comparison', 
                    fontsize=self.title_size, fontweight='bold')
        plt.tight_layout()
        
        return fig