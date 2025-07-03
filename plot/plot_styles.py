"""Plot styles and themes for subwoofer simulation visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import matplotlib.colors as mcolors


class PlotStyles:
    """Centralized plot styling configuration."""
    
    def __init__(self):
        self.init_default_styles()
    
    def init_default_styles(self):
        """Initialize default plot styles."""
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F7931E',
            'error': '#DC3545',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        # Source colors
        self.source_colors = {
            'positive': '#4CAF50',  # Green for positive polarity
            'negative': '#F44336',  # Red for negative polarity
            'neutral': '#2196F3'    # Blue for neutral
        }
        
        # Room boundary styles
        self.room_boundary_color = '#2E3440'
        self.room_boundary_width = 2
        self.room_boundary_style = '-'
        self.room_fill_color = '#ECEFF4'
        self.room_fill_alpha = 0.3
        
        # Target area styles
        self.target_area_color = '#A3BE8C'
        self.target_area_edge_color = '#5E81AC'
        self.target_area_alpha = 0.3
        
        # Avoidance area styles
        self.avoidance_area_color = '#BF616A'
        self.avoidance_area_edge_color = '#D08770'
        self.avoidance_area_alpha = 0.3
        
        # Placement area styles
        self.placement_area_color = '#EBCB8B'
        self.placement_area_edge_color = '#D08770'
        self.placement_area_alpha = 0.2
        
        # SPL colormap configurations
        self.spl_colormaps = {
            'default': 'viridis',
            'hot': 'hot',
            'cool': 'cool',
            'rainbow': 'rainbow',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma',
            'cividis': 'cividis',
            'turbo': 'turbo'
        }
        
        # Font configurations
        self.fonts = {
            'title': {'size': 14, 'weight': 'bold'},
            'axis_label': {'size': 12, 'weight': 'normal'},
            'tick_label': {'size': 10, 'weight': 'normal'},
            'legend': {'size': 10, 'weight': 'normal'},
            'annotation': {'size': 8, 'weight': 'normal'}
        }
        
        # Line styles
        self.line_styles = {
            'solid': '-',
            'dashed': '--',
            'dotted': ':',
            'dashdot': '-.'
        }
        
        # Marker styles
        self.markers = {
            'subwoofer': 'o',
            'measurement': 's',
            'target': '^',
            'avoidance': 'v',
            'placement': 'D'
        }
        
        # Grid styles
        self.grid_styles = {
            'major': {'alpha': 0.5, 'linewidth': 0.5, 'color': '#888888'},
            'minor': {'alpha': 0.3, 'linewidth': 0.3, 'color': '#BBBBBB'}
        }
        
        # Set matplotlib defaults
        self.apply_matplotlib_defaults()
    
    def apply_matplotlib_defaults(self):
        """Apply default matplotlib settings."""
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.transparent': False
        })
    
    def get_source_color(self, polarity: int) -> str:
        """Get color for source based on polarity.
        
        Args:
            polarity: Source polarity (+1 or -1)
            
        Returns:
            Color string
        """
        if polarity > 0:
            return self.source_colors['positive']
        elif polarity < 0:
            return self.source_colors['negative']
        else:
            return self.source_colors['neutral']
    
    def get_spl_colormap(self, name: str = 'default') -> str:
        """Get SPL colormap by name.
        
        Args:
            name: Colormap name
            
        Returns:
            Matplotlib colormap name
        """
        return self.spl_colormaps.get(name, self.spl_colormaps['default'])
    
    def get_font_config(self, element: str) -> Dict[str, Any]:
        """Get font configuration for plot element.
        
        Args:
            element: Plot element name
            
        Returns:
            Font configuration dictionary
        """
        return self.fonts.get(element, self.fonts['axis_label'])
    
    def create_custom_colormap(
        self,
        name: str,
        colors: List[str],
        n_colors: int = 256
    ) -> mcolors.LinearSegmentedColormap:
        """Create custom colormap.
        
        Args:
            name: Colormap name
            colors: List of color strings
            n_colors: Number of colors in colormap
            
        Returns:
            Custom colormap
        """
        cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=n_colors)
        return cmap
    
    def get_array_indicator_style(self, array_type: str) -> Dict[str, Any]:
        """Get style for array type indicator.
        
        Args:
            array_type: Type of array (cardioid, endfire, line_array, vortex)
            
        Returns:
            Style configuration dictionary
        """
        styles = {
            'cardioid': {
                'color': '#FF6B6B',
                'linewidth': 2,
                'linestyle': '-',
                'alpha': 0.7,
                'fill_alpha': 0.2
            },
            'endfire': {
                'color': '#4ECDC4',
                'linewidth': 2,
                'linestyle': '--',
                'alpha': 0.7,
                'fill_alpha': 0.2
            },
            'line_array': {
                'color': '#45B7D1',
                'linewidth': 2,
                'linestyle': '-.',
                'alpha': 0.7,
                'fill_alpha': 0.2
            },
            'vortex': {
                'color': '#F39C12',
                'linewidth': 2,
                'linestyle': ':',
                'alpha': 0.7,
                'fill_alpha': 0.2
            }
        }
        
        return styles.get(array_type, styles['cardioid'])
    
    def get_directivity_arrow_style(self) -> Dict[str, Any]:
        """Get style for directivity arrows.
        
        Returns:
            Arrow style configuration
        """
        return {
            'head_width': 0.1,
            'head_length': 0.1,
            'linewidth': 1,
            'alpha': 0.8,
            'zorder': 11
        }
    
    def get_area_label_style(self, area_type: str) -> Dict[str, Any]:
        """Get style for area labels.
        
        Args:
            area_type: Type of area (target, avoidance, placement)
            
        Returns:
            Label style configuration
        """
        styles = {
            'target': {
                'bbox': {'boxstyle': 'round,pad=0.3', 'facecolor': 'green', 'alpha': 0.7},
                'color': 'white',
                'fontsize': 8,
                'weight': 'bold',
                'ha': 'center',
                'va': 'center'
            },
            'avoidance': {
                'bbox': {'boxstyle': 'round,pad=0.3', 'facecolor': 'red', 'alpha': 0.7},
                'color': 'white',
                'fontsize': 8,
                'weight': 'bold',
                'ha': 'center',
                'va': 'center'
            },
            'placement': {
                'bbox': {'boxstyle': 'round,pad=0.3', 'facecolor': 'yellow', 'alpha': 0.7},
                'color': 'black',
                'fontsize': 8,
                'weight': 'bold',
                'ha': 'center',
                'va': 'center'
            }
        }
        
        return styles.get(area_type, styles['target'])
    
    def get_source_label_style(self) -> Dict[str, Any]:
        """Get style for source labels.
        
        Returns:
            Source label style configuration
        """
        return {
            'bbox': {'boxstyle': 'round,pad=0.2', 'facecolor': 'black', 'alpha': 0.7},
            'color': 'white',
            'fontsize': 8,
            'weight': 'bold',
            'ha': 'center',
            'va': 'bottom'
        }
    
    def get_spl_contour_levels(self, spl_min: float, spl_max: float, num_levels: int = 20) -> np.ndarray:
        """Get SPL contour levels.
        
        Args:
            spl_min: Minimum SPL value
            spl_max: Maximum SPL value
            num_levels: Number of contour levels
            
        Returns:
            Array of contour levels
        """
        return np.linspace(spl_min, spl_max, num_levels)
    
    def get_frequency_response_style(self) -> Dict[str, Any]:
        """Get style for frequency response plots.
        
        Returns:
            Frequency response style configuration
        """
        return {
            'linewidth': 2,
            'alpha': 0.8,
            'marker': 'o',
            'markersize': 4,
            'markerfacecolor': 'white',
            'markeredgecolor': self.colors['primary'],
            'markeredgewidth': 1
        }
    
    def get_optimization_progress_style(self) -> Dict[str, Any]:
        """Get style for optimization progress plots.
        
        Returns:
            Optimization progress style configuration
        """
        return {
            'linewidth': 2,
            'alpha': 0.8,
            'color': self.colors['accent'],
            'marker': 's',
            'markersize': 3,
            'markerfacecolor': self.colors['accent'],
            'markeredgecolor': 'white',
            'markeredgewidth': 1
        }
    
    def apply_dark_theme(self):
        """Apply dark theme to plots."""
        plt.style.use('dark_background')
        
        # Update colors for dark theme
        self.room_boundary_color = '#ECEFF4'
        self.room_fill_color = '#2E3440'
        self.room_fill_alpha = 0.3
        
        # Update matplotlib settings
        plt.rcParams.update({
            'figure.facecolor': '#2E3440',
            'axes.facecolor': '#3B4252',
            'axes.edgecolor': '#D8DEE9',
            'axes.labelcolor': '#D8DEE9',
            'xtick.color': '#D8DEE9',
            'ytick.color': '#D8DEE9',
            'text.color': '#D8DEE9',
            'grid.color': '#4C566A'
        })
    
    def apply_light_theme(self):
        """Apply light theme to plots."""
        plt.style.use('default')
        self.apply_matplotlib_defaults()
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes.
        
        Returns:
            List of theme names
        """
        return ['default', 'dark', 'light']
    
    def get_available_colormaps(self) -> List[str]:
        """Get list of available colormaps.
        
        Returns:
            List of colormap names
        """
        return list(self.spl_colormaps.keys())
    
    def create_spl_colorbar_config(self, frequency: float) -> Dict[str, Any]:
        """Create colorbar configuration for SPL plots.
        
        Args:
            frequency: Frequency for the SPL plot
            
        Returns:
            Colorbar configuration
        """
        return {
            'label': f'SPL at {frequency:.1f} Hz (dB)',
            'shrink': 0.8,
            'aspect': 20,
            'pad': 0.05,
            'extend': 'both'
        }
    
    def get_export_figure_config(self) -> Dict[str, Any]:
        """Get configuration for figure export.
        
        Returns:
            Export configuration dictionary
        """
        return {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'transparent': False,
            'pad_inches': 0.1
        }