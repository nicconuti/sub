"""Visualization and plotting functionality for subwoofer simulation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
try:
    # Try Qt6 first
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        # Fallback to generic Qt backend (works with both Qt5/Qt6)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        try:
            # Fallback to Qt5 if available
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
        except ImportError:
            raise ImportError("No suitable Qt backend found for matplotlib. Please install PyQt6, PyQt5, or PySide6.")
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from typing import Dict, List, Any, Optional, Tuple
import logging

from plot.plot_styles import PlotStyles

logger = logging.getLogger(__name__)


class MatplotlibCanvas(QWidget):
    """Custom matplotlib canvas widget for PyQt6."""
    
    def __init__(self, parent=None, width=8, height=7, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.plot_styles = PlotStyles()
        
        # Set default styles
        self.axes.set_aspect('equal')
        self.axes.grid(True, alpha=0.3)
        self.axes.set_xlabel('X Position (m)')
        self.axes.set_ylabel('Y Position (m)')
    
    def clear_plot(self):
        """Clear the plot."""
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.grid(True, alpha=0.3)
        self.axes.set_xlabel('X Position (m)')
        self.axes.set_ylabel('Y Position (m)')
    
    def refresh_canvas(self):
        """Refresh the canvas display."""
        self.canvas.draw()


class SubwooferVisualizer:
    """Main visualizer class for subwoofer simulation."""
    
    def __init__(self, canvas: MatplotlibCanvas):
        self.canvas = canvas
        self.axes = canvas.axes
        self.plot_styles = PlotStyles()
        self.logger = logging.getLogger(__name__)
        
        # Visualization state
        self.current_spl_map = None
        self.colorbar = None
        self.auto_scale_spl = True
        self.spl_range = (-10, 100)  # Default SPL range
        
        # Plot elements storage
        self.plot_elements = {
            'sources': [],
            'room_lines': [],
            'target_areas': [],
            'avoidance_areas': [],
            'placement_areas': [],
            'array_indicators': [],
            'background_image': None,
            'grid_lines': []
        }
    
    def plot_spl_map(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        SPL_grid: np.ndarray,
        frequency: float,
        colormap: str = 'viridis',
        show_colorbar: bool = True
    ) -> None:
        """Plot SPL heat map.
        
        Args:
            X_grid: X coordinate grid
            Y_grid: Y coordinate grid
            SPL_grid: SPL values grid
            frequency: Frequency in Hz
            colormap: Matplotlib colormap name
            show_colorbar: Whether to show colorbar
        """
        try:
            # Clear previous SPL map
            if self.current_spl_map is not None:
                self.current_spl_map.remove()
            
            if self.colorbar is not None:
                self.colorbar.remove()
            
            # Filter out invalid SPL values
            valid_mask = SPL_grid > -200
            SPL_filtered = np.where(valid_mask, SPL_grid, np.nan)
            
            # Determine SPL range
            if self.auto_scale_spl:
                valid_spl = SPL_filtered[~np.isnan(SPL_filtered)]
                if len(valid_spl) > 0:
                    vmin = np.percentile(valid_spl, 5)
                    vmax = np.percentile(valid_spl, 95)
                else:
                    vmin, vmax = 0, 100
            else:
                vmin, vmax = self.spl_range
            
            # Create contour plot
            self.current_spl_map = self.axes.contourf(
                X_grid, Y_grid, SPL_filtered,
                levels=50,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.8
            )
            
            # Add colorbar
            if show_colorbar:
                self.colorbar = self.canvas.figure.colorbar(
                    self.current_spl_map,
                    ax=self.axes,
                    label=f'SPL at {frequency:.1f} Hz (dB)'
                )
            
            self.logger.info(f"SPL map plotted for {frequency:.1f} Hz")
            
        except Exception as e:
            self.logger.error(f"Error plotting SPL map: {e}")
    
    def plot_sources(
        self,
        sources: np.ndarray,
        show_labels: bool = True,
        show_directivity: bool = True,
        show_delays: bool = False
    ) -> None:
        """Plot subwoofer sources.
        
        Args:
            sources: Array of source data
            show_labels: Whether to show source labels
            show_directivity: Whether to show directivity arrows
            show_delays: Whether to show delay information
        """
        try:
            # Clear previous sources
            for element in self.plot_elements['sources']:
                element.remove()
            self.plot_elements['sources'].clear()
            
            for i, source in enumerate(sources):
                x, y = source['x'], source['y']
                angle = source['angle']
                polarity = source['polarity']
                delay = source['delay_ms']
                gain_db = 20 * np.log10(source['gain_lin'])
                
                # Choose color based on polarity
                color = self.plot_styles.get_source_color(polarity)
                
                # Plot source marker
                marker = self.axes.scatter(
                    x, y,
                    c=color,
                    s=100,
                    marker='o',
                    edgecolors='black',
                    linewidth=2,
                    alpha=0.8,
                    zorder=10
                )
                self.plot_elements['sources'].append(marker)
                
                # Add source label
                if show_labels:
                    label = f'S{i+1}'
                    if show_delays:
                        label += f'\\n{delay:.1f}ms'
                    
                    text = self.axes.text(
                        x, y + 0.3,
                        label,
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        weight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                    )
                    self.plot_elements['sources'].append(text)
                
                # Add directivity arrow
                if show_directivity:
                    arrow_length = 0.4
                    dx = arrow_length * np.sin(angle)
                    dy = arrow_length * np.cos(angle)
                    
                    arrow = self.axes.arrow(
                        x, y, dx, dy,
                        head_width=0.1,
                        head_length=0.1,
                        fc=color,
                        ec='black',
                        linewidth=1,
                        alpha=0.8,
                        zorder=11
                    )
                    self.plot_elements['sources'].append(arrow)
            
            self.logger.info(f"Plotted {len(sources)} sources")
            
        except Exception as e:
            self.logger.error(f"Error plotting sources: {e}")
    
    def plot_room_boundaries(self, vertices: List[List[float]]) -> None:
        """Plot room boundaries.
        
        Args:
            vertices: List of [x, y] coordinates defining room boundary
        """
        try:
            # Clear previous room lines
            for element in self.plot_elements['room_lines']:
                element.remove()
            self.plot_elements['room_lines'].clear()
            
            if len(vertices) < 3:
                return
            
            # Close the polygon
            vertices_closed = vertices + [vertices[0]]
            
            # Plot room boundary
            x_coords = [v[0] for v in vertices_closed]
            y_coords = [v[1] for v in vertices_closed]
            
            line = self.axes.plot(
                x_coords, y_coords,
                color=self.plot_styles.room_boundary_color,
                linewidth=self.plot_styles.room_boundary_width,
                linestyle=self.plot_styles.room_boundary_style,
                alpha=0.8,
                zorder=5
            )[0]
            self.plot_elements['room_lines'].append(line)
            
            # Add room fill
            fill = self.axes.fill(
                x_coords, y_coords,
                color=self.plot_styles.room_fill_color,
                alpha=self.plot_styles.room_fill_alpha,
                zorder=1
            )[0]
            self.plot_elements['room_lines'].append(fill)
            
            self.logger.info("Room boundaries plotted")
            
        except Exception as e:
            self.logger.error(f"Error plotting room boundaries: {e}")
    
    def plot_target_areas(self, target_areas: List[Dict[str, Any]]) -> None:
        """Plot target areas.
        
        Args:
            target_areas: List of target area definitions
        """
        try:
            # Clear previous target areas
            for element in self.plot_elements['target_areas']:
                element.remove()
            self.plot_elements['target_areas'].clear()
            
            for area in target_areas:
                vertices = area.get('vertices', [])
                if len(vertices) < 3:
                    continue
                
                # Create polygon
                polygon = patches.Polygon(
                    vertices,
                    facecolor=self.plot_styles.target_area_color,
                    edgecolor=self.plot_styles.target_area_edge_color,
                    alpha=self.plot_styles.target_area_alpha,
                    linewidth=2,
                    linestyle='--',
                    zorder=3
                )
                
                patch = self.axes.add_patch(polygon)
                self.plot_elements['target_areas'].append(patch)
                
                # Add area label
                center_x = np.mean([v[0] for v in vertices])
                center_y = np.mean([v[1] for v in vertices])
                
                label = f"Target {area.get('id', '')}"
                min_spl = area.get('min_spl', None)
                if min_spl is not None:
                    label += f"\\n≥{min_spl:.1f} dB"
                
                text = self.axes.text(
                    center_x, center_y,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    weight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7)
                )
                self.plot_elements['target_areas'].append(text)
            
            self.logger.info(f"Plotted {len(target_areas)} target areas")
            
        except Exception as e:
            self.logger.error(f"Error plotting target areas: {e}")
    
    def plot_avoidance_areas(self, avoidance_areas: List[Dict[str, Any]]) -> None:
        """Plot avoidance areas.
        
        Args:
            avoidance_areas: List of avoidance area definitions
        """
        try:
            # Clear previous avoidance areas
            for element in self.plot_elements['avoidance_areas']:
                element.remove()
            self.plot_elements['avoidance_areas'].clear()
            
            for area in avoidance_areas:
                vertices = area.get('vertices', [])
                if len(vertices) < 3:
                    continue
                
                # Create polygon
                polygon = patches.Polygon(
                    vertices,
                    facecolor=self.plot_styles.avoidance_area_color,
                    edgecolor=self.plot_styles.avoidance_area_edge_color,
                    alpha=self.plot_styles.avoidance_area_alpha,
                    linewidth=2,
                    linestyle=':',
                    zorder=3
                )
                
                patch = self.axes.add_patch(polygon)
                self.plot_elements['avoidance_areas'].append(patch)
                
                # Add area label
                center_x = np.mean([v[0] for v in vertices])
                center_y = np.mean([v[1] for v in vertices])
                
                label = f"Avoid {area.get('id', '')}"
                max_spl = area.get('max_spl', None)
                if max_spl is not None:
                    label += f"\\n≤{max_spl:.1f} dB"
                
                text = self.axes.text(
                    center_x, center_y,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    weight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
                )
                self.plot_elements['avoidance_areas'].append(text)
            
            self.logger.info(f"Plotted {len(avoidance_areas)} avoidance areas")
            
        except Exception as e:
            self.logger.error(f"Error plotting avoidance areas: {e}")
    
    def plot_placement_areas(self, placement_areas: List[Dict[str, Any]]) -> None:
        """Plot subwoofer placement areas.
        
        Args:
            placement_areas: List of placement area definitions
        """
        try:
            # Clear previous placement areas
            for element in self.plot_elements['placement_areas']:
                element.remove()
            self.plot_elements['placement_areas'].clear()
            
            for area in placement_areas:
                vertices = area.get('vertices', [])
                if len(vertices) < 3:
                    continue
                
                # Create polygon
                polygon = patches.Polygon(
                    vertices,
                    facecolor=self.plot_styles.placement_area_color,
                    edgecolor=self.plot_styles.placement_area_edge_color,
                    alpha=self.plot_styles.placement_area_alpha,
                    linewidth=1,
                    linestyle='-.',
                    zorder=2
                )
                
                patch = self.axes.add_patch(polygon)
                self.plot_elements['placement_areas'].append(patch)
                
                # Add area label
                center_x = np.mean([v[0] for v in vertices])
                center_y = np.mean([v[1] for v in vertices])
                
                label = f"Placement {area.get('id', '')}"
                
                text = self.axes.text(
                    center_x, center_y,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    weight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
                self.plot_elements['placement_areas'].append(text)
            
            self.logger.info(f"Plotted {len(placement_areas)} placement areas")
            
        except Exception as e:
            self.logger.error(f"Error plotting placement areas: {e}")
    
    def plot_array_indicators(self, array_groups: Dict[str, Any]) -> None:
        """Plot array configuration indicators.
        
        Args:
            array_groups: Dictionary of array group configurations
        """
        try:
            # Clear previous array indicators
            for element in self.plot_elements['array_indicators']:
                element.remove()
            self.plot_elements['array_indicators'].clear()
            
            for group_id, group_data in array_groups.items():
                array_type = group_data.get('tipo', '')
                
                if array_type == 'cardioid':
                    self._plot_cardioid_indicator(group_data)
                elif array_type == 'endfire':
                    self._plot_endfire_indicator(group_data)
                elif array_type == 'line_array':
                    self._plot_line_array_indicator(group_data)
                elif array_type == 'vortex':
                    self._plot_vortex_indicator(group_data)
            
            self.logger.info(f"Plotted {len(array_groups)} array indicators")
            
        except Exception as e:
            self.logger.error(f"Error plotting array indicators: {e}")
    
    def _plot_cardioid_indicator(self, group_data: Dict[str, Any]) -> None:
        """Plot cardioid array indicator."""
        # Implementation placeholder
        pass
    
    def _plot_endfire_indicator(self, group_data: Dict[str, Any]) -> None:
        """Plot endfire array indicator."""
        # Implementation placeholder
        pass
    
    def _plot_line_array_indicator(self, group_data: Dict[str, Any]) -> None:
        """Plot line array indicator."""
        # Implementation placeholder
        pass
    
    def _plot_vortex_indicator(self, group_data: Dict[str, Any]) -> None:
        """Plot vortex array indicator."""
        # Implementation placeholder
        pass
    
    def plot_background_image(
        self,
        image_path: str,
        center_x: float = 0,
        center_y: float = 0,
        scale: float = 1.0,
        rotation_deg: float = 0,
        alpha: float = 0.5
    ) -> None:
        """Plot background image.
        
        Args:
            image_path: Path to image file
            center_x: X coordinate of image center
            center_y: Y coordinate of image center
            scale: Image scale factor
            rotation_deg: Rotation in degrees
            alpha: Image transparency
        """
        try:
            import matplotlib.image as mpimg
            
            # Clear previous background image
            if self.plot_elements['background_image'] is not None:
                self.plot_elements['background_image'].remove()
            
            # Load image
            img = mpimg.imread(image_path)
            
            # Create image artist
            im = self.axes.imshow(
                img,
                extent=[center_x - scale, center_x + scale, center_y - scale, center_y + scale],
                alpha=alpha,
                zorder=0
            )
            
            # Apply rotation if needed
            if rotation_deg != 0:
                transform = mtransforms.Affine2D().rotate_deg(rotation_deg) + self.axes.transData
                im.set_transform(transform)
            
            self.plot_elements['background_image'] = im
            
            self.logger.info(f"Background image plotted: {image_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting background image: {e}")
    
    def plot_grid(self, spacing: float = 1.0, show_major: bool = True, show_minor: bool = False) -> None:
        """Plot coordinate grid.
        
        Args:
            spacing: Grid spacing
            show_major: Whether to show major grid lines
            show_minor: Whether to show minor grid lines
        """
        try:
            # Clear previous grid
            for element in self.plot_elements['grid_lines']:
                element.remove()
            self.plot_elements['grid_lines'].clear()
            
            if show_major:
                self.axes.grid(True, which='major', alpha=0.5, linewidth=0.5)
            
            if show_minor:
                self.axes.grid(True, which='minor', alpha=0.3, linewidth=0.3)
                self.axes.minorticks_on()
            
            self.logger.info("Grid plotted")
            
        except Exception as e:
            self.logger.error(f"Error plotting grid: {e}")
    
    def auto_fit_view(self, margin: float = 0.1) -> None:
        """Auto-fit view to show all elements.
        
        Args:
            margin: Margin around elements as fraction of range
        """
        try:
            # Get bounds of all plot elements
            x_min, x_max = self.axes.get_xlim()
            y_min, y_max = self.axes.get_ylim()
            
            # Add margin
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_margin = x_range * margin
            y_margin = y_range * margin
            
            self.axes.set_xlim(x_min - x_margin, x_max + x_margin)
            self.axes.set_ylim(y_min - y_margin, y_max + y_margin)
            
            self.logger.info("View auto-fitted")
            
        except Exception as e:
            self.logger.error(f"Error auto-fitting view: {e}")
    
    def set_spl_range(self, min_spl: float, max_spl: float) -> None:
        """Set SPL display range.
        
        Args:
            min_spl: Minimum SPL value
            max_spl: Maximum SPL value
        """
        self.spl_range = (min_spl, max_spl)
        self.auto_scale_spl = False
        self.logger.info(f"SPL range set to {min_spl} - {max_spl} dB")
    
    def set_auto_scale_spl(self, auto_scale: bool) -> None:
        """Set automatic SPL scaling.
        
        Args:
            auto_scale: Whether to auto-scale SPL
        """
        self.auto_scale_spl = auto_scale
        self.logger.info(f"Auto-scale SPL: {auto_scale}")
    
    def clear_all(self) -> None:
        """Clear all plot elements."""
        self.canvas.clear_plot()
        
        # Clear stored elements
        for element_list in self.plot_elements.values():
            if isinstance(element_list, list):
                element_list.clear()
            else:
                element_list = None
        
        self.current_spl_map = None
        self.colorbar = None
        
        self.logger.info("All plot elements cleared")
    
    def refresh_display(self) -> None:
        """Refresh the display."""
        self.canvas.refresh_canvas()