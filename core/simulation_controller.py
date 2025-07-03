"""Controller for managing simulation state and business logic."""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import logging
from matplotlib.path import Path

from core.acoustic_engine import AcousticEngine
from core.config import SimulationConfig, SUB_DTYPE
from core.optimization import OptimizationWorker

logger = logging.getLogger(__name__)


class SimulationController(QObject):
    """Controls simulation state and coordinates between UI and business logic."""
    
    # Signals
    spl_map_updated = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # X, Y, SPL grids
    simulation_progress = pyqtSignal(str)  # Progress message
    simulation_error = pyqtSignal(str)  # Error message
    optimization_finished = pyqtSignal(dict)  # Results
    project_modified = pyqtSignal()
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__()
        
        self.config = config or SimulationConfig()
        self.acoustic_engine = AcousticEngine(self.config.acoustic_params.get('speed_of_sound', 343.0))
        
        # Simulation state
        self.current_frequency = 60.0
        self.current_spl_map: Optional[np.ndarray] = None
        self.auto_scale_spl = True
        self.spl_range = (60.0, 100.0)
        
        # Grid parameters
        self.grid_resolution = 0.1  # meters
        self.grid_bounds: Optional[Tuple[float, float, float, float]] = None
        
        # Optimization state
        self.optimization_thread: Optional[QThread] = None
        self.optimization_worker: Optional[OptimizationWorker] = None
        
        self.logger = logging.getLogger(__name__)
    
    def set_frequency(self, frequency: float):
        """Set current simulation frequency."""
        if frequency <= 0:
            self.logger.error(f"Invalid frequency: {frequency}")
            return
        
        self.current_frequency = frequency
        self.logger.info(f"Frequency set to {frequency} Hz")
    
    def set_speed_of_sound(self, speed: float):
        """Set speed of sound for calculations."""
        try:
            self.acoustic_engine.set_speed_of_sound(speed)
            self.config.acoustic_params['speed_of_sound'] = speed
        except ValueError as e:
            self.logger.error(f"Invalid speed of sound: {e}")
            self.simulation_error.emit(str(e))
    
    def set_grid_parameters(self, resolution: float, bounds: Optional[Tuple[float, float, float, float]] = None):
        """Set grid parameters for simulation."""
        if resolution <= 0:
            self.logger.error(f"Invalid grid resolution: {resolution}")
            return
        
        self.grid_resolution = resolution
        self.grid_bounds = bounds
        self.logger.info(f"Grid parameters set: resolution={resolution}, bounds={bounds}")
    
    def set_spl_range(self, min_spl: float, max_spl: float, auto_scale: bool = False):
        """Set SPL display range."""
        if min_spl >= max_spl:
            self.logger.error("Min SPL must be less than max SPL")
            return
        
        self.spl_range = (min_spl, max_spl)
        self.auto_scale_spl = auto_scale
        self.logger.info(f"SPL range set to [{min_spl}, {max_spl}], auto_scale={auto_scale}")
    
    def calculate_spl_map(self, sources: np.ndarray, room_vertices: List[List[float]]) -> bool:
        """Calculate SPL map for current configuration.
        
        Args:
            sources: Array of source data
            room_vertices: List of room boundary vertices
            
        Returns:
            True if calculation successful, False otherwise
        """
        try:
            if len(sources) == 0:
                self.simulation_error.emit("No sources defined")
                return False
            
            if len(room_vertices) < 3:
                self.simulation_error.emit("Room must have at least 3 vertices")
                return False
            
            # Validate sources
            if not self.acoustic_engine.validate_sources(sources):
                self.simulation_error.emit("Invalid source configuration")
                return False
            
            # Calculate room bounds
            if self.grid_bounds is None:
                room_points = np.array(room_vertices)
                x_min, x_max = room_points[:, 0].min(), room_points[:, 0].max()
                y_min, y_max = room_points[:, 1].min(), room_points[:, 1].max()
                
                # Add some padding
                padding = max((x_max - x_min), (y_max - y_min)) * 0.1
                bounds = (x_min - padding, x_max + padding, y_min - padding, y_max + padding)
            else:
                bounds = self.grid_bounds
            
            self.simulation_progress.emit("Calculating SPL field...")
            
            # Calculate grid resolution in points
            x_span = bounds[1] - bounds[0]
            y_span = bounds[3] - bounds[2]
            x_points = max(50, int(x_span / self.grid_resolution))
            y_points = max(50, int(y_span / self.grid_resolution))
            
            # Calculate SPL field
            X_grid, Y_grid, SPL_grid = self.acoustic_engine.calculate_spl_field(
                bounds, max(x_points, y_points), self.current_frequency, sources
            )
            
            # Apply room mask
            self.simulation_progress.emit("Applying room boundaries...")
            room_path = Path(room_vertices)
            points_check = np.vstack((X_grid.ravel(), Y_grid.ravel())).T
            room_mask = room_path.contains_points(points_check).reshape(X_grid.shape)
            
            # Mask SPL values outside room
            SPL_grid[~room_mask] = np.nan
            
            # Auto-scale if enabled
            if self.auto_scale_spl:
                valid_spl = SPL_grid[~np.isnan(SPL_grid)]
                if len(valid_spl) > 0:
                    min_spl = np.percentile(valid_spl, 5)
                    max_spl = np.percentile(valid_spl, 95)
                    self.spl_range = (float(min_spl), float(max_spl))
            
            # Store current map
            self.current_spl_map = SPL_grid
            
            # Emit signal with results
            self.spl_map_updated.emit(X_grid, Y_grid, SPL_grid)
            self.simulation_progress.emit("SPL calculation completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error calculating SPL map: {e}")
            self.simulation_error.emit(f"Calculation error: {e}")
            return False
    
    def calculate_point_spl(self, sources: np.ndarray, points: List[Tuple[float, float]]) -> List[float]:
        """Calculate SPL at specific points.
        
        Args:
            sources: Array of source data
            points: List of (x, y) coordinates
            
        Returns:
            List of SPL values at each point
        """
        try:
            if len(sources) == 0 or len(points) == 0:
                return []
            
            points_array = np.array(points)
            spl_values = self.acoustic_engine.calculate_point_spl(
                points_array, self.current_frequency, sources
            )
            
            return spl_values.tolist()
            
        except Exception as e:
            self.logger.error(f"Error calculating point SPL: {e}")
            return []
    
    def update_source_position(self, project_data: Dict[str, Any], source_idx: int, x: float, y: float):
        """Update source position in project data.
        
        Args:
            project_data: Project data dictionary
            source_idx: Index of source to update
            x: New X coordinate
            y: New Y coordinate
        """
        try:
            sources = project_data.get('sources', [])
            if 0 <= source_idx < len(sources):
                if isinstance(sources, np.ndarray):
                    sources[source_idx]['x'] = x
                    sources[source_idx]['y'] = y
                else:
                    sources[source_idx]['x'] = x
                    sources[source_idx]['y'] = y
                
                self.project_modified.emit()
                self.logger.info(f"Updated source {source_idx} position to ({x:.2f}, {y:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error updating source position: {e}")
    
    def update_source_angle(self, project_data: Dict[str, Any], source_idx: int, angle: float):
        """Update source angle in project data.
        
        Args:
            project_data: Project data dictionary
            source_idx: Index of source to update
            angle: New angle in radians
        """
        try:
            sources = project_data.get('sources', [])
            if 0 <= source_idx < len(sources):
                if isinstance(sources, np.ndarray):
                    sources[source_idx]['angle'] = angle
                else:
                    sources[source_idx]['angle'] = angle
                
                self.project_modified.emit()
                self.logger.info(f"Updated source {source_idx} angle to {np.degrees(angle):.1f}°")
            
        except Exception as e:
            self.logger.error(f"Error updating source angle: {e}")
    
    def update_area_vertex(self, project_data: Dict[str, Any], area_type: str, 
                          area_idx: int, vertex_idx: int, x: float, y: float):
        """Update area vertex position.
        
        Args:
            project_data: Project data dictionary
            area_type: Type of area ('target', 'avoid', 'placement')
            area_idx: Index of area
            vertex_idx: Index of vertex within area
            x: New X coordinate
            y: New Y coordinate
        """
        try:
            area_key = f"{area_type}_areas"
            if area_type == "avoid":
                area_key = "avoidance_areas"
            elif area_type == "placement":
                area_key = "sub_placement_areas"
            
            areas = project_data.get(area_key, [])
            if 0 <= area_idx < len(areas):
                area = areas[area_idx]
                vertices = area.get("punti", [])
                if 0 <= vertex_idx < len(vertices):
                    vertices[vertex_idx] = [x, y]
                    self.project_modified.emit()
                    self.logger.info(f"Updated {area_type} area {area_idx} vertex {vertex_idx} to ({x:.2f}, {y:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error updating area vertex: {e}")
    
    def update_room_vertex(self, project_data: Dict[str, Any], vertex_idx: int, x: float, y: float):
        """Update room vertex position.
        
        Args:
            project_data: Project data dictionary
            vertex_idx: Index of vertex
            x: New X coordinate
            y: New Y coordinate
        """
        try:
            room_vertices = project_data.get('room_vertices', [])
            if 0 <= vertex_idx < len(room_vertices):
                if isinstance(room_vertices[vertex_idx], dict):
                    room_vertices[vertex_idx]["pos"] = [x, y]
                else:
                    room_vertices[vertex_idx] = [x, y]
                
                self.project_modified.emit()
                self.logger.info(f"Updated room vertex {vertex_idx} to ({x:.2f}, {y:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error updating room vertex: {e}")
    
    def add_source(self, project_data: Dict[str, Any], x: float, y: float, **kwargs) -> int:
        """Add a new source to the project.
        
        Args:
            project_data: Project data dictionary
            x: X coordinate
            y: Y coordinate
            **kwargs: Additional source parameters
            
        Returns:
            Index of the newly added source
        """
        try:
            sources = project_data.get('sources', [])
            
            # Create new source data
            source_data = {
                'x': x,
                'y': y,
                'pressure_val_at_1m_relative_to_pref': kwargs.get('spl_rms', 85.0),
                'gain_lin': 10**(kwargs.get('gain_db', 0.0) / 20.0),
                'angle': kwargs.get('angle', 0.0),
                'delay_ms': kwargs.get('delay_ms', 0.0),
                'polarity': kwargs.get('polarity', 1),
                'width': kwargs.get('width', self.config.default_values['sub_width']),
                'depth': kwargs.get('depth', self.config.default_values['sub_depth']),
                'param_locks': kwargs.get('param_locks', {}),
                'group_id': kwargs.get('group_id', None),
                'id': kwargs.get('id', len(sources) + 1)
            }
            
            # Add to sources list
            if isinstance(sources, list):
                sources.append(source_data)
                project_data['sources'] = sources
            else:
                # Convert to structured array if needed
                new_source = np.array([
                    (x, y, source_data['pressure_val_at_1m_relative_to_pref'],
                     source_data['gain_lin'], source_data['angle'],
                     source_data['delay_ms'], source_data['polarity'])
                ], dtype=SUB_DTYPE)
                
                if len(sources) == 0:
                    project_data['sources'] = new_source
                else:
                    project_data['sources'] = np.concatenate([sources, new_source])
            
            self.project_modified.emit()
            new_idx = len(project_data['sources']) - 1
            self.logger.info(f"Added source at index {new_idx} at position ({x:.2f}, {y:.2f})")
            
            return new_idx
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return -1
    
    def remove_source(self, project_data: Dict[str, Any], source_idx: int) -> bool:
        """Remove a source from the project.
        
        Args:
            project_data: Project data dictionary
            source_idx: Index of source to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            sources = project_data.get('sources', [])
            if 0 <= source_idx < len(sources):
                if isinstance(sources, list):
                    sources.pop(source_idx)
                else:
                    # For numpy arrays
                    indices = list(range(len(sources)))
                    indices.pop(source_idx)
                    if indices:
                        project_data['sources'] = sources[indices]
                    else:
                        project_data['sources'] = np.array([], dtype=SUB_DTYPE)
                
                self.project_modified.emit()
                self.logger.info(f"Removed source at index {source_idx}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing source: {e}")
            return False
    
    def start_optimization(self, project_data: Dict[str, Any], optimization_params: Dict[str, Any]):
        """Start optimization process.
        
        Args:
            project_data: Project data dictionary
            optimization_params: Optimization parameters
        """
        try:
            if self.optimization_thread is not None:
                self.stop_optimization()
            
            # Create optimization worker
            self.optimization_worker = OptimizationWorker()
            self.optimization_thread = QThread()
            
            self.optimization_worker.moveToThread(self.optimization_thread)
            
            # Connect signals
            self.optimization_thread.started.connect(
                lambda: self.optimization_worker.run_optimization(project_data, optimization_params)
            )
            self.optimization_worker.finished.connect(self.optimization_thread.quit)
            self.optimization_worker.finished.connect(self.optimization_worker.deleteLater)
            self.optimization_worker.result_ready.connect(self.optimization_finished.emit)
            self.optimization_worker.progress_update.connect(self.simulation_progress.emit)
            self.optimization_worker.error_occurred.connect(self.simulation_error.emit)
            
            self.optimization_thread.finished.connect(self.optimization_thread.deleteLater)
            self.optimization_thread.finished.connect(self._on_optimization_finished)
            
            # Start optimization
            self.optimization_thread.start()
            self.simulation_progress.emit("Starting optimization...")
            
        except Exception as e:
            self.logger.error(f"Error starting optimization: {e}")
            self.simulation_error.emit(f"Optimization error: {e}")
    
    def stop_optimization(self):
        """Stop running optimization."""
        try:
            if self.optimization_worker:
                self.optimization_worker.request_stop()
            
            if self.optimization_thread and self.optimization_thread.isRunning():
                self.optimization_thread.quit()
                self.optimization_thread.wait(3000)  # Wait up to 3 seconds
                
                if self.optimization_thread.isRunning():
                    self.optimization_thread.terminate()
                    self.optimization_thread.wait(1000)
            
            self.simulation_progress.emit("Optimization stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping optimization: {e}")
    
    def _on_optimization_finished(self):
        """Handle optimization thread finished."""
        self.optimization_thread = None
        self.optimization_worker = None
        self.simulation_progress.emit("Optimization completed")
    
    def get_current_spl_range(self) -> Tuple[float, float]:
        """Get current SPL display range."""
        return self.spl_range
    
    def get_current_spl_map(self) -> Optional[np.ndarray]:
        """Get current SPL map."""
        return self.current_spl_map
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_optimization()
        self.logger.info("Simulation controller cleaned up")