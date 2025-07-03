"""Event handlers for canvas interactions and drag operations."""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from PyQt6.QtCore import QObject, pyqtSignal
import logging

logger = logging.getLogger(__name__)


class CanvasEventHandler(QObject):
    """Handles mouse events and interactions on the matplotlib canvas."""
    
    # Signals
    source_selected = pyqtSignal(int)  # source index
    source_moved = pyqtSignal(int, float, float)  # index, x, y
    source_rotated = pyqtSignal(int, float)  # index, angle
    vertex_selected = pyqtSignal(str, int, int)  # area_type, area_idx, vertex_idx
    vertex_moved = pyqtSignal(str, int, int, float, float)  # area_type, area_idx, vertex_idx, x, y
    room_vertex_selected = pyqtSignal(int)  # vertex index
    room_vertex_moved = pyqtSignal(int, float, float)  # vertex_idx, x, y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State tracking
        self.drag_object: Optional[Tuple] = None
        self.original_mouse_pos: Optional[Tuple[float, float]] = None
        self.original_object_pos: Optional[Tuple[float, float]] = None
        self.original_object_angle: Optional[float] = None
        self.original_group_states: List[Dict[str, Any]] = []
        
        # Grid and snapping
        self.grid_snap_enabled = False
        self.grid_snap_spacing = 0.25
        
        # Calibration mode
        self.is_in_calibration_mode = False
        self.calibration_points = []
        
        self.logger = logging.getLogger(__name__)
    
    def set_grid_snap_params(self, enabled: bool, spacing: float):
        """Set grid snapping parameters."""
        self.grid_snap_enabled = enabled
        self.grid_snap_spacing = spacing
    
    def snap_to_grid(self, value: float) -> float:
        """Snap a coordinate value to grid if enabled."""
        if not self.grid_snap_enabled or self.grid_snap_spacing <= 0:
            return value
        return round(value / self.grid_snap_spacing) * self.grid_snap_spacing
    
    def handle_press_event(self, event, project_data: Dict[str, Any]) -> bool:
        """Handle mouse press events on canvas.
        
        Args:
            event: Matplotlib mouse event
            project_data: Current project data
            
        Returns:
            True if event was handled, False otherwise
        """
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return False
        
        # Handle calibration mode
        if self.is_in_calibration_mode:
            return self._handle_calibration_click(event)
        
        # Check for area vertex selection
        if self._handle_area_vertex_selection(event, project_data):
            return True
        
        # Check for room vertex selection
        if self._handle_room_vertex_selection(event, project_data):
            return True
        
        # Check for source selection
        if self._handle_source_selection(event, project_data):
            return True
        
        # No object selected
        self.drag_object = None
        return False
    
    def handle_motion_event(self, event, project_data: Dict[str, Any]) -> bool:
        """Handle mouse motion events during drag operations.
        
        Args:
            event: Matplotlib mouse event
            project_data: Current project data
            
        Returns:
            True if event was handled, False otherwise
        """
        if (self.drag_object is None or event.inaxes is None or 
            event.xdata is None or event.ydata is None or 
            self.original_mouse_pos is None):
            return False
        
        dx = event.xdata - self.original_mouse_pos[0]
        dy = event.ydata - self.original_mouse_pos[1]
        obj_type = self.drag_object[0]
        
        if obj_type == "sub_pos":
            self._handle_source_position_drag(dx, dy, project_data)
        elif obj_type == "group_pos":
            self._handle_group_position_drag(dx, dy, project_data)
        elif obj_type == "sub_rotate":
            self._handle_source_rotation_drag(event, project_data)
        elif obj_type == "group_rotate":
            self._handle_group_rotation_drag(event, project_data)
        elif obj_type.endswith("_vtx"):
            self._handle_vertex_drag(dx, dy, project_data)
        elif obj_type == "stanza_vtx":
            self._handle_room_vertex_drag(dx, dy, project_data)
        
        return True
    
    def handle_release_event(self, event) -> bool:
        """Handle mouse release events.
        
        Args:
            event: Matplotlib mouse event
            
        Returns:
            True if event was handled, False otherwise
        """
        if self.drag_object is not None:
            self.drag_object = None
            self.original_mouse_pos = None
            self.original_object_pos = None
            self.original_object_angle = None
            self.original_group_states = []
            return True
        
        return False
    
    def _handle_calibration_click(self, event) -> bool:
        """Handle clicks during calibration mode."""
        if len(self.calibration_points) < 2:
            self.calibration_points.append((event.xdata, event.ydata))
            self.logger.info(f"Calibration point {len(self.calibration_points)}: ({event.xdata:.2f}, {event.ydata:.2f})")
            return True
        return False
    
    def _handle_area_vertex_selection(self, event, project_data: Dict[str, Any]) -> bool:
        """Handle selection of area vertices."""
        area_lists = [
            (project_data.get('target_areas', []), "target"),
            (project_data.get('avoidance_areas', []), "avoid"),
            (project_data.get('sub_placement_areas', []), "placement"),
        ]
        
        for area_list, type_prefix in area_lists:
            for area_idx, area_data in enumerate(area_list):
                if not area_data.get("active", False):
                    continue
                
                plots = area_data.get("plots", [])
                for vtx_idx, plot_artist in enumerate(plots):
                    if plot_artist and hasattr(plot_artist, 'contains') and plot_artist.contains(event)[0]:
                        self.vertex_selected.emit(type_prefix, area_idx, vtx_idx)
                        self.drag_object = (f"{type_prefix}_vtx", area_idx, vtx_idx)
                        self.original_mouse_pos = (event.xdata, event.ydata)
                        
                        # Get original position
                        punti = area_data.get("punti", [])
                        if vtx_idx < len(punti):
                            self.original_object_pos = tuple(punti[vtx_idx])
                        
                        return True
        
        return False
    
    def _handle_room_vertex_selection(self, event, project_data: Dict[str, Any]) -> bool:
        """Handle selection of room vertices."""
        room_vertices = project_data.get('room_vertices', [])
        
        for vtx_idx, vtx_data in enumerate(room_vertices):
            plot_artist = vtx_data.get("plot") if isinstance(vtx_data, dict) else None
            if plot_artist and hasattr(plot_artist, 'contains') and plot_artist.contains(event)[0]:
                self.room_vertex_selected.emit(vtx_idx)
                self.drag_object = ("stanza_vtx", vtx_idx)
                self.original_mouse_pos = (event.xdata, event.ydata)
                
                # Get original position
                if isinstance(vtx_data, dict):
                    self.original_object_pos = tuple(vtx_data["pos"])
                else:
                    self.original_object_pos = tuple(vtx_data)
                
                return True
        
        return False
    
    def _handle_source_selection(self, event, project_data: Dict[str, Any]) -> bool:
        """Handle selection of sources."""
        sources = project_data.get('sources', [])
        
        # Check in reverse order (last drawn first)
        for i in reversed(range(len(sources))):
            source = sources[i]
            
            # Check arrow artist (for rotation)
            arrow_artist = source.get("arrow_artist")
            if arrow_artist and hasattr(arrow_artist, 'contains') and arrow_artist.contains(event)[0]:
                # Check if angle is locked
                if source.get("param_locks", {}).get("angle", False):
                    self.logger.info(f"Source {i} angle is locked")
                    return True
                
                self.source_selected.emit(i)
                self.original_mouse_pos = (event.xdata, event.ydata)
                self.original_object_angle = source.get("angle", 0.0)
                
                drag_type = "group_rotate" if source.get("group_id") is not None else "sub_rotate"
                self.drag_object = (drag_type, i)
                
                if "group" in drag_type:
                    self.original_group_states = self._get_group_states(
                        source.get("group_id"), sources
                    )
                
                return True
            
            # Check rectangle artist (for position)
            rect_artist = source.get("rect_artist")
            if rect_artist and hasattr(rect_artist, 'contains') and rect_artist.contains(event)[0]:
                # Check if position is locked
                if source.get("param_locks", {}).get("position", False):
                    self.logger.info(f"Source {i} position is locked")
                    return True
                
                self.source_selected.emit(i)
                self.original_mouse_pos = (event.xdata, event.ydata)
                self.original_object_pos = (source.get("x", 0.0), source.get("y", 0.0))
                
                drag_type = "group_pos" if source.get("group_id") is not None else "sub_pos"
                self.drag_object = (drag_type, i)
                
                if "group" in drag_type:
                    self.original_group_states = self._get_group_states(
                        source.get("group_id"), sources
                    )
                
                return True
        
        return False
    
    def _handle_source_position_drag(self, dx: float, dy: float, project_data: Dict[str, Any]):
        """Handle dragging of individual source position."""
        if self.drag_object is None or self.original_object_pos is None:
            return
        
        main_idx = self.drag_object[1]
        sources = project_data.get('sources', [])
        
        if main_idx < len(sources):
            new_x = self.snap_to_grid(self.original_object_pos[0] + dx)
            new_y = self.snap_to_grid(self.original_object_pos[1] + dy)
            self.source_moved.emit(main_idx, new_x, new_y)
    
    def _handle_group_position_drag(self, dx: float, dy: float, project_data: Dict[str, Any]):
        """Handle dragging of source group position."""
        if not self.original_group_states:
            return
        
        for s_state in self.original_group_states:
            orig_x, orig_y = s_state["original_pos"]
            new_x = self.snap_to_grid(orig_x + dx)
            new_y = self.snap_to_grid(orig_y + dy)
            self.source_moved.emit(s_state["sub_idx"], new_x, new_y)
    
    def _handle_source_rotation_drag(self, event, project_data: Dict[str, Any]):
        """Handle dragging of individual source rotation."""
        if self.drag_object is None:
            return
        
        main_idx = self.drag_object[1]
        sources = project_data.get('sources', [])
        
        if main_idx < len(sources):
            source = sources[main_idx]
            sub_x = source.get("x", 0.0)
            sub_y = source.get("y", 0.0)
            
            # Calculate angle from source to mouse
            angle = np.arctan2(event.xdata - sub_x, event.ydata - sub_y)
            self.source_rotated.emit(main_idx, angle)
    
    def _handle_group_rotation_drag(self, event, project_data: Dict[str, Any]):
        """Handle dragging of source group rotation."""
        if not self.original_group_states:
            return
        
        group_center = self.original_group_states[0]["group_center"]
        
        # Calculate angle deltas
        initial_mouse_angle = np.arctan2(
            self.original_mouse_pos[1] - group_center[1],
            self.original_mouse_pos[0] - group_center[0],
        )
        current_mouse_angle = np.arctan2(
            event.ydata - group_center[1], 
            event.xdata - group_center[0]
        )
        angle_delta = current_mouse_angle - initial_mouse_angle
        
        # Apply rotation to each source in group
        for s_state in self.original_group_states:
            sub_idx = s_state["sub_idx"]
            orig_rel_pos = s_state["rel_pos"]
            orig_angle = s_state["original_angle"]
            
            # Rotate relative position
            new_rel_x = (orig_rel_pos[0] * np.cos(angle_delta) - 
                        orig_rel_pos[1] * np.sin(angle_delta))
            new_rel_y = (orig_rel_pos[0] * np.sin(angle_delta) + 
                        orig_rel_pos[1] * np.cos(angle_delta))
            
            # Update position
            new_x = self.snap_to_grid(group_center[0] + new_rel_x)
            new_y = self.snap_to_grid(group_center[1] + new_rel_y)
            self.source_moved.emit(sub_idx, new_x, new_y)
            
            # Update angle
            new_angle = orig_angle + angle_delta
            self.source_rotated.emit(sub_idx, new_angle)
    
    def _handle_vertex_drag(self, dx: float, dy: float, project_data: Dict[str, Any]):
        """Handle dragging of area vertices."""
        if self.drag_object is None or self.original_object_pos is None:
            return
        
        obj_type, area_idx, vtx_idx = self.drag_object
        area_type = obj_type.replace("_vtx", "")
        
        new_x = self.snap_to_grid(self.original_object_pos[0] + dx)
        new_y = self.snap_to_grid(self.original_object_pos[1] + dy)
        
        self.vertex_moved.emit(area_type, area_idx, vtx_idx, new_x, new_y)
    
    def _handle_room_vertex_drag(self, dx: float, dy: float, project_data: Dict[str, Any]):
        """Handle dragging of room vertices."""
        if self.drag_object is None or self.original_object_pos is None:
            return
        
        vtx_idx = self.drag_object[1]
        new_x = self.snap_to_grid(self.original_object_pos[0] + dx)
        new_y = self.snap_to_grid(self.original_object_pos[1] + dy)
        
        self.room_vertex_moved.emit(vtx_idx, new_x, new_y)
    
    def _get_group_states(self, group_id: Optional[int], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get state information for all sources in a group."""
        if group_id is None:
            return []
        
        group_sources = [s for i, s in enumerate(sources) if s.get("group_id") == group_id]
        if not group_sources:
            return []
        
        # Calculate group center
        center_x = np.mean([s.get("x", 0.0) for s in group_sources])
        center_y = np.mean([s.get("y", 0.0) for s in group_sources])
        group_center = (center_x, center_y)
        
        # Create state information for each source
        states = []
        for i, source in enumerate(sources):
            if source.get("group_id") == group_id:
                sub_x = source.get("x", 0.0)
                sub_y = source.get("y", 0.0)
                
                states.append({
                    "sub_idx": i,
                    "original_pos": (sub_x, sub_y),
                    "original_angle": source.get("angle", 0.0),
                    "rel_pos": (sub_x - center_x, sub_y - center_y),
                    "group_center": group_center,
                })
        
        return states
    
    def start_calibration_mode(self):
        """Start calibration mode for background image."""
        self.is_in_calibration_mode = True
        self.calibration_points = []
        self.logger.info("Started calibration mode")
    
    def stop_calibration_mode(self) -> List[Tuple[float, float]]:
        """Stop calibration mode and return calibration points."""
        self.is_in_calibration_mode = False
        points = self.calibration_points.copy()
        self.calibration_points = []
        self.logger.info(f"Stopped calibration mode with {len(points)} points")
        return points