"""Configuration constants and parameters for subwoofer simulation."""

import numpy as np
from typing import Dict, List, Tuple, Any

# --- Data Types ---
SUB_DTYPE = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('pressure_val_at_1m_relative_to_pref', np.float64),
    ('gain_lin', np.float64),
    ('angle', np.float64),
    ('delay_ms', np.float64),
    ('polarity', np.int32)
])

# --- Parameter Ranges ---
PARAM_RANGES = {
    "delay_ms": (0.0, 300.0),
    "gain_db": (-30.0, 1.0),
    "polarity": [-1, 1],
    "spl_rms": (70.0, 140.0),
    "sub_width_depth": (0.1, 2.0),
    "angle": (0.0, 2 * np.pi),
}

# --- Default Values ---
DEFAULT_SUB_SPL_RMS = 85.0
DEFAULT_SUB_WIDTH = 0.4
DEFAULT_SUB_DEPTH = 0.5
DEFAULT_ARRAY_FREQ = 60.0
DEFAULT_ARRAY_RADIUS = 1.0
DEFAULT_VORTEX_MODE = 1
DEFAULT_LINE_ARRAY_STEERING_DEG = 0.0
DEFAULT_ARRAY_START_ANGLE_DEG = 0.0
DEFAULT_LINE_ARRAY_COVERAGE_DEG = 0.0
DEFAULT_VORTEX_STEERING_DEG = 0.0

# --- Visualization Parameters ---
ARROW_LENGTH = 0.4
ARRAY_INDICATOR_CONE_WIDTH_DEG = 30.0
ARRAY_INDICATOR_RADIUS = 2.5

# --- Room and Area Defaults ---
DEFAULT_ROOM_VERTICES = [[-15, -15], [15, -15], [15, 15], [-15, 15]]
DEFAULT_TARGET_AREA_VERTICES = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
DEFAULT_AVOIDANCE_AREA_VERTICES = [[-4, 0], [-3, 0], [-3, 1], [-4, 1]]
DEFAULT_SUB_PLACEMENT_AREA_VERTICES = [[-10, -10], [10, -10], [10, 10], [-10, 10]]

# --- Optimization Parameters ---
DEFAULT_MAX_SPL_AVOIDANCE = 65.0
DEFAULT_TARGET_MIN_SPL_DESIRED = 80.0
DEFAULT_BALANCE_SLIDER_VALUE = 50
OPTIMIZATION_MUTATION_RATE = 0.1

# --- Acoustic Parameters ---
FRONT_DIRECTIVITY_BEAMWIDTH_RAD = np.pi / 6
FRONT_DIRECTIVITY_GAIN_LIN = 10 ** (1.0 / 20.0)
DEFAULT_SIM_SPEED_OF_SOUND = 343.0
P_REF_20UPA_CALC_DEFAULT = 10 ** (DEFAULT_SUB_SPL_RMS / 20.0)

# --- Grid and Snap Parameters ---
DEFAULT_GRID_SNAP_SPACING = 0.25
DEFAULT_GRID_SNAP_ENABLED = False
DEFAULT_GRID_SHOW_ENABLED = False

# --- Image Background Parameters ---
DEFAULT_BG_IMAGE_PROPS = {
    "path": None,
    "data": None,
    "artist": None,
    "center_x": 0,
    "center_y": 0,
    "scale": 1.0,
    "rotation_deg": 0,
    "alpha": 0.5,
    "anchor_pixel": None,
    "cached_transformed": None,
}

# --- Application Settings ---
DEFAULT_WINDOW_GEOMETRY = (50, 50, 1600, 1000)
DEFAULT_CANVAS_SIZE = (8, 7, 100)  # width, height, dpi


class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    def __init__(self):
        self.sub_dtype = SUB_DTYPE
        self.param_ranges = PARAM_RANGES.copy()
        self.default_values = self._get_default_values()
        self.acoustic_params = self._get_acoustic_params()
        self.visualization_params = self._get_visualization_params()
        self.optimization_params = self._get_optimization_params()
    
    def _get_default_values(self) -> Dict[str, Any]:
        """Get default values for simulation parameters."""
        return {
            "sub_spl_rms": DEFAULT_SUB_SPL_RMS,
            "sub_width": DEFAULT_SUB_WIDTH,
            "sub_depth": DEFAULT_SUB_DEPTH,
            "array_freq": DEFAULT_ARRAY_FREQ,
            "array_radius": DEFAULT_ARRAY_RADIUS,
            "vortex_mode": DEFAULT_VORTEX_MODE,
            "line_array_steering_deg": DEFAULT_LINE_ARRAY_STEERING_DEG,
            "array_start_angle_deg": DEFAULT_ARRAY_START_ANGLE_DEG,
            "line_array_coverage_deg": DEFAULT_LINE_ARRAY_COVERAGE_DEG,
            "vortex_steering_deg": DEFAULT_VORTEX_STEERING_DEG,
            "room_vertices": DEFAULT_ROOM_VERTICES,
            "target_area_vertices": DEFAULT_TARGET_AREA_VERTICES,
            "avoidance_area_vertices": DEFAULT_AVOIDANCE_AREA_VERTICES,
            "sub_placement_area_vertices": DEFAULT_SUB_PLACEMENT_AREA_VERTICES,
            "max_spl_avoidance": DEFAULT_MAX_SPL_AVOIDANCE,
            "target_min_spl_desired": DEFAULT_TARGET_MIN_SPL_DESIRED,
            "balance_slider_value": DEFAULT_BALANCE_SLIDER_VALUE,
        }
    
    def _get_acoustic_params(self) -> Dict[str, Any]:
        """Get acoustic simulation parameters."""
        return {
            "front_directivity_beamwidth_rad": FRONT_DIRECTIVITY_BEAMWIDTH_RAD,
            "front_directivity_gain_lin": FRONT_DIRECTIVITY_GAIN_LIN,
            "speed_of_sound": DEFAULT_SIM_SPEED_OF_SOUND,
            "p_ref_20upa": P_REF_20UPA_CALC_DEFAULT,
        }
    
    def _get_visualization_params(self) -> Dict[str, Any]:
        """Get visualization parameters."""
        return {
            "arrow_length": ARROW_LENGTH,
            "array_indicator_cone_width_deg": ARRAY_INDICATOR_CONE_WIDTH_DEG,
            "array_indicator_radius": ARRAY_INDICATOR_RADIUS,
            "grid_snap_spacing": DEFAULT_GRID_SNAP_SPACING,
            "grid_snap_enabled": DEFAULT_GRID_SNAP_ENABLED,
            "grid_show_enabled": DEFAULT_GRID_SHOW_ENABLED,
            "bg_image_props": DEFAULT_BG_IMAGE_PROPS.copy(),
        }
    
    def _get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters."""
        return {
            "mutation_rate": OPTIMIZATION_MUTATION_RATE,
        }
    
    def update_param_range(self, param_name: str, min_val: float, max_val: float) -> None:
        """Update parameter range.
        
        Args:
            param_name: Name of parameter
            min_val: Minimum value
            max_val: Maximum value
        """
        if param_name in self.param_ranges:
            self.param_ranges[param_name] = (min_val, max_val)
    
    def get_param_range(self, param_name: str) -> Tuple[float, float]:
        """Get parameter range.
        
        Args:
            param_name: Name of parameter
            
        Returns:
            Tuple of (min_value, max_value)
        """
        return self.param_ranges.get(param_name, (0.0, 1.0))
    
    def validate_param_value(self, param_name: str, value: float) -> bool:
        """Validate parameter value against range.
        
        Args:
            param_name: Name of parameter
            value: Value to validate
            
        Returns:
            True if value is within range, False otherwise
        """
        if param_name not in self.param_ranges:
            return True
        
        min_val, max_val = self.param_ranges[param_name]
        return min_val <= value <= max_val