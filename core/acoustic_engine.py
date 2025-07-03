"""Acoustic calculation engine for subwoofer simulation."""

import numpy as np
import numba
from typing import Tuple, Optional
import logging
from core.config import FRONT_DIRECTIVITY_BEAMWIDTH_RAD, FRONT_DIRECTIVITY_GAIN_LIN

logger = logging.getLogger(__name__)


@numba.jit(nopython=True, fastmath=True, cache=True)
def calculate_spl_vectorized(
    px: np.ndarray, 
    py: np.ndarray, 
    freq: float, 
    c_val: float, 
    current_sorgenti_array: np.ndarray
) -> np.ndarray:
    """Calculate Sound Pressure Level using vectorized operations.
    
    Args:
        px: X coordinates of calculation points
        py: Y coordinates of calculation points
        freq: Frequency in Hz
        c_val: Speed of sound in m/s
        current_sorgenti_array: Array of subwoofer sources
        
    Returns:
        SPL values in dB at each calculation point
    """
    if freq <= 0 or c_val <= 0:
        return np.full(px.shape, -np.inf)

    total_amplitude = np.zeros_like(px, dtype=np.complex128)
    wavelength = c_val / freq
    if wavelength == 0:
        return np.full(px.shape, -np.inf)
    k = 2 * np.pi / wavelength

    # Iterate over structured array
    for i in range(len(current_sorgenti_array)):
        sub_data = current_sorgenti_array[i]
        sub_x, sub_y = sub_data.x, sub_data.y

        distance = np.sqrt((px - sub_x) ** 2 + (py - sub_y) ** 2)
        distance[distance < 0.01] = 0.01  # Avoid division by zero

        base_amplitude_attenuation = (
            sub_data.pressure_val_at_1m_relative_to_pref * sub_data.gain_lin
        ) / distance

        sub_orientation_angle_nord = sub_data.angle
        v_sub_x = np.sin(sub_orientation_angle_nord)
        v_sub_y = np.cos(sub_orientation_angle_nord)
        v_point_x = px - sub_x
        v_point_y = py - sub_y
        dot_product = v_sub_x * v_point_x + v_sub_y * v_point_y
        mag_point = np.sqrt(v_point_x**2 + v_point_y**2)
        mag_point[mag_point < 1e-9] = 1e-9
        cos_delta_angle = np.clip(dot_product / mag_point, -1.0, 1.0)
        delta_angle = np.arccos(cos_delta_angle)

        directive_gain_lin = np.full(px.shape, 1.0)
        directive_gain_lin[np.abs(delta_angle) < FRONT_DIRECTIVITY_BEAMWIDTH_RAD] = (
            FRONT_DIRECTIVITY_GAIN_LIN
        )

        final_amplitude_component = base_amplitude_attenuation * directive_gain_lin

        phase_distance = -k * distance
        phase_delay = -2 * np.pi * freq * (sub_data.delay_ms / 1000.0)
        phase_polarity = np.pi if sub_data.polarity < 0 else 0.0
        total_phase = phase_distance + phase_delay + phase_polarity

        total_amplitude += final_amplitude_component * np.exp(1j * total_phase)

    magnitude = np.abs(total_amplitude)

    spl = np.full(magnitude.shape, -240.0)
    non_zero_mask = magnitude > 1e-12
    spl[non_zero_mask] = 20 * np.log10(magnitude[non_zero_mask])

    return spl


class AcousticEngine:
    """Main acoustic calculation engine."""
    
    def __init__(self, speed_of_sound: float = 343.0):
        """Initialize acoustic engine.
        
        Args:
            speed_of_sound: Speed of sound in m/s
        """
        self.speed_of_sound = speed_of_sound
        self.logger = logging.getLogger(__name__)
    
    def calculate_spl_field(
        self,
        room_bounds: Tuple[float, float, float, float],
        resolution: int,
        frequency: float,
        sources: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate SPL field over a room area.
        
        Args:
            room_bounds: (x_min, x_max, y_min, y_max)
            resolution: Number of points per dimension
            frequency: Frequency in Hz
            sources: Array of subwoofer sources
            
        Returns:
            Tuple of (X_grid, Y_grid, SPL_grid)
        """
        x_min, x_max, y_min, y_max = room_bounds
        
        x_points = np.linspace(x_min, x_max, resolution)
        y_points = np.linspace(y_min, y_max, resolution)
        X_grid, Y_grid = np.meshgrid(x_points, y_points)
        
        SPL_grid = calculate_spl_vectorized(
            X_grid, Y_grid, frequency, self.speed_of_sound, sources
        )
        
        return X_grid, Y_grid, SPL_grid
    
    def calculate_point_spl(
        self,
        points: np.ndarray,
        frequency: float,
        sources: np.ndarray
    ) -> np.ndarray:
        """Calculate SPL at specific points.
        
        Args:
            points: Array of (x, y) coordinates
            frequency: Frequency in Hz
            sources: Array of subwoofer sources
            
        Returns:
            SPL values at each point
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        px = points[:, 0]
        py = points[:, 1]
        
        spl_values = calculate_spl_vectorized(
            px, py, frequency, self.speed_of_sound, sources
        )
        
        return spl_values
    
    def calculate_frequency_response(
        self,
        point: Tuple[float, float],
        frequencies: np.ndarray,
        sources: np.ndarray
    ) -> np.ndarray:
        """Calculate frequency response at a point.
        
        Args:
            point: (x, y) coordinate
            frequencies: Array of frequencies in Hz
            sources: Array of subwoofer sources
            
        Returns:
            SPL values for each frequency
        """
        px = np.array([point[0]])
        py = np.array([point[1]])
        
        spl_response = np.zeros(len(frequencies))
        
        for i, freq in enumerate(frequencies):
            spl_response[i] = calculate_spl_vectorized(
                px, py, freq, self.speed_of_sound, sources
            )[0]
        
        return spl_response
    
    def calculate_directivity_pattern(
        self,
        source_position: Tuple[float, float],
        source_angle: float,
        radius: float,
        frequency: float,
        angular_resolution: int = 360
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate directivity pattern for a source.
        
        Args:
            source_position: (x, y) position of source
            source_angle: Orientation angle of source
            radius: Distance from source for measurements
            frequency: Frequency in Hz
            angular_resolution: Number of angular points
            
        Returns:
            Tuple of (angles, SPL_values)
        """
        from .config import SUB_DTYPE
        
        # Create a single source
        source = np.array([
            (source_position[0], source_position[1], 1.0, 1.0, source_angle, 0.0, 1)
        ], dtype=SUB_DTYPE)
        
        # Calculate measurement points in circle
        angles = np.linspace(0, 2*np.pi, angular_resolution)
        px = source_position[0] + radius * np.cos(angles)
        py = source_position[1] + radius * np.sin(angles)
        
        # Calculate SPL at each point
        spl_values = calculate_spl_vectorized(
            px, py, frequency, self.speed_of_sound, source
        )
        
        return angles, spl_values
    
    def validate_sources(self, sources: np.ndarray) -> bool:
        """Validate source array format.
        
        Args:
            sources: Array of subwoofer sources
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = ['x', 'y', 'pressure_val_at_1m_relative_to_pref', 
                             'gain_lin', 'angle', 'delay_ms', 'polarity']
            
            for field in required_fields:
                if field not in sources.dtype.names:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Check for reasonable values
            if np.any(sources['gain_lin'] < 0):
                self.logger.warning("Negative gain values detected")
            
            if np.any(np.abs(sources['delay_ms']) > 1000):
                self.logger.warning("Large delay values detected (>1000ms)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating sources: {e}")
            return False
    
    def set_speed_of_sound(self, speed: float) -> None:
        """Set speed of sound.
        
        Args:
            speed: Speed of sound in m/s
        """
        if speed <= 0:
            raise ValueError("Speed of sound must be positive")
        
        self.speed_of_sound = speed
        self.logger.info(f"Speed of sound set to {speed} m/s")
    
    def get_wavelength(self, frequency: float) -> float:
        """Get wavelength for given frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Wavelength in meters
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        return self.speed_of_sound / frequency