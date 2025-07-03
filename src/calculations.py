import numpy as np
import numba
import math
from matplotlib.path import Path

from .constants import sub_dtype, PARAM_RANGES, DEFAULT_SUB_SPL_RMS, DEFAULT_SUB_WIDTH, DEFAULT_SUB_DEPTH, DEFAULT_ARRAY_FREQ, DEFAULT_SIM_SPEED_OF_SOUND, FRONT_DIRECTIVITY_BEAMWIDTH_RAD, FRONT_DIRECTIVITY_GAIN_LIN

@numba.jit(nopython=True, fastmath=True, cache=True)
def calculate_spl_vectorized(px, py, freq, c_val, current_sorgenti_array):
    if freq <= 0 or c_val <= 0:
        return np.full(px.shape, -np.inf)

    total_amplitude = np.zeros_like(px, dtype=np.complex128)
    wavelength = c_val / freq
    if wavelength == 0:
        return np.full(px.shape, -np.inf)
    k = 2 * np.pi / wavelength

    for i in range(len(current_sorgenti_array)):
        sub_data = current_sorgenti_array[i]
        sub_x, sub_y = sub_data.x, sub_data.y

        distance = np.sqrt((px - sub_x) ** 2 + (py - sub_y) ** 2)
        distance[distance < 0.01] = 0.01

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

def normalize_delays(configs_list):
    if not configs_list:
        return
    all_delays = [config["delay_ms"] for config in configs_list]
    min_delay = min(all_delays)
    if min_delay != 0:
        for config in configs_list:
            config["delay_ms"] -= min_delay

def calculate_cardioid_configs(center_x, center_y, spacing, c, angle_deg, base_params):
    angle_rad = np.radians(angle_deg)
    dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
    front_sub = base_params.copy()
    front_sub.update(
        {
            "x": center_x + dir_x * spacing / 2,
            "y": center_y + dir_y * spacing / 2,
            "angle": angle_rad,
            "delay_ms": 0,
            "gain_db": 0.0,
            "is_group_master": True,
            "polarity": 1,
            "param_locks": {
                "angle": True,
                "delay": True,
                "gain": False,
                "polarity": True,
                "position": False,
            },
        }
    )
    rear_sub = base_params.copy()
    rear_sub.update(
        {
            "x": center_x - dir_x * spacing / 2,
            "y": center_y - dir_y * spacing / 2,
            "angle": angle_rad,
            "delay_ms": (spacing / c) * 1000.0,
            "gain_db": 0.0,
            "polarity": -1,
            "is_group_master": False,
            "param_locks": {
                "angle": True,
                "delay": True,
                "gain": False,
                "polarity": True,
                "position": False,
            },
        }
    )
    configs = [front_sub, rear_sub]
    normalize_delays(configs)
    return configs

def calculate_endfire_configs(center_x, center_y, num_elements, spacing, c, angle_deg, base_params):
    angle_rad = np.radians(angle_deg)
    dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
    new_configs = []
    start_offset = -(num_elements - 1) / 2.0 * spacing
    for k in range(num_elements):
        offset = start_offset + k * spacing
        sub_x = center_x + offset * dir_x
        sub_y = center_y + offset * dir_y
        delay_ms = (k * spacing / c) * 1000.0
        new_conf = base_params.copy()
        new_conf.update(
            {
                "x": sub_x,
                "y": sub_y,
                "angle": angle_rad,
                "delay_ms": delay_ms,
                "gain_db": 0.0,
                "is_group_master": (k == 0),
                "param_locks": {
                    "angle": True,
                    "delay": True,
                    "gain": False,
                    "polarity": False,
                    "position": False,
                },
            }
        )
        new_configs.append(new_conf)
    normalize_delays(new_configs)
    return new_configs

def calculate_line_array_steered_configs(
    center_x,
    center_y,
    num_elements,
    spacing,
    orientation_deg,
    steering_deg,
    coverage_deg,
    c,
    base_params,
):
    orientation_rad = np.radians(orientation_deg)
    steering_rad = np.radians(steering_deg)
    coverage_rad = np.radians(coverage_deg)
    sub_physical_orientation = orientation_rad
    start_offset = -(num_elements - 1) / 2.0 * spacing
    array_length = (num_elements - 1) * spacing
    new_configs = []
    line_dir_x, line_dir_y = np.cos(orientation_rad), -np.sin(orientation_rad)

    for i in range(num_elements):
        offset = start_offset + i * spacing
        sub_x = center_x + offset * line_dir_x
        sub_y = center_y + offset * line_dir_y

        steering_dir_x, steering_dir_y = np.sin(steering_rad), np.cos(steering_rad)
        dot_product = (sub_x - center_x) * steering_dir_x + (
            sub_y - center_y
        ) * steering_dir_y
        delay_steering_sec = dot_product / c

        delay_coverage_sec = 0.0
        if coverage_rad > np.radians(1) and array_length > 0:
            try:
                virtual_radius = (array_length / 2.0) / math.sin(coverage_rad / 2.0)
                if abs(offset) <= virtual_radius:
                    delay_coverage_sec = (
                        virtual_radius - math.sqrt(virtual_radius**2 - offset**2)
                    ) / c
            except (ValueError, ZeroDivisionError):
                pass

        total_delay_ms = (delay_steering_sec + delay_coverage_sec) * 1000.0
        new_conf = base_params.copy()
        new_conf.update(
            {
                "x": sub_x,
                "y": sub_y,
                "angle": sub_physical_orientation,
                "delay_ms": total_delay_ms,
                "gain_db": 0.0,
                "is_group_master": (i == num_elements // 2),
                "param_locks": {
                    "angle": True,
                    "delay": True,
                    "gain": False,
                    "polarity": False,
                    "position": False,
                },
            }
        )
        new_configs.append(new_conf)
    normalize_delays(new_configs)
    return new_configs

def calculate_vortex_array_configs(
    center_x,
    center_y,
    num_elements,
    radius,
    mode,
    freq,
    start_angle_deg,
    steering_deg,
    c,
    base_params,
):
    angle_step = 2 * np.pi / num_elements
    start_angle_rad = np.radians(start_angle_deg)
    steering_rad = np.radians(steering_deg)
    new_configs = []
    for n in range(num_elements):
        phi = start_angle_rad + n * angle_step
        sub_x = center_x + radius * np.sin(phi)
        sub_y = center_y + radius * np.cos(phi)
        sub_orientation = phi + np.pi / 2

        delay_vortex_ms = ((mode * n) / (num_elements * freq)) * 1000.0

        pos_vector_x = sub_x - center_x
        pos_vector_y = sub_y - center_y
        steering_dir_x, steering_dir_y = np.sin(steering_rad), np.cos(steering_rad)
        projected_distance = (
            pos_vector_x * steering_dir_x + pos_vector_y * steering_dir_y
        )
        delay_steering_ms = (-projected_distance / c) * 1000.0
        total_delay_ms = delay_vortex_ms + delay_steering_ms

        new_conf = base_params.copy()
        new_conf.update(
            {
                "x": sub_x,
                "y": sub_y,
                "angle": sub_orientation,
                "delay_ms": total_delay_ms,
                "gain_db": 0.0,
                "is_group_master": (n == 0),
                "param_locks": {
                    "angle": True,
                    "delay": True,
                    "gain": False,
                    "polarity": False,
                    "position": False,
                },
            }
        )
        new_configs.append(new_conf)
    normalize_delays(new_configs)
    return new_configs

def calculate_multi_cardioid_configs(center_x, center_y, num_elements, spacing, c_sound, base_params):
    if num_elements < 3:
        return []

    configs = []
    orientation_rad = 0.0
    dir_x, dir_y = np.cos(orientation_rad), -np.sin(orientation_rad)
    
    start_offset = -(num_elements - 1) / 2.0 * spacing

    for i in range(num_elements):
        offset = start_offset + i * spacing
        sub_x = center_x + offset * dir_x
        sub_y = center_y + offset * dir_y
        
        new_conf = base_params.copy()

        if (i + 1) % 3 == 0:
            is_reversed = True
            polarity = -1
            angle = orientation_rad + np.pi
            gain_db = +6.0
            delay_ms = (spacing / c_sound) * 1000.0
        else:
            is_reversed = False
            polarity = 1
            angle = orientation_rad
            gain_db = 0.0
            delay_ms = 0.0

        new_conf.update({
            "x": sub_x,
            "y": sub_y,
            "angle": angle,
            "delay_ms": delay_ms,
            "polarity": polarity,
            "gain_db": gain_db,
            "is_group_master": (i == 0),
            "param_locks": {
                'angle': True, 'delay': True, 'gain': True,
                'polarity': True, 'position': False
            },
        })
        configs.append(new_conf)
        
    normalize_delays(configs)
    return configs

def calculate_pressure_gradient_configs(center_x, center_y, num_elements, spacing, c_sound, angle_deg, base_params):
    if num_elements < 2:
        return []

    configs = []
    angle_rad = np.radians(angle_deg)
    dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
    
    start_offset = -(num_elements - 1) / 2.0 * spacing
    GAIN_SHADING_PER_STEP_DB = -3.0

    for i in range(num_elements):
        offset = start_offset + i * spacing
        sub_x = center_x + offset * dir_x
        sub_y = center_y + offset * dir_y
        
        new_conf = base_params.copy()
        new_conf.update({
            "x": sub_x,
            "y": sub_y,
            "angle": angle_rad,
            "delay_ms": (i * spacing / c_sound) * 1000.0,
            "gain_db": i * GAIN_SHADING_PER_STEP_DB,
            "polarity": 1,
            "is_group_master": (i == 0),
        })
        configs.append(new_conf)
        
    normalize_delays(configs)
    return configs

def calculate_steerable_3sub_cardioid_configs(
    center_x, center_y, spacing, orientation_deg, reversed_sub_position_index, c, base_params
):
    orientation_rad = np.radians(orientation_deg)
    line_dir_x = np.cos(orientation_rad)
    line_dir_y = -np.sin(orientation_rad)

    configs = []
    positions = []
    
    for i in range(3):
        offset = (i - 1) * spacing
        sub_x = center_x + offset * line_dir_x
        sub_y = center_y + offset * line_dir_y
        positions.append((sub_x, sub_y))

    for i in range(3):
        is_reversed = (i == reversed_sub_position_index)
        sub_pos = positions[i]
        
        new_conf = base_params.copy()

        if is_reversed:
            polarity = -1
            angle = orientation_rad + np.pi
            gain_db = +6.0
            
            if reversed_sub_position_index == 1:
                distance_to_ref = spacing
            else:
                distance_to_ref = 2 * spacing
            
            delay_ms = (distance_to_ref / c) * 1000.0
        else:
            polarity = 1
            angle = orientation_rad
            gain_db = 0.0
            
            if reversed_sub_position_index == 0 and i == 2:
                delay_ms = (spacing / c) * 1000.0
            elif reversed_sub_position_index == 2 and i == 0:
                delay_ms = (spacing / c) * 1000.0
            else:
                delay_ms = 0.0

        new_conf.update(
            {
                "x": sub_pos[0],
                "y": sub_pos[1],
                "angle": angle,
                "delay_ms": delay_ms,
                "polarity": polarity,
                "gain_db": gain_db,
                "is_group_master": (i == 1),
                "param_locks": {
                    'angle': True, 'delay': True, 'gain': True,
                    'polarity': True, 'position': False
                },
            }
        )
        configs.append(new_conf)

    normalize_delays(configs)
    return configs

def find_nearest_point_in_placement_area(x, y, placement_area):
    if not placement_area:
        return x, y
        
    area_path = Path(placement_area)
    if area_path.contains_point((x, y)):
        return x, y
        
    min_dist = float('inf')
    nearest_x, nearest_y = x, y
    
    for i in range(len(placement_area)):
        p1 = placement_area[i]
        p2 = placement_area[(i + 1) % len(placement_area)]
        
        seg_x, seg_y = project_point_on_segment(x, y, p1, p2)
        dist = np.sqrt((x - seg_x)**2 + (y - seg_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = seg_x, seg_y
            
    return nearest_x, nearest_y

def project_point_on_segment(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return x1, y1
        
    param = dot / len_sq
    
    if param < 0:
        return x1, y1
    elif param > 1:
        return x2, y2
    else:
        return x1 + param * C, y1 + param * D

def calculate_auto_spl_range(spl_map):
    if spl_map is None:
        return 70, 100
        
    valid_values = spl_map[~np.isnan(spl_map) & (spl_map > -200)]
    if valid_values.size == 0:
        return 70, 100
        
    min_spl = np.min(valid_values)
    max_spl = np.max(valid_values)
    
    center = (max_spl + min_spl) / 2.0
    return float(center - 15), float(center + 15)
