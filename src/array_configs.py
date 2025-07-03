import numpy as np
import math
from .calculations import normalize_delays
from .constants import DEFAULT_SIM_SPEED_OF_SOUND, DEFAULT_ARRAY_FREQ

def generate_array_configs(array_type, center_x, center_y, num_elements, spacing, c_sound, design_freq, base_params, array_params_ui):
    new_configs = []
    start_angle_deg = array_params_ui.get("start_angle_deg", 0.0)
    
    if array_type == "Coppia Cardioide (2 sub)":
        new_configs = _calculate_cardioid_configs(center_x, center_y, spacing, c_sound, start_angle_deg, base_params)
    
    elif array_type == "Array End-Fire":
        new_configs = _calculate_endfire_configs(center_x, center_y, num_elements, spacing, c_sound, start_angle_deg, base_params)
    
    elif array_type == "Array Lineare (Steering Elettrico)":
        steering_angle_deg = array_params_ui.get("steering_deg", 0.0)
        coverage_angle_deg = array_params_ui.get("coverage_deg", 0.0)
        new_configs = _calculate_line_array_steered_configs(center_x, center_y, num_elements, spacing, start_angle_deg, steering_angle_deg, coverage_angle_deg, c_sound, base_params)

    elif array_type == "Array Vortex":
        vortex_mode = array_params_ui.get("vortex_mode", 1)
        steering_deg = array_params_ui.get("steering_deg", 0.0)
        new_configs = _calculate_vortex_array_configs(center_x, center_y, num_elements, spacing, vortex_mode, design_freq, start_angle_deg, steering_deg, c_sound, base_params)
    
    elif array_type == "Cardioide a 3 Sub (Sterzante)":
        reversed_pos_index = array_params_ui.get("reversed_pos_index", 1)
        new_configs = _calculate_steerable_3sub_cardioid_configs(center_x, center_y, spacing, start_angle_deg, reversed_pos_index, c_sound, base_params)
    
    elif array_type == "Cardiodi Multipli (3+ sub)":
         new_configs = _calculate_multi_cardioid_configs(center_x, center_y, num_elements, spacing, c_sound, base_params)

    elif array_type == "Gradiente di Pressione":
        new_configs = _calculate_pressure_gradient_configs(center_x, center_y, num_elements, spacing, c_sound, start_angle_deg, base_params)
        
    return new_configs

def _calculate_cardioid_configs(
    center_x, center_y, spacing, c, angle_deg, base_params
):
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

def _calculate_endfire_configs(
    center_x, center_y, num_elements, spacing, c, angle_deg, base_params
):
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

def _calculate_line_array_steered_configs(
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

def _calculate_vortex_array_configs(
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

def _calculate_multi_cardioid_configs(center_x, center_y, num_elements, spacing, c_sound, base_params):
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

def _calculate_pressure_gradient_configs(center_x, center_y, num_elements, spacing, c_sound, angle_deg, base_params):
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

def _calculate_steerable_3sub_cardioid_configs(
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
