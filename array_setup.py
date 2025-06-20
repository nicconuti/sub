import math
import numpy as np

from constants import DEFAULT_SUB_SPL_RMS


class ArraySetupMixin:
    """Mixin providing array setup utilities."""

    def _normalize_delays(self, configs_list):
        if not configs_list:
            return
        all_delays = [config['delay_ms'] for config in configs_list]
        min_delay = min(all_delays)
        if min_delay != 0:
            for config in configs_list:
                config['delay_ms'] -= min_delay

    def _add_new_subs_as_group(self, configs_list, array_type, array_params, ref_sub_idx_to_remove=-1):
        if not configs_list:
            return

        if ref_sub_idx_to_remove != -1 and ref_sub_idx_to_remove < len(self.sorgenti):
            self.sorgenti.pop(ref_sub_idx_to_remove)

        self._update_max_group_id()
        new_group_id = self.next_group_id
        self.lista_gruppi_array[new_group_id] = {'type': array_type, **array_params}

        start_index_of_new_subs = len(self.sorgenti)
        for config in configs_list:
            new_sub_data = config.copy()
            new_sub_data['group_id'] = new_group_id
            defaults = {
                'id': self.next_sub_id,
                'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': True},
            }
            for k, v in defaults.items():
                new_sub_data.setdefault(k, v)
            new_sub_data['gain_lin'] = 10 ** (new_sub_data.get('gain_db', 0) / 20.0)
            new_sub_data['pressure_val_at_1m_relative_to_pref'] = 10 ** (
                new_sub_data.get('spl_rms', DEFAULT_SUB_SPL_RMS) / 20.0
            )
            self.sorgenti.append(new_sub_data)
            self.next_sub_id += 1

        master_idx_in_group = next((i for i, conf in enumerate(configs_list) if conf.get('is_group_master')), 0)
        self.current_sub_idx = start_index_of_new_subs + master_idx_in_group

        if array_type == "Lineare" and len(configs_list) > 1:
            first_sub_pos = np.array([configs_list[0]['x'], configs_list[0]['y']])
            last_sub_pos = np.array([configs_list[-1]['x'], configs_list[-1]['y']])
            total_length = np.linalg.norm(last_sub_pos - first_sub_pos)
            self.array_info_label.setText(f"Array Lineare creato. Lunghezza Totale: {total_length:.2f} m")
        else:
            self.array_info_label.setText(f"Gruppo {array_type} creato.")

        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def setup_line_array_steered(
        self,
        center_x,
        center_y,
        num_elements,
        spacing,
        orientation_deg,
        steering_deg,
        coverage_deg,
        c,
        base_params,
        array_params,
        ref_sub_idx,
    ):
        orientation_rad = np.radians(orientation_deg)
        steering_rad = np.radians(steering_deg)
        coverage_rad = np.radians(coverage_deg)
        sub_physical_orientation = orientation_rad
        start_offset = -(num_elements - 1) / 2.0 * spacing
        array_length = (num_elements - 1) * spacing
        new_configs = []

        line_dir_x = np.cos(orientation_rad)
        line_dir_y = -np.sin(orientation_rad)

        for i in range(num_elements):
            offset = start_offset + i * spacing
            sub_x = center_x + offset * line_dir_x
            sub_y = center_y + offset * line_dir_y
            steering_dir_x, steering_dir_y = np.sin(steering_rad), np.cos(steering_rad)
            dot_product = (sub_x - center_x) * steering_dir_x + (sub_y - center_y) * steering_dir_y
            delay_steering_sec = dot_product / c
            delay_coverage_sec = 0.0
            if coverage_rad > np.radians(1) and array_length > 0:
                try:
                    virtual_radius = (array_length / 2.0) / math.sin(coverage_rad / 2.0)
                    if abs(offset) <= virtual_radius:
                        delay_coverage_sec = (virtual_radius - math.sqrt(virtual_radius**2 - offset**2)) / c
                except (ValueError, ZeroDivisionError):
                    pass
            total_delay_ms = (delay_steering_sec + delay_coverage_sec) * 1000.0
            new_conf = base_params.copy()
            new_conf.update(
                {
                    'x': sub_x,
                    'y': sub_y,
                    'angle': sub_physical_orientation,
                    'delay_ms': total_delay_ms,
                    'is_group_master': (i == num_elements // 2),
                    'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': False},
                }
            )
            new_configs.append(new_conf)

        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "Lineare", array_params, ref_sub_idx)

    def setup_vortex_array(
        self,
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
        array_params,
        ref_sub_idx,
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
            projected_distance = pos_vector_x * steering_dir_x + pos_vector_y * steering_dir_y
            delay_steering_ms = (-projected_distance / c) * 1000.0
            total_delay_ms = delay_vortex_ms + delay_steering_ms
            new_conf = base_params.copy()
            new_conf.update(
                {
                    'x': sub_x,
                    'y': sub_y,
                    'angle': sub_orientation,
                    'delay_ms': total_delay_ms,
                    'is_group_master': (n == 0),
                    'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': False},
                }
            )
            new_configs.append(new_conf)

        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "Vortex", array_params, ref_sub_idx)
        self.status_bar.showMessage(f"Array Vortex di {num_elements} elementi creato.", 5000)

    def setup_cardioid_pair(self, center_x, center_y, spacing, c, angle_deg, base_params, ref_sub_idx):
        angle_rad = np.radians(angle_deg)
        dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
        front_sub = base_params.copy()
        front_sub.update(
            {
                'x': center_x + dir_x * spacing / 2,
                'y': center_y + dir_y * spacing / 2,
                'angle': angle_rad,
                'delay_ms': 0,
                'is_group_master': True,
                'polarity': 1,
                'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': True},
            }
        )
        rear_sub = base_params.copy()
        rear_sub.update(
            {
                'x': center_x - dir_x * spacing / 2,
                'y': center_y - dir_y * spacing / 2,
                'angle': angle_rad,
                'delay_ms': (spacing / c) * 1000.0,
                'polarity': -1,
                'is_group_master': False,
                'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': True},
            }
        )
        self._normalize_delays([front_sub, rear_sub])
        self._add_new_subs_as_group([front_sub, rear_sub], "Cardioide", {'steering_deg': angle_deg}, ref_sub_idx)
        self.status_bar.showMessage("Coppia Cardioide creata.", 5000)

    def setup_end_fire_array(self, center_x, center_y, num_elements, spacing, c, angle_deg, base_params, ref_sub_idx):
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
                    'x': sub_x,
                    'y': sub_y,
                    'angle': angle_rad,
                    'delay_ms': delay_ms,
                    'is_group_master': (k == 0),
                    'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': False},
                }
            )
            new_configs.append(new_conf)

        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "End-Fire", {'steering_deg': angle_deg}, ref_sub_idx)
        self.status_bar.showMessage(f"Array End-Fire di {num_elements} elementi creato.", 5000)

