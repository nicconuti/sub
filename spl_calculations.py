import numpy as np
from constants import FRONT_DIRECTIVITY_BEAMWIDTH_RAD, FRONT_DIRECTIVITY_GAIN_LIN


def calculate_spl_vectorized(px, py, freq, c_val, current_sorgenti_list):
    if freq <= 0 or c_val <= 0:
        return np.full(px.shape, -np.inf)

    total_amplitude = np.zeros_like(px, dtype=np.complex128)
    wavelength = c_val / freq
    if wavelength == 0:
        return np.full(px.shape, -np.inf)
    k = 2 * np.pi / wavelength

    for sub_data in current_sorgenti_list:
        sub_x, sub_y = sub_data['x'], sub_data['y']

        distance = np.sqrt((px - sub_x)**2 + (py - sub_y)**2)
        distance[distance < 0.01] = 0.01

        base_amplitude_attenuation = (
            sub_data.get('pressure_val_at_1m_relative_to_pref', 1.0)
            * sub_data.get('gain_lin', 1.0)
        ) / distance

        sub_orientation_angle_nord = sub_data.get('angle', 0)
        v_sub_x = np.sin(sub_orientation_angle_nord)
        v_sub_y = np.cos(sub_orientation_angle_nord)
        v_point_x = px - sub_x
        v_point_y = py - sub_y
        dot_product = v_sub_x * v_point_x + v_sub_y * v_point_y
        mag_point = np.sqrt(v_point_x**2 + v_point_y**2)
        mag_point[mag_point < 1e-9] = 1e-9
        cos_delta_angle = dot_product / mag_point
        delta_angle = np.arccos(np.clip(cos_delta_angle, -1.0, 1.0))

        directive_gain_lin = np.full(px.shape, 1.0)
        directive_gain_lin[np.abs(delta_angle) < FRONT_DIRECTIVITY_BEAMWIDTH_RAD] = FRONT_DIRECTIVITY_GAIN_LIN

        final_amplitude_component = base_amplitude_attenuation * directive_gain_lin

        phase_distance = -k * distance
        phase_delay = -2 * np.pi * freq * (sub_data.get('delay_ms', 0.0) / 1000.0)
        phase_polarity = np.pi if sub_data.get('polarity', 1) < 0 else 0.0
        total_phase = phase_distance + phase_delay + phase_polarity

        total_amplitude += final_amplitude_component * np.exp(1j * total_phase)

    magnitude = np.abs(total_amplitude)

    spl = np.full(magnitude.shape, -240.0)
    non_zero_mask = magnitude > 1e-12
    spl[non_zero_mask] = 20 * np.log10(magnitude[non_zero_mask])

    return spl
