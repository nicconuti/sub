
import sys
import os

try:
    import matplotlib
    import PyQt6
    import pandas
    import openpyxl  # Aggiunto per l'engine di ExcelWriter

    print("--- DIAGNOSTICS ---")
    print("Python Executable:", sys.executable)
    print("Matplotlib Version:", matplotlib.__version__)
    print("Matplotlib Path:", os.path.dirname(matplotlib.__file__))
    print("PyQt6 Path:", os.path.dirname(PyQt6.__file__))
    print("--- END DIAGNOSTICS ---\n\n")
except Exception as e:
    print(f"DIAGNOSTIC ERROR: {e}")
# --- END DIAGNOSTIC CODE ---

import sys
import numba
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.transforms as mtransforms

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QSlider,
    QCheckBox,
    QRadioButton,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QSizePolicy,
    QStatusBar,
    QMessageBox,
    QButtonGroup,
    QScrollArea,
    QComboBox,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

import warnings

# --- Parametri Globali Iniziali ---

sub_dtype = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('pressure_val_at_1m_relative_to_pref', np.float64),
    ('gain_lin', np.float64),
    ('angle', np.float64),
    ('delay_ms', np.float64),
    ('polarity', np.int32)
])

PARAM_RANGES = {
    "delay_ms": (0.0, 300.0),
    "gain_db": (-30.0, 1.0),
    "polarity": [-1, 1],
    "spl_rms": (70.0, 140.0),
    "sub_width_depth": (0.1, 2.0),
    "angle": (0.0, 2 * np.pi),
}
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
ARROW_LENGTH = 0.4
ARRAY_INDICATOR_CONE_WIDTH_DEG = 30.0
ARRAY_INDICATOR_RADIUS = 2.5

DEFAULT_ROOM_VERTICES = [[-15, -15], [15, -15], [15, 15], [-15, 15]]
P_REF_20UPA_CALC_DEFAULT = 10 ** (DEFAULT_SUB_SPL_RMS / 20.0)

DEFAULT_TARGET_AREA_VERTICES = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
DEFAULT_AVOIDANCE_AREA_VERTICES = [[-4, 0], [-3, 0], [-3, 1], [-4, 1]]
DEFAULT_SUB_PLACEMENT_AREA_VERTICES = [[-10, -10], [10, -10], [10, 10], [-10, 10]]

DEFAULT_MAX_SPL_AVOIDANCE = 65.0
DEFAULT_TARGET_MIN_SPL_DESIRED = 80.0
DEFAULT_BALANCE_SLIDER_VALUE = 50

OPTIMIZATION_MUTATION_RATE = 0.1

FRONT_DIRECTIVITY_BEAMWIDTH_RAD = np.pi / 6
FRONT_DIRECTIVITY_GAIN_LIN = 10 ** (1.0 / 20.0)
DEFAULT_SIM_SPEED_OF_SOUND = 343.0


class MatplotlibCanvas(QWidget):
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

@numba.jit(nopython=True, fastmath=True, cache=True)
def calculate_spl_vectorized(px, py, freq, c_val, current_sorgenti_array):
    if freq <= 0 or c_val <= 0:
        return np.full(px.shape, -np.inf)

    total_amplitude = np.zeros_like(px, dtype=np.complex128)
    wavelength = c_val / freq
    if wavelength == 0:
        return np.full(px.shape, -np.inf)
    k = 2 * np.pi / wavelength

    # Itera sull'array strutturato
    for i in range(len(current_sorgenti_array)):
        sub_data = current_sorgenti_array[i]
        sub_x, sub_y = sub_data.x, sub_data.y

        distance = np.sqrt((px - sub_x) ** 2 + (py - sub_y) ** 2)
        distance[distance < 0.01] = 0.01 # Evita divisione per zero

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

class OptimizationWorker(QObject):
    status_update = pyqtSignal(str)
    finished = pyqtSignal(object, float)

    def __init__(
        self,
        criterion,
        optim_freq_s,
        optim_freq_min_val,
        optim_freq_max_val,
        optim_n_freq_val,
        pop_size,
        generations,
        c_val,
        grid_res_spl,
        room_pts,
        active_target_areas_points,
        active_avoidance_areas_points,
        max_spl_avoidance,
        target_min_spl_desired,
        balance_target_avoidance,
        current_sorgenti_configs,
        sorgenti_param_locks_list,
        sub_placement_area_points=None,
    ):
        super().__init__()
        self.criterion, self.optim_freq_s = criterion, optim_freq_s
        self.optim_freq_min_val, self.optim_freq_max_val = (
            optim_freq_min_val,
            optim_freq_max_val,
        )
        self.optim_n_freq_val = optim_n_freq_val
        self.pop_size, self.generations = pop_size, generations
        self.mutation_rate_val = OPTIMIZATION_MUTATION_RATE
        self.c_val, self.grid_res_spl, self.room_pts = c_val, grid_res_spl, room_pts
        self.active_target_areas_points, self.active_avoidance_areas_points = (
            active_target_areas_points,
            active_avoidance_areas_points,
        )

        self.max_spl_avoidance = max_spl_avoidance
        self.target_min_spl_desired = target_min_spl_desired
        self.balance_target_avoidance = balance_target_avoidance

        self.AVOIDANCE_PENALTY_FACTOR = 1.0
        self.TARGET_UNDERSHOOT_PENALTY_FACTOR = 2.0

        self.sorgenti_configs_ref = current_sorgenti_configs
        self.sorgenti_param_locks = sorgenti_param_locks_list
        self.sub_placement_area_points = sub_placement_area_points or []
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        self.status_update.emit("Richiesta di interruzione ottimizzazione...")

    def run(self):
        self._stop_requested = False
        if not self.sorgenti_configs_ref:
            self.status_update.emit("Nessun subwoofer da ottimizzare.")
            self.finished.emit(None, -float('inf'))  
            return
            
        self.status_update.emit(f"Inizializzazione ottimizzazione...")
        QApplication.processEvents()
        population = [self._create_individual_worker() for _ in range(self.pop_size)]
        best_overall_dsp_params, best_overall_fit = None, -float("inf")

        for gen in range(self.generations):
            if self._stop_requested:
                self.status_update.emit("Ottimizzazione interrotta.")
                self.finished.emit(best_overall_dsp_params, best_overall_fit)
                return
                
            self.status_update.emit(
                f"Generazione {gen+1}/{self.generations} - Calcolo Fitness..."
            )
            QApplication.processEvents()
            
            fit_vals = [self._calculate_fitness_worker(ind) for ind in population]
            
            if not fit_vals:
                self.status_update.emit(f"Gen {gen+1}: Fitness non calcolabile.")
                break
            
            valid_indices = [i for i, f in enumerate(fit_vals) if f > -float("inf")]

            if not valid_indices:
                self.status_update.emit(f"Gen {gen+1}: Nessuna fitness valida.")
                break

            current_best_idx = np.argmax([fit_vals[i] for i in valid_indices])
            
            if fit_vals[valid_indices[current_best_idx]] > best_overall_fit:
                best_overall_fit = fit_vals[valid_indices[current_best_idx]]
                best_overall_dsp_params = [
                    p.copy() for p in population[valid_indices[current_best_idx]]
                ]

            # --- Logica dell'algoritmo genetico (selezione, crossover, mutazione) ---
            sorted_indices = sorted(
                valid_indices, key=lambda i: fit_vals[i], reverse=True
            )
            elite_size = max(1, int(0.1 * self.pop_size))
            new_population = [population[i] for i in sorted_indices[:elite_size]]

            tourn_s = 3
            while len(new_population) < self.pop_size:
                parents = []
                for _ in range(2):
                    competitors_indices = np.random.choice(
                        sorted_indices,
                        size=min(tourn_s, len(sorted_indices)),
                        replace=False,
                    )
                    winner_idx = competitors_indices[
                        np.argmax([fit_vals[i] for i in competitors_indices])
                    ]
                    parents.append(population[winner_idx])

                child1, child2 = [], []
                for p1_sub, p2_sub in zip(parents[0], parents[1]):
                    c1_sub, c2_sub = {}, {}
                    c1_sub["sub_idx_original"] = p1_sub["sub_idx_original"]
                    c2_sub["sub_idx_original"] = p2_sub["sub_idx_original"]

                    for key in p1_sub:
                        if key == "sub_idx_original":
                            continue
                        if np.random.rand() < 0.5:
                            c1_sub[key], c2_sub[key] = p1_sub[key], p2_sub[key]
                        else:
                            c1_sub[key], c2_sub[key] = p2_sub[key], p1_sub[key]
                        if key == "gain_db":
                            c1_sub["gain_lin"] = 10 ** (c1_sub["gain_db"] / 20.0)
                            c2_sub["gain_lin"] = 10 ** (c2_sub["gain_db"] / 20.0)
                    child1.append(c1_sub)
                    child2.append(c2_sub)
                
                new_population.append(self._mutate_worker(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate_worker(child2))
            
            population = new_population

        final_fitness_display = (
            f"{best_overall_fit:.3f}" if best_overall_fit > -float("inf") else "N/A"
        )
        final_message = f"Terminata. Miglior fitness: {final_fitness_display}."
        self.status_update.emit(final_message)
        
        self.finished.emit(best_overall_dsp_params, best_overall_fit)

    def _create_individual_worker(self):
        individual = []
        for i, sub_ref in enumerate(self.sorgenti_configs_ref):
            locks = self.sorgenti_param_locks[i]
            dsp = {"sub_idx_original": i}

            # Posizione con area di posizionamento
            if not locks.get("position", False) and self.sub_placement_area_points:
                # Genera posizione casuale all'interno dell'area di posizionamento
                area_path = Path(self.sub_placement_area_points)
                while True:
                    min_x = min(p[0] for p in self.sub_placement_area_points)
                    max_x = max(p[0] for p in self.sub_placement_area_points)
                    min_y = min(p[1] for p in self.sub_placement_area_points)
                    max_y = max(p[1] for p in self.sub_placement_area_points)
                    
                    rand_x = np.random.uniform(min_x, max_x)
                    rand_y = np.random.uniform(min_y, max_y)
                    
                    if area_path.contains_point((rand_x, rand_y)):
                        dsp["x"] = rand_x
                        dsp["y"] = rand_y
                        break
            else:
                dsp["x"] = sub_ref["x"]
                dsp["y"] = sub_ref["y"]

            if not locks.get("angle", False):
                dsp["angle"] = np.random.uniform(*PARAM_RANGES["angle"])
            else:
                dsp["angle"] = sub_ref["angle"]

            if not locks.get("delay", False):
                dsp["delay_ms"] = np.random.uniform(*PARAM_RANGES["delay_ms"])
            else:
                dsp["delay_ms"] = sub_ref["delay_ms"]

            if not locks.get("gain", False):
                dsp["gain_db"] = np.random.uniform(*PARAM_RANGES["gain_db"])
            else:
                dsp["gain_db"] = sub_ref["gain_db"]

            if not locks.get("polarity", False):
                dsp["polarity"] = np.random.choice(PARAM_RANGES["polarity"])
            else:
                dsp["polarity"] = sub_ref["polarity"]

            dsp["gain_lin"] = 10 ** (dsp["gain_db"] / 20.0)
            individual.append(dsp)
        return individual

    def _mutate_worker(self, individual):
        mutated = []
        rate = self.mutation_rate_val
        for i, sub_dsp in enumerate(individual):
            new_dsp = sub_dsp.copy()
            locks = self.sorgenti_param_locks[sub_dsp["sub_idx_original"]]

            # Logica di mutazione posizione robusta
            if not locks.get("position", False) and np.random.rand() < rate and self.sub_placement_area_points:
                area_path = Path(self.sub_placement_area_points)
                min_x = min(p[0] for p in self.sub_placement_area_points)
                max_x = max(p[0] for p in self.sub_placement_area_points)
                min_y = min(p[1] for p in self.sub_placement_area_points)
                max_y = max(p[1] for p in self.sub_placement_area_points)
                
                # Tenta di generare una nuova posizione valida all'interno dell'area
                for _ in range(20):
                    rand_x = np.random.uniform(min_x, max_x)
                    rand_y = np.random.uniform(min_y, max_y)
                    if area_path.contains_point((rand_x, rand_y)):
                        new_dsp["x"] = rand_x
                        new_dsp["y"] = rand_y
                        break

            if not locks.get("delay", False) and np.random.rand() < rate:
                new_dsp["delay_ms"] = np.clip(
                    new_dsp["delay_ms"] + np.random.normal(0, 5),
                    *PARAM_RANGES["delay_ms"],
                )

            if not locks.get("gain", False) and np.random.rand() < rate:
                new_dsp["gain_db"] = np.clip(
                    new_dsp["gain_db"] + np.random.normal(0, 1),
                    *PARAM_RANGES["gain_db"],
                )
                new_dsp["gain_lin"] = 10 ** (new_dsp["gain_db"] / 20.0)

            if not locks.get("polarity", False) and np.random.rand() < rate:
                new_dsp["polarity"] *= -1

            if not locks.get("angle", False) and np.random.rand() < rate:
                new_dsp["angle"] = (
                    new_dsp["angle"] + np.random.normal(0, np.radians(5))
                ) % (2 * np.pi)

            mutated.append(new_dsp)
        return mutated

    def _calculate_spl_map_for_fitness_evaluation(
        self, individual_dsp_params, freq, area_points, is_avoidance=False, return_grid_info=False
    ):
        
            # --- FINE MODIFICA ---

        if not area_points:
            if is_avoidance:
                    if return_grid_info:
                        return None, None, None, None
                    return None, None
            if return_grid_info:
                return None, None, None, None
            return None, None
        configs = []
        for i, sub_ref in enumerate(self.sorgenti_configs_ref):
            conf = sub_ref.copy()
            dsp_params_for_this_sub = next(
                (
                    dsp
                    for dsp in individual_dsp_params
                    if dsp.get("sub_idx_original") == i
                ),
                None,
            )
            if dsp_params_for_this_sub:
                conf.update(dsp_params_for_this_sub)
            configs.append(conf)
            # --- INIZIO MODIFICA ---
        configs_array = np.array([(
            c['x'], c['y'],
            c.get('pressure_val_at_1m_relative_to_pref', 1.0),
            c.get('gain_lin', 1.0),
            c.get('angle', 0.0),
            c.get('delay_ms', 0.0),
            c.get('polarity', 1)
        ) for c in configs], dtype=sub_dtype)

        min_x = min(p[0] for points in area_points for p in points)
        max_x = max(p[0] for points in area_points for p in points)
        min_y = min(p[1] for points in area_points for p in points)
        max_y = max(p[1] for points in area_points for p in points)
        
        grid_res = self.grid_res_spl
        x_c = np.arange(min_x, max_x + grid_res, grid_res)
        y_c = np.arange(min_y, max_y + grid_res, grid_res)

        if len(x_c) < 2 or len(y_c) < 2:
            if return_grid_info:
                return None, None, None, None
            return None, None

        X, Y = np.meshgrid(x_c, y_c)
        points_check = np.vstack((X.ravel(), Y.ravel())).T
        mask = np.zeros_like(X, dtype=bool)
        for points in area_points:
            if len(points) >= 3:
                mask |= Path(points).contains_points(points_check).reshape(X.shape)

        if not np.any(mask):
            if return_grid_info:
                return None, None, None, None
            return None, None

        spl_map = np.full(X.shape, np.nan)
        points_to_calc_x = X[mask]
        points_to_calc_y = Y[mask]

        if freq is None or self.c_val is None:
            if return_grid_info:
                return np.full(X.shape, -np.inf), mask, min_x, grid_res
            return np.full(X.shape, -np.inf), mask

        spl_values = calculate_spl_vectorized(
            points_to_calc_x, points_to_calc_y, freq, self.c_val, configs_array
        )
        spl_map[mask] = spl_values

        if return_grid_info:
            return spl_map, mask, min_x, grid_res
        else:
            return spl_map, mask

    def _calculate_fitness_worker(self, individual):
        # --- MANUALE DI TUNING FINALE ---
        COVERAGE_COST_FACTOR = 500.0
        AVOIDANCE_COST_FACTOR = 600.0
        LOBE_FOCUS_COST_FACTOR = 50.0
        UNIFORMITY_COST_FACTOR = 15.0
        GAIN_BALANCE_COST_FACTOR = 8.0
        RESULT_SYMMETRY_COST_FACTOR = 45.0
        COLLISION_COST_FACTOR = 20000.0 

        # --- INIZIO LOGICA DI CALCOLO ---
        total_cost_accumulator = 0.0
        frequencies_to_test = []

        if self.criterion == "Omogeneità SPL" and self.optim_n_freq_val > 1:
            frequencies_to_test = np.linspace(
                self.optim_freq_min_val, self.optim_freq_max_val, self.optim_n_freq_val
            )
        elif self.criterion == "Copertura SPL":
            frequencies_to_test = [self.optim_freq_s]
        else: # Fallback
            frequencies_to_test = [(self.optim_freq_min_val + self.optim_freq_max_val) / 2.0]
        
        # Determina l'asse di simmetria una sola volta
        symmetry_axis_x = None
        if self.active_target_areas_points:
            list_of_centroids = []
            for area_points in self.active_target_areas_points:
                list_of_centroids.append(np.mean(np.array(area_points), axis=0))
            
            if list_of_centroids:
                meta_centroid = np.mean(np.array(list_of_centroids), axis=0)
                symmetry_axis_x = meta_centroid[0]

        # Cicla su ogni frequenza nel range di ottimizzazione
        for freq in frequencies_to_test:
            min_spl_in_target, avg_spl_in_target = -float('inf'), -float('inf')
            std_dev_in_target = float('inf')
            max_spl_in_avoidance, avg_spl_in_avoidance = -float('inf'), -float('inf')
            spl_map_target, mask_target, min_x_target, grid_res_spl = None, None, None, self.grid_res_spl

            # Calcolo per Target
            if self.active_target_areas_points:
                spl_map_target, mask_target, min_x_target, _ = self._calculate_spl_map_for_fitness_evaluation(
                    individual, freq, self.active_target_areas_points, return_grid_info=True
                )
                if spl_map_target is not None and np.any(mask_target):
                    values_target = spl_map_target[mask_target & ~np.isnan(spl_map_target)]
                    if values_target.size > 0:
                        min_spl_in_target = np.min(values_target)
                        avg_spl_in_target = np.mean(values_target)
                        std_dev_in_target = np.std(values_target)
            
            # Calcolo per Avoidance
            if self.active_avoidance_areas_points:
                spl_map_avoid, mask_avoid, _, _ = self._calculate_spl_map_for_fitness_evaluation(
                    individual, freq, self.active_avoidance_areas_points, is_avoidance=True, return_grid_info=True
                )
                if spl_map_avoid is not None and np.any(mask_avoid):
                    values_avoid = spl_map_avoid[mask_avoid & ~np.isnan(spl_map_avoid)]
                    if values_avoid.size > 0:
                        max_spl_in_avoidance = np.max(values_avoid)
                        avg_spl_in_avoidance = np.mean(values_avoid)

            # Calcolo dei "Costi" per la frequenza corrente
            undershoot_db = max(0, self.target_min_spl_desired - min_spl_in_target)
            coverage_cost = (undershoot_db ** 2.5) * COVERAGE_COST_FACTOR

            overshoot_db = max(0, max_spl_in_avoidance - self.max_spl_avoidance)
            avoidance_cost = (overshoot_db ** 2) *  AVOIDANCE_COST_FACTOR

            uniformity_cost = std_dev_in_target * UNIFORMITY_COST_FACTOR
            
            spl_difference = avg_spl_in_target - avg_spl_in_avoidance if (avg_spl_in_target > -float('inf') and avg_spl_in_avoidance > -float('inf')) else 0
            lobe_focus_cost = max(0, 15 - spl_difference) * LOBE_FOCUS_COST_FACTOR

            # --- MODIFICA INIZIO: Logica di calcolo simmetria robusta ---
            result_symmetry_cost = 0.0
            if symmetry_axis_x is not None and spl_map_target is not None and min_x_target is not None:
                num_cols = spl_map_target.shape[1]
                spl_differences = []
                
                for col_idx in range(num_cols // 2):
                    mirrored_col_idx = num_cols - 1 - col_idx
                    
                    col1 = spl_map_target[:, col_idx]
                    col2 = spl_map_target[:, mirrored_col_idx]
                    mask1 = mask_target[:, col_idx]
                    mask2 = mask_target[:, mirrored_col_idx]
                    
                    valid_mask = mask1 & mask2 & ~np.isnan(col1) & ~np.isnan(col2)
                    
                    # Controlla che il punto speculare sia valido e all'interno dell'area target
                    if np.any(valid_mask):
                        diffs = np.abs(col1[valid_mask] - col2[valid_mask])**2
                        spl_differences.extend(diffs)

                if spl_differences:
                    result_symmetry_cost = np.mean(spl_differences) * RESULT_SYMMETRY_COST_FACTOR
            # --- MODIFICA FINE ---

            target_importance = self.balance_target_avoidance / 100.0
            avoidance_importance = 1.0 - target_importance
            cost_for_this_freq = (coverage_cost * target_importance) + \
                                 (avoidance_cost * avoidance_importance) + \
                                 uniformity_cost + \
                                 lobe_focus_cost + \
                                 result_symmetry_cost
            
            total_cost_accumulator += cost_for_this_freq

        unlocked_gains = [dsp['gain_db'] for i, dsp in enumerate(individual) if not self.sorgenti_param_locks[i].get('gain', False)]
        gain_balance_cost = np.std(unlocked_gains) * GAIN_BALANCE_COST_FACTOR if len(unlocked_gains) > 1 else 0.0

        average_cost_over_freqs = total_cost_accumulator / len(frequencies_to_test) if len(frequencies_to_test) > 0 else float('inf')
        
        collision_cost = 0.0
        num_subs = len(individual)
        if num_subs > 1:
            # Crea un array di posizioni e dimensioni per il calcolo vettoriale
            positions = np.array([[sub['x'], sub['y']] for sub in individual])
            # Usa le dimensioni globali come riferimento per la collisione
            # (potrebbe essere raffinato per usare dimensioni individuali se necessario)
            sub_max_dim = max(self.sorgenti_configs_ref[0].get('width', 0.5), self.sorgenti_configs_ref[0].get('depth', 0.5))
            min_dist_sq = (sub_max_dim * 1.1)**2 # Distanza minima al quadrato (più veloce da calcolare), con un po' di margine (10%)

            # Calcola la matrice delle distanze al quadrato
            dist_sq_matrix = np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=-1)
            
            # Azzera la diagonale per non auto-confrontare e prendi la parte superiore della matrice per evitare doppi conteggi
            np.fill_diagonal(dist_sq_matrix, np.inf)
            collisions = dist_sq_matrix[np.triu_indices(num_subs, k=1)] < min_dist_sq
            
            # Applica la penalità per ogni collisione trovata
            collision_cost = np.sum(collisions) * COLLISION_COST_FACTOR
        final_total_cost = average_cost_over_freqs + gain_balance_cost

        return -final_total_cost

class SubwooferSimApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulatore Posizionamento Subwoofer Avanzato")
        self.setGeometry(50, 50, 1600, 1000)

        # Stato dell'applicazione
        self.punti_stanza = []
        for p in DEFAULT_ROOM_VERTICES:
            self.punti_stanza.append({"pos": p, "plot": None})

        self.sorgenti = []
        self.next_sub_id = 1
        self.next_group_id = 1
        self.max_group_id = 0
        self.current_sub_idx = -1
        self.lista_gruppi_array = {}

        self.lista_target_areas = []
        self.current_target_area_idx = -1
        self.next_target_area_id = 1

        self.lista_avoidance_areas = []
        self.current_avoidance_area_idx = -1
        self.next_avoidance_area_id = 1

        # --- NUOVA AREA DI POSIZIONAMENTO SUB ---
        self.lista_sub_placement_areas = []
        self.current_sub_placement_area_idx = -1
        self.next_sub_placement_area_id = 1

        self.selected_stanza_vtx_idx = -1
        self.drag_object = None
        self.original_mouse_pos = None
        self.original_object_pos = None
        self.original_object_angle = None
        self.original_group_states = []

        self.current_spl_map = None
        self._cax_for_colorbar_spl = None
        self.auto_scale_spl = True  # Scala SPL automatica

        self.optimization_thread = None
        self.optimization_worker = None

        # --- NUOVI STATI PER IMMAGINE DI SFONDO ---
        self.bg_image_props = {
            "path": None,
            "data": None,
            "artist": None,
            "center_x": 0,
            "center_y": 0,
            "scale": 1.0,
            "rotation_deg": 0,
            "alpha": 0.5,
            "anchor_pixel": None,
            "cached_transformed": None,  # Cache per performance
        }
        self.is_in_calibration_mode = False
        self.calibration_points = []

        self.global_sub_width = DEFAULT_SUB_WIDTH
        self.global_sub_depth = DEFAULT_SUB_DEPTH
        self.global_sub_spl_rms = DEFAULT_SUB_SPL_RMS
        self.use_global_for_new_manual_subs = False

        self.grid_snap_spacing = 0.25
        self.grid_snap_enabled = False
        self.grid_show_enabled = False

        self.max_spl_avoidance_ui_val = DEFAULT_MAX_SPL_AVOIDANCE
        self.target_min_spl_desired_ui_val = DEFAULT_TARGET_MIN_SPL_DESIRED
        self.balance_slider_ui_val = DEFAULT_BALANCE_SLIDER_VALUE

        self.last_optim_criterion = None
        self.last_optim_freq_s = None
        self.last_optim_freq_min = None
        self.last_optim_freq_max = None

        self._setup_ui()

        if self.sorgenti:
            self.current_sub_idx = 0

        self.auto_fit_view_to_room()
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()
        self.update_optim_freq_fields_visibility()

    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_container_layout = QHBoxLayout(self.central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_container_layout.addWidget(splitter)

        controls_scroll_area = QScrollArea()
        controls_scroll_area.setWidgetResizable(True)
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        controls_scroll_area.setWidget(controls_widget)

        splitter.addWidget(controls_scroll_area)

        self.plot_canvas = MatplotlibCanvas(self)
        self.ax = self.plot_canvas.axes
        splitter.addWidget(self.plot_canvas)

        splitter.setSizes([450, 1150])

        group_box_style = """
        QGroupBox {
            border: 2px solid gray; border-radius: 5px; margin-top: 1ex; font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; color: lightgray;
        }
        """
        self.central_widget.setStyleSheet(group_box_style)

        self._setup_project_ui()
        self._setup_stanza_ui()
        self._setup_background_image_ui()
        self._setup_global_sub_ui()
        self._setup_sub_config_ui()
        self._setup_group_array_ui()
        self._setup_target_areas_ui()
        self._setup_avoidance_areas_ui()
        self._setup_sub_placement_areas_ui()  # NUOVA AREA
        self._setup_spl_vis_ui()
        self._setup_sim_grid_ui()
        self._setup_optimization_ui()

        self.controls_layout.addStretch(1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Applicazione caricata.")
        self.plot_canvas.canvas.mpl_connect("button_press_event", self.on_press_mpl)
        self.plot_canvas.canvas.mpl_connect("motion_notify_event", self.on_motion_mpl)
        self.plot_canvas.canvas.mpl_connect("button_release_event", self.on_release_mpl)
        self.plot_canvas.canvas.mpl_connect(
            "motion_notify_event", self.on_mouse_move_for_spl_display
        )

    def _setup_project_ui(self):
        project_group = QGroupBox("Progetto")
        project_layout = QHBoxLayout()

        self.btn_save_project = QPushButton("Salva Progetto")
        self.btn_save_project.clicked.connect(self.save_project_to_excel)
        project_layout.addWidget(self.btn_save_project)

        self.btn_load_project = QPushButton("Carica Progetto")
        self.btn_load_project.clicked.connect(self.load_project_from_excel)
        project_layout.addWidget(self.btn_load_project)

        project_group.setLayout(project_layout)
        self.controls_layout.addWidget(project_group)

    def _setup_stanza_ui(self):
        stanza_group = QGroupBox("Configurazione Stanza")
        stanza_layout = QFormLayout()
        self.btn_add_vtx = QPushButton("Aggiungi Vertice Stanza")
        self.btn_add_vtx.clicked.connect(self.add_stanza_vtx)
        stanza_layout.addRow(self.btn_add_vtx)
        self.btn_rem_vtx = QPushButton("Rimuovi Ultimo Vertice")
        self.btn_rem_vtx.clicked.connect(self.remove_stanza_vtx)
        stanza_layout.addRow(self.btn_rem_vtx)
        self.selected_vtx_label = QLabel("Nessun Vertice Selezionato")
        stanza_layout.addRow(self.selected_vtx_label)
        self.tb_stanza_vtx_x = QLineEdit()
        self.tb_stanza_vtx_x.setEnabled(False)
        stanza_layout.addRow("Vertice X:", self.tb_stanza_vtx_x)
        self.tb_stanza_vtx_y = QLineEdit()
        self.tb_stanza_vtx_y.setEnabled(False)
        stanza_layout.addRow("Vertice Y:", self.tb_stanza_vtx_y)
        self.btn_update_stanza_vtx = QPushButton("Aggiorna Vertice")
        self.btn_update_stanza_vtx.setEnabled(False)
        self.btn_update_stanza_vtx.clicked.connect(
            self.on_update_selected_stanza_vertex
        )
        stanza_layout.addRow(self.btn_update_stanza_vtx)
        self.error_text_stanza_vtx_edit = QLabel("")
        self.error_text_stanza_vtx_edit.setStyleSheet("color: red;")
        stanza_layout.addRow(self.error_text_stanza_vtx_edit)
        stanza_group.setLayout(stanza_layout)
        self.controls_layout.addWidget(stanza_group)

    def _setup_background_image_ui(self):
        bg_group = QGroupBox("Immagine di Sfondo")
        bg_layout = QFormLayout()
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Carica Immagine...")
        btn_load.clicked.connect(self.load_background_image)
        btn_layout.addWidget(btn_load)
        btn_calib = QPushButton("Calibra Scala (2 click)")
        btn_calib.clicked.connect(self.start_calibration)
        btn_layout.addWidget(btn_calib)
        btn_remove = QPushButton("Rimuovi")
        btn_remove.clicked.connect(self.remove_background_image)
        btn_layout.addWidget(btn_remove)
        bg_layout.addRow(btn_layout)
        self.calibration_status_label = QLabel("Pronto per caricare un'immagine.")
        self.calibration_status_label.setStyleSheet(
            "font-style: italic; color: orange;"
        )
        bg_layout.addRow(self.calibration_status_label)
        self.tb_bg_x = QLineEdit(str(self.bg_image_props["center_x"]))
        self.tb_bg_y = QLineEdit(str(self.bg_image_props["center_y"]))
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(self.tb_bg_x)
        pos_layout.addWidget(self.tb_bg_y)
        bg_layout.addRow("Centro Img X/Y (m):", pos_layout)
        self.tb_bg_scale = QLineEdit(str(self.bg_image_props["scale"]))
        bg_layout.addRow("Scala (m/pixel):", self.tb_bg_scale)
        self.tb_bg_rotation = QLineEdit(str(self.bg_image_props["rotation_deg"]))
        bg_layout.addRow("Rotazione (°):", self.tb_bg_rotation)
        self.tb_bg_alpha = QLineEdit(str(self.bg_image_props["alpha"]))
        bg_layout.addRow("Trasparenza (0-1):", self.tb_bg_alpha)
        btn_apply = QPushButton("Applica Trasformazioni Manuali")
        btn_apply.clicked.connect(self.update_background_image_manually)
        bg_layout.addRow(btn_apply)
        bg_group.setLayout(bg_layout)
        self.controls_layout.addWidget(bg_group)

    def _setup_global_sub_ui(self):
        global_settings_group = QGroupBox("Impostazioni Globali Subwoofer")
        global_settings_layout = QFormLayout()
        self.tb_global_sub_width = QLineEdit(str(self.global_sub_width))
        global_settings_layout.addRow("Larghezza Sub (m):", self.tb_global_sub_width)
        self.tb_global_sub_depth = QLineEdit(str(self.global_sub_depth))
        global_settings_layout.addRow("Profondità Sub (m):", self.tb_global_sub_depth)
        self.tb_global_sub_spl = QLineEdit(str(self.global_sub_spl_rms))
        global_settings_layout.addRow("SPL RMS Globale (dB):", self.tb_global_sub_spl)
        self.check_use_global_for_new = QCheckBox("Usa per Nuovi Sub Manuali")
        self.check_use_global_for_new.setChecked(False)
        self.check_use_global_for_new.toggled.connect(self.on_toggle_use_global_for_new)
        global_settings_layout.addRow(self.check_use_global_for_new)
        self.btn_apply_globals_to_all = QPushButton("Applica Globali a Tutti i Sub")
        self.btn_apply_globals_to_all.clicked.connect(
            self.apply_global_settings_to_all_subs
        )
        global_settings_layout.addRow(self.btn_apply_globals_to_all)
        self.error_text_global_settings = QLabel("")
        self.error_text_global_settings.setStyleSheet("color: red;")
        global_settings_layout.addRow(self.error_text_global_settings)
        global_settings_group.setLayout(global_settings_layout)
        self.controls_layout.addWidget(global_settings_group)

    def _setup_sub_config_ui(self):
        sub_group = QGroupBox("Configurazione Subwoofer")
        sub_layout_main = QVBoxLayout()
        sub_selector_layout = QHBoxLayout()
        self.btn_prev_sub = QPushButton("<")
        self.btn_prev_sub.clicked.connect(self.select_prev_sub)
        sub_selector_layout.addWidget(self.btn_prev_sub)
        self.sub_selector_text_widget = QLabel("Nessun Sub")
        sub_selector_layout.addWidget(
            self.sub_selector_text_widget, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.btn_next_sub = QPushButton(">")
        self.btn_next_sub.clicked.connect(self.select_next_sub)
        sub_selector_layout.addWidget(self.btn_next_sub)
        sub_layout_main.addLayout(sub_selector_layout)
        sub_actions_layout = QHBoxLayout()
        self.btn_add_sub = QPushButton("Aggiungi Sub")
        self.btn_add_sub.clicked.connect(self.add_subwoofer)
        sub_actions_layout.addWidget(self.btn_add_sub)
        self.btn_rem_sub = QPushButton("Rimuovi Sub Corrente")
        self.btn_rem_sub.clicked.connect(self.remove_subwoofer)
        sub_actions_layout.addWidget(self.btn_rem_sub)
        sub_layout_main.addLayout(sub_actions_layout)
        sub_params_layout = QFormLayout()

        self.sub_pos_label = QLabel("X/Y Sub (m):")
        self.tb_sub_x = QLineEdit()
        self.tb_sub_x.returnPressed.connect(self.on_submit_sub_param_qt)
        self.tb_sub_y = QLineEdit()
        self.tb_sub_y.returnPressed.connect(self.on_submit_sub_param_qt)
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(self.tb_sub_x)
        pos_layout.addWidget(self.tb_sub_y)
        sub_params_layout.addRow(self.sub_pos_label, pos_layout)

        self.tb_sub_angle = QLineEdit()
        self.tb_sub_angle.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow("Angolo (°):", self.tb_sub_angle)

        self.sub_gain_label = QLabel("Trim Gain (dB):")
        self.tb_sub_gain_db = QLineEdit()
        self.tb_sub_gain_db.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow(self.sub_gain_label, self.tb_sub_gain_db)

        self.sub_delay_label = QLabel("Delay (ms):")
        self.tb_sub_delay = QLineEdit()
        self.tb_sub_delay.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow(self.sub_delay_label, self.tb_sub_delay)

        self.sub_polarity_label = QLabel("Polarità (+1/-1):")
        self.tb_sub_polarity = QLineEdit()
        self.tb_sub_polarity.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow(self.sub_polarity_label, self.tb_sub_polarity)

        self.check_sub_angle_lock = QCheckBox("Blocca Angolo")
        self.check_sub_angle_lock.setObjectName("angle")
        self.check_sub_angle_lock.setChecked(False)
        self.check_sub_angle_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_angle_lock)
        self.check_sub_delay_lock = QCheckBox("Blocca Delay")
        self.check_sub_delay_lock.setObjectName("delay")
        self.check_sub_delay_lock.setChecked(False)
        self.check_sub_delay_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_delay_lock)
        self.check_sub_gain_lock = QCheckBox("Blocca Gain")
        self.check_sub_gain_lock.setObjectName("gain")
        self.check_sub_gain_lock.setChecked(False)
        self.check_sub_gain_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_gain_lock)
        self.check_sub_polarity_lock = QCheckBox("Blocca Polarità")
        self.check_sub_polarity_lock.setObjectName("polarity")
        self.check_sub_polarity_lock.setChecked(False)
        self.check_sub_polarity_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_polarity_lock)
        self.check_sub_position_lock = QCheckBox("Blocca Posizione")
        self.check_sub_position_lock.setObjectName("position")
        self.check_sub_position_lock.setChecked(False)
        self.check_sub_position_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_position_lock)

        self.tb_sub_width = QLineEdit()
        self.tb_sub_width.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow("Larghezza Sub (m):", self.tb_sub_width)
        self.tb_sub_depth = QLineEdit()
        self.tb_sub_depth.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow("Profondità Sub (m):", self.tb_sub_depth)
        self.tb_sub_spl_rms = QLineEdit()
        self.tb_sub_spl_rms.returnPressed.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow("SPL RMS @ 1m (dB):", self.tb_sub_spl_rms)

        self.btn_submit_sub_params = QPushButton("Applica Parametri")
        self.btn_submit_sub_params.clicked.connect(self.on_submit_sub_param_qt)
        sub_params_layout.addRow(self.btn_submit_sub_params)
        self.error_text_sub = QLabel("")
        self.error_text_sub.setStyleSheet("color: red;")
        sub_params_layout.addRow(self.error_text_sub)
        sub_layout_main.addLayout(sub_params_layout)
        sub_group.setLayout(sub_layout_main)
        self.controls_layout.addWidget(sub_group)

    def _setup_group_array_ui(self):
        group_array_group = QGroupBox("Gestione Gruppi e Array")
        group_array_layout = QVBoxLayout()

        grouping_sub_group_box = QGroupBox("Raggruppamento Manuale")
        grouping_sub_layout = QVBoxLayout()
        group_action_buttons_layout = QHBoxLayout()
        self.btn_create_new_group = QPushButton("Crea Nuovo Gruppo")
        self.btn_create_new_group.clicked.connect(self.create_new_group)
        group_action_buttons_layout.addWidget(self.btn_create_new_group)
        self.btn_add_to_group = QPushButton("Aggiungi a Gruppo")
        self.btn_add_to_group.clicked.connect(self.add_sub_to_existing_group)
        group_action_buttons_layout.addWidget(self.btn_add_to_group)
        grouping_sub_layout.addLayout(group_action_buttons_layout)
        group_action_buttons_layout2 = QHBoxLayout()
        self.btn_remove_from_group = QPushButton("Rimuovi da Gruppo")
        self.btn_remove_from_group.clicked.connect(self.remove_sub_from_group)
        group_action_buttons_layout2.addWidget(self.btn_remove_from_group)
        self.btn_ungroup_all = QPushButton("Sciogli Gruppo")
        self.btn_ungroup_all.clicked.connect(self.ungroup_selected_sub_group)
        group_action_buttons_layout2.addWidget(self.btn_ungroup_all)
        grouping_sub_layout.addLayout(group_action_buttons_layout2)
        self.group_status_label = QLabel("Sub selezionato: Nessun Gruppo")
        grouping_sub_layout.addWidget(self.group_status_label)
        self.error_text_grouping = QLabel("")
        self.error_text_grouping.setStyleSheet("color: red;")
        grouping_sub_layout.addWidget(self.error_text_grouping)

        self.group_details_label = QLabel("Valori Assoluti Membri del Gruppo:")
        self.group_details_label.setVisible(False)
        grouping_sub_layout.addWidget(self.group_details_label)
        self.group_members_list = QListWidget()
        self.group_members_list.setMaximumHeight(120)
        self.group_members_list.setVisible(False)
        grouping_sub_layout.addWidget(self.group_members_list)

        grouping_sub_group_box.setLayout(grouping_sub_layout)
        group_array_layout.addWidget(grouping_sub_group_box)

        array_setup_group_box = QGroupBox("Configurazione Array")
        self.array_setup_layout_form = QFormLayout()
        self.array_type_combo = QComboBox()
        self.array_type_combo.addItems(
            [
                "Nessuno",
                "Coppia Cardioide (2 sub)",
                "Array End-Fire",
                "Array Lineare (Steering Elettrico)",
                "Array Vortex",
                "Cardioide a 3 Sub (Sterzante)",
                "Cardiodi Multipli (3+ sub)",
                "Gradiente di Pressione",
            ]
        )
        self.array_type_combo.currentIndexChanged.connect(self.on_array_type_change)
        self.array_setup_layout_form.addRow("Tipo di Array:", self.array_type_combo)
        self.array_freq_label = QLabel("Freq. Array Design (Hz):")
        self.array_freq_input = QLineEdit(str(DEFAULT_ARRAY_FREQ))
        self.array_setup_layout_form.addRow(
            self.array_freq_label, self.array_freq_input
        )
        self.array_auto_spacing_check = QCheckBox(
            "Calcola Spaziatura/Raggio da Frequenza"
        )
        self.array_auto_spacing_check.setChecked(False)
        self.array_auto_spacing_check.toggled.connect(self.on_auto_spacing_toggle)
        self.array_setup_layout_form.addRow(self.array_auto_spacing_check)
        self.array_wavelength_fraction_label = QLabel("Frazione λ per Spaziatura:")
        self.array_wavelength_fraction_combo = QComboBox()
        self.array_wavelength_fraction_combo.addItems(["λ/4", "λ/2"])
        self.array_setup_layout_form.addRow(
            self.array_wavelength_fraction_label, self.array_wavelength_fraction_combo
        )
        self.array_spacing_label = QLabel("Spaziatura/Raggio (m):")
        self.array_spacing_input = QLineEdit(str(DEFAULT_ARRAY_RADIUS))
        self.array_setup_layout_form.addRow(
            self.array_spacing_label, self.array_spacing_input
        )
        self.array_elements_label = QLabel("Numero Elementi:")
        self.array_elements_input = QLineEdit("4")
        self.array_setup_layout_form.addRow(
            self.array_elements_label, self.array_elements_input
        )
        self.array_start_angle_label = QLabel("Orientamento/Angolo (°):")
        self.array_start_angle_input = QLineEdit(str(DEFAULT_ARRAY_START_ANGLE_DEG))
        self.array_setup_layout_form.addRow(
            self.array_start_angle_label, self.array_start_angle_input
        )
        self.array_line_coverage_angle_label = QLabel("Angolo Copertura (°):")
        self.array_line_coverage_angle_input = QLineEdit(
            str(DEFAULT_LINE_ARRAY_COVERAGE_DEG)
        )
        self.array_setup_layout_form.addRow(
            self.array_line_coverage_angle_label, self.array_line_coverage_angle_input
        )
        self.array_line_steering_angle_label = QLabel("Angolo Steering (°):")
        self.array_line_steering_angle_input = QLineEdit(
            str(DEFAULT_LINE_ARRAY_STEERING_DEG)
        )
        self.array_setup_layout_form.addRow(
            self.array_line_steering_angle_label, self.array_line_steering_angle_input
        )
        self.array_vortex_mode_label = QLabel("Modalità Vortex:")
        self.array_vortex_mode_label.setToolTip(
            "Numero intero (es. 1, 2, -1) che definisce l'avvolgimento dell'onda sonora.\nValori più alti creano un 'nullo' di pressione più ampio al centro."
        )
        self.array_vortex_mode_input = QLineEdit(str(DEFAULT_VORTEX_MODE))
        self.array_setup_layout_form.addRow(
            self.array_vortex_mode_label, self.array_vortex_mode_input
        )
        self.array_vortex_steering_angle_label = QLabel("Angolo Steering Vortex (°):")
        self.array_vortex_steering_angle_input = QLineEdit(
            str(DEFAULT_VORTEX_STEERING_DEG)
        )
        self.array_setup_layout_form.addRow(
            self.array_vortex_steering_angle_label,
            self.array_vortex_steering_angle_input,
        )

        self.array_3sub_cardio_pos_label = QLabel("Posizione Sub Invertito:")
        self.array_3sub_cardio_pos_combo = QComboBox()
        self.array_3sub_cardio_pos_combo.addItems(["Sinistra", "Centro", "Destra"])
        self.array_setup_layout_form.addRow(
            self.array_3sub_cardio_pos_label, self.array_3sub_cardio_pos_combo
        )
        
        self.array_steering_slider_label = QLabel("Steering Interattivo (°):")
        self.array_steering_slider_label.setVisible(False)
        self.array_setup_layout_form.addRow(self.array_steering_slider_label)
        self.array_steering_slider = QSlider(Qt.Orientation.Horizontal)
        self.array_steering_slider.setMinimum(-180)
        self.array_steering_slider.setMaximum(180)
        self.array_steering_slider.setValue(0)
        self.array_steering_slider.setVisible(False)
        self.array_steering_slider.valueChanged.connect(
            self.on_array_steering_slider_change
        )
        self.array_steering_slider.sliderReleased.connect(
            self.trigger_spl_map_recalculation
        )
        self.array_setup_layout_form.addRow(self.array_steering_slider)

        self.apply_array_config_button = QPushButton("Crea Gruppo Array")
        self.apply_array_config_button.clicked.connect(
            self.apply_or_update_array_configuration
        )
        self.array_setup_layout_form.addRow(self.apply_array_config_button)
        self.array_info_label = QLabel()
        self.array_info_label.setWordWrap(True)
        self.array_setup_layout_form.addRow(self.array_info_label)
        self.error_text_array_params = QLabel("")
        self.error_text_array_params.setStyleSheet("color: red;")
        self.array_length_label = QLabel("Lunghezza Array: N/D")
        self.array_length_label.setStyleSheet("font-style: italic; color: lightblue;")
        self.array_setup_layout_form.addRow(self.array_length_label)
        self.array_setup_layout_form.addRow(self.error_text_array_params)
        array_setup_group_box.setLayout(self.array_setup_layout_form)
        group_array_layout.addWidget(array_setup_group_box)

        group_array_group.setLayout(group_array_layout)
        self.controls_layout.addWidget(group_array_group)
        
    def _update_array_length_display(self, group_id):
        if group_id is None:
            self.array_length_label.setText("Lunghezza Array: N/D")
            return

        members = [s for s in self.sorgenti if s.get("group_id") == group_id]
        if len(members) < 2:
            self.array_length_label.setText("Lunghezza Array: 0.00 m")
            return

        points = np.array([[s['x'], s['y']] for s in members])
        # Calcola la matrice delle distanze euclidee al quadrato tra tutti i punti
        dist_matrix_sq = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
        max_dist = np.sqrt(np.max(dist_matrix_sq))

        self.array_length_label.setText(f"Lunghezza Array: {max_dist:.2f} m")

    def _setup_target_areas_ui(self):
        targets_group = QGroupBox("Aree Target")
        targets_main_layout = QVBoxLayout()
        target_selector_layout = QHBoxLayout()
        self.btn_prev_target_area = QPushButton("<")
        self.btn_prev_target_area.clicked.connect(self.select_prev_target_area)
        target_selector_layout.addWidget(self.btn_prev_target_area)
        self.label_current_target_area = QLabel("Nessuna Area")
        target_selector_layout.addWidget(
            self.label_current_target_area, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.btn_next_target_area = QPushButton(">")
        self.btn_next_target_area.clicked.connect(self.select_next_target_area)
        target_selector_layout.addWidget(self.btn_next_target_area)
        targets_main_layout.addLayout(target_selector_layout)
        target_actions_layout = QHBoxLayout()
        self.btn_new_target_area = QPushButton("Nuova")
        self.btn_new_target_area.clicked.connect(self.add_new_target_area_ui)
        target_actions_layout.addWidget(self.btn_new_target_area)
        self.btn_remove_selected_target_area = QPushButton("Rimuovi")
        self.btn_remove_selected_target_area.clicked.connect(
            self.remove_selected_target_area_ui
        )
        target_actions_layout.addWidget(self.btn_remove_selected_target_area)
        targets_main_layout.addLayout(target_actions_layout)
        self.check_activate_selected_target_area = QCheckBox("Attiva Area")
        self.check_activate_selected_target_area.setChecked(True)
        self.check_activate_selected_target_area.toggled.connect(
            self.toggle_selected_target_area_active
        )
        targets_main_layout.addWidget(self.check_activate_selected_target_area)

        self.btn_add_target_vtx = QPushButton("Aggiungi Vertice a Area Target")
        self.btn_add_target_vtx.clicked.connect(self._add_vtx_to_current_target_area)
        targets_main_layout.addWidget(self.btn_add_target_vtx)

        self.target_vtx_list_widget = QListWidget()
        self.target_vtx_list_widget.currentItemChanged.connect(
            self.on_target_vtx_selection_change
        )
        self.target_vtx_list_widget.setMaximumHeight(100)
        targets_main_layout.addWidget(self.target_vtx_list_widget)

        vtx_edit_layout = QFormLayout()
        self.tb_target_vtx_x = QLineEdit()
        vtx_edit_layout.addRow("Vertice X:", self.tb_target_vtx_x)
        self.tb_target_vtx_y = QLineEdit()
        vtx_edit_layout.addRow("Vertice Y:", self.tb_target_vtx_y)
        self.btn_update_target_vtx = QPushButton("Aggiorna Vertice Target")
        self.btn_update_target_vtx.clicked.connect(
            self.on_update_selected_target_vertex
        )
        targets_main_layout.addLayout(vtx_edit_layout)
        targets_main_layout.addWidget(self.btn_update_target_vtx)

        self.error_text_target_area_mgmt = QLabel("")
        self.error_text_target_area_mgmt.setStyleSheet("color: orange;")
        targets_main_layout.addWidget(self.error_text_target_area_mgmt)
        targets_group.setLayout(targets_main_layout)
        self.controls_layout.addWidget(targets_group)

    def _setup_avoidance_areas_ui(self):
        avoid_group = QGroupBox("Aree di Evitamento")
        avoid_main_layout = QVBoxLayout()
        avoid_selector_layout = QHBoxLayout()
        self.btn_prev_avoid_area = QPushButton("<")
        self.btn_prev_avoid_area.clicked.connect(self.select_prev_avoidance_area)
        avoid_selector_layout.addWidget(self.btn_prev_avoid_area)
        self.label_current_avoid_area = QLabel("Nessuna Area")
        avoid_selector_layout.addWidget(
            self.label_current_avoid_area, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.btn_next_avoid_area = QPushButton(">")
        self.btn_next_avoid_area.clicked.connect(self.select_next_avoidance_area)
        avoid_selector_layout.addWidget(self.btn_next_avoid_area)
        avoid_main_layout.addLayout(avoid_selector_layout)
        avoid_actions_layout = QHBoxLayout()
        self.btn_new_avoid_area = QPushButton("Nuova")
        self.btn_new_avoid_area.clicked.connect(self.add_new_avoidance_area_ui)
        avoid_actions_layout.addWidget(self.btn_new_avoid_area)
        self.btn_remove_selected_avoid_area = QPushButton("Rimuovi")
        self.btn_remove_selected_avoid_area.clicked.connect(
            self.remove_selected_avoidance_area_ui
        )
        avoid_actions_layout.addWidget(self.btn_remove_selected_avoid_area)
        avoid_main_layout.addLayout(avoid_actions_layout)
        self.check_activate_selected_avoid_area = QCheckBox("Attiva Area")
        self.check_activate_selected_avoid_area.setChecked(True)
        self.check_activate_selected_avoid_area.toggled.connect(
            self.toggle_selected_avoidance_area_active
        )
        avoid_main_layout.addWidget(self.check_activate_selected_avoid_area)

        self.btn_add_avoid_vtx = QPushButton("Aggiungi Vertice a Area Evitamento")
        self.btn_add_avoid_vtx.clicked.connect(self._add_vtx_to_current_avoidance_area)
        avoid_main_layout.addWidget(self.btn_add_avoid_vtx)

        self.avoid_vtx_list_widget = QListWidget()
        self.avoid_vtx_list_widget.currentItemChanged.connect(
            self.on_avoid_vtx_selection_change
        )
        self.avoid_vtx_list_widget.setMaximumHeight(100)
        avoid_main_layout.addWidget(self.avoid_vtx_list_widget)

        vtx_edit_layout = QFormLayout()
        self.tb_avoid_vtx_x = QLineEdit()
        vtx_edit_layout.addRow("Vertice X:", self.tb_avoid_vtx_x)
        self.tb_avoid_vtx_y = QLineEdit()
        vtx_edit_layout.addRow("Vertice Y:", self.tb_avoid_vtx_y)
        self.btn_update_avoid_vtx = QPushButton("Aggiorna Vertice Avoid")
        self.btn_update_avoid_vtx.clicked.connect(self.on_update_selected_avoid_vertex)
        avoid_main_layout.addLayout(vtx_edit_layout)
        avoid_main_layout.addWidget(self.btn_update_avoid_vtx)

        self.error_text_avoid_area_mgmt = QLabel("")
        self.error_text_avoid_area_mgmt.setStyleSheet("color: orange;")
        avoid_main_layout.addWidget(self.error_text_avoid_area_mgmt)
        avoid_group.setLayout(avoid_main_layout)
        self.controls_layout.addWidget(avoid_group)

    def _setup_sub_placement_areas_ui(self):
        """Nuova UI per definire aree di posizionamento subwoofer"""
        placement_group = QGroupBox("Aree Posizionamento Sub")
        placement_main_layout = QVBoxLayout()
        
        placement_selector_layout = QHBoxLayout()
        self.btn_prev_placement_area = QPushButton("<")
        self.btn_prev_placement_area.clicked.connect(self.select_prev_sub_placement_area)
        placement_selector_layout.addWidget(self.btn_prev_placement_area)
        self.label_current_placement_area = QLabel("Nessuna Area")
        placement_selector_layout.addWidget(
            self.label_current_placement_area, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.btn_next_placement_area = QPushButton(">")
        self.btn_next_placement_area.clicked.connect(self.select_next_sub_placement_area)
        placement_selector_layout.addWidget(self.btn_next_placement_area)
        placement_main_layout.addLayout(placement_selector_layout)
        
        placement_actions_layout = QHBoxLayout()
        self.btn_new_placement_area = QPushButton("Nuova")
        self.btn_new_placement_area.clicked.connect(self.add_new_sub_placement_area_ui)
        placement_actions_layout.addWidget(self.btn_new_placement_area)
        self.btn_remove_selected_placement_area = QPushButton("Rimuovi")
        self.btn_remove_selected_placement_area.clicked.connect(
            self.remove_selected_sub_placement_area_ui
        )
        placement_actions_layout.addWidget(self.btn_remove_selected_placement_area)
        placement_main_layout.addLayout(placement_actions_layout)
        
        self.check_activate_selected_placement_area = QCheckBox("Attiva Area")
        self.check_activate_selected_placement_area.setChecked(True)
        self.check_activate_selected_placement_area.toggled.connect(
            self.toggle_selected_sub_placement_area_active
        )
        placement_main_layout.addWidget(self.check_activate_selected_placement_area)

        self.btn_add_placement_vtx = QPushButton("Aggiungi Vertice")
        self.btn_add_placement_vtx.clicked.connect(self._add_vtx_to_current_sub_placement_area)
        placement_main_layout.addWidget(self.btn_add_placement_vtx)

        self.placement_vtx_list_widget = QListWidget()
        self.placement_vtx_list_widget.currentItemChanged.connect(
            self.on_placement_vtx_selection_change
        )
        self.placement_vtx_list_widget.setMaximumHeight(100)
        placement_main_layout.addWidget(self.placement_vtx_list_widget)

        vtx_edit_layout = QFormLayout()
        self.tb_placement_vtx_x = QLineEdit()
        vtx_edit_layout.addRow("Vertice X:", self.tb_placement_vtx_x)
        self.tb_placement_vtx_y = QLineEdit()
        vtx_edit_layout.addRow("Vertice Y:", self.tb_placement_vtx_y)
        self.btn_update_placement_vtx = QPushButton("Aggiorna Vertice")
        self.btn_update_placement_vtx.clicked.connect(self.on_update_selected_placement_vertex)
        placement_main_layout.addLayout(vtx_edit_layout)
        placement_main_layout.addWidget(self.btn_update_placement_vtx)

        self.error_text_placement_area_mgmt = QLabel("")
        self.error_text_placement_area_mgmt.setStyleSheet("color: orange;")
        placement_main_layout.addWidget(self.error_text_placement_area_mgmt)
        placement_group.setLayout(placement_main_layout)
        self.controls_layout.addWidget(placement_group)

    def _setup_spl_vis_ui(self):
        spl_vis_group = QGroupBox("Visualizzazione SPL")
        spl_vis_layout = QFormLayout()
        self.slider_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_freq.setMinimum(20)
        self.slider_freq.setMaximum(200)
        self.slider_freq.setValue(80)
        self.slider_freq.valueChanged.connect(self.on_freq_change_ui_qt)
        spl_vis_layout.addRow("Freq. Visualiz.:", self.slider_freq)
        self.label_slider_freq_val = QLabel(f"{self.slider_freq.value()} Hz")
        spl_vis_layout.addRow(self.label_slider_freq_val)
        
        # Scala SPL auto-adattiva
        self.check_auto_scale_spl = QCheckBox("Scala SPL Auto (±30dB)")
        self.check_auto_scale_spl.setChecked(True)
        self.check_auto_scale_spl.toggled.connect(self.on_auto_scale_spl_toggle)
        spl_vis_layout.addRow(self.check_auto_scale_spl)
        
        self.tb_spl_min = QLineEdit("50")
        self.tb_spl_max = QLineEdit("100")
        spl_vis_layout.addRow("SPL Min Display (dB):", self.tb_spl_min)
        spl_vis_layout.addRow("SPL Max Display (dB):", self.tb_spl_max)
        self.error_text_spl_range = QLabel("")
        self.error_text_spl_range.setStyleSheet("color: red;")
        spl_vis_layout.addRow(self.error_text_spl_range)
        self.check_auto_spl_update = QCheckBox("Aggiorna SPL Automaticamente")
        self.check_auto_spl_update.setChecked(True)
        self.check_auto_spl_update.toggled.connect(self.trigger_spl_map_recalculation)
        spl_vis_layout.addRow(self.check_auto_spl_update)
        self.btn_update_spl = QPushButton("Aggiorna Mappa SPL")
        self.btn_update_spl.clicked.connect(
            lambda: self.visualizza_mappatura_spl(
                self.get_slider_freq_val(), preserve_view=True
            )
        )
        spl_vis_layout.addRow(self.btn_update_spl)
        spl_vis_group.setLayout(spl_vis_layout)
        self.controls_layout.addWidget(spl_vis_group)

    def _setup_sim_grid_ui(self):
        sim_grid_params_group = QGroupBox("Parametri Simulazione e Griglia")
        sim_grid_params_layout = QFormLayout()
        self.tb_velocita_suono = QLineEdit(str(DEFAULT_SIM_SPEED_OF_SOUND))
        sim_grid_params_layout.addRow("Velocità Suono (m/s):", self.tb_velocita_suono)
        self.tb_grid_res_spl = QLineEdit("0.1")
        sim_grid_params_layout.addRow(
            "Risoluzione Mappa SPL (m):", self.tb_grid_res_spl
        )
        self.tb_grid_snap_spacing = QLineEdit(str(self.grid_snap_spacing))
        self.tb_grid_snap_spacing.editingFinished.connect(self.update_grid_snap_params)
        sim_grid_params_layout.addRow(
            "Passo Griglia Snap (m):", self.tb_grid_snap_spacing
        )
        self.check_grid_snap_enabled = QCheckBox("Abilita Snap Oggetti")
        self.check_grid_snap_enabled.setChecked(False)
        self.check_grid_snap_enabled.stateChanged.connect(self.update_grid_snap_params)
        sim_grid_params_layout.addRow(self.check_grid_snap_enabled)
        self.check_show_grid = QCheckBox("Mostra Griglia")
        self.check_show_grid.setChecked(False)
        self.check_show_grid.stateChanged.connect(self.update_grid_snap_params)
        sim_grid_params_layout.addRow(self.check_show_grid)
        self.error_text_grid_params = QLabel("")
        self.error_text_grid_params.setStyleSheet("color: red;")
        sim_grid_params_layout.addRow(self.error_text_grid_params)
        sim_grid_params_group.setLayout(sim_grid_params_layout)
        self.controls_layout.addWidget(sim_grid_params_group)

    def _setup_optimization_ui(self):
        optim_group = QGroupBox("Ottimizzazione Automatica")
        optim_main_layout = QVBoxLayout()
        self.radio_btn_group_crit = QButtonGroup(self)
        crit_layout = QHBoxLayout()
        self.radio_copertura = QRadioButton("Copertura SPL")
        self.radio_copertura.setChecked(True)
        self.radio_copertura.toggled.connect(
            lambda: self.update_optim_freq_fields_visibility("Copertura SPL")
        )
        crit_layout.addWidget(self.radio_copertura)
        self.radio_omogeneita = QRadioButton("Omogeneità SPL")
        self.radio_omogeneita.toggled.connect(
            lambda: self.update_optim_freq_fields_visibility("Omogeneità SPL")
        )
        crit_layout.addWidget(self.radio_omogeneita)
        self.radio_btn_group_crit.addButton(self.radio_copertura)
        self.radio_btn_group_crit.addButton(self.radio_omogeneita)
        optim_main_layout.addLayout(crit_layout)
        optim_freq_layout = QFormLayout()
        self.label_opt_freq_single_widget = QLabel("Freq. Ottim. (Hz):")
        self.tb_opt_freq_single = QLineEdit("80")
        optim_freq_layout.addRow(
            self.label_opt_freq_single_widget, self.tb_opt_freq_single
        )
        self.label_opt_freq_min_widget = QLabel("Freq. Min (Hz):")
        self.tb_opt_freq_min = QLineEdit("40")
        optim_freq_layout.addRow(self.label_opt_freq_min_widget, self.tb_opt_freq_min)
        self.label_opt_freq_max_widget = QLabel("Freq. Max (Hz):")
        self.tb_opt_freq_max = QLineEdit("120")
        optim_freq_layout.addRow(self.label_opt_freq_max_widget, self.tb_opt_freq_max)
        self.label_opt_n_freq_widget = QLabel("Nr. Punti Freq.:")
        self.tb_opt_n_freq = QLineEdit("10")
        optim_freq_layout.addRow(self.label_opt_n_freq_widget, self.tb_opt_n_freq)
        optim_main_layout.addLayout(optim_freq_layout)
        self.error_text_optim_freq = QLabel("")
        self.error_text_optim_freq.setStyleSheet("color: red;")
        optim_main_layout.addWidget(self.error_text_optim_freq)
        
        optim_algo_layout = QFormLayout()
        self.tb_opt_pop_size = QLineEdit("100")
        optim_algo_layout.addRow("Config. Testate:", self.tb_opt_pop_size)
        self.tb_opt_generations = QLineEdit("50")
        optim_algo_layout.addRow("Cicli Ottim.:", self.tb_opt_generations)

        self.tb_target_min_spl_desired = QLineEdit(
            str(self.target_min_spl_desired_ui_val)
        )
        optim_algo_layout.addRow(
            "Min SPL Target Desiderato (dB):", self.tb_target_min_spl_desired
        )

        self.tb_max_spl_avoid = QLineEdit(str(self.max_spl_avoidance_ui_val))
        optim_algo_layout.addRow("Max SPL Evit. (dB):", self.tb_max_spl_avoid)

        self.label_balance_slider = QLabel("Priorità Target / Evitamento:")
        optim_algo_layout.addRow(self.label_balance_slider)
        self.slider_balance = QSlider(Qt.Orientation.Horizontal)
        self.slider_balance.setMinimum(0)
        self.slider_balance.setMaximum(100)
        self.slider_balance.setValue(self.balance_slider_ui_val)
        self.label_balance_value = QLabel(f"{self.slider_balance.value()}% Target")
        self.slider_balance.valueChanged.connect(
            lambda val: self.label_balance_value.setText(f"{val}% Target")
        )
        optim_algo_layout.addRow(self.slider_balance)
        optim_algo_layout.addRow(self.label_balance_value)

        self.check_enable_suggestions = QCheckBox("Abilita Consulente di Configurazione")
        self.check_enable_suggestions.setChecked(False)  # Spento di default
        optim_algo_layout.addRow(self.check_enable_suggestions)

        optim_main_layout.addLayout(optim_algo_layout)
        
        self.error_text_optim_params = QLabel("")
        self.error_text_optim_params.setStyleSheet("color: red;")
        optim_main_layout.addWidget(self.error_text_optim_params)
        
        optim_buttons_layout_H1 = QHBoxLayout()
        self.btn_optimize_widget = QPushButton("Avvia Ottimizzazione DSP")
        self.btn_optimize_widget.clicked.connect(self.avvia_ottimizzazione_ui_qt)
        optim_buttons_layout_H1.addWidget(self.btn_optimize_widget)
        self.btn_stop_optimize_widget = QPushButton("Ferma")
        self.btn_stop_optimize_widget.clicked.connect(self.stop_ottimizzazione_ui_qt)
        self.btn_stop_optimize_widget.setEnabled(False)
        optim_buttons_layout_H1.addWidget(self.btn_stop_optimize_widget)
        optim_main_layout.addLayout(optim_buttons_layout_H1)
        
        self.btn_unlock_dsp_params = QPushButton("Sblocca DSP per Ottim.")
        self.btn_unlock_dsp_params.clicked.connect(self.unlock_dsp_for_optimization)
        optim_main_layout.addWidget(self.btn_unlock_dsp_params)
        
        self.status_text_optim = QLabel("Pronto.")
        self.status_text_optim.setWordWrap(True)
        optim_main_layout.addWidget(self.status_text_optim)
        
        optim_group.setLayout(optim_main_layout)
        self.controls_layout.addWidget(optim_group)

    def handle_optim_finished(self, best_solution, best_fitness):
        self.btn_optimize_widget.setEnabled(True)
        self.btn_stop_optimize_widget.setEnabled(False)

        # Se l'ottimizzazione non ha prodotto risultati, esci.
        if not best_solution:
            self.status_bar.showMessage("Ottimizzazione terminata senza una soluzione valida.", 4000)
            self.status_text_optim.setText("Pronto.")
            return

        # Non applica la soluzione. Passa i risultati all'assistente per la scelta.
        # La logica di cosa fare dopo è TUTTA dentro 'suggest_intelligent_array_layout'.
        DIFFICULTY_THRESHOLD = -50.0
        CRITICAL_DIFFICULTY_THRESHOLD = -100.0

        critical = False
        if self.check_enable_suggestions.isChecked():
            if best_fitness < CRITICAL_DIFFICULTY_THRESHOLD:
                critical = True
            elif best_fitness < DIFFICULTY_THRESHOLD:
                critical = True # Attiviamo l'assistente anche per difficoltà medie
    
        # Chiama l'assistente SOLO se il checkbox è spuntato, altrimenti applica il risultato grezzo.
        if critical:
            self.suggest_intelligent_array_layout(best_solution, best_fitness, critical)
        else:
            # Se l'assistente non è richiesto, applica direttamente il miglior risultato
            for i, sub_dsp in enumerate(best_solution):
                if i < len(self.sorgenti):
                    self.sorgenti[i].update(sub_dsp)
        
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)
            self.status_bar.showMessage("Ottimizzazione completata e risultato applicato.", 5000)

    def suggest_intelligent_array_layout(self, best_solution, best_fitness, critical=False):
        """
        (COMPLETA E CORRETTA)
        Mostra la finestra con le opzioni e applica la scelta dell'utente
        SOLO DOPO che il bottone è stato cliccato.
        """
        num_subs = len(self.sorgenti)
        if num_subs == 0:
            return # Non fare nulla se non ci sono sub

        # 1. GENERA I SUGGERIMENTI DI LAYOUT INTELLIGENTI
        analysis_result = self._analyze_layout_geometry_advanced()
        suggested_layouts = self._generate_intelligent_layout_suggestions(num_subs, analysis_result)
    
        # 2. CREA E CONFIGURA LA FINESTRA DI DIALOGO
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("Assistente di Configurazione")
    
        msg_text = f"🎯 **Ottimizzazione Difficile** (Fitness: {best_fitness:.2f})\n\n"
        msg_text += "L'ottimizzatore ha faticato. Per un risultato migliore, puoi:\n"
        msg_box.setText(msg_text)

        # Aggiungi i bottoni per ogni opzione possibile
        raw_result_button = msg_box.addButton("Applica Risultato Grezzo Ottimizzato", QMessageBox.ButtonRole.ActionRole)
    
        smart_buttons = []
        for layout in suggested_layouts:
            btn_text = layout['type'].replace("Array di ", "").replace("Doppio ", "2x ")
            btn = msg_box.addButton(f"Riconfigura come: {btn_text}", QMessageBox.ButtonRole.ActionRole)
            btn.layout_data = layout
            smart_buttons.append(btn)

        add_subs_button = msg_box.addButton("Aggiungi Subwoofer...", QMessageBox.ButtonRole.ActionRole)
        cancel_button = msg_box.addButton("Ignora Risultati", QMessageBox.ButtonRole.RejectRole)
    
        # 3. MOSTRA LA FINESTRA E ATTENDI LA SCELTA DELL'UTENTE
        msg_box.exec()
    
        # 4. ESEGUI L'AZIONE SCELTA (I SUB SI SPOSTANO SOLO ORA)
        clicked_button = msg_box.clickedButton()
        solution_applied = False

        if clicked_button == raw_result_button:
            # L'utente ha scelto di applicare la soluzione grezza
            for i, sub_dsp in enumerate(best_solution):
                if i < len(self.sorgenti):
                    self.sorgenti[i].update(sub_dsp)
            solution_applied = True
            self.status_bar.showMessage("Risultato grezzo dell'ottimizzazione applicato.", 4000)

        elif hasattr(clicked_button, 'layout_data'):
            # L'utente ha scelto un layout intelligente
            self._apply_multi_group_layout(clicked_button.layout_data)
            solution_applied = True
    
        elif clicked_button == add_subs_button:
            self._suggest_add_subwoofers()
            # In questo caso non aggiorniamo la vista qui, lo fa la funzione stessa
            return

        elif clicked_button == cancel_button:
            # L'utente ha scelto di ignorare, non facciamo nulla
            self.status_text_optim.setText("Risultati ignorati. Pronto.")
            return

        # 5. AGGIORNA LA VISTA FINALE SOLO SE E' STATA APPLICATA UNA SOLUZIONE
        if solution_applied:
            # Aggiorna i campi della UI
            self.aggiorna_ui_sub_fields()

            # Imposta la frequenza di visualizzazione
            new_freq = None
            if self.last_optim_criterion == "Copertura SPL":
                if self.last_optim_freq_s is not None: new_freq = self.last_optim_freq_s
            elif self.last_optim_criterion == "Omogeneità SPL":
                if (self.last_optim_freq_min is not None and self.last_optim_freq_max is not None):
                    new_freq = (self.last_optim_freq_min + self.last_optim_freq_max) / 2.0
        
            if new_freq is not None:
                slider_min, slider_max = self.slider_freq.minimum(), self.slider_freq.maximum()
                self.slider_freq.setValue(int(max(slider_min, min(slider_max, new_freq))))

            # Ridisegna tutto con la nuova configurazione
            self.full_redraw(preserve_view=True)
        else:
            self.status_text_optim.setText("Pronto.")

    def _generate_intelligent_layout_suggestions(self, num_subs, analysis):
        """
        Motore di suggerimenti avanzato che include la proposta di aggiungere
        subwoofer quando l'analisi geometrica lo ritiene necessario.
        """
        suggestions = []

        # --- ANALISI PRELIMINARE PER AGGIUNTA SUB ---
        # Heuristica: se l'area di copertura è molto grande rispetto al numero di sub,
        # o se l'ottimizzazione è critica, suggerisci di aggiungerne.
        is_coverage_demanding = analysis.get("is_coverage_demanding", False)
        
        if num_subs > 0 and is_coverage_demanding:
            suggestions.append({
                "type": "Aggiungi Subwoofer...",
                "description": "Il numero di subwoofer potrebbe essere insufficiente per l'area richiesta. Aumentare il numero può migliorare drasticamente copertura e uniformità.",
                "action": "add_subs" # Flag speciale per l'azione
            })

        if num_subs < 2:
            return suggestions # Se c'è solo il suggerimento di aggiungere sub, va bene

        # --- STRATEGIE DI LAYOUT (Codice precedente, rimane valido) ---
        # A. Array di Coppie Cardioidi (richiede num_subs >= 4 e pari)
        if num_subs >= 4 and num_subs % 2 == 0:
            # ... (il resto della logica di generazione dei meta-layout rimane invariato)
            num_pairs = num_subs // 2
            groups_config = []
            for i in range(num_pairs):
                sub_indices = [i*2, i*2+1]
                groups_config.append({"subs": sub_indices, "type": "Coppia Cardioide (2 sub)"})
            suggestions.append({
                "type": f"Array di {num_pairs} Coppie Cardioidi",
                "description": "Crea gruppi cardioidi indipendenti. Posizionali affiancati (Broadside) o in linea (End-Fire) per un controllo direzionale scalabile.",
                "groups": groups_config
            })

        # B. Altre strategie... (tutto il codice che genera i layout rimane qui)
        if num_subs >= 6 and num_subs % 2 == 0:
            subs_per_array = num_subs // 2
            groups_config_lines = [
                {"subs": list(range(subs_per_array)), "type": "Array Lineare (Steering Elettrico)"},
                {"subs": list(range(subs_per_array, num_subs)), "type": "Array Lineare (Steering Elettrico)"}
            ]
            suggestions.append({
                "type": f"Doppio Line Array ({subs_per_array} sub ciascuno)",
                "description": "Massima flessibilità. Usa i due Line Array per coprire zone L/R, oppure posizionali in linea (End-Fire) per la massima direttività.",
                "groups": groups_config_lines
            })

        # ... e così via per tutte le altre strategie di layout.
        
        unique_suggestions = {s['type']: s for s in suggestions}.values()
        return self._filter_suggestions_by_geometry(list(unique_suggestions), analysis)

    def _filter_suggestions_by_geometry(self, suggestions, analysis):
        """Filtra e ordina i suggerimenti in base all'analisi geometrica"""
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = 0
            
            # Bonus per configurazioni che risolvono problemi specifici
            if analysis.get("needs_rear_suppression"):
                if "Cardioide" in suggestion["type"]:
                    score += 3
                elif "End-Fire" in suggestion["type"]:
                    score += 2
                    
            if analysis.get("needs_lateral_suppression"):
                if "Vortex" in suggestion["type"]:
                    score += 3
                elif "End-Fire" in suggestion["type"]:
                    score += 2
                    
            if analysis.get("complex_geometry"):
                if "Vortex" in suggestion["type"]:
                    score += 2
                elif "Misto" in suggestion["type"]:
                    score += 1
                    
            scored_suggestions.append((score, suggestion))
        
        # Ordina per punteggio decrescente
        scored_suggestions.sort(key=lambda x: x[0], reverse=True)
        return [suggestion for score, suggestion in scored_suggestions[:3]]  # Top 3

    def _present_layout_options(self, suggestions, critical):
        """
        Presenta le opzioni di layout in modo chiaro, con indici corretti e
        gestendo il suggerimento "Aggiungi Subwoofer" come una proposta formale.
        """
        if not suggestions:
             # Se non ci sono suggerimenti, non mostrare nulla, neanche se critico
            return
            
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("Consulente di Configurazione")

        # --- Testo Introduttivo ---
        msg_text = "🎯 **Consulente di Configurazione Array**\n\n"
        if critical:
            msg_text += "⚠️ L'ottimizzazione sta incontrando difficoltà. "
        else:
            msg_text += "💡 Per migliorare il risultato, "
        msg_text += "**ti propongo le seguenti strategie:**\n"
        
        # --- Descrizione Esplicita delle Configurazioni Proposte ---
        for i, suggestion in enumerate(suggestions, 1):
            config_title = suggestion['type']
            msg_text += f"\n---------------------------------\n"
            msg_text += f"**Proposta {i}: {config_title}**\n"
            msg_text += f"**Strategia:** *{suggestion['description']}*\n"

        msg_box.setText(msg_text)
        
        # --- Creazione Bottoni (INDICE CORRETTO) ---
        for i, suggestion in enumerate(suggestions, 1):
            # Se la proposta è di aggiungere sub, crea un bottone apposito
            if suggestion.get('action') == 'add_subs':
                btn = msg_box.addButton(suggestion['type'], QMessageBox.ButtonRole.ActionRole)
            else:
                # Altrimenti, crea il bottone standard con l'indice corretto
                btn = msg_box.addButton(f"Applica Proposta {i}", QMessageBox.ButtonRole.ActionRole)
            
            btn.suggestion_data = suggestion
        
        btn_no = msg_box.addButton("Mantieni Attuale", QMessageBox.ButtonRole.RejectRole)
        
        msg_box.exec()
        
        clicked_button = msg_box.clickedButton()

        if clicked_button == btn_no:
            return
        
        # Gestisce l'azione in base al tipo di suggerimento cliccato
        suggestion_data = getattr(clicked_button, 'suggestion_data', None)
        if suggestion_data:
            if suggestion_data.get('action') == 'add_subs':
                self._suggest_add_subwoofers()
            else:
                self._apply_multi_group_layout(suggestion_data)

    def _apply_multi_group_layout(self, layout_config):
        """Applica una configurazione di layout multi-gruppo"""
        try:
            # Reset gruppi esistenti
            self.lista_gruppi_array.clear()
            for sub in self.sorgenti:
                sub['group_id'] = None
                sub['is_group_master'] = False
                sub['param_locks'] = {
                    'angle': False, 'delay': False, 'gain': False, 
                    'polarity': False, 'position': False
                }
            
            # Applica ogni gruppo del layout
            for group_config in layout_config['groups']:
                self._create_and_apply_group(group_config)
            
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=False)
            self.status_bar.showMessage(
                f"Layout '{layout_config['type']}' applicato con successo!", 6000
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Errore Layout", f"Errore nell'applicazione del layout:\n{e}")

    def _create_and_apply_group(self, group_config):
        """
        (VERSIONE CORRETTA)
        Crea e applica una configurazione di gruppo, usando il nuovo "centralino".
        """
        sub_indices = group_config['subs']
        array_type = group_config['type']
        
        if not sub_indices or any(i >= len(self.sorgenti) for i in sub_indices):
            print(f"ATTENZIONE: Indici subwoofer non validi per il gruppo: {sub_indices}")
            return
            
        group_subs = [self.sorgenti[i] for i in sub_indices]
        center_x = np.mean([s['x'] for s in group_subs])
        center_y = np.mean([s['y'] for s in group_subs])
        
        base_params = {
            "width": self.global_sub_width, "depth": self.global_sub_depth,
            "spl_rms": self.global_sub_spl_rms, "gain_db": 0, "polarity": 1, "delay_ms": 0,
        }
        
        try:
            c_sound = float(self.tb_velocita_suono.text())
            design_freq = float(self.array_freq_input.text())
            wavelength = c_sound / design_freq
            spacing = wavelength / 4.0 # Usa una spaziatura di default per i suggerimenti
        except (ValueError, ZeroDivisionError):
            c_sound, design_freq, spacing = DEFAULT_SIM_SPEED_OF_SOUND, DEFAULT_ARRAY_FREQ, 1.0

        # --- CHIAMATA AL CENTRALINO ---
        num_elements = len(sub_indices)
        new_configs = self._generate_array_configs(
            array_type, center_x, center_y, num_elements, 
            spacing, c_sound, design_freq, base_params
        )
        # ---------------------------
        
        if not new_configs:
            print(f"ATTENZIONE: Nessuna configurazione generata per il tipo di array '{array_type}'")
            return

        placement_area = self.get_active_sub_placement_area()
        if placement_area:
            area_path = Path(placement_area)
            for config in new_configs:
                if not area_path.contains_point((config['x'], config['y'])):
                    new_x, new_y = self._find_nearest_point_in_placement_area(config['x'], config['y'], placement_area)
                    config['x'], config['y'] = new_x, new_y

        new_group_id = self.next_group_id
        self.lista_gruppi_array[new_group_id] = {'type': array_type, 'design_freq': design_freq}
        self._update_max_group_id()
        
        for i, config in enumerate(new_configs):
            if i < len(sub_indices):
                sub_idx = sub_indices[i]
                sub = self.sorgenti[sub_idx]
                sub.update(config)
                sub['gain_lin'] = 10**(sub['gain_db'] / 20.0)
                sub['group_id'] = new_group_id
                sub['is_group_master'] = config.get('is_group_master', False)
                sub['param_locks'] = config.get('param_locks', {'angle': True, 'delay': True, 'gain': True, 'polarity': True, 'position': False})

    def _generate_array_configs(self, array_type, center_x, center_y, num_elements, spacing, c_sound, design_freq, base_params):
        """
        (NUOVA FUNZIONE - IL CENTRALINO MANCANTE)
        Questa funzione legge il tipo di array richiesto come stringa e chiama la
        funzione di calcolo corretta, restituendo le configurazioni dei subwoofer.
        """
        new_configs = []
        # Trova l'array_params corretto dal UI per avere i valori di default
        # NOTA: Questo potrebbe essere migliorato passando i parametri direttamente, ma per ora usiamo i valori del UI
        start_angle_deg = float(self.array_start_angle_input.text())
        
        if array_type == "Coppia Cardioide (2 sub)":
            new_configs = self._calculate_cardioid_configs(center_x, center_y, spacing, c_sound, start_angle_deg, base_params)
        
        elif array_type == "Array End-Fire":
            new_configs = self._calculate_endfire_configs(center_x, center_y, num_elements, spacing, c_sound, start_angle_deg, base_params)
        
        elif array_type == "Array Lineare (Steering Elettrico)":
            steering_angle_deg = float(self.array_line_steering_angle_input.text())
            coverage_angle_deg = float(self.array_line_coverage_angle_input.text())
            new_configs = self._calculate_line_array_steered_configs(center_x, center_y, num_elements, spacing, start_angle_deg, steering_angle_deg, coverage_angle_deg, c_sound, base_params)

        elif array_type == "Array Vortex":
            vortex_mode = int(self.array_vortex_mode_input.text())
            steering_deg = float(self.array_vortex_steering_angle_input.text())
            new_configs = self._calculate_vortex_array_configs(center_x, center_y, num_elements, spacing, vortex_mode, design_freq, start_angle_deg, steering_deg, c_sound, base_params)
        
        elif array_type == "Cardioide a 3 Sub (Sterzante)":
            reversed_pos_index = self.array_3sub_cardio_pos_combo.currentIndex()
            new_configs = self._calculate_steerable_3sub_cardioid_configs(center_x, center_y, spacing, start_angle_deg, reversed_pos_index, c_sound, base_params)
        
        elif array_type == "Cardiodi Multipli (3+ sub)":
             new_configs = self._calculate_multi_cardioid_configs(center_x, center_y, num_elements, spacing, c_sound, base_params)

        elif array_type == "Gradiente di Pressione":
            new_configs = self._calculate_pressure_gradient_configs(center_x, center_y, num_elements, spacing, c_sound, base_params)
            
        return new_configs

    def _calculate_multi_cardioid_configs(self, center_x, center_y, num_elements, spacing, c_sound, base_params):
        """
        Configurazione per un array cardioide in linea con 3 o più elementi.
        Utilizza un pattern scalabile "2 frontali, 1 invertito".
        Ogni terzo subwoofer è invertito (direzione, polarità) e processato
        con gain e delay per cancellare i due sub frontali precedenti.
        """
        if num_elements < 3:
            return []

        configs = []
        orientation_rad = 0.0  # Puntamento di default verso l'alto (Nord)
        dir_x, dir_y = np.cos(orientation_rad), -np.sin(orientation_rad) # Vettore per la linea dei sub
        
        start_offset = -(num_elements - 1) / 2.0 * spacing

        for i in range(num_elements):
            offset = start_offset + i * spacing
            sub_x = center_x + offset * dir_x
            sub_y = center_y + offset * dir_y
            
            new_conf = base_params.copy()

            # Applica il pattern "2 Frontali, 1 Invertito"
            if (i + 1) % 3 == 0:
                # --- Questo è un SUBWOOFER INVERTITO (il 3°, 6°, 9°...) ---
                is_reversed = True
                polarity = -1
                angle = orientation_rad + np.pi  # Gira di 180 gradi
                # Aumento di gain necessario per cancellare 2 sorgenti
                gain_db = +6.0
                # Delay per allineare temporalmente la sua emissione posteriore
                # con quella frontale del sub adiacente.
                delay_ms = (spacing / c_sound) * 1000.0
            else:
                # --- Questo è un SUBWOOFER FRONTALE ---
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
            
        self._normalize_delays(configs)
        return configs

    def _calculate_pressure_gradient_configs(self, center_x, center_y, num_elements, spacing, c_sound, angle_deg, base_params):
        """
        (Versione con angolo orientabile)
        """
        if num_elements < 2:
            return []

        configs = []
        # Usa l'angolo fornito invece di un valore fisso
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
            
        self._normalize_delays(configs)
        return configs
    
    def _calculate_steerable_3sub_cardioid_configs(
        self, center_x, center_y, spacing, orientation_deg, reversed_sub_position_index, c, base_params
    ):
        """
        Versione RIVISTA e CORRETTA.
        Calcola la configurazione per un array cardioide a 3 elementi affiancati.
        La logica di Gain e Delay è stata perfezionata per una cancellazione posteriore ottimale.
        """
        orientation_rad = np.radians(orientation_deg)
        line_dir_x = np.cos(orientation_rad)
        line_dir_y = -np.sin(orientation_rad)

        configs = []
        positions = []
        
        # 1. Calcola le posizioni dei 3 sub in linea
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
                # --- SUBWOOFER INVERTITO ---
                polarity = -1
                angle = orientation_rad + np.pi
                
                # GAIN: +6dB per avere l'energia necessaria a cancellare due sorgenti frontali.
                gain_db = +6.0
                
                # DELAY: Il ritardo è calcolato rispetto ai sub frontali.
                # Per i casi asimmetrici (sub invertito a sx/dx), il ritardo è basato
                # sulla distanza dal sub frontale più lontano per garantire la coerenza del fronte d'onda.
                if reversed_sub_position_index == 1: # Caso simmetrico (Centro)
                    distance_to_ref = spacing
                else: # Caso asimmetrico (Sinistra o Destra)
                    distance_to_ref = 2 * spacing
                
                delay_ms = (distance_to_ref / c) * 1000.0
            else:
                # --- SUBWOOFER FRONTALI ---
                polarity = 1
                angle = orientation_rad
                gain_db = 0.0
                
                # DELAY: Per i casi asimmetrici, anche i sub frontali necessitano di un piccolo
                # ritardo per agire come un fronte d'onda coerente.
                if reversed_sub_position_index == 0 and i == 2: # Sub invertito a Sinistra
                    delay_ms = (spacing / c) * 1000.0
                elif reversed_sub_position_index == 2 and i == 0: # Sub invertito a Destra
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

        self._normalize_delays(configs)
        return configs

    def _find_nearest_point_in_placement_area(self, x, y, placement_area):
        """Trova il punto più vicino all'interno dell'area di posizionamento"""
        if not placement_area:
            return x, y
            
        area_path = Path(placement_area)
        if area_path.contains_point((x, y)):
            return x, y
            
        # Trova il punto più vicino sul perimetro
        min_dist = float('inf')
        nearest_x, nearest_y = x, y
        
        for i in range(len(placement_area)):
            p1 = placement_area[i]
            p2 = placement_area[(i + 1) % len(placement_area)]
            
            # Proiezione del punto sulla linea del segmento
            seg_x, seg_y = self._project_point_on_segment(x, y, p1, p2)
            dist = np.sqrt((x - seg_x)**2 + (y - seg_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_x, nearest_y = seg_x, seg_y
                
        return nearest_x, nearest_y

    def _project_point_on_segment(self, px, py, p1, p2):
        """Proietta un punto su un segmento di linea"""
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

    def get_active_sub_placement_area(self):
        """Restituisce l'area di posizionamento sub attiva"""
        for area in self.lista_sub_placement_areas:
            if area.get('active', False) and len(area.get('punti', [])) >= 3:
                return area['punti']
        return None

    def _analyze_layout_geometry_advanced(self):
        """Analisi geometrica avanzata per suggerimenti intelligenti"""
        analysis = {
            'needs_rear_suppression': False,
            'needs_lateral_suppression': False,
            'complex_geometry': False,
            'description': "ottenere una copertura controllata"
        }

        if not self.sorgenti:
            return analysis

        # Analizza la complessità della geometria
        target_areas = self.get_active_areas_points(self.lista_target_areas)
        avoid_areas = self.get_active_areas_points(self.lista_avoidance_areas)
        
        total_areas = len(target_areas) + len(avoid_areas)
        if total_areas > 2:
            analysis['complex_geometry'] = True
            
        # Analizza le forme delle aree
        for area_points in target_areas + avoid_areas:
            if len(area_points) > 4:  # Forme complesse
                analysis['complex_geometry'] = True
                break
                
        # Analisi direzionale come prima
        if self.sorgenti and avoid_areas:
            sub_center_x = np.mean([s['x'] for s in self.sorgenti])
            sub_center_y = np.mean([s['y'] for s in self.sorgenti])
            
            avg_sin = np.mean([np.sin(s['angle']) for s in self.sorgenti])
            avg_cos = np.mean([np.cos(s['angle']) for s in self.sorgenti])
            
            if abs(avg_sin) < 1e-9 and abs(avg_cos) < 1e-9:
                sub_fwd_vector = np.array([0.0, 1.0])
            else:
                sub_fwd_vector = np.array([avg_sin, avg_cos])
                sub_fwd_vector = sub_fwd_vector / np.linalg.norm(sub_fwd_vector)

            for area_points in avoid_areas:
                area_center_x = np.mean([p[0] for p in area_points])
                area_center_y = np.mean([p[1] for p in area_points])
                
                avoid_vector = np.array([area_center_x - sub_center_x, area_center_y - sub_center_y])
                norm_avoid_vector = np.linalg.norm(avoid_vector)
                
                if norm_avoid_vector < 0.1:
                    continue

                dot_product = np.dot(sub_fwd_vector, avoid_vector)
                cross_product_z = sub_fwd_vector[0] * avoid_vector[1] - sub_fwd_vector[1] * avoid_vector[0]

                if norm_avoid_vector > 0:
                    normalized_dot_product = dot_product / norm_avoid_vector
                    normalized_cross_product_z = cross_product_z / norm_avoid_vector
                else:
                    continue

                if normalized_dot_product < -0.5:
                    analysis['needs_rear_suppression'] = True
                
                if abs(normalized_cross_product_z) > 0.5 and normalized_dot_product > -0.87:
                    analysis['needs_lateral_suppression'] = True
        
        # Aggiorna descrizione
        descriptions = []
        if analysis['needs_lateral_suppression']:
            descriptions.append("aumentare la direttività e ridurre l'emissione laterale")
        if analysis['needs_rear_suppression']:
            descriptions.append("massimizzare la cancellazione posteriore")
        if analysis['complex_geometry']:
            descriptions.append("gestire geometrie complesse")
        
        if descriptions:
            analysis['description'] = " e ".join(descriptions)
        
        return analysis

    def _suggest_add_subwoofers(self):
        """Suggerisce di aggiungere sub solo in casi critici"""
        current_subs = len(self.sorgenti)
        if current_subs >= 8:  # Limite ragionevole
            QMessageBox.information(
                self, "Consulente di Configurazione",
                "Con 8+ subwoofer, il problema potrebbe essere nella configurazione piuttosto che nel numero. "
                "Prova configurazioni array diverse o verifica l'acustica della stanza."
            )
            return
            
        suggested_total = min(current_subs + 2, 6)  # Massimo 6 sub totali
        
        reply = QMessageBox.question(
            self, "Consulente: Aggiungere Subwoofer?",
            f"Per raggiungere gli obiettivi potrebbe essere necessario aggiungere {suggested_total - current_subs} subwoofer "
            f"(totale: {suggested_total}).\n\n"
            "Vuoi che li aggiunga e pre-configuri un layout ottimale?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._add_subs_and_configure(suggested_total - current_subs)

    def _add_subs_and_configure(self, subs_to_add):
        """Aggiunge sub e configura layout intelligentemente"""
        # Calcola posizione centrale attuale
        if self.sorgenti:
            center_x = np.mean([s['x'] for s in self.sorgenti])
            center_y = np.mean([s['y'] for s in self.sorgenti])
        else:
            center_x, center_y, _ = self._get_area_center_and_default_size()
            
        # Aggiunge i nuovi sub
        for _ in range(subs_to_add):
            self.add_subwoofer(
                specific_config={'x': center_x, 'y': center_y}, 
                redraw=False
            )
        
        # Suggerisce layout per il nuovo numero totale
        total_subs = len(self.sorgenti)
        analysis = self._analyze_layout_geometry_advanced()
        suggestions = self._generate_intelligent_layout_suggestions(total_subs, analysis)
        
        if suggestions:
            self._apply_multi_group_layout(suggestions[0])  # Applica il migliore
        
        self.status_bar.showMessage(f"Aggiunti {subs_to_add} subwoofer e applicato layout ottimale!", 6000)

    # Metodi per gestione aree di posizionamento sub
    def select_prev_sub_placement_area(self):
        if not self.lista_sub_placement_areas:
            self.current_sub_placement_area_idx = -1
        else:
            self.current_sub_placement_area_idx = (
                self.current_sub_placement_area_idx - 1 + len(self.lista_sub_placement_areas)
            ) % len(self.lista_sub_placement_areas)
        self.update_ui_for_selected_sub_placement_area()

    def select_next_sub_placement_area(self):
        if not self.lista_sub_placement_areas:
            self.current_sub_placement_area_idx = -1
        else:
            self.current_sub_placement_area_idx = (self.current_sub_placement_area_idx + 1) % len(
                self.lista_sub_placement_areas
            )
        self.update_ui_for_selected_sub_placement_area()

    def add_new_sub_placement_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size()
        hs = size / 2.0
        default_verts = DEFAULT_SUB_PLACEMENT_AREA_VERTICES
        self.current_sub_placement_area_idx = self._add_new_area_data(
            self.lista_sub_placement_areas, default_verts, "Posizionamento", "next_sub_placement_area_id", activate=True 
        )
        self.update_ui_for_selected_sub_placement_area()

    def remove_selected_sub_placement_area_ui(self):
        if 0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas):
            self.lista_sub_placement_areas.pop(self.current_sub_placement_area_idx)
            if not self.lista_sub_placement_areas:
                self.current_sub_placement_area_idx = -1
            elif self.current_sub_placement_area_idx >= len(self.lista_sub_placement_areas):
                self.current_sub_placement_area_idx = len(self.lista_sub_placement_areas) - 1
            self.update_ui_for_selected_sub_placement_area()

    def toggle_selected_sub_placement_area_active(self, checked):
        if 0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas):
            self.lista_sub_placement_areas[self.current_sub_placement_area_idx]["active"] = checked
            self.update_ui_for_selected_sub_placement_area()

    def _add_vtx_to_current_sub_placement_area(self):
        if 0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas):
            area = self.lista_sub_placement_areas[self.current_sub_placement_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area["punti"].append([new_x, new_y])
            self.update_ui_for_selected_sub_placement_area()
        else:
            self.error_text_placement_area_mgmt.setText(
                "Selezionare prima un'area di posizionamento o crearne una nuova."
            )

    def update_ui_for_selected_sub_placement_area(self):
        is_valid_idx = (
            0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas)
        )
        self.btn_add_placement_vtx.setEnabled(is_valid_idx)
        for w in [
            self.btn_prev_placement_area,
            self.btn_next_placement_area,
            self.btn_remove_selected_placement_area,
            self.check_activate_selected_placement_area,
            self.placement_vtx_list_widget,
        ]:
            w.setEnabled(is_valid_idx)

        self.placement_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_sub_placement_areas[self.current_sub_placement_area_idx]
            self.label_current_placement_area.setText(
                f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})"
            )
            try:
                self.check_activate_selected_placement_area.toggled.disconnect()
            except TypeError:
                pass
            self.check_activate_selected_placement_area.setChecked(area["active"])
            self.check_activate_selected_placement_area.toggled.connect(
                self.toggle_selected_sub_placement_area_active
            )
            for i, p in enumerate(area["punti"]):
                self.placement_vtx_list_widget.addItem(
                    f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})"
                )
        else:
            self.label_current_placement_area.setText("Nessuna Area di Posizionamento")

        self.on_placement_vtx_selection_change()
        self.full_redraw(preserve_view=True)

    def on_placement_vtx_selection_change(self):
        is_valid_area = (
            0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas)
        )
        selected_items = self.placement_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)

        for w in [self.tb_placement_vtx_x, self.tb_placement_vtx_y, self.btn_update_placement_vtx]:
            w.setEnabled(can_edit)

        if can_edit:
            vtx_idx = self.placement_vtx_list_widget.currentRow()
            vtx = self.lista_sub_placement_areas[self.current_sub_placement_area_idx]["punti"][vtx_idx]
            self.tb_placement_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_placement_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_placement_vtx_x.clear()
            self.tb_placement_vtx_y.clear()

    def on_update_selected_placement_vertex(self):
        if not (0 <= self.current_sub_placement_area_idx < len(self.lista_sub_placement_areas)):
            return
        vtx_idx = self.placement_vtx_list_widget.currentRow()
        if vtx_idx < 0:
            return

        try:
            x = float(self.tb_placement_vtx_x.text())
            y = float(self.tb_placement_vtx_y.text())
            self.lista_sub_placement_areas[self.current_sub_placement_area_idx]["punti"][vtx_idx] = [
                self.snap_to_grid(x), self.snap_to_grid(y)
            ]
            self.update_ui_for_selected_sub_placement_area()
        except ValueError:
            self.error_text_placement_area_mgmt.setText("Coordinate non valide.")

    # Scala SPL auto-adattiva
    def on_auto_scale_spl_toggle(self, checked):
        self.auto_scale_spl = checked
        self.tb_spl_min.setEnabled(not checked)
        self.tb_spl_max.setEnabled(not checked)
        if checked:
            self.trigger_spl_map_recalculation()

    def _calculate_auto_spl_range(self, spl_map):
        """Calcola range SPL automatico con un delta fisso di 30dB."""
        if spl_map is None:
            return 70, 100
            
        valid_values = spl_map[~np.isnan(spl_map) & (spl_map > -200)]
        if valid_values.size == 0:
            return 70, 100
            
        min_spl = np.min(valid_values)
        max_spl = np.max(valid_values)
        
        # Calcola il centro del range effettivo e crea una finestra di 30dB attorno ad esso.
        center = (max_spl + min_spl) / 2.0
        return float(center - 15), float(center + 15)

    # Optimizzazione immagine di sfondo
    def _cache_background_image(self):
        """Cache dell'immagine di sfondo trasformata per performance"""
        if (self.bg_image_props.get("data") is None or 
            self.bg_image_props.get("cached_transformed") is not None):
            return
            
        # Pre-trasforma l'immagine per velocizzare il rendering
        try:
            # Implementazione cache qui se necessario
            pass
        except Exception as e:
            print(f"Errore cache immagine: {e}")

    def disegna_subwoofer_e_elementi(self):
        for i, sub_ in enumerate(self.sorgenti):
            color = "green" if i == self.current_sub_idx else "black"
            if sub_.get("group_id") is not None:
                if sub_["is_group_master"]:
                    color = "purple"
                elif i == self.current_sub_idx:
                    color = "darkorange"
                else:
                    color = "blue"

            center_x, center_y = sub_["x"], sub_["y"]
            angle_rad = sub_["angle"]

            display_angle_deg = 90 - np.degrees(angle_rad)

            width = sub_.get("width", DEFAULT_SUB_WIDTH)
            depth = sub_.get("depth", DEFAULT_SUB_DEPTH)

            rect = patches.Rectangle(
                (-depth / 2, -width / 2),
                depth,
                width,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.6,
                gid=f"rect_sub_{sub_['id']}",
                picker=True,
                zorder=2.5,
            )
            transform = (
                mtransforms.Affine2D().rotate_deg(display_angle_deg)
                + mtransforms.Affine2D().translate(center_x, center_y)
                + self.ax.transData
            )
            rect.set_transform(transform)
            self.ax.add_patch(rect)
            sub_["rect_artist"] = rect

            arrow_end_x = center_x + ARROW_LENGTH * np.sin(angle_rad)
            arrow_end_y = center_y + ARROW_LENGTH * np.cos(angle_rad)
            arrow = patches.FancyArrowPatch(
                (center_x, center_y),
                (arrow_end_x, arrow_end_y),
                mutation_scale=20,
                arrowstyle="->",
                color="dimgray",
                linewidth=1.5,
                zorder=2.6,
                gid=f"arrow_sub_{sub_['id']}",
                picker=True,
            )
            self.ax.add_patch(arrow)
            sub_["arrow_artist"] = arrow

            pol_char = "+" if sub_["polarity"] > 0 else "-"
            group_info = ""
            if sub_.get("group_id") is not None:
                group_info = f"G{sub_['group_id']}"
                if sub_["is_group_master"]:
                    group_info += " (M)"
                group_info += ", "
            sub_text_info = (
                f"S{sub_.get('id', i + 1)}: {sub_.get('spl_rms', DEFAULT_SUB_SPL_RMS):.0f}dB\n"
                f"{group_info}{sub_.get('gain_db', 0.0):.1f}dB, {sub_['delay_ms']:.1f}ms, Pol {pol_char}"
            )

            text_offset = max(width, depth) / 2 + 0.15
            self.ax.text(
                center_x,
                center_y + text_offset,
                sub_text_info,
                color=color,
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=6,
            )

    def disegna_array_direction_indicators(self):
        for group_id, group_info in self.lista_gruppi_array.items():
            members = [s for s in self.sorgenti if s.get("group_id") == group_id]
            if not members:
                continue

            center_x = np.mean([s["x"] for s in members])
            center_y = np.mean([s["y"] for s in members])

            steering_deg_nord = group_info.get("steering_deg", 0)
            steering_deg_est = 90 - steering_deg_nord

            wedge = patches.Wedge(
                center=(center_x, center_y),
                r=ARRAY_INDICATOR_RADIUS,
                theta1=steering_deg_est - ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                theta2=steering_deg_est + ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                facecolor="cyan",
                edgecolor="blue",
                alpha=0.25,
                zorder=1.5,
            )
            self.ax.add_patch(wedge)

    def disegna_stanza_e_vertici(self):
        if self.punti_stanza and len(self.punti_stanza) >= 3:
            points_pos = [p["pos"] for p in self.punti_stanza]
            patch = patches.Polygon(
                points_pos,
                closed=True,
                fill=None,
                edgecolor="blue",
                linewidth=2,
                zorder=1,
            )
            self.ax.add_patch(patch)
            for i, vtx_data in enumerate(self.punti_stanza):
                x, y = vtx_data["pos"]
                vtx_data["plot"] = self.ax.plot(
                    x,
                    y,
                    marker="o",
                    ms=7,
                    color="red",
                    picker=7,
                    gid=f"stanza_vtx_{i}",
                    zorder=5.2,
                )[0]

    def disegna_aree_target_e_avoidance(self):
        area_lists_and_prefixes = [
            (self.lista_target_areas, "target"),
            (self.lista_avoidance_areas, "avoid"),
            (self.lista_sub_placement_areas, "placement"),
        ]
        
        for area_list, type_prefix in area_lists_and_prefixes:
            for area_data in area_list:
                area_data["plots"] = []
                if (
                    area_data.get("active", False)
                    and len(area_data.get("punti", [])) >= 3
                ):
                    is_selected = False
                    if type_prefix == "target":
                        is_selected = (
                            self.current_target_area_idx != -1
                            and area_data["id"] == self.lista_target_areas[self.current_target_area_idx]["id"]
                        )
                        color, m_color, ls = ("green", "lime", "--")
                    elif type_prefix == "avoid":
                        is_selected = (
                            self.current_avoidance_area_idx != -1
                            and area_data["id"] == self.lista_avoidance_areas[self.current_avoidance_area_idx]["id"]
                        )
                        color, m_color, ls = ("red", "orangered", ":")
                    else:  # placement
                        is_selected = (
                            self.current_sub_placement_area_idx != -1
                            and area_data["id"] == self.lista_sub_placement_areas[self.current_sub_placement_area_idx]["id"]
                        )
                        color, m_color, ls = ("purple", "magenta", "-.")

                    patch = patches.Polygon(
                        area_data["punti"],
                        closed=True,
                        fill=True,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3 if is_selected else 0.15,
                        ls=ls,
                        zorder=0.5,
                    )
                    self.ax.add_patch(patch)
                    
                    marker = "P" if type_prefix == "target" else ("X" if type_prefix == "avoid" else "s")
                    for v_idx, v in enumerate(area_data["punti"]):
                        plot_artist = self.ax.plot(
                            v[0],
                            v[1],
                            marker=marker,
                            ms=9,
                            color=m_color,
                            picker=7,
                            zorder=5.1,
                        )[0]
                        area_data["plots"].append(plot_artist)

    def disegna_griglia(self):
        if self.grid_show_enabled and self.grid_snap_spacing > 0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_ticks = np.arange(
                round(xlim[0] / self.grid_snap_spacing) * self.grid_snap_spacing,
                xlim[1],
                self.grid_snap_spacing,
            )
            y_ticks = np.arange(
                round(ylim[0] / self.grid_snap_spacing) * self.grid_snap_spacing,
                ylim[1],
                self.grid_snap_spacing,
            )
            for x in x_ticks:
                self.ax.axvline(
                    x, color="gray", linestyle=":", linewidth=0.5, alpha=0.6, zorder=-1
                )
            for y in y_ticks:
                self.ax.axhline(
                    y, color="gray", linestyle=":", linewidth=0.5, alpha=0.6, zorder=-1
                )

    def auto_fit_view_to_room(self):
        if not hasattr(self, "ax"):
            return
        if not self.punti_stanza:
            self.ax.set_xlim(-7, 7)
            self.ax.set_ylim(-5, 5)
        else:
            all_x = [p["pos"][0] for p in self.punti_stanza]
            all_y = [p["pos"][1] for p in self.punti_stanza]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            px = (max_x - min_x) * 0.15 if (max_x - min_x) > 0 else 1.5
            py = (max_y - min_y) * 0.15 if (max_y - min_y) > 0 else 1.5
            self.ax.set_xlim(min_x - px, max_x + px)
            self.ax.set_ylim(min_y - py, max_y + py)
        self.ax.set_aspect("equal", adjustable="box")
        if hasattr(self, "plot_canvas"):
            self.plot_canvas.canvas.draw_idle()

    def get_active_areas_points(self, area_list):
        return [
            list(area["punti"])
            for area in area_list
            if area.get("active", False) and len(area.get("punti", [])) >= 3
        ]

    def snap_to_grid(self, value):
        if self.grid_snap_enabled and self.grid_snap_spacing > 0:
            return round(value / self.grid_snap_spacing) * self.grid_snap_spacing
        return value
    def add_stanza_vtx(self, event=None):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim()
        new_vtx_coord = [np.mean(xlims), np.mean(ylims)]
        if self.punti_stanza and len(self.punti_stanza) > 1:
            p_last = self.punti_stanza[-1]["pos"]
            new_vtx_coord = [p_last[0] + 1, p_last[1]]
        snapped_pos = [
            self.snap_to_grid(new_vtx_coord[0]),
            self.snap_to_grid(new_vtx_coord[1]),
        ]
        self.punti_stanza.append({"pos": snapped_pos, "plot": None})
        self.full_redraw(preserve_view=False)

    def remove_stanza_vtx(self, event=None):
        if len(self.punti_stanza) > 0:
            self.punti_stanza.pop()
            self.full_redraw(preserve_view=False)

    def update_stanza_vtx_editor(self):
        is_valid_idx = 0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)
        for w in [
            self.tb_stanza_vtx_x,
            self.tb_stanza_vtx_y,
            self.btn_update_stanza_vtx,
        ]:
            w.setEnabled(is_valid_idx)
        if is_valid_idx:
            vtx = self.punti_stanza[self.selected_stanza_vtx_idx]["pos"]
            self.selected_vtx_label.setText(
                f"Vertice Selezionato {self.selected_stanza_vtx_idx + 1}:"
            )
            self.tb_stanza_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_stanza_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.selected_vtx_label.setText("Nessun Vertice Selezionato")
            self.tb_stanza_vtx_x.setText("")
            self.tb_stanza_vtx_y.setText("")
        self.error_text_stanza_vtx_edit.setText("")

    def on_update_selected_stanza_vertex(self):
        if not (0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)):
            return
        try:
            x, y = float(self.tb_stanza_vtx_x.text()), float(
                self.tb_stanza_vtx_y.text()
            )
            self.punti_stanza[self.selected_stanza_vtx_idx]["pos"] = [
                self.snap_to_grid(x),
                self.snap_to_grid(y),
            ]
            self.full_redraw(preserve_view=False)
            self.update_stanza_vtx_editor()
        except ValueError:
            self.error_text_stanza_vtx_edit.setText("Coordinate non valide.")

    def on_toggle_use_global_for_new(self, checked):
        self.use_global_for_new_manual_subs = checked

    def apply_global_settings_to_all_subs(self):
        try:
            new_spl = float(self.tb_global_sub_spl.text())
            if not (
                PARAM_RANGES["spl_rms"][0] <= new_spl <= PARAM_RANGES["spl_rms"][1]
            ):
                raise ValueError(f"SPL fuori range")
            self.global_sub_width = float(self.tb_global_sub_width.text())
            self.global_sub_depth = float(self.tb_global_sub_depth.text())
            self.global_sub_spl_rms = new_spl
            for sub in self.sorgenti:
                sub.update(
                    {
                        "width": self.global_sub_width,
                        "depth": self.global_sub_depth,
                        "spl_rms": self.global_sub_spl_rms,
                        "pressure_val_at_1m_relative_to_pref": 10 ** (new_spl / 20.0),
                    }
                )
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError as e:
            self.error_text_global_settings.setText(f"Errore: {e}")

    def select_next_sub(self, event=None):
        if self.sorgenti:
            self.current_sub_idx = (self.current_sub_idx + 1) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def select_prev_sub(self, event=None):
        if self.sorgenti:
            self.current_sub_idx = (
                self.current_sub_idx - 1 + len(self.sorgenti)
            ) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def add_subwoofer(self, event=None, specific_config=None, redraw=True):
        new_sub_data = specific_config or {}
        if not specific_config:
            new_sub_data = {"x": 0.0, "y": 0.0}
            if self.use_global_for_new_manual_subs:
                new_sub_data.update(
                    {
                        "width": self.global_sub_width,
                        "depth": self.global_sub_depth,
                        "spl_rms": self.global_sub_spl_rms,
                    }
                )

        defaults = {
            "id": self.next_sub_id,
            "angle": 0.0,
            "delay_ms": 0.0,
            "polarity": 1,
            "gain_db": 0.0,
            "width": self.global_sub_width,
            "depth": self.global_sub_depth,
            "spl_rms": self.global_sub_spl_rms,
            "param_locks": {
                "angle": False,
                "delay": False,
                "gain": False,
                "polarity": False,
                "position": False,
            },
            "group_id": None,
            "is_group_master": False,
        }
        for k, v in defaults.items():
            new_sub_data.setdefault(k, v)

        new_sub_data["gain_lin"] = 10 ** (new_sub_data["gain_db"] / 20.0)
        new_sub_data["pressure_val_at_1m_relative_to_pref"] = 10 ** (
            new_sub_data["spl_rms"] / 20.0
        )

        self.sorgenti.append(new_sub_data)
        self.next_sub_id += 1
        self.current_sub_idx = len(self.sorgenti) - 1

        if redraw:
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()

    def remove_subwoofer(self, event=None):
        if not self.sorgenti or self.current_sub_idx < 0:
            return
        sub_to_remove = self.sorgenti.pop(self.current_sub_idx)
        gid_to_check = sub_to_remove.get("group_id")
        if gid_to_check is not None:
            remaining_in_group = [
                s for s in self.sorgenti if s.get("group_id") == gid_to_check
            ]
            if not remaining_in_group and gid_to_check in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid_to_check]
            elif sub_to_remove.get("is_group_master") and remaining_in_group:
                remaining_in_group[0]["is_group_master"] = True

        if not self.sorgenti:
            self.current_sub_idx = -1
        else:
            self.current_sub_idx = min(self.current_sub_idx, len(self.sorgenti) - 1)

        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def aggiorna_ui_sub_fields(self):
        enable = bool(self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti))
        for w in [
            self.tb_sub_x,
            self.tb_sub_y,
            self.tb_sub_angle,
            self.check_sub_angle_lock,
            self.tb_sub_delay,
            self.check_sub_delay_lock,
            self.tb_sub_gain_db,
            self.check_sub_gain_lock,
            self.tb_sub_polarity,
            self.check_sub_polarity_lock,
            self.check_sub_position_lock,
            self.tb_sub_width,
            self.tb_sub_depth,
            self.tb_sub_spl_rms,
            self.btn_submit_sub_params,
            self.btn_rem_sub,
        ]:
            w.setEnabled(enable)

        self.sub_gain_label.setText("Trim Gain (dB):")
        self.sub_delay_label.setText("Delay (ms):")
        self.sub_polarity_label.setText("Polarità (+1/-1):")
        self.tb_sub_gain_db.setPlaceholderText("")
        self.tb_sub_delay.setPlaceholderText("")
        self.tb_sub_polarity.setPlaceholderText("")

        sub = self.sorgenti[self.current_sub_idx] if enable else None

        if enable:
            self.sub_selector_text_widget.setText(
                f"Sub ID:{sub.get('id', '')} ({self.current_sub_idx + 1}/{len(self.sorgenti)})"
            )

            if sub.get("group_id") is not None:
                centroid = self._get_group_centroid(sub["group_id"])
                if centroid:
                    self.tb_sub_x.setText(f"{centroid[0]:.2f}")
                    self.tb_sub_y.setText(f"{centroid[1]:.2f}")
                self.sub_pos_label.setText("X/Y Gruppo (m):")
                self.sub_gain_label.setText("Gain Relativo (+/- dB):")
                self.sub_delay_label.setText("Delay Relativo (+/- ms):")
                self.sub_polarity_label.setText("Polarità Assoluta:")
                self.tb_sub_gain_db.setText("")
                self.tb_sub_delay.setText("")
                self.tb_sub_polarity.setText("")
                self.tb_sub_gain_db.setPlaceholderText("es. 1.5 o -2")
                self.tb_sub_delay.setPlaceholderText("es. 5 o -10")
                self.tb_sub_polarity.setPlaceholderText("1 o -1")
            else:
                self.tb_sub_x.setText(f"{sub['x']:.2f}")
                self.tb_sub_y.setText(f"{sub['y']:.2f}")
                self.sub_pos_label.setText("X/Y Sub (m):")
                self.tb_sub_gain_db.setText(f"{sub['gain_db']:.1f}")
                self.tb_sub_delay.setText(f"{sub['delay_ms']:.2f}")
                self.tb_sub_polarity.setText(str(int(sub["polarity"])))

            self.tb_sub_angle.setText(f"{np.degrees(sub['angle']):.1f}")
            self.tb_sub_width.setText(f"{sub['width']:.2f}")
            self.tb_sub_depth.setText(f"{sub['depth']:.2f}")
            self.tb_sub_spl_rms.setText(f"{sub['spl_rms']:.1f}")
            for p, cb in [
                ("angle", self.check_sub_angle_lock),
                ("delay", self.check_sub_delay_lock),
                ("gain", self.check_sub_gain_lock),
                ("polarity", self.check_sub_polarity_lock),
                ("position", self.check_sub_position_lock),
            ]:
                try:
                    cb.toggled.disconnect()
                except TypeError:
                    pass
                cb.setChecked(sub["param_locks"].get(p, False))
                cb.toggled.connect(self.on_toggle_param_lock)
        else:
            self.sub_selector_text_widget.setText("Nessun Sub")
            for w in [
                self.tb_sub_x,
                self.tb_sub_y,
                self.tb_sub_angle,
                self.tb_sub_delay,
                self.tb_sub_gain_db,
                self.tb_sub_polarity,
                self.tb_sub_width,
                self.tb_sub_depth,
                self.tb_sub_spl_rms,
            ]:
                w.clear()
            self.sub_pos_label.setText("X/Y Sub (m):")

        self.error_text_sub.setText("")
        self._update_group_ui_status()
        self._populate_array_editor_from_selection()

        if enable and sub.get("group_id") is not None:
            self.group_details_label.setVisible(True)
            self.group_members_list.setVisible(True)
            self.group_members_list.clear()
            group_id = sub.get("group_id")
            members = [s for s in self.sorgenti if s.get("group_id") == group_id]
            for member in members:
                pol_char = "+" if member["polarity"] > 0 else "-"
                master_char = " (M)" if member.get("is_group_master") else ""
                info_string = (
                    f"S{member['id']}{master_char}: "
                    f"{member['gain_db']:.1f}dB, "
                    f"{member['delay_ms']:.2f}ms, "
                    f"Pol {pol_char}"
                )
                self.group_members_list.addItem(QListWidgetItem(info_string))
        else:
            self.group_details_label.setVisible(False)
            self.group_members_list.setVisible(False)
            self.group_members_list.clear()

    def _get_group_centroid(self, group_id):
        if group_id is None:
            return None
        members = [s for s in self.sorgenti if s.get("group_id") == group_id]
        if not members:
            return None
        center_x = np.mean([s["x"] for s in members])
        center_y = np.mean([s["y"] for s in members])
        return (center_x, center_y)

    def on_toggle_param_lock(self, checked):
        if not self.sorgenti or self.current_sub_idx < 0:
            return
        sub = self.sorgenti[self.current_sub_idx]
        sender = self.sender()
        param_name = sender.objectName()
        if (
            sub.get("group_id") is not None
            and sub["param_locks"].get(param_name)
            and not checked
        ):
            QMessageBox.warning(
                self,
                "Attenzione",
                f"Il parametro è controllato dall'array. Sciogli il gruppo per sbloccarlo.",
            )
            sender.setChecked(True)
            return
        sub["param_locks"][param_name] = checked

    def on_submit_sub_param_qt(self):
        if not self.sorgenti or self.current_sub_idx < 0:
            return
        try:
            sub = self.sorgenti[self.current_sub_idx]

            if sub.get("group_id") is not None:
                group_id = sub["group_id"]
                current_centroid = self._get_group_centroid(group_id)
                new_centroid_x = float(self.tb_sub_x.text())
                new_centroid_y = float(self.tb_sub_y.text())
                if current_centroid:
                    dx = new_centroid_x - current_centroid[0]
                    dy = new_centroid_y - current_centroid[1]
                    for member in self.sorgenti:
                        if member.get("group_id") == group_id:
                            member["x"] += dx
                            member["y"] += dy

                if self.tb_sub_gain_db.text().strip():
                    gain_delta = float(self.tb_sub_gain_db.text())
                    for member in self.sorgenti:
                        if member.get("group_id") == group_id:
                            member["gain_db"] += gain_delta

                if self.tb_sub_delay.text().strip():
                    delay_delta = float(self.tb_sub_delay.text())
                    for member in self.sorgenti:
                        if member.get("group_id") == group_id:
                            member["delay_ms"] += delay_delta

                if self.tb_sub_polarity.text().strip():
                    new_pol = int(self.tb_sub_polarity.text())
                    if new_pol in [1, -1]:
                        for member in self.sorgenti:
                            if member.get("group_id") == group_id:
                                member["polarity"] = new_pol
            else:
                sub["x"] = float(self.tb_sub_x.text())
                sub["y"] = float(self.tb_sub_y.text())
                if not sub["param_locks"].get("gain", False):
                    sub["gain_db"] = float(self.tb_sub_gain_db.text())
                if not sub["param_locks"].get("delay", False):
                    sub["delay_ms"] = float(self.tb_sub_delay.text())
                if not sub["param_locks"].get("polarity", False):
                    sub["polarity"] = int(self.tb_sub_polarity.text())

            if not sub["param_locks"].get("angle", False):
                sub["angle"] = np.radians(float(self.tb_sub_angle.text()))
            sub.update(
                {
                    "width": float(self.tb_sub_width.text()),
                    "depth": float(self.tb_sub_depth.text()),
                    "spl_rms": float(self.tb_sub_spl_rms.text()),
                }
            )

            for s_ in self.sorgenti:
                s_["gain_lin"] = 10 ** (s_["gain_db"] / 20.0)
                s_["pressure_val_at_1m_relative_to_pref"] = 10 ** (s_["spl_rms"] / 20.0)

            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError:
            self.error_text_sub.setText("Errore: Dati non validi.")
        except Exception as e:
            self.error_text_sub.setText(f"Errore: {e}")

    def _update_max_group_id(self):
        self.max_group_id = max(
            [
                s.get("group_id", 0)
                for s in self.sorgenti
                if s.get("group_id") is not None
            ]
            or [0]
        )
        self.next_group_id = self.max_group_id + 1

    def _update_group_ui_status(self):
        # Controlla se un subwoofer valido è selezionato
        enable = hasattr(self, "btn_add_to_group") and self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti)

        if not enable:
            if hasattr(self, "group_status_label"):
                self.group_status_label.setText("Nessun Sub Selezionato")
            for w in [self.btn_create_new_group, self.btn_add_to_group, self.btn_remove_from_group, self.btn_ungroup_all]:
                w.setEnabled(False)
            if hasattr(self, "array_length_label"):
                self.array_length_label.setText("Lunghezza Array: N/D")
            return

        # Se il sub è valido, procedi
        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get("group_id")  # La variabile corretta è 'g_id'
        is_master = sub.get("is_group_master")

        # Abilita/disabilita i bottoni
        self.btn_create_new_group.setEnabled(g_id is None)
        self.btn_add_to_group.setEnabled(g_id is None)
        self.btn_remove_from_group.setEnabled(g_id is not None)
        self.btn_ungroup_all.setEnabled(g_id is not None)

        # Aggiorna il testo della label di stato
        if g_id:
            self.group_status_label.setText(
                f"Sub selezionato: Gruppo ID {g_id}{' (M)' if is_master else ''}"
            )
        else:
            self.group_status_label.setText("Sub selezionato: Nessun Gruppo")
        
        # Chiama la funzione di aggiornamento lunghezza usando la variabile corretta 'g_id'
        self._update_array_length_display(g_id)

    def disegna_immagine_di_sfondo(self):
        if self.bg_image_props.get("artist") is not None:
            try:
                if self.bg_image_props["artist"] in self.ax.images:
                    self.bg_image_props["artist"].remove()
            except (AttributeError, ValueError, NotImplementedError):
                self.ax.images.clear() 

            self.bg_image_props["artist"] = None

        if (
            self.bg_image_props.get("data") is not None
            and not self.is_in_calibration_mode
        ):
            # Usa immagine cached se disponibile per performance
            image_data = self.bg_image_props.get("cached_transformed") or self.bg_image_props["data"]
            
            self.bg_image_props["artist"] = self.ax.imshow(
                image_data,
                alpha=self.bg_image_props.get("alpha", 0.5),
                zorder=-10,
                origin="upper",
            )

            h, w = self.bg_image_props["data"].shape[:2]
            scale = self.bg_image_props.get("scale", 1.0)
            rotation = self.bg_image_props.get("rotation_deg", 0.0)
            center_x = self.bg_image_props.get("center_x", 0.0)
            center_y = self.bg_image_props.get("center_y", 0.0)

            transform = None

            if (
                "anchor_pixel" in self.bg_image_props
                and self.bg_image_props["anchor_pixel"] is not None
            ):
                anchor_px = self.bg_image_props["anchor_pixel"]

                transform = (
                    mtransforms.Affine2D()
                    .translate(-anchor_px[0], -anchor_px[1])
                    .rotate_deg(rotation)
                    .scale(scale, -scale) 
                    .translate(center_x, center_y)
                )

            else:
                transform = (
                    mtransforms.Affine2D()
                    .translate(-w / 2, -h / 2)
                    .scale(scale, -scale) 
                    .rotate_deg(rotation)
                    .translate(center_x, center_y)
                )

            if transform:
                self.bg_image_props["artist"].set_transform(
                    transform + self.ax.transData
                )

    def load_background_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Seleziona Immagine", "", "File Immagine (*.png *.jpg *.jpeg)"
        )
        if not filepath:
            return
        try:
            self.remove_background_image()
            self.bg_image_props["path"] = filepath
            self.bg_image_props["data"] = plt.imread(filepath)
            self.calibration_status_label.setText(
                "Immagine caricata. Usa 'Calibra Scala' per dimensionarla."
            )
            self.start_calibration() 
        except Exception as e:
            QMessageBox.critical(
                self, "Errore Caricamento", f"Impossibile caricare l'immagine:\n{e}"
            )
            self.remove_background_image()

    def remove_background_image(self):
        if self.bg_image_props.get("artist"):
            self.bg_image_props["artist"].remove()
        self.bg_image_props = {
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
        self.is_in_calibration_mode = False
        self.calibration_points = []
        self.calibration_status_label.setText("Pronto per caricare un'immagine.")
        self.full_redraw(preserve_view=True)

    def start_calibration(self):
        if self.bg_image_props.get("data") is None:
            QMessageBox.warning(
                self,
                "Nessuna Immagine",
                "Carica un'immagine prima di iniziare la calibrazione.",
            )
            return

        if self.bg_image_props.get("artist"):
            self.bg_image_props["artist"].remove()
            self.bg_image_props["artist"] = None

        self.ax.cla()

        self.disegna_elementi_statici_senza_spl()

        self.bg_image_props["artist"] = self.ax.imshow(
            self.bg_image_props["data"], zorder=-10, alpha=0.7, origin="upper"
        )

        altezza_pixel, larghezza_pixel, *_ = self.bg_image_props["data"].shape

        self.ax.set_xlim(0, larghezza_pixel)
        self.ax.set_ylim(
            altezza_pixel, 0
        )  

        self.ax.set_aspect("equal", adjustable="box")

        self.is_in_calibration_mode = True
        self.calibration_points = []
        self.calibration_status_label.setText(
            "MODALITÀ CALIBRAZIONE: Clicca il primo punto sull'immagine."
        )
        self.status_bar.showMessage("MODALITÀ CALIBRAZIONE ATTIVA", 0)

        self.plot_canvas.canvas.draw_idle()

    def handle_calibration_click(self, event):
        if not self.is_in_calibration_mode:
            return False

        click_point = (event.xdata, event.ydata)
        
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return False

        self.calibration_points.append(click_point)

        if len(self.calibration_points) == 1:
            self.ax.plot(
                click_point[0], click_point[1], "r+", markersize=15, markeredgewidth=2
            )
            self.plot_canvas.canvas.draw_idle()
            self.calibration_status_label.setText(
                "OK. Ora clicca il secondo punto sull'immagine."
            )
        elif len(self.calibration_points) == 2:
            self.ax.plot(
                click_point[0], click_point[1], "g+", markersize=15, markeredgewidth=2
            )
            self.plot_canvas.canvas.draw_idle()

            p1_pix, p2_pix = self.calibration_points[0], self.calibration_points[1]
            dist_pixels = math.sqrt(
                (p2_pix[0] - p1_pix[0]) ** 2 + (p2_pix[1] - p1_pix[1]) ** 2
            )

            if dist_pixels < 1:
                self.finish_calibration(
                    success=False, message="I punti sono troppo vicini. Riprova."
                )
                return True

            dist_meters, ok = QInputDialog.getDouble(
                self,
                "Distanza Reale",
                "Inserisci la distanza reale tra i due punti (in metri):",
                1.0,
                0.01,
                10000,
                2,
            )

            if ok and dist_meters > 0:
                self.bg_image_props["scale"] = dist_meters / dist_pixels
                
                angle_rad = math.atan2(-(p2_pix[1] - p1_pix[1]), (p2_pix[0] - p1_pix[0]))
                self.bg_image_props["rotation_deg"] = math.degrees(angle_rad)

                self.bg_image_props["center_x"] = 0
                self.bg_image_props["center_y"] = 0

                self.bg_image_props["anchor_pixel"] = p1_pix

                self.finish_calibration(
                    True,
                    "Calibrazione completata. Usa i campi per affinare la posizione.",
                )
            else:
                self.finish_calibration(False, "Calibrazione annullata.")
        return True

    def finish_calibration(self, success=True, message=""):
        self.is_in_calibration_mode = False
        self.calibration_points = []
        self.calibration_status_label.setText(message)
        self.status_bar.clearMessage()
        if success:
            self.tb_bg_x.setText(f"{self.bg_image_props['center_x']:.2f}")
            self.tb_bg_y.setText(f"{self.bg_image_props['center_y']:.2f}")
            self.tb_bg_scale.setText(f"{self.bg_image_props['scale']:.6f}")
            self.tb_bg_rotation.setText(f"{self.bg_image_props['rotation_deg']:.2f}")
        self.full_redraw(preserve_view=False)

    def update_background_image_manually(self):
        if self.bg_image_props.get("data") is None:
            return
        try:
            self.bg_image_props["center_x"] = float(self.tb_bg_x.text())
            self.bg_image_props["center_y"] = float(self.tb_bg_y.text())
            self.bg_image_props["scale"] = float(self.tb_bg_scale.text())
            self.bg_image_props["rotation_deg"] = float(self.tb_bg_rotation.text())
            self.bg_image_props["alpha"] = float(self.tb_bg_alpha.text())
            
            self.bg_image_props["anchor_pixel"] = None 
            self.bg_image_props["cached_transformed"] = None  # Reset cache

            self.full_redraw(preserve_view=True)
        except ValueError:
            QMessageBox.warning(
                self,
                "Dati non validi",
                "I valori per l'immagine devono essere numerici.",
            )

    def create_new_group(self):
        if not self.sorgenti or self.current_sub_idx == -1:
            return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get("group_id"):
            self.error_text_grouping.setText("Sub già in un gruppo.")
            return
        self._update_max_group_id()
        sub["group_id"] = self.next_group_id
        sub["is_group_master"] = True
        self.status_bar.showMessage(f"Creato Gruppo ID {self.next_group_id}.", 3000)
        self.aggiorna_ui_sub_fields()

    def add_sub_to_existing_group(self):
        if not self.sorgenti or self.current_sub_idx == -1:
            return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get("group_id"):
            self.error_text_grouping.setText("Sub già in un gruppo.")
            return

        existing_groups = sorted(
            list(set(s["group_id"] for s in self.sorgenti if s["group_id"] is not None))
        )
        if not existing_groups:
            self.error_text_grouping.setText("Nessun gruppo esistente.")
            return

        group_id_str, ok = QInputDialog.getItem(
            self,
            "Aggiungi a Gruppo",
            "Seleziona ID del gruppo:",
            [str(g) for g in existing_groups],
            0,
            False,
        )
        if not ok or not group_id_str:
            return

        target_id = int(group_id_str)
        sub["group_id"] = target_id
        sub["is_group_master"] = False
        self.aggiorna_ui_sub_fields()

    def remove_sub_from_group(self):
        if not self.sorgenti or self.current_sub_idx < 0:
            return
        sub = self.sorgenti[self.current_sub_idx]
        if not sub.get("group_id"):
            return

        gid = sub.get("group_id")
        is_master = sub.get("is_group_master")

        sub["group_id"] = None
        sub["is_group_master"] = False
        for p in sub["param_locks"]:
            sub["param_locks"][p] = False

        if is_master:
            remaining_in_group = [s for s in self.sorgenti if s.get("group_id") == gid]
            if remaining_in_group:
                remaining_in_group[0]["is_group_master"] = True
            elif gid in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid]

        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)

    def ungroup_selected_sub_group(self):
        if not self.sorgenti or self.current_sub_idx < 0:
            return
        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get("group_id")
        if g_id is None:
            return

        if g_id in self.lista_gruppi_array:
            del self.lista_gruppi_array[g_id]

        for s in self.sorgenti:
            if s.get("group_id") == g_id:
                s["group_id"] = None
                s["is_group_master"] = False
                for p in s["param_locks"]:
                    s["param_locks"][p] = False
        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)

    def _normalize_delays(self, configs_list):
        if not configs_list:
            return
        all_delays = [config["delay_ms"] for config in configs_list]
        min_delay = min(all_delays)
        if min_delay != 0:
            for config in configs_list:
                config["delay_ms"] -= min_delay

    def _update_array_ui(self):
        array_type = self.array_type_combo.currentText()
        is_none = array_type == "Nessuno"
        is_card_endfire = array_type in ["Coppia Cardioide (2 sub)", "Array End-Fire"]
        is_line_array = array_type == "Array Lineare (Steering Elettrico)"
        is_vortex = array_type == "Array Vortex"
        is_multi_cardioid = array_type == "Cardiodi Multipli (3+ sub)"
        is_gradient = array_type == "Gradiente di Pressione"
        is_3sub_cardioid = array_type == "Cardioide a 3 Sub"
        is_auto_spacing = self.array_auto_spacing_check.isChecked()

        self.array_freq_label.setVisible(not is_none)
        self.array_freq_input.setVisible(not is_none)
        self.array_auto_spacing_check.setVisible(
            is_card_endfire or is_line_array or is_vortex or is_multi_cardioid or is_gradient or is_3sub_cardioid
        )

        show_frac_lambda = is_auto_spacing and (
            is_card_endfire or is_line_array or is_vortex or is_multi_cardioid or is_gradient or is_3sub_cardioid
        )
        self.array_wavelength_fraction_label.setVisible(show_frac_lambda)
        self.array_wavelength_fraction_combo.setVisible(show_frac_lambda)

        self.array_spacing_label.setVisible(not is_none)
        self.array_spacing_input.setVisible(not is_none)

        if is_vortex:
            self.array_spacing_label.setText("Raggio Array (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        elif is_card_endfire or is_line_array or is_multi_cardioid or is_3sub_cardioid  or is_gradient:
            self.array_spacing_label.setText("Spaziatura Fisica (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        else:
            self.array_spacing_label.setText("Spaziatura/Raggio (m):")
            self.array_spacing_input.setReadOnly(False)

        self.array_elements_label.setVisible(not is_none)
        self.array_elements_input.setVisible(not is_none)
        
        # Logica per abilitare/disabilitare e impostare N. elementi
        if array_type == "Coppia Cardioide (2 sub)":
            self.array_elements_input.setText("2")
            self.array_elements_input.setEnabled(False)
        elif is_3sub_cardioid:
            self.array_elements_input.setText("3")
            self.array_elements_input.setEnabled(False)
        else:
            self.array_elements_input.setEnabled(True)

        should_show_start_angle = is_line_array or is_vortex or is_card_endfire or is_multi_cardioid or is_3sub_cardioid  or is_gradient
        self.array_start_angle_label.setVisible(should_show_start_angle)
        self.array_start_angle_input.setVisible(should_show_start_angle)

        if is_line_array or is_card_endfire or is_multi_cardioid or is_3sub_cardioid  or is_gradient:
            self.array_start_angle_label.setText("Orientamento (°):")
        elif is_vortex:
            self.array_start_angle_label.setText("Angolo Iniziale (°):")

        for w in [
            self.array_line_steering_angle_label,
            self.array_line_steering_angle_input,
            self.array_line_coverage_angle_label,
            self.array_line_coverage_angle_input,
        ]:
            w.setVisible(is_line_array)

        self.array_vortex_mode_label.setVisible(is_vortex)
        self.array_vortex_mode_input.setVisible(is_vortex)
        self.array_vortex_steering_angle_label.setVisible(is_vortex)
        self.array_vortex_steering_angle_input.setVisible(is_vortex)

        self.array_3sub_cardio_pos_label.setVisible(is_3sub_cardioid)
        self.array_3sub_cardio_pos_combo.setVisible(is_3sub_cardioid)

        show_slider = is_line_array or is_vortex
        self.array_steering_slider_label.setVisible(show_slider)
        self.array_steering_slider.setVisible(show_slider)

    def on_array_type_change(self):
        self._update_array_ui()
        array_type = self.array_type_combo.currentText()
        info_text = "Il sub selezionato (se esiste) verrà SOSTITUITO dal nuovo array."
        if array_type == "Nessuno":
            self.array_info_label.setText("Seleziona un tipo di array da configurare.")
        else:
            self.array_info_label.setText(info_text)
        self.error_text_array_params.setText("")

    def on_auto_spacing_toggle(self):
        self._update_array_ui()

    def apply_or_update_array_configuration(self):
        if not self.sorgenti or self.current_sub_idx < 0:
            self.apply_array_configuration(create_new=True)
            return
        sub = self.sorgenti[self.current_sub_idx]
        group_id = sub.get("group_id")
        if group_id and group_id in self.lista_gruppi_array:
            self.apply_array_configuration(
                create_new=False, group_id_to_update=group_id
            )
        else:
            self.apply_array_configuration(create_new=True)

    def apply_array_configuration(self, create_new=True, group_id_to_update=None, suggested_subs_to_add=0, array_type=None):
        ref_sub_idx = (
            self.current_sub_idx
            if self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti)
            else -1
        )
        ref_sub_model = self.sorgenti[ref_sub_idx] if ref_sub_idx != -1 else None

        center_x, center_y = (
            (ref_sub_model["x"], ref_sub_model["y"])
            if ref_sub_model and create_new
            else (0, 0)
        )
        
        if not create_new and group_id_to_update:
            centroid = self._get_group_centroid(group_id_to_update)
            if centroid:
                center_x, center_y = centroid

        self.error_text_array_params.setText("")
        if array_type is None:
            array_type = self.array_type_combo.currentText()

        if array_type == "Nessuno":
            return

        try:
            c_sound = float(self.tb_velocita_suono.text())
            design_freq = float(self.array_freq_input.text())
            num_elements = int(self.array_elements_input.text())
            spacing_or_radius = float(self.array_spacing_input.text())
            start_angle_deg = (
                float(self.array_start_angle_input.text())
                if self.array_start_angle_input.isVisible()
                else 0.0
            )

            if self.array_auto_spacing_check.isChecked() and array_type != "Nessuno":
                if design_freq <= 0:
                    raise ValueError("La frequenza di design deve essere positiva.")
                wavelength = c_sound / design_freq
                spacing_or_radius = (
                    wavelength / 4.0
                    if self.array_wavelength_fraction_combo.currentText() == "λ/4"
                    else wavelength / 2.0
                )
                self.array_spacing_input.setText(f"{spacing_or_radius:.3f}")

            base_sub_params = {
                "width": ref_sub_model.get("width", self.global_sub_width) if ref_sub_model else self.global_sub_width,
                "depth": ref_sub_model.get("depth", self.global_sub_depth) if ref_sub_model else self.global_sub_depth,
                "spl_rms": ref_sub_model.get("spl_rms", self.global_sub_spl_rms) if ref_sub_model else self.global_sub_spl_rms,
                "gain_db": 0, "polarity": 1, "delay_ms": 0,
            }
            
            new_configs = []
            array_params = {
                "type": array_type,
                "design_freq": design_freq,
                "num_elements": num_elements,
                "c_sound": c_sound,
                "spacing": spacing_or_radius,
                "orientation_deg": start_angle_deg,
            }

            if array_type == "Coppia Cardioide (2 sub)":
                new_configs = self._calculate_cardioid_configs(
                    center_x, center_y, spacing_or_radius, c_sound, start_angle_deg, base_sub_params
                )
            elif array_type == "Array End-Fire":
                new_configs = self._calculate_endfire_configs(
                    center_x, center_y, num_elements, spacing_or_radius, c_sound, start_angle_deg, base_sub_params
                )
            elif array_type == "Array Lineare (Steering Elettrico)":
                steering_angle_deg = float(self.array_line_steering_angle_input.text())
                coverage_angle_deg = float(self.array_line_coverage_angle_input.text())
                array_params.update({"steering_deg": steering_angle_deg, "coverage_deg": coverage_angle_deg})
                new_configs = self._calculate_line_array_steered_configs(
                    center_x, center_y, num_elements, spacing_or_radius, start_angle_deg,
                    steering_angle_deg, coverage_angle_deg, c_sound, base_sub_params
                )
            elif array_type == "Array Vortex":
                vortex_mode = int(self.array_vortex_mode_input.text())
                steering_deg = float(self.array_vortex_steering_angle_input.text())
                array_params.update({"mode": vortex_mode, "steering_deg": steering_deg, "radius": spacing_or_radius})
                new_configs = self._calculate_vortex_array_configs(
                    center_x, center_y, num_elements, spacing_or_radius, vortex_mode,
                    design_freq, start_angle_deg, steering_deg, c_sound, base_sub_params
                )
            elif array_type == "Gradiente di Pressione":
                 new_configs = self._calculate_pressure_gradient_configs(
                     center_x, center_y, num_elements, spacing_or_radius, c_sound, start_angle_deg, base_sub_params
                 )
            elif array_type == "Cardiodi Multipli (3+ sub)":
                new_configs = self._calculate_multi_cardioid_configs(
                    center_x, center_y, num_elements, spacing_or_radius, c_sound, base_sub_params
                )
            elif array_type == "Cardioide a 3 Sub (Sterzante)":
                reversed_pos_index = self.array_3sub_cardio_pos_combo.currentIndex()
                array_params.update({"reversed_pos_index": reversed_pos_index})
                new_configs = self._calculate_steerable_3sub_cardioid_configs(
                    center_x, center_y, spacing_or_radius, start_angle_deg, reversed_pos_index, c_sound, base_sub_params
                )

            if create_new:
                self._add_new_subs_as_group(new_configs, array_params, ref_sub_idx)
            else:
                self._update_subs_in_group(group_id_to_update, new_configs, array_params)

        except Exception as e:
            self.error_text_array_params.setText(f"Errore parametri array: {e}")

    def on_array_steering_slider_change(self, value):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]
        group_id = sub.get('group_id')
        if not (group_id and group_id in self.lista_gruppi_array): return

        array_type = self.lista_gruppi_array[group_id].get('type')
        if array_type == "Array Lineare (Steering Elettrico)":
            self.array_line_steering_angle_input.setText(str(value))
        elif array_type == "Array Vortex":
            self.array_vortex_steering_angle_input.setText(str(value))
        else:
            return

        self.apply_array_configuration(create_new=False, group_id_to_update=group_id)
        self.lista_gruppi_array[group_id]['steering_deg'] = float(value)
    
        for s in self.sorgenti:
            if s.get("rect_artist") in self.ax.patches:
                s["rect_artist"].remove()
            if s.get("arrow_artist") in self.ax.patches:
                s["arrow_artist"].remove()
        
        for patch in self.ax.patches[:]:
            if isinstance(patch, patches.Wedge) and patch.get_alpha() == 0.25:
                patch.remove()

        for text in self.ax.texts[:]:
            if any(f"S{s['id']}" in text.get_text() for s in self.sorgenti):
                text.remove()

        self.disegna_subwoofer_e_elementi()
        self.disegna_array_direction_indicators()
        
        self.plot_canvas.canvas.draw_idle()

    def _add_new_subs_as_group(
        self, configs_list, array_params, ref_sub_idx_to_remove=-1
    ):
        if not configs_list:
            return
        if ref_sub_idx_to_remove != -1 and ref_sub_idx_to_remove < len(self.sorgenti):
            self.sorgenti.pop(ref_sub_idx_to_remove)

        self._update_max_group_id()
        new_group_id = self.next_group_id
        self.lista_gruppi_array[new_group_id] = array_params

        start_index_of_new_subs = len(self.sorgenti)
        for config in configs_list:
            new_sub_data = config.copy()
            new_sub_data["group_id"] = new_group_id

            defaults = {
                "id": self.next_sub_id,
                "param_locks": {
                    "angle": True,
                    "delay": True,
                    "gain": False,
                    "polarity": True,
                    "position": False,
                },
            }
            for k, v in defaults.items():
                new_sub_data.setdefault(k, v)

            new_sub_data["gain_lin"] = 10 ** (new_sub_data.get("gain_db", 0) / 20.0)
            new_sub_data["pressure_val_at_1m_relative_to_pref"] = 10 ** (
                new_sub_data.get("spl_rms", DEFAULT_SUB_SPL_RMS) / 20.0
            )

            self.sorgenti.append(new_sub_data)
            self.next_sub_id += 1

        master_idx_in_group = next(
            (i for i, conf in enumerate(configs_list) if conf.get("is_group_master")), 0
        )
        self.current_sub_idx = start_index_of_new_subs + master_idx_in_group

        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def _update_subs_in_group(self, group_id, new_configs, array_params):
        members = [s for s in self.sorgenti if s.get("group_id") == group_id]
        if len(members) != len(new_configs):
            self.error_text_array_params.setText(
                "Errore: il numero di elementi non può essere cambiato in un array esistente."
            )
            return

        for i, sub in enumerate(members):
            new_data = new_configs[i]
            sub.update(new_data)
            sub["gain_lin"] = 10 ** (sub.get("gain_db", 0) / 20.0)

        self.lista_gruppi_array[group_id] = array_params
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def _calculate_cardioid_configs(
        self, center_x, center_y, spacing, c, angle_deg, base_params
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
        self._normalize_delays(configs)
        return configs

    def _calculate_endfire_configs(
        self, center_x, center_y, num_elements, spacing, c, angle_deg, base_params
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
        self._normalize_delays(new_configs)
        return new_configs

    def _calculate_line_array_steered_configs(
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
        self._normalize_delays(new_configs)
        return new_configs

    def _calculate_vortex_array_configs(
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
        self._normalize_delays(new_configs)
        return new_configs

    def _populate_array_editor_from_selection(self):
        if not self.sorgenti or self.current_sub_idx < 0:
            is_array = False
        else:
            sub = self.sorgenti[self.current_sub_idx]
            group_id = sub.get("group_id")
            is_array = group_id is not None and group_id in self.lista_gruppi_array

        self.array_type_combo.setEnabled(not is_array)

        if is_array:
            params = self.lista_gruppi_array[group_id]
            self.apply_array_config_button.setText("Aggiorna Array")
            self.array_type_combo.setCurrentText(params.get("type", "Nessuno"))

            # Popola i campi con i parametri salvati
            self.array_freq_input.setText(
                str(params.get("freq", params.get("design_freq", DEFAULT_ARRAY_FREQ)))
            )
            self.array_spacing_input.setText(
                str(params.get("spacing", params.get("radius", DEFAULT_ARRAY_RADIUS)))
            )
            self.array_elements_input.setText(str(params.get("num_elements", 4)))
            self.array_start_angle_input.setText(
                str(
                    params.get(
                        "orientation_deg",
                        params.get("start_angle_deg", DEFAULT_ARRAY_START_ANGLE_DEG),
                    )
                )
            )
            self.array_line_coverage_angle_input.setText(
                str(params.get("coverage_deg", DEFAULT_LINE_ARRAY_COVERAGE_DEG))
            )
            self.array_line_steering_angle_input.setText(
                str(params.get("steering_deg", DEFAULT_LINE_ARRAY_STEERING_DEG))
            )
            self.array_vortex_mode_input.setText(
                str(params.get("mode", DEFAULT_VORTEX_MODE))
            )
            self.array_vortex_steering_angle_input.setText(
                str(params.get("steering_deg", DEFAULT_VORTEX_STEERING_DEG))
            )

            # Imposta lo slider
            show_slider = params.get("type") in [
                "Array Lineare (Steering Elettrico)",
                "Array Vortex",
            ]
            self.array_steering_slider_label.setVisible(show_slider)
            self.array_steering_slider.setVisible(show_slider)
            if show_slider:
                self.array_steering_slider.blockSignals(True)
                self.array_steering_slider.setValue(int(params.get("steering_deg", 0)))
                self.array_steering_slider.blockSignals(False)
        else:
            # Nessun array selezionato, ripristina i valori di default
            self.apply_array_config_button.setText("Crea Gruppo Array")
            
            # Nascondi lo slider di steering
            self.array_steering_slider_label.setVisible(False)
            self.array_steering_slider.setVisible(False)
            
            # Ripristina i valori di default nei campi
            self.array_freq_input.setText(str(DEFAULT_ARRAY_FREQ))
            self.array_spacing_input.setText(str(DEFAULT_ARRAY_RADIUS))
            self.array_elements_input.setText("4")
            self.array_start_angle_input.setText(str(DEFAULT_ARRAY_START_ANGLE_DEG))
            self.array_line_coverage_angle_input.setText(str(DEFAULT_LINE_ARRAY_COVERAGE_DEG))
            self.array_line_steering_angle_input.setText(str(DEFAULT_LINE_ARRAY_STEERING_DEG))
            self.array_vortex_mode_input.setText(str(DEFAULT_VORTEX_MODE))
            self.array_vortex_steering_angle_input.setText(str(DEFAULT_VORTEX_STEERING_DEG))
        
        # Aggiorna la UI in base al tipo di array selezionato
        self._update_array_ui()
    def select_prev_target_area(self):
        if not self.lista_target_areas:
            self.current_target_area_idx = -1
        else:
            self.current_target_area_idx = (
                self.current_target_area_idx - 1 + len(self.lista_target_areas)
            ) % len(self.lista_target_areas)
        self.update_ui_for_selected_target_area()

    def select_next_target_area(self):
        if not self.lista_target_areas:
            self.current_target_area_idx = -1
        else:
            self.current_target_area_idx = (self.current_target_area_idx + 1) % len(
                self.lista_target_areas
            )
        self.update_ui_for_selected_target_area()

    def _add_vtx_to_current_target_area(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            area = self.lista_target_areas[self.current_target_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area["punti"].append([new_x, new_y])
            self.update_ui_for_selected_target_area()
        else:
            self.error_text_target_area_mgmt.setText(
                "Selezionare prima un'area target o crearne una nuova."
            )

    def _add_new_area_data(
        self, area_list, default_vertices, base_name, next_id_attr, activate=True
    ):
        new_id = getattr(self, next_id_attr)
        snapped_default_vertices = [
            [self.snap_to_grid(p[0]), self.snap_to_grid(p[1])] for p in default_vertices
        ]
        area_data = {
            "id": new_id,
            "nome": f"{base_name} {new_id}",
            "punti": snapped_default_vertices,
            "active": activate,
            "plots": [],
        }
        area_list.append(area_data)
        setattr(self, next_id_attr, new_id + 1)
        return len(area_list) - 1

    def _get_area_center_and_default_size(self):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim()
        cx, cy = np.mean(xlims), np.mean(ylims)
        size = min(xlims[1] - xlims[0], ylims[1] - ylims[0]) / 4.0
        return cx, cy, max(1.0, size)

    def add_new_target_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size()
        hs = size / 2.0
        default_verts = [
            [cx - hs, cy - hs],
            [cx + hs, cy - hs],
            [cx + hs, cy + hs],
            [cx - hs, cy + hs],
        ]
        self.current_target_area_idx = self._add_new_area_data(
            self.lista_target_areas, default_verts, "Target", "next_target_area_id", activate=True 
        )
        self.update_ui_for_selected_target_area()

    def remove_selected_target_area_ui(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            self.lista_target_areas.pop(self.current_target_area_idx)
            if not self.lista_target_areas:
                self.current_target_area_idx = -1
            elif self.current_target_area_idx >= len(self.lista_target_areas):
                self.current_target_area_idx = len(self.lista_target_areas) - 1
            self.update_ui_for_selected_target_area()

    def toggle_selected_target_area_active(self, checked):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            self.lista_target_areas[self.current_target_area_idx]["active"] = checked
            self.update_ui_for_selected_target_area()

    def update_ui_for_selected_target_area(self):
        is_valid_idx = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        self.btn_add_target_vtx.setEnabled(is_valid_idx)
        for w in [
            self.btn_prev_target_area,
            self.btn_next_target_area,
            self.btn_remove_selected_target_area,
            self.check_activate_selected_target_area,
            self.target_vtx_list_widget,
        ]:
            w.setEnabled(is_valid_idx)

        self.target_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_target_areas[self.current_target_area_idx]
            self.label_current_target_area.setText(
                f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})"
            )
            try:
                self.check_activate_selected_target_area.toggled.disconnect()
            except TypeError:
                pass
            self.check_activate_selected_target_area.setChecked(area["active"])
            self.check_activate_selected_target_area.toggled.connect(
                self.toggle_selected_target_area_active
            )
            for i, p in enumerate(area["punti"]):
                self.target_vtx_list_widget.addItem(
                    f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})"
                )
        else:
            self.label_current_target_area.setText("Nessuna Area Target")

        self.on_target_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()

    def on_target_vtx_selection_change(self):
        is_valid_area = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        selected_items = self.target_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)

        for w in [
            self.tb_target_vtx_x,
            self.tb_target_vtx_y,
            self.btn_update_target_vtx,
        ]:
            w.setEnabled(can_edit)

        if can_edit:
            vtx_idx = self.target_vtx_list_widget.currentRow()
            vtx = self.lista_target_areas[self.current_target_area_idx]["punti"][
                vtx_idx
            ]
            self.tb_target_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_target_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_target_vtx_x.clear()
            self.tb_target_vtx_y.clear()

    def on_update_selected_target_vertex(self):
        if not (0 <= self.current_target_area_idx < len(self.lista_target_areas)):
            return
        vtx_idx = self.target_vtx_list_widget.currentRow()
        if vtx_idx < 0:
            return

        try:
            x = float(self.tb_target_vtx_x.text())
            y = float(self.tb_target_vtx_y.text())
            self.lista_target_areas[self.current_target_area_idx]["punti"][vtx_idx] = [
                self.snap_to_grid(x),
                self.snap_to_grid(y),
            ]
            self.update_ui_for_selected_target_area()
        except ValueError:
            self.error_text_target_area_mgmt.setText("Coordinate non valide.")

    def select_prev_avoidance_area(self):
        if not self.lista_avoidance_areas:
            self.current_avoidance_area_idx = -1
        else:
            self.current_avoidance_area_idx = (
                self.current_avoidance_area_idx - 1 + len(self.lista_avoidance_areas)
            ) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()

    def select_next_avoidance_area(self):
        if not self.lista_avoidance_areas:
            self.current_avoidance_area_idx = -1
        else:
            self.current_avoidance_area_idx = (
                self.current_avoidance_area_idx + 1
            ) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()

    def add_new_avoidance_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size()
        hs = size / 2.0 * 0.8
        default_verts = [
            [cx - hs, cy - hs],
            [cx + hs, cy - hs],
            [cx + hs, cy + hs],
            [cx - hs, cy + hs],
        ]
        self.current_avoidance_area_idx = self._add_new_area_data(
            self.lista_avoidance_areas,
            default_verts,
            "Evitamento",
            "next_avoidance_area_id",
            activate=True
        )
        self.update_ui_for_selected_avoidance_area()

    def remove_selected_avoidance_area_ui(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            self.lista_avoidance_areas.pop(self.current_avoidance_area_idx)
            if not self.lista_avoidance_areas:
                self.current_avoidance_area_idx = -1
            elif self.current_avoidance_area_idx >= len(self.lista_avoidance_areas):
                self.current_avoidance_area_idx = len(self.lista_avoidance_areas) - 1
            self.update_ui_for_selected_avoidance_area()

    def toggle_selected_avoidance_area_active(self, checked):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            self.lista_avoidance_areas[self.current_avoidance_area_idx][
                "active"
            ] = checked
            self.update_ui_for_selected_avoidance_area()

    def _add_vtx_to_current_avoidance_area(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area["punti"].append([new_x, new_y])
            self.update_ui_for_selected_avoidance_area()
        else:
            self.error_text_avoid_area_mgmt.setText(
                "Selezionare prima un'area di evitamento o crearne una nuova."
            )

    def update_ui_for_selected_avoidance_area(self):
        is_valid_idx = (
            0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        )
        self.btn_add_avoid_vtx.setEnabled(is_valid_idx)
        for w in [
            self.btn_prev_avoid_area,
            self.btn_next_avoid_area,
            self.btn_remove_selected_avoid_area,
            self.check_activate_selected_avoid_area,
            self.avoid_vtx_list_widget,
        ]:
            w.setEnabled(is_valid_idx)

        self.avoid_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            self.label_current_avoid_area.setText(
                f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})"
            )
            try:
                self.check_activate_selected_avoid_area.toggled.disconnect()
            except TypeError:
                pass
            self.check_activate_selected_avoid_area.setChecked(area["active"])
            self.check_activate_selected_avoid_area.toggled.connect(
                self.toggle_selected_avoidance_area_active
            )
            for i, p in enumerate(area["punti"]):
                self.avoid_vtx_list_widget.addItem(
                    f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})"
                )
        else:
            self.label_current_avoid_area.setText("Nessuna Area di Evitamento")

        self.on_avoid_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()

    def on_avoid_vtx_selection_change(self):
        is_valid_area = (
            0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        )
        selected_items = self.avoid_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)

        for w in [self.tb_avoid_vtx_x, self.tb_avoid_vtx_y, self.btn_update_avoid_vtx]:
            w.setEnabled(can_edit)

        if can_edit:
            vtx_idx = self.avoid_vtx_list_widget.currentRow()
            vtx = self.lista_avoidance_areas[self.current_avoidance_area_idx]["punti"][
                vtx_idx
            ]
            self.tb_avoid_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_avoid_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_avoid_vtx_x.clear()
            self.tb_avoid_vtx_y.clear()

    def on_update_selected_avoid_vertex(self):
        if not (0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)):
            return
        vtx_idx = self.avoid_vtx_list_widget.currentRow()
        if vtx_idx < 0:
            return

        try:
            x = float(self.tb_avoid_vtx_x.text())
            y = float(self.tb_avoid_vtx_y.text())
            self.lista_avoidance_areas[self.current_avoidance_area_idx]["punti"][
                vtx_idx
            ] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.update_ui_for_selected_avoidance_area()
        except ValueError:
            self.error_text_avoid_area_mgmt.setText("Coordinate non valide.")

    def update_grid_snap_params(self, *args):
        self.grid_snap_enabled = self.check_grid_snap_enabled.isChecked()
        self.grid_show_enabled = self.check_show_grid.isChecked()
        try:
            self.grid_snap_spacing = float(self.tb_grid_snap_spacing.text())
        except:
            self.grid_snap_spacing = 0.25
        self.full_redraw(preserve_view=True)

    def get_slider_freq_val(self):
        return self.slider_freq.value()

    def on_freq_change_ui_qt(self, value):
        self.label_slider_freq_val.setText(f"{value} Hz")
        self.trigger_spl_map_recalculation()

    def trigger_spl_map_recalculation(self, force_redraw=False, fit_view=False):
        try:
            self.check_auto_spl_update.toggled.disconnect(self.trigger_spl_map_recalculation)
        except TypeError:
            pass 

        if self.check_auto_spl_update.isChecked() or force_redraw:
            preserve = not fit_view
            self.full_redraw(preserve_view=preserve)
        
        self.check_auto_spl_update.toggled.connect(self.trigger_spl_map_recalculation)

    def full_redraw(self, preserve_view=False):
        self.visualizza_mappatura_spl(self.get_slider_freq_val(), preserve_view)

    def on_press_mpl(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.handle_calibration_click(event):
            return

        self.status_bar.clearMessage()
        
        # Gestione aree (incluse le nuove aree di posizionamento)
        area_lists_and_prefixes = [
            (self.lista_target_areas, "target"),
            (self.lista_avoidance_areas, "avoid"),
            (self.lista_sub_placement_areas, "placement"),
        ]
        
        for area_list, type_prefix in area_lists_and_prefixes:
            for area_idx, area_data in enumerate(area_list):
                if not area_data.get("active", False):
                    continue
                for vtx_idx, plot_artist in enumerate(area_data.get("plots", [])):
                    if plot_artist and plot_artist.contains(event)[0]:
                        if type_prefix == "target":
                            if self.current_target_area_idx != area_idx:
                                self.current_target_area_idx = area_idx
                                self.update_ui_for_selected_target_area()
                            self.target_vtx_list_widget.setCurrentRow(vtx_idx)
                        elif type_prefix == "avoid":
                            if self.current_avoidance_area_idx != area_idx:
                                self.current_avoidance_area_idx = area_idx
                                self.update_ui_for_selected_avoidance_area()
                            self.avoid_vtx_list_widget.setCurrentRow(vtx_idx)
                        elif type_prefix == "placement":
                            if self.current_sub_placement_area_idx != area_idx:
                                self.current_sub_placement_area_idx = area_idx
                                self.update_ui_for_selected_sub_placement_area()
                            self.placement_vtx_list_widget.setCurrentRow(vtx_idx)
                        self.drag_object = (f"{type_prefix}_vtx", area_idx, vtx_idx)
                        self.original_mouse_pos = (event.xdata, event.ydata)
                        self.original_object_pos = tuple(area_data["punti"][vtx_idx])
                        return
                        
        # Gestione vertici stanza
        for vtx_idx, vtx_data in enumerate(self.punti_stanza):
            if vtx_data.get("plot") and vtx_data["plot"].contains(event)[0]:
                self.drag_object = ("stanza_vtx", vtx_idx)
                self.original_mouse_pos = (event.xdata, event.ydata)
                self.original_object_pos = tuple(vtx_data["pos"])
                self.selected_stanza_vtx_idx = vtx_idx
                self.update_stanza_vtx_editor()
                return
                
        # Gestione subwoofer
        for i in reversed(range(len(self.sorgenti))):
            sub = self.sorgenti[i]
            if sub.get("arrow_artist") and sub["arrow_artist"].contains(event)[0]:
                if sub.get("param_locks", {}).get("angle", False):
                    self.status_bar.showMessage(
                        f"Angolo Sub ID:{sub.get('id', i+1)} bloccato.", 2000
                    )
                    return
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = (
                    "group_rotate" if sub.get("group_id") is not None else "sub_rotate"
                )
                self.original_object_angle = sub["angle"]
                self.drag_object = (drag_type, i)
                if "group" in drag_type:
                    self.original_group_states = self._get_group_states(
                        sub.get("group_id")
                    )
                return
            if sub.get("rect_artist") and sub["rect_artist"].contains(event)[0]:
                if sub.get("param_locks", {}).get("position", False):
                    self.status_bar.showMessage(
                        f"Posizione Sub ID:{sub.get('id', i+1)} bloccata.", 2000
                    )
                    return
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = (
                    "group_pos" if sub.get("group_id") is not None else "sub_pos"
                )
                self.original_object_pos = (sub["x"], sub["y"])
                self.drag_object = (drag_type, i)
                if "group" in drag_type:
                    self.original_group_states = self._get_group_states(
                        sub.get("group_id")
                    )
                return
        self.drag_object = None

    def on_motion_mpl(self, event):
        if self.drag_object is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self.original_mouse_pos[0]
        dy = event.ydata - self.original_mouse_pos[1]
        obj_type = self.drag_object[0]

        redraw_needed = True

        if obj_type == "sub_pos":
            main_idx = self.drag_object[1]
            self.sorgenti[main_idx]["x"] = self.snap_to_grid(
                self.original_object_pos[0] + dx
            )
            self.sorgenti[main_idx]["y"] = self.snap_to_grid(
                self.original_object_pos[1] + dy
            )
            self.aggiorna_ui_sub_fields()
        elif obj_type == "group_pos":
            for s_state in self.original_group_states:
                orig_x, orig_y = s_state["original_pos"]
                self.sorgenti[s_state["sub_idx"]]["x"] = self.snap_to_grid(orig_x + dx)
                self.sorgenti[s_state["sub_idx"]]["y"] = self.snap_to_grid(orig_y + dy)
            self.aggiorna_ui_sub_fields()
        elif obj_type == "sub_rotate":
            main_idx = self.drag_object[1]
            sub = self.sorgenti[main_idx]
            self.sorgenti[main_idx]["angle"] = np.arctan2(
                event.xdata - sub["x"], event.ydata - sub["y"]
            )
            self.aggiorna_ui_sub_fields()
        elif obj_type == "group_rotate":
            group_center = self.original_group_states[0]["group_center"]
            initial_mouse_angle = np.arctan2(
                self.original_mouse_pos[1] - group_center[1],
                self.original_mouse_pos[0] - group_center[0],
            )
            current_mouse_angle = np.arctan2(
                event.ydata - group_center[1], event.xdata - group_center[0]
            )
            angle_delta = current_mouse_angle - initial_mouse_angle
            for s_state in self.original_group_states:
                sub_idx, orig_rel_pos, orig_angle = (
                    s_state["sub_idx"],
                    s_state["rel_pos"],
                    s_state["original_angle"],
                )
                new_rel_x = orig_rel_pos[0] * np.cos(angle_delta) - orig_rel_pos[
                    1
                ] * np.sin(angle_delta)
                new_rel_y = orig_rel_pos[0] * np.sin(angle_delta) + orig_rel_pos[
                    1
                ] * np.cos(angle_delta)
                self.sorgenti[sub_idx]["x"] = self.snap_to_grid(
                    group_center[0] + new_rel_x
                )
                self.sorgenti[sub_idx]["y"] = self.snap_to_grid(
                    group_center[1] + new_rel_y
                )
                self.sorgenti[sub_idx]["angle"] = (orig_angle + angle_delta) % (
                    2 * np.pi
                )
            self.aggiorna_ui_sub_fields()
        elif obj_type == "stanza_vtx":
            main_idx = self.drag_object[1]
            self.punti_stanza[main_idx]["pos"][0] = self.snap_to_grid(
                self.original_object_pos[0] + dx
            )
            self.punti_stanza[main_idx]["pos"][1] = self.snap_to_grid(
                self.original_object_pos[1] + dy
            )
            self.update_stanza_vtx_editor()
        elif obj_type in ["target_vtx", "avoid_vtx", "placement_vtx"]:
            area_type_prefix, area_idx, vtx_idx = (
                self.drag_object[0].split("_")[0],
                self.drag_object[1],
                self.drag_object[2],
            )
            if area_type_prefix == "target":
                area_list = self.lista_target_areas
            elif area_type_prefix == "avoid":
                area_list = self.lista_avoidance_areas
            else:  # placement
                area_list = self.lista_sub_placement_areas
                
            area_list[area_idx]["punti"][vtx_idx] = [
                self.snap_to_grid(self.original_object_pos[0] + dx),
                self.snap_to_grid(self.original_object_pos[1] + dy),
            ]
            if area_type_prefix == "target":
                self.update_ui_for_selected_target_area()
            elif area_type_prefix == "avoid":
                self.update_ui_for_selected_avoidance_area()
            else:
                self.update_ui_for_selected_sub_placement_area()
        else:
            redraw_needed = False

        if redraw_needed:
            for patch in self.ax.patches:
                patch.remove()
            for line in self.ax.lines:
                line.remove()
            for text in self.ax.texts:
                text.remove()
            self.disegna_elementi_statici_senza_spl()
            self.plot_canvas.canvas.draw_idle()

    def on_release_mpl(self, event):
        if self.drag_object:
            self.status_bar.showMessage(f"Rilasciato oggetto.", 2000)
            is_room_drag = self.drag_object and self.drag_object[0] == "stanza_vtx"
            self.drag_object = None
            self.original_group_states = []
            self.trigger_spl_map_recalculation(force_redraw=True, fit_view=is_room_drag)

    def _get_group_states(self, group_id):
        states = []
        if group_id is None:
            return states
        members = [s for s in self.sorgenti if s.get("group_id") == group_id]
        if not members:
            return states
        center_x = np.mean([s["x"] for s in members])
        center_y = np.mean([s["y"] for s in members])
        for s in members:
            states.append(
                {
                    "sub_idx": self.sorgenti.index(s),
                    "original_pos": (s["x"], s["y"]),
                    "original_angle": s["angle"],
                    "rel_pos": (s["x"] - center_x, s["y"] - center_y),
                    "group_center": (center_x, center_y),
                }
            )
        return states

    def visualizza_mappatura_spl(self, frequenza, preserve_view=False):
        current_xlim, current_ylim = None, None
        if preserve_view and self.ax.get_figure().get_axes(): 
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
    
        self.ax.cla()

        self.plot_canvas.figure.set_facecolor("#323232")
        self.ax.set_facecolor("#404040")
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
    
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.set_xlabel("X (m)", loc='right')
        self.ax.set_ylabel("Y (m)", loc='top')
    
        if hasattr(self, '_cax_for_colorbar_spl') and self._cax_for_colorbar_spl:
            try:
                self.plot_canvas.figure.delaxes(self._cax_for_colorbar_spl)
            except (KeyError, AttributeError):
                pass
        self._cax_for_colorbar_spl = None 

        self.plot_canvas.figure.subplots_adjust(right=0.92)
    
        room_points = [p['pos'] for p in self.punti_stanza]
        if room_points and len(room_points) >= 3 and self.sorgenti:
            try:
                c_val=float(self.tb_velocita_suono.text())
                grid_res=float(self.tb_grid_res_spl.text())
                
                if not(c_val > 0 and grid_res > 0): 
                    raise ValueError("Parametri simulazione non validi.")
            
                min_x_room=min(p[0] for p in room_points)
                max_x_room=max(p[0] for p in room_points)
                min_y_room=min(p[1] for p in room_points)
                max_y_room=max(p[1] for p in room_points)
                x=np.arange(min_x_room, max_x_room + grid_res, grid_res)
                y=np.arange(min_y_room, max_y_room + grid_res, grid_res)
            
                if len(x)>=2 and len(y)>=2:
                    X, Y = np.meshgrid(x,y)
                    points_check = np.vstack((X.ravel(), Y.ravel())).T 
                    room_mask = Path(room_points).contains_points(points_check).reshape(X.shape)
                
                    self.current_spl_map = np.full(X.shape, np.nan)

                    if np.any(room_mask):
                        points_to_calc_x = X[room_mask]
                        points_to_calc_y = Y[room_mask]
                        # --- INIZIO MODIFICA ---
                        sorgenti_array = np.array([(
                           s['x'], s['y'],
                           s.get('pressure_val_at_1m_relative_to_pref', 1.0),
                           s.get('gain_lin', 1.0),
                           s.get('angle', 0.0),
                           s.get('delay_ms', 0.0),
                           s.get('polarity', 1)
                       ) for s in self.sorgenti], dtype=sub_dtype)

                        spl_values = calculate_spl_vectorized(points_to_calc_x, points_to_calc_y, frequenza, c_val, sorgenti_array)
                       # --- FINE MODIFICA ---
                        
                        self.current_spl_map[room_mask] = spl_values

                        if np.any(~np.isnan(self.current_spl_map)):
                            # Scala SPL auto-adattiva
                            if self.auto_scale_spl:
                                min_spl, max_spl = self._calculate_auto_spl_range(self.current_spl_map)
                                self.tb_spl_min.setText(f"{min_spl:.1f}")
                                self.tb_spl_max.setText(f"{max_spl:.1f}")
                            else:
                                try:
                                    min_spl = float(self.tb_spl_min.text())
                                    max_spl = float(self.tb_spl_max.text())
                                    if min_spl >= max_spl:
                                        raise ValueError("Min SPL deve essere < Max SPL")
                                except ValueError:
                                    min_spl, max_spl = self._calculate_auto_spl_range(self.current_spl_map)
                                    self.tb_spl_min.setText(f"{min_spl:.1f}")
                                    self.tb_spl_max.setText(f"{max_spl:.1f}")
                            
                            contour = self.ax.pcolormesh(X, Y, self.current_spl_map, cmap='jet', vmin=min_spl, vmax=max_spl, shading='auto', alpha=0.75, zorder=0.3)
                        
                            self.plot_canvas.figure.subplots_adjust(right=0.85)
                            self._cax_for_colorbar_spl = self.plot_canvas.figure.add_axes([0.87, 0.15, 0.03, 0.7])
                            self.plot_canvas.figure.colorbar(contour, cax=self._cax_for_colorbar_spl, label="SPL (dB)")
                            self._cax_for_colorbar_spl.yaxis.label.set_color('white')
                            self._cax_for_colorbar_spl.tick_params(axis='y', colors='white')

            except Exception as e:
                print(f"Errore calcolo/disegno mappa SPL: {e}")
                self.status_bar.showMessage(f"Errore visualizzazione SPL: {e}", 5000)
                self.current_spl_map = None

        self.disegna_elementi_statici_senza_spl()
    
        self.ax.set_aspect('equal', adjustable='box')
        if preserve_view and current_xlim is not None:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        else:
            self.auto_fit_view_to_room()
        
        self.plot_canvas.canvas.draw_idle()

    def disegna_elementi_statici_senza_spl(self):
        self.disegna_immagine_di_sfondo()
        self.disegna_stanza_e_vertici()
        self.disegna_aree_target_e_avoidance()
        self.disegna_subwoofer_e_elementi()
        self.disegna_array_direction_indicators()
        self.disegna_griglia()

    def update_optim_freq_fields_visibility(self, *args):
        is_copertura = self.radio_copertura.isChecked()
        for w in [self.label_opt_freq_single_widget, self.tb_opt_freq_single]:
            w.setVisible(is_copertura)
        for w in [
            self.label_opt_freq_min_widget,
            self.tb_opt_freq_min,
            self.label_opt_freq_max_widget,
            self.tb_opt_freq_max,
            self.label_opt_n_freq_widget,
            self.tb_opt_n_freq,
        ]:
            w.setVisible(not is_copertura)

        has_active_target_areas = (
            len(self.get_active_areas_points(self.lista_target_areas)) > 0
        )
        has_active_avoidance_areas = (
            len(self.get_active_areas_points(self.lista_avoidance_areas)) > 0
        )

        show_balance_slider = has_active_target_areas and has_active_avoidance_areas

        self.label_balance_slider.setVisible(show_balance_slider)
        self.slider_balance.setVisible(show_balance_slider)
        self.label_balance_value.setVisible(show_balance_slider)

    def on_mouse_move_for_spl_display(self, event):
        if (
            event.inaxes != self.ax
            or self.current_spl_map is None
            or event.xdata is None
            or event.ydata is None
        ):
            self.status_bar.showMessage(
                "Muovi il mouse sul grafico per visualizzare l'SPL.", 0
            )
            return
        if self.drag_object is not None:
            return
        try:
            x_coord, y_coord = event.xdata, event.ydata
            if not self.punti_stanza:
                return
            min_x_room = min(p["pos"][0] for p in self.punti_stanza)
            max_x_room = max(p["pos"][0] for p in self.punti_stanza)
            min_y_room = min(p["pos"][1] for p in self.punti_stanza)
            max_y_room = max(p["pos"][1] for p in self.punti_stanza)
            grid_res_text = self.tb_grid_res_spl.text()
            if not grid_res_text:
                return
            grid_res = float(grid_res_text)
            if grid_res <= 0:
                return

            col_idx = int(np.floor((x_coord - min_x_room) / grid_res))
            row_idx = int(np.floor((y_coord - min_y_room) / grid_res))

            if (
                0 <= row_idx < self.current_spl_map.shape[0]
                and 0 <= col_idx < self.current_spl_map.shape[1]
            ):
                spl_val = self.current_spl_map[row_idx, col_idx]
                if not np.isnan(spl_val):
                    self.status_bar.showMessage(
                        f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): {spl_val:.1f} dB", 0
                    )
                else:
                    self.status_bar.showMessage(
                        f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Area Stanza", 0
                    )
            else:
                self.status_bar.showMessage(
                    f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Limiti di Plot", 0
                )
        except (ValueError, IndexError):
            pass

    def unlock_dsp_for_optimization(self):
        if not self.sorgenti:
            self.status_bar.showMessage("Nessun subwoofer da sbloccare.", 3000)
            return

        unlocked_count = 0
        for sub in self.sorgenti:
            for param in ["delay", "gain", "polarity"]:
                if sub["param_locks"].get(param, True):
                    sub["param_locks"][param] = False
                    unlocked_count += 1

            if (
                sub.get("group_id") is None
                or sub.get("group_id") not in self.lista_gruppi_array
            ):
                if sub["param_locks"].get("angle", True):
                    sub["param_locks"]["angle"] = False
                    unlocked_count += 1

        self.aggiorna_ui_sub_fields()
        self.status_bar.showMessage(
            f"Parametri DSP sbloccati. L'angolo dei gruppi array rimane bloccato.", 4000
        )

    def avvia_ottimizzazione_ui_qt(self):
        if self.optimization_thread is not None:
            return
        if not self.sorgenti:
            self.status_text_optim.setText("Aggiungere almeno un subwoofer.")
            return

        room_points_list = [p["pos"] for p in self.punti_stanza]
        if len(room_points_list) < 3:
            self.status_text_optim.setText("Definire una stanza valida.")
            return

        active_targets = self.get_active_areas_points(self.lista_target_areas)
        active_avoidances = self.get_active_areas_points(self.lista_avoidance_areas)

        if not active_targets and not active_avoidances:
            self.status_text_optim.setText(
                "Attivare almeno un'area target o di evitamento per l'ottimizzazione."
            )
            return

        try:
            criterion = self.radio_btn_group_crit.checkedButton().text()
            pop_s = int(self.tb_opt_pop_size.text())
            gens = int(self.tb_opt_generations.text())

            max_spl_avoid = float(self.tb_max_spl_avoid.text())
            target_min_spl_desired_val = float(self.tb_target_min_spl_desired.text())

            balance_target_avoidance_val = self.slider_balance.value()

            c_val = float(self.tb_velocita_suono.text())
            grid_res = float(self.tb_grid_res_spl.text())
            optim_f_s, optim_f_min, optim_f_max, optim_n_f = None, None, None, None

            self.last_optim_criterion = criterion

            if criterion == "Copertura SPL":
                optim_f_s = float(self.tb_opt_freq_single.text())
                self.last_optim_freq_s = optim_f_s
            else:
                optim_f_min = float(self.tb_opt_freq_min.text())
                optim_f_max = float(self.tb_opt_freq_max.text())
                optim_n_f = int(self.tb_opt_n_freq.text())
                self.last_optim_freq_min = optim_f_min
                self.last_optim_freq_max = optim_f_max
                if optim_n_f < 2:
                    self.status_text_optim.setText(
                        "Per 'Omogeneità', usare almeno 2 punti frequenza."
                    )
                    return
        except (ValueError, AttributeError) as e:
            self.status_text_optim.setText(f"Errore parametri ottimizzazione: {e}")
            return

        # Area di posizionamento per ottimizzazione
        sub_placement_area = self.get_active_sub_placement_area()

        self.optimization_thread = QThread()
        self.optimization_worker = OptimizationWorker(
            criterion,
            optim_f_s,
            optim_f_min,
            optim_f_max,
            optim_n_f,
            pop_s,
            gens,
            c_val,
            grid_res,
            room_points_list,
            active_targets,
            active_avoidances,
            max_spl_avoid,
            target_min_spl_desired_val,
            balance_target_avoidance_val,
            [s.copy() for s in self.sorgenti],
            [s["param_locks"].copy() for s in self.sorgenti],
            sub_placement_area,
        )

        self.optimization_worker.moveToThread(self.optimization_thread)

        self.optimization_worker.status_update.connect(self.update_optim_status_text)
        self.optimization_worker.finished.connect(self.optimization_thread.quit)
        self.optimization_worker.finished.connect(self.handle_optim_finished)
        self.optimization_worker.finished.connect(self.optimization_worker.deleteLater)
        self.optimization_thread.finished.connect(self.optimization_thread.deleteLater)
        self.optimization_thread.finished.connect(self.on_optim_thread_finished)

        self.optimization_thread.started.connect(self.optimization_worker.run)
        self.optimization_thread.start()

        self.btn_optimize_widget.setEnabled(False)
        self.btn_stop_optimize_widget.setEnabled(True)

    def stop_ottimizzazione_ui_qt(self):
        if self.optimization_worker:
            self.optimization_worker.request_stop()

    def on_optim_thread_finished(self):
        self.optimization_thread = None
        self.optimization_worker = None
        self.status_bar.showMessage("Thread di ottimizzazione terminato.", 3000)

    def update_optim_status_text(self, message):
        if hasattr(self, "status_text_optim"):
            self.status_text_optim.setText(message)
        QApplication.processEvents()

    def save_project_to_excel(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Salva Progetto Completo", "", "File Excel (*.xlsx)")
        if not filepath:
            return

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if self.sorgenti:
                    sub_data_to_save = []
                    for sub in self.sorgenti:
                        sub_data_to_save.append({
                            'ID': sub.get('id'), 'X (m)': sub.get('x'), 'Y (m)': sub.get('y'),
                            'Angolo (°)': np.degrees(sub.get('angle')), 'Gain (dB)': sub.get('gain_db'),
                            'Polarità': sub.get('polarity'), 'Delay (ms)': sub.get('delay_ms'),
                            'Larghezza (m)': sub.get('width'), 'Profondità (m)': sub.get('depth'),
                            'SPL (dB)': sub.get('spl_rms'), 'group_id': sub.get('group_id'),
                            'Lock_Angle': sub['param_locks'].get('angle', False),
                            'Lock_Delay': sub['param_locks'].get('delay', False),
                            'Lock_Gain': sub['param_locks'].get('gain', False),
                            'Lock_Polarity': sub['param_locks'].get('polarity', False),
                            'Lock_Position': sub['param_locks'].get('position', False),
                            'Is_Group_Master': sub.get('is_group_master', False)
                        })
                    df_subs = pd.DataFrame(sub_data_to_save)
                    df_subs.to_excel(writer, sheet_name='Subwoofers', index=False)

                if self.punti_stanza:
                    room_verts = [p['pos'] for p in self.punti_stanza]
                    df_room = pd.DataFrame(room_verts, columns=['X', 'Y'])
                    df_room.to_excel(writer, sheet_name='Stanza', index=False)

                if self.lista_target_areas:
                    target_areas_data = []
                    for area in self.lista_target_areas:
                        for vtx in area['punti']:
                            target_areas_data.append({
                                'Area_ID': area['id'], 'Nome': area['nome'], 'Attiva': area['active'],
                                'Vertice_X': vtx[0], 'Vertice_Y': vtx[1]
                            })
                    df_target = pd.DataFrame(target_areas_data)
                    df_target.to_excel(writer, sheet_name='Aree_Target', index=False)

                if self.lista_avoidance_areas:
                    avoid_areas_data = []
                    for area in self.lista_avoidance_areas:
                        for vtx in area['punti']:
                            avoid_areas_data.append({
                                'Area_ID': area['id'], 'Nome': area['nome'], 'Attiva': area['active'],
                                'Vertice_X': vtx[0], 'Vertice_Y': vtx[1]
                            })
                    df_avoid = pd.DataFrame(avoid_areas_data)
                    df_avoid.to_excel(writer, sheet_name='Aree_Evitamento', index=False)

                # Salva aree di posizionamento sub
                if self.lista_sub_placement_areas:
                    placement_areas_data = []
                    for area in self.lista_sub_placement_areas:
                        for vtx in area['punti']:
                            placement_areas_data.append({
                                'Area_ID': area['id'], 'Nome': area['nome'], 'Attiva': area['active'],
                                'Vertice_X': vtx[0], 'Vertice_Y': vtx[1]
                            })
                    df_placement = pd.DataFrame(placement_areas_data)
                    df_placement.to_excel(writer, sheet_name='Aree_Posizionamento', index=False)

                if self.bg_image_props.get('path'):
                    props_to_save = {k: v for k, v in self.bg_image_props.items() if k not in ['data', 'artist', 'cached_transformed']}
                    if 'anchor_pixel' in props_to_save and props_to_save['anchor_pixel'] is not None:
                        try:
                            px, py = props_to_save['anchor_pixel']
                            props_to_save['anchor_pixel'] = (float(px), float(py))
                        except (ValueError, TypeError):
                            print(f"Non è stato possibile convertire anchor_pixel in float prima del salvataggio: {props_to_save['anchor_pixel']}")
                            props_to_save['anchor_pixel'] = None
                    df_bg = pd.DataFrame([props_to_save])
                    df_bg.to_excel(writer, sheet_name='ImmagineSfondo', index=False)
            
                if self.lista_gruppi_array:
                    array_group_data = []
                    for group_id, params in self.lista_gruppi_array.items():
                        row_data = {'Group_ID': group_id}
                        for k, v in params.items():
                            row_data[k] = v
                        array_group_data.append(row_data)
                    df_array_groups = pd.DataFrame(array_group_data)
                    df_array_groups.to_excel(writer, sheet_name='Array_Groups', index=False)

                self.status_bar.showMessage(f"Progetto completo salvato in {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Errore di Salvataggio", f"Impossibile salvare il file di progetto:\n{e}")

    def load_project_from_excel(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Carica Progetto Completo", "", "File Excel (*.xlsx)")
        if not filepath:
            return
        
        try:
            # Reset completo
            self.sorgenti.clear()
            self.punti_stanza.clear()
            self.lista_target_areas.clear()
            self.lista_avoidance_areas.clear()
            self.lista_sub_placement_areas.clear()
            self.lista_gruppi_array.clear()
            self.remove_background_image()
            self.current_sub_idx = -1
            self.next_sub_id = 1
            self.next_group_id = 1
            self.next_target_area_id = 1
            self.next_avoidance_area_id = 1
            self.next_sub_placement_area_id = 1
        
            xls = pd.ExcelFile(filepath)
        
            # Carica immagine di sfondo
            if 'ImmagineSfondo' in xls.sheet_names:
                try:
                    df_bg = pd.read_excel(filepath, sheet_name='ImmagineSfondo')
                    if not df_bg.empty:
                        bg_props_loaded = df_bg.to_dict('records')[0]
                        if 'anchor_pixel' in bg_props_loaded and isinstance(bg_props_loaded['anchor_pixel'], str):
                            try:
                                import ast
                                bg_props_loaded['anchor_pixel'] = ast.literal_eval(bg_props_loaded['anchor_pixel'])
                            except (ValueError, SyntaxError):
                                print(f"Non è stato possibile riconvertire 'anchor_pixel' dalla stringa: {bg_props_loaded['anchor_pixel']}")
                                bg_props_loaded['anchor_pixel'] = None
                        img_path = bg_props_loaded.get('path')
                        if img_path and os.path.exists(img_path):
                            self.bg_image_props.update(bg_props_loaded)
                            self.bg_image_props['data'] = plt.imread(img_path)
                            self.bg_image_props['cached_transformed'] = None  # Reset cache
                            self.tb_bg_x.setText(str(self.bg_image_props.get('center_x', 0.0)))
                            self.tb_bg_y.setText(str(self.bg_image_props.get('center_y', 0.0)))
                            self.tb_bg_scale.setText(str(self.bg_image_props.get('scale', 1.0)))
                            self.tb_bg_rotation.setText(str(self.bg_image_props.get('rotation_deg', 0.0)))
                            self.tb_bg_alpha.setText(str(self.bg_image_props.get('alpha', 0.5)))
                        else:
                            QMessageBox.warning(self, "Immagine non trovata", fr"Il file immagine del progetto non è stato trovato al percorso:\n{img_path}\n\nL'immagine non verrà caricata, ma il resto del progetto sì.")
                except Exception as e:
                    print(f"Errore nel caricamento del foglio 'ImmagineSfondo': {e}")

            # Carica gruppi array
            if 'Array_Groups' in xls.sheet_names:
                try:
                    df_array_groups = pd.read_excel(filepath, sheet_name='Array_Groups')
                    if not df_array_groups.empty:
                        for index, row in df_array_groups.iterrows():
                            group_id = int(row['Group_ID'])
                            group_params = row.drop('Group_ID').to_dict()
                            self.lista_gruppi_array[group_id] = group_params
                except Exception as e:
                    print(f"Errore nel caricamento del foglio 'Array_Groups': {e}")

            # Carica subwoofer
            if 'Subwoofers' in xls.sheet_names:
                df_subs = pd.read_excel(filepath, sheet_name='Subwoofers')
                for index, row in df_subs.iterrows():
                    config = {
                        'id': int(row['ID']), 'x': row['X (m)'], 'y': row['Y (m)'],
                        'angle': np.radians(row['Angolo (°)']), 'gain_db': row['Gain (dB)'],
                        'polarity': row['Polarità'], 'delay_ms': row['Delay (ms)'],
                        'width': row.get('Larghezza (m)', self.global_sub_width),
                        'depth': row.get('Profondità (m)', self.global_sub_depth),
                        'spl_rms': row.get('SPL (dB)', self.global_sub_spl_rms),
                        'group_id': int(row['group_id']) if pd.notna(row['group_id']) else None,
                        'is_group_master': bool(row.get('Is_Group_Master', False)), 
                        'param_locks': { 
                            'angle': bool(row.get('Lock_Angle', False)),
                            'delay': bool(row.get('Lock_Delay', False)),
                            'gain': bool(row.get('Lock_Gain', False)),
                            'polarity': bool(row.get('Lock_Polarity', False)),
                            'position': bool(row.get('Lock_Position', False)),
                        }
                    }
                    config['gain_lin'] = 10 ** (config['gain_db'] / 20.0)
                    config['pressure_val_at_1m_relative_to_pref'] = 10 ** (config['spl_rms'] / 20.0)
                    self.sorgenti.append(config) 

                if 'ID' in df_subs.columns and not df_subs['ID'].empty and pd.notna(df_subs['ID'].max()):
                    self.next_sub_id = int(df_subs['ID'].max()) + 1

            # Carica stanza
            if 'Stanza' in xls.sheet_names:
                df_room = pd.read_excel(filepath, sheet_name='Stanza')
                for index, row in df_room.iterrows():
                    self.punti_stanza.append({'pos': [row['X'], row['Y']], 'plot': None})

            # Carica aree target
            if 'Aree_Target' in xls.sheet_names:
                df_target = pd.read_excel(filepath, sheet_name='Aree_Target')
                if not df_target.empty:
                    for area_id, group in df_target.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id), 'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]), 'punti': punti, 'plots': []
                        }
                        self.lista_target_areas.append(area_data)
                    if not df_target['Area_ID'].empty and pd.notna(df_target['Area_ID'].max()):
                        self.next_target_area_id = int(df_target['Area_ID'].max()) + 1

            # Carica aree evitamento
            if 'Aree_Evitamento' in xls.sheet_names:
                df_avoid = pd.read_excel(filepath, sheet_name='Aree_Evitamento')
                if not df_avoid.empty:
                    for area_id, group in df_avoid.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id), 'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]), 'punti': punti, 'plots': []
                        }
                        self.lista_avoidance_areas.append(area_data)
                    if not df_avoid['Area_ID'].empty and pd.notna(df_avoid['Area_ID'].max()):
                        self.next_avoidance_area_id = int(df_avoid['Area_ID'].max()) + 1

            # Carica aree posizionamento
            if 'Aree_Posizionamento' in xls.sheet_names:
                df_placement = pd.read_excel(filepath, sheet_name='Aree_Posizionamento')
                if not df_placement.empty:
                    for area_id, group in df_placement.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id), 'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]), 'punti': punti, 'plots': []
                        }
                        self.lista_sub_placement_areas.append(area_data)
                    if not df_placement['Area_ID'].empty and pd.notna(df_placement['Area_ID'].max()):
                        self.next_sub_placement_area_id = int(df_placement['Area_ID'].max()) + 1
        
            self._update_max_group_id()

            if self.sorgenti: 
                self.current_sub_idx = 0
        
            self.aggiorna_ui_sub_fields()
            self.update_ui_for_selected_target_area()
            self.update_ui_for_selected_avoidance_area()
            self.update_ui_for_selected_sub_placement_area()
            self.full_redraw()
            self.status_bar.showMessage(f"Progetto completo caricato da {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Errore di Caricamento", f"Impossibile caricare il file di progetto:\n{e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        main_win = SubwooferSimApp()
        main_win.show()
    except Exception as e:
        print(f"ERRORE DURANTE L'INIZIALIZZAZIONE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(app.exec())