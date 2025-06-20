# --- START DIAGNOSTIC CODE ---
import sys
import os
try:
    import matplotlib
    import PyQt6
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
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.transforms as mtransforms

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QSlider, QCheckBox, QRadioButton,
                             QGroupBox, QFormLayout, QGridLayout, QSizePolicy, QStatusBar,
                             QMessageBox, QButtonGroup, QScrollArea, QComboBox, QInputDialog,
                             QListWidget, QListWidgetItem, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

import warnings

# --- Parametri Globali Iniziali ---
PARAM_RANGES = {
    'delay_ms': (0.0, 500.0),
    'gain_db': (-20.0, 6.0),
    'polarity': [-1, 1], # Note: polarity is a list, not a tuple, for np.random.choice
    'spl_rms': (70.0, 140.0),
    'sub_width_depth': (0.1, 2.0),
    'angle': (0.0, 2 * np.pi) # Angle should be a tuple for uniform
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

DEFAULT_ROOM_VERTICES = [ [-15, -15], [15, -15], [15, 15], [-15, 15] ]
P_REF_20UPA_CALC_DEFAULT = 10**(DEFAULT_SUB_SPL_RMS/20.0)

DEFAULT_TARGET_AREA_VERTICES = [ [-1, -1], [1, -1], [1, 1], [-1, 1] ]
DEFAULT_AVOIDANCE_AREA_VERTICES = [ [-4, 0], [-3, 0], [-3, 1], [-4, 1] ]

DEFAULT_MAX_SPL_AVOIDANCE = 65.0
DEFAULT_TARGET_MIN_SPL_DESIRED = 80.0
DEFAULT_BALANCE_SLIDER_VALUE = 50 # Default for 50/50 target/avoidance balance

OPTIMIZATION_MUTATION_RATE = 0.1 # Fixed mutation rate (10%)


FRONT_DIRECTIVITY_BEAMWIDTH_RAD = np.pi / 6
FRONT_DIRECTIVITY_GAIN_LIN = 10**(1.0/20.0)
DEFAULT_SIM_SPEED_OF_SOUND = 343.0

class MatplotlibCanvas(QWidget):
    def __init__(self, parent=None, width=8, height=7, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(); layout.addWidget(self.toolbar); layout.addWidget(self.canvas); self.setLayout(layout)

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
        
        base_amplitude_attenuation = (sub_data.get('pressure_val_at_1m_relative_to_pref', 1.0) * sub_data.get('gain_lin', 1.0)) / distance

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


class OptimizationWorker(QObject):
    status_update = pyqtSignal(str)
    finished = pyqtSignal(object)
    def __init__(self, criterion, optim_freq_s, optim_freq_min_val, optim_freq_max_val, optim_n_freq_val,
                 pop_size, generations, c_val, grid_res_spl, room_pts,
                 active_target_areas_points, active_avoidance_areas_points,
                 max_spl_avoidance, target_min_spl_desired, balance_target_avoidance,
                 current_sorgenti_configs,
                 sorgenti_param_locks_list):
        super().__init__()
        self.criterion, self.optim_freq_s = criterion, optim_freq_s
        self.optim_freq_min_val, self.optim_freq_max_val = optim_freq_min_val, optim_freq_max_val
        self.optim_n_freq_val = optim_n_freq_val
        self.pop_size, self.generations = pop_size, generations
        self.mutation_rate_val = OPTIMIZATION_MUTATION_RATE # Fixed mutation rate
        self.c_val, self.grid_res_spl, self.room_pts = c_val, grid_res_spl, room_pts
        self.active_target_areas_points, self.active_avoidance_areas_points = active_target_areas_points, active_avoidance_areas_points
        
        self.max_spl_avoidance = max_spl_avoidance
        self.target_min_spl_desired = target_min_spl_desired
        self.balance_target_avoidance = balance_target_avoidance # 0-100 value from slider

        self.AVOIDANCE_PENALTY_FACTOR = 1.0 # How much each dB above avoidance threshold counts
        self.TARGET_UNDERSHOOT_PENALTY_FACTOR = 2.0 # How much each dB below target min SPL counts

        self.sorgenti_configs_ref = current_sorgenti_configs
        self.sorgenti_param_locks = sorgenti_param_locks_list
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        self.status_update.emit("Richiesta di interruzione ottimizzazione...")

    def run(self):
        self._stop_requested = False
        if not self.sorgenti_configs_ref:
            self.status_update.emit("Nessun subwoofer da ottimizzare."); self.finished.emit(None); return
        self.status_update.emit(f"Inizializzazione ottimizzazione...")
        QApplication.processEvents()
        population = [self._create_individual_worker() for _ in range(self.pop_size)]
        best_overall_dsp_params, best_overall_fit = None, -float('inf')

        for gen in range(self.generations):
            if self._stop_requested: self.status_update.emit("Ottimizzazione interrotta."); self.finished.emit(best_overall_dsp_params); return
            self.status_update.emit(f"Generazione {gen+1}/{self.generations} - Calcolo Fitness...")
            QApplication.processEvents()
            fit_vals = [self._calculate_fitness_worker(ind) for ind in population]
            if not fit_vals: self.status_update.emit(f"Gen {gen+1}: Fitness non calcolabile."); break
            current_best_idx = np.argmax(fit_vals) if fit_vals else -1
            if current_best_idx != -1 and fit_vals[current_best_idx] > best_overall_fit:
                best_overall_fit = fit_vals[current_best_idx]; best_overall_dsp_params = [p.copy() for p in population[current_best_idx]]
            if not fit_vals or all(f == -float('inf') for f in fit_vals): self.status_update.emit(f"Gen {gen+1}: Nessun individuo valido."); break
            
            valid_indices = [i for i, f in enumerate(fit_vals) if f > -float('inf')]
            if not valid_indices: self.status_update.emit(f"Gen {gen+1}: Nessuna fitness valida."); break
            
            sorted_indices = sorted(valid_indices, key=lambda i: fit_vals[i], reverse=True)
            elite_size = max(1, int(0.1 * self.pop_size))
            new_population = [population[i] for i in sorted_indices[:elite_size]]
            
            tourn_s = 3
            while len(new_population) < self.pop_size:
                parents = []
                for _ in range(2):
                    competitors_indices = np.random.choice(sorted_indices, size=min(tourn_s, len(sorted_indices)), replace=False)
                    winner_idx = competitors_indices[np.argmax([fit_vals[i] for i in competitors_indices])]
                    parents.append(population[winner_idx])
                
                child1, child2 = [], []
                # Crossover logic needs to handle each sub's DSP parameters individually
                for p1_sub, p2_sub in zip(parents[0], parents[1]):
                    c1_sub, c2_sub = {}, {}
                    # Always copy the original sub_idx for tracking locks
                    c1_sub['sub_idx_original'] = p1_sub['sub_idx_original']
                    c2_sub['sub_idx_original'] = p2_sub['sub_idx_original']

                    for key in p1_sub:
                        if key == 'sub_idx_original': # Skip the tracking index
                            continue
                        
                        # Apply crossover for actual DSP parameters
                        if np.random.rand() < 0.5:
                            c1_sub[key], c2_sub[key] = p1_sub[key], p2_sub[key]
                        else:
                            c1_sub[key], c2_sub[key] = p2_sub[key], p1_sub[key]
                        
                        # Recalculate gain_lin if gain_db was crossed over
                        if key == 'gain_db':
                            c1_sub['gain_lin'] = 10**(c1_sub['gain_db']/20.0)
                            c2_sub['gain_lin'] = 10**(c2_sub['gain_db']/20.0)
                    child1.append(c1_sub)
                    child2.append(c2_sub)
                
                new_population.append(self._mutate_worker(child1));
                if len(new_population) < self.pop_size: new_population.append(self._mutate_worker(child2))
            population = new_population

        final_fitness_display = f"{best_overall_fit:.3f}" if best_overall_fit > -float('inf') else "N/A"
        final_message = f"Terminata. Miglior fitness: {final_fitness_display}."
        self.status_update.emit(final_message); self.finished.emit(best_overall_dsp_params)

    def _create_individual_worker(self):
        individual = []
        for i, sub_ref in enumerate(self.sorgenti_configs_ref):
            locks = self.sorgenti_param_locks[i]
            dsp = {'sub_idx_original': i} # Store original index for lock lookup
            
            # Angle
            if not locks.get('angle', False):
                dsp['angle'] = np.random.uniform(*PARAM_RANGES['angle'])
            else:
                dsp['angle'] = sub_ref['angle']

            # Delay
            if not locks.get('delay', False):
                dsp['delay_ms'] = np.random.uniform(*PARAM_RANGES['delay_ms'])
            else:
                dsp['delay_ms'] = sub_ref['delay_ms']
            
            # Gain
            if not locks.get('gain', False):
                dsp['gain_db'] = np.random.uniform(*PARAM_RANGES['gain_db'])
            else:
                dsp['gain_db'] = sub_ref['gain_db']
            
            # Polarity
            if not locks.get('polarity', False):
                dsp['polarity'] = np.random.choice(PARAM_RANGES['polarity'])
            else:
                dsp['polarity'] = sub_ref['polarity']
            
            dsp['gain_lin'] = 10**(dsp['gain_db']/20.0)
            individual.append(dsp)
        return individual

    def _mutate_worker(self, individual):
        mutated = []
        rate = self.mutation_rate_val
        for i, sub_dsp in enumerate(individual):
            new_dsp = sub_dsp.copy()
            locks = self.sorgenti_param_locks[sub_dsp['sub_idx_original']] # Use original index for locks
            
            if not locks.get('delay', False) and np.random.rand() < rate: 
                new_dsp['delay_ms'] = np.clip(new_dsp['delay_ms'] + np.random.normal(0, 5), *PARAM_RANGES['delay_ms'])
            
            if not locks.get('gain', False) and np.random.rand() < rate: 
                new_dsp['gain_db'] = np.clip(new_dsp['gain_db'] + np.random.normal(0, 1), *PARAM_RANGES['gain_db'])
                new_dsp['gain_lin'] = 10**(new_dsp['gain_db']/20.0)
            
            if not locks.get('polarity', False) and np.random.rand() < rate: 
                new_dsp['polarity'] *= -1
            
            if not locks.get('angle', False) and np.random.rand() < rate: # Mutate angle if not locked
                new_dsp['angle'] = (new_dsp['angle'] + np.random.normal(0, np.radians(5))) % (2 * np.pi) # Smaller mutation step for angle
                
            mutated.append(new_dsp)
        return mutated

    def _calculate_spl_map_for_fitness_evaluation(self, individual_dsp_params, freq, area_points, is_avoidance=False):
        configs = []
        for i, sub_ref in enumerate(self.sorgenti_configs_ref):
            conf = sub_ref.copy()
            # Find the DSP parameters for the current subwoofer in the individual
            # Use original index for robust lookup
            dsp_params_for_this_sub = next((dsp for dsp in individual_dsp_params if dsp.get('sub_idx_original') == i), None)
            if dsp_params_for_this_sub:
                # Apply only relevant DSP parameters (angle, delay_ms, gain_db, polarity, gain_lin)
                conf['angle'] = dsp_params_for_this_sub.get('angle', conf['angle'])
                conf['delay_ms'] = dsp_params_for_this_sub.get('delay_ms', conf['delay_ms'])
                conf['gain_db'] = dsp_params_for_this_sub.get('gain_db', conf['gain_db'])
                conf['gain_lin'] = dsp_params_for_this_sub.get('gain_lin', conf['gain_lin'])
                conf['polarity'] = dsp_params_for_this_sub.get('polarity', conf['polarity'])
            
            configs.append(conf)
        
        if not area_points:
            room_points_list = [p['pos'] for p in self.room_pts]
            if not is_avoidance: area_points = [room_points_list]
            else: return None, None

        min_x=min(p[0] for points in area_points for p in points); max_x=max(p[0] for points in area_points for p in points)
        min_y=min(p[1] for points in area_points for p in points); max_y=max(p[1] for points in area_points for p in points)
        x_c=np.arange(min_x,max_x+self.grid_res_spl,self.grid_res_spl); y_c=np.arange(min_y,max_y+self.grid_res_spl,self.grid_res_spl)
        if len(x_c)<2 or len(y_c)<2: return None,None
        
        X,Y=np.meshgrid(x_c,y_c); points_check = np.vstack((X.ravel(), Y.ravel())).T
        mask = np.zeros_like(X, dtype=bool)
        for points in area_points:
            if len(points) >= 3: mask |= Path(points).contains_points(points_check).reshape(X.shape)
        
        if not np.any(mask): return None, None
        
        spl_map = np.full(X.shape, np.nan)
        points_to_calc_x = X[mask]
        points_to_calc_y = Y[mask]
        
        if freq is None or self.c_val is None:
            return np.full(X.shape, -np.inf), mask 

        spl_values = calculate_spl_vectorized(points_to_calc_x, points_to_calc_y, freq, self.c_val, configs)
        spl_map[mask] = spl_values
        
        return spl_map, mask

    def _calculate_fitness_worker(self, individual):
        score_target_component = 0.0 # Base per il contributo del target
        score_avoidance_penalty = 0.0 # Base per la penalità di evitamento

        # Caso iniziale: Se non ci sono né aree target attive né aree di evitamento attive, l'ottimizzazione non ha un obiettivo.
        if not self.active_target_areas_points and not self.active_avoidance_areas_points:
            return -float('inf') 
        
        ### LOGICA FITNESS AGGIORNATA: CALCOLO AREA TARGET ###
        if self.active_target_areas_points:
            freq_for_level_target = self.optim_freq_s if self.criterion == 'Copertura SPL' else (self.optim_freq_min_val + self.optim_freq_max_val) / 2.0

            spl_map_target, mask_target = self._calculate_spl_map_for_fitness_evaluation(
                individual, freq_for_level_target, self.active_target_areas_points
            )
            
            if spl_map_target is not None:
                values_target = spl_map_target[mask_target & ~np.isnan(spl_map_target)]
                
                if values_target.size > 0:
                    mean_target_spl = np.mean(values_target)
                    min_target_spl = np.min(values_target)

                    base_target_score = 0.0
                    if self.criterion == 'Copertura SPL':
                        base_target_score = mean_target_spl
                    elif self.criterion == 'Omogeneità SPL':
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            freqs = np.logspace(np.log10(self.optim_freq_min_val), np.log10(self.optim_freq_max_val), self.optim_n_freq_val)
                            maps_for_homogeneity = [self._calculate_spl_map_for_fitness_evaluation(individual, f, self.active_target_areas_points)[0] for f in freqs]
                            
                            valid_homogeneity_maps = [m for m in maps_for_homogeneity if m is not None and np.any(~np.isnan(m))]
                            
                            if valid_homogeneity_maps:
                                stacked_homogeneity_maps = np.stack(valid_homogeneity_maps)
                                if stacked_homogeneity_maps.shape[0] >= 2:
                                    variations = np.nanstd(stacked_homogeneity_maps, axis=0)
                                else:
                                    variations = np.full(valid_homogeneity_maps[0].shape, 0.0)
                                
                                base_target_score = -np.nanmean(variations) if not np.all(np.isnan(variations)) else -float('inf')
                            else:
                                base_target_score = -float('inf')

                    score_target_component = base_target_score # Fixed base weight

                    if min_target_spl < self.target_min_spl_desired:
                        undershoot_amount = (self.target_min_spl_desired - min_target_spl)
                        score_target_component -= (undershoot_amount * self.TARGET_UNDERSHOOT_PENALTY_FACTOR)
                else:
                    score_target_component = -float('inf')
            else:
                score_target_component = -float('inf')


        ### LOGICA FITNESS AGGIORNATA: CALCOLO AREA DI EVITAMENTO ###
        if self.active_avoidance_areas_points:
            freq_avoid = self.optim_freq_s if self.criterion == 'Copertura SPL' else (self.optim_freq_min_val + self.optim_freq_max_val) / 2.0
            
            spl_map_avoid, mask_avoid = self._calculate_spl_map_for_fitness_evaluation(
                individual, freq_avoid, self.active_avoidance_areas_points, is_avoidance=True
            )
            
            if spl_map_avoid is not None:
                values_avoid = spl_map_avoid[mask_avoid & ~np.isnan(spl_map_avoid)]
                
                if values_avoid.size > 0:
                    mean_spl_avoid = np.mean(values_avoid)
                    if mean_spl_avoid > self.max_spl_avoidance:
                        penalty = (mean_spl_avoid - self.max_spl_avoidance) * self.AVOIDANCE_PENALTY_FACTOR
                        score_avoidance_penalty = penalty
        
        # FITNESS FINALE: Bilanciamento tra target e avoidance
        if self.active_target_areas_points and self.active_avoidance_areas_points:
            balance_norm = self.balance_target_avoidance / 100.0 # Normalizza da 0 a 1
            total_fitness = (score_target_component * balance_norm) - (score_avoidance_penalty * (1 - balance_norm))
        elif self.active_target_areas_points:
            total_fitness = score_target_component
        elif self.active_avoidance_areas_points:
            total_fitness = -score_avoidance_penalty 
        else:
            total_fitness = -float('inf')
            
        return total_fitness if not np.isnan(total_fitness) else -float('inf')


class SubwooferSimApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulatore Posizionamento Subwoofer Avanzato")
        self.setGeometry(50, 50, 1600, 1000)

        # Stato dell'applicazione
        self.punti_stanza = []
        for p in DEFAULT_ROOM_VERTICES:
            self.punti_stanza.append({'pos': p, 'plot': None})

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
        
        self.selected_stanza_vtx_idx = -1
        self.drag_object = None
        self.original_mouse_pos = None
        self.original_object_pos = None
        self.original_object_angle = None
        self.original_group_states = []
        
        self.current_spl_map = None
        self._cax_for_colorbar_spl = None
                                
        self.optimization_thread = None
        self.optimization_worker = None
        
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

        # --- INIZIO NUOVO CODICE ---
        # Memorizza gli ultimi parametri usati per l'ottimizzazione
        self.last_optim_criterion = None
        self.last_optim_freq_s = None
        self.last_optim_freq_min = None
        self.last_optim_freq_max = None
        # --- FINE NUOVO CODICE ---


        self._setup_ui()
        
        if self.sorgenti: self.current_sub_idx = 0
        
        self.auto_fit_view_to_room()
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()
        self.update_optim_freq_fields_visibility()
        # _update_array_ui() is now called within _setup_group_array_ui

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
            border: 2px solid gray;
            border-radius: 5px;
            margin-top: 1ex; /* leave space at the top for the title */
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center; /* position at the top center */
            padding: 0 3px;
            color: lightgray;
        }
        """
        self.central_widget.setStyleSheet(group_box_style)


        self._setup_project_ui()
        self._setup_stanza_ui()
        self._setup_global_sub_ui()
        self._setup_sub_config_ui()
        self._setup_group_array_ui() # Nuova sezione unificata
        self._setup_target_areas_ui()
        self._setup_avoidance_areas_ui()
        self._setup_spl_vis_ui()
        self._setup_sim_grid_ui()
        self._setup_optimization_ui()
        
        self.controls_layout.addStretch(1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Applicazione caricata.")
        self.plot_canvas.canvas.mpl_connect('button_press_event', self.on_press_mpl)
        self.plot_canvas.canvas.mpl_connect('motion_notify_event', self.on_motion_mpl)
        self.plot_canvas.canvas.mpl_connect('button_release_event', self.on_release_mpl)
        self.plot_canvas.canvas.mpl_connect('motion_notify_event', self.on_mouse_move_for_spl_display)


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
        self.btn_update_stanza_vtx.clicked.connect(self.on_update_selected_stanza_vertex)
        stanza_layout.addRow(self.btn_update_stanza_vtx)
        self.error_text_stanza_vtx_edit = QLabel("")
        self.error_text_stanza_vtx_edit.setStyleSheet("color: red;")
        stanza_layout.addRow(self.error_text_stanza_vtx_edit)
        stanza_group.setLayout(stanza_layout)
        self.controls_layout.addWidget(stanza_group)

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
        self.check_use_global_for_new.toggled.connect(self.on_toggle_use_global_for_new)
        global_settings_layout.addRow(self.check_use_global_for_new)
        self.btn_apply_globals_to_all = QPushButton("Applica Globali a Tutti i Sub")
        self.btn_apply_globals_to_all.clicked.connect(self.apply_global_settings_to_all_subs)
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
        sub_selector_layout.addWidget(self.sub_selector_text_widget, 1, alignment=Qt.AlignmentFlag.AlignCenter)
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
        self.tb_sub_y = QLineEdit()
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(self.tb_sub_x)
        pos_layout.addWidget(self.tb_sub_y)
        sub_params_layout.addRow(self.sub_pos_label, pos_layout)
        
        self.tb_sub_angle = QLineEdit()
        sub_params_layout.addRow("Angolo (°):", self.tb_sub_angle)
        
        self.sub_gain_label = QLabel("Trim Gain (dB):")
        self.tb_sub_gain_db = QLineEdit()
        sub_params_layout.addRow(self.sub_gain_label, self.tb_sub_gain_db)
        
        self.sub_delay_label = QLabel("Delay (ms):")
        self.tb_sub_delay = QLineEdit()
        sub_params_layout.addRow(self.sub_delay_label, self.tb_sub_delay)
        
        self.sub_polarity_label = QLabel("Polarità (+1/-1):")
        self.tb_sub_polarity = QLineEdit()
        sub_params_layout.addRow(self.sub_polarity_label, self.tb_sub_polarity)
        
        self.check_sub_angle_lock = QCheckBox("Blocca Angolo")
        self.check_sub_angle_lock.setObjectName("angle")
        self.check_sub_angle_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_angle_lock)
        self.check_sub_delay_lock = QCheckBox("Blocca Delay")
        self.check_sub_delay_lock.setObjectName("delay")
        self.check_sub_delay_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_delay_lock)
        self.check_sub_gain_lock = QCheckBox("Blocca Gain")
        self.check_sub_gain_lock.setObjectName("gain")
        self.check_sub_gain_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_gain_lock)
        self.check_sub_polarity_lock = QCheckBox("Blocca Polarità")
        self.check_sub_polarity_lock.setObjectName("polarity")
        self.check_sub_polarity_lock.toggled.connect(self.on_toggle_param_lock)
        sub_params_layout.addRow(self.check_sub_polarity_lock)
        self.tb_sub_width = QLineEdit()
        sub_params_layout.addRow("Larghezza Sub (m):", self.tb_sub_width)
        self.tb_sub_depth = QLineEdit()
        sub_params_layout.addRow("Profondità Sub (m):", self.tb_sub_depth)
        self.tb_sub_spl_rms = QLineEdit()
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
        
    def _setup_group_array_ui(self): # Nuova sezione unificata
        group_array_group = QGroupBox("Gestione Gruppi e Array")
        group_array_layout = QVBoxLayout()

        # Raggruppamento Subwoofer (spostato e adattato)
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
        self.group_details_label.setVisible(False) # Nascosto di default
        grouping_sub_layout.addWidget(self.group_details_label)

        self.group_members_list = QListWidget()
        self.group_members_list.setMaximumHeight(120) # Limitiamo l'altezza
        self.group_members_list.setVisible(False) # Nascosto di default
        grouping_sub_layout.addWidget(self.group_members_list)
        grouping_sub_group_box.setLayout(grouping_sub_layout)
        group_array_layout.addWidget(grouping_sub_group_box)

        # Creazione Array (spostato e adattato)
        array_setup_group_box = QGroupBox("Configurazione Array")
        self.array_setup_layout_form = QFormLayout()
        self.array_type_combo = QComboBox()
        self.array_type_combo.addItems(["Nessuno", "Coppia Cardioide (2 sub)", "Array End-Fire", "Array Lineare (Steering Elettrico)", "Array Vortex"])
        self.array_type_combo.currentIndexChanged.connect(self.on_array_type_change)
        self.array_setup_layout_form.addRow("Tipo di Array:", self.array_type_combo)
        self.array_freq_label = QLabel("Freq. Array Design (Hz):")
        self.array_freq_input = QLineEdit(str(DEFAULT_ARRAY_FREQ))
        self.array_setup_layout_form.addRow(self.array_freq_label, self.array_freq_input)
        self.array_auto_spacing_check = QCheckBox("Calcola Spaziatura/Raggio da Frequenza")
        self.array_auto_spacing_check.toggled.connect(self.on_auto_spacing_toggle)
        self.array_setup_layout_form.addRow(self.array_auto_spacing_check)
        self.array_wavelength_fraction_label = QLabel("Frazione λ per Spaziatura:")
        self.array_wavelength_fraction_combo = QComboBox()
        self.array_wavelength_fraction_combo.addItems(["λ/4", "λ/2"])
        self.array_setup_layout_form.addRow(self.array_wavelength_fraction_label, self.array_wavelength_fraction_combo)
        self.array_spacing_label = QLabel("Spaziatura/Raggio (m):")
        self.array_spacing_input = QLineEdit(str(DEFAULT_ARRAY_RADIUS))
        self.array_setup_layout_form.addRow(self.array_spacing_label, self.array_spacing_input)
        self.array_elements_label = QLabel("Numero Elementi:")
        self.array_elements_input = QLineEdit("4")
        self.array_setup_layout_form.addRow(self.array_elements_label, self.array_elements_input)
        self.array_start_angle_label = QLabel("Orientamento/Angolo (°):")
        self.array_start_angle_input = QLineEdit(str(DEFAULT_ARRAY_START_ANGLE_DEG))
        self.array_setup_layout_form.addRow(self.array_start_angle_label, self.array_start_angle_input)
        self.array_line_coverage_angle_label = QLabel("Angolo Copertura (°):")
        self.array_line_coverage_angle_input = QLineEdit(str(DEFAULT_LINE_ARRAY_COVERAGE_DEG))
        self.array_setup_layout_form.addRow(self.array_line_coverage_angle_label, self.array_line_coverage_angle_input)
        self.array_line_steering_angle_label = QLabel("Angolo Steering (°):")
        self.array_line_steering_angle_input = QLineEdit(str(DEFAULT_LINE_ARRAY_STEERING_DEG))
        self.array_setup_layout_form.addRow(self.array_line_steering_angle_label, self.array_line_steering_angle_input)
        self.array_vortex_mode_label = QLabel("Modalità Vortex:")
        self.array_vortex_mode_label.setToolTip("Numero intero (es. 1, 2, -1) che definisce l'avvolgimento dell'onda sonora.\nValori più alti creano un 'nullo' di pressione più ampio al centro.")
        self.array_vortex_mode_input = QLineEdit(str(DEFAULT_VORTEX_MODE))
        self.array_setup_layout_form.addRow(self.array_vortex_mode_label, self.array_vortex_mode_input)
        self.array_vortex_steering_angle_label = QLabel("Angolo Steering Vortex (°):")
        self.array_vortex_steering_angle_input = QLineEdit(str(DEFAULT_VORTEX_STEERING_DEG))
        self.array_setup_layout_form.addRow(self.array_vortex_steering_angle_label, self.array_vortex_steering_angle_input)

        self.apply_array_config_button = QPushButton("Crea Gruppo Array")
        self.apply_array_config_button.clicked.connect(self.apply_array_configuration)
        self.array_setup_layout_form.addRow(self.apply_array_config_button)
        self.array_info_label = QLabel()
        self.array_info_label.setWordWrap(True)
        self.array_setup_layout_form.addRow(self.array_info_label)
        self.error_text_array_params = QLabel("")
        self.error_text_array_params.setStyleSheet("color: red;")
        self.array_setup_layout_form.addRow(self.error_text_array_params)
        array_setup_group_box.setLayout(self.array_setup_layout_form)
        group_array_layout.addWidget(array_setup_group_box)

        group_array_group.setLayout(group_array_layout)
        self.controls_layout.addWidget(group_array_group)
        
    def _setup_target_areas_ui(self):
        targets_group = QGroupBox("Aree Target")
        targets_main_layout = QVBoxLayout()
        target_selector_layout = QHBoxLayout()
        self.btn_prev_target_area = QPushButton("<")
        self.btn_prev_target_area.clicked.connect(self.select_prev_target_area)
        target_selector_layout.addWidget(self.btn_prev_target_area)
        self.label_current_target_area = QLabel("Nessuna Area")
        target_selector_layout.addWidget(self.label_current_target_area, 1, Qt.AlignmentFlag.AlignCenter)
        self.btn_next_target_area = QPushButton(">")
        self.btn_next_target_area.clicked.connect(self.select_next_target_area)
        target_selector_layout.addWidget(self.btn_next_target_area)
        targets_main_layout.addLayout(target_selector_layout)
        target_actions_layout = QHBoxLayout()
        self.btn_new_target_area = QPushButton("Nuova")
        self.btn_new_target_area.clicked.connect(self.add_new_target_area_ui)
        target_actions_layout.addWidget(self.btn_new_target_area)
        self.btn_remove_selected_target_area = QPushButton("Rimuovi")
        self.btn_remove_selected_target_area.clicked.connect(self.remove_selected_target_area_ui)
        target_actions_layout.addWidget(self.btn_remove_selected_target_area)
        targets_main_layout.addLayout(target_actions_layout)
        self.check_activate_selected_target_area = QCheckBox("Attiva Area")
        self.check_activate_selected_target_area.toggled.connect(self.toggle_selected_target_area_active)
        targets_main_layout.addWidget(self.check_activate_selected_target_area)

        self.btn_add_target_vtx = QPushButton("Aggiungi Vertice a Area Target")
        self.btn_add_target_vtx.clicked.connect(self._add_vtx_to_current_target_area)
        targets_main_layout.addWidget(self.btn_add_target_vtx)

        self.target_vtx_list_widget = QListWidget()
        self.target_vtx_list_widget.currentItemChanged.connect(self.on_target_vtx_selection_change)
        self.target_vtx_list_widget.setMaximumHeight(100)
        targets_main_layout.addWidget(self.target_vtx_list_widget)
        
        vtx_edit_layout = QFormLayout()
        self.tb_target_vtx_x = QLineEdit()
        vtx_edit_layout.addRow("Vertice X:", self.tb_target_vtx_x)
        self.tb_target_vtx_y = QLineEdit()
        vtx_edit_layout.addRow("Vertice Y:", self.tb_target_vtx_y)
        self.btn_update_target_vtx = QPushButton("Aggiorna Vertice Target")
        self.btn_update_target_vtx.clicked.connect(self.on_update_selected_target_vertex)
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
        avoid_selector_layout.addWidget(self.label_current_avoid_area, 1, Qt.AlignmentFlag.AlignCenter)
        self.btn_next_avoid_area = QPushButton(">")
        self.btn_next_avoid_area.clicked.connect(self.select_next_avoidance_area)
        avoid_selector_layout.addWidget(self.btn_next_avoid_area)
        avoid_main_layout.addLayout(avoid_selector_layout)
        avoid_actions_layout = QHBoxLayout()
        self.btn_new_avoid_area = QPushButton("Nuova")
        self.btn_new_avoid_area.clicked.connect(self.add_new_avoidance_area_ui)
        avoid_actions_layout.addWidget(self.btn_new_avoid_area)
        self.btn_remove_selected_avoid_area = QPushButton("Rimuovi")
        self.btn_remove_selected_avoid_area.clicked.connect(self.remove_selected_avoidance_area_ui)
        avoid_actions_layout.addWidget(self.btn_remove_selected_avoid_area)
        avoid_main_layout.addLayout(avoid_actions_layout)
        self.check_activate_selected_avoid_area = QCheckBox("Attiva Area")
        self.check_activate_selected_avoid_area.toggled.connect(self.toggle_selected_avoidance_area_active)
        avoid_main_layout.addWidget(self.check_activate_selected_avoid_area) # Corrected typo here

        self.btn_add_avoid_vtx = QPushButton("Aggiungi Vertice a Area Evitamento")
        self.btn_add_avoid_vtx.clicked.connect(self._add_vtx_to_current_avoidance_area)
        avoid_main_layout.addWidget(self.btn_add_avoid_vtx)

        self.avoid_vtx_list_widget = QListWidget()
        self.avoid_vtx_list_widget.currentItemChanged.connect(self.on_avoid_vtx_selection_change)
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
        self.tb_spl_min = QLineEdit("50")
        self.tb_spl_max = QLineEdit("100")
        spl_vis_layout.addRow("SPL Min Display (dB):", self.tb_spl_min)
        spl_vis_layout.addRow("SPL Max Display (dB):", self.tb_spl_max)
        self.error_text_spl_range = QLabel("")
        self.error_text_spl_range.setStyleSheet("color: red;")
        spl_vis_layout.addRow(self.error_text_spl_range)
        self.check_auto_spl_update = QCheckBox("Aggiorna SPL Automaticamente")
        self.check_auto_spl_update.setChecked(True)
        spl_vis_layout.addRow(self.check_auto_spl_update)
        self.btn_update_spl = QPushButton("Aggiorna Mappa SPL")
        self.btn_update_spl.clicked.connect(lambda: self.visualizza_mappatura_spl(self.get_slider_freq_val(), preserve_view=True))
        spl_vis_layout.addRow(self.btn_update_spl)
        spl_vis_group.setLayout(spl_vis_layout)
        self.controls_layout.addWidget(spl_vis_group)
        
    def _setup_sim_grid_ui(self):
        sim_grid_params_group = QGroupBox("Parametri Simulazione e Griglia")
        sim_grid_params_layout = QFormLayout()
        self.tb_velocita_suono = QLineEdit(str(DEFAULT_SIM_SPEED_OF_SOUND))
        sim_grid_params_layout.addRow("Velocità Suono (m/s):", self.tb_velocita_suono)
        self.tb_grid_res_spl = QLineEdit("0.1")
        sim_grid_params_layout.addRow("Risoluzione Mappa SPL (m):", self.tb_grid_res_spl)
        self.tb_grid_snap_spacing = QLineEdit(str(self.grid_snap_spacing))
        self.tb_grid_snap_spacing.editingFinished.connect(self.update_grid_snap_params)
        sim_grid_params_layout.addRow("Passo Griglia Snap (m):", self.tb_grid_snap_spacing)
        self.check_grid_snap_enabled = QCheckBox("Abilita Snap Oggetti")
        self.check_grid_snap_enabled.setChecked(self.grid_snap_enabled)
        self.check_grid_snap_enabled.stateChanged.connect(self.update_grid_snap_params)
        sim_grid_params_layout.addRow(self.check_grid_snap_enabled)
        self.check_show_grid = QCheckBox("Mostra Griglia")
        self.check_show_grid.setChecked(self.grid_show_enabled)
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
        self.radio_copertura.toggled.connect(lambda: self.update_optim_freq_fields_visibility("Copertura SPL"))
        crit_layout.addWidget(self.radio_copertura)
        self.radio_omogeneita = QRadioButton("Omogeneità SPL")
        self.radio_omogeneita.toggled.connect(lambda: self.update_optim_freq_fields_visibility("Omogeneità SPL"))
        crit_layout.addWidget(self.radio_omogeneita)
        self.radio_btn_group_crit.addButton(self.radio_copertura)
        self.radio_btn_group_crit.addButton(self.radio_omogeneita)
        optim_main_layout.addLayout(crit_layout)
        optim_freq_layout = QFormLayout()
        self.label_opt_freq_single_widget = QLabel("Freq. Ottim. (Hz):")
        self.tb_opt_freq_single = QLineEdit("80")
        optim_freq_layout.addRow(self.label_opt_freq_single_widget, self.tb_opt_freq_single)
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
        self.tb_opt_pop_size = QLineEdit("50")
        optim_algo_layout.addRow("Config. Testate:", self.tb_opt_pop_size)
        self.tb_opt_generations = QLineEdit("30")
        optim_algo_layout.addRow("Cicli Ottim.:", self.tb_opt_generations)
        
        self.tb_target_min_spl_desired = QLineEdit(str(self.target_min_spl_desired_ui_val))
        optim_algo_layout.addRow("Min SPL Target Desiderato (dB):", self.tb_target_min_spl_desired)
        
        self.tb_max_spl_avoid = QLineEdit(str(self.max_spl_avoidance_ui_val))
        optim_algo_layout.addRow("Max SPL Evit. (dB):", self.tb_max_spl_avoid)
        
        self.label_balance_slider = QLabel("Bilanciamento Target / Evitamento:")
        optim_algo_layout.addRow(self.label_balance_slider)
        self.slider_balance = QSlider(Qt.Orientation.Horizontal)
        self.slider_balance.setMinimum(0)
        self.slider_balance.setMaximum(100)
        self.slider_balance.setValue(self.balance_slider_ui_val)
        self.label_balance_value = QLabel(f"{self.slider_balance.value()}% Target")
        self.slider_balance.valueChanged.connect(lambda val: self.label_balance_value.setText(f"{val}% Target"))
        optim_algo_layout.addRow(self.slider_balance)
        optim_algo_layout.addRow(self.label_balance_value)

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

    def disegna_subwoofer_e_elementi(self):
        for i, sub_ in enumerate(self.sorgenti):
            color = "green" if i == self.current_sub_idx else "black"
            if sub_.get('group_id') is not None:
                if sub_['is_group_master']: color = "purple"
                elif i == self.current_sub_idx: color = "darkorange"
                else: color = "blue"

            center_x, center_y = sub_['x'], sub_['y']
            angle_rad = sub_['angle']
            
            display_angle_deg = 90 - np.degrees(angle_rad)
            
            width = sub_.get('width', DEFAULT_SUB_WIDTH)
            depth = sub_.get('depth', DEFAULT_SUB_DEPTH)
            
            rect = patches.Rectangle((-depth / 2, -width / 2), depth, width, linewidth=1.5, edgecolor=color,
                                     facecolor=color, alpha=0.6, gid=f"rect_sub_{sub_['id']}", picker=True, zorder=2.5)
            transform = mtransforms.Affine2D().rotate_deg(display_angle_deg) + mtransforms.Affine2D().translate(center_x, center_y) + self.ax.transData
            rect.set_transform(transform)
            self.ax.add_patch(rect)
            sub_['rect_artist'] = rect 

            arrow_end_x = center_x + ARROW_LENGTH * np.sin(angle_rad)
            arrow_end_y = center_y + ARROW_LENGTH * np.cos(angle_rad)
            arrow = patches.FancyArrowPatch((center_x, center_y), (arrow_end_x, arrow_end_y),
                                            mutation_scale=20, arrowstyle='->', color='dimgray', linewidth=1.5,
                                            zorder=2.6, gid=f"arrow_sub_{sub_['id']}", picker=True)
            self.ax.add_patch(arrow)
            sub_['arrow_artist'] = arrow 

            pol_char = '+' if sub_['polarity'] > 0 else '-'
            group_info = ""
            if sub_.get('group_id') is not None:
                group_info = f"G{sub_['group_id']}"
                if sub_['is_group_master']: group_info += " (M)"
                group_info += ", "
            sub_text_info = (f"S{sub_.get('id', i + 1)}: {sub_.get('spl_rms', DEFAULT_SUB_SPL_RMS):.0f}dB\n"
                             f"{group_info}{sub_.get('gain_db', 0.0):.1f}dB, {sub_['delay_ms']:.1f}ms, Pol {pol_char}")
            
            text_offset = max(width, depth) / 2 + 0.15 
            self.ax.text(center_x, center_y + text_offset, sub_text_info, color=color, fontsize=6,
                         ha="center", va="bottom", zorder=6)

    def disegna_array_direction_indicators(self):
        for group_id, group_info in self.lista_gruppi_array.items():
            members = [s for s in self.sorgenti if s.get('group_id') == group_id]
            if not members: continue

            center_x = np.mean([s['x'] for s in members])
            center_y = np.mean([s['y'] for s in members])
            
            steering_deg_nord = group_info.get('steering_deg', 0)
            steering_deg_est = 90 - steering_deg_nord
            
            wedge = patches.Wedge(
                center=(center_x, center_y), r=ARRAY_INDICATOR_RADIUS,
                theta1=steering_deg_est - ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                theta2=steering_deg_est + ARRAY_INDICATOR_CONE_WIDTH_DEG / 2,
                facecolor='cyan', edgecolor='blue', alpha=0.25, zorder=1.5
            )
            self.ax.add_patch(wedge)

    def disegna_stanza_e_vertici(self):
        if self.punti_stanza and len(self.punti_stanza) >=3:
            points_pos = [p['pos'] for p in self.punti_stanza]
            patch = patches.Polygon(points_pos, closed=True, fill=None, edgecolor="blue", linewidth=2, zorder=1)
            self.ax.add_patch(patch)
            for i, vtx_data in enumerate(self.punti_stanza):
                x, y = vtx_data['pos']
                vtx_data['plot'] = self.ax.plot(x,y,marker="o",ms=7,color="red",picker=7,gid=f'stanza_vtx_{i}', zorder=5.2)[0]

    def disegna_aree_target_e_avoidance(self):
        for area_list, type_prefix in [(self.lista_target_areas, 'target'), (self.lista_avoidance_areas, 'avoid')]:
            for area_data in area_list:
                area_data['plots'] = [] 
                if area_data.get('active', False) and len(area_data.get('punti', [])) >= 3:
                    is_selected_target = type_prefix == 'target' and self.current_target_area_idx != -1 and area_data['id'] == self.lista_target_areas[self.current_target_area_idx]['id']
                    is_selected_avoid = type_prefix == 'avoid' and self.current_avoidance_area_idx != -1 and area_data['id'] == self.lista_avoidance_areas[self.current_avoidance_area_idx]['id']
                    is_selected = is_selected_target or is_selected_avoid
                    color, m_color, ls = ('green', 'lime', '--') if type_prefix == 'target' else ('red', 'orangered', ':')
                    patch = patches.Polygon(area_data['punti'], closed=True, fill=True, edgecolor=color, facecolor=color, alpha=0.3 if is_selected else 0.15, ls=ls, zorder=0.5)
                    self.ax.add_patch(patch)
                    for v_idx, v in enumerate(area_data['punti']):
                        plot_artist = self.ax.plot(v[0], v[1], marker='P' if type_prefix=='target' else 'X', ms=9, color=m_color, picker=7, zorder=5.1)[0]
                        area_data['plots'].append(plot_artist)

    def disegna_griglia(self):
        if self.grid_show_enabled and self.grid_snap_spacing > 0:
            xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
            x_ticks = np.arange(round(xlim[0]/self.grid_snap_spacing)*self.grid_snap_spacing, xlim[1], self.grid_snap_spacing)
            y_ticks = np.arange(round(ylim[0]/self.grid_snap_spacing)*self.grid_snap_spacing, ylim[1], self.grid_snap_spacing)
            for x in x_ticks: self.ax.axvline(x, color='gray', linestyle=':', linewidth=0.5, alpha=0.6, zorder=-1)
            for y in y_ticks: self.ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.6, zorder=-1)

    def auto_fit_view_to_room(self):
        if not hasattr(self, 'ax'): return
        if not self.punti_stanza:
            self.ax.set_xlim(-7, 7)
            self.ax.set_ylim(-5, 5)
        else:
            all_x = [p['pos'][0] for p in self.punti_stanza]
            all_y = [p['pos'][1] for p in self.punti_stanza]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            px = (max_x - min_x) * 0.15 if (max_x - min_x) > 0 else 1.5
            py = (max_y - min_y) * 0.15 if (max_y - min_y) > 0 else 1.5
            self.ax.set_xlim(min_x - px, max_x + px)
            self.ax.set_ylim(min_y - py, max_y + py)
        self.ax.set_aspect('equal', adjustable='box')
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.canvas.draw_idle()

    def get_active_areas_points(self, area_list):
        return [list(area['punti']) for area in area_list if area.get('active', False) and len(area.get('punti', [])) >= 3]

    def snap_to_grid(self, value):
        return round(value / self.grid_snap_spacing) * self.grid_snap_spacing if self.grid_snap_enabled and self.grid_snap_spacing > 0 else value
        
    def add_stanza_vtx(self, event=None):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim()
        new_vtx_coord = [np.mean(xlims), np.mean(ylims)]
        if self.punti_stanza and len(self.punti_stanza) > 1: p_last = self.punti_stanza[-1]['pos']; new_vtx_coord = [p_last[0] + 1, p_last[1]]
        snapped_pos = [self.snap_to_grid(new_vtx_coord[0]), self.snap_to_grid(new_vtx_coord[1])]
        self.punti_stanza.append({'pos': snapped_pos, 'plot': None})
        self.full_redraw(preserve_view=False)

    def remove_stanza_vtx(self, event=None):
        if len(self.punti_stanza) > 0:
            self.punti_stanza.pop()
            self.full_redraw(preserve_view=False)

    def update_stanza_vtx_editor(self):
        is_valid_idx = 0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)
        for w in [self.tb_stanza_vtx_x, self.tb_stanza_vtx_y, self.btn_update_stanza_vtx]: w.setEnabled(is_valid_idx)
        if is_valid_idx:
            vtx = self.punti_stanza[self.selected_stanza_vtx_idx]['pos']
            self.selected_vtx_label.setText(f"Vertice Selezionato {self.selected_stanza_vtx_idx + 1}:")
            self.tb_stanza_vtx_x.setText(f"{vtx[0]:.2f}"); self.tb_stanza_vtx_y.setText(f"{vtx[1]:.2f}")
        else: self.selected_vtx_label.setText("Nessun Vertice Selezionato"); self.tb_stanza_vtx_x.setText(""); self.tb_stanza_vtx_y.setText("")
        self.error_text_stanza_vtx_edit.setText("")

    def on_update_selected_stanza_vertex(self):
        if not (0 <= self.selected_stanza_vtx_idx < len(self.punti_stanza)): return
        try:
            x, y = float(self.tb_stanza_vtx_x.text()), float(self.tb_stanza_vtx_y.text())
            self.punti_stanza[self.selected_stanza_vtx_idx]['pos'] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.full_redraw(preserve_view=False)
            self.update_stanza_vtx_editor()
        except ValueError: self.error_text_stanza_vtx_edit.setText("Coordinate non valide.")

    def on_toggle_use_global_for_new(self, checked): self.use_global_for_new_manual_subs = checked
    
    def apply_global_settings_to_all_subs(self):
        try:
            new_spl = float(self.tb_global_sub_spl.text())
            if not (PARAM_RANGES['spl_rms'][0] <= new_spl <= PARAM_RANGES['spl_rms'][1]): raise ValueError(f"SPL fuori range")
            self.global_sub_width = float(self.tb_global_sub_width.text()); self.global_sub_depth = float(self.tb_global_sub_depth.text()); self.global_sub_spl_rms = new_spl
            for sub in self.sorgenti: sub.update({'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms, 'pressure_val_at_1m_relative_to_pref': 10**(new_spl/20.0)})
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError as e: self.error_text_global_settings.setText(f"Errore: {e}")

    def select_next_sub(self, event=None):
        if self.sorgenti: 
            self.current_sub_idx = (self.current_sub_idx + 1) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)
    def select_prev_sub(self, event=None):
        if self.sorgenti: 
            self.current_sub_idx = (self.current_sub_idx - 1 + len(self.sorgenti)) % len(self.sorgenti)
            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def add_subwoofer(self, event=None, specific_config=None, redraw=True):
        new_sub_data = specific_config or {}
        if not specific_config:
            new_sub_data = {'x': 0.0, 'y': 0.0}
            if self.use_global_for_new_manual_subs: new_sub_data.update({'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms})
        
        defaults = {'id': self.next_sub_id, 'angle': 0.0, 'delay_ms': 0.0, 'polarity': 1, 'gain_db': 0.0, 
                    'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms, 
                    'param_locks': {'angle': False, 'delay': False, 'gain': False, 'polarity': False}, 
                    'group_id': None, 'is_group_master': False}
        for k, v in defaults.items(): new_sub_data.setdefault(k, v)
        
        new_sub_data['gain_lin'] = 10**(new_sub_data['gain_db']/20.0)
        new_sub_data['pressure_val_at_1m_relative_to_pref'] = 10**(new_sub_data['spl_rms']/20.0)
        
        self.sorgenti.append(new_sub_data)
        self.next_sub_id += 1
        self.current_sub_idx = len(self.sorgenti) - 1
        
        if redraw:
            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()

    def remove_subwoofer(self, event=None):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub_to_remove = self.sorgenti.pop(self.current_sub_idx)
        gid_to_check = sub_to_remove.get('group_id')
        if gid_to_check is not None:
            remaining_in_group = [s for s in self.sorgenti if s.get('group_id') == gid_to_check]
            if not remaining_in_group and gid_to_check in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid_to_check]
            elif sub_to_remove.get('is_group_master') and remaining_in_group:
                remaining_in_group[0]['is_group_master'] = True
        
        if not self.sorgenti:
            self.current_sub_idx = -1
        else:
            self.current_sub_idx = min(self.current_sub_idx, len(self.sorgenti) - 1)
            
        self.full_redraw(preserve_view=True)
        self.aggiorna_ui_sub_fields()

    def aggiorna_ui_sub_fields(self):
        enable = bool(self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti))
        for w in [self.tb_sub_x, self.tb_sub_y, self.tb_sub_angle, self.check_sub_angle_lock, self.tb_sub_delay, self.check_sub_delay_lock, self.tb_sub_gain_db, self.check_sub_gain_lock, self.tb_sub_polarity, self.check_sub_polarity_lock, self.tb_sub_width, self.tb_sub_depth, self.tb_sub_spl_rms, self.btn_submit_sub_params, self.btn_rem_sub, self.apply_array_config_button]: w.setEnabled(enable)
        
        # Resetta le etichette al loro stato di default
        self.sub_gain_label.setText("Trim Gain (dB):")
        self.sub_delay_label.setText("Delay (ms):")
        self.sub_polarity_label.setText("Polarità (+1/-1):")
        self.tb_sub_gain_db.setPlaceholderText("")
        self.tb_sub_delay.setPlaceholderText("")
        self.tb_sub_polarity.setPlaceholderText("")

        if enable:
            sub = self.sorgenti[self.current_sub_idx]
            self.sub_selector_text_widget.setText(f"Sub ID:{sub.get('id', '')} ({self.current_sub_idx + 1}/{len(self.sorgenti)})")
            
            if sub.get('group_id') is not None:
                centroid = self._get_group_centroid(sub['group_id'])
                if centroid:
                    self.tb_sub_x.setText(f"{centroid[0]:.2f}")
                    self.tb_sub_y.setText(f"{centroid[1]:.2f}")
                self.sub_pos_label.setText("X/Y Gruppo (m):")
                # NUOVA FUNZIONALITÀ: Modifica UI per controllo di gruppo
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
                self.tb_sub_polarity.setText(str(int(sub['polarity'])))

            self.tb_sub_angle.setText(f"{np.degrees(sub['angle']):.1f}")
            self.tb_sub_width.setText(f"{sub['width']:.2f}"); self.tb_sub_depth.setText(f"{sub['depth']:.2f}"); self.tb_sub_spl_rms.setText(f"{sub['spl_rms']:.1f}")
            for p, cb in [('angle', self.check_sub_angle_lock), ('delay', self.check_sub_delay_lock), ('gain', self.check_sub_gain_lock), ('polarity', self.check_sub_polarity_lock)]:
                try: cb.toggled.disconnect()
                except TypeError: pass
                cb.setChecked(sub['param_locks'].get(p, False)); cb.toggled.connect(self.on_toggle_param_lock)
        else:
            self.sub_selector_text_widget.setText("Nessun Sub")
            for w in [self.tb_sub_x, self.tb_sub_y, self.tb_sub_angle, self.tb_sub_delay, self.tb_sub_gain_db, self.tb_sub_polarity, self.tb_sub_width, self.tb_sub_depth, self.tb_sub_spl_rms]:
                w.clear()
            self.sub_pos_label.setText("X/Y Sub (m):")

        self.error_text_sub.setText(""); self._update_group_ui_status()
        if enable and sub.get('group_id') is not None:
           self.group_details_label.setVisible(True)
           self.group_members_list.setVisible(True)
           self.group_members_list.clear()

           group_id = sub.get('group_id')
           members = [s for s in self.sorgenti if s.get('group_id') == group_id]
           
           for member in members:
               pol_char = '+' if member['polarity'] > 0 else '-'
               master_char = " (M)" if member.get('is_group_master') else ""
               info_string = (f"S{member['id']}{master_char}: "
                              f"{member['gain_db']:.1f}dB, "
                              f"{member['delay_ms']:.2f}ms, "
                              f"Pol {pol_char}")
               self.group_members_list.addItem(QListWidgetItem(info_string))
        else:
           # Se il sub non è in un gruppo, nascondi i dettagli
           self.group_details_label.setVisible(False)
           self.group_members_list.setVisible(False)
           self.group_members_list.clear()
        
    def _get_group_centroid(self, group_id):
        if group_id is None: return None
        members = [s for s in self.sorgenti if s.get('group_id') == group_id]
        if not members: return None
        center_x = np.mean([s['x'] for s in members])
        center_y = np.mean([s['y'] for s in members])
        return (center_x, center_y)

    def on_toggle_param_lock(self, checked):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]; sender = self.sender(); param_name = sender.objectName()
        if sub.get('group_id') is not None and sub['param_locks'].get(param_name) and not checked: QMessageBox.warning(self, "Attenzione", f"Il parametro è controllato dall'array. Sciogli il gruppo per sbloccarlo."); sender.setChecked(True); return
        sub['param_locks'][param_name] = checked

    def on_submit_sub_param_qt(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        try:
            sub = self.sorgenti[self.current_sub_idx]
            
            if sub.get('group_id') is not None:
                group_id = sub['group_id']
                current_centroid = self._get_group_centroid(group_id)
                new_centroid_x = float(self.tb_sub_x.text())
                new_centroid_y = float(self.tb_sub_y.text())
                if current_centroid:
                    dx = new_centroid_x - current_centroid[0]
                    dy = new_centroid_y - current_centroid[1]
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id:
                            member['x'] += dx
                            member['y'] += dy
                
                if self.tb_sub_gain_db.text().strip():
                    gain_delta = float(self.tb_sub_gain_db.text())
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id: member['gain_db'] += gain_delta
                
                if self.tb_sub_delay.text().strip():
                    delay_delta = float(self.tb_sub_delay.text())
                    for member in self.sorgenti:
                        if member.get('group_id') == group_id: member['delay_ms'] += delay_delta

                if self.tb_sub_polarity.text().strip():
                    new_pol = int(self.tb_sub_polarity.text())
                    if new_pol in [1, -1]:
                        for member in self.sorgenti:
                            if member.get('group_id') == group_id: member['polarity'] = new_pol

            else:
                sub['x']=float(self.tb_sub_x.text())
                sub['y']=float(self.tb_sub_y.text())
                if not sub['param_locks'].get('gain', False): sub['gain_db']=float(self.tb_sub_gain_db.text())
                if not sub['param_locks'].get('delay', False): sub['delay_ms']=float(self.tb_sub_delay.text())
                if not sub['param_locks'].get('polarity', False): sub['polarity']=int(self.tb_sub_polarity.text())

            if not sub['param_locks'].get('angle', False): sub['angle']=np.radians(float(self.tb_sub_angle.text()))
            sub.update({'width':float(self.tb_sub_width.text()), 'depth':float(self.tb_sub_depth.text()), 'spl_rms':float(self.tb_sub_spl_rms.text())})
            
            for s_ in self.sorgenti:
                s_['gain_lin'] = 10**(s_['gain_db']/20.0)
                s_['pressure_val_at_1m_relative_to_pref'] = 10**(s_['spl_rms'] / 20.0)

            self.full_redraw(preserve_view=True)
            self.aggiorna_ui_sub_fields()
        except ValueError: self.error_text_sub.setText("Errore: Dati non validi.")
        except Exception as e: self.error_text_sub.setText(f"Errore: {e}")

    def _update_max_group_id(self):
        self.max_group_id = max([s.get('group_id', 0) for s in self.sorgenti if s.get('group_id') is not None] or [0])
        self.next_group_id = self.max_group_id + 1
        
    def _update_group_ui_status(self):
        if not hasattr(self, 'btn_add_to_group') or not self.sorgenti:
            if hasattr(self, 'group_status_label'): self.group_status_label.setText("Nessun Sub")
            for w in [self.btn_create_new_group, self.btn_add_to_group, self.btn_remove_from_group, self.btn_ungroup_all]: w.setEnabled(False)
            return
            
        enable = 0 <= self.current_sub_idx < len(self.sorgenti)
        if not enable:
            self.group_status_label.setText("Nessun Sub Selezionato")
            for w in [self.btn_create_new_group, self.btn_add_to_group, self.btn_remove_from_group, self.btn_ungroup_all]: w.setEnabled(False)
            return

        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get('group_id')
        is_master = sub.get('is_group_master')
        self.btn_create_new_group.setEnabled(g_id is None)
        self.btn_add_to_group.setEnabled(g_id is None)
        self.btn_remove_from_group.setEnabled(g_id is not None)
        self.btn_ungroup_all.setEnabled(g_id is not None)
        
        if g_id:
            self.group_status_label.setText(f"Sub selezionato: Gruppo ID {g_id}{' (M)' if is_master else ''}")
        else:
            self.group_status_label.setText("Sub selezionato: Nessun Gruppo")

    def create_new_group(self):
        if not self.sorgenti or self.current_sub_idx == -1: return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get('group_id'): self.error_text_grouping.setText("Sub già in un gruppo."); return
        self._update_max_group_id(); sub['group_id'] = self.next_group_id; sub['is_group_master'] = True
        self.status_bar.showMessage(f"Creato Gruppo ID {self.next_group_id}.", 3000)
        self.aggiorna_ui_sub_fields()
        
    def add_sub_to_existing_group(self):
        if not self.sorgenti or self.current_sub_idx == -1: return
        sub = self.sorgenti[self.current_sub_idx]
        if sub.get('group_id'): self.error_text_grouping.setText("Sub già in un gruppo."); return
        
        existing_groups = sorted(list(set(s['group_id'] for s in self.sorgenti if s['group_id'] is not None)))
        if not existing_groups:
            self.error_text_grouping.setText("Nessun gruppo esistente."); return
            
        group_id_str, ok = QInputDialog.getItem(self, "Aggiungi a Gruppo", "Seleziona ID del gruppo:", [str(g) for g in existing_groups], 0, False)
        if not ok or not group_id_str: return
        
        target_id = int(group_id_str)
        sub['group_id'] = target_id
        sub['is_group_master'] = False
        self.aggiorna_ui_sub_fields()
        
    def remove_sub_from_group(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]
        if not sub.get('group_id'): return
        
        gid = sub.get('group_id')
        is_master = sub.get('is_group_master')
        
        sub['group_id'] = None
        sub['is_group_master'] = False
        for p in sub['param_locks']: sub['param_locks'][p] = False

        if is_master:
            remaining_in_group = [s for s in self.sorgenti if s.get('group_id') == gid]
            if remaining_in_group:
                remaining_in_group[0]['is_group_master'] = True
            elif gid in self.lista_gruppi_array:
                del self.lista_gruppi_array[gid]

        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)
        
    def ungroup_selected_sub_group(self):
        if not self.sorgenti or self.current_sub_idx < 0: return
        sub = self.sorgenti[self.current_sub_idx]
        g_id = sub.get('group_id')
        if g_id is None: return

        if g_id in self.lista_gruppi_array:
            del self.lista_gruppi_array[g_id]

        for s in self.sorgenti:
            if s.get('group_id') == g_id:
                s['group_id'] = None; s['is_group_master'] = False
                for p in s['param_locks']: s['param_locks'][p] = False
        self.aggiorna_ui_sub_fields()
        self.full_redraw(preserve_view=True)
        
    def _normalize_delays(self, configs_list):
        if not configs_list: return
        all_delays = [config['delay_ms'] for config in configs_list]
        min_delay = min(all_delays)
        if min_delay != 0:
            for config in configs_list:
                config['delay_ms'] -= min_delay

    def _update_array_ui(self):
        array_type = self.array_type_combo.currentText()
        is_none = array_type == "Nessuno"
        is_card_endfire = array_type in ["Coppia Cardioide (2 sub)", "Array End-Fire"]
        is_line_array = array_type == "Array Lineare (Steering Elettrico)"
        is_vortex = array_type == "Array Vortex"
        is_auto_spacing = self.array_auto_spacing_check.isChecked()

        self.array_freq_label.setVisible(not is_none)
        self.array_freq_input.setVisible(not is_none)
        self.array_auto_spacing_check.setVisible(is_card_endfire or is_line_array or is_vortex)
        
        show_frac_lambda = is_auto_spacing and (is_card_endfire or is_line_array or is_vortex)
        self.array_wavelength_fraction_label.setVisible(show_frac_lambda)
        self.array_wavelength_fraction_combo.setVisible(show_frac_lambda)
        
        self.array_spacing_label.setVisible(not is_none)
        self.array_spacing_input.setVisible(not is_none)
        
        if is_vortex:
            self.array_spacing_label.setText("Raggio Array (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        elif is_card_endfire or is_line_array:
            self.array_spacing_label.setText("Spaziatura Fisica (m):")
            self.array_spacing_input.setReadOnly(is_auto_spacing)
        else:
            self.array_spacing_label.setText("Spaziatura/Raggio (m):")
            self.array_spacing_input.setReadOnly(False)
            
        self.array_elements_label.setVisible(not is_none)
        self.array_elements_input.setVisible(not is_none)
        self.array_elements_input.setEnabled(array_type not in ["Nessuno", "Coppia Cardioide (2 sub)"])
        if array_type == "Coppia Cardioide (2 sub)": self.array_elements_input.setText("2")
        
        should_show_start_angle = is_line_array or is_vortex or is_card_endfire
        self.array_start_angle_label.setVisible(should_show_start_angle)
        self.array_start_angle_input.setVisible(should_show_start_angle)
        
        if is_line_array or is_card_endfire:
            self.array_start_angle_label.setText("Orientamento (°):")
        elif is_vortex:
            self.array_start_angle_label.setText("Angolo Iniziale (°):")

        for w in [self.array_line_steering_angle_label, self.array_line_steering_angle_input, self.array_line_coverage_angle_label, self.array_line_coverage_angle_input]: w.setVisible(is_line_array)
        
        self.array_vortex_mode_label.setVisible(is_vortex)
        self.array_vortex_mode_input.setVisible(is_vortex)
        self.array_vortex_steering_angle_label.setVisible(is_vortex)
        self.array_vortex_steering_angle_input.setVisible(is_vortex)

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

    def apply_array_configuration(self):
        ref_sub_idx = self.current_sub_idx if self.sorgenti and 0 <= self.current_sub_idx < len(self.sorgenti) else -1
        ref_sub_model = self.sorgenti[ref_sub_idx] if ref_sub_idx != -1 else None
        
        center_x, center_y = (ref_sub_model['x'], ref_sub_model['y']) if ref_sub_model else (0,0)
        base_sub_params = {
            'width': self.global_sub_width, 'depth': self.global_sub_depth, 'spl_rms': self.global_sub_spl_rms,
            'gain_db': 0, 'polarity': 1, 'delay_ms': 0
        }
        if ref_sub_model:
            base_sub_params.update({k: v for k, v in ref_sub_model.items() if k in base_sub_params})

        self.error_text_array_params.setText("")
        array_type = self.array_type_combo.currentText()
        if array_type == "Nessuno": return
        
        try:
            c_sound = float(self.tb_velocita_suono.text())
            design_freq = float(self.array_freq_input.text())
            num_elements = int(self.array_elements_input.text())
            spacing_or_radius = float(self.array_spacing_input.text())
            start_angle_deg = float(self.array_start_angle_input.text()) if self.array_start_angle_input.isVisible() else 0.0

            if self.array_auto_spacing_check.isChecked() and array_type != "Nessuno":
                if design_freq <= 0: raise ValueError("La frequenza di design deve essere positiva.")
                wavelength = c_sound / design_freq
                spacing_or_radius = wavelength / 4.0 if self.array_wavelength_fraction_combo.currentText() == "λ/4" else wavelength / 2.0
                self.array_spacing_input.setText(f"{spacing_or_radius:.3f}")
            
            array_params = {}
            if array_type == "Coppia Cardioide (2 sub)":
                self.setup_cardioid_pair(center_x, center_y, spacing_or_radius, c_sound, start_angle_deg, base_sub_params, ref_sub_idx)
            elif array_type == "Array End-Fire":
                self.setup_end_fire_array(center_x, center_y, num_elements, spacing_or_radius, c_sound, start_angle_deg, base_sub_params, ref_sub_idx)
            elif array_type == "Array Lineare (Steering Elettrico)":
                steering_angle_deg = float(self.array_line_steering_angle_input.text())
                coverage_angle_deg = float(self.array_line_coverage_angle_input.text())
                array_params = {'steering_deg': steering_angle_deg, 'coverage_deg': coverage_angle_deg}
                self.setup_line_array_steered(center_x, center_y, num_elements, spacing_or_radius, start_angle_deg, steering_angle_deg, coverage_angle_deg, c_sound, base_sub_params, array_params, ref_sub_idx)
            elif array_type == "Array Vortex":
                vortex_mode = int(self.array_vortex_mode_input.text())
                steering_deg = float(self.array_vortex_steering_angle_input.text())
                array_params = {'steering_deg': steering_deg, 'mode': vortex_mode}
                self.setup_vortex_array(center_x, center_y, num_elements, spacing_or_radius, vortex_mode, design_freq, start_angle_deg, steering_deg, c_sound, base_sub_params, array_params, ref_sub_idx)

        except Exception as e:
            self.error_text_array_params.setText(f"Errore parametri array: {e}")
            import traceback
            traceback.print_exc()

    def _add_new_subs_as_group(self, configs_list, array_type, array_params, ref_sub_idx_to_remove=-1):
        if not configs_list: return

        if ref_sub_idx_to_remove != -1 and ref_sub_idx_to_remove < len(self.sorgenti):
            self.sorgenti.pop(ref_sub_idx_to_remove)
        
        self._update_max_group_id()
        new_group_id = self.next_group_id
        self.lista_gruppi_array[new_group_id] = {'type': array_type, **array_params}

        start_index_of_new_subs = len(self.sorgenti)
        for config in configs_list:
            new_sub_data = config.copy()
            new_sub_data['group_id'] = new_group_id
            
            defaults = {'id': self.next_sub_id, 'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': True}}
            for k, v in defaults.items(): new_sub_data.setdefault(k, v)
            
            new_sub_data['gain_lin'] = 10**(new_sub_data.get('gain_db',0)/20.0)
            new_sub_data['pressure_val_at_1m_relative_to_pref'] = 10**(new_sub_data.get('spl_rms', DEFAULT_SUB_SPL_RMS)/20.0)
            
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

    def setup_line_array_steered(self, center_x, center_y, num_elements, spacing, orientation_deg, steering_deg, coverage_deg, c, base_params, array_params, ref_sub_idx):
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
                except (ValueError, ZeroDivisionError): pass
            
            total_delay_ms = (delay_steering_sec + delay_coverage_sec) * 1000.0
            new_conf = base_params.copy()
            new_conf.update({'x':sub_x, 'y':sub_y, 'angle':sub_physical_orientation, 'delay_ms':total_delay_ms, 'is_group_master':(i == num_elements // 2), 'param_locks':{'angle':True,'delay':True,'gain':False,'polarity':False}})
            new_configs.append(new_conf)
            
        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "Lineare", array_params, ref_sub_idx)

    def setup_vortex_array(self, center_x, center_y, num_elements, radius, mode, freq, start_angle_deg, steering_deg, c, base_params, array_params, ref_sub_idx):
        angle_step = 2 * np.pi / num_elements
        start_angle_rad = np.radians(start_angle_deg)
        steering_rad = np.radians(steering_deg)
        new_configs = []

        for n in range(num_elements):
            phi = start_angle_rad + n * angle_step
            sub_x = center_x + radius * np.sin(phi)
            sub_y = center_y + radius * np.cos(phi)
            sub_orientation = phi + np.pi/2
            
            delay_vortex_ms = ((mode * n) / (num_elements * freq)) * 1000.0
            
            pos_vector_x = sub_x - center_x
            pos_vector_y = sub_y - center_y
            steering_dir_x, steering_dir_y = np.sin(steering_rad), np.cos(steering_rad)
            projected_distance = pos_vector_x * steering_dir_x + pos_vector_y * steering_dir_y
            delay_steering_ms = (-projected_distance / c) * 1000.0
            total_delay_ms = delay_vortex_ms + delay_steering_ms

            new_conf = base_params.copy()
            new_conf.update({'x': sub_x, 'y': sub_y, 'angle': sub_orientation, 'delay_ms': total_delay_ms, 
                             'is_group_master': (n==0), 'param_locks': {'angle': True, 'delay': True, 'gain': False, 'polarity': False}})
            new_configs.append(new_conf)
            
        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "Vortex", array_params, ref_sub_idx)
        self.status_bar.showMessage(f"Array Vortex di {num_elements} elementi creato.", 5000)

    def setup_cardioid_pair(self, center_x, center_y, spacing, c, angle_deg, base_params, ref_sub_idx):
        angle_rad = np.radians(angle_deg)
        dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)

        front_sub = base_params.copy()
        front_sub.update({'x': center_x + dir_x * spacing / 2, 'y': center_y + dir_y * spacing / 2,
                          'angle': angle_rad, 'delay_ms': 0, 'is_group_master': True, 'polarity': 1,
                          'param_locks':{'angle':True,'delay':True,'gain':False,'polarity':True}})
        
        rear_sub = base_params.copy()
        rear_sub.update({'x': center_x - dir_x * spacing / 2, 'y': center_y - dir_y * spacing / 2,
                         'angle': angle_rad, 'delay_ms': (spacing / c) * 1000.0, 'polarity': -1, 'is_group_master': False,
                         'param_locks':{'angle':True,'delay':True,'gain':False,'polarity':True}})
        
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
            new_conf.update({'x':sub_x, 'y':sub_y, 'angle': angle_rad, 'delay_ms':delay_ms, 'is_group_master':(k==0),  # Master set to the first element
                             'param_locks':{'angle':True,'delay':True,'gain':False,'polarity':False}})
            new_configs.append(new_conf)

        self._normalize_delays(new_configs)
        self._add_new_subs_as_group(new_configs, "End-Fire", {'steering_deg': angle_deg}, ref_sub_idx)
        self.status_bar.showMessage(f"Array End-Fire di {num_elements} elementi creato.", 5000)
    
    def select_prev_target_area(self):
        if not self.lista_target_areas: self.current_target_area_idx = -1
        else: self.current_target_area_idx = (self.current_target_area_idx - 1 + len(self.lista_target_areas)) % len(self.lista_target_areas)
        self.update_ui_for_selected_target_area()
    def select_next_target_area(self):
        if not self.lista_target_areas: self.current_target_area_idx = -1
        else: self.current_target_area_idx = (self.current_target_area_idx + 1) % len(self.lista_target_areas)
        self.update_ui_for_selected_target_area()

    def _add_vtx_to_current_target_area(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            area = self.lista_target_areas[self.current_target_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area['punti'].append([new_x, new_y])
            self.update_ui_for_selected_target_area()
        else:
            self.error_text_target_area_mgmt.setText("Selezionare prima un'area target o crearne una nuova.")

    def _add_new_area_data(self, area_list, default_vertices, base_name, next_id_attr, activate=True):
        new_id = getattr(self, next_id_attr)
        snapped_default_vertices = [[self.snap_to_grid(p[0]), self.snap_to_grid(p[1])] for p in default_vertices]
        area_data = { 'id': new_id, 'nome': f"{base_name} {new_id}", 'punti': snapped_default_vertices, 'active': activate, 'plots': [] }
        area_list.append(area_data); setattr(self, next_id_attr, new_id + 1)
        return len(area_list) - 1
    def _get_area_center_and_default_size(self):
        xlims, ylims = self.ax.get_xlim(), self.ax.get_ylim(); cx, cy = np.mean(xlims), np.mean(ylims)
        size = min(xlims[1]-xlims[0], ylims[1]-ylims[0]) / 4.0
        return cx, cy, max(1.0, size)
    def add_new_target_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size(); hs = size/2.0
        default_verts = [ [cx-hs, cy-hs], [cx+hs, cy-hs], [cx+hs, cy+hs], [cx-hs, cy+hs] ]
        self.current_target_area_idx = self._add_new_area_data(self.lista_target_areas, default_verts, "Target", 'next_target_area_id')
        self.update_ui_for_selected_target_area()
    def remove_selected_target_area_ui(self):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas):
            self.lista_target_areas.pop(self.current_target_area_idx)
            if not self.lista_target_areas: self.current_target_area_idx = -1
            elif self.current_target_area_idx >= len(self.lista_target_areas): self.current_target_area_idx = len(self.lista_target_areas) - 1
            self.update_ui_for_selected_target_area()
    def toggle_selected_target_area_active(self, checked):
        if 0 <= self.current_target_area_idx < len(self.lista_target_areas): self.lista_target_areas[self.current_target_area_idx]['active'] = checked; self.update_ui_for_selected_target_area()
    
    def update_ui_for_selected_target_area(self):
        is_valid_idx = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        self.btn_add_target_vtx.setEnabled(is_valid_idx)

        for w in [self.btn_prev_target_area, self.btn_next_target_area, self.btn_remove_selected_target_area, self.check_activate_selected_target_area, self.target_vtx_list_widget]: w.setEnabled(is_valid_idx)
        
        self.target_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_target_areas[self.current_target_area_idx]
            self.label_current_target_area.setText(f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})")
            try: self.check_activate_selected_target_area.toggled.disconnect()
            except TypeError: pass
            self.check_activate_selected_target_area.setChecked(area['active']); self.check_activate_selected_target_area.toggled.connect(self.toggle_selected_target_area_active)
            for i, p in enumerate(area['punti']):
                self.target_vtx_list_widget.addItem(f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
        else:
            self.label_current_target_area.setText("Nessuna Area Target")
        
        self.on_target_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()
        
    def on_target_vtx_selection_change(self):
        is_valid_area = 0 <= self.current_target_area_idx < len(self.lista_target_areas)
        selected_items = self.target_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)
        
        for w in [self.tb_target_vtx_x, self.tb_target_vtx_y, self.btn_update_target_vtx]:
            w.setEnabled(can_edit)
            
        if can_edit:
            vtx_idx = self.target_vtx_list_widget.currentRow()
            vtx = self.lista_target_areas[self.current_target_area_idx]['punti'][vtx_idx]
            self.tb_target_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_target_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_target_vtx_x.clear()
            self.tb_target_vtx_y.clear()

    def on_update_selected_target_vertex(self):
        if not (0 <= self.current_target_area_idx < len(self.lista_target_areas)): return
        vtx_idx = self.target_vtx_list_widget.currentRow()
        if vtx_idx < 0: return
        
        try:
            x = float(self.tb_target_vtx_x.text())
            y = float(self.tb_target_vtx_y.text())
            self.lista_target_areas[self.current_target_area_idx]['punti'][vtx_idx] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.update_ui_for_selected_target_area()
        except ValueError:
            self.error_text_target_area_mgmt.setText("Coordinate non valide.")

    def select_prev_avoidance_area(self):
        if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
        else: self.current_avoidance_area_idx = (self.current_avoidance_area_idx - 1 + len(self.lista_avoidance_areas)) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()
    def select_next_avoidance_area(self):
        if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
        else: self.current_avoidance_area_idx = (self.current_avoidance_area_idx + 1) % len(self.lista_avoidance_areas)
        self.update_ui_for_selected_avoidance_area()
    def add_new_avoidance_area_ui(self):
        cx, cy, size = self._get_area_center_and_default_size(); hs = size/2.0 * 0.8
        default_verts = [ [cx-hs, cy-hs], [cx+hs, cy-hs], [cx+hs, cy+hs], [cx-hs, cy+hs] ]
        self.current_avoidance_area_idx = self._add_new_area_data(self.lista_avoidance_areas, default_verts, "Evitamento", 'next_avoidance_area_id')
        self.update_ui_for_selected_avoidance_area()
    def remove_selected_avoidance_area_ui(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            self.lista_avoidance_areas.pop(self.current_avoidance_area_idx)
            if not self.lista_avoidance_areas: self.current_avoidance_area_idx = -1
            elif self.current_avoidance_area_idx >= len(self.lista_avoidance_areas): self.current_avoidance_area_idx = len(self.lista_avoidance_areas) - 1
            self.update_ui_for_selected_avoidance_area()
    def toggle_selected_avoidance_area_active(self, checked):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas): self.lista_avoidance_areas[self.current_avoidance_area_idx]['active'] = checked; self.update_ui_for_selected_avoidance_area()
    
    def _add_vtx_to_current_avoidance_area(self):
        if 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas):
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            cx_ax, cy_ax = self.ax.get_xlim(), self.ax.get_ylim()
            new_x = self.snap_to_grid(np.mean(cx_ax))
            new_y = self.snap_to_grid(np.mean(cy_ax))
            area['punti'].append([new_x, new_y])
            self.update_ui_for_selected_avoidance_area()
        else:
            self.error_text_avoid_area_mgmt.setText("Selezionare prima un'area di evitamento o crearne una nuova.")


    def update_ui_for_selected_avoidance_area(self):
        is_valid_idx = 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        self.btn_add_avoid_vtx.setEnabled(is_valid_idx)

        for w in [self.btn_prev_avoid_area, self.btn_next_avoid_area, self.btn_remove_selected_avoid_area, self.check_activate_selected_avoid_area, self.avoid_vtx_list_widget]: w.setEnabled(is_valid_idx)
        
        self.avoid_vtx_list_widget.clear()
        if is_valid_idx:
            area = self.lista_avoidance_areas[self.current_avoidance_area_idx]
            self.label_current_avoid_area.setText(f"{area['nome']} ({'Attiva' if area['active'] else 'Non Attiva'})")
            try: self.check_activate_selected_avoid_area.toggled.disconnect()
            except TypeError: pass
            self.check_activate_selected_avoid_area.setChecked(area['active']); self.check_activate_selected_avoid_area.toggled.connect(self.toggle_selected_avoidance_area_active)
            for i, p in enumerate(area['punti']):
                self.avoid_vtx_list_widget.addItem(f"Vertice {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
        else:
            self.label_current_avoid_area.setText("Nessuna Area di Evitamento")
        
        self.on_avoid_vtx_selection_change()
        self.full_redraw(preserve_view=True)
        self.update_optim_freq_fields_visibility()

    def on_avoid_vtx_selection_change(self):
        is_valid_area = 0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)
        selected_items = self.avoid_vtx_list_widget.selectedItems()
        can_edit = is_valid_area and bool(selected_items)
        
        for w in [self.tb_avoid_vtx_x, self.tb_avoid_vtx_y, self.btn_update_avoid_vtx]:
            w.setEnabled(can_edit)
            
        if can_edit:
            vtx_idx = self.avoid_vtx_list_widget.currentRow()
            vtx = self.lista_avoidance_areas[self.current_avoidance_area_idx]['punti'][vtx_idx]
            self.tb_avoid_vtx_x.setText(f"{vtx[0]:.2f}")
            self.tb_avoid_vtx_y.setText(f"{vtx[1]:.2f}")
        else:
            self.tb_avoid_vtx_x.clear()
            self.tb_avoid_vtx_y.clear()

    def on_update_selected_avoid_vertex(self):
        if not (0 <= self.current_avoidance_area_idx < len(self.lista_avoidance_areas)): return
        vtx_idx = self.avoid_vtx_list_widget.currentRow()
        if vtx_idx < 0: return
        
        try:
            x = float(self.tb_avoid_vtx_x.text())
            y = float(self.tb_avoid_vtx_y.text())
            self.lista_avoidance_areas[self.current_avoidance_area_idx]['punti'][vtx_idx] = [self.snap_to_grid(x), self.snap_to_grid(y)]
            self.update_ui_for_selected_avoidance_area()
        except ValueError:
            self.error_text_avoid_area_mgmt.setText("Coordinate non valide.")

    def update_grid_snap_params(self, *args):
        self.grid_snap_enabled = self.check_grid_snap_enabled.isChecked(); self.grid_show_enabled = self.check_show_grid.isChecked()
        try: self.grid_snap_spacing = float(self.tb_grid_snap_spacing.text())
        except: self.grid_snap_spacing = 0.25
        self.full_redraw(preserve_view=True)
        
    def get_slider_freq_val(self): return self.slider_freq.value()
    def on_freq_change_ui_qt(self, value): self.label_slider_freq_val.setText(f"{value} Hz"); self.trigger_spl_map_recalculation()
    def trigger_spl_map_recalculation(self, force_redraw=False, fit_view=False):
        if self.check_auto_spl_update.isChecked() or force_redraw:
            self.full_redraw(preserve_view=not fit_view)

    def full_redraw(self, preserve_view=False):
        self.visualizza_mappatura_spl(self.get_slider_freq_val(), preserve_view)

    def on_press_mpl(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        self.status_bar.clearMessage()
        
        for area_list, type_prefix in [(self.lista_target_areas, 'target'), (self.lista_avoidance_areas, 'avoid')]:
            for area_idx, area_data in enumerate(area_list):
                if not area_data.get('active', False): continue
                for vtx_idx, plot_artist in enumerate(area_data.get('plots', [])): 
                    if plot_artist and plot_artist.contains(event)[0]:
                        if type_prefix == 'target':
                            if self.current_target_area_idx != area_idx:
                                self.current_target_area_idx = area_idx
                                self.update_ui_for_selected_target_area()
                            self.target_vtx_list_widget.setCurrentRow(vtx_idx)
                        elif type_prefix == 'avoid':
                            if self.current_avoidance_area_idx != area_idx:
                                self.current_avoidance_area_idx = area_idx
                                self.update_ui_for_selected_avoidance_area()
                            self.avoid_vtx_list_widget.setCurrentRow(vtx_idx)

                        self.drag_object = (f'{type_prefix}_vtx', area_idx, vtx_idx)
                        self.original_mouse_pos = (event.xdata, event.ydata)
                        self.original_object_pos = tuple(area_data['punti'][vtx_idx]) 
                        return

        for vtx_idx, vtx_data in enumerate(self.punti_stanza):
            if vtx_data.get('plot') and vtx_data['plot'].contains(event)[0]:
                self.drag_object = ('stanza_vtx', vtx_idx)
                self.original_mouse_pos = (event.xdata, event.ydata)
                self.original_object_pos = tuple(vtx_data['pos'])
                self.selected_stanza_vtx_idx = vtx_idx
                self.update_stanza_vtx_editor()
                return

        for i in reversed(range(len(self.sorgenti))):
            sub = self.sorgenti[i]
            if sub.get('arrow_artist') and sub['arrow_artist'].contains(event)[0]:
                if sub.get('param_locks', {}).get('angle', False):
                    self.status_bar.showMessage(f"Angolo Sub ID:{sub.get('id', i+1)} bloccato.", 2000)
                    return
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = 'group_rotate' if sub.get('group_id') is not None else 'sub_rotate'
                self.original_object_angle = sub['angle']
                self.drag_object = (drag_type, i)
                if 'group' in drag_type: self.original_group_states = self._get_group_states(sub.get('group_id'))
                return
            
            if sub.get('rect_artist') and sub['rect_artist'].contains(event)[0]:
                self.current_sub_idx = i
                self.aggiorna_ui_sub_fields()
                self.original_mouse_pos = (event.xdata, event.ydata)
                drag_type = 'group_pos' if sub.get('group_id') is not None else 'sub_pos'
                self.original_object_pos = (sub['x'], sub['y'])
                self.drag_object = (drag_type, i)
                if 'group' in drag_type: self.original_group_states = self._get_group_states(sub.get('group_id'))
                return
                                
        self.drag_object = None

    def on_motion_mpl(self, event):
        # Always update SPL display if mouse is over axes, regardless of drag state
        # self.on_mouse_move_for_spl_display(event) 

        if self.drag_object is None or event.inaxes != self.ax or event.xdata is None: 
            return

        dx = event.xdata - self.original_mouse_pos[0]; dy = event.ydata - self.original_mouse_pos[1]
        obj_type = self.drag_object[0]
        
        redraw_needed = True

        if obj_type == 'sub_pos':
            main_idx = self.drag_object[1]
            self.sorgenti[main_idx]['x'] = self.snap_to_grid(self.original_object_pos[0] + dx)
            self.sorgenti[main_idx]['y'] = self.snap_to_grid(self.original_object_pos[1] + dy)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'group_pos':
            for s_state in self.original_group_states:
                orig_x, orig_y = s_state['original_pos']
                self.sorgenti[s_state['sub_idx']]['x'] = self.snap_to_grid(orig_x + dx)
                self.sorgenti[s_state['sub_idx']]['y'] = self.snap_to_grid(orig_y + dy)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'sub_rotate':
            main_idx = self.drag_object[1]
            sub = self.sorgenti[main_idx]
            self.sorgenti[main_idx]['angle'] = np.arctan2(event.xdata - sub['x'], event.ydata - sub['y'])
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'group_rotate':
            group_center = self.original_group_states[0]['group_center']
            initial_mouse_angle = np.arctan2(self.original_mouse_pos[1] - group_center[1], self.original_mouse_pos[0] - group_center[0])
            current_mouse_angle = np.arctan2(event.ydata - group_center[1], event.xdata - group_center[0])
            angle_delta = current_mouse_angle - initial_mouse_angle
            for s_state in self.original_group_states:
                sub_idx, orig_rel_pos, orig_angle = s_state['sub_idx'], s_state['rel_pos'], s_state['original_angle']
                new_rel_x = orig_rel_pos[0] * np.cos(angle_delta) - orig_rel_pos[1] * np.sin(angle_delta)
                new_rel_y = orig_rel_pos[0] * np.sin(angle_delta) + orig_rel_pos[1] * np.cos(angle_delta)
                self.sorgenti[sub_idx]['x'] = self.snap_to_grid(group_center[0] + new_rel_x)
                self.sorgenti[sub_idx]['y'] = self.snap_to_grid(group_center[1] + new_rel_y)
                self.sorgenti[sub_idx]['angle'] = (orig_angle + angle_delta) % (2 * np.pi)
            self.aggiorna_ui_sub_fields()
        elif obj_type == 'stanza_vtx':
            main_idx = self.drag_object[1]
            self.punti_stanza[main_idx]['pos'][0] = self.snap_to_grid(self.original_object_pos[0] + dx)
            self.punti_stanza[main_idx]['pos'][1] = self.snap_to_grid(self.original_object_pos[1] + dy)
            self.update_stanza_vtx_editor()
        elif obj_type in ['target_vtx', 'avoid_vtx']:
            area_type_prefix, area_idx, vtx_idx = self.drag_object[0].split('_')[0], self.drag_object[1], self.drag_object[2]
            area_list = self.lista_target_areas if 'target' in area_type_prefix else self.lista_avoidance_areas
            area_list[area_idx]['punti'][vtx_idx] = [self.snap_to_grid(self.original_object_pos[0] + dx), self.snap_to_grid(self.original_object_pos[1] + dy)]
            if area_type_prefix == 'target':
                self.update_ui_for_selected_target_area()
            else: 
                self.update_ui_for_selected_avoidance_area()
        else:
            redraw_needed = False

        if redraw_needed:
            self.full_redraw(preserve_view=True)

    def on_release_mpl(self, event):
        if self.drag_object:
            self.status_bar.showMessage(f"Rilasciato oggetto.", 2000)
            is_room_drag = self.drag_object and self.drag_object[0] == 'stanza_vtx'
            self.drag_object = None
            self.original_group_states = []
            self.trigger_spl_map_recalculation(force_redraw=True, fit_view=is_room_drag)

    def _get_group_states(self, group_id):
        states = [];
        if group_id is None: return states
        members = [s for s in self.sorgenti if s.get('group_id') == group_id]
        if not members: return states
        center_x = np.mean([s['x'] for s in members]); center_y = np.mean([s['y'] for s in members])
        for s in members:
            states.append({'sub_idx': self.sorgenti.index(s), 'original_pos': (s['x'], s['y']), 'original_angle': s['angle'], 'rel_pos': (s['x'] - center_x, s['y'] - center_y), 'group_center': (center_x, center_y)})
        return states

    def visualizza_mappatura_spl(self, frequenza, preserve_view=False):
        current_xlim, current_ylim = None, None
        if preserve_view and self.ax.lines:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
        
        self.ax.cla()

        # --- START: FIX FOR DARK THEME ---
        # Set a visible background color for the figure and the axes area
        self.plot_canvas.figure.set_facecolor("#323232")  # Dark gray for the outer area
        self.ax.set_facecolor("#404040")                 # Lighter gray for the plot area

        # Make the axes lines (spines) and labels white so they are visible
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        # --- END: FIX FOR DARK THEME ---
        
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.set_xlabel("X (m)", loc='right')
        self.ax.set_ylabel("Y (m)", loc='top')
        
        if hasattr(self, '_cax_for_colorbar_spl') and self._cax_for_colorbar_spl:
            try: self.plot_canvas.figure.delaxes(self._cax_for_colorbar_spl)
            except (KeyError, AttributeError): pass
        self._cax_for_colorbar_spl = None

        self.plot_canvas.figure.subplots_adjust(right=0.92)
        
        room_points = [p['pos'] for p in self.punti_stanza]
        if room_points and len(room_points) >= 3 and self.sorgenti:
            try:
                min_spl=float(self.tb_spl_min.text()); max_spl=float(self.tb_spl_max.text())
                c_val=float(self.tb_velocita_suono.text()); grid_res=float(self.tb_grid_res_spl.text())
                if not(c_val > 0 and grid_res > 0 and min_spl < max_spl): raise ValueError()
                
                min_x_room=min(p[0] for p in room_points); max_x_room=max(p[0] for p in room_points)
                min_y_room=min(p[1] for p in room_points); max_y_room=max(p[1] for p in room_points)
                x=np.arange(min_x_room, max_x_room + grid_res, grid_res); y=np.arange(min_y_room, max_y_room + grid_res, grid_res)
                
                if len(x)>=2 and len(y)>=2:
                    X, Y = np.meshgrid(x,y)
                    SPL_map_plot = np.full(X.shape, np.nan)
                    room_mask = Path(room_points).contains_points(np.vstack((X.ravel(),Y.ravel())).T).reshape(X.shape)
                    
                    if np.any(room_mask):
                        points_to_calc_x = X[room_mask]
                        points_to_calc_y = Y[room_mask]
                        spl_values = calculate_spl_vectorized(points_to_calc_x, points_to_calc_y, frequenza, c_val, self.sorgenti)
                        self.current_spl_map = np.copy(SPL_map_plot) # Store the current SPL map for mouse hovering
                        self.current_spl_map[room_mask] = spl_values

                        masked_spl_map = np.ma.masked_where(~room_mask, self.current_spl_map)
                        if masked_spl_map.count() > 0:
                            contour = self.ax.contourf(X, Y, masked_spl_map, levels=100, cmap='jet', alpha=0.75, extend='both', vmin=min_spl, vmax=max_spl, zorder=0.3)
                            self.plot_canvas.figure.subplots_adjust(right=0.85)
                            self._cax_for_colorbar_spl = self.plot_canvas.figure.add_axes([0.87, 0.15, 0.03, 0.7])
                            self.plot_canvas.figure.colorbar(contour, cax=self._cax_for_colorbar_spl, label="SPL (dB)")
            except Exception as e:
                print(f"Errore calcolo/disegno mappa SPL: {e}")
                self.current_spl_map = None # Reset map on error

        self.disegna_elementi_statici_senza_spl()
        
        self.ax.set_aspect('equal', adjustable='box')
        if preserve_view and current_xlim is not None:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        else:
            self.auto_fit_view_to_room()
            
        self.plot_canvas.canvas.draw_idle()
    
    def disegna_elementi_statici_senza_spl(self):
        self.disegna_stanza_e_vertici()
        self.disegna_aree_target_e_avoidance()
        self.disegna_subwoofer_e_elementi()
        self.disegna_array_direction_indicators()
        self.disegna_griglia()

    def update_optim_freq_fields_visibility(self, *args):
        is_copertura = self.radio_copertura.isChecked()
        for w in [self.label_opt_freq_single_widget, self.tb_opt_freq_single]: w.setVisible(is_copertura)
        for w in [self.label_opt_freq_min_widget, self.tb_opt_freq_min, self.label_opt_freq_max_widget, self.tb_opt_freq_max, self.label_opt_n_freq_widget, self.tb_opt_n_freq]: w.setVisible(not is_copertura)
        
        has_active_target_areas = len(self.get_active_areas_points(self.lista_target_areas)) > 0
        has_active_avoidance_areas = len(self.get_active_areas_points(self.lista_avoidance_areas)) > 0
        
        show_balance_slider = has_active_target_areas and has_active_avoidance_areas

        self.label_balance_slider.setVisible(show_balance_slider)
        self.slider_balance.setVisible(show_balance_slider)
        self.label_balance_value.setVisible(show_balance_slider)

    def on_mouse_move_for_spl_display(self, event):
        if event.inaxes != self.ax or self.current_spl_map is None or event.xdata is None or event.ydata is None:
            self.status_bar.showMessage("Muovi il mouse sul grafico per visualizzare l'SPL.", 0)
            return

        try:
            x_coord = event.xdata
            y_coord = event.ydata
            
            if not self.punti_stanza:
                self.status_bar.showMessage("Definisci la stanza per visualizzare l'SPL al mouse.", 0)
                return

            min_x_room=min(p['pos'][0] for p in self.punti_stanza); max_x_room=max(p['pos'][0] for p in self.punti_stanza)
            min_y_room=min(p['pos'][1] for p in self.punti_stanza); max_y_room=max(p['pos'][1] for p in self.punti_stanza)
            
            grid_res_text = self.tb_grid_res_spl.text()
            if not grid_res_text:
                self.status_bar.showMessage("Risoluzione griglia SPL non definita. Impossibile calcolare SPL al mouse.", 0)
                return
            grid_res = float(grid_res_text)

            if grid_res <= 0:
                self.status_bar.showMessage("Risoluzione griglia SPL non valida. Impossibile calcolare SPL al mouse.", 0)
                return
            
            col_idx = int(np.floor((x_coord - min_x_room) / grid_res))
            row_idx = int(np.floor((y_coord - min_y_room) / grid_res))
            
            if 0 <= row_idx < self.current_spl_map.shape[0] and 0 <= col_idx < self.current_spl_map.shape[1]:
                spl_val = self.current_spl_map[row_idx, col_idx]
                if not np.isnan(spl_val):
                    self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): {spl_val:.1f} dB", 0)
                else:
                    self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Area Stanza", 0)
            else:
                self.status_bar.showMessage(f"SPL a ({x_coord:.2f}m, {y_coord:.2f}m): Fuori Limiti di Plot", 0)

        except ValueError:
            self.status_bar.showMessage("Dati non numerici per calcolo SPL al mouse.", 0)
        except Exception as e:
            self.status_bar.showMessage(f"Errore: {e}", 0)


    def unlock_dsp_for_optimization(self):
        if not self.sorgenti: self.status_bar.showMessage("Nessun subwoofer da sbloccare.", 3000); return
        unlocked_count = 0
        for sub in self.sorgenti:
            if sub.get('group_id') is None:
                for param in ['delay', 'gain', 'polarity', 'angle']:
                    if sub['param_locks'].get(param, False):
                        sub['param_locks'][param] = False
                        unlocked_count += 1
        self.aggiorna_ui_sub_fields(); self.status_bar.showMessage(f"Parametri DSP sbloccati.", 3000)

    def avvia_ottimizzazione_ui_qt(self):
        if self.optimization_thread is not None: return
        if not self.sorgenti: self.status_text_optim.setText("Aggiungere almeno un subwoofer."); return
        
        room_points_list = [p['pos'] for p in self.punti_stanza]
        if len(room_points_list) < 3: self.status_text_optim.setText("Definire una stanza valida."); return

        active_targets = self.get_active_areas_points(self.lista_target_areas)
        active_avoidances = self.get_active_areas_points(self.lista_avoidance_areas)
        
        if not active_targets and not active_avoidances:
            self.status_text_optim.setText("Attivare almeno un'area target o di evitamento per l'ottimizzazione."); 
            return

        try:
            criterion = self.radio_btn_group_crit.checkedButton().text(); pop_s=int(self.tb_opt_pop_size.text()); gens=int(self.tb_opt_generations.text()); 
            
            max_spl_avoid=float(self.tb_max_spl_avoid.text())
            target_min_spl_desired_val = float(self.tb_target_min_spl_desired.text())
            
            balance_target_avoidance_val = self.slider_balance.value()
            
            c_val=float(self.tb_velocita_suono.text()); grid_res=float(self.tb_grid_res_spl.text())
            optim_f_s, optim_f_min, optim_f_max, optim_n_f = None, None, None, None
            self.last_optim_criterion = criterion

            if criterion == 'Copertura SPL':
                optim_f_s = float(self.tb_opt_freq_single.text())
                self.last_optim_freq_s = optim_f_s
            else:
                optim_f_min=float(self.tb_opt_freq_min.text())
                optim_f_max=float(self.tb_opt_freq_max.text())
                optim_n_f=int(self.tb_opt_n_freq.text())
                self.last_optim_freq_min = optim_f_min
                self.last_optim_freq_max = optim_f_max
                if optim_n_f < 2:
                    self.status_text_optim.setText("Per 'Omogeneità', usare almeno 2 punti frequenza.")
                    return
        except (ValueError, AttributeError) as e: self.status_text_optim.setText(f"Errore parametri ottimizzazione: {e}"); return
        
        self.optimization_thread = QThread()
        self.optimization_worker = OptimizationWorker(criterion, optim_f_s, optim_f_min, optim_f_max, optim_n_f, pop_s, gens, c_val, grid_res, 
            room_points_list, active_targets, active_avoidances,
            max_spl_avoid, target_min_spl_desired_val, balance_target_avoidance_val,
            [s.copy() for s in self.sorgenti], [s['param_locks'].copy() for s in self.sorgenti])
        
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
        if self.optimization_worker: self.optimization_worker.request_stop()
        
    def handle_optim_finished(self, best_solution):
        self.btn_optimize_widget.setEnabled(True)
        self.btn_stop_optimize_widget.setEnabled(False)

        new_freq = None
        if self.last_optim_criterion == 'Copertura SPL':
            if self.last_optim_freq_s is not None:
                new_freq = self.last_optim_freq_s
        elif self.last_optim_criterion == 'Omogeneità SPL':
            if self.last_optim_freq_min is not None and self.last_optim_freq_max is not None:
            # Usiamo la media aritmetica del range come frequenza di visualizzazione
                new_freq = (self.last_optim_freq_min + self.last_optim_freq_max) / 2.0

        if new_freq is not None:
            # Assicuriamoci che il valore sia nei limiti dello slider per evitare errori
            slider_min = self.slider_freq.minimum()
            slider_max = self.slider_freq.maximum()
            clamped_freq = max(slider_min, min(slider_max, new_freq))
        
        # Impostiamo il valore dello slider. Questo aggiornerà automaticamente
        # anche l'etichetta del valore (es. "80 Hz")
        self.slider_freq.setValue(int(clamped_freq))
        
        if best_solution:
            for i, sub_dsp in enumerate(best_solution):
                current_sub = self.sorgenti[i]
                current_sub['delay_ms'] = sub_dsp['delay_ms']
                current_sub['gain_db'] = sub_dsp['gain_db']
                current_sub['gain_lin'] = sub_dsp['gain_lin']
                current_sub['polarity'] = sub_dsp['polarity']
                if 'angle' in sub_dsp and not current_sub['param_locks'].get('angle', False):
                    current_sub['angle'] = sub_dsp['angle']

            self.aggiorna_ui_sub_fields()
            self.full_redraw(preserve_view=True)

    def on_optim_thread_finished(self):
        self.optimization_thread = None
        self.optimization_worker = None
        self.status_bar.showMessage("Thread di ottimizzazione terminato.", 3000)

    def update_optim_status_text(self, message):
        if hasattr(self, 'status_text_optim'):
            self.status_text_optim.setText(message)
        QApplication.processEvents()
    
    # ASSICURATI CHE QUESTA FUNZIONE SIA INDENTATA CORRETTAMENTE DENTRO LA CLASSE SubwooferSimApp

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
                            'SPL (dB)': sub.get('spl_rms'), 'group_id': sub.get('group_id')
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
            
            self.status_bar.showMessage(f"Progetto completo salvato in {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Errore di Salvataggio", f"Impossibile salvare il file di progetto:\n{e}")

    # CANCELLA LA VECCHIA FUNZIONE E SOSTITUISCILA CON QUESTA
# ASSICURATI CHE SIA INDENTATA CORRETTAMENTE DENTRO LA CLASSE

    def load_project_from_excel(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Carica Progetto Completo", "", "File Excel (*.xlsx)")
        if not filepath:
            return
            
        try:
            # PASSO 1: PULIZIA DELLO STATO CORRENTE
            self.sorgenti.clear()
            self.punti_stanza.clear()
            self.lista_target_areas.clear()
            self.lista_avoidance_areas.clear()
            self.lista_gruppi_array.clear()
            self.current_sub_idx = -1
            self.next_sub_id = 1
            self.next_group_id = 1
            self.next_target_area_id = 1
            self.next_avoidance_area_id = 1
            
            # PASSO 2: CARICAMENTO DATI DAI FOGLI
            try:
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
                    }
                    self.add_subwoofer(specific_config=config, redraw=False)
                if 'ID' in df_subs.columns and pd.notna(df_subs['ID'].max()):
                    self.next_sub_id = int(df_subs['ID'].max()) + 1
            except Exception:
                print("Foglio 'Subwoofers' non trovato o errore nel caricarlo. Ignorato.")

            try:
                df_room = pd.read_excel(filepath, sheet_name='Stanza')
                for index, row in df_room.iterrows():
                    self.punti_stanza.append({'pos': [row['X'], row['Y']], 'plot': None})
            except Exception:
                print("Foglio 'Stanza' non trovato o errore nel caricarlo. Ignorato.")

            try:
                df_target = pd.read_excel(filepath, sheet_name='Aree_Target')
                if not df_target.empty:
                    for area_id, group in df_target.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id), 'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]), 'punti': punti, 'plots': []
                        }
                        self.lista_target_areas.append(area_data)
                    if pd.notna(df_target['Area_ID'].max()):
                        self.next_target_area_id = int(df_target['Area_ID'].max()) + 1
            except Exception:
                print("Foglio 'Aree_Target' non trovato o errore nel caricarlo. Ignorato.")

            try:
                df_avoid = pd.read_excel(filepath, sheet_name='Aree_Evitamento')
                if not df_avoid.empty:
                    for area_id, group in df_avoid.groupby('Area_ID'):
                        punti = group[['Vertice_X', 'Vertice_Y']].values.tolist()
                        area_data = {
                            'id': int(area_id), 'nome': group['Nome'].iloc[0],
                            'active': bool(group['Attiva'].iloc[0]), 'punti': punti, 'plots': []
                        }
                        self.lista_avoidance_areas.append(area_data)
                    if pd.notna(df_avoid['Area_ID'].max()):
                        self.next_avoidance_area_id = int(df_avoid['Area_ID'].max()) + 1
            except Exception:
                print("Foglio 'Aree_Evitamento' non trovato o errore nel caricarlo. Ignorato.")
            
            # PASSO 3: AGGIORNAMENTO FINALE
            all_group_ids = {s['group_id'] for s in self.sorgenti if s['group_id'] is not None}
            for gid in all_group_ids:
                group_members = [s for s in self.sorgenti if s.get('group_id') == gid]
                if group_members:
                    group_members[0]['is_group_master'] = True

            if self.sorgenti: self.current_sub_idx = 0
            
            self.aggiorna_ui_sub_fields()
            self.update_ui_for_selected_target_area()
            self.update_ui_for_selected_avoidance_area()
            self.full_redraw()
            self.status_bar.showMessage(f"Progetto completo caricato da {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Errore di Caricamento", f"Impossibile caricare il file di progetto:\n{e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
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