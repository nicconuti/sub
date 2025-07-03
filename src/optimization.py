import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication
from matplotlib.path import Path

from .constants import OPTIMIZATION_MUTATION_RATE, PARAM_RANGES, sub_dtype
from .calculations import calculate_spl_vectorized

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

            if not locks.get("position", False) and self.sub_placement_area_points:
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

            if not locks.get("position", False) and np.random.rand() < rate and self.sub_placement_area_points:
                area_path = Path(self.sub_placement_area_points)
                min_x = min(p[0] for p in self.sub_placement_area_points)
                max_x = max(p[0] for p in self.sub_placement_area_points)
                min_y = min(p[1] for p in self.sub_placement_area_points)
                max_y = max(p[1] for p in self.sub_placement_area_points)
                
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
        COVERAGE_COST_FACTOR = 500.0
        AVOIDANCE_COST_FACTOR = 600.0
        LOBE_FOCUS_COST_FACTOR = 50.0
        UNIFORMITY_COST_FACTOR = 15.0
        GAIN_BALANCE_COST_FACTOR = 8.0
        RESULT_SYMMETRY_COST_FACTOR = 45.0
        COLLISION_COST_FACTOR = 20000.0 

        total_cost_accumulator = 0.0
        frequencies_to_test = []

        if self.criterion == "Omogeneità SPL" and self.optim_n_freq_val > 1:
            frequencies_to_test = np.linspace(
                self.optim_freq_min_val, self.optim_freq_max_val, self.optim_n_freq_val
            )
        elif self.criterion == "Copertura SPL":
            frequencies_to_test = [self.optim_freq_s]
        else: 
            frequencies_to_test = [(self.optim_freq_min_val + self.optim_freq_max_val) / 2.0]
        
        symmetry_axis_x = None
        if self.active_target_areas_points:
            list_of_centroids = []
            for area_points in self.active_target_areas_points:
                list_of_centroids.append(np.mean(np.array(area_points), axis=0))
            
            if list_of_centroids:
                meta_centroid = np.mean(np.array(list_of_centroids), axis=0)
                symmetry_axis_x = meta_centroid[0]

        for freq in frequencies_to_test:
            min_spl_in_target, avg_spl_in_target = -float('inf'), -float('inf')
            std_dev_in_target = float('inf')
            max_spl_in_avoidance, avg_spl_in_avoidance = -float('inf'), -float('inf')
            spl_map_target, mask_target, min_x_target, grid_res_spl = None, None, None, self.grid_res_spl

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
            
            if self.active_avoidance_areas_points:
                spl_map_avoid, mask_avoid, _, _ = self._calculate_spl_map_for_fitness_evaluation(
                    individual, freq, self.active_avoidance_areas_points, is_avoidance=True, return_grid_info=True
                )
                if spl_map_avoid is not None and np.any(mask_avoid):
                    values_avoid = spl_map_avoid[mask_avoid & ~np.isnan(spl_map_avoid)]
                    if values_avoid.size > 0:
                        max_spl_in_avoidance = np.max(values_avoid)
                        avg_spl_in_avoidance = np.mean(values_avoid)

            undershoot_db = max(0, self.target_min_spl_desired - min_spl_in_target)
            coverage_cost = (undershoot_db ** 2.5) * COVERAGE_COST_FACTOR

            overshoot_db = max(0, max_spl_in_avoidance - self.max_spl_avoidance)
            avoidance_cost = (overshoot_db ** 2) *  AVOIDANCE_COST_FACTOR

            uniformity_cost = std_dev_in_target * UNIFORMITY_COST_FACTOR
            
            spl_difference = avg_spl_in_target - avg_spl_in_avoidance if (avg_spl_in_target > -float('inf') and avg_spl_in_avoidance > -float('inf')) else 0
            lobe_focus_cost = max(0, 15 - spl_difference) * LOBE_FOCUS_COST_FACTOR

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
                    
                    if np.any(valid_mask):
                        diffs = np.abs(col1[valid_mask] - col2[valid_mask])**2
                        spl_differences.extend(diffs)

                if spl_differences:
                    result_symmetry_cost = np.mean(spl_differences) * RESULT_SYMMETRY_COST_FACTOR

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
            positions = np.array([[sub['x'], sub['y']] for sub in individual])
            sub_max_dim = max(self.sorgenti_configs_ref[0].get('width', 0.5), self.sorgenti_configs_ref[0].get('depth', 0.5))
            min_dist_sq = (sub_max_dim * 1.1)**2

            dist_sq_matrix = np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=-1)
            
            np.fill_diagonal(dist_sq_matrix, np.inf)
            collisions = dist_sq_matrix[np.triu_indices(num_subs, k=1)] < min_dist_sq
            
            collision_cost = np.sum(collisions) * COLLISION_COST_FACTOR
        final_total_cost = average_cost_over_freqs + gain_balance_cost

        return -final_total_cost
