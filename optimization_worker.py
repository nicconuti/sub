import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from matplotlib.path import Path
import warnings

from constants import PARAM_RANGES, OPTIMIZATION_MUTATION_RATE
from spl_calculations import calculate_spl_vectorized


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


