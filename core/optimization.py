"""Optimization algorithms for subwoofer placement and configuration."""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable
import logging
from PyQt6.QtCore import QObject, pyqtSignal
from core.acoustic_engine import AcousticEngine
from core.config import SimulationConfig, OPTIMIZATION_MUTATION_RATE

logger = logging.getLogger(__name__)


class OptimizationWorker(QObject):
    """Worker class for running optimization in separate thread."""
    
    status_update = pyqtSignal(str)
    finished = pyqtSignal(object, float)

    def __init__(
        self,
        criterion: str,
        optim_freq_s: List[float],
        optim_freq_min_val: float,
        optim_freq_max_val: float,
        optim_n_freq_val: int,
        pop_size: int,
        generations: int,
        c_val: float,
        grid_res_spl: int,
        room_bounds: Tuple[float, float, float, float],
        target_areas: List[Dict],
        avoidance_areas: List[Dict],
        sub_placement_areas: List[Dict],
        existing_sources: np.ndarray,
        optimization_params: Dict[str, Any]
    ):
        super().__init__()
        self.criterion = criterion
        self.optim_freq_s = optim_freq_s
        self.optim_freq_min_val = optim_freq_min_val
        self.optim_freq_max_val = optim_freq_max_val
        self.optim_n_freq_val = optim_n_freq_val
        self.pop_size = pop_size
        self.generations = generations
        self.c_val = c_val
        self.grid_res_spl = grid_res_spl
        self.room_bounds = room_bounds
        self.target_areas = target_areas
        self.avoidance_areas = avoidance_areas
        self.sub_placement_areas = sub_placement_areas
        self.existing_sources = existing_sources
        self.optimization_params = optimization_params
        
        self.acoustic_engine = AcousticEngine(speed_of_sound=c_val)
        self.config = SimulationConfig()
        
        # Optimization state
        self.best_fitness = -np.inf
        self.best_solution = None
        self.generation_count = 0
        
    def run_optimization(self) -> None:
        """Run the optimization process."""
        try:
            self.status_update.emit("Starting optimization...")
            
            # Determine optimization frequencies
            if self.criterion == "single_freq":
                frequencies = self.optim_freq_s
            else:
                frequencies = np.linspace(
                    self.optim_freq_min_val, 
                    self.optim_freq_max_val, 
                    self.optim_n_freq_val
                )
            
            # Initialize population
            population = self._initialize_population()
            
            # Run genetic algorithm
            for generation in range(self.generations):
                self.generation_count = generation
                
                # Evaluate fitness for each individual
                fitness_scores = []
                for individual in population:
                    fitness = self._evaluate_fitness(individual, frequencies)
                    fitness_scores.append(fitness)
                
                # Update best solution
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > self.best_fitness:
                    self.best_fitness = fitness_scores[max_fitness_idx]
                    self.best_solution = population[max_fitness_idx].copy()
                
                # Report progress
                self.status_update.emit(
                    f"Generation {generation + 1}/{self.generations} - "
                    f"Best fitness: {self.best_fitness:.2f}"
                )
                
                # Create next generation
                population = self._create_next_generation(population, fitness_scores)
            
            self.status_update.emit("Optimization completed!")
            self.finished.emit(self.best_solution, self.best_fitness)
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            self.status_update.emit(f"Optimization failed: {e}")
            self.finished.emit(None, -np.inf)
    
    def _initialize_population(self) -> List[np.ndarray]:
        """Initialize random population of solutions."""
        population = []
        
        # Determine number of variables per individual
        num_sources = len(self.existing_sources)
        if num_sources == 0:
            raise ValueError("No sources to optimize")
        
        # Variables per source: x, y, delay, gain, polarity, angle
        vars_per_source = 6
        
        for _ in range(self.pop_size):
            individual = np.zeros(num_sources * vars_per_source)
            
            for i in range(num_sources):
                base_idx = i * vars_per_source
                
                # Position (x, y) - random within placement areas
                if self.sub_placement_areas:
                    area = np.random.choice(self.sub_placement_areas)
                    vertices = area['vertices']
                    x_coords = [v[0] for v in vertices]
                    y_coords = [v[1] for v in vertices]
                    individual[base_idx] = np.random.uniform(min(x_coords), max(x_coords))
                    individual[base_idx + 1] = np.random.uniform(min(y_coords), max(y_coords))
                else:
                    # Use room bounds
                    x_min, x_max, y_min, y_max = self.room_bounds
                    individual[base_idx] = np.random.uniform(x_min, x_max)
                    individual[base_idx + 1] = np.random.uniform(y_min, y_max)
                
                # Delay (ms)
                delay_range = self.config.get_param_range("delay_ms")
                individual[base_idx + 2] = np.random.uniform(*delay_range)
                
                # Gain (dB)
                gain_range = self.config.get_param_range("gain_db")
                individual[base_idx + 3] = np.random.uniform(*gain_range)
                
                # Polarity
                individual[base_idx + 4] = np.random.choice([-1, 1])
                
                # Angle (rad)
                angle_range = self.config.get_param_range("angle")
                individual[base_idx + 5] = np.random.uniform(*angle_range)
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: np.ndarray, frequencies: np.ndarray) -> float:
        """Evaluate fitness of an individual solution."""
        try:
            # Convert individual to source array
            sources = self._individual_to_sources(individual)
            
            # Calculate SPL field for each frequency
            total_cost = 0.0
            
            for freq in frequencies:
                X_grid, Y_grid, SPL_grid = self.acoustic_engine.calculate_spl_field(
                    self.room_bounds, self.grid_res_spl, freq, sources
                )
                
                # Calculate cost based on target and avoidance areas
                freq_cost = self._calculate_area_cost(X_grid, Y_grid, SPL_grid)
                total_cost += freq_cost
            
            # Average cost over frequencies
            average_cost = total_cost / len(frequencies)
            
            # Add penalty for constraint violations
            penalty = self._calculate_constraint_penalty(sources)
            
            final_fitness = average_cost - penalty
            
            return final_fitness
            
        except Exception as e:
            logger.error(f"Fitness evaluation error: {e}")
            return -np.inf
    
    def _individual_to_sources(self, individual: np.ndarray) -> np.ndarray:
        """Convert individual array to source array."""
        from .config import SUB_DTYPE
        
        num_sources = len(self.existing_sources)
        vars_per_source = 6
        
        sources = np.zeros(num_sources, dtype=SUB_DTYPE)
        
        for i in range(num_sources):
            base_idx = i * vars_per_source
            
            sources[i]['x'] = individual[base_idx]
            sources[i]['y'] = individual[base_idx + 1]
            
            # Convert gain from dB to linear
            gain_db = individual[base_idx + 3]
            sources[i]['gain_lin'] = 10 ** (gain_db / 20.0)
            
            sources[i]['delay_ms'] = individual[base_idx + 2]
            sources[i]['polarity'] = int(individual[base_idx + 4])
            sources[i]['angle'] = individual[base_idx + 5]
            
            # Copy fixed parameters from existing sources
            sources[i]['pressure_val_at_1m_relative_to_pref'] = \
                self.existing_sources[i]['pressure_val_at_1m_relative_to_pref']
        
        return sources
    
    def _calculate_area_cost(self, X_grid: np.ndarray, Y_grid: np.ndarray, SPL_grid: np.ndarray) -> float:
        """Calculate cost based on target and avoidance areas."""
        total_cost = 0.0
        
        # Target areas - reward high SPL
        for area in self.target_areas:
            mask = self._create_area_mask(X_grid, Y_grid, area['vertices'])
            if np.any(mask):
                target_spl = area.get('min_spl', 80.0)
                area_spl = SPL_grid[mask]
                # Reward SPL above target
                cost = np.mean(np.maximum(area_spl - target_spl, 0))
                total_cost += cost
        
        # Avoidance areas - penalize high SPL
        for area in self.avoidance_areas:
            mask = self._create_area_mask(X_grid, Y_grid, area['vertices'])
            if np.any(mask):
                max_spl = area.get('max_spl', 65.0)
                area_spl = SPL_grid[mask]
                # Penalize SPL above maximum
                penalty = np.mean(np.maximum(area_spl - max_spl, 0))
                total_cost -= penalty
        
        return total_cost
    
    def _create_area_mask(self, X_grid: np.ndarray, Y_grid: np.ndarray, vertices: List[List[float]]) -> np.ndarray:
        """Create boolean mask for area defined by vertices."""
        from matplotlib.path import Path
        
        # Create path from vertices
        path = Path(vertices)
        
        # Create mask
        points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        mask = path.contains_points(points)
        
        return mask.reshape(X_grid.shape)
    
    def _calculate_constraint_penalty(self, sources: np.ndarray) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Minimum distance between sources
        min_distance = 0.5  # meters
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                dist = np.sqrt(
                    (sources[i]['x'] - sources[j]['x'])**2 + 
                    (sources[i]['y'] - sources[j]['y'])**2
                )
                if dist < min_distance:
                    penalty += (min_distance - dist) * 100
        
        # Sources outside placement areas
        for i, source in enumerate(sources):
            if not self._is_point_in_placement_areas(source['x'], source['y']):
                penalty += 1000
        
        return penalty
    
    def _is_point_in_placement_areas(self, x: float, y: float) -> bool:
        """Check if point is within any placement area."""
        if not self.sub_placement_areas:
            return True
        
        from matplotlib.path import Path
        
        for area in self.sub_placement_areas:
            path = Path(area['vertices'])
            if path.contains_point((x, y)):
                return True
        
        return False
    
    def _create_next_generation(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Convert fitness scores to probabilities
        fitness_array = np.array(fitness_scores)
        if np.all(fitness_array == fitness_array[0]):
            # All fitness scores are the same, use uniform selection
            selection_probs = np.ones(len(population)) / len(population)
        else:
            # Normalize fitness scores
            min_fitness = np.min(fitness_array)
            if min_fitness < 0:
                fitness_array = fitness_array - min_fitness
            
            if np.sum(fitness_array) == 0:
                selection_probs = np.ones(len(population)) / len(population)
            else:
                selection_probs = fitness_array / np.sum(fitness_array)
        
        # Create new population
        for _ in range(self.pop_size):
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents."""
        if np.random.random() < 0.8:  # Crossover probability
            # Uniform crossover
            mask = np.random.random(len(parent1)) < 0.5
            child = np.where(mask, parent1, parent2)
        else:
            child = parent1.copy()
        
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate an individual."""
        mutated = individual.copy()
        
        num_sources = len(self.existing_sources)
        vars_per_source = 6
        
        for i in range(num_sources):
            base_idx = i * vars_per_source
            
            # Position mutation
            if np.random.random() < OPTIMIZATION_MUTATION_RATE:
                mutated[base_idx] += np.random.normal(0, 0.5)
                mutated[base_idx + 1] += np.random.normal(0, 0.5)
            
            # Delay mutation
            if np.random.random() < OPTIMIZATION_MUTATION_RATE:
                delay_range = self.config.get_param_range("delay_ms")
                mutated[base_idx + 2] += np.random.normal(0, 5)
                mutated[base_idx + 2] = np.clip(mutated[base_idx + 2], *delay_range)
            
            # Gain mutation
            if np.random.random() < OPTIMIZATION_MUTATION_RATE:
                gain_range = self.config.get_param_range("gain_db")
                mutated[base_idx + 3] += np.random.normal(0, 1)
                mutated[base_idx + 3] = np.clip(mutated[base_idx + 3], *gain_range)
            
            # Polarity mutation
            if np.random.random() < OPTIMIZATION_MUTATION_RATE:
                mutated[base_idx + 4] *= -1
            
            # Angle mutation
            if np.random.random() < OPTIMIZATION_MUTATION_RATE:
                mutated[base_idx + 5] += np.random.normal(0, 0.2)
                mutated[base_idx + 5] = mutated[base_idx + 5] % (2 * np.pi)
        
        return mutated


class OptimizationManager:
    """Manager class for different optimization algorithms."""
    
    def __init__(self, acoustic_engine: AcousticEngine):
        self.acoustic_engine = acoustic_engine
        self.config = SimulationConfig()
        self.logger = logging.getLogger(__name__)
    
    def optimize_array_placement(
        self,
        room_bounds: Tuple[float, float, float, float],
        target_areas: List[Dict],
        avoidance_areas: List[Dict],
        sub_placement_areas: List[Dict],
        optimization_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """Optimize subwoofer array placement.
        
        Args:
            room_bounds: (x_min, x_max, y_min, y_max)
            target_areas: List of target area definitions
            avoidance_areas: List of avoidance area definitions
            sub_placement_areas: List of valid placement areas
            optimization_params: Optimization parameters
            
        Returns:
            Tuple of (optimized_sources, final_fitness)
        """
        # This would typically be run in a separate thread
        # For now, return placeholder results
        self.logger.info("Starting array placement optimization")
        
        # Placeholder implementation
        num_sources = optimization_params.get('num_sources', 4)
        from .config import SUB_DTYPE
        
        sources = np.zeros(num_sources, dtype=SUB_DTYPE)
        
        # Random placement as placeholder
        x_min, x_max, y_min, y_max = room_bounds
        for i in range(num_sources):
            sources[i]['x'] = np.random.uniform(x_min, x_max)
            sources[i]['y'] = np.random.uniform(y_min, y_max)
            sources[i]['pressure_val_at_1m_relative_to_pref'] = 1.0
            sources[i]['gain_lin'] = 1.0
            sources[i]['angle'] = 0.0
            sources[i]['delay_ms'] = 0.0
            sources[i]['polarity'] = 1
        
        fitness = 0.0
        
        return sources, fitness