# Constants for subwoofer simulation
import numpy as np

# --- Parametri Globali Iniziali ---
PARAM_RANGES = {
    'delay_ms': (0.0, 500.0),
    'gain_db': (-20.0, 6.0),
    'polarity': [-1, 1],  # Note: polarity is a list, not a tuple, for np.random.choice
    'spl_rms': (70.0, 140.0),
    'sub_width_depth': (0.1, 2.0),
    'angle': (0.0, 2 * np.pi)  # Angle should be a tuple for uniform
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

DEFAULT_MAX_SPL_AVOIDANCE = 65.0
DEFAULT_TARGET_MIN_SPL_DESIRED = 80.0
DEFAULT_BALANCE_SLIDER_VALUE = 50  # Default for 50/50 target/avoidance balance

OPTIMIZATION_MUTATION_RATE = 0.1  # Fixed mutation rate (10%)

FRONT_DIRECTIVITY_BEAMWIDTH_RAD = np.pi / 6
FRONT_DIRECTIVITY_GAIN_LIN = 10 ** (1.0 / 20.0)
DEFAULT_SIM_SPEED_OF_SOUND = 343.0
