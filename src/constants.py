import numpy as np

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
