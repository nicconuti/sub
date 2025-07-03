import unittest
from core.config import SimulationConfig


class TestSimulationConfig(unittest.TestCase):
    """Tests for SimulationConfig."""

    def test_defaults_present(self):
        cfg = SimulationConfig()
        self.assertIn('sub_spl_rms', cfg.default_values)
        self.assertIsInstance(cfg.acoustic_params['speed_of_sound'], float)

    def test_param_range_update_and_get(self):
        cfg = SimulationConfig()
        cfg.update_param_range('gain_db', -10.0, 10.0)
        self.assertEqual(cfg.get_param_range('gain_db'), (-10.0, 10.0))

    def test_validate_param_value(self):
        cfg = SimulationConfig()
        self.assertTrue(cfg.validate_param_value('gain_db', 0.0))
        self.assertFalse(cfg.validate_param_value('gain_db', -40.0))


if __name__ == '__main__':
    unittest.main()
