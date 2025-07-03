import unittest
import numpy as np

from core.acoustic_engine import AcousticEngine, calculate_spl_vectorized
from core.config import SUB_DTYPE


class TestAcousticEngine(unittest.TestCase):
    """Tests for acoustic engine calculations."""

    def test_calculate_spl_vectorized_basic(self):
        px = np.array([0.0])
        py = np.array([0.0])
        freq = 60.0
        c_val = 343.0
        sources = np.array([(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1)], dtype=SUB_DTYPE)
        result = calculate_spl_vectorized(px, py, freq, c_val, sources)
        self.assertEqual(result.shape, px.shape)

    def test_get_wavelength(self):
        engine = AcousticEngine()
        wl = engine.get_wavelength(100.0)
        self.assertAlmostEqual(wl, 343.0 / 100.0)

    def test_validate_sources(self):
        engine = AcousticEngine()
        valid_sources = np.array([(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1)], dtype=SUB_DTYPE)
        self.assertTrue(engine.validate_sources(valid_sources))

        invalid_dtype = np.array([(0.0, 0.0)], dtype=[('x', float), ('y', float)])
        self.assertFalse(engine.validate_sources(invalid_dtype))


if __name__ == '__main__':
    unittest.main()
