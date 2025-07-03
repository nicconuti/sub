"""Test diagnostics module."""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.diagnostics import run_diagnostics, setup_logging


class TestDiagnostics(unittest.TestCase):
    """Test cases for diagnostics module."""
    
    def test_run_diagnostics(self):
        """Test that diagnostics run without errors."""
        diagnostics = run_diagnostics()
        
        # Check that diagnostics dict has expected keys
        self.assertIn('python_executable', diagnostics)
        self.assertIn('libraries', diagnostics)
        self.assertIn('errors', diagnostics)
        
        # Check that python executable is a string
        self.assertIsInstance(diagnostics['python_executable'], str)
        
        # Check that libraries is a dict
        self.assertIsInstance(diagnostics['libraries'], dict)
        
        # Check that errors is a list
        self.assertIsInstance(diagnostics['errors'], list)
    
    def test_setup_logging(self):
        """Test that logging setup works."""
        # This should not raise an exception
        setup_logging()


if __name__ == '__main__':
    unittest.main()