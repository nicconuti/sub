"""Smoke test for the subwoofer simulation tool."""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.diagnostics import run_diagnostics, setup_logging
from core.config import SimulationConfig, SUB_DTYPE
from core.data_loader import DataLoader
from core.acoustic_engine import AcousticEngine
from core.exporter import ProjectExporter


def test_diagnostics():
    """Test diagnostics functionality."""
    print("Testing diagnostics...")
    
    diagnostics = run_diagnostics()
    
    print(f"✓ Python executable: {diagnostics['python_executable']}")
    print(f"✓ Found {len(diagnostics['libraries'])} libraries")
    
    if diagnostics['errors']:
        print(f"⚠ Found {len(diagnostics['errors'])} errors:")
        for error in diagnostics['errors']:
            print(f"  - {error}")
    else:
        print("✓ No diagnostic errors")
    
    return len(diagnostics['errors']) == 0


def test_config():
    """Test configuration module."""
    print("\\nTesting configuration...")
    
    config = SimulationConfig()
    
    # Test parameter ranges
    assert config.param_ranges['delay_ms'] == (0.0, 300.0)
    print("✓ Parameter ranges loaded")
    
    # Test default values
    assert 'sub_spl_rms' in config.default_values
    print("✓ Default values loaded")
    
    # Test validation
    assert config.validate_param_value('delay_ms', 150.0) == True
    assert config.validate_param_value('delay_ms', 500.0) == False
    print("✓ Parameter validation works")
    
    return True


def test_acoustic_engine():
    """Test acoustic engine."""
    print("\\nTesting acoustic engine...")
    
    engine = AcousticEngine()
    
    # Create test sources
    sources = np.array([
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1),
        (1.0, 0.0, 1.0, 1.0, 0.0, 5.0, 1)
    ], dtype=SUB_DTYPE)
    
    # Test validation
    assert engine.validate_sources(sources) == True
    print("✓ Source validation works")
    
    # Test SPL calculation
    points = np.array([[2.0, 0.0], [0.0, 2.0]])
    spl_values = engine.calculate_point_spl(points, 60.0, sources)
    
    assert len(spl_values) == 2
    assert not np.any(np.isnan(spl_values))
    print("✓ SPL calculation works")
    
    # Test wavelength calculation
    wavelength = engine.get_wavelength(60.0)
    expected = 343.0 / 60.0
    assert abs(wavelength - expected) < 0.01
    print("✓ Wavelength calculation works")
    
    return True


def test_data_loader():
    """Test data loader."""
    print("\\nTesting data loader...")
    
    loader = DataLoader()
    
    # Test file format validation
    assert loader.validate_file_format("test.xlsx") == True
    assert loader.validate_file_format("test.txt") == False
    print("✓ File format validation works")
    
    # Test config creation
    test_config = {
        'test_param': 123,
        'test_string': 'hello'
    }
    
    test_file = project_root / 'test_config.json'
    
    try:
        loader.save_json_config(test_config, test_file)
        loaded_config = loader.load_json_config(test_file)
        
        assert loaded_config['test_param'] == 123
        assert loaded_config['test_string'] == 'hello'
        print("✓ JSON config save/load works")
        
        # Clean up
        test_file.unlink()
        
    except Exception as e:
        print(f"✗ JSON config test failed: {e}")
        return False
    
    return True


def test_exporter():
    """Test exporter."""
    print("\\nTesting exporter...")
    
    exporter = ProjectExporter()
    
    # Test project data validation
    valid_sources = np.array([
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1)
    ], dtype=SUB_DTYPE)
    
    project_data = {
        'sources': valid_sources,
        'name': 'Test Project'
    }
    
    assert exporter.validate_export_data(project_data) == True
    print("✓ Export data validation works")
    
    # Test supported formats
    formats = exporter.get_supported_formats()
    assert 'xlsx' in formats
    assert 'json' in formats
    print("✓ Supported formats available")
    
    return True


def test_imports():
    """Test that all major modules can be imported."""
    print("\\nTesting module imports...")
    
    try:
        # Test core imports
        from core import diagnostics, config, data_loader, acoustic_engine, optimization, exporter
        print("✓ Core modules imported")
        
        # Test plot imports
        from plot import visualizer, plot_styles
        print("✓ Plot modules imported")
        
        # Test UI imports (this might fail if PyQt6 is not available)
        try:
            from ui import main_window, control_panel, dialogs
            print("✓ UI modules imported")
        except ImportError as e:
            print(f"⚠ UI modules not available: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    setup_logging(level=logging.WARNING)  # Reduce log noise
    
    print("=" * 50)
    print("SUBWOOFER SIMULATION TOOL - SMOKE TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_diagnostics,
        test_config,
        test_acoustic_engine,
        test_data_loader,
        test_exporter
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\\n" + "=" * 50)
    print("SMOKE TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All smoke tests PASSED!")
        return 0
    else:
        print("❌ Some smoke tests FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())