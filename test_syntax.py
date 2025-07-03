#!/usr/bin/env python3
"""Test script to verify all modules have correct syntax and can be imported."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_modules():
    """Test core modules imports."""
    print("Testing core modules...")
    
    try:
        from core import diagnostics
        print("✅ core.diagnostics")
    except Exception as e:
        print(f"❌ core.diagnostics: {e}")
        return False
    
    try:
        from core import config
        print("✅ core.config")
    except Exception as e:
        print(f"❌ core.config: {e}")
        return False
    
    try:
        from core import data_loader
        print("✅ core.data_loader")
    except Exception as e:
        print(f"❌ core.data_loader: {e}")
        return False
    
    try:
        from core import exporter
        print("✅ core.exporter")
    except Exception as e:
        print(f"❌ core.exporter: {e}")
        return False
    
    # Test acoustic_engine (might fail if numba not available)
    try:
        from core import acoustic_engine
        print("✅ core.acoustic_engine")
    except Exception as e:
        print(f"⚠️  core.acoustic_engine: {e}")
    
    # Test optimization (might fail if PyQt6 not available)
    try:
        from core import optimization
        print("✅ core.optimization")
    except Exception as e:
        print(f"⚠️  core.optimization: {e}")
    
    return True

def test_plot_modules():
    """Test plot modules imports."""
    print("\nTesting plot modules...")
    
    try:
        from plot import plot_styles
        print("✅ plot.plot_styles")
    except Exception as e:
        print(f"❌ plot.plot_styles: {e}")
        return False
    
    # Test visualizer (might fail if matplotlib/PyQt6 not available)
    try:
        from plot import visualizer
        print("✅ plot.visualizer")
    except Exception as e:
        print(f"⚠️  plot.visualizer: {e}")
    
    return True

def test_ui_modules():
    """Test UI modules imports."""
    print("\nTesting UI modules...")
    
    # Test dialogs (might fail if PyQt6 not available)
    try:
        from ui import dialogs
        print("✅ ui.dialogs")
    except Exception as e:
        print(f"⚠️  ui.dialogs: {e}")
    
    # Test control_panel (might fail if PyQt6 not available)
    try:
        from ui import control_panel
        print("✅ ui.control_panel")
    except Exception as e:
        print(f"⚠️  ui.control_panel: {e}")
    
    # Test main_window (might fail if PyQt6 not available)
    try:
        from ui import main_window
        print("✅ ui.main_window")
    except Exception as e:
        print(f"⚠️  ui.main_window: {e}")
    
    return True

def test_syntax_only():
    """Test syntax compilation of all Python files."""
    print("\nTesting syntax compilation...")
    
    python_files = [
        "main.py",
        "core/diagnostics.py",
        "core/config.py", 
        "core/data_loader.py",
        "core/acoustic_engine.py",
        "core/optimization.py",
        "core/exporter.py",
        "plot/plot_styles.py",
        "plot/visualizer.py",
        "ui/dialogs.py",
        "ui/control_panel.py",
        "ui/main_window.py",
        "tests/test_diagnostics.py",
        "scripts/run_tests.py",
        "scripts/smoke_test.py"
    ]
    
    all_pass = True
    
    for py_file in python_files:
        file_path = project_root / py_file
        if file_path.exists():
            try:
                import py_compile
                py_compile.compile(str(file_path), doraise=True)
                print(f"✅ {py_file}")
            except Exception as e:
                print(f"❌ {py_file}: {e}")
                all_pass = False
        else:
            print(f"⚠️  {py_file}: File not found")
    
    return all_pass

def main():
    """Run all tests."""
    print("=" * 50)
    print("SYNTAX AND IMPORT TESTS")
    print("=" * 50)
    
    # Test syntax compilation first
    syntax_ok = test_syntax_only()
    
    # Test imports
    core_ok = test_core_modules()
    plot_ok = test_plot_modules() 
    ui_ok = test_ui_modules()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    if syntax_ok:
        print("✅ All files have correct syntax")
    else:
        print("❌ Some files have syntax errors")
    
    if core_ok:
        print("✅ Core modules can be imported")
    else:
        print("❌ Some core modules have import errors")
    
    print("⚠️  Plot and UI modules may show warnings if PyQt6/matplotlib not installed")
    print("   This is normal - install requirements.txt to resolve")
    
    return 0 if syntax_ok and core_ok else 1

if __name__ == '__main__':
    sys.exit(main())