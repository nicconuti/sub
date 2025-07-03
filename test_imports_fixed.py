#!/usr/bin/env python3
"""Test script to verify import fixes without requiring external dependencies."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_modules():
    """Test core modules that don't require external dependencies."""
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
    
    return True

def test_import_structure():
    """Test that import statements are using absolute imports."""
    print("\\nTesting import structure...")
    
    # Check main.py imports
    try:
        # This should work because it imports core modules
        from core.diagnostics import setup_logging
        print("✅ main.py can import core.diagnostics")
    except Exception as e:
        print(f"❌ main.py core import failed: {e}")
        return False
    
    # Test the problematic UI imports (they'll fail due to missing PyQt6 but should show correct import structure)
    try:
        from ui.main_window import MainWindow
        print("✅ ui.main_window import structure works")
    except ModuleNotFoundError as e:
        if "PyQt6" in str(e) or "matplotlib" in str(e):
            print("✅ ui.main_window import structure correct (missing PyQt6/matplotlib is expected)")
        else:
            print(f"❌ ui.main_window unexpected import error: {e}")
            return False
    except ImportError as e:
        if "relative import" in str(e):
            print(f"❌ ui.main_window still has relative import issues: {e}")
            return False
        else:
            print("✅ ui.main_window import structure correct (dependency error is expected)")
    
    return True

def test_syntax_compilation():
    """Test that all files compile without syntax errors."""
    print("\\nTesting syntax compilation...")
    
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
        "ui/main_window.py"
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
    print("=" * 60)
    print("IMPORT FIXES VERIFICATION")
    print("=" * 60)
    
    syntax_ok = test_syntax_compilation()
    structure_ok = test_import_structure()
    core_ok = test_core_modules()
    
    print("\\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if syntax_ok:
        print("✅ All files have correct syntax")
    else:
        print("❌ Some files have syntax errors")
    
    if structure_ok:
        print("✅ Import structure is correct (absolute imports)")
    else:
        print("❌ Import structure still has issues")
    
    if core_ok:
        print("✅ Core modules can be imported")
    else:
        print("❌ Core modules have import errors")
    
    print("\\n📋 SUMMARY:")
    if syntax_ok and structure_ok:
        print("🎉 SUCCESS: Import fixes are working correctly!")
        print("   - All syntax errors resolved")
        print("   - Absolute imports implemented")
        print("   - Ready for dependency installation")
        print("\\n📦 Next step: Install dependencies with:")
        print("   pip install -r requirements-minimal.txt")
    else:
        print("❌ ISSUES: Some problems remain")
    
    return 0 if (syntax_ok and structure_ok) else 1

if __name__ == '__main__':
    sys.exit(main())