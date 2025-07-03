"""Diagnostics module for checking library versions and environment."""

import sys
import os
import logging
from typing import Dict, Any


def run_diagnostics() -> Dict[str, Any]:
    """Run diagnostic checks for required libraries.
    
    Returns:
        Dict containing diagnostic information about the environment.
    """
    diagnostics = {
        "python_executable": sys.executable,
        "libraries": {},
        "errors": []
    }
    
    try:
        import matplotlib
        diagnostics["libraries"]["matplotlib"] = {
            "version": matplotlib.__version__,
            "path": os.path.dirname(matplotlib.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"matplotlib: {e}")
    
    try:
        import PyQt6
        diagnostics["libraries"]["PyQt6"] = {
            "path": os.path.dirname(PyQt6.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"PyQt6: {e}")
    
    try:
        import pandas
        diagnostics["libraries"]["pandas"] = {
            "version": pandas.__version__,
            "path": os.path.dirname(pandas.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"pandas: {e}")
    
    try:
        import openpyxl
        diagnostics["libraries"]["openpyxl"] = {
            "version": openpyxl.__version__,
            "path": os.path.dirname(openpyxl.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"openpyxl: {e}")
    
    try:
        import numba
        diagnostics["libraries"]["numba"] = {
            "version": numba.__version__,
            "path": os.path.dirname(numba.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"numba: {e}")
    
    try:
        import numpy
        diagnostics["libraries"]["numpy"] = {
            "version": numpy.__version__,
            "path": os.path.dirname(numpy.__file__)
        }
    except Exception as e:
        diagnostics["errors"].append(f"numpy: {e}")
    
    return diagnostics


def print_diagnostics() -> None:
    """Print diagnostic information to console."""
    print("--- DIAGNOSTICS ---")
    
    diagnostics = run_diagnostics()
    
    print(f"Python Executable: {diagnostics['python_executable']}")
    
    for lib_name, lib_info in diagnostics["libraries"].items():
        if "version" in lib_info:
            print(f"{lib_name} Version: {lib_info['version']}")
        print(f"{lib_name} Path: {lib_info['path']}")
    
    if diagnostics["errors"]:
        print("\nDIAGNOSTIC ERRORS:")
        for error in diagnostics["errors"]:
            print(f"  {error}")
    
    print("--- END DIAGNOSTICS ---\n")


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('subwoofer_sim.log')
        ]
    )


if __name__ == "__main__":
    print_diagnostics()