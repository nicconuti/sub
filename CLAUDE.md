# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Subwoofer Simulation Tool** - a PyQt6-based desktop application for acoustic engineering. The tool simulates subwoofer array configurations and visualizes sound pressure level (SPL) patterns in 2D space using advanced numerical acoustics calculations.

## Architecture

### Core Architecture
- **main.py**: Application entry point with PyQt6 setup and high-DPI scaling
- **core/**: Core simulation engine and acoustic calculations
  - `acoustic_engine.py`: Numba-accelerated SPL calculations with vectorized operations
  - `config.py`: Configuration management and default parameters
  - `data_loader.py`: Project data loading and management
  - `exporter.py`: Project export functionality
  - `optimization.py`: Array optimization algorithms
  - `diagnostics.py`: System diagnostics and logging setup
- **ui/**: PyQt6 GUI components
  - `main_window.py`: Main application window with plotting canvas
  - `control_panel.py`: Parameter controls and array configuration
  - `dialogs.py`: File dialogs and preferences
- **plot/**: Matplotlib visualization layer
  - `visualizer.py`: Matplotlib canvas integration with PyQt6
  - `plot_styles.py`: Visual styling and themes
- **src/**: Additional application modules (legacy structure)
  - `app.py`: Alternative entry point or legacy GUI
  - `array_configs.py`: Array configuration definitions
  - `calculations.py`: Mathematical calculations
  - `constants.py`: Application constants and data types
  - `optimization.py`: Optimization algorithms
  - `plotting.py`: Plotting utilities

### Key Technologies
- **PyQt6**: Modern GUI framework with Qt6 bindings
- **NumPy + Numba**: High-performance numerical computing with JIT compilation
- **Matplotlib**: Scientific plotting integrated with PyQt6
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing algorithms

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running the Application
```bash
# Main application
python main.py

# Alternative entry point (if available)
python src/app.py
```

### Testing
```bash
# Run all tests using custom test runner
python scripts/run_tests.py

# Run tests with pytest (development)
pytest

# Run tests with coverage
pytest --cov=core --cov=ui --cov=plot

# Run smoke tests
python scripts/smoke_test.py
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .

# Remove unused imports
autoflake --remove-all-unused-imports --in-place --recursive .
```

### Performance Analysis
```bash
# Memory profiling
python -m memory_profiler main.py

# Line profiling (add @profile decorators first)
kernprof -l -v main.py
```

## Key Implementation Details

### Acoustic Engine
- Uses **Numba JIT compilation** for performance-critical SPL calculations
- Implements **vectorized operations** for efficient array processing
- Supports **complex amplitude calculations** with phase relationships
- Includes **directivity patterns** and **distance-based attenuation**

### GUI Architecture
- **QMainWindow** with **QSplitter** layout for resizable panels
- **MatplotlibCanvas** embedded in PyQt6 for real-time visualization
- **Signal-slot pattern** for component communication
- **Threaded operations** for non-blocking simulations

### Data Management
- **Structured NumPy arrays** for subwoofer data (sub_dtype)
- **Pandas DataFrames** for parameter management
- **JSON/Excel export** capabilities for project data
- **Configuration persistence** across sessions

## Development Notes

### Performance Considerations
- Critical path calculations use **@numba.jit** decorators
- Large arrays processed with **vectorized NumPy operations**
- GUI updates use **Qt signals** to avoid blocking the main thread
- Memory usage monitored for large simulation grids

### GUI Best Practices
- All file operations use **QFileDialog** for native OS integration
- **High-DPI scaling** automatically handled for modern displays
- **Fusion style** applied for consistent cross-platform appearance
- **Status bar** provides real-time feedback during operations

### Testing Strategy
- **Unit tests** in `tests/` directory using standard unittest framework
- **Integration tests** via custom `scripts/run_tests.py` runner
- **Smoke tests** for basic functionality verification
- **Coverage reporting** available through pytest-cov