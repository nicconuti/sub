"""Dialog windows for the subwoofer simulation application."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QTabWidget, QWidget, QFileDialog, QMessageBox,
    QTextEdit, QSlider, QGroupBox, QListWidget, QListWidgetItem,
    QDialogButtonBox, QColorDialog, QFontDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FileDialogs:
    """Helper class for file dialogs."""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.last_directory = str(Path.home())
    
    def get_open_filename(
        self,
        title: str = "Open File",
        filters: str = "All files (*.*)",
        directory: Optional[str] = None
    ) -> str:
        """Get filename for opening."""
        if directory is None:
            directory = self.last_directory
        
        filename, _ = QFileDialog.getOpenFileName(
            self.parent, title, directory, filters
        )
        
        if filename:
            self.last_directory = str(Path(filename).parent)
        
        return filename
    
    def get_save_filename(
        self,
        title: str = "Save File",
        filters: str = "All files (*.*)",
        directory: Optional[str] = None
    ) -> str:
        """Get filename for saving."""
        if directory is None:
            directory = self.last_directory
        
        filename, _ = QFileDialog.getSaveFileName(
            self.parent, title, directory, filters
        )
        
        if filename:
            self.last_directory = str(Path(filename).parent)
        
        return filename
    
    def get_directory(
        self,
        title: str = "Select Directory",
        directory: Optional[str] = None
    ) -> str:
        """Get directory path."""
        if directory is None:
            directory = self.last_directory
        
        dirname = QFileDialog.getExistingDirectory(
            self.parent, title, directory
        )
        
        if dirname:
            self.last_directory = dirname
        
        return dirname


class PreferencesDialog(QDialog):
    """Preferences/Settings dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(500, 400)
        
        self._init_ui()
        self._load_defaults()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_general_tab()
        self._create_display_tab()
        self._create_simulation_tab()
        self._create_performance_tab()
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_preferences)
        button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        
        layout.addWidget(button_box)
    
    def _create_general_tab(self):
        """Create general preferences tab."""
        tab = QWidget()
        layout = QFormLayout()
        tab.setLayout(layout)
        
        # Application settings
        self.auto_save_checkbox = QCheckBox("Auto-save projects")
        layout.addRow("", self.auto_save_checkbox)
        
        self.auto_save_interval_spinbox = QSpinBox()
        self.auto_save_interval_spinbox.setRange(1, 60)
        self.auto_save_interval_spinbox.setValue(5)
        self.auto_save_interval_spinbox.setSuffix(" minutes")
        layout.addRow("Auto-save interval:", self.auto_save_interval_spinbox)
        
        self.remember_window_state_checkbox = QCheckBox("Remember window state")
        layout.addRow("", self.remember_window_state_checkbox)
        
        self.show_tooltips_checkbox = QCheckBox("Show tooltips")
        layout.addRow("", self.show_tooltips_checkbox)
        
        # Default values
        self.default_frequency_spinbox = QDoubleSpinBox()
        self.default_frequency_spinbox.setRange(20.0, 200.0)
        self.default_frequency_spinbox.setValue(60.0)
        self.default_frequency_spinbox.setSuffix(" Hz")
        layout.addRow("Default frequency:", self.default_frequency_spinbox)
        
        self.default_spl_spinbox = QDoubleSpinBox()
        self.default_spl_spinbox.setRange(70.0, 140.0)
        self.default_spl_spinbox.setValue(85.0)
        self.default_spl_spinbox.setSuffix(" dB")
        layout.addRow("Default SPL:", self.default_spl_spinbox)
        
        self.tab_widget.addTab(tab, "General")
    
    def _create_display_tab(self):
        """Create display preferences tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Visualization settings
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout()
        viz_group.setLayout(viz_layout)
        
        self.default_colormap_combo = QComboBox()
        self.default_colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'])
        viz_layout.addRow("Default colormap:", self.default_colormap_combo)
        
        self.plot_dpi_spinbox = QSpinBox()
        self.plot_dpi_spinbox.setRange(72, 300)
        self.plot_dpi_spinbox.setValue(100)
        viz_layout.addRow("Plot DPI:", self.plot_dpi_spinbox)
        
        self.anti_aliasing_checkbox = QCheckBox("Anti-aliasing")
        viz_layout.addRow("", self.anti_aliasing_checkbox)
        
        layout.addWidget(viz_group)
        
        # Colors
        colors_group = QGroupBox("Colors")
        colors_layout = QGridLayout()
        colors_group.setLayout(colors_layout)
        
        self.positive_source_color_button = QPushButton("Positive Sources")
        self.negative_source_color_button = QPushButton("Negative Sources")
        self.room_boundary_color_button = QPushButton("Room Boundary")
        self.target_area_color_button = QPushButton("Target Areas")
        self.avoidance_area_color_button = QPushButton("Avoidance Areas")
        
        colors_layout.addWidget(QLabel("Positive Sources:"), 0, 0)
        colors_layout.addWidget(self.positive_source_color_button, 0, 1)
        colors_layout.addWidget(QLabel("Negative Sources:"), 1, 0)
        colors_layout.addWidget(self.negative_source_color_button, 1, 1)
        colors_layout.addWidget(QLabel("Room Boundary:"), 2, 0)
        colors_layout.addWidget(self.room_boundary_color_button, 2, 1)
        colors_layout.addWidget(QLabel("Target Areas:"), 3, 0)
        colors_layout.addWidget(self.target_area_color_button, 3, 1)
        colors_layout.addWidget(QLabel("Avoidance Areas:"), 4, 0)
        colors_layout.addWidget(self.avoidance_area_color_button, 4, 1)
        
        layout.addWidget(colors_group)
        
        # Fonts
        fonts_group = QGroupBox("Fonts")
        fonts_layout = QFormLayout()
        fonts_group.setLayout(fonts_layout)
        
        self.axis_font_button = QPushButton("Select Font")
        self.label_font_button = QPushButton("Select Font")
        self.title_font_button = QPushButton("Select Font")
        
        fonts_layout.addRow("Axis labels:", self.axis_font_button)
        fonts_layout.addRow("Text labels:", self.label_font_button)
        fonts_layout.addRow("Titles:", self.title_font_button)
        
        layout.addWidget(fonts_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Display")
    
    def _create_simulation_tab(self):
        """Create simulation preferences tab."""
        tab = QWidget()
        layout = QFormLayout()
        tab.setLayout(layout)
        
        # Calculation settings
        self.default_grid_resolution_spinbox = QSpinBox()
        self.default_grid_resolution_spinbox.setRange(50, 500)
        self.default_grid_resolution_spinbox.setValue(100)
        layout.addRow("Default grid resolution:", self.default_grid_resolution_spinbox)
        
        self.speed_of_sound_spinbox = QDoubleSpinBox()
        self.speed_of_sound_spinbox.setRange(300.0, 400.0)
        self.speed_of_sound_spinbox.setValue(343.0)
        self.speed_of_sound_spinbox.setSuffix(" m/s")
        layout.addRow("Speed of sound:", self.speed_of_sound_spinbox)
        
        self.auto_calculate_checkbox = QCheckBox("Auto-calculate on parameter change")
        layout.addRow("", self.auto_calculate_checkbox)
        
        # Optimization settings
        self.default_population_size_spinbox = QSpinBox()
        self.default_population_size_spinbox.setRange(10, 200)
        self.default_population_size_spinbox.setValue(50)
        layout.addRow("Default population size:", self.default_population_size_spinbox)
        
        self.default_generations_spinbox = QSpinBox()
        self.default_generations_spinbox.setRange(10, 500)
        self.default_generations_spinbox.setValue(100)
        layout.addRow("Default generations:", self.default_generations_spinbox)
        
        self.default_mutation_rate_spinbox = QDoubleSpinBox()
        self.default_mutation_rate_spinbox.setRange(0.01, 0.5)
        self.default_mutation_rate_spinbox.setValue(0.1)
        self.default_mutation_rate_spinbox.setSingleStep(0.01)
        layout.addRow("Default mutation rate:", self.default_mutation_rate_spinbox)
        
        self.tab_widget.addTab(tab, "Simulation")
    
    def _create_performance_tab(self):
        """Create performance preferences tab."""
        tab = QWidget()
        layout = QFormLayout()
        tab.setLayout(layout)
        
        # Threading settings
        self.use_multithreading_checkbox = QCheckBox("Use multithreading")
        layout.addRow("", self.use_multithreading_checkbox)
        
        self.max_threads_spinbox = QSpinBox()
        self.max_threads_spinbox.setRange(1, 16)
        self.max_threads_spinbox.setValue(4)
        layout.addRow("Max threads:", self.max_threads_spinbox)
        
        # Memory settings
        self.cache_size_spinbox = QSpinBox()
        self.cache_size_spinbox.setRange(100, 2000)
        self.cache_size_spinbox.setValue(500)
        self.cache_size_spinbox.setSuffix(" MB")
        layout.addRow("Cache size:", self.cache_size_spinbox)
        
        self.enable_gpu_checkbox = QCheckBox("Enable GPU acceleration (if available)")
        layout.addRow("", self.enable_gpu_checkbox)
        
        # Logging
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        self.log_level_combo.setCurrentText('INFO')
        layout.addRow("Log level:", self.log_level_combo)
        
        self.log_to_file_checkbox = QCheckBox("Log to file")
        layout.addRow("", self.log_to_file_checkbox)
        
        self.tab_widget.addTab(tab, "Performance")
    
    def _load_defaults(self):
        """Load default values."""
        # Set default values for all controls
        self.auto_save_checkbox.setChecked(True)
        self.remember_window_state_checkbox.setChecked(True)
        self.show_tooltips_checkbox.setChecked(True)
        self.anti_aliasing_checkbox.setChecked(True)
        self.auto_calculate_checkbox.setChecked(False)
        self.use_multithreading_checkbox.setChecked(True)
        self.enable_gpu_checkbox.setChecked(False)
        self.log_to_file_checkbox.setChecked(True)
    
    def load_preferences(self, config):
        """Load preferences from configuration."""
        # Load preferences from config object
        # This would read from the actual configuration
        pass
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get current preferences."""
        return {
            'general': {
                'auto_save': self.auto_save_checkbox.isChecked(),
                'auto_save_interval': self.auto_save_interval_spinbox.value(),
                'remember_window_state': self.remember_window_state_checkbox.isChecked(),
                'show_tooltips': self.show_tooltips_checkbox.isChecked(),
                'default_frequency': self.default_frequency_spinbox.value(),
                'default_spl': self.default_spl_spinbox.value()
            },
            'display': {
                'default_colormap': self.default_colormap_combo.currentText(),
                'plot_dpi': self.plot_dpi_spinbox.value(),
                'anti_aliasing': self.anti_aliasing_checkbox.isChecked()
            },
            'simulation': {
                'default_grid_resolution': self.default_grid_resolution_spinbox.value(),
                'speed_of_sound': self.speed_of_sound_spinbox.value(),
                'auto_calculate': self.auto_calculate_checkbox.isChecked(),
                'default_population_size': self.default_population_size_spinbox.value(),
                'default_generations': self.default_generations_spinbox.value(),
                'default_mutation_rate': self.default_mutation_rate_spinbox.value()
            },
            'performance': {
                'use_multithreading': self.use_multithreading_checkbox.isChecked(),
                'max_threads': self.max_threads_spinbox.value(),
                'cache_size': self.cache_size_spinbox.value(),
                'enable_gpu': self.enable_gpu_checkbox.isChecked(),
                'log_level': self.log_level_combo.currentText(),
                'log_to_file': self.log_to_file_checkbox.isChecked()
            }
        }
    
    def apply_preferences(self):
        """Apply current preferences without closing dialog."""
        # Apply preferences to the application
        pass
    
    def restore_defaults(self):
        """Restore default preferences."""
        self._load_defaults()


class SourcePropertiesDialog(QDialog):
    """Dialog for editing source properties."""
    
    def __init__(self, parent=None, source_data=None):
        super().__init__(parent)
        
        self.setWindowTitle("Source Properties")
        self.setModal(True)
        self.resize(400, 300)
        
        self.source_data = source_data or {}
        
        self._init_ui()
        self._load_source_data()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Source properties form
        form_layout = QFormLayout()
        
        # ID and Name
        self.source_id_edit = QLineEdit()
        self.source_id_edit.setReadOnly(True)
        form_layout.addRow("ID:", self.source_id_edit)
        
        self.source_name_edit = QLineEdit()
        form_layout.addRow("Name:", self.source_name_edit)
        
        # Position
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-100.0, 100.0)
        self.x_spinbox.setSuffix(" m")
        form_layout.addRow("X Position:", self.x_spinbox)
        
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-100.0, 100.0)
        self.y_spinbox.setSuffix(" m")
        form_layout.addRow("Y Position:", self.y_spinbox)
        
        # Acoustic properties
        self.spl_spinbox = QDoubleSpinBox()
        self.spl_spinbox.setRange(70.0, 140.0)
        self.spl_spinbox.setSuffix(" dB")
        form_layout.addRow("SPL @ 1m:", self.spl_spinbox)
        
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(-30.0, 10.0)
        self.gain_spinbox.setSuffix(" dB")
        form_layout.addRow("Gain:", self.gain_spinbox)
        
        self.delay_spinbox = QDoubleSpinBox()
        self.delay_spinbox.setRange(0.0, 300.0)
        self.delay_spinbox.setSuffix(" ms")
        form_layout.addRow("Delay:", self.delay_spinbox)
        
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItems(['Positive (+1)', 'Negative (-1)'])
        form_layout.addRow("Polarity:", self.polarity_combo)
        
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0.0, 360.0)
        self.angle_spinbox.setSuffix("°")
        form_layout.addRow("Angle:", self.angle_spinbox)
        
        # Physical properties
        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setRange(0.1, 2.0)
        self.width_spinbox.setSuffix(" m")
        form_layout.addRow("Width:", self.width_spinbox)
        
        self.depth_spinbox = QDoubleSpinBox()
        self.depth_spinbox.setRange(0.1, 2.0)
        self.depth_spinbox.setSuffix(" m")
        form_layout.addRow("Depth:", self.depth_spinbox)
        
        layout.addLayout(form_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
    
    def _load_source_data(self):
        """Load source data into the form."""
        if self.source_data:
            self.source_id_edit.setText(str(self.source_data.get('id', '')))
            self.source_name_edit.setText(self.source_data.get('name', ''))
            self.x_spinbox.setValue(self.source_data.get('x', 0.0))
            self.y_spinbox.setValue(self.source_data.get('y', 0.0))
            self.spl_spinbox.setValue(self.source_data.get('spl', 85.0))
            self.gain_spinbox.setValue(self.source_data.get('gain', 0.0))
            self.delay_spinbox.setValue(self.source_data.get('delay', 0.0))
            self.polarity_combo.setCurrentIndex(0 if self.source_data.get('polarity', 1) > 0 else 1)
            self.angle_spinbox.setValue(self.source_data.get('angle', 0.0))
            self.width_spinbox.setValue(self.source_data.get('width', 0.4))
            self.depth_spinbox.setValue(self.source_data.get('depth', 0.5))
    
    def get_source_data(self) -> Dict[str, Any]:
        """Get source data from the form."""
        return {
            'id': self.source_id_edit.text(),
            'name': self.source_name_edit.text(),
            'x': self.x_spinbox.value(),
            'y': self.y_spinbox.value(),
            'spl': self.spl_spinbox.value(),
            'gain': self.gain_spinbox.value(),
            'delay': self.delay_spinbox.value(),
            'polarity': 1 if self.polarity_combo.currentIndex() == 0 else -1,
            'angle': self.angle_spinbox.value(),
            'width': self.width_spinbox.value(),
            'depth': self.depth_spinbox.value()
        }


class OptimizationProgressDialog(QDialog):
    """Dialog showing optimization progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Optimization Progress")
        self.setModal(True)
        self.resize(400, 300)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Progress information
        self.status_label = QLabel("Initializing optimization...")
        layout.addWidget(self.status_label)
        
        # Progress details
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        layout.addWidget(self.progress_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.close_button.clicked.connect(self.accept)
    
    def update_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)
        self.progress_text.append(message)
    
    def optimization_finished(self):
        """Handle optimization finished."""
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)


class ExportDialog(QDialog):
    """Dialog for export options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Export Options")
        self.setModal(True)
        self.resize(400, 300)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Export format
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()
        format_group.setLayout(format_layout)
        
        self.excel_radio = QRadioButton("Excel (.xlsx)")
        self.json_radio = QRadioButton("JSON (.json)")
        self.csv_radio = QRadioButton("CSV (.csv)")
        self.excel_radio.setChecked(True)
        
        format_layout.addWidget(self.excel_radio)
        format_layout.addWidget(self.json_radio)
        format_layout.addWidget(self.csv_radio)
        
        layout.addWidget(format_group)
        
        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        self.include_metadata_checkbox = QCheckBox("Include metadata")
        self.include_sources_checkbox = QCheckBox("Include sources")
        self.include_areas_checkbox = QCheckBox("Include areas")
        self.include_results_checkbox = QCheckBox("Include simulation results")
        
        self.include_metadata_checkbox.setChecked(True)
        self.include_sources_checkbox.setChecked(True)
        self.include_areas_checkbox.setChecked(True)
        self.include_results_checkbox.setChecked(True)
        
        options_layout.addWidget(self.include_metadata_checkbox)
        options_layout.addWidget(self.include_sources_checkbox)
        options_layout.addWidget(self.include_areas_checkbox)
        options_layout.addWidget(self.include_results_checkbox)
        
        layout.addWidget(options_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
    
    def get_export_settings(self) -> Dict[str, Any]:
        """Get export settings."""
        if self.excel_radio.isChecked():
            format_type = 'excel'
        elif self.json_radio.isChecked():
            format_type = 'json'
        else:
            format_type = 'csv'
        
        return {
            'format': format_type,
            'include_metadata': self.include_metadata_checkbox.isChecked(),
            'include_sources': self.include_sources_checkbox.isChecked(),
            'include_areas': self.include_areas_checkbox.isChecked(),
            'include_results': self.include_results_checkbox.isChecked()
        }