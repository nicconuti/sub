"""Control panel for subwoofer simulation parameters."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QSlider, QCheckBox, QRadioButton, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea,
    QButtonGroup, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, List, Any, Optional
import logging
import numpy as np

from core.config import SimulationConfig

logger = logging.getLogger(__name__)


class ControlPanel(QWidget):
    """Main control panel for simulation parameters."""
    
    # Signals
    parameter_changed = pyqtSignal()
    simulation_requested = pyqtSignal()
    optimization_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.config = SimulationConfig()
        
        # Parameter tracking
        self.parameters = {}
        self.updating_ui = False
        
        self._init_ui()
        self._connect_signals()
        
        self.logger.info("Control panel initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setFixedWidth(300)
        
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create scroll area for controls
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        main_layout.addWidget(scroll_area)
        
        # Create tab widget for different parameter categories
        self.tab_widget = QTabWidget()
        scroll_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_simulation_tab()
        self._create_sources_tab()
        self._create_room_tab()
        self._create_areas_tab()
        self._create_arrays_tab()
        self._create_optimization_tab()
        
        # Control buttons
        self._create_control_buttons()
        main_layout.addWidget(self.control_buttons_widget)
    
    def _create_simulation_tab(self):
        """Create simulation parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Frequency settings
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QFormLayout()
        freq_group.setLayout(freq_layout)
        
        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setRange(20.0, 200.0)
        self.frequency_spinbox.setValue(60.0)
        self.frequency_spinbox.setSuffix(" Hz")
        freq_layout.addRow("Frequency:", self.frequency_spinbox)
        
        self.speed_of_sound_spinbox = QDoubleSpinBox()
        self.speed_of_sound_spinbox.setRange(300.0, 400.0)
        self.speed_of_sound_spinbox.setValue(343.0)
        self.speed_of_sound_spinbox.setSuffix(" m/s")
        freq_layout.addRow("Speed of Sound:", self.speed_of_sound_spinbox)
        
        layout.addWidget(freq_group)
        
        # Grid settings
        grid_group = QGroupBox("Calculation Grid")
        grid_layout = QFormLayout()
        grid_group.setLayout(grid_layout)
        
        self.grid_resolution_spinbox = QSpinBox()
        self.grid_resolution_spinbox.setRange(50, 500)
        self.grid_resolution_spinbox.setValue(100)
        grid_layout.addRow("Resolution:", self.grid_resolution_spinbox)
        
        self.auto_bounds_checkbox = QCheckBox("Auto bounds")
        self.auto_bounds_checkbox.setChecked(True)
        grid_layout.addRow("", self.auto_bounds_checkbox)
        
        layout.addWidget(grid_group)
        
        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout()
        display_group.setLayout(display_layout)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'])
        display_layout.addRow("Colormap:", self.colormap_combo)
        
        self.auto_scale_checkbox = QCheckBox("Auto-scale SPL")
        self.auto_scale_checkbox.setChecked(True)
        display_layout.addRow("", self.auto_scale_checkbox)
        
        self.min_spl_spinbox = QDoubleSpinBox()
        self.min_spl_spinbox.setRange(-50.0, 150.0)
        self.min_spl_spinbox.setValue(60.0)
        self.min_spl_spinbox.setSuffix(" dB")
        display_layout.addRow("Min SPL:", self.min_spl_spinbox)
        
        self.max_spl_spinbox = QDoubleSpinBox()
        self.max_spl_spinbox.setRange(-50.0, 150.0)
        self.max_spl_spinbox.setValue(100.0)
        self.max_spl_spinbox.setSuffix(" dB")
        display_layout.addRow("Max SPL:", self.max_spl_spinbox)
        
        layout.addWidget(display_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Simulation")
    
    def _create_sources_tab(self):
        """Create sources parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Source list and controls
        sources_group = QGroupBox("Subwoofer Sources")
        sources_layout = QVBoxLayout()
        sources_group.setLayout(sources_layout)
        
        # Source list buttons
        list_buttons_layout = QHBoxLayout()
        self.add_source_button = QPushButton("Add Source")
        self.remove_source_button = QPushButton("Remove Source")
        self.duplicate_source_button = QPushButton("Duplicate")
        
        list_buttons_layout.addWidget(self.add_source_button)
        list_buttons_layout.addWidget(self.remove_source_button)
        list_buttons_layout.addWidget(self.duplicate_source_button)
        sources_layout.addLayout(list_buttons_layout)
        
        layout.addWidget(sources_group)
        
        # Current source settings
        current_source_group = QGroupBox("Current Source Settings")
        current_layout = QFormLayout()
        current_source_group.setLayout(current_layout)
        
        # Position
        self.source_x_spinbox = QDoubleSpinBox()
        self.source_x_spinbox.setRange(-50.0, 50.0)
        self.source_x_spinbox.setValue(0.0)
        self.source_x_spinbox.setSuffix(" m")
        current_layout.addRow("X Position:", self.source_x_spinbox)
        
        self.source_y_spinbox = QDoubleSpinBox()
        self.source_y_spinbox.setRange(-50.0, 50.0)
        self.source_y_spinbox.setValue(0.0)
        self.source_y_spinbox.setSuffix(" m")
        current_layout.addRow("Y Position:", self.source_y_spinbox)
        
        # SPL and Gain
        self.source_spl_spinbox = QDoubleSpinBox()
        self.source_spl_spinbox.setRange(70.0, 140.0)
        self.source_spl_spinbox.setValue(85.0)
        self.source_spl_spinbox.setSuffix(" dB")
        current_layout.addRow("SPL @ 1m:", self.source_spl_spinbox)
        
        self.source_gain_spinbox = QDoubleSpinBox()
        self.source_gain_spinbox.setRange(-30.0, 10.0)
        self.source_gain_spinbox.setValue(0.0)
        self.source_gain_spinbox.setSuffix(" dB")
        current_layout.addRow("Gain:", self.source_gain_spinbox)
        
        # Delay and Polarity
        self.source_delay_spinbox = QDoubleSpinBox()
        self.source_delay_spinbox.setRange(0.0, 300.0)
        self.source_delay_spinbox.setValue(0.0)
        self.source_delay_spinbox.setSuffix(" ms")
        current_layout.addRow("Delay:", self.source_delay_spinbox)
        
        self.source_polarity_combo = QComboBox()
        self.source_polarity_combo.addItems(['Positive (+1)', 'Negative (-1)'])
        current_layout.addRow("Polarity:", self.source_polarity_combo)
        
        # Angle
        self.source_angle_spinbox = QDoubleSpinBox()
        self.source_angle_spinbox.setRange(0.0, 360.0)
        self.source_angle_spinbox.setValue(0.0)
        self.source_angle_spinbox.setSuffix("°")
        current_layout.addRow("Angle:", self.source_angle_spinbox)
        
        layout.addWidget(current_source_group)
        
        # Global source settings
        global_group = QGroupBox("Global Settings")
        global_layout = QFormLayout()
        global_group.setLayout(global_layout)
        
        self.global_width_spinbox = QDoubleSpinBox()
        self.global_width_spinbox.setRange(0.1, 2.0)
        self.global_width_spinbox.setValue(0.4)
        self.global_width_spinbox.setSuffix(" m")
        global_layout.addRow("Default Width:", self.global_width_spinbox)
        
        self.global_depth_spinbox = QDoubleSpinBox()
        self.global_depth_spinbox.setRange(0.1, 2.0)
        self.global_depth_spinbox.setValue(0.5)
        self.global_depth_spinbox.setSuffix(" m")
        global_layout.addRow("Default Depth:", self.global_depth_spinbox)
        
        layout.addWidget(global_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Sources")
    
    def _create_room_tab(self):
        """Create room parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Room geometry
        room_group = QGroupBox("Room Geometry")
        room_layout = QVBoxLayout()
        room_group.setLayout(room_layout)
        
        # Room type selection
        room_type_layout = QHBoxLayout()
        self.room_type_group = QButtonGroup()
        
        self.rectangular_radio = QRadioButton("Rectangular")
        self.custom_radio = QRadioButton("Custom Shape")
        self.rectangular_radio.setChecked(True)
        
        self.room_type_group.addButton(self.rectangular_radio, 0)
        self.room_type_group.addButton(self.custom_radio, 1)
        
        room_type_layout.addWidget(self.rectangular_radio)
        room_type_layout.addWidget(self.custom_radio)
        room_layout.addLayout(room_type_layout)
        
        # Rectangular room settings
        rect_layout = QFormLayout()
        
        self.room_width_spinbox = QDoubleSpinBox()
        self.room_width_spinbox.setRange(1.0, 100.0)
        self.room_width_spinbox.setValue(30.0)
        self.room_width_spinbox.setSuffix(" m")
        rect_layout.addRow("Width:", self.room_width_spinbox)
        
        self.room_height_spinbox = QDoubleSpinBox()
        self.room_height_spinbox.setRange(1.0, 100.0)
        self.room_height_spinbox.setValue(30.0)
        self.room_height_spinbox.setSuffix(" m")
        rect_layout.addRow("Height:", self.room_height_spinbox)
        
        room_layout.addLayout(rect_layout)
        
        # Room control buttons
        room_buttons_layout = QHBoxLayout()
        self.edit_room_button = QPushButton("Edit Vertices")
        self.load_background_button = QPushButton("Load Background")
        
        room_buttons_layout.addWidget(self.edit_room_button)
        room_buttons_layout.addWidget(self.load_background_button)
        room_layout.addLayout(room_buttons_layout)
        
        layout.addWidget(room_group)
        
        # Grid and snap settings
        grid_group = QGroupBox("Grid & Snap")
        grid_layout = QFormLayout()
        grid_group.setLayout(grid_layout)
        
        self.show_grid_checkbox = QCheckBox("Show Grid")
        grid_layout.addRow("", self.show_grid_checkbox)
        
        self.snap_to_grid_checkbox = QCheckBox("Snap to Grid")
        grid_layout.addRow("", self.snap_to_grid_checkbox)
        
        self.grid_spacing_spinbox = QDoubleSpinBox()
        self.grid_spacing_spinbox.setRange(0.1, 5.0)
        self.grid_spacing_spinbox.setValue(0.25)
        self.grid_spacing_spinbox.setSuffix(" m")
        grid_layout.addRow("Grid Spacing:", self.grid_spacing_spinbox)
        
        layout.addWidget(grid_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Room")
    
    def _create_areas_tab(self):
        """Create areas parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Target areas
        target_group = QGroupBox("Target Areas")
        target_layout = QVBoxLayout()
        target_group.setLayout(target_layout)
        
        target_buttons_layout = QHBoxLayout()
        self.add_target_button = QPushButton("Add Target")
        self.remove_target_button = QPushButton("Remove Target")
        
        target_buttons_layout.addWidget(self.add_target_button)
        target_buttons_layout.addWidget(self.remove_target_button)
        target_layout.addLayout(target_buttons_layout)
        
        target_settings_layout = QFormLayout()
        
        self.target_min_spl_spinbox = QDoubleSpinBox()
        self.target_min_spl_spinbox.setRange(50.0, 120.0)
        self.target_min_spl_spinbox.setValue(80.0)
        self.target_min_spl_spinbox.setSuffix(" dB")
        target_settings_layout.addRow("Min SPL:", self.target_min_spl_spinbox)
        
        target_layout.addLayout(target_settings_layout)
        layout.addWidget(target_group)
        
        # Avoidance areas
        avoidance_group = QGroupBox("Avoidance Areas")
        avoidance_layout = QVBoxLayout()
        avoidance_group.setLayout(avoidance_layout)
        
        avoidance_buttons_layout = QHBoxLayout()
        self.add_avoidance_button = QPushButton("Add Avoidance")
        self.remove_avoidance_button = QPushButton("Remove Avoidance")
        
        avoidance_buttons_layout.addWidget(self.add_avoidance_button)
        avoidance_buttons_layout.addWidget(self.remove_avoidance_button)
        avoidance_layout.addLayout(avoidance_buttons_layout)
        
        avoidance_settings_layout = QFormLayout()
        
        self.avoidance_max_spl_spinbox = QDoubleSpinBox()
        self.avoidance_max_spl_spinbox.setRange(40.0, 100.0)
        self.avoidance_max_spl_spinbox.setValue(65.0)
        self.avoidance_max_spl_spinbox.setSuffix(" dB")
        avoidance_settings_layout.addRow("Max SPL:", self.avoidance_max_spl_spinbox)
        
        avoidance_layout.addLayout(avoidance_settings_layout)
        layout.addWidget(avoidance_group)
        
        # Placement areas
        placement_group = QGroupBox("Placement Areas")
        placement_layout = QVBoxLayout()
        placement_group.setLayout(placement_layout)
        
        placement_buttons_layout = QHBoxLayout()
        self.add_placement_button = QPushButton("Add Placement")
        self.remove_placement_button = QPushButton("Remove Placement")
        
        placement_buttons_layout.addWidget(self.add_placement_button)
        placement_buttons_layout.addWidget(self.remove_placement_button)
        placement_layout.addLayout(placement_buttons_layout)
        
        layout.addWidget(placement_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Areas")
    
    def _create_arrays_tab(self):
        """Create array configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Array type selection
        array_type_group = QGroupBox("Array Configuration")
        array_layout = QVBoxLayout()
        array_type_group.setLayout(array_layout)
        
        self.array_type_combo = QComboBox()
        self.array_type_combo.addItems(['None', 'Cardioid', 'Endfire', 'Line Array', 'Vortex'])
        array_layout.addWidget(self.array_type_combo)
        
        layout.addWidget(array_type_group)
        
        # Array parameters
        array_params_group = QGroupBox("Array Parameters")
        array_params_layout = QFormLayout()
        array_params_group.setLayout(array_params_layout)
        
        self.array_frequency_spinbox = QDoubleSpinBox()
        self.array_frequency_spinbox.setRange(20.0, 200.0)
        self.array_frequency_spinbox.setValue(60.0)
        self.array_frequency_spinbox.setSuffix(" Hz")
        array_params_layout.addRow("Array Frequency:", self.array_frequency_spinbox)
        
        self.array_radius_spinbox = QDoubleSpinBox()
        self.array_radius_spinbox.setRange(0.5, 10.0)
        self.array_radius_spinbox.setValue(1.0)
        self.array_radius_spinbox.setSuffix(" m")
        array_params_layout.addRow("Array Radius:", self.array_radius_spinbox)
        
        self.array_steering_spinbox = QDoubleSpinBox()
        self.array_steering_spinbox.setRange(-180.0, 180.0)
        self.array_steering_spinbox.setValue(0.0)
        self.array_steering_spinbox.setSuffix("°")
        array_params_layout.addRow("Steering Angle:", self.array_steering_spinbox)
        
        layout.addWidget(array_params_group)
        
        # Array generation
        array_gen_group = QGroupBox("Generate Array")
        array_gen_layout = QVBoxLayout()
        array_gen_group.setLayout(array_gen_layout)
        
        gen_layout = QFormLayout()
        
        self.num_sources_spinbox = QSpinBox()
        self.num_sources_spinbox.setRange(2, 20)
        self.num_sources_spinbox.setValue(4)
        gen_layout.addRow("Number of Sources:", self.num_sources_spinbox)
        
        array_gen_layout.addLayout(gen_layout)
        
        self.generate_array_button = QPushButton("Generate Array")
        array_gen_layout.addWidget(self.generate_array_button)
        
        layout.addWidget(array_gen_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Arrays")
    
    def _create_optimization_tab(self):
        """Create optimization parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Optimization settings
        optim_group = QGroupBox("Optimization Settings")
        optim_layout = QFormLayout()
        optim_group.setLayout(optim_layout)
        
        self.optim_criterion_combo = QComboBox()
        self.optim_criterion_combo.addItems(['Single Frequency', 'Frequency Range'])
        optim_layout.addRow("Criterion:", self.optim_criterion_combo)
        
        self.optim_frequency_spinbox = QDoubleSpinBox()
        self.optim_frequency_spinbox.setRange(20.0, 200.0)
        self.optim_frequency_spinbox.setValue(60.0)
        self.optim_frequency_spinbox.setSuffix(" Hz")
        optim_layout.addRow("Frequency:", self.optim_frequency_spinbox)
        
        self.optim_freq_min_spinbox = QDoubleSpinBox()
        self.optim_freq_min_spinbox.setRange(20.0, 200.0)
        self.optim_freq_min_spinbox.setValue(40.0)
        self.optim_freq_min_spinbox.setSuffix(" Hz")
        optim_layout.addRow("Min Frequency:", self.optim_freq_min_spinbox)
        
        self.optim_freq_max_spinbox = QDoubleSpinBox()
        self.optim_freq_max_spinbox.setRange(20.0, 200.0)
        self.optim_freq_max_spinbox.setValue(80.0)
        self.optim_freq_max_spinbox.setSuffix(" Hz")
        optim_layout.addRow("Max Frequency:", self.optim_freq_max_spinbox)
        
        self.optim_num_freq_spinbox = QSpinBox()
        self.optim_num_freq_spinbox.setRange(3, 20)
        self.optim_num_freq_spinbox.setValue(5)
        optim_layout.addRow("Number of Frequencies:", self.optim_num_freq_spinbox)
        
        layout.addWidget(optim_group)
        
        # Algorithm settings
        algo_group = QGroupBox("Algorithm Settings")
        algo_layout = QFormLayout()
        algo_group.setLayout(algo_layout)
        
        self.population_size_spinbox = QSpinBox()
        self.population_size_spinbox.setRange(10, 200)
        self.population_size_spinbox.setValue(50)
        algo_layout.addRow("Population Size:", self.population_size_spinbox)
        
        self.generations_spinbox = QSpinBox()
        self.generations_spinbox.setRange(10, 500)
        self.generations_spinbox.setValue(100)
        algo_layout.addRow("Generations:", self.generations_spinbox)
        
        self.mutation_rate_spinbox = QDoubleSpinBox()
        self.mutation_rate_spinbox.setRange(0.01, 0.5)
        self.mutation_rate_spinbox.setValue(0.1)
        self.mutation_rate_spinbox.setSingleStep(0.01)
        algo_layout.addRow("Mutation Rate:", self.mutation_rate_spinbox)
        
        layout.addWidget(algo_group)
        
        # Balance settings
        balance_group = QGroupBox("Balance Settings")
        balance_layout = QFormLayout()
        balance_group.setLayout(balance_layout)
        
        self.target_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_weight_slider.setRange(0, 100)
        self.target_weight_slider.setValue(50)
        self.target_weight_label = QLabel("50%")
        
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(self.target_weight_slider)
        weight_layout.addWidget(self.target_weight_label)
        balance_layout.addRow("Target Weight:", weight_layout)
        
        layout.addWidget(balance_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Optimization")
    
    def _create_control_buttons(self):
        """Create main control buttons."""
        self.control_buttons_widget = QWidget()
        layout = QVBoxLayout()
        self.control_buttons_widget.setLayout(layout)
        
        # Simulation buttons
        sim_layout = QHBoxLayout()
        
        self.run_simulation_button = QPushButton("Run Simulation")
        self.run_simulation_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.stop_simulation_button = QPushButton("Stop")
        self.stop_simulation_button.setEnabled(False)
        
        sim_layout.addWidget(self.run_simulation_button)
        sim_layout.addWidget(self.stop_simulation_button)
        layout.addLayout(sim_layout)
        
        # Optimization button
        self.run_optimization_button = QPushButton("Run Optimization")
        self.run_optimization_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        layout.addWidget(self.run_optimization_button)
        
        # Utility buttons
        util_layout = QHBoxLayout()
        
        self.clear_all_button = QPushButton("Clear All")
        self.auto_fit_button = QPushButton("Auto Fit")
        
        util_layout.addWidget(self.clear_all_button)
        util_layout.addWidget(self.auto_fit_button)
        layout.addLayout(util_layout)
    
    def _connect_signals(self):
        """Connect UI signals to handlers."""
        # Simulation parameters
        self.frequency_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.speed_of_sound_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.grid_resolution_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.auto_scale_checkbox.toggled.connect(self._on_parameter_changed)
        
        # Sources
        self.source_x_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.source_y_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.source_spl_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.source_gain_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.source_delay_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.source_polarity_combo.currentIndexChanged.connect(self._on_parameter_changed)
        self.source_angle_spinbox.valueChanged.connect(self._on_parameter_changed)
        
        # Room
        self.room_type_group.buttonClicked.connect(self._on_parameter_changed)
        self.room_width_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.room_height_spinbox.valueChanged.connect(self._on_parameter_changed)
        
        # Areas
        self.target_min_spl_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.avoidance_max_spl_spinbox.valueChanged.connect(self._on_parameter_changed)
        
        # Arrays
        self.array_type_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.array_frequency_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.array_radius_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.array_steering_spinbox.valueChanged.connect(self._on_parameter_changed)
        
        # Optimization
        self.optim_criterion_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.target_weight_slider.valueChanged.connect(self._on_target_weight_changed)
        
        # Control buttons
        self.run_simulation_button.clicked.connect(self.simulation_requested.emit)
        self.run_optimization_button.clicked.connect(self.optimization_requested.emit)
    
    def _on_parameter_changed(self):
        """Handle parameter change."""
        if not self.updating_ui:
            self.parameter_changed.emit()
    
    def _on_target_weight_changed(self, value):
        """Handle target weight slider change."""
        self.target_weight_label.setText(f"{value}%")
        self._on_parameter_changed()
    
    def get_simulation_parameters(self) -> Dict[str, Any]:
        """Get current simulation parameters."""
        return {
            'frequency': self.frequency_spinbox.value(),
            'speed_of_sound': self.speed_of_sound_spinbox.value(),
            'grid_resolution': self.grid_resolution_spinbox.value(),
            'colormap': self.colormap_combo.currentText(),
            'auto_scale_spl': self.auto_scale_checkbox.isChecked(),
            'min_spl': self.min_spl_spinbox.value(),
            'max_spl': self.max_spl_spinbox.value()
        }
    
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return {
            'criterion': self.optim_criterion_combo.currentText(),
            'frequency': self.optim_frequency_spinbox.value(),
            'freq_min': self.optim_freq_min_spinbox.value(),
            'freq_max': self.optim_freq_max_spinbox.value(),
            'num_frequencies': self.optim_num_freq_spinbox.value(),
            'population_size': self.population_size_spinbox.value(),
            'generations': self.generations_spinbox.value(),
            'mutation_rate': self.mutation_rate_spinbox.value(),
            'target_weight': self.target_weight_slider.value()
        }
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all current parameters."""
        params = {}
        params.update(self.get_simulation_parameters())
        params.update(self.get_optimization_parameters())
        
        # Add other parameters
        params.update({
            'room_type': 'rectangular' if self.rectangular_radio.isChecked() else 'custom',
            'room_width': self.room_width_spinbox.value(),
            'room_height': self.room_height_spinbox.value(),
            'show_grid': self.show_grid_checkbox.isChecked(),
            'snap_to_grid': self.snap_to_grid_checkbox.isChecked(),
            'grid_spacing': self.grid_spacing_spinbox.value(),
            'array_type': self.array_type_combo.currentText(),
            'array_frequency': self.array_frequency_spinbox.value(),
            'array_radius': self.array_radius_spinbox.value(),
            'array_steering': self.array_steering_spinbox.value(),
            'num_sources': self.num_sources_spinbox.value()
        })
        
        return params
    
    def load_project_data(self, project_data: Dict[str, Any]):
        """Load project data into the control panel."""
        self.updating_ui = True
        
        try:
            # Load simulation parameters
            sim_params = project_data.get('simulation_params', {})
            
            if 'frequency' in sim_params:
                self.frequency_spinbox.setValue(sim_params['frequency'])
            if 'speed_of_sound' in sim_params:
                self.speed_of_sound_spinbox.setValue(sim_params['speed_of_sound'])
            if 'grid_resolution' in sim_params:
                self.grid_resolution_spinbox.setValue(sim_params['grid_resolution'])
            
            # Load other parameters as needed
            # ... (additional parameter loading)
            
            self.logger.info("Project data loaded into control panel")
            
        except Exception as e:
            self.logger.error(f"Error loading project data: {e}")
        
        finally:
            self.updating_ui = False
    
    def set_simulation_running(self, running: bool):
        """Set simulation running state."""
        self.run_simulation_button.setEnabled(not running)
        self.stop_simulation_button.setEnabled(running)
        self.run_optimization_button.setEnabled(not running)