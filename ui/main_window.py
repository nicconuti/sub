"""Main window for the subwoofer simulation application."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QFileDialog, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from typing import Dict, List, Any, Optional
import logging
import numpy as np

from plot.visualizer import MatplotlibCanvas, SubwooferVisualizer
from core.config import SimulationConfig, DEFAULT_WINDOW_GEOMETRY
from core.data_loader import DataLoader
from core.exporter import ProjectExporter
from core.diagnostics import run_diagnostics
from ui.control_panel import ControlPanel
from ui.dialogs import FileDialogs, PreferencesDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    # Signals
    project_changed = pyqtSignal()
    simulation_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration and components
        self.config = SimulationConfig()
        self.data_loader = DataLoader()
        self.exporter = ProjectExporter()
        self.file_dialogs = FileDialogs(self)
        
        # Application state
        self.current_project_path = None
        self.project_modified = False
        self.simulation_running = False
        
        # Project data
        self.project_data = self._init_project_data()
        
        # Initialize UI
        self._init_ui()
        self._init_menu_bar()
        self._init_status_bar()
        self._connect_signals()
        
        # Run diagnostics
        self._run_initial_diagnostics()
        
        self.logger.info("Main window initialized")
    
    def _init_project_data(self) -> Dict[str, Any]:
        """Initialize empty project data structure."""
        return {
            'name': 'Untitled Project',
            'description': '',
            'version': '1.0',
            'author': '',
            'sources': np.array([], dtype=self.config.sub_dtype),
            'room_vertices': self.config.default_values['room_vertices'],
            'target_areas': [],
            'avoidance_areas': [],
            'sub_placement_areas': [],
            'array_groups': {},
            'simulation_params': self.config.default_values.copy(),
            'optimization_results': {},
            'background_image': None
        }
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Subwoofer Simulation Tool")
        self.setGeometry(*DEFAULT_WINDOW_GEOMETRY)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # Create control panel
        self.control_panel = ControlPanel(self)
        main_splitter.addWidget(self.control_panel)
        
        # Create visualization area
        self.canvas = MatplotlibCanvas(self, width=12, height=9)
        self.visualizer = SubwooferVisualizer(self.canvas)
        main_splitter.addWidget(self.canvas)
        
        # Set splitter proportions
        main_splitter.setSizes([300, 900])
        
        # Set minimum sizes
        self.control_panel.setMinimumWidth(250)
        self.canvas.setMinimumWidth(600)
    
    def _init_menu_bar(self):
        """Initialize the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # New project
        new_action = QAction('&New Project', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create a new project')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # Open project
        open_action = QAction('&Open Project...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open an existing project')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Save project
        save_action = QAction('&Save Project', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save the current project')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Save project as
        save_as_action = QAction('Save Project &As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.setStatusTip('Save the project with a new name')
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Import/Export submenu
        import_export_menu = file_menu.addMenu('&Import/Export')
        
        # Export to Excel
        export_excel_action = QAction('Export to &Excel...', self)
        export_excel_action.setStatusTip('Export project data to Excel file')
        export_excel_action.triggered.connect(self.export_to_excel)
        import_export_menu.addAction(export_excel_action)
        
        # Export to JSON
        export_json_action = QAction('Export to &JSON...', self)
        export_json_action.setStatusTip('Export project data to JSON file')
        export_json_action.triggered.connect(self.export_to_json)
        import_export_menu.addAction(export_json_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        
        # Preferences
        preferences_action = QAction('&Preferences...', self)
        preferences_action.setStatusTip('Open preferences dialog')
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Auto-fit view
        auto_fit_action = QAction('&Auto-fit View', self)
        auto_fit_action.setShortcut('Ctrl+F')
        auto_fit_action.setStatusTip('Auto-fit view to show all elements')
        auto_fit_action.triggered.connect(self.auto_fit_view)
        view_menu.addAction(auto_fit_action)
        
        # Refresh display
        refresh_action = QAction('&Refresh Display', self)
        refresh_action.setShortcut('F5')
        refresh_action.setStatusTip('Refresh the display')
        refresh_action.triggered.connect(self.refresh_display)
        view_menu.addAction(refresh_action)
        
        # Simulation menu
        simulation_menu = menubar.addMenu('&Simulation')
        
        # Run simulation
        run_sim_action = QAction('&Run Simulation', self)
        run_sim_action.setShortcut('Ctrl+R')
        run_sim_action.setStatusTip('Run acoustic simulation')
        run_sim_action.triggered.connect(self.run_simulation)
        simulation_menu.addAction(run_sim_action)
        
        # Stop simulation
        stop_sim_action = QAction('&Stop Simulation', self)
        stop_sim_action.setShortcut('Ctrl+T')
        stop_sim_action.setStatusTip('Stop running simulation')
        stop_sim_action.triggered.connect(self.stop_simulation)
        simulation_menu.addAction(stop_sim_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About
        about_action = QAction('&About', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Diagnostics
        diagnostics_action = QAction('&Diagnostics', self)
        diagnostics_action.setStatusTip('Show system diagnostics')
        diagnostics_action.triggered.connect(self.show_diagnostics)
        help_menu.addAction(diagnostics_action)
    
    def _init_status_bar(self):
        """Initialize the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.project_changed.connect(self.on_project_changed)
        self.simulation_updated.connect(self.on_simulation_updated)
        
        # Connect control panel signals
        self.control_panel.parameter_changed.connect(self.on_parameter_changed)
        self.control_panel.simulation_requested.connect(self.run_simulation)
        self.control_panel.optimization_requested.connect(self.run_optimization)
    
    def _run_initial_diagnostics(self):
        """Run initial system diagnostics."""
        try:
            diagnostics = run_diagnostics()
            if diagnostics['errors']:
                self.logger.warning(f"Diagnostics found {len(diagnostics['errors'])} errors")
                self.status_bar.showMessage(f"System diagnostics: {len(diagnostics['errors'])} warnings")
            else:
                self.logger.info("System diagnostics: All OK")
                self.status_bar.showMessage("System diagnostics: All OK")
        except Exception as e:
            self.logger.error(f"Error running diagnostics: {e}")
    
    # Project management methods
    def new_project(self):
        """Create a new project."""
        try:
            if self.project_modified:
                reply = QMessageBox.question(
                    self, 'Unsaved Changes',
                    'The current project has unsaved changes. Do you want to save them?',
                    QMessageBox.StandardButton.Save | 
                    QMessageBox.StandardButton.Discard | 
                    QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Save:
                    if not self.save_project():
                        return
                elif reply == QMessageBox.StandardButton.Cancel:
                    return
            
            # Reset to new project
            self.project_data = self._init_project_data()
            self.current_project_path = None
            self.project_modified = False
            
            # Update UI
            self.control_panel.load_project_data(self.project_data)
            self.visualizer.clear_all()
            self.update_display()
            
            self.project_changed.emit()
            self.status_bar.showMessage("New project created")
            self.logger.info("New project created")
            
        except Exception as e:
            self.logger.error(f"Error creating new project: {e}")
            QMessageBox.critical(self, 'Error', f'Error creating new project: {e}')
    
    def open_project(self):
        """Open an existing project."""
        try:
            file_path = self.file_dialogs.get_open_filename(
                title="Open Project",
                filters="Excel files (*.xlsx);;JSON files (*.json);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # Load project data
            if file_path.endswith('.xlsx'):
                loaded_data = self.data_loader.load_excel_project(file_path)
            elif file_path.endswith('.json'):
                loaded_data = self.data_loader.load_json_config(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Update project data
            self.project_data.update(loaded_data)
            self.current_project_path = file_path
            self.project_modified = False
            
            # Update UI
            self.control_panel.load_project_data(self.project_data)
            self.update_display()
            
            self.project_changed.emit()
            self.status_bar.showMessage(f"Project opened: {file_path}")
            self.logger.info(f"Project opened: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error opening project: {e}")
            QMessageBox.critical(self, 'Error', f'Error opening project: {e}')
    
    def save_project(self) -> bool:
        """Save the current project."""
        try:
            if self.current_project_path is None:
                return self.save_project_as()
            
            # Save project data
            if self.current_project_path.endswith('.xlsx'):
                self.exporter.export_to_excel(self.current_project_path, self.project_data)
            elif self.current_project_path.endswith('.json'):
                self.data_loader.save_json_config(self.project_data, self.current_project_path)
            else:
                raise ValueError(f"Unsupported file format: {self.current_project_path}")
            
            self.project_modified = False
            self.status_bar.showMessage(f"Project saved: {self.current_project_path}")
            self.logger.info(f"Project saved: {self.current_project_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving project: {e}")
            QMessageBox.critical(self, 'Error', f'Error saving project: {e}')
            return False
    
    def save_project_as(self) -> bool:
        """Save the project with a new name."""
        try:
            file_path = self.file_dialogs.get_save_filename(
                title="Save Project As",
                filters="Excel files (*.xlsx);;JSON files (*.json);;All files (*.*)"
            )
            
            if not file_path:
                return False
            
            # Save project data
            if file_path.endswith('.xlsx'):
                self.exporter.export_to_excel(file_path, self.project_data)
            elif file_path.endswith('.json'):
                self.data_loader.save_json_config(self.project_data, file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.current_project_path = file_path
            self.project_modified = False
            
            self.status_bar.showMessage(f"Project saved as: {file_path}")
            self.logger.info(f"Project saved as: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving project: {e}")
            QMessageBox.critical(self, 'Error', f'Error saving project: {e}')
            return False
    
    def export_to_excel(self):
        """Export project data to Excel."""
        try:
            file_path = self.file_dialogs.get_save_filename(
                title="Export to Excel",
                filters="Excel files (*.xlsx);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            self.exporter.export_to_excel(file_path, self.project_data)
            self.status_bar.showMessage(f"Data exported to Excel: {file_path}")
            self.logger.info(f"Data exported to Excel: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            QMessageBox.critical(self, 'Error', f'Error exporting to Excel: {e}')
    
    def export_to_json(self):
        """Export project data to JSON."""
        try:
            file_path = self.file_dialogs.get_save_filename(
                title="Export to JSON",
                filters="JSON files (*.json);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            self.exporter.export_to_json(file_path, self.project_data)
            self.status_bar.showMessage(f"Data exported to JSON: {file_path}")
            self.logger.info(f"Data exported to JSON: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            QMessageBox.critical(self, 'Error', f'Error exporting to JSON: {e}')
    
    # Simulation methods
    def run_simulation(self):
        """Run acoustic simulation."""
        try:
            if len(self.project_data['sources']) == 0:
                QMessageBox.warning(self, 'Warning', 'No sources defined. Please add sources first.')
                return
            
            self.simulation_running = True
            self.status_bar.showMessage("Running simulation...")
            
            # Get simulation parameters
            sim_params = self.control_panel.get_simulation_parameters()
            
            # Update project data
            self.project_data['simulation_params'].update(sim_params)
            
            # Run simulation (this would typically be done in a separate thread)
            # For now, just update the display
            self.update_display()
            
            self.simulation_running = False
            self.simulation_updated.emit()
            self.status_bar.showMessage("Simulation completed")
            self.logger.info("Simulation completed")
            
        except Exception as e:
            self.simulation_running = False
            self.logger.error(f"Error running simulation: {e}")
            QMessageBox.critical(self, 'Error', f'Error running simulation: {e}')
    
    def stop_simulation(self):
        """Stop running simulation."""
        if self.simulation_running:
            self.simulation_running = False
            self.status_bar.showMessage("Simulation stopped")
            self.logger.info("Simulation stopped")
    
    def run_optimization(self):
        """Run optimization algorithm."""
        try:
            if len(self.project_data['sources']) == 0:
                QMessageBox.warning(self, 'Warning', 'No sources defined. Please add sources first.')
                return
            
            self.status_bar.showMessage("Running optimization...")
            
            # Get optimization parameters
            optim_params = self.control_panel.get_optimization_parameters()
            
            # Run optimization (this would typically be done in a separate thread)
            # For now, just show a message
            QMessageBox.information(self, 'Optimization', 'Optimization feature will be implemented.')
            
            self.status_bar.showMessage("Optimization completed")
            self.logger.info("Optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error running optimization: {e}")
            QMessageBox.critical(self, 'Error', f'Error running optimization: {e}')
    
    # UI update methods
    def update_display(self):
        """Update the visualization display."""
        try:
            # Clear previous plots
            self.visualizer.clear_all()
            
            # Plot room boundaries
            if self.project_data['room_vertices']:
                self.visualizer.plot_room_boundaries(self.project_data['room_vertices'])
            
            # Plot sources
            if len(self.project_data['sources']) > 0:
                self.visualizer.plot_sources(self.project_data['sources'])
            
            # Plot areas
            if self.project_data['target_areas']:
                self.visualizer.plot_target_areas(self.project_data['target_areas'])
            
            if self.project_data['avoidance_areas']:
                self.visualizer.plot_avoidance_areas(self.project_data['avoidance_areas'])
            
            if self.project_data['sub_placement_areas']:
                self.visualizer.plot_placement_areas(self.project_data['sub_placement_areas'])
            
            # Plot array indicators
            if self.project_data['array_groups']:
                self.visualizer.plot_array_indicators(self.project_data['array_groups'])
            
            # Auto-fit view
            self.visualizer.auto_fit_view()
            
            # Refresh display
            self.visualizer.refresh_display()
            
        except Exception as e:
            self.logger.error(f"Error updating display: {e}")
    
    def auto_fit_view(self):
        """Auto-fit the view to show all elements."""
        self.visualizer.auto_fit_view()
        self.visualizer.refresh_display()
        self.status_bar.showMessage("View auto-fitted")
    
    def refresh_display(self):
        """Refresh the display."""
        self.visualizer.refresh_display()
        self.status_bar.showMessage("Display refreshed")
    
    # Dialog methods
    def show_preferences(self):
        """Show preferences dialog."""
        try:
            dialog = PreferencesDialog(self)
            dialog.load_preferences(self.config)
            
            if dialog.exec() == dialog.DialogCode.Accepted:
                # Update configuration
                preferences = dialog.get_preferences()
                # Apply preferences (implementation depends on preferences structure)
                self.logger.info("Preferences updated")
                
        except Exception as e:
            self.logger.error(f"Error showing preferences: {e}")
            QMessageBox.critical(self, 'Error', f'Error showing preferences: {e}')
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Subwoofer Simulation Tool</h2>
        <p>Version 1.0</p>
        <p>A professional tool for subwoofer placement simulation and optimization.</p>
        <p>Features:</p>
        <ul>
            <li>Acoustic simulation with vectorized SPL calculations</li>
            <li>Genetic algorithm optimization</li>
            <li>Interactive visualization</li>
            <li>Multiple array configurations</li>
            <li>Project management with Excel/JSON support</li>
        </ul>
        <p>Built with Python, PyQt6, NumPy, and Matplotlib.</p>
        """
        
        QMessageBox.about(self, 'About', about_text)
    
    def show_diagnostics(self):
        """Show system diagnostics."""
        try:
            diagnostics = run_diagnostics()
            
            diag_text = f"""
            <h3>System Diagnostics</h3>
            <p><b>Python Executable:</b> {diagnostics['python_executable']}</p>
            <h4>Libraries:</h4>
            <ul>
            """
            
            for lib_name, lib_info in diagnostics['libraries'].items():
                version = lib_info.get('version', 'Unknown')
                path = lib_info.get('path', 'Unknown')
                diag_text += f"<li><b>{lib_name}:</b> {version} ({path})</li>"
            
            diag_text += "</ul>"
            
            if diagnostics['errors']:
                diag_text += "<h4>Errors:</h4><ul>"
                for error in diagnostics['errors']:
                    diag_text += f"<li>{error}</li>"
                diag_text += "</ul>"
            
            QMessageBox.information(self, 'System Diagnostics', diag_text)
            
        except Exception as e:
            self.logger.error(f"Error showing diagnostics: {e}")
            QMessageBox.critical(self, 'Error', f'Error showing diagnostics: {e}')
    
    # Event handlers
    def on_project_changed(self):
        """Handle project changed signal."""
        self.project_modified = True
        self.update_window_title()
    
    def on_simulation_updated(self):
        """Handle simulation updated signal."""
        self.update_display()
    
    def on_parameter_changed(self):
        """Handle parameter changed signal."""
        self.project_modified = True
        # Update project data with new parameters
        self.project_data.update(self.control_panel.get_all_parameters())
    
    def update_window_title(self):
        """Update the window title."""
        title = "Subwoofer Simulation Tool"
        
        if self.current_project_path:
            title += f" - {self.current_project_path}"
        else:
            title += f" - {self.project_data['name']}"
        
        if self.project_modified:
            title += " *"
        
        self.setWindowTitle(title)
    
    # Close event
    def closeEvent(self, event):
        """Handle application close event."""
        if self.project_modified:
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'The current project has unsaved changes. Do you want to save them?',
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        
        self.logger.info("Application closing")
        event.accept()