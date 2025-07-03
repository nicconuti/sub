"""Main entry point for the Subwoofer Simulation Tool."""

import sys
import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.diagnostics import setup_logging, print_diagnostics
from ui.main_window import MainWindow


def setup_application():
    """Setup the QApplication with proper configuration."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Subwoofer Simulation Tool")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Acoustic Simulation Labs")
    app.setOrganizationDomain("acousticsim.com")
    
    # Set application style
    app.setStyle("Fusion")  # Modern cross-platform style
    
    # Enable high DPI scaling (if available)
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # These attributes are not available in newer PyQt6 versions
        # High DPI scaling is enabled by default
        pass
    
    # Set application icon (if available)
    icon_path = project_root / "resources" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Print diagnostics
        print_diagnostics()
        
        # Create application
        app = setup_application()
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run application
        return app.exec()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        logging.error(f"Failed to start application: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())