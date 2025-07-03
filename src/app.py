"""
DEPRECATED: Legacy app.py - Use main.py instead

This file is maintained for compatibility with existing imports and references.
The actual application logic has been moved to modular components.

For new development, use: python main.py
"""

import sys
import os
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the new modular components
from ui.main_window import MainWindow

# Show deprecation warning
warnings.warn(
    "src/app.py is deprecated. Use 'python main.py' instead for the new modular application.",
    DeprecationWarning,
    stacklevel=2
)


class SubwooferSimApp(MainWindow):
    """
    DEPRECATED: Legacy compatibility class.
    
    This class inherits from the new MainWindow to maintain compatibility
    with any existing code that imports SubwooferSimApp directly.
    
    For new development, use MainWindow directly or launch via main.py
    """
    
    def __init__(self, parent=None):
        """Initialize legacy app using new modular architecture."""
        super().__init__()
        
        # Note: Legacy bridge available separately if needed
        
        print("⚠️  DEPRECATION WARNING: You are using the legacy src/app.py")
        print("   For better performance and maintainability, use: python main.py")
        print("   The new modular architecture provides:")
        print("   - Better code organization")
        print("   - Improved performance")
        print("   - Enhanced testability")
        print("   - Modern PyQt6 patterns")


# Legacy compatibility - maintain the original entry point
if __name__ == "__main__":
    print("="*60)
    print("🚨 LEGACY MODE - CONSIDER MIGRATION")
    print("="*60)
    print("You are running the legacy src/app.py file.")
    print("For the best experience, use the new modular application:")
    print("")
    print("  python main.py")
    print("")
    print("The legacy mode will work but uses deprecated patterns.")
    print("="*60)
    
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    try:
        main_win = SubwooferSimApp()
        main_win.show()
        print("✓ Legacy application started")
        print("  (Consider migrating to: python main.py)")
    except Exception as e:
        print(f"❌ ERROR DURING INITIALIZATION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(app.exec())