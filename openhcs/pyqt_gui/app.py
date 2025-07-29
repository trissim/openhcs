"""
OpenHCS PyQt6 Application

Main application class that initializes the PyQt6 application and
manages global configuration and services.
"""

import sys
import logging
import asyncio
from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.io.base import storage_registry
from openhcs.io.filemanager import FileManager

from openhcs.pyqt_gui.main import OpenHCSMainWindow

logger = logging.getLogger(__name__)


class OpenHCSPyQtApp(QApplication):
    """
    OpenHCS PyQt6 Application.
    
    Main application class that manages global state, configuration,
    and the main window lifecycle.
    """
    
    def __init__(self, argv: list, global_config: Optional[GlobalPipelineConfig] = None):
        """
        Initialize the OpenHCS PyQt6 application.
        
        Args:
            argv: Command line arguments
            global_config: Global configuration (uses default if None)
        """
        super().__init__(argv)
        
        # Application metadata
        self.setApplicationName("OpenHCS")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("OpenHCS Development Team")
        self.setOrganizationDomain("openhcs.org")
        
        # Global configuration
        self.global_config = global_config or get_default_global_config()
        
        # Shared components
        self.storage_registry = storage_registry
        self.file_manager = FileManager(self.storage_registry)
        
        # Main window
        self.main_window: Optional[OpenHCSMainWindow] = None
        
        # Setup application
        self.setup_application()
        
        logger.info("OpenHCS PyQt6 application initialized")
    
    def setup_application(self):
        """Setup application-wide configuration."""
        # Setup GPU registry
        setup_global_gpu_registry(global_config=self.global_config)
        logger.info("GPU registry setup completed")

        # Set application icon (if available)
        icon_path = Path(__file__).parent / "resources" / "openhcs_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Setup exception handling
        sys.excepthook = self.handle_exception
    
    def create_main_window(self) -> OpenHCSMainWindow:
        """
        Create and show the main window.
        
        Returns:
            Created main window
        """
        if self.main_window is None:
            self.main_window = OpenHCSMainWindow(self.global_config)
            
            # Connect application-level signals
            self.main_window.config_changed.connect(self.on_config_changed)
            
        return self.main_window
    
    def show_main_window(self):
        """Show the main window."""
        if self.main_window is None:
            self.create_main_window()
        
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()
    
    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.
        
        Args:
            new_config: New global configuration
        """
        self.global_config = new_config
        logger.info("Global configuration updated")
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Handle uncaught exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Handle Ctrl+C gracefully
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Show error dialog
        error_msg = f"An unexpected error occurred:\n\n{exc_type.__name__}: {exc_value}"
        
        if self.main_window:
            QMessageBox.critical(
                self.main_window,
                "Unexpected Error",
                error_msg
            )
        else:
            # No main window - application is in invalid state
            raise RuntimeError("Uncaught exception occurred but no main window available for error dialog")
    
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Application exit code
        """
        # Show main window
        self.show_main_window()

        # Start event loop
        return self.exec()


if __name__ == "__main__":
    # Don't run directly - use launch.py instead
    print("Use 'python -m openhcs.pyqt_gui' or 'python -m openhcs.pyqt_gui.launch' to start the GUI")
    sys.exit(1)
