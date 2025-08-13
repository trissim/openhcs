"""
OpenHCS PyQt6 Main Window

Main application window implementing QDockWidget system to replace
textual-window floating windows with native Qt docking.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QDockWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QMenuBar, QStatusBar, QToolBar, QSplitter,
    QMessageBox, QFileDialog, QDialog
)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry

from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter

logger = logging.getLogger(__name__)


class OpenHCSMainWindow(QMainWindow):
    """
    Main OpenHCS PyQt6 application window.
    
    Implements QDockWidget system to replace textual-window floating windows
    with native Qt docking, providing better desktop integration.
    """
    
    # Signals for application events
    config_changed = pyqtSignal(object)  # GlobalPipelineConfig
    status_message = pyqtSignal(str)  # Status message
    
    def __init__(self, global_config: Optional[GlobalPipelineConfig] = None):
        """
        Initialize the main OpenHCS window.
        
        Args:
            global_config: Global configuration (uses default if None)
        """
        super().__init__()
        
        # Core configuration
        self.global_config = global_config or get_default_global_config()
        
        # Create shared components
        self.storage_registry = storage_registry
        self.file_manager = FileManager(self.storage_registry)
        
        # Service adapter for Qt integration
        self.service_adapter = PyQtServiceAdapter(self)
        
        # Floating windows registry (replaces dock widgets)
        self.floating_windows: Dict[str, QDialog] = {}
        
        # Settings for window state persistence
        self.settings = QSettings("OpenHCS", "PyQt6GUI")
        
        # Initialize UI
        self.setup_ui()
        self.setup_dock_system()
        self.create_floating_windows()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()

        # Apply initial theme
        self.apply_initial_theme()

        # Restore window state
        self.restore_window_state()

        # Show default windows (plate manager and pipeline editor visible by default)
        self.show_default_windows()

        logger.info("OpenHCS PyQt6 main window initialized")
    
    def setup_ui(self):
        """Setup basic UI structure."""
        self.setWindowTitle("OpenHCS - High-Content Screening Platform")
        self.setMinimumSize(640, 480)

        # Make main window floating (not tiled) like other OpenHCS components
        from PyQt6.QtCore import Qt
        self.setWindowFlags(Qt.WindowType.Dialog)
        
        # Central widget with system monitor background
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        
        # System monitor widget (background)
        from openhcs.pyqt_gui.widgets.system_monitor import SystemMonitorWidget
        self.system_monitor = SystemMonitorWidget()
        central_layout.addWidget(self.system_monitor)
        
        self.setCentralWidget(central_widget)

    def apply_initial_theme(self):
        """Apply initial color scheme to the main window."""
        # Get theme manager from service adapter
        theme_manager = self.service_adapter.get_theme_manager()

        # Note: ServiceAdapter already applied dark theme globally in its __init__
        # Just register for theme change notifications, don't re-apply
        theme_manager.register_theme_change_callback(self.on_theme_changed)

        logger.debug("Registered for theme change notifications (theme already applied by ServiceAdapter)")

    def on_theme_changed(self, color_scheme):
        """
        Handle theme change notifications.

        Args:
            color_scheme: New color scheme that was applied
        """
        # Update any main window specific styling if needed
        # Most styling is handled automatically by the theme manager
        logger.debug("Main window received theme change notification")
    
    def setup_dock_system(self):
        """Setup window system mirroring Textual TUI floating windows."""
        # In Textual TUI, widgets are floating windows, not docked
        # We'll create windows on-demand when menu items are clicked
        # Only the system monitor stays as the central background widget
        pass
    
    def create_floating_windows(self):
        """Create floating windows mirroring Textual TUI window system."""
        # Windows are created on-demand when menu items are clicked
        # This mirrors the Textual TUI pattern where windows are mounted dynamically
        self.floating_windows = {}  # Track created windows

    def show_default_windows(self):
        """Show plate manager and pipeline editor by default (like Textual TUI)."""
        # Show plate manager by default
        self.show_plate_manager()

        # Show pipeline editor by default
        self.show_pipeline_editor()

    def show_plate_manager(self):
        """Show plate manager window (mirrors Textual TUI pattern)."""
        if "plate_manager" not in self.floating_windows:
            from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

            # Create floating window
            window = QDialog(self)
            window.setWindowTitle("Plate Manager")
            window.setModal(False)
            window.resize(600, 400)

            # Add widget to window
            layout = QVBoxLayout(window)
            plate_widget = PlateManagerWidget(
                self.file_manager,
                self.service_adapter,
                self.service_adapter.get_current_color_scheme()
            )
            layout.addWidget(plate_widget)

            self.floating_windows["plate_manager"] = window

            # Connect to pipeline editor if it exists (mirrors Textual TUI)
            self._connect_plate_to_pipeline_manager(plate_widget)

        # Show the window
        self.floating_windows["plate_manager"].show()
        self.floating_windows["plate_manager"].raise_()
        self.floating_windows["plate_manager"].activateWindow()

    def show_pipeline_editor(self):
        """Show pipeline editor window (mirrors Textual TUI pattern)."""
        if "pipeline_editor" not in self.floating_windows:
            from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

            # Create floating window
            window = QDialog(self)
            window.setWindowTitle("Pipeline Editor")
            window.setModal(False)
            window.resize(800, 600)

            # Add widget to window
            layout = QVBoxLayout(window)
            pipeline_widget = PipelineEditorWidget(
                self.file_manager,
                self.service_adapter,
                self.service_adapter.get_current_color_scheme()
            )
            layout.addWidget(pipeline_widget)

            self.floating_windows["pipeline_editor"] = window

            # Connect to plate manager for current plate selection (mirrors Textual TUI)
            self._connect_pipeline_to_plate_manager(pipeline_widget)

        # Show the window
        self.floating_windows["pipeline_editor"].show()
        self.floating_windows["pipeline_editor"].raise_()
        self.floating_windows["pipeline_editor"].activateWindow()



    def show_log_viewer(self):
        """Show log viewer window (mirrors Textual TUI pattern)."""
        if "log_viewer" not in self.floating_windows:
            from openhcs.pyqt_gui.widgets.log_viewer import LogViewerWindow
            from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

            # Create floating window
            window = QDialog(self)
            window.setWindowTitle("Log Viewer")
            window.setModal(False)
            window.resize(900, 700)

            # Add widget to window
            layout = QVBoxLayout(window)
            log_viewer_widget = LogViewerWindow(self.file_manager, self.service_adapter)
            layout.addWidget(log_viewer_widget)

            self.floating_windows["log_viewer"] = window

            # Connect to plate manager signals if it exists
            if "plate_manager" in self.floating_windows:
                plate_dialog = self.floating_windows["plate_manager"]
                # Find the PlateManagerWidget inside the dialog
                plate_widget = plate_dialog.findChild(PlateManagerWidget)
                if plate_widget and hasattr(plate_widget, 'clear_subprocess_logs'):
                    plate_widget.clear_subprocess_logs.connect(log_viewer_widget.clear_subprocess_logs)
                    plate_widget.subprocess_log_started.connect(log_viewer_widget.start_monitoring)
                    plate_widget.subprocess_log_stopped.connect(log_viewer_widget.stop_monitoring)

        # Show the window
        self.floating_windows["log_viewer"].show()
        self.floating_windows["log_viewer"].raise_()
        self.floating_windows["log_viewer"].activateWindow()

    def setup_menu_bar(self):
        """Setup application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New pipeline action
        new_action = QAction("&New Pipeline", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_pipeline)
        file_menu.addAction(new_action)
        
        # Open pipeline action
        open_action = QAction("&Open Pipeline", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pipeline)
        file_menu.addAction(open_action)
        
        # Save pipeline action
        save_action = QAction("&Save Pipeline", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_pipeline)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")

        # Plate Manager window
        plate_action = QAction("&Plate Manager", self)
        plate_action.setShortcut("Ctrl+P")
        plate_action.triggered.connect(self.show_plate_manager)
        view_menu.addAction(plate_action)

        # Pipeline Editor window
        pipeline_action = QAction("Pipeline &Editor", self)
        pipeline_action.setShortcut("Ctrl+E")
        pipeline_action.triggered.connect(self.show_pipeline_editor)
        view_menu.addAction(pipeline_action)



        # Log Viewer window
        log_action = QAction("&Log Viewer", self)
        log_action.setShortcut("Ctrl+L")
        log_action.triggered.connect(self.show_log_viewer)
        view_menu.addAction(log_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Configuration action
        config_action = QAction("&Configuration", self)
        config_action.triggered.connect(self.show_configuration)
        tools_menu.addAction(config_action)

        tools_menu.addSeparator()

        # Theme submenu
        theme_menu = tools_menu.addMenu("&Theme")

        # Dark theme action
        dark_theme_action = QAction("&Dark Theme", self)
        dark_theme_action.triggered.connect(self.switch_to_dark_theme)
        theme_menu.addAction(dark_theme_action)

        # Light theme action
        light_theme_action = QAction("&Light Theme", self)
        light_theme_action.triggered.connect(self.switch_to_light_theme)
        theme_menu.addAction(light_theme_action)

        theme_menu.addSeparator()

        # Load theme from file action
        load_theme_action = QAction("&Load Theme from File...", self)
        load_theme_action.triggered.connect(self.load_theme_from_file)
        theme_menu.addAction(load_theme_action)

        # Save theme to file action
        save_theme_action = QAction("&Save Theme to File...", self)
        save_theme_action.triggered.connect(self.save_theme_to_file)
        theme_menu.addAction(save_theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")

        # General help action
        help_action = QAction("&OpenHCS Help", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        help_menu.addSeparator()

        # About action
        about_action = QAction("&About OpenHCS", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup application status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("OpenHCS PyQt6 GUI Ready")
        
        # Connect status message signal
        self.status_message.connect(self.status_bar.showMessage)
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect config changes
        self.config_changed.connect(self.on_config_changed)
        
        # Connect service adapter to application
        self.service_adapter.set_global_config(self.global_config)
        
        # Setup auto-save timer for window state
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.save_window_state)
        self.auto_save_timer.start(30000)  # Save every 30 seconds
    
    def restore_window_state(self):
        """Restore window state from settings."""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            
            window_state = self.settings.value("windowState")
            if window_state:
                self.restoreState(window_state)
                
        except Exception as e:
            logger.warning(f"Failed to restore window state: {e}")
    
    def save_window_state(self):
        """Save window state to settings."""
        # Skip settings save for now to prevent hanging
        # TODO: Investigate QSettings hanging issue
        logger.debug("Skipping window state save to prevent hanging")
    
    # Menu action handlers
    def new_pipeline(self):
        """Create new pipeline."""
        if "pipeline_editor" in self.dock_widgets:
            pipeline_widget = self.dock_widgets["pipeline_editor"].widget()
            if hasattr(pipeline_widget, 'new_pipeline'):
                pipeline_widget.new_pipeline()
    
    def open_pipeline(self):
        """Open existing pipeline."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pipeline",
            "",
            "Function Files (*.func);;All Files (*)"
        )
        
        if file_path and "pipeline_editor" in self.dock_widgets:
            pipeline_widget = self.dock_widgets["pipeline_editor"].widget()
            if hasattr(pipeline_widget, 'load_pipeline'):
                pipeline_widget.load_pipeline(Path(file_path))
    
    def save_pipeline(self):
        """Save current pipeline."""
        if "pipeline_editor" in self.dock_widgets:
            pipeline_widget = self.dock_widgets["pipeline_editor"].widget()
            if hasattr(pipeline_widget, 'save_pipeline'):
                pipeline_widget.save_pipeline()
    
    def show_configuration(self):
        """Show configuration dialog with lazy loading support."""
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow
        from openhcs.core.lazy_config import create_pipeline_config_for_editing, PipelineConfig

        # Create lazy PipelineConfig for editing with proper thread-local context
        current_lazy_config = create_pipeline_config_for_editing(self.global_config)

        def handle_config_save(new_config):
            """Handle configuration save (mirrors Textual TUI pattern)."""
            # Convert lazy PipelineConfig back to GlobalPipelineConfig
            global_config = new_config.to_base_config()

            self.global_config = global_config

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import set_current_pipeline_config
            set_current_pipeline_config(global_config)

            # Emit signal for other components to update
            self.config_changed.emit(global_config)

            # Save config to cache for future sessions (matches TUI)
            self._save_config_to_cache(global_config)

        # Use lazy PipelineConfig instead of GlobalPipelineConfig for placeholder support
        config_window = ConfigWindow(
            PipelineConfig,        # config_class (lazy wrapper)
            current_lazy_config,   # current_config (lazy instance)
            handle_config_save,    # on_save_callback
            self.service_adapter.get_current_color_scheme(),  # color_scheme
            self                   # parent
        )
        # Show as non-modal window (like plate manager and pipeline editor)
        config_window.show()
        config_window.raise_()
        config_window.activateWindow()

    def _connect_pipeline_to_plate_manager(self, pipeline_widget):
        """Connect pipeline editor to plate manager (mirrors Textual TUI pattern)."""
        # Get plate manager if it exists
        if "plate_manager" in self.floating_windows:
            plate_manager_window = self.floating_windows["plate_manager"]

            # Find the actual plate manager widget
            plate_manager_widget = None
            for child in plate_manager_window.findChildren(QWidget):
                if hasattr(child, 'selected_plate_path') and hasattr(child, 'orchestrators'):
                    plate_manager_widget = child
                    break

            if plate_manager_widget:
                # Connect plate selection signal to pipeline editor (mirrors Textual TUI)
                plate_manager_widget.plate_selected.connect(pipeline_widget.set_current_plate)

                # Set pipeline editor reference in plate manager
                if hasattr(plate_manager_widget, 'set_pipeline_editor'):
                    plate_manager_widget.set_pipeline_editor(pipeline_widget)

                # Set current plate if one is already selected
                if plate_manager_widget.selected_plate_path:
                    pipeline_widget.set_current_plate(plate_manager_widget.selected_plate_path)

                logger.debug("Connected pipeline editor to plate manager")
            else:
                logger.warning("Could not find plate manager widget to connect")
        else:
            logger.debug("Plate manager not yet created - connection will be made when both exist")

    def _connect_plate_to_pipeline_manager(self, plate_manager_widget):
        """Connect plate manager to pipeline editor (reverse direction)."""
        # Get pipeline editor if it exists
        if "pipeline_editor" in self.floating_windows:
            pipeline_editor_window = self.floating_windows["pipeline_editor"]

            # Find the actual pipeline editor widget
            pipeline_editor_widget = None
            for child in pipeline_editor_window.findChildren(QWidget):
                if hasattr(child, 'set_current_plate') and hasattr(child, 'pipeline_steps'):
                    pipeline_editor_widget = child
                    break

            if pipeline_editor_widget:
                # Connect plate selection signal to pipeline editor (mirrors Textual TUI)
                plate_manager_widget.plate_selected.connect(pipeline_editor_widget.set_current_plate)

                # Set pipeline editor reference in plate manager
                if hasattr(plate_manager_widget, 'set_pipeline_editor'):
                    plate_manager_widget.set_pipeline_editor(pipeline_editor_widget)

                # Set current plate if one is already selected
                if plate_manager_widget.selected_plate_path:
                    pipeline_editor_widget.set_current_plate(plate_manager_widget.selected_plate_path)

                logger.debug("Connected plate manager to pipeline editor")
            else:
                logger.warning("Could not find pipeline editor widget to connect")
        else:
            logger.debug("Pipeline editor not yet created - connection will be made when both exist")

    def show_help(self):
        """Show general OpenHCS help - reuses Textual TUI help system."""
        from openhcs.pyqt_gui.windows.help_window import HelpWindow

        # Create and show help window (reuses existing help content)
        help_window = HelpWindow(parent=self)
        help_window.show()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About OpenHCS",
            "OpenHCS - High-Content Screening Platform\n\n"
            "A comprehensive platform for microscopy image processing\n"
            "and high-content screening analysis.\n\n"
            "PyQt6 GUI Version 1.0.0"
        )
    
    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """Handle global configuration changes."""
        self.global_config = new_config
        self.service_adapter.set_global_config(new_config)

        # Notify all floating windows of config change
        for window in self.floating_windows.values():
            # Get the widget from the window's layout
            layout = window.layout()
            widget = layout.itemAt(0).widget()
            # Only call on_config_changed if the widget has this method
            if hasattr(widget, 'on_config_changed'):
                widget.on_config_changed(new_config)

    def _save_config_to_cache(self, config):
        """Save config to cache asynchronously (matches TUI pattern)."""
        try:
            from openhcs.pyqt_gui.services.config_cache_adapter import get_global_config_cache
            cache = get_global_config_cache()
            cache.save_config_to_cache_async(config)
            logger.info("Global config save to cache initiated")
        except Exception as e:
            logger.error(f"Error saving global config to cache: {e}")

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Starting application shutdown...")

        try:
            # Stop system monitor first with timeout
            if hasattr(self, 'system_monitor'):
                logger.info("Stopping system monitor...")
                self.system_monitor.stop_monitoring()

            # Close floating windows and cleanup their resources
            for window_name, window in list(self.floating_windows.items()):
                try:
                    layout = window.layout()
                    if layout and layout.count() > 0:
                        widget = layout.itemAt(0).widget()
                        if hasattr(widget, 'cleanup'):
                            widget.cleanup()
                    window.close()
                    window.deleteLater()
                except Exception as e:
                    logger.warning(f"Error cleaning up window {window_name}: {e}")

            # Clear floating windows dict
            self.floating_windows.clear()

            # Save window state
            self.save_window_state()

            # Force Qt to process pending events before shutdown
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

            # Additional cleanup - force garbage collection
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        # Accept close event
        event.accept()
        logger.info("OpenHCS PyQt6 application closed")

        # Force application quit with a short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: QApplication.instance().quit())

    # ========== THEME MANAGEMENT METHODS ==========

    def switch_to_dark_theme(self):
        """Switch to dark theme variant."""
        self.service_adapter.switch_to_dark_theme()
        self.status_message.emit("Switched to dark theme")

    def switch_to_light_theme(self):
        """Switch to light theme variant."""
        self.service_adapter.switch_to_light_theme()
        self.status_message.emit("Switched to light theme")

    def load_theme_from_file(self):
        """Load theme from JSON configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Theme Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            success = self.service_adapter.load_theme_from_config(file_path)
            if success:
                self.status_message.emit(f"Loaded theme from {Path(file_path).name}")
            else:
                QMessageBox.warning(
                    self,
                    "Theme Load Error",
                    f"Failed to load theme from {Path(file_path).name}"
                )

    def save_theme_to_file(self):
        """Save current theme to JSON configuration file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Theme Configuration",
            "pyqt6_color_scheme.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            success = self.service_adapter.save_current_theme(file_path)
            if success:
                self.status_message.emit(f"Saved theme to {Path(file_path).name}")
            else:
                QMessageBox.warning(
                    self,
                    "Theme Save Error",
                    f"Failed to save theme to {Path(file_path).name}"
                )
