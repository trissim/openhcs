"""
OpenHCS PyQt6 Main Window

Main application window using WindowManager for clean window abstraction.
"""

import logging
from typing import Optional, Callable
from pathlib import Path
import webbrowser

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
    QDialog,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QAction, QKeySequence, QDesktopServices, QShowEvent

from openhcs.core.config import GlobalPipelineConfig
from polystore.filemanager import FileManager
from polystore.base import storage_registry

from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.config_framework.object_state import ObjectState
from pyqt_reactive.animation.flash_overlay_opengl import prewarm_opengl
from pyqt_reactive.animation import WindowFlashOverlay
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.widgets.system_monitor import SystemMonitorWidget
from pyqt_reactive.widgets.editors.simple_code_editor import QScintillaCodeEditorDialog
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowEmbeddedWidgets,
    MainWindowLifecycleWorkflow,
    MainWindowPipelineActions,
    MainWindowTimeTravelWorkflow,
    TimeTravelShortcutEventFilter,
    MainWindowUiBridgeLifecycle,
    MainWindowWidgetConnector,
    build_main_window_specs,
)
from openhcs.pyqt_gui.services.time_travel_navigation import (
    TimeTravelNavigationTarget,
    TimeTravelSourceScope,
    TimeTravelWindowRequest,
    parse_function_scope_ref,
    make_function_token_target,
    make_field_path_target,
    resolve_fallback_field_path,
    should_include_time_travel_scope,
    should_replace_navigation_target,
)

logger = logging.getLogger(__name__)


class MainWindowUiServices(PyQtServiceAdapter):
    """Qt services plus embedded-widget construction owned by the main window."""

    def __init__(self, main_window: QWidget, *, widget_gui_config) -> None:
        super().__init__(main_window)
        self.widget_gui_config = widget_gui_config

    def create_window(self, spec) -> QDialog:
        return spec.window_class(self.main_window, self)

    def create_plate_manager_widget(self):
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        return PlateManagerWidget(
            self,
            self.get_current_color_scheme(),
            gui_config=self.widget_gui_config,
        )

    def create_zmq_server_manager_widget(self, ports_to_scan):
        from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import (
            ZMQServerManagerWidget,
        )

        return ZMQServerManagerWidget(
            ports_to_scan=ports_to_scan,
            title="ZMQ Servers",
            style_generator=self.get_style_generator(),
        )

    def create_pipeline_editor_widget(self):
        from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

        return PipelineEditorWidget(
            self,
            self.get_current_color_scheme(),
            gui_config=self.widget_gui_config,
        )


class OpenHCSMainWindow(QMainWindow):
    """
    Main OpenHCS PyQt6 application window.

    Implements QDockWidget system to replace textual-window floating windows
    with native Qt docking, providing better desktop integration.
    """

    # Signals for application events
    config_changed = pyqtSignal(object)  # GlobalPipelineConfig
    status_message = pyqtSignal(str)  # Status message

    def __init__(
        self,
        *,
        runtime_context: PyQtGuiRuntimeContext,
    ):
        """
        Initialize the main OpenHCS window.
        """

        super().__init__()

        # Core configuration
        self.runtime_context = runtime_context
        self.bridge_config = runtime_context.bridge_config
        self.global_config = runtime_context.pipeline_runtime

        # Create shared components
        self.storage_registry = storage_registry
        self.file_manager = FileManager(self.storage_registry)

        # Service adapter for Qt integration
        main_window_services = MainWindowUiServices(
            self,
            widget_gui_config=runtime_context.widget_config(),
        )
        self.window_services = main_window_services
        self.widget_services = main_window_services
        self.theme_manager_services = main_window_services
        self.window_color_scheme_services = main_window_services
        self.theme_file_services = main_window_services
        self.config_services = main_window_services

        self.embedded_widgets = MainWindowEmbeddedWidgets()
        self.floating_windows: dict[str, QWidget] = {}
        self.ui_bridge_lifecycle = MainWindowUiBridgeLifecycle()

        # Declarative window specs
        self.window_specs = self._get_window_specs()

        # Settings for window state persistence
        self.settings = QSettings("OpenHCS", "PyQt6GUI")

        # Pre-warm OpenGL context in background (zero-delay window creation)
        prewarm_opengl()

        # Initialize UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()

        # Apply initial theme
        self.apply_initial_theme()

        # Restore window state
        self.restore_window_state()

        logger.info(
            "OpenHCS PyQt6 main window initialized (deferred initialization pending)"
        )

    @property
    def pipeline_runtime_config(self) -> GlobalPipelineConfig:
        return self.runtime_context.pipeline_runtime

    def set_pipeline_runtime_config(self, new_config: GlobalPipelineConfig) -> None:
        self.runtime_context = self.runtime_context.with_pipeline_runtime(new_config)
        self.global_config = new_config

    @property
    def service_adapter(self):
        return self.config_services

    @service_adapter.setter
    def service_adapter(self, value):
        self.window_services = value
        self.widget_services = value
        self.theme_manager_services = value
        self.window_color_scheme_services = value
        self.theme_file_services = value
        self.config_services = value

    def deferred_initialization(self):
        """
        Deferred initialization that happens after window is visible.

        This includes:
        - Log viewer initialization (file I/O) - IMMEDIATE
        - Default windows (pipeline editor with config cache warming) - IMMEDIATE

        Note: System monitor is now created during __init__ so startup screen appears immediately
        """
        # Initialize log viewer (hidden) for continuous log monitoring - IMMEDIATE
        self.show_window("log_viewer")

        # Show default windows (plate manager and pipeline editor visible by default) - IMMEDIATE
        self.show_default_windows()
        self._start_ui_bridge_if_enabled()

        logger.info("Deferred initialization complete (UI ready)")

    def _start_ui_bridge_if_enabled(self) -> None:
        bridge_config = self.bridge_config
        if not bridge_config.enabled:
            logger.debug("OpenHCS UI bridge is disabled")
            return

        try:
            from openhcs.pyqt_gui.services.ui_bridge_composition import (
                OpenHCSUiBridgeCompositionRoot,
            )
            from openhcs.pyqt_gui.services.ui_bridge_server import UiBridgeControlServer

            server = UiBridgeControlServer(
                OpenHCSUiBridgeCompositionRoot.for_main_window(self).build_service(),
                bridge_config,
            )
            binding = server.start()
            self.ui_bridge_lifecycle.set_server(server)
            logger.info(
                "OpenHCS UI bridge started on %s:%s; descriptor=%s",
                binding.connection.host,
                binding.connection.port,
                binding.descriptor_file_path,
            )
        except Exception as exc:
            logger.error("Failed to start OpenHCS UI bridge: %s", exc, exc_info=True)

    def _get_window_specs(self):
        """Return declarative window specifications."""
        return build_main_window_specs()

    def _create_window_factory(self, window_id: str) -> Callable[[], QDialog]:
        """Create factory function for a window."""
        spec = self.window_specs[window_id]

        def factory() -> QDialog:
            window = self.window_services.create_window(spec)
            return window

        return factory

    def show_window(self, window_id: str, hide_if_startup: bool = True) -> None:
        """Show window using WindowManager."""
        factory = self._create_window_factory(window_id)
        window = WindowManager.show_or_focus(window_id, factory)

        spec = self.window_specs[window_id]
        if hide_if_startup and spec.initialize_on_startup and window_id == "log_viewer":
            window.hide()

        self._ensure_flash_overlay(window)

    def setup_ui(self):
        """Setup basic UI structure."""
        self.setWindowTitle("OpenHCS")
        self.setMinimumSize(1024, 768)
        self.resize(self.minimumSize())

        # Make main window floating (not tiled) like other OpenHCS components
        self.setWindowFlags(Qt.WindowType.Dialog)

        # Central widget with main layout
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Main vertical splitter: System Monitor (top) vs rest (bottom)
        from PyQt6.QtWidgets import QSplitter

        top_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: System Monitor
        self.system_monitor = SystemMonitorWidget()
        self.embedded_widgets.system_monitor = self.system_monitor
        top_splitter.addWidget(self.system_monitor)

        # Connect system monitor button signals to main window actions
        self.system_monitor.show_global_config.connect(self.show_configuration)
        self.system_monitor.show_log_viewer.connect(self.show_log_viewer)
        self.system_monitor.show_custom_functions.connect(
            self._on_manage_custom_functions
        )
        self.system_monitor.show_test_plate_generator.connect(
            self.show_synthetic_plate_generator
        )

        # Bottom section: Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT SIDE: Vertical splitter with Plate Manager (top) and ZMQ Browser (bottom)
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plate Manager (top of left side)
        # Auto-registers with ServiceRegistry via AutoRegisterServiceMixin
        self.plate_manager_widget = self.widget_services.create_plate_manager_widget()
        self.embedded_widgets.plate_manager = self.plate_manager_widget
        left_splitter.addWidget(self.plate_manager_widget)

        # ZMQ Server Manager (bottom of left side)
        ports_to_scan = self.zmq_server_manager_ports_to_scan()
        self.zmq_manager_widget = self.widget_services.create_zmq_server_manager_widget(
            ports_to_scan
        )
        self.embedded_widgets.zmq_manager = self.zmq_manager_widget
        self.zmq_manager_widget.log_file_opened.connect(self._open_log_file_in_viewer)
        left_splitter.addWidget(self.zmq_manager_widget)

        # Set sizes for left splitter (70% plate manager, 30% ZMQ)
        left_splitter.setSizes([350, 150])

        main_splitter.addWidget(left_splitter)

        # RIGHT SIDE: Pipeline Editor
        self.pipeline_editor_widget = self.widget_services.create_pipeline_editor_widget()
        self.embedded_widgets.pipeline_editor = self.pipeline_editor_widget
        main_splitter.addWidget(self.pipeline_editor_widget)

        # Connect plate manager to pipeline editor (mirrors Textual TUI and ManagedWindow pattern)
        MainWindowWidgetConnector().connect(
            self.plate_manager_widget,
            self.pipeline_editor_widget,
        )
        self._register_embedded_code_document_windows()

        # Set sizes for main splitter (50/50)
        main_splitter.setSizes([500, 500])

        # Add main splitter to top splitter (bottom section)
        top_splitter.addWidget(main_splitter)

        # Set sizes for top splitter - system monitor takes minimum, bottom takes all available space
        top_splitter.setSizes([1, 1000])
        top_splitter.setStretchFactor(0, 0)  # System monitor doesn't stretch
        top_splitter.setStretchFactor(1, 1)  # Bottom section takes all available space

        central_layout.addWidget(top_splitter, 1)  # Stretch to fill remaining space

        # Store references
        self.central_layout = central_layout
        self.top_splitter = top_splitter
        self.main_splitter = main_splitter
        self.left_splitter = left_splitter
        self.embedded_widgets.top_splitter = top_splitter
        self.embedded_widgets.main_splitter = main_splitter
        self.embedded_widgets.left_splitter = left_splitter

        self.setCentralWidget(central_widget)

    def _register_embedded_code_document_windows(self) -> None:
        """Register embedded widgets that expose shared code-mode documents."""
        from pyqt_reactive.services.window_manager import WindowManager
        from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId

        code_document_driver = self.pipeline_editor_widget.code_document_driver()
        WindowManager.register(
            OpenHCSUiWindowId.pipeline_editor,
            self.pipeline_editor_widget,
            code_document_driver=code_document_driver,
        )

    def zmq_server_manager_ports_to_scan(self) -> list[int]:
        from openhcs.core.config import get_all_streaming_ports

        ports_to_scan = list(get_all_streaming_ports(num_ports_per_type=10))
        bridge_port = self.bridge_config.port
        if bridge_port not in ports_to_scan:
            ports_to_scan.append(bridge_port)
        return ports_to_scan

    def apply_initial_theme(self):
        """Apply initial color scheme to the main window."""
        # Get theme manager from service adapter
        theme_manager = self.theme_manager_services.get_theme_manager()

        # Note: ServiceAdapter already applied dark theme globally in its __init__
        # Just register for theme change notifications, don't re-apply
        theme_manager.register_theme_change_callback(self.on_theme_changed)

        logger.debug(
            "Registered for theme change notifications (theme already applied by ServiceAdapter)"
        )

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

    def _ensure_flash_overlay(self, window: QWidget) -> None:
        """Eagerly create flash overlay for a window to avoid first-paint glitches."""
        WindowFlashOverlay.get_for_window(window)

    def show_default_windows(self):
        """Show plate manager by default."""
        # Plate Manager and Pipeline Editor are now embedded in the main window
        # Just ensure they're visible (in case they were hidden)
        self.embedded_widgets.show_defaults()

        # Log viewer is still a separate window (on-demand)

    def show_plate_manager(self):
        """Show plate manager widget if not already visible."""
        self.embedded_widgets.show_plate_manager()

    def show_pipeline_editor(self):
        """Show pipeline editor widget if not already visible."""
        self.embedded_widgets.show_pipeline_editor()

    def show_zmq_server_manager(self):
        """Show ZMQ server manager widget if not already visible."""
        self.embedded_widgets.show_zmq_manager()

    def show_image_browser(self):
        """Show image browser window."""
        self.show_window("image_browser")

    def show_log_viewer(self):
        """Show log viewer window."""
        self.show_window("log_viewer", hide_if_startup=False)

    def _open_log_file_in_viewer(self, log_file_path: str):
        """
        Open a log file in the log viewer.

        Args:
            log_file_path: Path to log file to open
        """
        self.show_log_viewer()

        window = self._get_managed_window("log_viewer")
        if window:
            log_viewer_widget = window.get_widget()
            log_viewer_widget.switch_to_log(Path(log_file_path))
            logger.info(f"Switched log viewer to: {log_file_path}")

    def setup_menu_bar(self):
        """Setup application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Theme submenu
        theme_menu = file_menu.addMenu("&Theme")

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

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu - shortcuts come from declarative ShortcutConfig
        from openhcs.pyqt_gui.config import get_shortcut_config

        shortcuts = get_shortcut_config()

        view_menu = menubar.addMenu("&View")

        # Plate Manager window
        plate_action = QAction("&Plate Manager", self)
        plate_action.setShortcut(shortcuts.show_plate_manager.key)
        plate_action.triggered.connect(self.show_plate_manager)
        view_menu.addAction(plate_action)

        # Pipeline Editor window
        pipeline_action = QAction("Pipeline &Editor", self)
        pipeline_action.setShortcut(shortcuts.show_pipeline_editor.key)
        pipeline_action.triggered.connect(self.show_pipeline_editor)
        view_menu.addAction(pipeline_action)

        # Image Browser window
        image_browser_action = QAction("&Image Browser", self)
        image_browser_action.setShortcut(shortcuts.show_image_browser.key)
        image_browser_action.triggered.connect(self.show_image_browser)
        view_menu.addAction(image_browser_action)

        # Log Viewer window
        log_action = QAction("&Log Viewer", self)
        log_action.setShortcut(shortcuts.show_log_viewer.key)
        log_action.triggered.connect(self.show_log_viewer)
        view_menu.addAction(log_action)

        # ZMQ Server Manager window
        zmq_server_action = QAction("&ZMQ Server Manager", self)
        zmq_server_action.setShortcut(shortcuts.show_zmq_server_manager.key)
        zmq_server_action.triggered.connect(self.show_zmq_server_manager)
        view_menu.addAction(zmq_server_action)

        # Configuration action
        config_action = QAction("&Global Configuration", self)
        config_action.setShortcut(shortcuts.show_configuration.key)
        config_action.triggered.connect(self.show_configuration)
        view_menu.addAction(config_action)

        # Generate Synthetic Plate action
        generate_plate_action = QAction("Generate &Synthetic Plate", self)
        generate_plate_action.setShortcut(shortcuts.show_synthetic_plate_generator.key)
        generate_plate_action.triggered.connect(self.show_synthetic_plate_generator)
        view_menu.addAction(generate_plate_action)

        view_menu.addSeparator()

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Custom Functions submenu
        custom_functions_menu = tools_menu.addMenu("&Custom Functions")

        # Create new custom function action
        create_function_action = QAction("&Create New Function...", self)
        create_function_action.triggered.connect(self._on_create_custom_function)
        custom_functions_menu.addAction(create_function_action)

        # Manage custom functions action
        manage_functions_action = QAction("&Manage Functions...", self)
        manage_functions_action.triggered.connect(self._on_manage_custom_functions)
        custom_functions_menu.addAction(manage_functions_action)

        tools_menu.addSeparator()

        # Analysis Consolidation submenu
        analysis_menu = tools_menu.addMenu("&Analysis Consolidation")

        # Consolidate Results action
        consolidate_action = QAction("&Consolidate Results Directory...", self)
        consolidate_action.triggered.connect(self._on_consolidate_results)
        analysis_menu.addAction(consolidate_action)

        # Merge MetaXpress Summaries action
        merge_summaries_action = QAction("&Merge MetaXpress Summaries...", self)
        merge_summaries_action.triggered.connect(self._on_merge_metaxpress_summaries)
        analysis_menu.addAction(merge_summaries_action)

        # Concatenate MetaXpress Summaries (keep all headers) action
        concat_summaries_action = QAction(
            "&Concatenate MetaXpress Summaries (Keep Headers)...", self
        )
        concat_summaries_action.triggered.connect(self._on_concat_metaxpress_summaries)
        analysis_menu.addAction(concat_summaries_action)

        analysis_menu.addSeparator()

        # Run Experimental Analysis action
        experimental_analysis_action = QAction("Run &Experimental Analysis...", self)
        experimental_analysis_action.triggered.connect(
            self._on_run_experimental_analysis
        )
        analysis_menu.addAction(experimental_analysis_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # General help action
        help_action = QAction("&Documentation", self)
        help_action.setShortcut(shortcuts.show_help.key)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def setup_status_bar(self):
        """Setup application status bar."""
        self.status_bar = self.statusBar()

        # Add time-travel and pipeline debug controls to LEFT side of status bar.
        # Note: Don't use showMessage() as it hides addWidget() widgets
        from openhcs.pyqt_gui.widgets.shared.time_travel_widget import TimeTravelWidget

        color_scheme = self.window_color_scheme_services.get_current_color_scheme()
        self.bottom_control_panel = QWidget(self)
        bottom_control_layout = QVBoxLayout(self.bottom_control_panel)
        bottom_control_layout.setContentsMargins(0, 0, 0, 0)
        bottom_control_layout.setSpacing(1)

        self.time_travel_widget = TimeTravelWidget(color_scheme=color_scheme)
        bottom_control_layout.addWidget(self.time_travel_widget)

        self.status_bar.addWidget(self.bottom_control_panel, 1)

        self._status_progress_bar = QProgressBar()
        self._status_progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self._status_progress_bar)

        # Connect status message signal
        self.status_message.connect(self.status_bar.showMessage)

        self.lifecycle_workflow = MainWindowLifecycleWorkflow(
            main_window=self,
            embedded_widgets=self.embedded_widgets,
            floating_windows=self.floating_windows,
            status_progress_bar=self._status_progress_bar,
            ui_bridge_lifecycle=self.ui_bridge_lifecycle,
        )

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect config changes
        self.config_changed.connect(self.on_config_changed)

        # Connect service adapter to application
        self.config_services.set_global_config(self.pipeline_runtime_config)

        # Setup auto-save timer for window state
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.save_window_state)
        self.auto_save_timer.start(30000)  # Save every 30 seconds

        # Subscribe to time-travel completion to reopen windows for dirty states
        from openhcs.config_framework.object_state import ObjectStateRegistry

        ObjectStateRegistry.add_time_travel_complete_callback(
            self._on_time_travel_complete
        )

        # Subscribe to ObjectState unregistration to auto-close associated windows
        # This ensures windows close when time-traveling removes their backing state
        ObjectStateRegistry.add_unregister_callback(self._on_object_state_unregistered)

        # Register OpenHCS window handlers with the generic factory
        from openhcs.pyqt_gui.services.window_handlers import (
            register_openhcs_window_handlers,
        )

        register_openhcs_window_handlers()

        # Setup global keyboard shortcuts from declarative config
        self._setup_global_shortcuts()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._log_window_size("shown")

    def _log_window_size(self, context: str) -> None:
        size = self.size()
        logger.info(
            "Main window %s size=%dx%d pos=%d,%d",
            context,
            size.width(),
            size.height(),
            self.x(),
            self.y(),
        )

    def _setup_global_shortcuts(self):
        """Setup global keyboard shortcuts from declarative ShortcutConfig.

        Uses event filter to intercept Ctrl+Z/Y BEFORE input widgets get them,
        so time-travel always takes priority over widget-level undo/redo.
        """
        from PyQt6.QtWidgets import QApplication
        from openhcs.pyqt_gui.config import get_shortcut_config

        shortcuts = get_shortcut_config()
        time_travel_workflow = MainWindowTimeTravelWorkflow(
            refresh_time_travel_widget=self.time_travel_widget.refresh
        )

        # Time travel functions
        def time_travel_back():
            time_travel_workflow.back()

        def time_travel_forward():
            time_travel_workflow.forward()

        def time_travel_to_head():
            time_travel_workflow.to_head()

        # Store actions for event filter
        self._time_travel_actions = {
            shortcuts.time_travel_back.key: time_travel_back,
            shortcuts.time_travel_forward.key: time_travel_forward,
            shortcuts.time_travel_to_head.key: time_travel_to_head,
        }

        self._event_filter = TimeTravelShortcutEventFilter(self._time_travel_actions)
        QApplication.instance().installEventFilter(self._event_filter)

        logger.info(
            f"Global shortcuts (event filter): {shortcuts.time_travel_back.key}=back, "
            f"{shortcuts.time_travel_forward.key}=forward, "
            f"{shortcuts.time_travel_to_head.key}=head"
        )

    def restore_window_state(self):
        """Restore window state from settings."""
        logger.debug("Skipping window state restore; persistence disabled")
        return
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
        MainWindowPipelineActions(self, self.pipeline_editor_widget).new_pipeline()

    def open_pipeline(self):
        """Open existing pipeline."""
        MainWindowPipelineActions(self, self.pipeline_editor_widget).open_pipeline()

    def save_pipeline(self):
        """Save current pipeline."""
        MainWindowPipelineActions(self, self.pipeline_editor_widget).save_pipeline()

    def show_configuration(self):
        """Show configuration dialog for global config editing."""
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow

        def handle_config_save(new_config):
            """Handle configuration save (mirrors Textual TUI pattern)."""
            # new_config is already a GlobalPipelineConfig (concrete class)
            self.set_pipeline_runtime_config(new_config)

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import GlobalPipelineConfig
            from openhcs.config_framework.global_config import (
                set_global_config_for_editing,
            )

            set_global_config_for_editing(GlobalPipelineConfig, new_config)

            # Emit signal for other components to update
            self.config_changed.emit(new_config)

            # Save config to cache for future sessions (matches TUI)
            self._save_config_to_cache(new_config)

        # Use concrete GlobalPipelineConfig for global config editing (static context)
        # CRITICAL: Pass scope_id="" to match the ObjectState registered in app.py
        # This ensures ConfigWindow reuses the existing ObjectState instead of creating a new one
        config_window = ConfigWindow(
            GlobalPipelineConfig,  # config_class (concrete class for static context)
            self.config_services.get_global_config(),  # current_config (concrete instance)
            handle_config_save,  # on_save_callback
            self.window_color_scheme_services.get_current_color_scheme(),  # color_scheme
            self,  # parent
            scope_id="",  # Global scope - matches app.py registration
        )
        # Show as non-modal window (like plate manager and pipeline editor)
        config_window.show()
        config_window.raise_()
        config_window.activateWindow()

    def _on_object_state_unregistered(self, scope_id: str, state: "ObjectState"):
        """Handle ObjectState unregistration by closing associated windows.

        When time-travel removes a step/config (unregisters its ObjectState),
        any open editor window for that scope should automatically close.
        This ensures UI stays in sync with the ObjectState registry.

        Args:
            scope_id: Scope of the unregistered ObjectState
            state: The ObjectState being unregistered
        """
        from pyqt_reactive.services.window_manager import WindowManager

        if WindowManager.is_open(scope_id):
            WindowManager.close_window(scope_id)
            logger.info(
                f"⏱️ TIME_TRAVEL: Auto-closed window for unregistered state: {scope_id}"
            )

    def _on_time_travel_complete(self, dirty_states, triggering_scope: str | None):
        """Handle time-travel completion by reopening windows for dirty ObjectStates.

        When time-travel restores state with unsaved changes, this callback
        reopens the appropriate editor windows so the user can see/save the changes.

        Opens windows for the triggering mutation scope when available. Older
        dirty states may still exist in the full registry snapshot, but undoing
        one edit should not reopen unrelated editors.

        Args:
            dirty_states: List of (scope_id, ObjectState) tuples with unsaved changes
            triggering_scope: Scope that triggered the snapshot (for logging only)
        """
        from pyqt_reactive.services.scope_window_navigation import (
            ScopeWindowNavigationService,
        )
        from pyqt_reactive.services.window_navigation import WindowNavigationRequest

        logger.debug(
            f"⏱️ TIME_TRAVEL_CALLBACK: triggering_scope={triggering_scope!r} dirty_count={len(dirty_states)}"
        )

        pending = self._build_time_travel_window_requests(
            dirty_states,
            triggering_scope,
        )

        for request in pending.values():
            field_path = (
                request.target.to_field_path() if request.target is not None else None
            )
            result = ScopeWindowNavigationService.navigate(
                WindowNavigationRequest(
                    scope_id=request.scope_id,
                    object_state=request.object_state,
                    field_path=field_path,
                    create_if_missing=True,
                )
            )
            if result.created:
                logger.info(
                    f"⏱️ TIME_TRAVEL: Reopened window for dirty state: {request.scope_id}"
                )
            if result.window is not None:
                self._select_tab_for_time_travel(request.scope_id, request.target)

    def _build_time_travel_window_requests(
        self,
        dirty_states,
        triggering_scope: str | None,
    ) -> dict[str, TimeTravelWindowRequest]:
        """Project changed ObjectStates into canonical window reopen requests."""
        from objectstate import ObjectStateRegistry

        pending: dict[str, TimeTravelWindowRequest] = {}

        for entry in dirty_states:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                continue
            scope_id, state = entry
            if not isinstance(scope_id, str) or not isinstance(state, ObjectState):
                continue
            if not should_include_time_travel_scope(
                TimeTravelSourceScope(
                    changed_scope_id=scope_id,
                    triggering_scope=triggering_scope,
                )
            ):
                continue

            use_scope_id = scope_id
            use_state = state
            target: TimeTravelNavigationTarget | None = None

            function_scope = parse_function_scope_ref(scope_id)
            if function_scope is not None:
                use_scope_id = function_scope.step_scope_id
                parent_state = ObjectStateRegistry.get_by_scope(use_scope_id)
                if isinstance(parent_state, ObjectState):
                    use_state = parent_state
                target = make_function_token_target(function_scope.function_token)

            if target is None:
                field_path = resolve_fallback_field_path(
                    use_state.last_changed_field,
                    use_state.dirty_fields,
                )
                logger.debug(
                    f"⏱️ TIME_TRAVEL_NAV: scope={use_scope_id} last_changed_field={field_path}"
                )
                if field_path:
                    target = make_field_path_target(field_path)

            request = TimeTravelWindowRequest(
                scope_id=use_scope_id,
                object_state=use_state,
                target=target,
            )
            existing = pending.get(use_scope_id)
            if existing is None:
                pending[use_scope_id] = request
            elif should_replace_navigation_target(existing.target, target):
                pending[use_scope_id] = request

        return pending

    def _select_tab_for_time_travel(
        self, scope_id: str, target: TimeTravelNavigationTarget | None
    ) -> None:
        """Select appropriate tab in step editor after time-travel.

        If 'func' parameter was modified, switch to Function Pattern tab.
        Otherwise, stay on Step Settings tab.
        """
        from pyqt_reactive.services.window_manager import WindowManager
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

        window = WindowManager.get_window(scope_id)
        if not isinstance(window, DualEditorWindow):
            return

        if window.tab_widget is None:
            return

        is_function_scope = parse_function_scope_ref(scope_id) is not None
        if is_function_scope or (
            target is not None and target.is_function_target
        ):
            window.tab_widget.setCurrentIndex(1)
            logger.debug("[TAB_SELECT] Time-travel: Function Pattern tab")
        else:
            window.tab_widget.setCurrentIndex(0)
            logger.debug("[TAB_SELECT] Time-travel: Step Settings tab")

    def show_synthetic_plate_generator(self):
        """Show synthetic plate generator window."""
        from openhcs.pyqt_gui.windows.synthetic_plate_generator_window import (
            SyntheticPlateGeneratorWindow,
        )

        # Create and show the generator window
        generator_window = SyntheticPlateGeneratorWindow(
            color_scheme=self.window_color_scheme_services.get_current_color_scheme(),
            parent=self,
        )

        # Connect the plate_generated signal to add the plate to the manager
        generator_window.plate_generated.connect(self._on_synthetic_plate_generated)

        # Show the window
        generator_window.exec()

    def _on_synthetic_plate_generated(self, output_dir: str, pipeline_path: str):
        """
        Handle synthetic plate generation completion.

        Args:
            output_dir: Path to the generated plate directory
            pipeline_path: Path to the test pipeline to load
        """
        from pathlib import Path

        # Ensure plate manager exists (create if needed)
        self.show_plate_manager()

        # Load the test pipeline FIRST (this will create pipeline editor if needed)
        # Pass the plate path so pipeline editor knows which plate to save the pipeline for
        # This ensures the pipeline is saved to plate_pipelines[plate_path]
        self._load_pipeline_file(pipeline_path, plate_path=output_dir)

        # Get the plate manager widget from ServiceRegistry
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
        from pyqt_reactive.services.service_registry import ServiceRegistry

        plate_manager = ServiceRegistry.get(PlateManagerWidget)

        if not plate_manager:
            raise RuntimeError("Plate manager widget not found in ServiceRegistry")

        # Add the generated plate - this triggers plate_selected signal
        # which automatically updates pipeline editor via existing connections
        # (pipeline editor now exists and is connected, so it will receive the signal)
        plate_manager.add_plate_callback([Path(output_dir)])

        logger.info(f"Added synthetic plate and loaded test pipeline: {output_dir}")

    def _load_pipeline_file(self, pipeline_path: str, plate_path: str = None):
        """
        Load a pipeline file into the pipeline editor.

        Args:
            pipeline_path: Path to the pipeline file to load
            plate_path: Optional plate path to associate the pipeline with
        """
        try:
            # Ensure pipeline editor exists (create if needed)
            self.show_pipeline_editor()

            # Get the pipeline editor widget from ServiceRegistry
            from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
            from pyqt_reactive.services.service_registry import ServiceRegistry

            pipeline_editor = ServiceRegistry.get(PipelineEditorWidget)

            if not pipeline_editor:
                raise RuntimeError(
                    "Pipeline editor widget not found in ServiceRegistry"
                )

            # If plate_path is provided, set it as current_plate BEFORE loading
            # This ensures _apply_executed_code() can save to plate_pipelines[current_plate]
            if plate_path:
                pipeline_editor.current_plate = plate_path
                logger.debug(
                    f"Set current_plate to {plate_path} before loading pipeline"
                )

            # Load the pipeline file
            from pathlib import Path

            pipeline_file = Path(pipeline_path)

            if not pipeline_file.exists():
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

            # For .py files, read code and use existing _handle_edited_code
            if pipeline_file.suffix == ".py":
                with open(pipeline_file, "r") as f:
                    code = f.read()

                # Use existing infrastructure that already handles code execution
                pipeline_editor._handle_edited_code(code)
                logger.info(f"Loaded pipeline from Python file: {pipeline_path}")
            else:
                # For pickled files, use existing infrastructure
                pipeline_editor.load_pipeline_from_file(pipeline_file)
                logger.info(f"Loaded pipeline: {pipeline_path}")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            raise

    def _on_consolidate_results(self):
        """Open file dialog to select results directory and consolidate analysis results."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        # Select results directory
        results_dir = QFileDialog.getExistingDirectory(
            self, "Select Results Directory", "", QFileDialog.Option.ShowDirsOnly
        )

        if not results_dir:
            return

        results_path = Path(results_dir)

        # Check for CSV files
        csv_files = list(results_path.glob("*.csv"))
        csv_files = [
            f
            for f in csv_files
            if not any(
                pattern in f.name.lower()
                for pattern in ["metaxpress", "summary", "consolidated", "global"]
            )
        ]

        if not csv_files:
            QMessageBox.warning(
                self, "No CSV Files", f"No CSV files found in:\n{results_dir}"
            )
            return

        try:
            from openhcs.processing.backends.analysis.consolidate_analysis_results import (
                consolidate_analysis_results,
            )
            from openhcs.config_framework.global_config import get_current_global_config
            from openhcs.core.config import GlobalPipelineConfig

            # Get global config
            global_config = get_current_global_config(GlobalPipelineConfig)

            # Run consolidation
            summary_df = consolidate_analysis_results(
                results_directory=str(results_path),
                consolidation_config=global_config.analysis_consolidation_config,
                plate_metadata_config=global_config.plate_metadata_config,
            )

            output_file = (
                results_path
                / global_config.analysis_consolidation_config.output_filename
            )

            QMessageBox.information(
                self,
                "Consolidation Complete",
                f"Successfully consolidated {len(csv_files)} CSV files from {len(summary_df)} wells.\n\n"
                f"Output: {output_file.name}",
            )

        except Exception as e:
            logger.error(f"Failed to consolidate results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Consolidation Failed",
                f"Failed to consolidate results:\n\n{str(e)}",
            )

    def _on_merge_metaxpress_summaries(self):
        """Open file dialog to select multiple MetaXpress summaries and merge them (concat rows)."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        # Open file dialog to select multiple CSV files
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("MetaXpress CSV (*.csv)")
        file_dialog.setWindowTitle(
            "Select MetaXpress Summary Files to Merge (Concat Rows)"
        )

        if not file_dialog.exec():
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        # Ask for output location
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Merged Summary As",
            "merged_metaxpress_summary.csv",
            "CSV Files (*.csv)",
        )

        if not output_file:
            return

        try:
            from openhcs.processing.backends.analysis.consolidate_analysis_results import (
                consolidate_multi_plate_summaries,
            )

            # Extract plate names from file paths (parent directory name)
            plate_names = [Path(f).parent.name for f in selected_files]

            # Merge the summaries (concat rows from different plates)
            consolidate_multi_plate_summaries(
                summary_paths=selected_files,
                output_path=output_file,
                plate_names=plate_names,
            )

            QMessageBox.information(
                self,
                "Merge Complete",
                f"Successfully merged {len(selected_files)} summaries into:\n{output_file}",
            )

        except Exception as e:
            logger.error(f"Failed to merge summaries: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Merge Failed", f"Failed to merge summaries:\n\n{str(e)}"
            )

    def _on_concat_metaxpress_summaries(self):
        """Open file dialog to select multiple MetaXpress summaries and concatenate them (keep all headers)."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        # Open file dialog to select multiple CSV files
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("MetaXpress CSV (*.csv)")
        file_dialog.setWindowTitle(
            "Select MetaXpress Summary Files to Concatenate (Keep All Headers)"
        )

        if not file_dialog.exec():
            return

        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        # Ask for output location
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Concatenated Summary As",
            "concatenated_metaxpress_summary.csv",
            "CSV Files (*.csv)",
        )

        if not output_file:
            return

        try:
            # Read all files and concatenate with headers
            with open(output_file, "w") as outfile:
                for i, input_file in enumerate(selected_files):
                    with open(input_file, "r") as infile:
                        content = infile.read()
                        outfile.write(content)
                        # Add blank line between files (except after last file)
                        if i < len(selected_files) - 1:
                            outfile.write("\n")

            QMessageBox.information(
                self,
                "Concatenation Complete",
                f"Successfully concatenated {len(selected_files)} summaries (with all headers) into:\n{output_file}",
            )

        except Exception as e:
            logger.error(f"Failed to concatenate summaries: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Concatenation Failed",
                f"Failed to concatenate summaries:\n\n{str(e)}",
            )

    def _on_run_experimental_analysis(self):
        """Open file dialog to select directory and run experimental analysis."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        # Select directory containing config.xlsx and metaxpress_style_summary.csv
        analysis_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Experimental Analysis Directory",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )

        if not analysis_dir:
            return

        analysis_path = Path(analysis_dir)
        config_file = analysis_path / "config.xlsx"
        results_file = analysis_path / "metaxpress_style_summary.csv"

        # Check if required files exist
        if not config_file.exists():
            QMessageBox.warning(
                self,
                "Config File Missing",
                f"Expected config.xlsx not found in:\n{analysis_dir}",
            )
            return

        if not results_file.exists():
            QMessageBox.warning(
                self,
                "Results File Missing",
                f"Expected metaxpress_style_summary.csv not found in:\n{analysis_dir}",
            )
            return

        try:
            from openhcs.formats.experimental_analysis import run_experimental_analysis

            # Define output paths
            compiled_results = analysis_path / "compiled_results_normalized.xlsx"
            heatmaps = analysis_path / "heatmaps.xlsx"

            # Run analysis
            run_experimental_analysis(
                results_path=str(results_file),
                config_file=str(config_file),
                compiled_results_path=str(compiled_results),
                heatmap_path=str(heatmaps),
            )

            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Experimental analysis complete!\n\n"
                f"Compiled results: {compiled_results.name}\n"
                f"Heatmaps: {heatmaps.name}",
            )

        except Exception as e:
            logger.error(f"Failed to run experimental analysis: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"Failed to run experimental analysis:\n\n{str(e)}",
            )

    def show_help(self):
        """Opens documentation URL in default web browser."""
        from openhcs.constants.constants import DOCUMENTATION_URL

        url = DOCUMENTATION_URL
        if not QDesktopServices.openUrl(QUrl.fromUserInput(url)):
            # fallback for wsl users because it wants to be special
            webbrowser.open(url)

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """Handle global configuration changes."""
        self.set_pipeline_runtime_config(new_config)
        self.config_services.set_global_config(new_config)
        self.lifecycle_workflow.propagate_config(new_config)

    def _save_config_to_cache(self, config):
        """Save config to cache asynchronously (matches TUI pattern)."""
        try:
            from openhcs.pyqt_gui.services.config_cache_adapter import (
                get_global_config_cache,
            )

            cache = get_global_config_cache()
            cache.save_config_to_cache_async(config)
            logger.info("Global config save to cache initiated")
        except Exception as e:
            logger.error(f"Error saving global config to cache: {e}")

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Starting application shutdown...")

        try:
            self.lifecycle_workflow.close()

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
        self.theme_manager_services.switch_to_dark_theme()
        self.status_message.emit("Switched to dark theme")

    def switch_to_light_theme(self):
        """Switch to light theme variant."""
        self.theme_manager_services.switch_to_light_theme()
        self.status_message.emit("Switched to light theme")

    def load_theme_from_file(self):
        """Load theme from JSON configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Theme Configuration", "", "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            theme_loaded = self.theme_file_services.load_theme_from_config(file_path)
            if theme_loaded:
                self.status_message.emit(f"Loaded theme from {Path(file_path).name}")
            else:
                QMessageBox.warning(
                    self,
                    "Theme Load Error",
                    f"Failed to load theme from {Path(file_path).name}",
                )

    def save_theme_to_file(self):
        """Save current theme to JSON configuration file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Theme Configuration",
            "pyqt6_color_scheme.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if file_path:
            theme_saved = self.theme_file_services.save_current_theme(file_path)
            if theme_saved:
                self.status_message.emit(f"Saved theme to {Path(file_path).name}")
            else:
                QMessageBox.warning(
                    self,
                    "Theme Save Error",
                    f"Failed to save theme to {Path(file_path).name}",
                )

    def _on_plate_progress_started(self, max_value: int):
        """Handle plate manager progress started signal."""
        self.lifecycle_workflow.progress_started(max_value)

    def _on_plate_progress_updated(self, value: int):
        """Handle plate manager progress updated signal."""
        self.lifecycle_workflow.progress_updated(value)

    def _on_plate_progress_finished(self):
        """Handle plate manager progress finished signal."""
        self.lifecycle_workflow.progress_finished()

    def _on_create_custom_function(self):
        """Handle create custom function action."""
        from openhcs.processing.custom_functions.templates import get_default_template
        from openhcs.processing.custom_functions import CustomFunctionManager
        from openhcs.processing.custom_functions.validation import ValidationError

        # Get default template (numpy backend)
        template = get_default_template()

        # Open code editor (LLM assist always available via button)
        editor = QScintillaCodeEditorDialog(
            parent=self,
            initial_content=template,
            title="Create Custom Function",
            code_type="function",
        )

        if editor.exec():
            # User clicked Save
            code = editor.get_content()
            manager = CustomFunctionManager()

            try:
                functions = manager.register_from_code(code)
                func_names = ", ".join(f.__name__ for f in functions)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Function(s) '{func_names}' registered successfully!",
                )
            except ValidationError as e:
                # Validation failed - show specific error
                QMessageBox.critical(
                    self,
                    "Validation Failed",
                    f"Function code validation failed:\n\n{str(e)}",
                )
            # Let other exceptions propagate (fail-loud)

    def _on_manage_custom_functions(self):
        """Open custom function manager dialog."""
        from openhcs.pyqt_gui.dialogs.custom_function_manager_dialog import (
            CustomFunctionManagerDialog,
        )

        dialog = CustomFunctionManagerDialog(parent=self)
        dialog.exec()
