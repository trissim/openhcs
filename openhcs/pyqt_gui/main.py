"""
OpenHCS PyQt6 Main Window

Main application window using WindowManager for clean window abstraction.
"""

import logging
from types import FunctionType
from typing import TYPE_CHECKING, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
    QDialog,
    QLabel,
    QProgressBar,
    QSizePolicy,
)
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShowEvent

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.progress.projection import ExecutionRuntimeProjection
from openhcs.agent.ui_bridge_identities import (
    MainWindowWidgetIdentity,
    UiLiveOverviewStateSurfaceIdentityDeclaration,
)
from polystore.filemanager import FileManager
from polystore.base import storage_registry

from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, UIConfig
from openhcs.pyqt_gui.services.function_catalog_projection import (
    ZMQFunctionCatalogProjectionService,
)
from zmqruntime.startup import EndpointStartupStatus
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.pyqt_gui.services.desktop_update import (
    DesktopUpdateCheckFailure,
    DesktopUpdateCheckOrigin,
    DesktopUpdateCheckResult,
    DesktopUpdateDialogPresenter,
    DesktopUpdateError,
    DesktopRuntimeEnvironment,
    DesktopRestartSession,
    DesktopUpdateService,
)
from openhcs.pyqt_gui.services.desktop_restart import DesktopSessionRestart
from openhcs.pyqt_gui.services.zmq_version_restart import (
    ZMQVersionRestartDialogPresenter,
)
from objectstate.object_state import ObjectState
from pyqt_reactive.animation import WindowFlashOverlay
from pyqt_reactive.services.zmq_server_scan_service import (
    EndpointObservationSnapshot,
)
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.widgets.system_monitor import SystemMonitorWidget
from pyqt_reactive.widgets import (
    StatusIndicator,
    StatusState,
)
from pyqt_reactive.widgets.editors.simple_code_editor import QScintillaCodeEditorDialog
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowDockLayoutStore,
    MainWindowDockPane,
    MainWindowEmbeddedWidgets,
    MainWindowLifecycleWorkflow,
    MainWindowPipelineActions,
    MainWindowTimeTravelWorkflow,
    MainWindowShortcutLifecycle,
    MainWindowUiBridgeLifecycle,
    MainWindowWidgetConnector,
    build_main_window_specs,
)
from openhcs.pyqt_gui.services.embedded_code_documents import (
    EmbeddedCodeDocumentRegistrationABC,
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
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiOwnedStateSurfaceDeclaration,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId

if TYPE_CHECKING:
    from openhcs.runtime.zmq_application import OpenHCSEndpointCompatibility

logger = logging.getLogger(__name__)


class MainWindowUiServices(PyQtServiceAdapter):
    """Qt services plus embedded-widget construction owned by the main window."""

    def __init__(
        self,
        main_window: QWidget,
        *,
        widget_gui_config,
        function_catalog_projection: ZMQFunctionCatalogProjectionService,
    ) -> None:
        super().__init__(main_window)
        self.widget_gui_config = widget_gui_config
        self.function_catalog_projection = function_catalog_projection

    def create_window(self, spec) -> QDialog:
        return spec.window_class(self.main_window, self)

    def create_system_monitor_widget(self):
        return SystemMonitorWidget(config=self.widget_gui_config.performance_monitor)

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
            color_scheme=self.get_current_color_scheme(),
            config=self.widget_gui_config.zmq,
            progress_config=self.widget_gui_config.progress,
        )

    def create_pipeline_editor_widget(self):
        from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

        return PipelineEditorWidget(
            self,
            self.get_current_color_scheme(),
        )


class OpenHCSMainWindow(QMainWindow):
    """
    Main OpenHCS PyQt6 application window.

    Implements QDockWidget system to replace textual-window floating windows
    with native Qt docking, providing better desktop integration.
    """

    # Signals for application events
    UI_STATE_SURFACE_DECLARATIONS = (
        UiOwnedStateSurfaceDeclaration(
            identity=UiLiveOverviewStateSurfaceIdentityDeclaration,
            title="UI live overview",
            payload_schema="openhcs.ui.live_overview_state.v1",
        ),
    )
    UI_BRIDGE_WIDGET_IDENTITY = MainWindowWidgetIdentity

    config_changed = pyqtSignal(object)  # GlobalPipelineConfig
    ui_config_changed = pyqtSignal(object)  # UIConfig
    status_message = pyqtSignal(str)  # Status message
    zmq_endpoint_restart_completed = pyqtSignal()
    zmq_endpoint_restart_failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        runtime_context: PyQtGuiRuntimeContext,
        function_catalog_projection: ZMQFunctionCatalogProjectionService,
    ):
        """
        Initialize the main OpenHCS window.
        """

        super().__init__()

        # Core configuration
        self.runtime_context = runtime_context
        self.function_catalog_projection = function_catalog_projection

        # Create shared components
        self.storage_registry = storage_registry
        self.file_manager = FileManager(self.storage_registry)

        # Service adapter for Qt integration
        main_window_services = MainWindowUiServices(
            self,
            widget_gui_config=runtime_context.ui_config,
            function_catalog_projection=function_catalog_projection,
        )
        self.window_services = main_window_services
        self.widget_services = main_window_services
        self.theme_manager_services = main_window_services
        self.window_color_scheme_services = main_window_services
        self.theme_file_services = main_window_services
        self.config_services = main_window_services
        self.desktop_update_service = DesktopUpdateService(self)
        self.desktop_update_presenter = DesktopUpdateDialogPresenter(
            main_window_services,
        )
        self.zmq_version_restart_presenter = ZMQVersionRestartDialogPresenter(
            main_window_services,
        )
        self._pending_zmq_session_restart: DesktopSessionRestart | None = None
        self.desktop_update_service.check_completed.connect(
            self._on_update_check_completed
        )
        self.desktop_update_service.check_failed.connect(self._on_update_check_failed)
        self.zmq_endpoint_restart_completed.connect(
            self._complete_zmq_version_restart
        )
        self.zmq_endpoint_restart_failed.connect(self._fail_zmq_version_restart)

        self.embedded_widgets = MainWindowEmbeddedWidgets()
        self.dock_layout_store = MainWindowDockLayoutStore.for_current_application()
        self.floating_windows: dict[str, QWidget] = {}
        self.ui_bridge_lifecycle = MainWindowUiBridgeLifecycle()
        application = QApplication.instance()
        if application is None:
            raise RuntimeError("OpenHCSMainWindow requires an active QApplication.")
        self.shortcut_lifecycle = MainWindowShortcutLifecycle(application)

        # Declarative window specs
        self.window_specs = self._get_window_specs()

        # Initialize UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()

        # Apply initial theme
        self.apply_initial_theme()

        logger.info(
            "OpenHCS PyQt6 main window initialized (deferred initialization pending)"
        )

    @property
    def pipeline_runtime_config(self) -> GlobalPipelineConfig:
        return self.runtime_context.pipeline_runtime

    def set_pipeline_runtime_config(self, new_config: GlobalPipelineConfig) -> None:
        self.runtime_context = self.runtime_context.with_pipeline_runtime(new_config)

    def set_ui_config(self, new_config: UIConfig) -> None:
        """Apply and publish one UI configuration without partial live state."""

        if type(new_config) is not UIConfig:
            raise TypeError(
                "OpenHCSMainWindow.set_ui_config requires UIConfig; "
                f"got {type(new_config).__name__}."
            )
        previous_config = self.runtime_context.ui_config
        try:
            if new_config.logging != previous_config.logging:
                from openhcs.pyqt_gui.services.logging_config import (
                    configure_gui_logging,
                )

                configure_gui_logging(new_config.logging)
            self._apply_ui_config_consumers(new_config)
        except Exception:
            try:
                if new_config.logging != previous_config.logging:
                    from openhcs.pyqt_gui.services.logging_config import (
                        configure_gui_logging,
                    )

                    configure_gui_logging(previous_config.logging)
                self._apply_ui_config_consumers(previous_config)
            except Exception:
                logger.exception(
                    "Failed to restore live UI consumers after configuration "
                    "application failed"
                )
            raise
        self.runtime_context = self.runtime_context.with_ui_config(new_config)
        self.window_services.widget_gui_config = new_config
        self.ui_config_changed.emit(new_config)

    def _apply_ui_config_consumers(self, config: UIConfig) -> None:
        self._reconcile_ui_bridge(config)
        self.system_monitor.update_config(config.performance_monitor)
        self.plate_manager_widget.set_ui_config(config)
        self.shortcut_lifecycle.apply(config.shortcuts)
        self.zmq_manager_widget.set_zmq_config(
            config.zmq,
            self.zmq_server_manager_ports_to_scan(config),
        )
        self.zmq_manager_widget.set_progress_config(config.progress)

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
        - Default Plate Manager and Pipeline Editor windows - IMMEDIATE

        Note: System monitor is now created during __init__ so startup screen appears immediately
        """
        # Initialize log viewer (hidden) for continuous log monitoring - IMMEDIATE
        self.show_window("log_viewer")

        # Show default windows (plate manager and pipeline editor visible by default) - IMMEDIATE
        self.show_default_windows()
        self.window_services.execute_async_operation(self._prepare_execution_services)
        self._start_ui_bridge_if_enabled()
        self._check_for_updates_on_startup()

        logger.info("Deferred initialization complete (UI ready)")

    async def _prepare_execution_services(self) -> None:
        """Start the shared endpoint, then prewarm its callable catalog."""

        import asyncio

        await self.plate_manager_widget.ensure_execution_server()
        await asyncio.wrap_future(self.function_catalog_projection.prepare())

    def _start_ui_bridge_if_enabled(self) -> None:
        try:
            binding = self._reconcile_ui_bridge(self.runtime_context.ui_config)
            if binding is None:
                logger.debug("OpenHCS UI bridge is disabled")
                return
            logger.info(
                "OpenHCS UI bridge started on %s:%s; descriptor=%s",
                binding.connection.host,
                binding.connection.port,
                binding.descriptor_file_path,
            )
            self.zmq_manager_widget.set_zmq_config(
                self.runtime_context.ui_config.zmq,
                self.zmq_server_manager_ports_to_scan(),
            )
            self.zmq_manager_widget.set_progress_config(
                self.runtime_context.ui_config.progress
            )
        except Exception as exc:
            logger.error("Failed to start OpenHCS UI bridge: %s", exc, exc_info=True)

    def _reconcile_ui_bridge(self, config: UIConfig):
        return self.ui_bridge_lifecycle.reconcile(
            config=config.agent_bridge,
            transport_config=config.zmq,
            create_server=self._create_ui_bridge_server,
        )

    def _create_ui_bridge_server(self, bridge_config, transport_config):
        from openhcs.pyqt_gui.services.ui_bridge_composition import (
            OpenHCSUiBridgeCompositionRoot,
        )
        from openhcs.pyqt_gui.services.ui_bridge_server import UiBridgeControlServer

        return UiBridgeControlServer(
            OpenHCSUiBridgeCompositionRoot.for_main_window(self).build_service(),
            bridge_config,
            transport_config,
        )

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

    def show_window(self, window_id: str, hide_if_startup: bool = True) -> QWidget:
        """Show window using WindowManager."""
        factory = self._create_window_factory(window_id)
        window = WindowManager.show_or_focus(window_id, factory)

        spec = self.window_specs[window_id]
        spec.apply_startup_presentation(window, requested=hide_if_startup)

        self._ensure_flash_overlay(window)
        return window

    def setup_ui(self):
        """Compose the native Qt docking workspace."""
        self.setWindowTitle("OpenHCS")
        self.setMinimumSize(1024, 768)
        self.resize(self.minimumSize())

        # Make main window floating (not tiled) like other OpenHCS components
        self.setWindowFlags(Qt.WindowType.Dialog)

        self.embedded_widgets.configure_host(self)
        self.setCorner(
            Qt.Corner.TopLeftCorner,
            Qt.DockWidgetArea.TopDockWidgetArea,
        )
        self.setCorner(
            Qt.Corner.TopRightCorner,
            Qt.DockWidgetArea.TopDockWidgetArea,
        )

        self.system_monitor = self.widget_services.create_system_monitor_widget()
        system_monitor_pane = MainWindowDockPane.create(
            main_window=self,
            window_id=OpenHCSUiWindowId.system_monitor,
            title="System Monitor",
            widget=self.system_monitor,
            manager_header=self.system_monitor.manager_header,
            docked_content_height=self.system_monitor.embedded_content_height,
        )
        self.system_monitor.embedded_content_height_changed.connect(
            system_monitor_pane.set_docked_content_height
        )
        self.embedded_widgets.register(system_monitor_pane)
        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            system_monitor_pane.dock_widget,
        )

        # Connect system monitor button signals to main window actions
        self.system_monitor.show_global_config.connect(self.show_configuration)
        self.system_monitor.show_log_viewer.connect(self.show_log_viewer)
        self.system_monitor.show_custom_functions.connect(
            self._on_manage_custom_functions
        )
        self.system_monitor.show_test_plate_generator.connect(
            self.show_synthetic_plate_generator
        )

        self.plate_manager_widget = self.widget_services.create_plate_manager_widget()
        plate_manager_pane = MainWindowDockPane.create(
            main_window=self,
            window_id=OpenHCSUiWindowId.plate_manager,
            title="Plate Manager",
            widget=self.plate_manager_widget,
            manager_header=self.plate_manager_widget.manager_header,
        )
        self.embedded_widgets.register(plate_manager_pane)
        self.splitDockWidget(
            system_monitor_pane.dock_widget,
            plate_manager_pane.dock_widget,
            Qt.Orientation.Vertical,
        )

        ports_to_scan = self.zmq_server_manager_ports_to_scan()
        self.zmq_manager_widget = self.widget_services.create_zmq_server_manager_widget(
            ports_to_scan
        )
        self.zmq_manager_widget.log_file_opened.connect(self._open_log_file_in_viewer)
        zmq_manager_pane = MainWindowDockPane.create(
            main_window=self,
            window_id=OpenHCSUiWindowId.zmq_server_manager,
            title="ZMQ Server Manager",
            widget=self.zmq_manager_widget,
            manager_header=self.zmq_manager_widget.manager_header,
            preferred_floating_size=QSize(960, 640),
        )
        self.embedded_widgets.register(zmq_manager_pane)

        self.pipeline_editor_widget = (
            self.widget_services.create_pipeline_editor_widget()
        )
        pipeline_editor_pane = MainWindowDockPane.create(
            main_window=self,
            window_id=OpenHCSUiWindowId.pipeline_editor,
            title="Pipeline Editor",
            widget=self.pipeline_editor_widget,
            manager_header=self.pipeline_editor_widget.manager_header,
        )
        self.embedded_widgets.register(pipeline_editor_pane)
        self.splitDockWidget(
            plate_manager_pane.dock_widget,
            pipeline_editor_pane.dock_widget,
            Qt.Orientation.Horizontal,
        )
        self.splitDockWidget(
            plate_manager_pane.dock_widget,
            zmq_manager_pane.dock_widget,
            Qt.Orientation.Vertical,
        )

        # Connect the two manager workflow surfaces.
        MainWindowWidgetConnector().connect(
            self.plate_manager_widget,
            self.pipeline_editor_widget,
        )
        self._register_embedded_code_document_windows()

        self.resizeDocks(
            [plate_manager_pane.dock_widget, pipeline_editor_pane.dock_widget],
            [1, 1],
            Qt.Orientation.Horizontal,
        )
        system_monitor_height = max(
            1,
            system_monitor_pane.dock_widget.sizeHint().height(),
        )
        lower_workspace_height = max(
            system_monitor_height,
            self.height() - system_monitor_height,
        )
        self.resizeDocks(
            [
                system_monitor_pane.dock_widget,
                plate_manager_pane.dock_widget,
                pipeline_editor_pane.dock_widget,
            ],
            [
                system_monitor_height,
                lower_workspace_height,
                lower_workspace_height,
            ],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [plate_manager_pane.dock_widget, zmq_manager_pane.dock_widget],
            [7, 3],
            Qt.Orientation.Vertical,
        )
        self.dock_layout_store.restore(self)
        self.embedded_widgets.ensure_all_visible()

    def _register_embedded_code_document_windows(self) -> None:
        """Register embedded widgets that expose shared code-mode documents."""
        EmbeddedCodeDocumentRegistrationABC.register_all_for_main_window(self)

    def zmq_server_manager_ports_to_scan(
        self,
        ui_config: UIConfig | None = None,
    ) -> list[int]:
        from openhcs.core.config import get_all_streaming_ports

        config = self.runtime_context.ui_config if ui_config is None else ui_config
        zmq_config = config.zmq
        ports_to_scan = [
            zmq_config.default_port,
            *get_all_streaming_ports(
                num_ports_per_type=zmq_config.ports_per_server_type
            ),
        ]
        bridge_port = self.ui_bridge_lifecycle.bound_port
        if bridge_port is not None and bridge_port not in ports_to_scan:
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

    def show_system_monitor(self):
        """Show the system monitor pane if it was closed."""
        self.embedded_widgets.show_system_monitor()

    def show_image_browser(self):
        """Show image browser window."""
        self.show_window("image_browser")

    def show_log_viewer(self) -> QWidget:
        """Show log viewer window."""
        return self.show_window("log_viewer", hide_if_startup=False)

    def _open_log_file_in_viewer(self, log_file_path: str):
        """
        Open a log file in the log viewer.

        Args:
            log_file_path: Path to log file to open
        """
        window = self.show_log_viewer()
        window.switch_to_log(Path(log_file_path))
        logger.info("Switched log viewer to: %s", log_file_path)

    def setup_menu_bar(self):
        """Setup application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("&Load…", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_orchestrator_configuration)
        file_menu.addAction(load_action)

        save_action = QAction("&Save…", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_orchestrator_configuration)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.quit_app,
            exit_action,
        )

        view_menu = menubar.addMenu("&View")

        system_monitor_action = QAction("&System Monitor", self)
        system_monitor_action.triggered.connect(self.show_system_monitor)
        view_menu.addAction(system_monitor_action)

        # Plate Manager window
        plate_action = QAction("&Plate Manager", self)
        plate_action.triggered.connect(self.show_plate_manager)
        view_menu.addAction(plate_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_plate_manager,
            plate_action,
        )

        # Pipeline Editor window
        pipeline_action = QAction("Pipeline &Editor", self)
        pipeline_action.triggered.connect(self.show_pipeline_editor)
        view_menu.addAction(pipeline_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_pipeline_editor,
            pipeline_action,
        )

        # Image Browser window
        image_browser_action = QAction("&Image Browser", self)
        image_browser_action.triggered.connect(self.show_image_browser)
        view_menu.addAction(image_browser_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_image_browser,
            image_browser_action,
        )

        # Log Viewer window
        log_action = QAction("&Log Viewer", self)
        log_action.triggered.connect(self.show_log_viewer)
        view_menu.addAction(log_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_log_viewer,
            log_action,
        )

        # ZMQ Server Manager window
        zmq_server_action = QAction("&ZMQ Server Manager", self)
        zmq_server_action.triggered.connect(self.show_zmq_server_manager)
        view_menu.addAction(zmq_server_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_zmq_server_manager,
            zmq_server_action,
        )

        # Configuration action
        config_action = QAction("&Global Configuration", self)
        config_action.triggered.connect(self.show_configuration)
        view_menu.addAction(config_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_configuration,
            config_action,
        )

        # Generate Synthetic Plate action
        generate_plate_action = QAction("Generate &Synthetic Plate", self)
        generate_plate_action.triggered.connect(self.show_synthetic_plate_generator)
        view_menu.addAction(generate_plate_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_synthetic_plate_generator,
            generate_plate_action,
        )

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
        help_action = QAction("&Knowledge Base", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        self.shortcut_lifecycle.bind_menu_action(
            lambda config: config.show_help,
            help_action,
        )

        help_menu.addSeparator()

        self.check_for_updates_action = QAction("Check for &Updates…", self)
        self.check_for_updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(self.check_for_updates_action)

        help_menu.addSeparator()

        about_action = QAction("&About OpenHCS", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup application status bar."""
        self.status_bar = self.statusBar()

        # Add time-travel controls to the ordinary left lane. Transient
        # QStatusBar messages hide ordinary widgets, so application status is
        # rendered by a permanent right-lane label below instead.
        from openhcs.pyqt_gui.widgets.shared.time_travel_widget import TimeTravelWidget

        color_scheme = self.window_color_scheme_services.get_current_color_scheme()
        self.bottom_control_panel = QWidget(self)
        bottom_control_layout = QVBoxLayout(self.bottom_control_panel)
        bottom_control_layout.setContentsMargins(0, 0, 0, 0)
        bottom_control_layout.setSpacing(1)

        time_travel_workflow = MainWindowTimeTravelWorkflow(
            refresh_time_travel_widget=lambda: self.time_travel_widget.refresh(),
            before_restore=(
                self.plate_manager_widget.require_pipeline_definition_mutation_allowed
            ),
        )
        self.time_travel_widget = TimeTravelWidget(
            color_scheme=color_scheme,
            time_travel_workflow=time_travel_workflow,
        )
        bottom_control_layout.addWidget(self.time_travel_widget)

        self.status_bar.addWidget(self.bottom_control_panel, 1)

        self._status_message_label = QLabel("", self)
        self._status_message_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._status_message_label.setMaximumWidth(360)
        self._status_message_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self.status_bar.addPermanentWidget(self._status_message_label)

        self._zmq_status_indicator = StatusIndicator(
            check_fn=None,
            color_scheme=color_scheme,
            show_refresh=False,
            parent=self,
        )
        self._zmq_status_indicator.set_state(
            StatusState.DISCONNECTED,
            "ZMQ: Not connected",
        )
        self._zmq_status_indicator.setToolTip(
            "The configured execution endpoint has not been observed"
        )
        self.status_bar.addPermanentWidget(self._zmq_status_indicator)
        self._status_progress_bar = QProgressBar()
        self._status_progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self._status_progress_bar)

        self.status_message.connect(self._status_message_label.setText)

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
        self.plate_manager_widget.progress_started.connect(
            self._on_plate_progress_started
        )
        self.plate_manager_widget.progress_updated.connect(
            self._on_plate_progress_updated
        )
        self.plate_manager_widget.progress_finished.connect(
            self._on_plate_progress_finished
        )
        self.plate_manager_widget.runtime_progress_projection_changed.connect(
            self._on_runtime_progress_projection_changed
        )
        self._connect_zmq_lifecycle()

        # Connect service adapter to application
        self.config_services.set_global_config(self.pipeline_runtime_config)

        # Subscribe to time-travel completion to reopen windows for dirty states
        from objectstate.object_state import ObjectStateRegistry

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

    def _connect_zmq_lifecycle(self) -> None:
        """Project endpoint authority and use lifecycle events as invalidations."""

        self.plate_manager_widget.zmq_connection_status_changed.connect(
            self._observe_zmq_startup_status
        )
        self.plate_manager_widget.zmq_endpoint_compatibility_observed.connect(
            self._observe_zmq_endpoint_compatibility
        )
        self.zmq_manager_widget.endpoint_snapshot_changed.connect(
            self._apply_zmq_endpoint_snapshot
        )
        self.zmq_manager_widget.endpoint_terminated.connect(
            self.plate_manager_widget.zmq_client_service.endpoint_terminated
        )

    def _observe_zmq_startup_status(
        self,
        status: EndpointStartupStatus,
    ) -> None:
        """Commit startup activity into endpoint authority and request a fresh scan."""

        self.zmq_manager_widget.observe_endpoint_startup(
            self.runtime_context.ui_config.zmq.default_port,
            status,
        )
        self.status_message.emit(status.message)
        self.zmq_manager_widget.refresh_servers()

    def _observe_zmq_endpoint_compatibility(
        self,
        compatibility: "OpenHCSEndpointCompatibility",
    ) -> None:
        """Offer one state-preserving replacement for a mismatched endpoint."""

        if compatibility.matches or self._pending_zmq_session_restart is not None:
            return
        if not self.zmq_version_restart_presenter.confirm_restart(compatibility):
            return
        try:
            self._pending_zmq_session_restart = DesktopSessionRestart.capture(self)
        except Exception as exc:
            logger.exception("Failed to capture the ZMQ version restart session")
            self.zmq_version_restart_presenter.show_failure(
                "OpenHCS could not save the current session for restart.\n\n"
                f"{exc}"
            )
            return
        self.status_message.emit("Replacing the mismatched ZMQ execution server…")
        self.window_services.execute_async_operation(
            self._restart_zmq_endpoint_for_version_match
        )

    async def _restart_zmq_endpoint_for_version_match(self) -> None:
        try:
            await self.plate_manager_widget.zmq_client_service.restart_endpoint(
                persistent=True,
            )
        except Exception as exc:
            self.zmq_endpoint_restart_failed.emit(str(exc))
        else:
            self.zmq_endpoint_restart_completed.emit()

    def _complete_zmq_version_restart(self) -> None:
        transaction = self._pending_zmq_session_restart
        self._pending_zmq_session_restart = None
        if transaction is None:
            return
        if transaction.start():
            self.status_message.emit("ZMQ server matched; restarting OpenHCS…")
            self.close()
            return
        transaction.discard()
        self.zmq_version_restart_presenter.show_failure(
            "The matching ZMQ server started, but OpenHCS could not launch its "
            "session restart. The current application remains open."
        )

    def _fail_zmq_version_restart(self, message: str) -> None:
        transaction = self._pending_zmq_session_restart
        self._pending_zmq_session_restart = None
        if transaction is not None:
            transaction.discard()
        self.status_message.emit("ZMQ server restart failed")
        self.zmq_version_restart_presenter.show_failure(
            "OpenHCS could not replace the mismatched ZMQ server. The current "
            f"application and session remain open.\n\n{message}"
        )

    def _apply_zmq_endpoint_snapshot(
        self,
        snapshot: EndpointObservationSnapshot,
    ) -> None:
        """Project the browser's authoritative endpoint snapshot into the status bar."""

        port = self.runtime_context.ui_config.zmq.default_port
        status = snapshot.status_for_port(port)
        self.plate_manager_widget.zmq_client_service.reconcile_endpoint_presence(
            port,
            present=status.phase.expects_endpoint_presence,
        )
        status.present(self._zmq_status_indicator, "ZMQ")
        self._zmq_status_indicator.setToolTip(
            f"Execution endpoint {port}: {status.message.lower()}"
        )

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
        time_travel_workflow = self.time_travel_widget.time_travel_workflow

        # Time travel functions
        def time_travel_back():
            time_travel_workflow.back()

        def time_travel_forward():
            time_travel_workflow.forward()

        def time_travel_to_head():
            time_travel_workflow.to_head()

        self.shortcut_lifecycle.bind_time_travel_command(
            lambda config: config.time_travel_back,
            "Step back in history",
            time_travel_back,
        )
        self.shortcut_lifecycle.bind_time_travel_command(
            lambda config: config.time_travel_forward,
            "Step forward in history",
            time_travel_forward,
        )
        self.shortcut_lifecycle.bind_time_travel_command(
            lambda config: config.time_travel_to_head,
            "Return to history head",
            time_travel_to_head,
        )
        shortcuts = self.runtime_context.ui_config.shortcuts
        self.shortcut_lifecycle.apply(shortcuts)

        logger.info(
            "Global shortcuts (event filter): %s=back, %s=forward, %s=head",
            shortcuts.time_travel_back,
            shortcuts.time_travel_forward,
            shortcuts.time_travel_to_head,
        )

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

    def load_orchestrator_configuration(self) -> None:
        """Open Plate Manager code mode for loading an orchestrator document."""
        self.plate_manager_widget.action_code_plate()

    def save_orchestrator_configuration(self) -> None:
        """Open Plate Manager code mode with every orchestrator for export."""
        from openhcs.core.selection import SelectedAllSelectionMode

        self.plate_manager_widget.action_code_plate(
            selection_mode=SelectedAllSelectionMode.ALL,
        )

    def show_configuration(self):
        """Open the registered application configuration window."""
        from pyqt_reactive.services.scope_window_factory import WindowFactory

        WindowFactory.create_window_for_scope("")

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
        logger.debug(
            f"⏱️ TIME_TRAVEL_CALLBACK: triggering_scope={triggering_scope!r} dirty_count={len(dirty_states)}"
        )

        from objectstate.time_travel_profile import TimeTravelProfiler

        with TimeTravelProfiler.phase(
            "openhcs.main.build_time_travel_window_requests",
            dirty_states=len(dirty_states),
        ):
            pending = self._build_time_travel_window_requests(
                dirty_states,
                triggering_scope,
            )
        if not pending:
            return

        self._defer_time_travel_navigation(
            lambda: self._execute_time_travel_window_requests(
                pending,
                triggering_scope,
            )
        )

    def _defer_time_travel_navigation(self, callback) -> None:
        """Run time-travel navigation after queued restore refreshes settle."""
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, lambda: QTimer.singleShot(0, callback))

    def _execute_time_travel_window_requests(
        self,
        pending: dict[str, TimeTravelWindowRequest],
        triggering_scope: str | None,
    ) -> None:
        """Open/focus the windows targeted by a completed time-travel restore."""
        from pyqt_reactive.services.scope_window_navigation import (
            ScopeWindowNavigationService,
        )
        from pyqt_reactive.services.window_navigation import WindowNavigationRequest
        from objectstate.time_travel_profile import TimeTravelProfiler

        with TimeTravelProfiler.phase(
            "openhcs.main.execute_time_travel_window_requests",
            requests=len(pending),
        ):
            for request in pending.values():
                field_path = (
                    request.target.to_field_path()
                    if request.target is not None
                    else None
                )
                with TimeTravelProfiler.phase(
                    "openhcs.main.navigate_window",
                    scope=request.scope_id,
                    field_path=field_path,
                ):
                    result = ScopeWindowNavigationService.navigate(
                        WindowNavigationRequest(
                            scope_id=request.scope_id,
                            object_state=request.object_state,
                            field_path=field_path,
                            create_if_missing=triggering_scope is not None,
                        )
                    )
                if result.created:
                    logger.info(
                        f"⏱️ TIME_TRAVEL: Opened window for changed state: {request.scope_id}"
                    )
                if result.window is not None:
                    with TimeTravelProfiler.phase(
                        "openhcs.main.select_time_travel_tab",
                        scope=request.scope_id,
                    ):
                        self._select_tab_for_time_travel(
                            request.scope_id, request.target
                        )

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
        if is_function_scope or (target is not None and target.is_function_target):
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
            from objectstate.global_config import get_current_global_config
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

        # Select the directory projected by ExperimentalAnalysisConfig below.
        analysis_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Experimental Analysis Directory",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )

        if not analysis_dir:
            return

        analysis_path = Path(analysis_dir)
        from openhcs.core.config import ExperimentalAnalysisConfig
        from openhcs.processing.backends.experimental_analysis import (
            ExperimentalAnalysisEngine,
        )

        analysis_config = ExperimentalAnalysisConfig()
        config_file = analysis_path / analysis_config.config_file_name
        results_file = analysis_path / analysis_config.results_file_name

        # Check if required files exist
        if not config_file.exists():
            QMessageBox.warning(
                self,
                "Config File Missing",
                f"Expected {analysis_config.config_file_name} not found in:\n"
                f"{analysis_dir}",
            )
            return

        if not results_file.exists():
            QMessageBox.warning(
                self,
                "Results File Missing",
                f"Expected {analysis_config.results_file_name} not found in:\n"
                f"{analysis_dir}",
            )
            return

        try:
            compiled_results = (
                analysis_path / analysis_config.compiled_results_file_name
            )
            raw_results = analysis_path / analysis_config.raw_results_file_name
            heatmaps = analysis_path / analysis_config.heatmap_file_name

            ExperimentalAnalysisEngine(analysis_config).run_directory(analysis_path)

            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Experimental analysis complete!\n\n"
                f"Compiled results: {compiled_results.name}\n"
                + (
                    f"Raw results: {raw_results.name}\n"
                    if analysis_config.export_raw_results
                    else ""
                )
                + (
                    f"Heatmaps: {heatmaps.name}"
                    if analysis_config.export_heatmaps
                    else "Heatmap export disabled"
                ),
            )

        except Exception as e:
            logger.error(f"Failed to run experimental analysis: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"Failed to run experimental analysis:\n\n{str(e)}",
            )

    def show_help(self):
        """Open the source-backed OpenHCS knowledge browser."""
        from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId

        self.show_window(OpenHCSUiWindowId.knowledge_base, hide_if_startup=False)

    def show_about(self) -> None:
        """Open the package-owned OpenHCS identity and version window."""

        self.show_window(OpenHCSUiWindowId.about, hide_if_startup=False)

    def check_for_updates(self) -> None:
        """Start an explicit, asynchronous stable-release check."""
        if not self.desktop_update_service.check_for_updates(
            DesktopUpdateCheckOrigin.EXPLICIT
        ):
            return
        self.check_for_updates_action.setEnabled(False)
        self.status_message.emit("Checking for OpenHCS updates…")

    def _check_for_updates_on_startup(self) -> None:
        """Start one quiet update check after the desktop UI is ready."""

        if not self.runtime_context.ui_config.check_for_updates_on_startup:
            return
        if self.desktop_update_service.check_for_updates(
            DesktopUpdateCheckOrigin.STARTUP
        ):
            self.check_for_updates_action.setEnabled(False)

    def _on_update_check_completed(
        self,
        result: DesktopUpdateCheckResult,
    ) -> None:
        self.check_for_updates_action.setEnabled(True)
        update = result.update
        if not update.update_available:
            if result.origin is DesktopUpdateCheckOrigin.EXPLICIT:
                self.status_message.emit("OpenHCS is up to date")
                self.desktop_update_presenter.show_up_to_date(update)
            return

        self.status_message.emit(f"OpenHCS {update.latest_version} is available")
        if not self.desktop_update_presenter.confirm_update(update):
            return
        session = None
        try:
            runtime = DesktopRuntimeEnvironment.current()
            session = DesktopRestartSession.capture(self)
            started = self.desktop_update_service.start_update(
                update,
                runtime=runtime,
                session=session,
            )
        except DesktopUpdateError as exc:
            if session is not None:
                session.discard()
            logger.warning("Automatic OpenHCS update is unavailable: %s", exc)
            self.desktop_update_presenter.show_warning(
                f"OpenHCS could not start the automatic update.\n\n{exc}",
            )
            return
        except Exception as exc:
            if session is not None:
                session.discard()
            logger.exception("Failed to prepare the OpenHCS update")
            self.desktop_update_presenter.show_warning(
                f"OpenHCS could not save and start the update.\n\n{exc}",
            )
            return
        if not started:
            session.discard()
            self.desktop_update_presenter.show_warning(
                "OpenHCS could not start the background updater. The current "
                "application and session are unchanged.",
            )
            return

        self.status_message.emit("OpenHCS update prepared; restarting…")
        self.close()

    def _on_update_check_failed(self, failure: DesktopUpdateCheckFailure) -> None:
        self.check_for_updates_action.setEnabled(True)
        if failure.origin is DesktopUpdateCheckOrigin.STARTUP:
            logger.warning(
                "Startup OpenHCS update check failed: %s",
                failure.message,
            )
            return
        self.status_message.emit("OpenHCS update check failed")
        self.desktop_update_presenter.show_warning(
            "OpenHCS could not check the official release service.\n\n"
            f"{failure.message}",
        )

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """Handle global configuration changes."""
        self.set_pipeline_runtime_config(new_config)
        self.config_services.set_global_config(new_config)
        self.lifecycle_workflow.propagate_config(new_config)

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Starting application shutdown...")

        try:
            self.dock_layout_store.save(self)
            self.shortcut_lifecycle.close()
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

    def _on_runtime_progress_projection_changed(
        self,
        projection: ExecutionRuntimeProjection,
    ) -> None:
        """Render the current progress-registry projection in the status bar."""

        self.lifecycle_workflow.runtime_progress_changed(projection)

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
            declaration_type=FunctionType,
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
