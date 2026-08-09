"""Workflow objects owned by the PyQt main window."""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import QByteArray, QEvent, QObject, QSize, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QMainWindow,
    QProgressBar,
    QStyle,
    QToolButton,
    QWidget,
)
from pyqt_reactive.widgets.shared.manager_ui_scaffold import ManagerHeaderParts
from pyqt_reactive.services.window_manager import WindowManager

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.execution_state import ManagerExecutionState
from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.progress.projection import ExecutionRuntimeProjection
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.services.window_config import (
    StartupWindowPresentation,
    WindowSpec,
)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.config import AgentUiBridgeConfig, ShortcutConfig
    from openhcs.pyqt_gui.services.ui_bridge_server import UiBridgeControlServer
    from openhcs.runtime.zmq_config import OpenHCSZMQConfig

logger = logging.getLogger(__name__)


class SignalConnectionSurface(ABC):
    @abstractmethod
    def connect(self, callback) -> None:
        raise NotImplementedError


class SignalEmissionSurface(ABC):
    @abstractmethod
    def emit(self, value) -> None:
        raise NotImplementedError


class ConfigChangeSurface(ABC):
    @abstractmethod
    def on_config_changed(self, new_config: GlobalPipelineConfig) -> None:
        raise NotImplementedError


class PipelineEditorWorkflowSurface(ConfigChangeSurface):
    pipeline_steps: list
    pipeline_changed: SignalEmissionSurface
    plate_manager: "PlateManagerWorkflowSurface"

    @abstractmethod
    def set_current_plate(self, plate_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_item_list(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_button_states(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_orchestrator_state_changed(
        self,
        plate_path: str,
        state: OrchestratorState,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_manager_execution_state_changed(
        self,
        state: ManagerExecutionState,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_cellprofiler_pipeline_imported(self, plate_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_pipeline_data_changed(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def show_debug_snapshot(self, notification) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_pipeline_from_file(self, file_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_pipeline_to_file(self, file_path: Path) -> None:
        raise NotImplementedError


class PlateManagerWorkflowSurface(ConfigChangeSurface):
    plate_selected: SignalConnectionSurface
    orchestrator_config_changed: SignalConnectionSurface
    orchestrator_state_changed: SignalConnectionSurface
    manager_execution_state_changed: SignalConnectionSurface
    pipeline_data_changed: SignalConnectionSurface
    cellprofiler_pipeline_imported: SignalConnectionSurface
    debug_snapshot_available: SignalConnectionSurface
    selected_plate_path: str | None

    @abstractmethod
    def require_pipeline_definition_mutation_allowed(
        self,
        plate_path: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def refresh_prepared_cellprofiler_pipelines(self) -> None:
        raise NotImplementedError


class QtShortcutSequenceAuthority:
    """Convert Qt key events into configured shortcut strings."""

    @classmethod
    def from_event(cls, event) -> str | None:
        if event.type() != QEvent.Type.KeyPress:
            return None
        sequence = QKeySequence(event.keyCombination())
        if sequence.isEmpty():
            return None
        return sequence.toString(QKeySequence.SequenceFormat.PortableText)


class CodeEditorFocusAuthority:
    """Decide whether global shortcuts should yield to a focused code editor."""

    @staticmethod
    def allows_global_time_travel() -> bool:
        from PyQt6.Qsci import QsciScintilla

        widget = QApplication.focusWidget()
        while widget is not None:
            if isinstance(widget, QsciScintilla):
                return False
            widget = widget.parentWidget()
        return True


class TimeTravelShortcutEventFilter(QObject):
    """Qt event filter that routes configured shortcuts to time-travel actions."""

    def __init__(self, actions: Mapping[str, Callable[[], None]]) -> None:
        super().__init__()
        self._actions = dict(actions)

    def replace_actions(self, actions: Mapping[str, Callable[[], None]]) -> None:
        """Replace the complete configured key-to-command projection."""

        self._actions = dict(actions)

    def eventFilter(self, obj, event):
        del obj
        sequence = QtShortcutSequenceAuthority.from_event(event)
        if sequence is None:
            return False
        if sequence not in self._actions:
            return False
        if not CodeEditorFocusAuthority.allows_global_time_travel():
            return False
        self._actions[sequence]()
        return True


class MainWindowShortcutLifecycle:
    """Own live key projection onto main-window actions and event routing."""

    def __init__(self, application: QApplication) -> None:
        self._application = application
        self._menu_actions: list[tuple[Callable[["ShortcutConfig"], str], QAction]] = []
        self._time_travel_commands: list[
            tuple[
                Callable[["ShortcutConfig"], str],
                str,
                Callable[[], None],
            ]
        ] = []
        self._event_filter = TimeTravelShortcutEventFilter({})
        self._application.installEventFilter(self._event_filter)

    def bind_menu_action(
        self,
        key_from_config: Callable[["ShortcutConfig"], str],
        action: QAction,
    ) -> None:
        """Bind one concrete action to a typed configuration projection."""

        self._menu_actions.append((key_from_config, action))

    def bind_time_travel_command(
        self,
        key_from_config: Callable[["ShortcutConfig"], str],
        label: str,
        command: Callable[[], None],
    ) -> None:
        """Bind one event-filter command to a typed config projection."""

        self._time_travel_commands.append((key_from_config, label, command))

    def apply(self, config: "ShortcutConfig") -> None:
        """Validate and atomically project all configured key sequences."""

        projected_menu = tuple(
            (
                action,
                self._validated_sequence(
                    action.text().replace("&", ""),
                    key_from_config(config),
                ),
            )
            for key_from_config, action in self._menu_actions
        )
        projected_time_travel = tuple(
            (
                command,
                self._validated_sequence(label, key_from_config(config)),
            )
            for key_from_config, label, command in self._time_travel_commands
        )
        normalized = tuple(
            sequence.toString(QKeySequence.SequenceFormat.PortableText)
            for _consumer, sequence in (*projected_menu, *projected_time_travel)
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("Application keyboard shortcuts must be unique.")

        for action, sequence in projected_menu:
            action.setShortcut(sequence)
        self._event_filter.replace_actions(
            {
                sequence.toString(QKeySequence.SequenceFormat.PortableText): command
                for command, sequence in projected_time_travel
            }
        )

    def close(self) -> None:
        self._application.removeEventFilter(self._event_filter)
        self._menu_actions.clear()
        self._time_travel_commands.clear()

    @staticmethod
    def _validated_sequence(field_name: str, value: str) -> QKeySequence:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Shortcut {field_name} must be a non-empty key sequence.")
        sequence = QKeySequence(value)
        if sequence.isEmpty():
            raise ValueError(f"Shortcut {field_name} is not a valid Qt key sequence.")
        return sequence


@dataclass(frozen=True, slots=True)
class MainWindowSpecDefinition:
    """Declarative record for one WindowManager-managed window."""

    window_id: str
    title: str
    window_class: type[QDialog]
    initialize_on_startup: bool = False
    startup_presentation: StartupWindowPresentation = (
        StartupWindowPresentation.KEEP_VISIBLE
    )

    def build(self) -> WindowSpec:
        return WindowSpec(
            window_id=self.window_id,
            title=self.title,
            window_class=self.window_class,
            initialize_on_startup=self.initialize_on_startup,
            startup_presentation=self.startup_presentation,
        )


def build_main_window_specs() -> dict[str, WindowSpec]:
    """Build all WindowManager-managed window specifications."""

    from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
    from openhcs.pyqt_gui.windows.about_window import AboutOpenHCSWindow
    from openhcs.pyqt_gui.windows.help_window import HelpWindow
    from openhcs.pyqt_gui.windows.managed_windows import (
        ImageBrowserWindow,
        LogViewerWindowWrapper,
        PipelineEditorWindow,
        PlateManagerWindow,
        ZMQServerManagerWindow,
    )

    definitions = (
        MainWindowSpecDefinition(
            "plate_manager",
            "Plate Manager",
            PlateManagerWindow,
            True,
        ),
        MainWindowSpecDefinition(
            "pipeline_editor",
            "Pipeline Editor",
            PipelineEditorWindow,
        ),
        MainWindowSpecDefinition(
            "image_browser",
            "Image Browser",
            ImageBrowserWindow,
        ),
        MainWindowSpecDefinition(
            "log_viewer",
            "Log Viewer",
            LogViewerWindowWrapper,
            True,
            StartupWindowPresentation.HIDE,
        ),
        MainWindowSpecDefinition(
            "zmq_server_manager",
            "ZMQ Server Manager",
            ZMQServerManagerWindow,
        ),
        MainWindowSpecDefinition(
            OpenHCSUiWindowId.knowledge_base,
            "OpenHCS Knowledge Base",
            HelpWindow,
        ),
        MainWindowSpecDefinition(
            OpenHCSUiWindowId.about,
            "About OpenHCS",
            AboutOpenHCSWindow,
        ),
    )
    return {definition.window_id: definition.build() for definition in definitions}


@dataclass(slots=True, weakref_slot=True)
class MainWindowDockFloatController:
    """Toggle one dock while preserving its complete pre-float workspace slot."""

    main_window: QMainWindow
    dock_widget: QDockWidget
    docked_content_height: int | None = None
    preferred_floating_size: QSize | None = None
    _docked_state: QByteArray | None = None
    _floating_size: QSize | None = field(init=False, repr=False)
    _content_minimum_height: int = field(init=False, repr=False)
    _content_maximum_height: int = field(init=False, repr=False)
    _dock_transition_timer: QTimer = field(init=False, repr=False)
    _top_level_sync_timer: QTimer = field(init=False, repr=False)
    _pending_top_level_state: bool | None = field(default=None, init=False, repr=False)
    _redock_pending: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        content = self.dock_widget.widget()
        if content is None:
            raise RuntimeError("Dock float controller requires a content widget")
        if self.preferred_floating_size is not None and (
            self.preferred_floating_size.width() <= 0
            or self.preferred_floating_size.height() <= 0
        ):
            raise ValueError("Preferred floating size must be positive")
        self._floating_size = (
            QSize(self.preferred_floating_size)
            if self.preferred_floating_size is not None
            else None
        )
        self._content_minimum_height = content.minimumHeight()
        self._content_maximum_height = content.maximumHeight()
        self._dock_transition_timer = QTimer(self.main_window)
        self._dock_transition_timer.setSingleShot(True)
        self._dock_transition_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._dock_transition_timer.setInterval(1)
        self._dock_transition_timer.timeout.connect(self._restore_docked_workspace)
        self._top_level_sync_timer = QTimer(self.main_window)
        self._top_level_sync_timer.setSingleShot(True)
        self._top_level_sync_timer.setInterval(0)
        self._top_level_sync_timer.timeout.connect(self._apply_pending_top_level_state)
        self.sync_top_level(self.dock_widget.isFloating())

    def sync_top_level(self, is_floating: bool) -> None:
        """Apply embedded-only sizing while leaving floating panes resizable."""

        if self.docked_content_height is None:
            return
        content = self.dock_widget.widget()
        if content is None:
            return
        if is_floating:
            content.setMinimumHeight(self._content_minimum_height)
            content.setMaximumHeight(self._content_maximum_height)
            return
        if self._redock_pending:
            return
        content.setFixedHeight(self.docked_content_height)

    def set_docked_content_height(self, height: int) -> None:
        """Update the content owner's embedded height projection."""

        if height <= 0:
            raise ValueError("Docked content height must be positive")
        if height == self.docked_content_height:
            return
        self.docked_content_height = height
        self.sync_top_level(self.dock_widget.isFloating())

    def schedule_top_level_sync(self, is_floating: bool) -> None:
        """Apply dock constraints after Qt completes its native transition."""

        self._pending_top_level_state = is_floating
        self._top_level_sync_timer.start()

    def _apply_pending_top_level_state(self) -> None:
        is_floating = self._pending_top_level_state
        if is_floating is None:
            return
        self._pending_top_level_state = None
        self.sync_top_level(is_floating)

    def toggle(self) -> None:
        if not self.dock_widget.isFloating():
            self._docked_state = self.main_window.saveState()
            self.dock_widget.setFloating(True)
            self.dock_widget.show()
            if self._floating_size is not None:
                self.dock_widget.resize(self._floating_size)
            QTimer.singleShot(0, self._reveal_after_layout)
            return

        self._floating_size = self.dock_widget.size()
        self._redock_pending = True
        self.dock_widget.setFloating(False)
        if self._docked_state is None:
            self._redock_pending = False
            self.sync_top_level(False)
            self._reveal_after_layout()
            return

        # A resized floating QDockWidget completes its native dock transition
        # on the next event-loop turn. Restore the one authoritative workspace
        # snapshot after that transition so the first click recovers the prior
        # slot geometry instead of retaining the floating window dimensions.
        # Qt emits topLevelChanged(False) before its dock layout has consumed
        # a resized floating window. The first timed event-loop turn is the
        # earliest boundary at which restoreState can recover the saved slot.
        self._dock_transition_timer.start()

    def _restore_docked_workspace(self) -> None:
        # Reapply embedded-only constraints before asking Qt to consume the
        # saved layout. A resized floating pane otherwise contributes its
        # top-level width to restoreState on slower event loops, leaving a
        # fixed-height dock partially redocked at the floating geometry.
        self._redock_pending = False
        self.sync_top_level(False)
        if self._docked_state is not None:
            if not self.main_window.restoreState(self._docked_state):
                self.dock_widget.setFloating(False)
        QTimer.singleShot(0, self._reveal_after_layout)

    def _reveal_after_layout(self) -> None:
        self.dock_widget.show()
        self.dock_widget.raise_()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MainWindowDockPane:
    """One logical embedded pane and its native Qt geometry owner."""

    window_id: str
    title: str
    widget: QWidget
    dock_widget: QDockWidget
    float_controller: MainWindowDockFloatController | None = None
    float_button: QToolButton | None = None

    @classmethod
    def create(
        cls,
        *,
        main_window: QMainWindow,
        window_id: str,
        title: str,
        widget: QWidget,
        manager_header: ManagerHeaderParts | None = None,
        docked_content_height: int | None = None,
        preferred_floating_size: QSize | None = None,
    ) -> "MainWindowDockPane":
        dock_widget = QDockWidget(title, main_window)
        dock_widget.setObjectName(window_id)
        dock_widget.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock_widget.toggleViewAction().setVisible(False)
        dock_widget.setWidget(widget)
        float_button = None
        float_controller = None
        if manager_header is not None:
            content_layout = widget.layout()
            if content_layout is None:
                raise RuntimeError(
                    f"Dock pane {window_id!r} has no content layout for its manager header"
                )
            content_layout.removeWidget(manager_header.header)
            manager_header.present_as_dock_title()
            float_button = cls._title_button(
                dock_widget=dock_widget,
                object_name=f"{window_id}_dock_float_button",
                tooltip="Float pane",
            )
            float_controller = MainWindowDockFloatController(
                main_window=main_window,
                dock_widget=dock_widget,
                docked_content_height=docked_content_height,
                preferred_floating_size=preferred_floating_size,
            )

            def sync_float_button(is_floating: bool) -> None:
                standard_icon = (
                    QStyle.StandardPixmap.SP_TitleBarNormalButton
                    if is_floating
                    else QStyle.StandardPixmap.SP_TitleBarMaxButton
                )
                float_button.setIcon(dock_widget.style().standardIcon(standard_icon))
                float_button.setToolTip("Dock pane" if is_floating else "Float pane")

            float_button.clicked.connect(float_controller.toggle)
            dock_widget.topLevelChanged.connect(sync_float_button)
            dock_widget.topLevelChanged.connect(
                float_controller.schedule_top_level_sync
            )
            sync_float_button(dock_widget.isFloating())
            manager_header.title_layout.add_right_widget(float_button)
            dock_widget.setTitleBarWidget(manager_header.header)
        return cls(
            window_id=window_id,
            title=title,
            widget=widget,
            dock_widget=dock_widget,
            float_controller=float_controller,
            float_button=float_button,
        )

    @staticmethod
    def _title_button(
        *,
        dock_widget: QDockWidget,
        object_name: str,
        tooltip: str,
    ) -> QToolButton:
        button = QToolButton(dock_widget)
        button.setObjectName(object_name)
        button.setAutoRaise(True)
        button.setIconSize(QSize(12, 12))
        button.setFixedSize(18, 18)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

    def set_docked_content_height(self, height: int) -> None:
        """Delegate a content-owned embedded height to the dock controller."""

        if self.float_controller is None:
            raise RuntimeError(f"Dock pane {self.window_id!r} has no float controller")
        self.float_controller.set_docked_content_height(height)

    def show(self) -> None:
        """Reveal and focus this pane without changing its dock geometry."""

        self.dock_widget.show()
        self.dock_widget.raise_()
        self.widget.setFocus(Qt.FocusReason.OtherFocusReason)


@dataclass(slots=True)
class MainWindowEmbeddedWidgets:
    """Authoritative runtime graph for the main-window dock panes."""

    _panes: dict[str, MainWindowDockPane] = field(default_factory=dict)

    def configure_host(self, main_window: QMainWindow) -> None:
        """Configure stable native docking without reentrant drag animation."""

        main_window.setDockNestingEnabled(True)
        main_window.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

    def register(self, pane: MainWindowDockPane) -> None:
        if pane.window_id in self._panes:
            raise ValueError(f"Duplicate main-window pane id: {pane.window_id!r}")
        self._panes[pane.window_id] = pane

    def panes(self) -> tuple[MainWindowDockPane, ...]:
        return tuple(self._panes.values())

    def require_pane(self, window_id: str) -> MainWindowDockPane:
        try:
            return self._panes[window_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Main-window pane {window_id!r} has not been initialized"
            ) from exc

    def require_plate_manager(self) -> QWidget:
        return self.require_pane(OpenHCSUiWindowId.plate_manager).widget

    def require_pipeline_editor(self) -> QWidget:
        return self.require_pane(OpenHCSUiWindowId.pipeline_editor).widget

    def require_zmq_manager(self) -> QWidget:
        return self.require_pane(OpenHCSUiWindowId.zmq_server_manager).widget

    def require_system_monitor(self) -> QWidget:
        return self.require_pane(OpenHCSUiWindowId.system_monitor).widget

    def show_defaults(self) -> None:
        self.ensure_all_visible()

    def ensure_all_visible(self) -> None:
        """Keep every permanent workspace pane either docked or floating."""

        for pane in self.panes():
            pane.dock_widget.show()

    def show_plate_manager(self) -> None:
        self.require_pane(OpenHCSUiWindowId.plate_manager).show()

    def show_pipeline_editor(self) -> None:
        self.require_pane(OpenHCSUiWindowId.pipeline_editor).show()

    def show_zmq_manager(self) -> None:
        self.require_pane(OpenHCSUiWindowId.zmq_server_manager).show()

    def show_system_monitor(self) -> None:
        self.require_pane(OpenHCSUiWindowId.system_monitor).show()


@dataclass(slots=True)
class MainWindowDockLayoutStore:
    """Persist opaque native-Qt dock geometry outside scientific edit state."""

    settings: QSettings

    STATE_KEY = "main_window/dock_layout_state"
    STATE_VERSION = 2

    @classmethod
    def for_current_application(cls) -> "MainWindowDockLayoutStore":
        return cls(settings=QSettings())

    def restore(self, main_window: QMainWindow) -> bool:
        stored_state = self.settings.value(self.STATE_KEY)
        if stored_state is None:
            return False

        default_state = main_window.saveState(self.STATE_VERSION)
        if not isinstance(stored_state, QByteArray):
            self._discard_invalid_state()
            return False
        if main_window.restoreState(stored_state, self.STATE_VERSION):
            return True

        main_window.restoreState(default_state, self.STATE_VERSION)
        self._discard_invalid_state()
        return False

    def save(self, main_window: QMainWindow) -> None:
        self.settings.setValue(
            self.STATE_KEY,
            main_window.saveState(self.STATE_VERSION),
        )
        self.settings.sync()

    def _discard_invalid_state(self) -> None:
        self.settings.remove(self.STATE_KEY)
        self.settings.sync()


@dataclass(frozen=True, slots=True)
class MainWindowWidgetConnector:
    """Owns cross-widget wiring between plate and pipeline widgets."""

    def connect(
        self,
        plate_manager: PlateManagerWorkflowSurface,
        pipeline_editor: PipelineEditorWorkflowSurface,
    ) -> None:
        plate_manager.plate_selected.connect(pipeline_editor.set_current_plate)
        plate_manager.orchestrator_config_changed.connect(
            pipeline_editor.on_orchestrator_config_changed
        )
        plate_manager.orchestrator_state_changed.connect(
            pipeline_editor.on_orchestrator_state_changed
        )
        plate_manager.manager_execution_state_changed.connect(
            pipeline_editor.on_manager_execution_state_changed
        )
        plate_manager.pipeline_data_changed.connect(
            pipeline_editor.on_pipeline_data_changed
        )
        plate_manager.cellprofiler_pipeline_imported.connect(
            pipeline_editor.on_cellprofiler_pipeline_imported
        )
        plate_manager.debug_snapshot_available.connect(
            pipeline_editor.show_debug_snapshot
        )
        pipeline_editor.plate_manager = plate_manager

        plate_manager.refresh_prepared_cellprofiler_pipelines()

        if plate_manager.selected_plate_path:
            pipeline_editor.set_current_plate(plate_manager.selected_plate_path)

        logger.debug("Connected plate manager and pipeline editor widgets")


@dataclass(frozen=True, slots=True)
class MainWindowPipelineActions:
    """File-menu actions for the embedded pipeline editor."""

    main_window: QWidget
    pipeline_editor: PipelineEditorWorkflowSurface

    def new_pipeline(self) -> None:
        self.pipeline_editor.pipeline_steps = []
        self.pipeline_editor.update_item_list()
        self.pipeline_editor.update_button_states()
        self.pipeline_editor.pipeline_changed.emit(self.pipeline_editor.pipeline_steps)

    def open_pipeline(self, selected_path: Path | None = None) -> None:
        file_path = selected_path
        if file_path is None:
            from PyQt6.QtWidgets import QFileDialog

            selected, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Open Pipeline",
                "",
                "Function Files (*.func);;CellProfiler Pipelines (*.cppipe);;All Files (*)",
            )
            if selected:
                file_path = Path(selected)
            else:
                return

        self.pipeline_editor.load_pipeline_from_file(file_path)

    def save_pipeline(self, selected_path: Path | None = None) -> None:
        file_path = selected_path
        if file_path is None:
            from PyQt6.QtWidgets import QFileDialog

            selected, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Pipeline",
                "pipeline.func",
                "Function Files (*.func);;All Files (*)",
            )
            if selected:
                file_path = Path(selected)
            else:
                return

        self.pipeline_editor.save_pipeline_to_file(file_path)


@dataclass(frozen=True, slots=True)
class MainWindowLifecycleWorkflow:
    """Main-window lifecycle behavior that spans multiple child widgets."""

    main_window: QWidget
    embedded_widgets: MainWindowEmbeddedWidgets
    floating_windows: dict[str, QWidget]
    status_progress_bar: QProgressBar
    ui_bridge_lifecycle: "MainWindowUiBridgeLifecycle"

    def propagate_config(self, new_config: GlobalPipelineConfig) -> None:
        self.embedded_widgets.require_plate_manager().on_config_changed(new_config)
        self.embedded_widgets.require_pipeline_editor().on_config_changed(new_config)

    def progress_started(self, max_value: int) -> None:
        self.status_progress_bar.setMaximum(max_value)
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setVisible(True)

    def progress_updated(self, value: int) -> None:
        self.status_progress_bar.setValue(value)

    def progress_finished(self) -> None:
        self.status_progress_bar.setVisible(False)

    def runtime_progress_changed(
        self,
        projection: ExecutionRuntimeProjection,
    ) -> None:
        """Render the progress registry's current projection without retaining it."""

        self.status_progress_bar.setRange(0, 100)
        self.status_progress_bar.setValue(round(projection.overall_percent))
        self.status_progress_bar.setVisible(projection.has_active_work)

    def close(self) -> None:
        self.ui_bridge_lifecycle.close()

        logger.info("Stopping system monitor...")
        system_monitor = self.embedded_widgets.require_system_monitor()
        system_monitor.stop_monitoring()

        for scope_id in WindowManager.get_open_scopes():
            try:
                WindowManager.close_window(scope_id)
            except Exception as exc:
                logger.warning("Error closing managed window %s: %s", scope_id, exc)

        for window_name, window in list(self.floating_windows.items()):
            try:
                window.close()
                window.deleteLater()
            except Exception as exc:
                logger.warning("Error cleaning up window %s: %s", window_name, exc)

        self.floating_windows.clear()

        for widget in QApplication.topLevelWidgets():
            if widget is self.main_window:
                continue
            try:
                widget.close()
            except Exception as exc:
                logger.warning("Error closing top-level widget: %s", exc)

        QApplication.processEvents()
        gc.collect()


@dataclass(slots=True)
class MainWindowUiBridgeLifecycle:
    """Mutable owner for the optional main-window UI bridge server."""

    server: "UiBridgeControlServer | None" = None

    @property
    def bound_port(self) -> int | None:
        if self.server is None or not self.server.is_running:
            return None
        return self.server.binding.connection.port

    def reconcile(
        self,
        *,
        config: "AgentUiBridgeConfig",
        transport_config: "OpenHCSZMQConfig",
        create_server: Callable[
            ["AgentUiBridgeConfig", "OpenHCSZMQConfig"],
            "UiBridgeControlServer",
        ],
    ):
        """Make the running server exactly match the requested configuration.

        A failed replacement restarts the previous server before propagating
        the error, so configuration Save cannot strand the live bridge.
        """

        current = self.server
        if not config.enabled:
            self.close()
            return None
        if (
            current is not None
            and current.is_running
            and current.config == config
            and current.transport_config == transport_config
        ):
            return current.binding

        if current is not None:
            current.stop()
            self.server = None
        candidate = None
        try:
            candidate = create_server(config, transport_config)
            binding = candidate.start()
        except Exception as replacement_error:
            if candidate is not None:
                try:
                    candidate.stop()
                except Exception as cleanup_error:
                    replacement_error.add_note(
                        "Candidate UI bridge cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if current is not None:
                try:
                    current.start()
                except Exception as rollback_error:
                    raise RuntimeError(
                        "UI bridge replacement failed and the previous bridge "
                        "could not be restored."
                    ) from ExceptionGroup(
                        "UI bridge replacement and rollback errors",
                        (replacement_error, rollback_error),
                    )
                else:
                    self.server = current
            raise
        self.server = candidate
        return binding

    def close(self) -> None:
        if self.server is None:
            return
        try:
            self.server.stop()
        except Exception as exc:
            logger.warning("Error stopping UI bridge server: %s", exc)
        finally:
            self.server = None


@dataclass(frozen=True, slots=True)
class MainWindowTimeTravelWorkflow:
    """Time-travel shortcut actions and widget refresh."""

    refresh_time_travel_widget: Callable[[], None]
    before_restore: Callable[[], None]

    def back(self) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        return self._run(ObjectStateRegistry.time_travel_back)

    def forward(self) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        return self._run(ObjectStateRegistry.time_travel_forward)

    def to_head(self) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        return self._run(ObjectStateRegistry.time_travel_to_head)

    def to_index(self, index: int) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        return self._run(lambda: ObjectStateRegistry.time_travel_to(index))

    def switch_branch(self, branch: str) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        return self._run(lambda: ObjectStateRegistry.switch_branch(branch))

    def delete_branch(self, branch: str) -> bool:
        from objectstate.object_state import ObjectStateRegistry

        def delete() -> bool:
            if ObjectStateRegistry.get_current_branch() == branch:
                ObjectStateRegistry.switch_branch("main")
            return ObjectStateRegistry.delete_branch(branch)

        return self._run(delete)

    def _run(self, operation: Callable[[], object]) -> bool:
        """Authorize and execute one ObjectState restore operation."""

        try:
            self.before_restore()
            result = operation()
        except Exception as exc:
            logger.warning("Time-travel mutation rejected: %s", exc)
            return False
        self.refresh_time_travel_widget()
        if isinstance(result, bool):
            return result
        return True
