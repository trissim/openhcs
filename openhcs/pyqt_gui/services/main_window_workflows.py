"""Workflow objects owned by the PyQt main window."""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QProgressBar, QSplitter, QWidget

from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.services.window_config import WindowSpec
from pyqt_reactive.services.window_manager import WindowManager

if TYPE_CHECKING:
    from openhcs.pyqt_gui.services.ui_bridge_server import UiBridgeControlServer

logger = logging.getLogger(__name__)


class SignalConnectionSurface(ABC):
    @abstractmethod
    def connect(self, callback) -> None:
        raise NotImplementedError


class SignalEmissionSurface(ABC):
    @abstractmethod
    def emit(self, value) -> None:
        raise NotImplementedError


class MainWindowPersistenceSurface(ABC):
    @abstractmethod
    def save_window_state(self) -> None:
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
    def on_orchestrator_state_changed(self, plate_path: str, state: str) -> None:
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
    selected_plate_path: str | None

    @abstractmethod
    def set_pipeline_editor(
        self,
        pipeline_editor: PipelineEditorWorkflowSurface,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def notify_pipeline_definition_changed(self, plate_path: str) -> None:
        raise NotImplementedError


class QtShortcutSequenceAuthority:
    """Convert Qt key events into configured shortcut strings."""

    _KEY_NAMES = {
        Qt.Key.Key_Z: "Z",
        Qt.Key.Key_Y: "Y",
    }

    @classmethod
    def from_event(cls, event) -> str | None:
        if event.type() != QEvent.Type.KeyPress:
            return None
        key_name = cls._key_name(event.key())
        if key_name is None:
            return None
        return cls._modifier_prefix(event.modifiers()) + key_name

    @classmethod
    def _key_name(cls, key: int) -> str | None:
        if key in cls._KEY_NAMES:
            return cls._KEY_NAMES[key]
        return None

    @staticmethod
    def _modifier_prefix(modifiers) -> str:
        parts: list[str] = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if not parts:
            return ""
        return "+".join(parts) + "+"


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


@dataclass(frozen=True, slots=True)
class MainWindowSpecDefinition:
    """Declarative record for one WindowManager-managed window."""

    window_id: str
    title: str
    window_class: type[QDialog]
    initialize_on_startup: bool = False

    def build(self) -> WindowSpec:
        return WindowSpec(
            window_id=self.window_id,
            title=self.title,
            window_class=self.window_class,
            initialize_on_startup=self.initialize_on_startup,
        )

    @classmethod
    def from_row(
        cls,
        row: tuple[str, str, type[QDialog], bool],
    ) -> MainWindowSpecDefinition:
        return cls(
            window_id=row[0],
            title=row[1],
            window_class=row[2],
            initialize_on_startup=row[3],
        )


def build_main_window_specs() -> dict[str, WindowSpec]:
    """Build all WindowManager-managed window specifications."""

    from openhcs.pyqt_gui.windows.managed_windows import (
        ImageBrowserWindow,
        LogViewerWindowWrapper,
        PipelineEditorWindow,
        PlateManagerWindow,
        ZMQServerManagerWindow,
    )

    rows = (
        ("plate_manager", "Plate Manager", PlateManagerWindow, True),
        ("pipeline_editor", "Pipeline Editor", PipelineEditorWindow, False),
        ("image_browser", "Image Browser", ImageBrowserWindow, False),
        ("log_viewer", "Log Viewer", LogViewerWindowWrapper, True),
        ("zmq_server_manager", "ZMQ Server Manager", ZMQServerManagerWindow, False),
    )
    definitions = tuple(MainWindowSpecDefinition.from_row(row) for row in rows)
    return {definition.window_id: definition.build() for definition in definitions}


@dataclass(slots=True)
class MainWindowEmbeddedWidgets:
    """Explicit owned-widget graph for the embedded main window layout."""

    system_monitor: QWidget | None = None
    plate_manager: QWidget | None = None
    pipeline_editor: QWidget | None = None
    zmq_manager: QWidget | None = None
    left_splitter: QSplitter | None = None
    main_splitter: QSplitter | None = None
    top_splitter: QSplitter | None = None

    def require_plate_manager(self) -> QWidget:
        if self.plate_manager is None:
            raise RuntimeError("Plate manager widget has not been initialized")
        return self.plate_manager

    def require_pipeline_editor(self) -> QWidget:
        if self.pipeline_editor is None:
            raise RuntimeError("Pipeline editor widget has not been initialized")
        return self.pipeline_editor

    def require_zmq_manager(self) -> QWidget:
        if self.zmq_manager is None:
            raise RuntimeError("ZMQ manager widget has not been initialized")
        return self.zmq_manager

    def require_system_monitor(self) -> QWidget:
        if self.system_monitor is None:
            raise RuntimeError("System monitor widget has not been initialized")
        return self.system_monitor

    def show_defaults(self) -> None:
        self.require_plate_manager().show()
        self.require_pipeline_editor().show()

    def show_plate_manager(self) -> None:
        self._show_widget(self.require_plate_manager())
        self._set_splitter_ratios(self.left_splitter, (0.7, 0.3))
        self._set_splitter_ratios(self.main_splitter, (0.6, 0.4))

    def show_pipeline_editor(self) -> None:
        self._show_widget(self.require_pipeline_editor())
        self._set_splitter_ratios(self.main_splitter, (0.4, 0.6))

    def show_zmq_manager(self) -> None:
        self._show_widget(self.require_zmq_manager())
        self._set_splitter_ratios(self.left_splitter, (0.5, 0.5))
        self._set_splitter_ratios(self.main_splitter, (0.6, 0.4))

    @staticmethod
    def _show_widget(widget: QWidget) -> None:
        if not widget.isVisible():
            widget.show()

    @staticmethod
    def _set_splitter_ratios(splitter: QSplitter | None, ratios: tuple[float, float]) -> None:
        if splitter is None:
            return
        sizes = splitter.sizes()
        if len(sizes) == 2:
            total = sum(sizes)
            splitter.setSizes([int(total * ratios[0]), int(total * ratios[1])])


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
        plate_manager.set_pipeline_editor(pipeline_editor)
        pipeline_editor.plate_manager = plate_manager

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

    main_window: MainWindowPersistenceSurface
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

        self.main_window.save_window_state()
        QApplication.processEvents()
        gc.collect()


@dataclass(slots=True)
class MainWindowUiBridgeLifecycle:
    """Mutable owner for the optional main-window UI bridge server."""

    server: "UiBridgeControlServer | None" = None

    def set_server(self, server: "UiBridgeControlServer") -> None:
        self.server = server

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

    def back(self) -> None:
        from openhcs.config_framework.object_state import ObjectStateRegistry

        ObjectStateRegistry.time_travel_back()
        self.refresh_time_travel_widget()

    def forward(self) -> None:
        from openhcs.config_framework.object_state import ObjectStateRegistry

        ObjectStateRegistry.time_travel_forward()
        self.refresh_time_travel_widget()

    def to_head(self) -> None:
        from openhcs.config_framework.object_state import ObjectStateRegistry

        ObjectStateRegistry.time_travel_to_head()
        self.refresh_time_travel_widget()
