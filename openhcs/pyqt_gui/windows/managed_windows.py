"""
Managed window implementations using WindowManager.show_or_focus().

Each window is created by a factory function passed to WindowManager.
"""

from pathlib import Path

from PyQt6.QtWidgets import QDialog, QVBoxLayout

from openhcs.pyqt_gui.services.main_window_workflows import MainWindowWidgetConnector


class ManagedPlatePipelineConnector:
    """Connects managed plate and pipeline windows when both are open."""

    PLATE_WINDOW_ID = "plate_manager"
    PIPELINE_WINDOW_ID = "pipeline_editor"

    def connect_plate(self, plate_widget) -> None:
        pipeline_widget = self._open_widget(self.PIPELINE_WINDOW_ID)
        if pipeline_widget is not None:
            MainWindowWidgetConnector().connect(plate_widget, pipeline_widget)

    def connect_pipeline(self, pipeline_widget) -> None:
        plate_widget = self._open_widget(self.PLATE_WINDOW_ID)
        if plate_widget is not None:
            MainWindowWidgetConnector().connect(plate_widget, pipeline_widget)

    @staticmethod
    def _open_widget(window_id: str):
        from pyqt_reactive.services.window_manager import WindowManager

        window = WindowManager._scoped_windows.get(window_id)
        return window.widget if window is not None else None


class PlateManagerWindow(QDialog):
    def __init__(self, main_window, service_adapter):
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setWindowTitle("Plate Manager")
        self.setModal(False)
        self.resize(600, 400)

        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        layout = QVBoxLayout(self)
        self.widget = PlateManagerWidget(
            self.service_adapter,
            self.service_adapter.get_current_color_scheme(),
            gui_config=self.service_adapter.widget_gui_config,
        )
        layout.addWidget(self.widget)
        self._setup_connections()

    def _setup_connections(self):
        self.widget.global_config_changed.connect(
            lambda: self.main_window.on_config_changed(
                self.service_adapter.get_global_config()
            )
        )

        self._setup_progress_signals()

        self._connect_to_pipeline_editor()

    def _setup_progress_signals(self):
        self.widget.progress_started.connect(
            self.main_window._on_plate_progress_started
        )
        self.widget.progress_updated.connect(
            self.main_window._on_plate_progress_updated
        )
        self.widget.progress_finished.connect(
            self.main_window._on_plate_progress_finished
        )

    def _connect_to_pipeline_editor(self):
        ManagedPlatePipelineConnector().connect_plate(self.widget)


class PipelineEditorWindow(QDialog):
    def __init__(self, main_window, service_adapter):
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setWindowTitle("Pipeline Editor")
        self.setModal(False)
        self.resize(800, 600)

        from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

        layout = QVBoxLayout(self)
        self.widget = PipelineEditorWidget(
            self.service_adapter,
            self.service_adapter.get_current_color_scheme(),
        )
        layout.addWidget(self.widget)
        self._setup_connections()

    def _setup_connections(self):
        ManagedPlatePipelineConnector().connect_pipeline(self.widget)


class ImageBrowserWindow(QDialog):
    def __init__(self, main_window, service_adapter):
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setWindowTitle("Image Browser")
        self.setModal(False)
        self.resize(900, 600)

        from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserWidget

        layout = QVBoxLayout(self)
        self.widget = ImageBrowserWidget(
            orchestrator=None,
            color_scheme=self.service_adapter.get_current_color_scheme(),
            zmq_config=self.main_window.runtime_context.ui_config.zmq,
        )
        self.main_window.ui_config_changed.connect(
            lambda config: self.widget.set_zmq_config(config.zmq)
        )
        layout.addWidget(self.widget)
        self._setup_connections()

    def _setup_connections(self):
        from pyqt_reactive.services.window_manager import WindowManager

        plate_widgets = []
        embedded_plate_widget = self.main_window.embedded_widgets.plate_manager
        if embedded_plate_widget is not None:
            plate_widgets.append(embedded_plate_widget)

        plate_window = WindowManager._scoped_windows.get("plate_manager")
        if plate_window is not None and plate_window.widget not in plate_widgets:
            plate_widgets.append(plate_window.widget)

        for plate_widget in plate_widgets:
            plate_widget.plate_selected.connect(
                lambda _plate_path=None, plate_widget=plate_widget: self._update_orchestrator(
                    plate_widget
                )
            )
            self._update_orchestrator(plate_widget)

    def _update_orchestrator(self, plate_widget):
        orchestrator = plate_widget.get_selected_orchestrator()
        if orchestrator:
            self.widget.set_orchestrator(orchestrator)


class LogViewerWindowWrapper(QDialog):
    def __init__(self, main_window, service_adapter):
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setWindowTitle("Log Viewer")
        self.setModal(False)
        self.resize(900, 700)

        from pyqt_reactive.widgets.log_viewer import LogViewerWindow

        layout = QVBoxLayout(self)
        self.widget = LogViewerWindow(
            self.main_window.file_manager, self.service_adapter
        )
        layout.addWidget(self.widget)

    def switch_to_log(self, log_file_path: Path) -> None:
        """Display one server log through the wrapped log-viewer owner."""
        self.widget.switch_to_log(log_file_path)

    def closeEvent(self, event) -> None:
        """Close the composed viewer through its generic lifecycle authority."""
        self.widget.cleanup()
        super().closeEvent(event)


class ZMQServerManagerWindow(QDialog):
    def __init__(self, main_window, service_adapter):
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setWindowTitle("ZMQ Server Manager")
        self.setModal(False)
        self.resize(600, 400)

        from PyQt6.QtWidgets import QVBoxLayout
        from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import (
            ZMQServerManagerWidget,
        )

        layout = QVBoxLayout(self)

        self.widget = ZMQServerManagerWidget(
            ports_to_scan=self.main_window.zmq_server_manager_ports_to_scan(),
            title="ZMQ Servers (Execution + UI Bridge + Napari + Fiji)",
            style_generator=self.service_adapter.get_style_generator(),
            config=self.main_window.runtime_context.ui_config.zmq,
        )
        self.main_window.ui_config_changed.connect(
            lambda config: self.widget.set_zmq_config(
                config.zmq,
                self.main_window.zmq_server_manager_ports_to_scan(),
            )
        )
        layout.addWidget(self.widget)
        self.widget.log_file_opened.connect(self.main_window._open_log_file_in_viewer)
