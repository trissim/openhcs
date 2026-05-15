"""
OpenHCS PyQt6 Widgets

Widget components for the OpenHCS PyQt6 GUI application.
All widgets migrated from Textual TUI with full feature parity.
"""

from pyqt_reactive.widgets.system_monitor import SystemMonitorWidget
from openhcs.pyqt_gui.widgets.debug_toolbar import DebugToolbarWidget
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.source_bindings_editor import SourceBindingsEditorWidget
from pyqt_reactive.widgets.function_pane import FunctionPaneWidget
from pyqt_reactive.widgets.status_bar import StatusBarWidget

__all__ = [
    "SystemMonitorWidget",
    "DebugToolbarWidget",
    "PlateManagerWidget",
    "PipelineEditorWidget",
    "SourceBindingsEditorWidget",
    "FunctionPaneWidget",
    "StatusBarWidget",
]
