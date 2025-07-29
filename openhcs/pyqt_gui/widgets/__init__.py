"""
OpenHCS PyQt6 Widgets

Widget components for the OpenHCS PyQt6 GUI application.
All widgets migrated from Textual TUI with full feature parity.
"""

from openhcs.pyqt_gui.widgets.system_monitor import SystemMonitorWidget
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.function_pane import FunctionPaneWidget
from openhcs.pyqt_gui.widgets.status_bar import StatusBarWidget

__all__ = [
    "SystemMonitorWidget",
    "PlateManagerWidget", 
    "PipelineEditorWidget",
    "FunctionPaneWidget",
    "StatusBarWidget"
]
