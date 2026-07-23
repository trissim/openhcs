"""
OpenHCS PyQt6 Windows

Window components for OpenHCS PyQt6 GUI application.
All windows migrated from Textual TUI with full feature parity.
"""

from pyqt_reactive.widgets.shared import BaseFormDialog
from openhcs.pyqt_gui.windows.config_window import ConfigWindow
from openhcs.pyqt_gui.windows.help_window import HelpWindow
from openhcs.pyqt_gui.windows.file_browser_window import FileBrowserWindow
from openhcs.pyqt_gui.windows.debug_inspector_window import DebugInspectorWindow
from openhcs.pyqt_gui.windows.synthetic_plate_generator_window import (
    SyntheticPlateGeneratorWindow,
)
from openhcs.pyqt_gui.windows.managed_windows import (
    PlateManagerWindow,
    PipelineEditorWindow,
    ImageBrowserWindow,
    LogViewerWindowWrapper,
    ZMQServerManagerWindow,
)

__all__ = [
    "BaseFormDialog",
    "ConfigWindow",
    "DebugInspectorWindow",
    "HelpWindow",
    "FileBrowserWindow",
    "SyntheticPlateGeneratorWindow",
    "PlateManagerWindow",
    "PipelineEditorWindow",
    "ImageBrowserWindow",
    "LogViewerWindowWrapper",
    "ZMQServerManagerWindow",
]
