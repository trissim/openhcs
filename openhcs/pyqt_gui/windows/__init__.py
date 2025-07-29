"""
OpenHCS PyQt6 Windows

Window components for the OpenHCS PyQt6 GUI application.
All windows migrated from Textual TUI with full feature parity.
"""

from openhcs.pyqt_gui.windows.config_window import ConfigWindow
from openhcs.pyqt_gui.windows.help_window import HelpWindow
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
from openhcs.pyqt_gui.windows.file_browser_window import FileBrowserWindow
from openhcs.pyqt_gui.windows.function_selector_window import FunctionSelectorWindow

__all__ = [
    "ConfigWindow",
    "HelpWindow", 
    "DualEditorWindow",
    "FileBrowserWindow",
    "FunctionSelectorWindow"
]
