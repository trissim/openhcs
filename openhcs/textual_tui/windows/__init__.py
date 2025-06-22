"""OpenHCS TUI Windows package."""

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.windows.help_window import HelpWindow
from openhcs.textual_tui.windows.config_window import ConfigWindow
from openhcs.textual_tui.windows.dual_editor_window import DualEditorWindow
from openhcs.textual_tui.windows.pipeline_plate_window import PipelinePlateWindow
from openhcs.textual_tui.windows.file_browser_window import FileBrowserWindow, BrowserMode, open_file_browser_window
from openhcs.textual_tui.windows.function_selector_window import FunctionSelectorWindow
from openhcs.textual_tui.windows.group_by_selector_window import GroupBySelectorWindow
from openhcs.textual_tui.windows.help_windows import DocstringHelpWindow, ParameterHelpWindow, HelpableWidget

__all__ = [
    "BaseOpenHCSWindow",
    "HelpWindow",
    "ConfigWindow",
    "DualEditorWindow",
    "PipelinePlateWindow",
    "FileBrowserWindow",
    "BrowserMode",
    "open_file_browser_window",
    "FunctionSelectorWindow",
    "GroupBySelectorWindow",
    "DocstringHelpWindow",
    "ParameterHelpWindow",
    "HelpableWidget"
]