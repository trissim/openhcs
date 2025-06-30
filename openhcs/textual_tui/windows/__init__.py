"""OpenHCS TUI Windows package."""

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.windows.help_window import HelpWindow
from openhcs.textual_tui.windows.config_window import ConfigWindow
from openhcs.textual_tui.windows.dual_editor_window import DualEditorWindow
from openhcs.textual_tui.windows.pipeline_plate_window import PipelinePlateWindow
from openhcs.textual_tui.windows.file_browser_window import FileBrowserWindow, BrowserMode, open_file_browser_window
from openhcs.textual_tui.windows.function_selector_window import FunctionSelectorWindow
from openhcs.textual_tui.windows.group_by_selector_window import GroupBySelectorWindow
from openhcs.textual_tui.windows.help_windows import DocstringHelpWindow, ParameterHelpWindow
from openhcs.textual_tui.windows.terminal_window import TerminalWindow
from openhcs.textual_tui.windows.advanced_terminal_window import AdvancedTerminalWindow
from openhcs.textual_tui.windows.debug_class_explorer import DebugClassExplorerWindow
from openhcs.textual_tui.windows.multi_orchestrator_config_window import MultiOrchestratorConfigWindow
from openhcs.textual_tui.windows.toolong_window import ToolongWindow

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
    "TerminalWindow",
    "AdvancedTerminalWindow",
    "DebugClassExplorerWindow",
    "MultiOrchestratorConfigWindow",
    "ToolongWindow"
]