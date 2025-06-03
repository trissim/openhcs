"""
Complex editor components for OpenHCS TUI.

This module contains specialized editors for complex data structures:
- StepParameterEditor: Pure AbstractStep parameter editing
- FunctionPatternEditor: Function pattern editor with parameter editing
- DualEditorPane: Clean tab coordinator for step + function editing
- FileManagerBrowser: File browser with FileManager integration
"""

from .step_parameter_editor import StepParameterEditor
from .function_pattern_editor import FunctionPatternEditor
from .dual_editor_pane import DualEditorPane
from .file_browser import FileManagerBrowser

# Legacy DualStepFuncEditorPane moved to archive/

__all__ = [
    'StepParameterEditor',
    'FunctionPatternEditor',
    'DualEditorPane',
    'FileManagerBrowser'
]
