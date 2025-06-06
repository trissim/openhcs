"""
Complex editor components for OpenHCS TUI.

This module contains specialized editors for complex data structures:
- StepParameterEditor: Pure AbstractStep parameter editing
- FunctionPatternView: Function pattern editor with parameter editing (via views module)
- DualEditorPane: Clean tab coordinator for step + function editing
- FileManagerBrowser: File browser with FileManager integration
"""

from .step_parameter_editor import StepParameterEditor
# FunctionPatternEditor replaced by FunctionPatternView in views module
from .dual_editor_pane import DualEditorPane
from .file_browser import FileManagerBrowser

# Legacy DualStepFuncEditorPane moved to archive/

__all__ = [
    'StepParameterEditor',
    'DualEditorPane',
    'FileManagerBrowser'
    # FunctionPatternEditor replaced by FunctionPatternView in views module
]
