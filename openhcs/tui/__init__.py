"""
Terminal User Interface (TUI) for OpenHCS.

This package provides a TUI for interacting with OpenHCS, featuring:
- Clean layout architecture
- Modular editor components
- Pipeline visualization
- Step configuration
"""

# Import key components from organized submodules
from openhcs.tui.layout import CanonicalTUILayout, SimpleOpenHCSTUILauncher
from openhcs.tui.editors import StepParameterEditor, FunctionPatternEditor, DualEditorPane

__all__ = [
    'CanonicalTUILayout',
    'SimpleOpenHCSTUILauncher',
    'StepParameterEditor',
    'FunctionPatternEditor',
    'DualEditorPane',
]
