"""
Hybrid TUI implementation for OpenHCS.

Combines TUI2's clean MVC architecture with TUI's working components,
removing schema dependencies and using static analysis for parameter introspection.

Architecture:
- Controllers: Manage state and coordinate between components
- Components: Handle UI rendering and user interaction
- Utils: Pure functions for dialogs, file operations, static analysis
- Interfaces: Component interface definitions

Key Features:
- Schema-free parameter introspection using inspect module
- Working function pattern editor with dict key management
- Complete step settings editor with dynamic form generation
- Clean MVC separation with async/await support
"""

__version__ = "1.0.0"
from .controllers import AppController, DualEditorController
from .components import (
    FunctionPatternEditor,
    StepSettingsEditor,
    ParameterEditor,
    GroupedDropdown,
    FileManagerBrowser
)
from .main import HybridTUIApp, run_tui

__all__ = [
    'AppController',
    'DualEditorController',
    'FunctionPatternEditor',
    'StepSettingsEditor',
    'ParameterEditor',
    'GroupedDropdown',
    'FileManagerBrowser',
    'HybridTUIApp',
    'run_tui'
]

# Version info for tracking hybrid implementation
HYBRID_VERSION = {
    'version': __version__,
    'architecture': 'TUI2_MVC + TUI_Components',
    'schema_free': True,
    'static_analysis': True
}
