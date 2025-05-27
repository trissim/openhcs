"""
Controllers for hybrid TUI.

Contains controller classes that manage state and coordinate between components,
following the MVC architecture pattern.
"""

from .dual_editor_controller import DualEditorController
from .app_controller import AppController

# Controllers to be implemented if needed
# from .pipeline_editor_controller import PipelineEditorController
# from .plate_manager_controller import PlateManagerController

__all__ = [
    'DualEditorController',
    'AppController'
]
