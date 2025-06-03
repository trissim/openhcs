"""
Main UI panes for OpenHCS TUI.

This module contains the primary UI panes that make up the main interface:
- PlateManagerPane: Plate management and orchestrator operations
- PipelineEditorPane: Pipeline editing and step management
"""

from .plate_manager import PlateManagerPane
from .pipeline_editor import PipelineEditorPane

__all__ = [
    'PlateManagerPane',
    'PipelineEditorPane'
]
