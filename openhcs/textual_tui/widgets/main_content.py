"""
MainContent Widget for OpenHCS Textual TUI

Main content area with horizontal split between PlateManager and PipelineEditor.
Matches the layout from the current prompt-toolkit TUI.
"""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from textual.widget import Widget

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager

from .plate_manager import PlateManagerWidget
from .pipeline_editor import PipelineEditorWidget

logger = logging.getLogger(__name__)


class MainContent(Widget):
    """
    Main content area widget.
    
    Layout: Horizontal split with PlateManager (left) and PipelineEditor (right)
    Uses proper frame containers with titles matching the current TUI.
    """
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """
        Initialize the main content area.
        
        Args:
            filemanager: FileManager instance for file operations
            global_config: Global configuration
        """
        super().__init__()
        self.filemanager = filemanager
        self.global_config = global_config
        logger.debug("MainContent initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the main content layout."""
        with Horizontal():
            # Left pane: Plate Manager with proper border title
            plate_container = Container(id="plate_manager_container")
            plate_container.border_title = "Plate Manager"
            with plate_container:
                yield PlateManagerWidget(
                    filemanager=self.filemanager,
                    global_config=self.global_config
                )

            # Right pane: Pipeline Editor with proper border title
            pipeline_container = Container(id="pipeline_editor_container")
            pipeline_container.border_title = "Pipeline Editor"
            with pipeline_container:
                yield PipelineEditorWidget(
                    filemanager=self.filemanager,
                    global_config=self.global_config
                )
    
    def on_mount(self) -> None:
        """Called when the main content is mounted."""
        logger.info("MainContent mounted")
        
        # Set up communication between panes
        plate_manager = self.query_one(PlateManagerWidget)
        pipeline_editor = self.query_one(PipelineEditorWidget)

        # Set up bidirectional references
        plate_manager.pipeline_editor = pipeline_editor
        pipeline_editor.plate_manager = plate_manager

        # Connect plate selection to pipeline editor
        def on_plate_selected(plate_path: str):
            """Handle plate selection from PlateManager."""
            pipeline_editor.current_plate = plate_path
            # Also send plate status for constraint checking
            plate_status = plate_manager.get_plate_status(plate_path)
            pipeline_editor.current_plate_status = plate_status
            logger.debug(f"Plate selected: {plate_path} (status: {plate_status})")

        # Store the callback for future use
        plate_manager.on_plate_selected = on_plate_selected
