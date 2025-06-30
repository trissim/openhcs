"""Shared Pipeline/Plate Manager window for OpenHCS TUI."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.textual_tui.widgets.plate_manager import PlateManagerWidget


class PipelinePlateWindow(BaseOpenHCSWindow):
    """Shared window containing both Pipeline Editor and Plate Manager widgets."""

    DEFAULT_CSS = """
    PipelinePlateWindow {
        width: 80; height: 20;
        min-width: 80; min-height: 20;
    }
    PipelinePlateWindow #content_pane {
        padding: 0;  /* Remove all padding from content pane */
    }
    """

    def __init__(self, filemanager, global_config, **kwargs):
        """
        Initialize the shared pipeline/plate window.
        
        Args:
            filemanager: File manager instance
            global_config: Global configuration instance
        """
        super().__init__(
            window_id="pipeline_plate",
            title="Main",  # Changed to "Main" as requested
            mode="permanent",  # Use permanent mode so state persists
            allow_maximize=True,
            **kwargs
        )
        self.filemanager = filemanager
        self.global_config = global_config
        self.current_view = "pipeline"
        
        # Create both widgets as instance variables
        self.pipeline_widget = PipelineEditorWidget(filemanager, global_config)
        self.plate_widget = PlateManagerWidget(filemanager, global_config)

    def compose(self) -> ComposeResult:
        """Compose the shared window with both widgets in horizontal layout with titles."""
        with Horizontal():
            # Left pane: Plate Manager with proper border title
            plate_container = Container(id="plate_manager_container")
            plate_container.border_title = "Plate Manager"
            with plate_container:
                yield self.plate_widget

            # Right pane: Pipeline Editor with proper border title
            pipeline_container = Container(id="pipeline_editor_container")
            pipeline_container.border_title = "Pipeline Editor"
            with pipeline_container:
                yield self.pipeline_widget

    def on_mount(self):
        """Set up widget coordination - EXACT copy from MainContent.on_mount()."""
        # Both widgets are always visible in the horizontal layout
        # No need to hide anything - this matches the original MainContent behavior

        # Set up bidirectional widget references (preserve existing relationships)
        self.pipeline_widget.plate_manager = self.plate_widget
        self.plate_widget.pipeline_editor = self.pipeline_widget

        # Set up plate selection callback (exact coordination logic from MainContent)
        def on_plate_selected(plate_path: str):
            self.pipeline_widget.current_plate = plate_path

        self.plate_widget.on_plate_selected = on_plate_selected

    def show_pipeline_editor(self):
        """Focus on pipeline editor (but both are always visible)."""
        self.current_view = "pipeline"
        # Both widgets are always visible in horizontal layout
        # This method is kept for compatibility but doesn't hide anything

    def show_plate_manager(self):
        """Focus on plate manager (but both are always visible)."""
        self.current_view = "plate"
        # Both widgets are always visible in horizontal layout
        # This method is kept for compatibility but doesn't hide anything

    def show_both(self):
        """Show both pipeline editor and plate manager together (default behavior)."""
        self.current_view = "both"
        # Both widgets are always visible in horizontal layout - this is the default
