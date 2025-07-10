"""Help window for OpenHCS Textual TUI."""

from textual.app import ComposeResult
from textual.widgets import Static, Button
from textual.containers import ScrollableContainer, Horizontal

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow


class HelpWindow(BaseOpenHCSWindow):
    """Help window using textual-window system."""

    DEFAULT_CSS = """
    HelpWindow ScrollableContainer {
        text-align: left;
        align: left top;
    }

    HelpWindow Static {
        text-align: left;
    }
    """

    def __init__(self, content: str = "Help content goes here.", **kwargs):
        self.content = content

        super().__init__(
            window_id="help",
            title="OpenHCS Help",
            mode="temporary",
            **kwargs
        )

        # Calculate dynamic minimum size based on content
        self._calculate_dynamic_size()

    HELP_TEXT = """OpenHCS - Open High-Content Screening

🔬 Visual Programming for Cell Biology Research

Key Features:
• GPU-accelerated image processing
• Visual pipeline building
• Multi-backend storage support
• Real-time parameter editing

Workflow:
1. Add Plate → Select microscopy data
2. Edit Step → Visual function selection
3. Compile → Create execution plan
4. Run → Process images

For detailed documentation, see Nature Methods
publication."""

    def compose(self) -> ComposeResult:
        """Compose the help window content with scrollable area."""
        # Scrollable content area
        with ScrollableContainer():
            yield Static(self.HELP_TEXT)

        # Close button at bottom - wrapped in Horizontal for automatic centering
        with Horizontal():
            yield Button("Close", id="close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.close_window()

    def _calculate_dynamic_size(self) -> None:
        """Calculate and set dynamic minimum window size based on content."""
        try:
            # Calculate width based on content lines
            max_width = 50  # Base minimum width

            content_lines = self.HELP_TEXT.split('\n')
            for line in content_lines:
                max_width = max(max_width, len(line) + 10)

            # Calculate height based on number of lines
            min_height = 15  # Base minimum height
            content_line_count = len(content_lines)
            min_height = max(min_height, content_line_count + 5)  # Content + margins + button

            # Cap maximum size to reasonable limits
            max_width = min(max_width, 120)
            min_height = min(min_height, 40)

            # Set dynamic minimum size
            self.styles.min_width = max_width
            self.styles.min_height = min_height

        except Exception:
            # Fallback to default sizes if calculation fails
            self.styles.min_width = 50
            self.styles.min_height = 15
    