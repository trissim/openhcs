"""Help window for OpenHCS Textual TUI."""

from textual.app import ComposeResult
from textual.widgets import Static, Button
from textual.containers import ScrollableContainer

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow


class HelpWindow(BaseOpenHCSWindow):
    """Help window using textual-window system."""

    DEFAULT_CSS = """
    HelpWindow {
        width: 60; height: 20;
        min-width: 60; min-height: 20;
    }
    HelpWindow #dialog_container {
        width: 80;
        height: 35;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(
            window_id="help",
            title="OpenHCS Help",
            mode="temporary",
            **kwargs
        )

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

        # Close button at bottom
        yield Button("Close", id="close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.close_window()
    