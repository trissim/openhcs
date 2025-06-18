"""Help dialog screen for OpenHCS Textual TUI."""

from textual.app import ComposeResult
from textual.widgets import Static, Button
from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow


class HelpDialogScreen(BaseFloatingWindow):
    """Help dialog using the global floating window system."""

    DEFAULT_CSS = """
    HelpDialogScreen #dialog_container {
        width: 80;
        height: 35;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(title="OpenHCS Help", **kwargs)

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

    def compose_content(self) -> ComposeResult:
        """Compose the help dialog content."""
        yield Static(self.HELP_TEXT)

    def compose_buttons(self) -> ComposeResult:
        """Provide Close button."""
        yield Button("Close", id="close", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Close button dismisses dialog."""
        return False  # Dismiss with False result
    