"""Help dialog screen for OpenHCS Textual TUI."""

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Button, Static


class HelpDialogScreen(ModalScreen):
    """Help dialog with static content from old TUI - floating window style."""

    DEFAULT_CSS = """
    HelpDialogScreen {
        align: center middle;
        background: $background 60%;
    }

    #help_dialog {
        /* Compact sizing */
        width: auto;
        height: auto;
        max-width: 60;
        max-height: 25;
        padding: 1 2;  /* Reasonable padding */
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        margin: 0 0 1 0;  /* Tighter spacing */
    }
    
    #help_content {
        height: auto;
        margin: 0;  /* No extra margin */
    }
    
    .dialog-buttons {
        /* Don't dock - let it flow naturally */
        height: auto;
        align: center middle;
        margin-top: 1;  /* Small gap before button */
    }
    
    .dialog-buttons Button {
        width: 16;
        margin: 0;
    }
    """

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

For detailed documentation, see Nature Methods publication."""

    def compose(self) -> ComposeResult:
        """Compose the floating help dialog."""
        with Container(id="help_dialog", classes="dialog"):
            yield Static("OpenHCS Help", classes="dialog-title")
            
            # Just a container - no need for scroll on small content
            yield Container(
                Static(self.HELP_TEXT),
                id="help_content"
            )
            
            with Container(classes="dialog-buttons"):
                yield Button("Close", id="close_btn", variant="default", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close_btn":
            self.dismiss(None)