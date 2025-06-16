"""
Simple floating window system for OpenHCS Textual TUI.

Provides content-aware floating dialogs using standard Textual patterns.
"""
from typing import List, Optional, Any
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static, Button


class BaseFloatingWindow(ModalScreen):
    """
    Simple content-aware floating window using standard Textual patterns.
    """

    DEFAULT_CSS = """
    BaseFloatingWindow {
        align: center middle;
    }

    #dialog_container {
        width: auto;
        height: auto;
        max-width: 80;
        max-height: 80%;
        background: $surface;
        padding: 1;
    }

    .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    .dialog-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    .dialog-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str = "", **kwargs):
        """Initialize floating window with title."""
        super().__init__(**kwargs)
        self.window_title = title

    def compose_content(self) -> ComposeResult:
        """Override this method to provide the main window content."""
        yield Static("Override compose_content() to add your content")

    def compose_buttons(self) -> ComposeResult:
        """Override this method to provide custom buttons."""
        yield Button("OK", id="ok", compact=True)

    def compose(self) -> ComposeResult:
        """Compose the complete floating window structure."""
        with Container(id="dialog_container") as container:
            container.styles.border = ("solid", "white")
            # Title (if provided)
            if self.window_title:
                yield Static(self.window_title, classes="dialog-title")

            # Main content area
            yield from self.compose_content()

            # Button area
            with Container(classes="dialog-buttons"):
                yield from self.compose_buttons()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses with automatic dialog dismissal."""
        # Ignore form reset buttons - only handle dialog's main buttons
        if event.button.id.startswith("reset_"):
            return  # Let the form handle reset buttons

        # Convert button label (Content object) to string
        button_text = str(event.button.label) if event.button.label else ""
        result = self.handle_button_action(event.button.id, button_text)

        # Auto-dismiss unless explicitly prevented
        if result is not False:
            self.dismiss(result)

    def handle_button_action(self, button_id: str, button_text: str) -> Any:
        """
        Override this method to handle button actions.

        Returns:
            Value to return when dialog is dismissed, or False to prevent dismissal
        """
        return True  # Default: dismiss with True


class SimpleTextWindow(BaseFloatingWindow):
    """Simple floating window for displaying text with buttons."""

    def __init__(self, title: str, content: str, buttons: Optional[List[str]] = None, **kwargs):
        self.content_text = content
        self.button_texts = buttons or ['OK']
        super().__init__(title=title, **kwargs)

    def compose_content(self) -> ComposeResult:
        yield Static(self.content_text)

    def compose_buttons(self) -> ComposeResult:
        from textual.containers import Horizontal
        with Horizontal():
            for i, text in enumerate(self.button_texts):
                button_id = f"btn_{i}_{text.lower().replace(' ', '_')}"
                yield Button(text, id=button_id, compact=True)


class ConfirmationWindow(SimpleTextWindow):
    """Specialized confirmation dialog with Yes/No buttons."""

    def __init__(self, title: str, message: str, **kwargs):
        super().__init__(title=title, content=message, buttons=['Yes', 'No'], **kwargs)

    def handle_button_action(self, button_id: str, button_text: str) -> Any:
        return button_text.lower() == 'yes'
