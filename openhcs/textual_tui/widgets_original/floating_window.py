"""
Floating window widgets for OpenHCS TUI.

Provides base classes for modal dialogs and floating windows.
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
        width: 80;
        height: 25;
        background: $surface;
        padding: 0;
        border: thick $background 80%;
    }

    .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 0;
        padding: 1 2;
    }

    .dialog-buttons {
        height: auto;
        align: center middle;
        margin-top: 0;
        padding: 1 2;
    }

    .dialog-buttons Button {
        margin: 0 1;
    }

    .dialog-content {
        padding: 1 2;
        text-align: center;
    }

    .button-row {
        width: 100%;
        height: auto;
        content-align: center middle;
    }

    /* Compact confirmation dialogs */
    ConfirmationWindow #dialog_container {
        max-width: 50;
        height: auto;
        padding: 0;
    }

    ConfirmationWindow .dialog-title {
        padding: 1 2 0 2;
        margin-bottom: 0;
    }

    ConfirmationWindow .dialog-content {
        padding: 1 2;
        margin: 0;
    }

    ConfirmationWindow .dialog-buttons {
        padding: 0 2 1 2;
        margin: 0;
    }

    ConfirmationWindow .button-row {
        content-align: center middle;
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
        with Container(id="dialog_container", classes="dialog") as container:
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

        # Auto-dismiss unless explicitly prevented (None means don't dismiss)
        if result is not None:
            self.dismiss(result)

    def handle_button_action(self, button_id: str, button_text: str) -> Any:
        """
        Override this method to handle button actions.

        Returns:
            Value to return when dialog is dismissed, or None to prevent dismissal
        """
        return True  # Default: dismiss with True


class SimpleTextWindow(BaseFloatingWindow):
    """Simple floating window for displaying text with buttons."""

    def __init__(self, title: str, content: str, buttons: Optional[List[str]] = None, **kwargs):
        self.content_text = content
        self.button_texts = buttons or ['OK']
        super().__init__(title=title, **kwargs)

    def compose_content(self) -> ComposeResult:
        with Container(classes="dialog-content"):
            yield Static(self.content_text)

    def compose_buttons(self) -> ComposeResult:
        from textual.containers import Horizontal
        with Horizontal(classes="button-row"):
            for i, text in enumerate(self.button_texts):
                button_id = f"btn_{i}_{text.lower().replace(' ', '_')}"
                yield Button(text, id=button_id, compact=True)


class ConfirmationWindow(SimpleTextWindow):
    """Specialized confirmation dialog with Yes/No buttons."""

    def __init__(self, title: str, message: str, **kwargs):
        super().__init__(title=title, content=message, buttons=['Yes', 'No'], **kwargs)

    def handle_button_action(self, button_id: str, button_text: str) -> Any:
        # Return True for Yes, False for No
        if button_text.lower() == 'yes':
            return True
        elif button_text.lower() == 'no':
            return False
        return False
