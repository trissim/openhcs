"""Base modal widget for consistent modal patterns."""

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static


class BaseModalScreen(ModalScreen):
    """Base class for all modal screens with consistent patterns."""
    
    def __init__(self, title: str, can_cancel: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.can_cancel = can_cancel
    
    def compose(self) -> ComposeResult:
        """Compose the base modal structure."""
        with Container(id="modal_dialog"):
            yield Static(self.title, id="dialog_title")
            yield from self.compose_content()
            yield from self.compose_buttons()
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide modal content."""
        yield Static("Base modal content")
    
    def compose_buttons(self) -> ComposeResult:
        """Compose standard modal buttons."""
        with Horizontal(id="dialog_buttons"):
            yield Button("OK", id="ok_btn", variant="primary", compact=True)
            if self.can_cancel:
                yield Button("Cancel", id="cancel_btn", compact=True)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle standard button presses."""
        if event.button.id == "ok_btn":
            self.on_ok()
        elif event.button.id == "cancel_btn":
            self.on_cancel()
    
    def on_ok(self) -> None:
        """Override in subclasses for OK button handling."""
        self.dismiss(True)
    
    def on_cancel(self) -> None:
        """Override in subclasses for Cancel button handling."""
        self.dismiss(None)
