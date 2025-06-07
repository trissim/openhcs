"""Config dialog screen for OpenHCS Textual TUI."""

from typing import Any, Type
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Horizontal, Center, Middle
from textual.widgets import Button, Static

from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.textual_tui.widgets.config_form import ConfigFormWidget


class ConfigDialogScreen(ModalScreen):
    """Configuration dialog with form generation from dataclass - floating window style."""

    DEFAULT_CSS = """
    ConfigDialogScreen {
        align: center middle;
        background: $background 60%;
    }

    #config_dialog {
        /* Inherit app's dialog style */
        max-width: 90%;
        max-height: 80%;
        min-width: 60;
        min-height: 25;
    }
    """

    def __init__(self, config_class: Type, current_config: Any, **kwargs):
        super().__init__(**kwargs)
        self.config_class = config_class
        self.current_config = current_config

        # Analyze config structure
        self.field_specs = FieldIntrospector.analyze_dataclass(config_class, current_config)

        # Create form widget
        self.config_form = ConfigFormWidget(self.field_specs)

    def compose(self) -> ComposeResult:
        """Compose the floating config dialog."""
        # Use app's dialog class for consistent styling
        with Container(id="config_dialog", classes="dialog"):
            yield Static("Configuration", classes="dialog-title")

            with ScrollableContainer():
                yield self.config_form

            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save_btn", variant="primary", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_btn":
            self.on_save()
        elif event.button.id == "cancel_btn":
            self.on_cancel()
    
    def on_save(self) -> None:
        """Handle save button - create new config instance."""
        try:
            # Get form values
            form_values = self.config_form.get_config_values()
            
            # Create new config instance
            new_config = self.config_class(**form_values)
            
            # Return new config
            self.dismiss(new_config)
            
        except Exception as e:
            # TODO: Show error message to user
            # For now, just dismiss without saving
            self.dismiss(None)
    
    def on_cancel(self) -> None:
        """Handle cancel button."""
        self.dismiss(None)
