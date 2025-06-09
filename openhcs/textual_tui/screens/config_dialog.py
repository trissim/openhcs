"""Config dialog screen for OpenHCS Textual TUI."""

from typing import Any, Type
from textual.app import ComposeResult
from textual.widgets import Button

from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow


class ConfigDialogScreen(BaseFloatingWindow):
    """Configuration dialog with form generation from dataclass using global floating window system."""

    def __init__(self, config_class: Type, current_config: Any, **kwargs):
        self.config_class = config_class
        self.current_config = current_config

        # Analyze config structure
        self.field_specs = FieldIntrospector.analyze_dataclass(config_class, current_config)

        # Create form widget
        self.config_form = ConfigFormWidget(self.field_specs)

        super().__init__(title="Configuration", **kwargs)



    def compose_content(self) -> ComposeResult:
        """Compose the config dialog content."""
        yield self.config_form

    def compose_buttons(self) -> ComposeResult:
        """Provide Save/Cancel buttons."""
        yield Button("Save", id="save", compact=True)
        yield Button("Cancel", id="cancel", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Save/Cancel logic."""
        if button_text == 'Save':
            return self._handle_save()
        elif button_text == 'Cancel':
            return None  # Dismiss with None
        return None

    def _handle_save(self):
        """Handle save button - create new config instance."""
        try:
            # Get form values
            form_values = self.config_form.get_config_values()

            # Create new config instance
            new_config = self.config_class(**form_values)

            # Return new config
            return new_config

        except Exception as e:
            # TODO: Show error message to user
            # For now, just return None (don't save)
            return None


