"""Configuration window for OpenHCS Textual TUI."""

from typing import Type, Any, Callable, Optional
from textual.app import ComposeResult
from textual.widgets import Button
from textual.containers import Container

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector
from openhcs.textual_tui.widgets.config_form import ConfigFormWidget


class ConfigWindow(BaseOpenHCSWindow):
    """Configuration window using textual-window system."""

    DEFAULT_CSS = """
    ConfigWindow {
        width: 60; height: 20;
        min-width: 60; min-height: 20;
    }
    ConfigWindow #dialog_container {
        width: 80;
        height: 30;
    }
    """

    def __init__(self, config_class: Type, current_config: Any,
                 on_save_callback: Optional[Callable] = None, **kwargs):
        """
        Initialize config window.

        Args:
            config_class: Configuration class type
            current_config: Current configuration instance
            on_save_callback: Function to call when config is saved
        """
        super().__init__(
            window_id="config_dialog",
            title="Configuration",
            mode="temporary",
            **kwargs
        )
        self.config_class = config_class
        self.current_config = current_config
        self.on_save_callback = on_save_callback

        # Analyze config structure (same as original ConfigDialogScreen)
        self.field_specs = FieldIntrospector.analyze_dataclass(config_class, current_config)

        # Create the form widget (reuse existing logic)
        self.config_form = ConfigFormWidget(self.field_specs)

    def calculate_content_height(self) -> int:
        """Calculate dialog height based on number of fields."""
        # Base height for title, buttons, padding
        base_height = 8

        # Height per field (label + input + spacing)
        field_height = 2

        # Count total fields (including nested)
        total_fields = len(self.field_specs)

        # Add extra height for nested dataclasses
        for spec in self.field_specs:
            if hasattr(spec.actual_type, '__dataclass_fields__'):
                # Nested dataclass adds extra height for collapsible
                total_fields += len(spec.actual_type.__dataclass_fields__) + 1

        calculated = base_height + (total_fields * field_height)

        # Clamp between reasonable bounds
        return min(max(calculated, 15), 40)



    def compose(self) -> ComposeResult:
        """Compose the config window content."""
        with Container(classes="dialog-content"):
            yield self.config_form

        # Buttons
        yield Button("Save", id="save", compact=True)
        yield Button("Cancel", id="cancel", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save":
            self._handle_save()
        elif event.button.id == "cancel":
            self.close_window()

    def _handle_save(self):
        """Handle save button - reuse existing logic from ConfigDialogScreen."""
        # Get form values (same method as original)
        form_values = self.config_form.get_config_values()

        # Create new config instance (same as original)
        new_config = self.config_class(**form_values)

        # Call the callback if provided
        if self.on_save_callback:
            self.on_save_callback(new_config)

        self.close_window()


