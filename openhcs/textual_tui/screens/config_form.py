"""
ConfigFormScreen - Dynamic configuration form using ConfigReflectionService.

Provides plate-specific configuration editing with automatic form generation
from dataclass definitions.
"""

import logging
from typing import Dict, Any, Optional, Generator
from pathlib import Path

from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Static, Input, Checkbox, Select
from textual.reactive import reactive

from openhcs.core.config import GlobalPipelineConfig
from openhcs.textual_tui.services.config_reflection_service import FieldIntrospector

logger = logging.getLogger(__name__)


class ConfigFormScreen(ModalScreen):
    """
    Modal screen for editing plate-specific configuration.
    
    Uses ConfigReflectionService to dynamically generate forms from dataclass
    definitions, providing a flexible and maintainable configuration interface.
    """
    
    # Reactive state
    has_changes = reactive(False)
    
    def __init__(self, 
                 plate_path: str,
                 current_config: GlobalPipelineConfig,
                 title: str = "Plate Configuration"):
        """
        Initialize configuration form.
        
        Args:
            plate_path: Path to the plate being configured
            current_config: Current configuration to edit
            title: Dialog title
        """
        super().__init__()
        self.plate_path = plate_path
        self.current_config = current_config
        self.title = title
        
        # Initialize services
        self.reflection_service = FieldIntrospector()
        
        # Create working copy for editing
        self.editing_config = self._clone_config(current_config)
        self.original_config = self._clone_config(current_config)
        
        # Form widgets storage
        self.form_widgets = {}
        
        logger.info(f"ConfigFormScreen initialized for plate: {plate_path}")
    
    def _clone_config(self, config: GlobalPipelineConfig) -> GlobalPipelineConfig:
        """Create a deep copy of configuration."""
        # Use dataclass fields to create new instance
        import dataclasses
        return dataclasses.replace(config)
    
    def compose(self) -> ComposeResult:
        """Compose the configuration form."""
        with Container(id="config_dialog"):
            # Dialog title
            plate_name = Path(self.plate_path).name
            yield Static(f"{self.title} - {plate_name}", id="dialog_title")
            
            # Status bar
            yield Static("Ready", id="status_bar")
            
            # Scrollable form area
            with ScrollableContainer(id="config_form"):
                yield from self._create_form_fields()
            
            # Action buttons
            with Horizontal(id="dialog_buttons"):
                yield Button("Save", id="save_btn", variant="primary", compact=True, disabled=True)
                yield Button("Reset", id="reset_btn", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)
    
    def _create_form_fields(self) -> Generator[Any, None, None]:
        """Create form fields using FieldIntrospector."""
        # Analyze the configuration dataclass
        field_specs = self.reflection_service.analyze_dataclass(
            GlobalPipelineConfig,
            self.editing_config
        )

        for field_spec in field_specs:
            field_name = field_spec.name
            field_type = field_spec.actual_type
            current_value = field_spec.current_value
            description = field_spec.label
            
            # Create field container
            with Vertical(classes="config_field"):
                # Field label with description
                label_text = f"{field_name.replace('_', ' ').title()}"
                if description:
                    label_text += f" - {description}"
                yield Static(label_text, classes="field_label")
                
                # Create appropriate widget for field type
                widget = self._create_field_widget(field_name, field_type, current_value)
                if widget:
                    self.form_widgets[field_name] = widget
                    yield widget
    
    def _create_field_widget(self, field_name: str, field_type: type, current_value: Any) -> Optional[Any]:
        """Create appropriate widget for field type."""
        widget_id = f"field_{field_name}"
        
        if field_type is bool:
            # Boolean checkbox
            widget = Checkbox(
                label="",
                value=bool(current_value),
                id=widget_id
            )
            widget.on_changed = lambda checked: self._on_field_change(field_name, checked)
            return widget
            
        elif field_type is int:
            # Integer input
            widget = Input(
                value=str(current_value) if current_value is not None else "0",
                placeholder="Enter integer value",
                id=widget_id
            )
            widget.on_changed = lambda value: self._on_field_change(field_name, self._parse_int(value))
            return widget
            
        elif field_type is float:
            # Float input
            widget = Input(
                value=str(current_value) if current_value is not None else "0.0",
                placeholder="Enter decimal value",
                id=widget_id
            )
            widget.on_changed = lambda value: self._on_field_change(field_name, self._parse_float(value))
            return widget
            
        elif field_type is str or field_type is Path:
            # String/Path input
            widget = Input(
                value=str(current_value) if current_value is not None else "",
                placeholder=f"Enter {field_name.replace('_', ' ')}",
                id=widget_id
            )
            widget.on_changed = lambda value: self._on_field_change(field_name, value)
            return widget
            
        else:
            # Fallback to string representation
            widget = Input(
                value=str(current_value) if current_value is not None else "",
                placeholder=f"Enter {field_name.replace('_', ' ')}",
                id=widget_id
            )
            widget.on_changed = lambda value: self._on_field_change(field_name, value)
            return widget
    
    def _parse_int(self, value: str) -> int:
        """Parse integer value with error handling."""
        try:
            return int(value) if value.strip() else 0
        except ValueError:
            return 0
    
    def _parse_float(self, value: str) -> float:
        """Parse float value with error handling."""
        try:
            return float(value) if value.strip() else 0.0
        except ValueError:
            return 0.0
    
    def _on_field_change(self, field_name: str, value: Any) -> None:
        """Handle field value changes."""
        logger.debug(f"Config field changed: {field_name} = {value}")
        
        # Update editing config
        try:
            setattr(self.editing_config, field_name, value)
            self._update_change_tracking()
            self._update_status(f"Modified {field_name.replace('_', ' ')}")
        except Exception as e:
            logger.error(f"Failed to update config field {field_name}: {e}")
            self._update_status(f"Error updating {field_name}")
    
    def _update_change_tracking(self) -> None:
        """Update change tracking state."""
        # Compare current config with original
        has_changes = not self._configs_equal(self.editing_config, self.original_config)
        self.has_changes = has_changes
        
        # Update save button state
        try:
            save_btn = self.query_one("#save_btn", Button)
            save_btn.disabled = not has_changes
        except Exception:
            pass
    
    def _configs_equal(self, config1: GlobalPipelineConfig, config2: GlobalPipelineConfig) -> bool:
        """Compare two configurations for equality."""
        import dataclasses
        return config1 == config2
    
    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        try:
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(message)
        except Exception:
            pass
    
    def watch_has_changes(self, has_changes: bool) -> None:
        """React to changes in has_changes state."""
        # Update dialog title to show unsaved changes
        try:
            title = self.query_one("#dialog_title", Static)
            plate_name = Path(self.plate_path).name
            base_title = f"{self.title} - {plate_name}"
            if has_changes:
                title.update(f"{base_title} *")
            else:
                title.update(base_title)
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_btn":
            self._handle_save()
        elif event.button.id == "reset_btn":
            self._handle_reset()
        elif event.button.id == "cancel_btn":
            self._handle_cancel()
    
    def _handle_save(self) -> None:
        """Handle save button."""
        logger.info(f"Saving configuration for plate: {self.plate_path}")
        self._update_status("Configuration saved")
        self.dismiss(self.editing_config)
    
    def _handle_reset(self) -> None:
        """Handle reset button - restore original values."""
        logger.info("Resetting configuration to original values")
        
        # Reset editing config to original
        self.editing_config = self._clone_config(self.original_config)
        
        # Update form widgets
        self._reset_form_widgets()
        
        # Update change tracking
        self._update_change_tracking()
        self._update_status("Configuration reset to original values")
    
    def _reset_form_widgets(self) -> None:
        """Reset all form widgets to original values."""
        field_specs = self.reflection_service.analyze_dataclass(
            GlobalPipelineConfig,
            self.original_config
        )

        for field_spec in field_specs:
            field_name = field_spec.name
            original_value = field_spec.current_value
            
            widget = self.form_widgets.get(field_name)
            if widget:
                if isinstance(widget, Checkbox):
                    widget.value = bool(original_value)
                elif isinstance(widget, Input):
                    widget.value = str(original_value) if original_value is not None else ""
    
    def _handle_cancel(self) -> None:
        """Handle cancel button."""
        if self.has_changes:
            logger.info("Cancelling configuration with unsaved changes")
            self._update_status("Cancelled - changes discarded")
        else:
            logger.info("Cancelling configuration without changes")
            self._update_status("Cancelled")
        
        self.dismiss(None)
