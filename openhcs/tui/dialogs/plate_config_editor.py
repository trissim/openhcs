from prompt_toolkit.layout import Container
"""
Plate Configuration Editor Pane for OpenHCS TUI.

This module implements a pane for editing plate-specific configurations,
which are overrides of the GlobalPipelineConfig.
"""
import asyncio
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING, List, Union
import copy # For deepcopying config
import dataclasses # To inspect dataclass fields

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, Dimension, Window, FormattedTextControl, ScrollablePane
from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioList, Checkbox, Dialog
from prompt_toolkit.formatted_text import HTML

from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig, VFSConfig # Import config classes
from openhcs.constants import Microscope # Import Microscope enum
from enum import Enum # For isinstance check

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.config import GlobalPipelineConfig

logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]

class PlateConfigEditorPane:
    """
    A pane for editing plate-specific configurations.
    """
    def __init__(self, state: "TUIState", orchestrator: "PipelineOrchestrator"):
        self.state = state
        self.orchestrator = orchestrator
        
        base_config: "GlobalPipelineConfig"
        if hasattr(self.orchestrator, 'config_override') and self.orchestrator.config_override is not None:
            base_config = self.orchestrator.config_override
            logger.info(f"PlateConfigEditorPane: Editing existing override for plate {getattr(orchestrator, 'plate_id', 'N/A')}")
        elif hasattr(self.orchestrator, 'config') and self.orchestrator.config is not None:
            base_config = self.orchestrator.config 
            logger.info(f"PlateConfigEditorPane: Editing a copy of current config for plate {getattr(orchestrator, 'plate_id', 'N/A')}")
        else: # Fallback to global config from state if orchestrator has no config
            base_config = self.state.global_config
            logger.info(f"PlateConfigEditorPane: Editing a copy of global config for plate {getattr(orchestrator, 'plate_id', 'N/A')}")

        if base_config is None:
            logger.error("PlateConfigEditorPane: Base configuration is None. Cannot initialize editor.")
            self.editing_config: Optional["GlobalPipelineConfig"] = None
        else:
            self.editing_config = copy.deepcopy(base_config)

        self.config_param_inputs: Dict[str, Any] = {} # To store UI input widgets

        # UI Components
        self.save_button = SafeButton("Save", handler=self._handle_save)
        self.close_button = SafeButton("Close", handler=self._handle_close)
        self.save_button.disabled = True # Disabled until a change is made
        
        self._container = self._build_layout()

    def _config_value_changed(self, field_path: str, new_value: Any, widget_type: str = "text_area"):
        """
        Callback when a config value is changed in the UI.
        Updates self.editing_config and enables the save button.
        """
        if not self._validate_config_state():
            return

        logger.debug(f"Config value attempting change: {field_path} = {new_value} (from {widget_type})")

        try:
            obj_ptr, field_name = self._navigate_to_field(field_path)
            field_meta = self._get_field_metadata(obj_ptr, field_name)
            current_value_type = self._extract_field_type(field_meta.type)

            converted_value = self._convert_value(new_value, current_value_type)
            self._update_field_if_changed(obj_ptr, field_name, field_path, converted_value)

            get_app().invalidate()
        except (ValueError, TypeError, KeyError) as e:
            self._handle_conversion_error(e, field_path, new_value, type(e).__name__)
        except Exception as e:
            logger.error(f"Unexpected error updating config field '{field_path}': {e}", exc_info=True)

    def _validate_config_state(self) -> bool:
        """Validate that editing_config is available."""
        if self.editing_config is None:
            logger.error("_config_value_changed called but editing_config is None.")
            return False
        return True

    def _navigate_to_field(self, field_path: str) -> tuple:
        """Navigate to the field object and return object pointer and field name."""
        obj_ptr = self.editing_config
        parts = field_path.split('.')

        for part in parts[:-1]:
            obj_ptr = getattr(obj_ptr, part)

        return obj_ptr, parts[-1]

    def _get_field_metadata(self, obj_ptr: Any, field_name: str):
        """Get field metadata from dataclass fields."""
        return next(f for f in dataclasses.fields(obj_ptr) if f.name == field_name)

    def _extract_field_type(self, field_type: type) -> type:
        """Extract the actual type from Optional types."""
        # Handle Optional types by getting the underlying type
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # Assuming Optional[T] is Union[T, NoneType]
            args = [arg for arg in field_type.__args__ if arg is not type(None)]
            if args:
                return args[0]
        return field_type

    def _convert_value(self, new_value: Any, current_value_type: type) -> Any:
        """Convert new value to the appropriate type."""
        if current_value_type == int:
            return int(new_value)
        elif current_value_type == str:
            return str(new_value)
        elif current_value_type == bool:
            return self._convert_bool_value(new_value)
        elif issubclass(current_value_type, Enum):
            return current_value_type[new_value]  # Converts string name to Enum member
        else:
            return new_value

    def _convert_bool_value(self, new_value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(new_value, str):  # e.g. "True" from a text-based widget
            return new_value.lower() == 'true'
        else:  # Assumes new_value is already bool
            return bool(new_value)

    def _update_field_if_changed(self, obj_ptr: Any, field_name: str, field_path: str, converted_value: Any) -> None:
        """Update field if value has actually changed."""
        if getattr(obj_ptr, field_name) != converted_value:
            setattr(obj_ptr, field_name, converted_value)
            self.save_button.disabled = False
            logger.info(f"Config field '{field_path}' updated to '{converted_value}'. Save enabled.")
        else:
            logger.debug(f"Config field '{field_path}' value '{converted_value}' is same as current. No change.")

    def _handle_conversion_error(self, error: Exception, field_path: str, new_value: Any, error_type: str) -> None:
        """Handle conversion errors with appropriate logging."""
        if error_type == "ValueError":
            logger.error(f"ValueError converting value '{new_value}' for field '{field_path}'.")
        elif error_type == "TypeError":
            logger.error(f"TypeError for field '{field_path}' with value '{new_value}'.")
        elif error_type == "KeyError":
            logger.error(f"KeyError: '{new_value}' is not a valid member name for Enum field '{field_path}'.")

    def _create_config_widgets(self, config_obj: Any, parent_path: str = "") -> List[Any]:
        """Recursively creates UI widgets for a config object's fields."""
        if not dataclasses.is_dataclass(config_obj):
            return [Label(f"Non-dataclass object: {type(config_obj)}")]

        widgets = []
        for dc_field in dataclasses.fields(config_obj):
            widget = self._create_field_widget(dc_field, config_obj, parent_path)
            if widget:
                widgets.append(widget)

        return widgets

    def _create_field_widget(self, dc_field, config_obj: Any, parent_path: str) -> Any:
        """Create widget for a single field."""
        field_info = self._extract_field_widget_info(dc_field, config_obj, parent_path)

        # Handle nested dataclass
        if dataclasses.is_dataclass(field_info['current_value']):
            return self._create_nested_dataclass_widget(field_info)

        # Create input widget
        input_widget = self._create_input_widget(field_info)
        if input_widget:
            self.config_param_inputs[field_info['handler_key']] = input_widget
            return VSplit([field_info['label'], input_widget], padding=0, width=Dimension(max=100))

        return None

    def _extract_field_widget_info(self, dc_field, config_obj: Any, parent_path: str) -> dict:
        """Extract field information for widget creation."""
        field_name = dc_field.name
        current_value = getattr(config_obj, field_name)
        full_path = f"{parent_path}.{field_name}" if parent_path else field_name

        label_text = field_name.replace('_', ' ').title()
        label = Label(f"{label_text}: ", width=35)

        # Determine the actual type for Optional fields
        field_type, is_optional = self._extract_field_type_info(dc_field.type)

        return {
            'field_name': field_name,
            'current_value': current_value,
            'full_path': full_path,
            'label': label,
            'label_text': label_text,
            'field_type': field_type,
            'is_optional': is_optional,
            'handler_key': full_path
        }

    def _extract_field_type_info(self, field_type: type) -> tuple:
        """Extract field type and optional status."""
        is_optional = False
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            args = [arg for arg in field_type.__args__ if arg is not type(None)]
            if len(args) == 1 and type(None) in field_type.__args__:
                field_type = args[0]
                is_optional = True
        return field_type, is_optional

    def _create_nested_dataclass_widget(self, field_info: dict) -> HSplit:
        """Create widget for nested dataclass fields."""
        return HSplit([
            Label(HTML(f"<b><u>{field_info['label_text']}</u></b>")),
            Box(HSplit(self._create_config_widgets(field_info['current_value'], field_info['full_path'])), padding_left=2),
            Window(height=1)  # Spacer
        ])

    def _create_input_widget(self, field_info: dict) -> Any:
        """Create appropriate input widget based on field type."""
        field_type = field_info['field_type']
        current_value = field_info['current_value']
        full_path = field_info['full_path']
        is_optional = field_info['is_optional']

        if field_type == int:
            return self._create_int_widget(current_value, full_path)
        elif field_type == str:
            return self._create_str_widget(current_value, full_path, is_optional)
        elif field_type == bool:
            field_info['handler_key'] = (full_path, "checkbox")
            return Checkbox(checked=bool(current_value))
        elif issubclass(field_type, Enum):
            field_info['handler_key'] = (full_path, "radiolist")
            return self._create_enum_widget(field_type, current_value)
        else:
            return Label(f"Unsupported: {field_type}")

    def _create_int_widget(self, current_value: Any, full_path: str) -> TextArea:
        """Create text area widget for integer fields."""
        widget = TextArea(text=str(current_value), multiline=False, height=1)
        def int_accept_handler(path_capture=full_path, widget_capture=widget):
            self._config_value_changed(path_capture, widget_capture.text, "int_text_area")
        widget.accept_handler = int_accept_handler
        return widget

    def _create_str_widget(self, current_value: Any, full_path: str, is_optional: bool) -> TextArea:
        """Create text area widget for string fields."""
        text_val = str(current_value) if current_value is not None else ("" if is_optional else "None")
        widget = TextArea(text=text_val, multiline=False, height=1)
        def str_accept_handler(path_capture=full_path, widget_capture=widget):
            self._config_value_changed(path_capture, widget_capture.text, "str_text_area")
        widget.accept_handler = str_accept_handler
        return widget

    def _create_enum_widget(self, field_type: type, current_value: Any) -> RadioList:
        """Create radio list widget for enum fields."""
        enum_values = [(member.name, member.name) for member in field_type]
        return RadioList(values=enum_values, current_value=current_value.name)

    def _build_layout(self) -> Container:
        """Builds the main layout for the editor pane."""
        if self.editing_config is None:
            return Frame(Label("Error: Configuration not loaded."), title="Error")

        self.config_param_inputs.clear() # Clear previous inputs before rebuilding
        config_widgets_list = self._create_config_widgets(self.editing_config)
        
        # Fallback if no widgets created (e.g., empty config or all unsupported)
        if not config_widgets_list:
            config_widgets_list = [Label("No editable configuration fields found.")]

        self.config_view_container = ScrollablePane(HSplit(config_widgets_list))

        buttons = VSplit([
            self.save_button,
            self.close_button,
        ], padding=1, align="RIGHT")

        title_text = f"Edit Config: {getattr(self.orchestrator, 'plate_id', 'Unknown Plate')}"
        
        return Frame(
            HSplit([
                self.config_view_container,
                Window(height=1, char='-'), # Separator
                buttons,
            ]),
            title=title_text
        )

    @property
    def container(self) -> Container:
        return self._container

    def _handle_save(self) -> None:
        logger.info("PlateConfigEditorPane: Save clicked.")
        if self.editing_config is None:
            logger.error("PlateConfigEditorPane: No configuration to save.")
            # Optionally notify user via TUIState error event
            return

        # Phase 1: Ensure self.editing_config is fully up-to-date from all UI widgets.
        # TextAreas with accept_handler would have already called _config_value_changed.
        # This loop explicitly processes RadioLists and Checkboxes.
        logger.debug("Updating editing_config from UI widgets before save...")
        for key, widget in self.config_param_inputs.items():
            field_path: str
            widget_identifier: str # e.g., "checkbox", "radiolist"

            if isinstance(key, tuple):
                field_path, widget_identifier = key
            else: # Assumed to be TextArea path, already handled by accept_handler
                continue

            current_widget_value: Any = None
            if widget_identifier == "radiolist" and isinstance(widget, RadioList):
                current_widget_value = widget.current_value # This is the string member name
            elif widget_identifier == "checkbox" and isinstance(widget, Checkbox):
                current_widget_value = widget.checked
            
            if current_widget_value is not None:
                # Call _config_value_changed to ensure consistent type conversion and update
                # This will also re-enable save_button if it makes a change, which is fine.
                self._config_value_changed(field_path, current_widget_value, widget_type=widget_identifier)
        
        logger.debug("Finished updating editing_config from UI widgets.")

        # Phase 2: Apply the fully updated self.editing_config to the orchestrator.
        try:
            # Directly update the orchestrator's config attribute with the edited copy.
            # This aligns with the understanding that orchestrator stores its config in self.config.
            if hasattr(self.orchestrator, 'config'):
                self.orchestrator.config = self.editing_config
                logger.info(f"PlateConfigEditorPane: Orchestrator 'config' attribute updated for plate {getattr(self.orchestrator, 'plate_id', 'N/A')}.")
            else:
                logger.error(f"PlateConfigEditorPane: Orchestrator for plate {getattr(self.orchestrator, 'plate_id', 'N/A')} does not have a 'config' attribute to update.")
                async def notify_error_no_config_attr():
                    await self.state.notify("error_message", {"message": "Save failed: Orchestrator has no 'config' attribute."})
                asyncio.create_task(notify_error_no_config_attr())
                return

            # If save was successful:
            self.save_button.disabled = True
            get_app().invalidate()

            async def notify_save_success():
                await self.state.notify('plate_config_saved', {'orchestrator_id': getattr(self.orchestrator, 'plate_id', None)})
            asyncio.create_task(notify_save_success())

        except Exception as e:
            logger.error(f"Error applying updated config to orchestrator for plate {getattr(self.orchestrator, 'plate_id', 'N/A')}: {e}", exc_info=True)
            async def notify_apply_error():
                await self.state.notify("error_message", {"message": f"Error saving plate config: {e}"})
            asyncio.create_task(notify_apply_error())


    def _handle_close(self) -> None:
        logger.info("PlateConfigEditorPane: Close clicked.")
        async def notify_cancel():
            await self.state.notify('plate_config_editing_cancelled')
        asyncio.create_task(notify_cancel())

    async def shutdown(self):
        """Cleanup resources if any."""
        logger.info(f"PlateConfigEditorPane for {getattr(self.orchestrator, 'plate_id', 'N/A')} shutting down.")