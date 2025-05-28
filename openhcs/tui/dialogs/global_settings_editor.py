"""
Global Settings Editor Dialog for OpenHCS TUI.

This module implements a dialog for viewing and editing the
GlobalPipelineConfig.
"""
import asyncio
import logging
from typing import Any, Optional, List, Union, get_args, get_origin, Type, Dict, Literal # Added Literal
from enum import Enum
import dataclasses # For field introspection if using dataclasses

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, ScrollablePane, Dimension, Float # Added ScrollablePane, Dimension, Float
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, Checkbox, Box # Added Box

from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig, MicroscopeConfig # Ensure MicroscopeConfig is imported
from openhcs.constants.constants import Backend, Microscope # For dropdowns
# from ..tui_architecture import TUIState # Avoid circular import
from ..utils import show_error_dialog # Assuming show_error_dialog is available

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

class GlobalSettingsEditorDialog:
    """
    A dialog for editing global OpenHCS settings (GlobalPipelineConfig).
    """
    def __init__(self, state: Any, initial_config: GlobalPipelineConfig):
        self.state = state
        self.original_config = initial_config
        # Pydantic models have a model_copy method for deep copies
        self.editing_config = initial_config.model_copy(deep=True) 

        self.input_widgets: Dict[str, Any] = {}
        self.error_label: Label = Label("", style="class:error-text") # For displaying save errors
        
        self.save_button = SafeButton("Save", handler=self._save_settings)
        self.save_button.disabled = True # Initially disabled
        self.cancel_button = SafeButton("Cancel", handler=self._cancel)
        
        self.dialog: Optional[Dialog] = None
        self._build_dialog() # Call to build the dialog structure

    def _get_config_value(self, path: str) -> Any:
        """Gets a value from the editing_config using a dot-separated path."""
        obj = self.editing_config
        for part in path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning(f"Path part '{part}' not found in config for path '{path}' during get.")
                return None # Path not found
        return obj

    def _set_config_value(self, config_obj: Any, field_path_parts: List[str], value: Any) -> bool:
        """Sets a value in a (potentially nested) config object using parts of a path."""
        obj = config_obj
        for part in field_path_parts[:-1]: # Navigate to the parent object
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.error(f"Error setting config value: Path part '{part}' not found.")
                return False
        
        last_part = field_path_parts[-1]
        if hasattr(obj, last_part):
            try:
                setattr(obj, last_part, value)
                return True
            except Exception as e: # Catch potential Pydantic validation errors or other issues
                logger.error(f"Error setting attribute '{last_part}' to '{value}': {e}")
                return False
        logger.error(f"Error setting config value: Attribute '{last_part}' not found in parent object.")
        return False

    def _on_setting_changed(self, *args, **kwargs):
        """Callback when any setting widget changes. Enables/disables save button."""
        is_changed = False
        temp_compare_config = self.original_config.model_copy(deep=True)

        for path, widget in self.input_widgets.items():
            try:
                field_type = self._get_field_type_by_path(path)
                if field_type is None: continue

                current_widget_value = self._get_field_value_from_widget(widget, field_type)
                
                # Get original value for comparison
                original_obj_segment = self.original_config
                path_parts = path.split('.')
                for part in path_parts[:-1]:
                    original_obj_segment = getattr(original_obj_segment, part)
                original_value = getattr(original_obj_segment, path_parts[-1])

                if current_widget_value != original_value:
                    is_changed = True
                    break
            except ValueError: # If current widget value is invalid (e.g., non-int for int field)
                is_changed = True # Treat as changed to enable save and trigger validation
                break
            except Exception as e: # Catch other errors during comparison
                logger.debug(f"Error comparing field {path} for change detection: {e}")
                is_changed = True # Assume change if comparison fails
                break
                
        self.save_button.disabled = not is_changed
        get_app().invalidate()

    def _get_field_type_by_path(self, path: str) -> Optional[Type]:
        """Gets the type hint of a field by its dot-separated path from GlobalPipelineConfig."""
        obj_type: Type = GlobalPipelineConfig
        parts = path.split('.')
        
        for i, part_name in enumerate(parts):
            is_last_part = (i == len(parts) - 1)
            
            current_fields = {}
            if hasattr(obj_type, 'model_fields'): # Pydantic model
                current_fields = obj_type.model_fields
            elif dataclasses.is_dataclass(obj_type): # Standard dataclass
                current_fields = {f.name: f for f in dataclasses.fields(obj_type)}
            else:
                logger.warning(f"Cannot determine fields for non-model/dataclass type: {obj_type} at '{part_name}' in path '{path}'")
                return None

            if part_name not in current_fields:
                logger.warning(f"Field '{part_name}' not found in {obj_type} for path '{path}'.")
                return None
            
            field_def = current_fields[part_name]
            field_annotation = field_def.annotation if hasattr(field_def, 'annotation') else field_def.type # Pydantic vs dataclass

            if is_last_part:
                return field_annotation
            
            obj_type = field_annotation
            # Handle Optional[SomeConfigClass] before next iteration
            if get_origin(obj_type) is Union and type(None) in get_args(obj_type):
                non_none_args = [t for t in get_args(obj_type) if t is not type(None)]
                if non_none_args:
                    obj_type = non_none_args[0]
                else: # Should not happen for config classes
                    return None
        return None


    def _build_dialog_rows_for_model(self, model_instance: Any, model_type: Type, prefix: str = "") -> List[Any]:
        """Helper to recursively build UI rows for a given model instance and type."""
        field_definitions = self._get_field_definitions(model_type)
        if not field_definitions:
            return []

        rows = []
        for field_name, field_obj in field_definitions.items():
            field_path = f"{prefix}{field_name}"
            field_info = self._extract_field_info(field_obj, model_instance, field_name)

            if self._is_nested_model(field_info['actual_type']):
                nested_rows = self._build_nested_model_rows(field_info, field_path, prefix)
                rows.extend(nested_rows)
            else:
                widget = self._create_field_widget(field_info)
                if widget:
                    self._register_widget(field_path, widget, field_info['field_label_text'], prefix)
                    rows.append(self._create_field_row(field_info['field_label_text'], widget, prefix))

        return rows

    def _get_field_definitions(self, model_type: Type) -> dict:
        """Get field definitions for the model type."""
        if hasattr(model_type, 'model_fields'):  # Pydantic
            return model_type.model_fields
        elif dataclasses.is_dataclass(model_type):  # Standard dataclasses
            return {f.name: f for f in dataclasses.fields(model_type)}
        return {}

    def _extract_field_info(self, field_obj: Any, model_instance: Any, field_name: str) -> dict:
        """Extract field information for UI creation."""
        field_type_hint = field_obj.annotation if hasattr(field_obj, 'annotation') else field_obj.type
        current_value = getattr(model_instance, field_name, None)

        actual_type = field_type_hint
        is_optional = get_origin(field_type_hint) is Union and type(None) in get_args(field_type_hint)
        if is_optional:
            actual_type = next((t for t in get_args(field_type_hint) if t is not type(None)), actual_type)

        return {
            'field_name': field_name,
            'field_label_text': field_name.replace('_', ' ').title(),
            'field_type_hint': field_type_hint,
            'actual_type': actual_type,
            'is_optional': is_optional,
            'current_value': current_value,
            'origin_type': get_origin(actual_type)
        }

    def _is_nested_model(self, actual_type: Type) -> bool:
        """Check if the type is a nested model."""
        return hasattr(actual_type, 'model_fields') or dataclasses.is_dataclass(actual_type)

    def _build_nested_model_rows(self, field_info: dict, field_path: str, prefix: str) -> List[Any]:
        """Build rows for nested model fields."""
        rows = []
        field_label_text = field_info['field_label_text']
        current_value = field_info['current_value']
        actual_type = field_info['actual_type']
        is_optional = field_info['is_optional']

        # Add section title
        padding_left = len(prefix.split('.')) - 1 if prefix else 0
        rows.append(VSplit([Label(f"{field_label_text} Settings:", style="class:dialog.section-title")],
                          height=1, padding=Dimension(left=padding_left)))

        nested_prefix = f"{field_path}."

        if current_value is None and not is_optional:
            logger.error(f"Nested model {field_info['field_name']} is None but not Optional. Cannot build UI.")
            rows.append(Label(f"Error: {field_label_text} is None", style="class:error-text"))
        elif current_value is not None:
            rows.extend(self._build_dialog_rows_for_model(current_value, actual_type, nested_prefix))
        elif is_optional:
            rows.append(Label(f"{field_label_text}: (Not set)", style="class:italic"))

        return rows

    def _create_field_widget(self, field_info: dict) -> Any:
        """Create appropriate widget for field type."""
        actual_type = field_info['actual_type']
        origin_type = field_info['origin_type']
        current_value = field_info['current_value']
        is_optional = field_info['is_optional']

        if actual_type is bool:
            return self._create_bool_widget(current_value)
        elif origin_type is list or origin_type is List:
            return self._create_list_widget(current_value)
        elif origin_type is Literal:
            return self._create_literal_widget(actual_type, current_value)
        elif isinstance(actual_type, type) and issubclass(actual_type, Enum):
            return self._create_enum_widget(actual_type, current_value, is_optional)
        elif actual_type in [int, str, float]:
            return self._create_basic_type_widget(current_value)
        else:
            return Label(text=f"{str(current_value)} (Unsupported type: {actual_type})")

    def _create_bool_widget(self, current_value: Any) -> Checkbox:
        """Create checkbox widget for boolean fields."""
        widget = Checkbox(checked=bool(current_value))
        original_mouse_handler = widget.control.mouse_handler

        def create_checkbox_mouse_handler(cb_widget_instance):
            def new_mouse_handler(mouse_event):
                res = original_mouse_handler(mouse_event)
                self._on_setting_changed()
                return res
            return new_mouse_handler

        widget.control.mouse_handler = create_checkbox_mouse_handler(widget)
        return widget

    def _create_list_widget(self, current_value: Any) -> TextArea:
        """Create text area widget for list fields."""
        widget = TextArea(text=str(current_value or []), multiline=True, height=3, style="class:input-field")
        widget.buffer.on_text_changed += lambda b: self._on_setting_changed()
        return widget

    def _create_literal_widget(self, actual_type: Type, current_value: Any) -> RadioList:
        """Create radio list widget for literal fields."""
        options = [(val, str(val)) for val in get_args(actual_type)]
        widget = RadioList(values=options, current_value=current_value)
        widget.on_value_changed += lambda w_val: self._on_setting_changed()
        return widget

    def _create_enum_widget(self, actual_type: Type, current_value: Any, is_optional: bool) -> RadioList:
        """Create radio list widget for enum fields."""
        options = [(member.value, member.name) for name, member in actual_type.__members__.items()]
        enum_current_value = current_value.value if current_value else None

        if is_optional and current_value is None:
            options = [(None, "(Not set)")] + options

        widget = RadioList(values=options, current_value=enum_current_value)
        widget.on_value_changed += lambda w_val: self._on_setting_changed()
        return widget

    def _create_basic_type_widget(self, current_value: Any) -> TextArea:
        """Create text area widget for basic types (int, str, float)."""
        widget = TextArea(text=str(current_value) if current_value is not None else "",
                         multiline=False, height=1, style="class:input-field")
        widget.buffer.on_text_changed += lambda b: self._on_setting_changed()
        return widget

    def _register_widget(self, field_path: str, widget: Any, field_label_text: str, prefix: str) -> None:
        """Register widget in the input_widgets dictionary."""
        self.input_widgets[field_path] = widget

    def _create_field_row(self, field_label_text: str, widget: Any, prefix: str) -> Any:
        """Create a UI row for the field."""
        padding_left = len(prefix.split('.')) - 1 if prefix else 0
        return Box(HSplit([Label(f"{field_label_text}:", width=30), widget]),
                  padding=Dimension(left=padding_left))

    def _build_dialog(self):
        self.input_widgets.clear()
        
        title_label = Label("Global Pipeline Settings", style="class:dialog.title")
        # Main sections - you can add more specific ones if needed
        general_rows_title = Label("General Settings", style="class:dialog.section-title")
        
        all_rows = [title_label, VSplit([],height=1)] # Add a small spacer after title
        all_rows.extend(self._build_dialog_rows_for_model(self.editing_config, GlobalPipelineConfig))
        
        all_rows.append(self.error_label)

        scrollable_body = ScrollablePane(HSplit(all_rows, padding=Dimension(left=1, right=1)))

        self.dialog = Dialog(
            title="Global Settings Editor",
            body=scrollable_body,
            buttons=[self.save_button, self.cancel_button],
            width=Dimension(preferred=90, max=120),
            height=Dimension(preferred=30, max=50),
            modal=True
        )

    def _get_field_value_from_widget(self, widget: Any, field_type_hint: Type) -> Any:
        """Extract value from widget based on field type."""
        type_info = self._extract_type_info(field_type_hint)

        if isinstance(widget, Checkbox):
            return self._get_checkbox_value(widget)
        elif isinstance(widget, RadioList):
            return self._get_radiolist_value(widget, type_info)
        elif isinstance(widget, TextArea):
            return self._get_textarea_value(widget, type_info)
        else:
            raise ValueError(f"Unsupported widget type for value retrieval: {type(widget)}")

    def _extract_type_info(self, field_type_hint: Type) -> dict:
        """Extract type information from field type hint."""
        actual_type = field_type_hint
        is_optional = get_origin(field_type_hint) is Union and type(None) in get_args(field_type_hint)
        if is_optional:
            actual_type = next((t for t in get_args(field_type_hint) if t is not type(None)), actual_type)

        return {
            'actual_type': actual_type,
            'is_optional': is_optional,
            'origin_type': get_origin(actual_type)
        }

    def _get_checkbox_value(self, widget: Checkbox) -> bool:
        """Get value from checkbox widget."""
        return widget.checked

    def _get_radiolist_value(self, widget: RadioList, type_info: dict) -> Any:
        """Get value from radio list widget."""
        current_radio_value = widget.current_value
        actual_type = type_info['actual_type']
        is_optional = type_info['is_optional']

        if is_optional and current_radio_value is None:
            return None

        if isinstance(actual_type, type) and issubclass(actual_type, Enum):
            return actual_type(current_radio_value) if current_radio_value is not None else None

        return current_radio_value  # For Literals

    def _get_textarea_value(self, widget: TextArea, type_info: dict) -> Any:
        """Get value from text area widget."""
        text = widget.text.strip()
        actual_type = type_info['actual_type']
        is_optional = type_info['is_optional']
        origin_type = type_info['origin_type']

        if not text and is_optional:
            return None

        if actual_type is int:
            return int(text)
        elif actual_type is float:
            return float(text)
        elif actual_type is str:
            return text
        elif origin_type is list or origin_type is List:
            return self._parse_list_value(text)
        else:
            return text

    def _parse_list_value(self, text: str) -> list:
        """Parse list value from text."""
        try:
            if text.startswith('[') and text.endswith(']'):
                return eval(text)
            return [item.strip() for item in text.split(',') if item.strip()] if text else []
        except Exception as e:
            raise ValueError(f"Invalid list format: {e}")

    async def _save_settings(self):
        logger.info("GlobalSettingsEditorDialog: Attempting to save settings.")
        self.error_label.text = ""
        
        # Create a temporary config to validate changes against
        # This is crucial for Pydantic models which validate on assignment
        temp_validation_config = self.original_config.model_copy(deep=True)

        try:
            for path, widget in self.input_widgets.items():
                field_type = self._get_field_type_by_path(path)
                if field_type is None: continue
                
                value = self._get_field_value_from_widget(widget, field_type)
                
                # Apply value to the temp_validation_config
                if not self._set_config_value(temp_validation_config, path.split('.'), value):
                    # _set_config_value logs specifics, raise here to populate dialog error
                    raise ValueError(f"Failed to apply value for '{path}'. Check logs.")

            # If all values applied successfully to temp_validation_config without Pydantic errors,
            # it means the types are correct. Now, commit to editing_config.
            self.editing_config = temp_validation_config.model_copy(deep=True)
            self.original_config = self.editing_config.model_copy(deep=True) # Persist changes to original
            
            await self.state.notify('global_config_changed', self.original_config)
            self.save_button.disabled = True
            
            if self.dialog and hasattr(self.dialog, '__ohcs_future__'):
                future = getattr(self.dialog, '__ohcs_future__', None)
                if future and not future.done(): future.set_result(self.original_config)
            get_app().invalidate()

        except ValueError as e: # Catch type conversion errors from _get_field_value_from_widget
            logger.error(f"Validation/Conversion error: {e}")
            self.error_label.text = f"Error: {e}"
            get_app().invalidate()
        except Exception as e: # Catch Pydantic validation errors or other issues from _set_config_value
            logger.error(f"Error applying settings: {e}", exc_info=True)
            self.error_label.text = f"Error: {e}"
            get_app().invalidate()

    async def _cancel(self):
        logger.info("GlobalSettingsEditorDialog: Cancelled.")
        if self.dialog and hasattr(self.dialog, '__ohcs_future__'):
            future = getattr(self.dialog, '__ohcs_future__', None)
            if future and not future.done():
                future.set_result(None)

    async def show(self) -> Optional[GlobalPipelineConfig]:
        if not self.dialog: self._build_dialog()
        app = get_app()
        previous_focus = app.layout.current_window if hasattr(app.layout, 'current_window') else None
        
        # Ensure dialog width and height are Dimension objects if they are dynamic
        dialog_width = self.dialog.width if isinstance(self.dialog.width, Dimension) else Dimension(preferred=self.dialog.width)
        dialog_height = self.dialog.height if isinstance(self.dialog.height, Dimension) else Dimension(preferred=self.dialog.height)

        float_ = Float(content=self.dialog, width=dialog_width, height=dialog_height)
        setattr(self.dialog, '__ohcs_float__', float_) 
        
        future = asyncio.Future()
        setattr(self.dialog, '__ohcs_future__', future)

        app.layout.container.floats.append(float_)
        app.layout.focus(self.dialog)
        
        try:
            return await future
        finally:
            if float_ in app.layout.container.floats: app.layout.container.floats.remove(float_)
            if previous_focus and hasattr(app.layout, 'walk'):
                try:
                    is_still_in_layout = any(elem == previous_focus for elem in app.layout.walk(skip_hidden=True))
                    if is_still_in_layout: app.layout.focus(previous_focus)
                    else: app.layout.focus_last() 
                except Exception as e:
                    logger.warning(f"Could not restore focus: {e}")
                    try: app.layout.focus_last()
                    except Exception: pass
            elif hasattr(app.layout, 'focus_last'):
                 app.layout.focus_last()