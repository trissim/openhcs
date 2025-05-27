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
        
        self.save_button = Button("Save", handler=self._save_settings)
        self.save_button.disabled = True # Initially disabled
        self.cancel_button = Button("Cancel", handler=self._cancel)
        
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
        rows: List[Any] = []
        
        field_definitions = {}
        if hasattr(model_type, 'model_fields'): # Pydantic
            field_definitions = model_type.model_fields
        elif dataclasses.is_dataclass(model_type): # Standard dataclasses
            field_definitions = {f.name: f for f in dataclasses.fields(model_type)}
        else:
            return [] # Not a supported model type

        for field_name, field_obj in field_definitions.items():
            field_path = f"{prefix}{field_name}"
            field_label_text = field_name.replace('_', ' ').title()
            
            field_type_hint = field_obj.annotation if hasattr(field_obj, 'annotation') else field_obj.type
            current_value = getattr(model_instance, field_name, None)
            widget: Any = None

            actual_type = field_type_hint
            is_optional = get_origin(field_type_hint) is Union and type(None) in get_args(field_type_hint)
            if is_optional:
                actual_type = next((t for t in get_args(field_type_hint) if t is not type(None)), actual_type)
            
            origin_type = get_origin(actual_type)

            # Handle nested models (recursive call)
            if hasattr(actual_type, 'model_fields') or dataclasses.is_dataclass(actual_type):
                rows.append(VSplit([Label(f"{field_label_text} Settings:", style="class:dialog.section-title")], height=1, padding=Dimension(left=len(prefix.split('.'))-1 if prefix else 0)))
                nested_prefix = f"{field_path}."
                # Ensure current_value (the nested model instance) is not None
                if current_value is None and not is_optional: # Should not happen if config is well-defined
                     logger.error(f"Nested model {field_name} is None but not Optional. Cannot build UI.")
                     rows.append(Label(f"Error: {field_label_text} is None", style="class:error-text"))
                     continue
                elif current_value is not None: # Only recurse if nested object exists
                    rows.extend(self._build_dialog_rows_for_model(current_value, actual_type, nested_prefix))
                elif is_optional: # If it's Optional and None, just show a label
                    rows.append(Label(f"{field_label_text}: (Not set)", style="class:italic"))
                continue

            elif actual_type is bool:
                widget = Checkbox(checked=bool(current_value))
                # Mouse handler for Checkbox to trigger change detection
                original_mouse_handler = widget.control.mouse_handler
                def create_checkbox_mouse_handler(cb_widget_instance):
                    def new_mouse_handler(mouse_event):
                        res = original_mouse_handler(mouse_event)
                        self._on_setting_changed() # Generic change trigger
                        return res
                    return new_mouse_handler
                widget.control.mouse_handler = create_checkbox_mouse_handler(widget)

            elif origin_type is list or origin_type is List: # Basic list handling
                widget = TextArea(text=str(current_value or []), multiline=True, height=3, style="class:input-field")
                widget.buffer.on_text_changed += lambda b: self._on_setting_changed()

            elif origin_type is Literal:
                options = [(val, str(val)) for val in get_args(actual_type)]
                widget = RadioList(values=options, current_value=current_value)
                widget.on_value_changed += lambda w_val: self._on_setting_changed()
            
            elif isinstance(actual_type, type) and issubclass(actual_type, Enum):
                options = [(member.value, member.name) for name, member in actual_type.__members__.items()]
                enum_current_value = current_value.value if current_value else None
                if is_optional and current_value is None: # Handle Optional[Enum] being None
                    options = [ (None, "(Not set)") ] + options # Add a "Not set" option
                widget = RadioList(values=options, current_value=enum_current_value)
                widget.on_value_changed += lambda w_val: self._on_setting_changed()
            
            elif actual_type in [int, str, float]:
                widget = TextArea(text=str(current_value) if current_value is not None else "", multiline=False, height=1, style="class:input-field")
                widget.buffer.on_text_changed += lambda b: self._on_setting_changed()
            
            else:
                widget = Label(text=f"{str(current_value)} (Unsupported type: {actual_type})")

            if widget:
                self.input_widgets[field_path] = widget
                # Use Box for consistent padding and alignment
                rows.append(Box(HSplit([Label(f"{field_label_text}:", width=30), widget]), padding=Dimension(left=len(prefix.split('.'))-1 if prefix else 0)))
        return rows

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
        actual_type = field_type_hint
        is_optional = get_origin(field_type_hint) is Union and type(None) in get_args(field_type_hint)
        if is_optional:
            actual_type = next((t for t in get_args(field_type_hint) if t is not type(None)), actual_type)
        
        origin_type = get_origin(actual_type)

        if isinstance(widget, Checkbox):
            return widget.checked
        elif isinstance(widget, RadioList):
            current_radio_value = widget.current_value
            if is_optional and current_radio_value is None: # Explicit "Not set" for Optional Enums
                return None
            if isinstance(actual_type, type) and issubclass(actual_type, Enum):
                return actual_type(current_radio_value) if current_radio_value is not None else None
            return current_radio_value # For Literals
        elif isinstance(widget, TextArea):
            text = widget.text.strip()
            if not text and is_optional: return None
            if actual_type is int: return int(text)
            if actual_type is float: return float(text)
            if actual_type is str: return text
            if origin_type is list or origin_type is List:
                try:
                    if text.startswith('[') and text.endswith(']'): return eval(text)
                    return [item.strip() for item in text.split(',') if item.strip()] if text else []
                except Exception as e: raise ValueError(f"Invalid list format: {e}")
            return text
        raise ValueError(f"Unsupported widget type for value retrieval: {type(widget)}")

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