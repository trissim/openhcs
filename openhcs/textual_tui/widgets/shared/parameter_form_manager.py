# File: openhcs/textual_tui/widgets/shared/parameter_form_manager.py

import dataclasses
import ast
import logging
from enum import Enum
from typing import Any, Dict, get_origin, get_args, Union, Optional, Type

logger = logging.getLogger(__name__)
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Collapsible
from textual.app import ComposeResult

from .typed_widget_factory import TypedWidgetFactory
from .signature_analyzer import SignatureAnalyzer
from .clickable_help_label import ClickableParameterLabel, HelpIndicator
from ..different_values_input import DifferentValuesInput

# Import simplified abstraction layer
from openhcs.ui.shared.parameter_form_abstraction import (
    ParameterFormAbstraction, apply_lazy_default_placeholder
)
from openhcs.ui.shared.widget_creation_registry import create_textual_registry
from openhcs.ui.shared.textual_widget_strategies import create_different_values_widget

class ParameterFormManager:
    """Mathematical: (parameters, types, field_id) → parameter form"""

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type], field_id: str, parameter_info: Dict = None, is_global_config_editing: bool = False, global_config_type: Optional[Type] = None, placeholder_prefix: str = "Pipeline default"):
        # Initialize simplified abstraction layer
        self.form_abstraction = ParameterFormAbstraction(
            parameters, parameter_types, field_id, create_textual_registry(), parameter_info
        )

        # Maintain backward compatibility
        self.parameters = parameters.copy()
        self.parameter_types = parameter_types
        self.field_id = field_id
        self.parameter_info = parameter_info or {}
        self.is_global_config_editing = is_global_config_editing
        self.global_config_type = global_config_type
        self.placeholder_prefix = placeholder_prefix
    
    def build_form(self) -> ComposeResult:
        """Build parameter form - pure function with recursive dataclass support."""
        with Vertical() as form:
            form.styles.height = "auto"

            for param_name, param_type in self.parameter_types.items():
                current_value = self.parameters[param_name]

                # Handle Optional[dataclass] types with checkbox wrapper
                if self._is_optional_dataclass(param_type):
                    inner_dataclass_type = self._get_optional_inner_type(param_type)
                    yield from self._build_optional_dataclass_form(param_name, inner_dataclass_type, current_value)
                # Handle nested dataclasses recursively
                elif dataclasses.is_dataclass(param_type):
                    yield from self._build_nested_dataclass_form(param_name, param_type, current_value)
                else:
                    yield from self._build_regular_parameter_form(param_name, param_type, current_value)

    def _build_nested_dataclass_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for nested dataclass parameter."""
        # Create collapsible widget (no ID - just structure)
        collapsible = TypedWidgetFactory.create_widget(param_type, current_value, None)

        # Analyze nested dataclass
        nested_param_info = SignatureAnalyzer.analyze(param_type)

        # Get current values from nested dataclass instance
        nested_parameters = {}
        nested_parameter_types = {}

        for nested_name, nested_info in nested_param_info.items():
            if current_value:
                # For lazy dataclasses, preserve None values for placeholder behavior
                if hasattr(current_value, '_resolve_field_value'):
                    nested_current_value = object.__getattribute__(current_value, nested_name) if hasattr(current_value, nested_name) else nested_info.default_value
                else:
                    nested_current_value = getattr(current_value, nested_name, nested_info.default_value)
            else:
                nested_current_value = nested_info.default_value
            nested_parameters[nested_name] = nested_current_value
            nested_parameter_types[nested_name] = nested_info.param_type

        # Create nested form manager with hierarchical underscore notation and parameter info
        nested_field_id = f"{self.field_id}_{param_name}"
        nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id, nested_param_info)

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = param_type

        # Store reference to nested form manager for updates
        if not hasattr(self, 'nested_managers'):
            self.nested_managers = {}
        self.nested_managers[param_name] = nested_form_manager

        # Build nested form and add to collapsible
        with collapsible:
            yield from nested_form_manager.build_form()

        yield collapsible

    def _is_optional_dataclass(self, param_type: type) -> bool:
        """Check if parameter type is Optional[dataclass]."""
        from typing import get_origin, get_args, Union
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                inner_type = next(arg for arg in args if arg is not type(None))
                return dataclasses.is_dataclass(inner_type)
        return False

    def _get_optional_inner_type(self, param_type: type) -> type:
        """Extract the inner type from Optional[T]."""
        from typing import get_origin, get_args, Union
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                return next(arg for arg in args if arg is not type(None))
        return param_type

    def _build_optional_dataclass_form(self, param_name: str, dataclass_type: type, current_value: Any) -> ComposeResult:
        """Build form for Optional[dataclass] parameter with checkbox toggle."""
        from textual.widgets import Checkbox

        # Checkbox
        checkbox_id = f"{self.field_id}_{param_name}_enabled"
        checkbox = Checkbox(
            value=current_value is not None,
            label=f"Enable {param_name.replace('_', ' ').title()}",
            id=checkbox_id,
            compact=True
        )
        yield checkbox

        # Collapsible dataclass widget
        collapsible = TypedWidgetFactory.create_widget(dataclass_type, current_value, None)
        collapsible.collapsed = (current_value is None)

        # Setup nested form
        nested_param_info = SignatureAnalyzer.analyze(dataclass_type)
        nested_parameters = {}
        for name, info in nested_param_info.items():
            if current_value:
                # For lazy dataclasses, preserve None values for placeholder behavior
                if hasattr(current_value, '_resolve_field_value'):
                    value = object.__getattribute__(current_value, name) if hasattr(current_value, name) else info.default_value
                else:
                    value = getattr(current_value, name, info.default_value)
            else:
                value = info.default_value
            nested_parameters[name] = value
        nested_parameter_types = {name: info.param_type for name, info in nested_param_info.items()}

        nested_form_manager = ParameterFormManager(
            nested_parameters, nested_parameter_types, f"{self.field_id}_{param_name}", nested_param_info,
            is_global_config_editing=self.is_global_config_editing
        )

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = dataclass_type

        # Store references
        if not hasattr(self, 'nested_managers'):
            self.nested_managers = {}
        if not hasattr(self, 'optional_checkboxes'):
            self.optional_checkboxes = {}
        self.nested_managers[param_name] = nested_form_manager
        self.optional_checkboxes[param_name] = checkbox

        with collapsible:
            yield from nested_form_manager.build_form()
        yield collapsible

    def _build_regular_parameter_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for regular (non-dataclass) parameter."""
        # Check if this field has different values across orchestrators
        config_analysis = getattr(self, 'config_analysis', {})
        field_analysis = config_analysis.get(param_name, {})

        # Create widget using hierarchical underscore notation
        widget_id = f"{self.field_id}_{param_name}"

        # Handle different values or create normal widget
        if field_analysis.get('type') == 'different':
            default_value = field_analysis.get('default')
            input_widget = create_different_values_widget(param_name, param_type, default_value, widget_id)
        else:
            # Use registry for widget creation and apply placeholder
            widget_value = current_value.value if hasattr(current_value, 'value') else current_value
            input_widget = self.form_abstraction.create_widget_for_parameter(param_name, param_type, widget_value)
            apply_lazy_default_placeholder(input_widget, param_name, current_value, self.parameter_types, 'textual',
                                          is_global_config_editing=self.is_global_config_editing,
                                          global_config_type=self.global_config_type,
                                          placeholder_prefix=self.placeholder_prefix)

        # Get parameter info for help functionality
        param_info = self._get_parameter_info(param_name)
        param_description = param_info.description if param_info else None

        # 3-column layout: label + input + reset
        with Horizontal() as row:
            row.styles.height = "auto"

            # Parameter label with help (auto width - sizes to content)
            # Always use clickable label with help - provide default description if none exists
            description = param_description or f"Parameter: {param_name.replace('_', ' ')}"
            label = ClickableParameterLabel(
                param_name,
                description,
                param_type,
                classes="param-label clickable"
            )

            label.styles.width = "auto"
            label.styles.text_align = "left"
            label.styles.height = "1"
            yield label

            # Input widget (flexible width, left aligned, with left margin for spacing)
            input_widget.styles.width = "1fr"
            input_widget.styles.text_align = "left"
            input_widget.styles.margin = (0, 0, 0, 1)  # top, right, bottom, left margin
            yield input_widget

            # Reset button (auto width)
            reset_btn = Button("Reset", id=f"reset_{widget_id}", compact=True)
            reset_btn.styles.width = "auto"
            yield reset_btn
    
    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value with centralized enum conversion and nested dataclass support."""
        # Debug: Check if None values are being received and processed (path_planning only)
        if param_name == 'output_dir_suffix' or param_name == 'path_planning':
            logger.info(f"*** TEXTUAL UPDATE DEBUG *** {param_name} update_parameter called with: {value} (type: {type(value)})")
            if param_name == 'path_planning':
                import traceback
                logger.info(f"*** PATH_PLANNING SOURCE *** Call stack:")
                for line in traceback.format_stack()[-5:]:
                    logger.info(f"*** PATH_PLANNING SOURCE *** {line.strip()}")
        # Parse hierarchical parameter name (e.g., "path_planning_global_output_folder")
        # Split and check if this is a nested parameter
        parts = param_name.split('_')
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = '_'.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name from remaining parts
                    nested_field = '_'.join(parts[i:])

                    # Update nested form manager
                    if potential_nested == 'path_planning':
                        logger.info(f"*** NESTED MANAGER UPDATE *** Updating {potential_nested}.{nested_field} = {value}")
                    self.nested_managers[potential_nested].update_parameter(nested_field, value)

                    # Rebuild nested dataclass instance with lazy/concrete mixed behavior
                    nested_values = self.nested_managers[potential_nested].get_current_values()

                    # Debug: Check what values the nested manager is returning
                    if potential_nested == 'path_planning':
                        logger.info(f"*** NESTED VALUES DEBUG *** nested_values from {potential_nested}: {nested_values}")
                        if 'output_dir_suffix' in nested_values:
                            logger.info(f"*** NESTED VALUES DEBUG *** output_dir_suffix in nested_values: {nested_values['output_dir_suffix']} (type: {type(nested_values['output_dir_suffix'])})")

                        # Also check what's in the nested manager's parameters directly
                        nested_params = self.nested_managers[potential_nested].parameters
                        logger.info(f"*** NESTED VALUES DEBUG *** nested_manager.parameters: {nested_params}")
                        if 'output_dir_suffix' in nested_params:
                            logger.info(f"*** NESTED VALUES DEBUG *** output_dir_suffix in nested_manager.parameters: {nested_params['output_dir_suffix']} (type: {type(nested_params['output_dir_suffix'])})")

                    nested_type = self.parameter_types[potential_nested]

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(nested_type):
                        nested_type = self._get_optional_inner_type(nested_type)

                    # Create lazy dataclass instance with mixed concrete/lazy fields
                    if self.is_global_config_editing:
                        # Global config editing: use concrete dataclass
                        self.parameters[potential_nested] = nested_type(**nested_values)
                    else:
                        # Lazy context: always create lazy instance for thread-local resolution
                        # Even if all values are None (especially after reset), we want lazy resolution
                        from openhcs.core.lazy_config import LazyDataclassFactory

                        # Determine the correct field path using type inspection
                        field_path = self._get_field_path_for_nested_type(nested_type)

                        lazy_nested_type = LazyDataclassFactory.make_lazy_thread_local(
                            base_class=nested_type,
                            field_path=field_path,
                            lazy_class_name=f"Mixed{nested_type.__name__}"
                        )
                        # Pass ALL fields: concrete values for edited fields, None for lazy resolution
                        self.parameters[potential_nested] = lazy_nested_type(**nested_values)
                    return

        # Handle regular parameters (direct match)
        if param_name in self.parameters:
            # Handle literal "None" string - convert back to Python None
            if isinstance(value, str) and value == "None":
                value = None

            # Convert string back to proper type (comprehensive conversion)
            # Skip type conversion for None values (preserve for lazy placeholder behavior)
            if param_name in self.parameter_types and value is not None:
                param_type = self.parameter_types[param_name]
                if hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                    value = param_type(value)  # Convert string → enum
                elif self._is_list_of_enums(param_type):
                    # Handle List[Enum] types (like List[VariableComponents])
                    enum_type = self._get_enum_from_list(param_type)
                    if enum_type:
                        # Convert string value to enum, then wrap in list
                        enum_value = enum_type(value)
                        value = [enum_value]
                elif param_type == float:
                    # Convert string → float, handle empty based on parameter requirements
                    try:
                        if value == "":
                            # For empty values, we need to check if parameter is required
                            # This requires access to parameter info, but we don't have it here
                            # For now, convert empty to None (safer than 0.0)
                            value = None
                        else:
                            value = float(value)
                    except (ValueError, TypeError):
                        value = None  # Use None instead of 0.0 for failed conversions
                elif param_type == int:
                    # Convert string → int, handle empty based on parameter requirements
                    try:
                        if value == "":
                            # For empty values, convert to None (safer than 0)
                            value = None
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        value = None  # Use None instead of 0 for failed conversions
                elif param_type == bool:
                    # Convert string → bool
                    if isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                # Add more type conversions as needed

            self.parameters[param_name] = value

            # FALLBACK: If this is a nested field that bypassed the nested logic, update the nested manager
            if param_name == 'output_dir_suffix':
                logger.info(f"*** FALLBACK DEBUG *** Checking fallback for {param_name}")
                logger.info(f"*** FALLBACK DEBUG *** hasattr nested_managers: {hasattr(self, 'nested_managers')}")
                if hasattr(self, 'nested_managers'):
                    logger.info(f"*** FALLBACK DEBUG *** nested_managers keys: {list(self.nested_managers.keys())}")
                    for nested_name, nested_manager in self.nested_managers.items():
                        logger.info(f"*** FALLBACK DEBUG *** Checking {nested_name}, parameter_types: {list(nested_manager.parameter_types.keys())}")
                        if param_name in nested_manager.parameter_types:
                            logger.info(f"*** FALLBACK UPDATE *** Updating nested manager {nested_name}.{param_name} = {value}")
                            nested_manager.parameters[param_name] = value
                            break
                        else:
                            logger.info(f"*** FALLBACK DEBUG *** {param_name} not found in {nested_name}")
                else:
                    logger.info(f"*** FALLBACK DEBUG *** No nested_managers attribute")
            elif hasattr(self, 'nested_managers'):
                for nested_name, nested_manager in self.nested_managers.items():
                    if param_name in nested_manager.parameter_types:
                        nested_manager.parameters[param_name] = value
                        break

            # Debug: Check what was actually stored (path_planning only)
            if param_name == 'output_dir_suffix' or param_name == 'path_planning':
                stored_value = self.parameters.get(param_name)
                logger.info(f"*** TEXTUAL UPDATE DEBUG *** {param_name} stored as: {stored_value} (type: {type(stored_value)})")
    
    def reset_parameter(self, param_name: str, default_value: Any = None):
        """Reset parameter to appropriate default value based on lazy vs concrete dataclass context."""
        # Determine the correct reset value if not provided
        if default_value is None:
            default_value = self._get_reset_value_for_parameter(param_name)

        # Parse hierarchical parameter name for nested parameters
        parts = param_name.split('_')
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = '_'.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name
                    nested_field = '_'.join(parts[i:])

                    # Get appropriate reset value for nested field
                    nested_reset_value = self._get_reset_value_for_nested_parameter(potential_nested, nested_field)

                    # Reset in nested form manager
                    self.nested_managers[potential_nested].reset_parameter(nested_field, nested_reset_value)

                    # Rebuild nested dataclass instance
                    nested_values = self.nested_managers[potential_nested].get_current_values()

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(self.parameter_types[potential_nested]):
                        nested_type = self._get_optional_inner_type(self.parameter_types[potential_nested])
                    else:
                        nested_type = self.parameter_types[potential_nested]

                    # Create lazy dataclass instance with mixed concrete/lazy fields
                    if self.is_global_config_editing:
                        # Global config editing: use concrete dataclass
                        self.parameters[potential_nested] = nested_type(**nested_values)
                    else:
                        # Lazy context: always create lazy instance for thread-local resolution
                        # Even if all values are None (especially after reset), we want lazy resolution
                        from openhcs.core.lazy_config import LazyDataclassFactory

                        # Determine the correct field path using type inspection
                        field_path = self._get_field_path_for_nested_type(nested_type)

                        lazy_nested_type = LazyDataclassFactory.make_lazy_thread_local(
                            base_class=nested_type,
                            field_path=field_path,
                            lazy_class_name=f"Mixed{nested_type.__name__}"
                        )
                        # Pass ALL fields: concrete values for edited fields, None for lazy resolution
                        self.parameters[potential_nested] = lazy_nested_type(**nested_values)
                    return

        # Handle regular parameters
        if param_name in self.parameters:
            self.parameters[param_name] = default_value

            # Handle special reset behavior for DifferentValuesInput widgets
            self._handle_different_values_reset(param_name)

            # Re-apply placeholder styling if value is None (for reset functionality)
            if default_value is None:
                self._reapply_placeholder_if_needed(param_name)

    def _reapply_placeholder_if_needed(self, param_name: str):
        """Re-apply placeholder styling to a widget when its value is set to None."""
        # For Textual, we need to find the widget and re-apply placeholder
        # This is more complex than PyQt since Textual widgets are reactive
        # For now, we'll rely on the reactive nature of Textual widgets
        # The placeholder should be re-applied automatically when the value changes to None
        pass

    def _get_reset_value_for_parameter(self, param_name: str) -> Any:
        """
        Get the appropriate reset value for a parameter based on lazy vs concrete dataclass context.

        For concrete dataclasses (like GlobalPipelineConfig):
        - Reset to static class defaults

        For lazy dataclasses (like PipelineConfig for orchestrator configs):
        - Reset to None to preserve placeholder behavior and inheritance hierarchy
        """
        if param_name not in self.parameter_info:
            return None

        param_info = self.parameter_info[param_name]
        param_type = self.parameter_types[param_name]

        # For global config editing, always use static defaults
        if self.is_global_config_editing:
            return param_info.default_value

        # For nested dataclass fields, check if we should use concrete values
        if hasattr(param_type, '__dataclass_fields__'):
            # This is a dataclass field - determine if it should be concrete or None
            current_value = self.parameters.get(param_name)
            if self._should_use_concrete_nested_values(current_value):
                # Use static default for concrete nested dataclass
                return param_info.default_value
            else:
                # Use None for lazy nested dataclass to preserve placeholder behavior
                return None

        # For non-dataclass fields in lazy context, use None to preserve placeholder behavior
        # This allows the field to inherit from the parent config hierarchy
        if not self.is_global_config_editing:
            return None

        # Fallback to static default
        return param_info.default_value

    def _get_reset_value_for_nested_parameter(self, nested_param_name: str, nested_field_name: str) -> Any:
        """Get appropriate reset value for a nested parameter field."""
        nested_type = self.parameter_types[nested_param_name]
        nested_param_info = SignatureAnalyzer.analyze(nested_type)

        if nested_field_name not in nested_param_info:
            return None

        nested_field_info = nested_param_info[nested_field_name]

        # For global config editing, always use static defaults
        if self.is_global_config_editing:
            return nested_field_info.default_value

        # For lazy context, check if nested dataclass should use concrete values
        current_nested_value = self.parameters.get(nested_param_name)
        if self._should_use_concrete_nested_values(current_nested_value):
            return nested_field_info.default_value
        else:
            return None

    def _get_field_path_for_nested_type(self, nested_type: Type) -> Optional[str]:
        """
        Automatically determine the field path for a nested dataclass type using type inspection.

        This method examines the GlobalPipelineConfig fields and their type annotations
        to find which field corresponds to the given nested_type. This eliminates the need
        for hardcoded string mappings and automatically works with new nested dataclass fields.

        Args:
            nested_type: The dataclass type to find the field path for

        Returns:
            The field path string (e.g., 'path_planning', 'vfs') or None if not found
        """
        try:
            from openhcs.core.config import GlobalPipelineConfig
            from dataclasses import fields
            import typing

            # Get all fields from GlobalPipelineConfig
            global_config_fields = fields(GlobalPipelineConfig)

            for field in global_config_fields:
                field_type = field.type

                # Handle Optional types (Union[Type, None])
                if hasattr(typing, 'get_origin') and typing.get_origin(field_type) is typing.Union:
                    # Get the non-None type from Optional[Type]
                    args = typing.get_args(field_type)
                    if len(args) == 2 and type(None) in args:
                        field_type = args[0] if args[1] is type(None) else args[1]

                # Check if the field type matches our nested type
                if field_type == nested_type:
                    return field.name



            return None

        except Exception as e:
            # Fallback to None if type inspection fails
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to determine field path for {nested_type.__name__}: {e}")
            return None

    def _should_use_concrete_nested_values(self, current_value: Any) -> bool:
        """
        Determine if nested dataclass fields should use concrete values or None for placeholders.
        This mirrors the logic from the PyQt form manager.

        Returns True if:
        1. Global config editing (always concrete)
        2. Regular concrete dataclass (always concrete)

        Returns False if:
        1. Lazy dataclass (supports mixed lazy/concrete states per field)
        2. None values (show placeholders)

        Note: This method now supports mixed states within nested dataclasses.
        Individual fields can be lazy (None) or concrete within the same dataclass.
        """
        # Global config editing always uses concrete values
        if self.is_global_config_editing:
            return True

        # If current_value is None, use placeholders
        if current_value is None:
            return False

        # If current_value is a concrete dataclass instance, use its values
        if hasattr(current_value, '__dataclass_fields__') and not hasattr(current_value, '_resolve_field_value'):
            return True

        # For lazy dataclasses, always return False to enable mixed lazy/concrete behavior
        # Individual field values will be checked separately in the nested form creation
        if hasattr(current_value, '_resolve_field_value'):
            return False

        # Default to placeholder behavior for lazy contexts
        return False

    def handle_optional_checkbox_change(self, param_name: str, enabled: bool):
        """Handle checkbox change for Optional[dataclass] parameters."""
        if param_name in self.parameter_types and self._is_optional_dataclass(self.parameter_types[param_name]):
            dataclass_type = self._get_optional_inner_type(self.parameter_types[param_name])
            nested_managers = getattr(self, 'nested_managers', {})
            self.parameters[param_name] = (
                dataclass_type(**nested_managers[param_name].get_current_values())
                if enabled and param_name in nested_managers
                else dataclass_type() if enabled
                else None
            )

    def _handle_different_values_reset(self, param_name: str):
        """Handle reset behavior for DifferentValuesInput widgets."""
        # Check if this field has different values across orchestrators
        config_analysis = getattr(self, 'config_analysis', {})
        field_analysis = config_analysis.get(param_name, {})

        if field_analysis.get('type') == 'different':
            # For different values fields, reset means go back to "DIFFERENT VALUES" state
            # We need to find the widget and call its reset method
            widget_id = f"{self.field_id}_{param_name}"

            # This will be handled by the screen/container that manages the widgets
            # The widget itself will handle the reset via its reset_to_different() method
            # We just need to ensure the parameter value reflects the "different" state
            pass  # Widget-level reset will be handled by the containing screen

    def reset_all_parameters(self, defaults: Dict[str, Any] = None):
        """Reset all parameters to appropriate defaults based on lazy vs concrete dataclass context."""
        # If no defaults provided, generate them based on context
        if defaults is None:
            defaults = {}
            for param_name in self.parameters.keys():
                defaults[param_name] = self._get_reset_value_for_parameter(param_name)

        for param_name, default_value in defaults.items():
            if param_name in self.parameters:
                # Handle nested dataclasses
                if dataclasses.is_dataclass(self.parameter_types.get(param_name)):
                    if hasattr(self, 'nested_managers') and param_name in self.nested_managers:
                        # Generate appropriate reset values for nested parameters
                        nested_type = self.parameter_types[param_name]
                        nested_param_info = SignatureAnalyzer.analyze(nested_type)

                        # Use lazy-aware reset logic for nested parameters with mixed state support
                        nested_defaults = {}
                        for nested_field_name in nested_param_info.keys():
                            # For nested fields in lazy contexts, always reset to None to preserve lazy behavior
                            # This ensures individual fields can maintain placeholder behavior regardless of other field states
                            if not self.is_global_config_editing:
                                nested_defaults[nested_field_name] = None
                            else:
                                nested_defaults[nested_field_name] = self._get_reset_value_for_nested_parameter(param_name, nested_field_name)

                        self.nested_managers[param_name].reset_all_parameters(nested_defaults)

                        # Rebuild nested dataclass instance
                        nested_values = self.nested_managers[param_name].get_current_values()

                        # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                        if self._is_optional_dataclass(nested_type):
                            nested_type = self._get_optional_inner_type(nested_type)

                        # Create lazy dataclass instance with mixed concrete/lazy fields
                        if self.is_global_config_editing:
                            # Global config editing: use concrete dataclass
                            self.parameters[param_name] = nested_type(**nested_values)
                        else:
                            # Lazy context: always create lazy instance for thread-local resolution
                            # Even if all values are None (especially after reset), we want lazy resolution
                            from openhcs.core.lazy_config import LazyDataclassFactory

                            # Determine the correct field path using type inspection
                            field_path = self._get_field_path_for_nested_type(nested_type)

                            lazy_nested_type = LazyDataclassFactory.make_lazy_thread_local(
                                base_class=nested_type,
                                field_path=field_path,
                                lazy_class_name=f"Mixed{nested_type.__name__}"
                            )
                            # Pass ALL fields: concrete values for edited fields, None for lazy resolution
                            self.parameters[param_name] = lazy_nested_type(**nested_values)
                    else:
                        self.parameters[param_name] = default_value
                else:
                    self.parameters[param_name] = default_value

    def reset_parameter_by_path(self, parameter_path: str):
        """Reset a parameter by its full path (supports nested parameters).

        Args:
            parameter_path: Either a simple parameter name (e.g., 'num_workers')
                          or a nested path (e.g., 'path_planning.output_dir_suffix')
        """
        if '.' in parameter_path:
            # Handle nested parameter
            parts = parameter_path.split('.', 1)
            nested_name = parts[0]
            nested_param = parts[1]

            if hasattr(self, 'nested_managers') and nested_name in self.nested_managers:
                nested_manager = self.nested_managers[nested_name]
                if '.' in nested_param:
                    # Further nesting
                    nested_manager.reset_parameter_by_path(nested_param)
                else:
                    # Direct nested parameter
                    nested_manager.reset_parameter(nested_param)
            else:
                logger.warning(f"Nested manager '{nested_name}' not found for parameter path '{parameter_path}'")
        else:
            # Handle top-level parameter
            self.reset_parameter(parameter_path)

    def _is_list_of_enums(self, param_type) -> bool:
        """Check if parameter type is List[Enum]."""
        try:
            # Check if it's a generic type (like List[Something])
            origin = get_origin(param_type)
            if origin is list:
                # Get the type arguments (e.g., VariableComponents from List[VariableComponents])
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    # Check if the inner type is an enum
                    return hasattr(inner_type, '__bases__') and Enum in inner_type.__bases__
            return False
        except Exception:
            return False

    def _get_enum_from_list(self, param_type):
        """Extract enum type from List[Enum] type."""
        return self._get_enum_from_list_static(param_type)

    @staticmethod
    def _is_list_of_enums_static(param_type) -> bool:
        """Static version of _is_list_of_enums for use in convert_string_to_type."""
        try:
            # Check if it's a generic type (like List[Something])
            origin = get_origin(param_type)
            if origin is list:
                # Get the type arguments (e.g., VariableComponents from List[VariableComponents])
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    # Check if the inner type is an enum
                    return hasattr(inner_type, '__bases__') and Enum in inner_type.__bases__
            return False
        except Exception:
            return False

    @staticmethod
    def _get_enum_from_list_static(param_type):
        """Static version of _get_enum_from_list for use in convert_string_to_type."""
        try:
            args = get_args(param_type)
            if args and len(args) > 0:
                return args[0]  # Return the enum type (e.g., VariableComponents)
            return None
        except Exception:
            return None

    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self.parameters.copy()

    def _get_parameter_info(self, param_name: str):
        """Get parameter info for help functionality."""
        return self.parameter_info.get(param_name)

    # Old placeholder methods removed - now using centralized abstraction layer

    @staticmethod
    def convert_string_to_type(string_value: str, param_type: type, strict: bool = False) -> Any:
        """
        Convert string input to expected type using existing type conversion logic.

        Args:
            string_value: The string value from user input
            param_type: The expected type from function signature
            strict: If True, raise errors on conversion failure. If False, return None.

        Returns:
            Converted value of the expected type

        Raises:
            ValueError: If strict=True and conversion fails with specific error message
        """
        # Handle empty/None values - let compiler validate if required
        if string_value == "" or string_value is None:
            return None

        try:
            # Handle Union types (like Optional[List[float]] which is Union[List[float], None])
            origin = get_origin(param_type)
            if origin is Union:
                # Try each type in the Union until one works
                union_args = get_args(param_type)
                last_error = None

                for union_type in union_args:
                    # Skip NoneType - we handle None separately
                    if union_type is type(None):
                        continue

                    try:
                        # Recursively try to convert to this union member type
                        return ParameterFormManager.convert_string_to_type(string_value, union_type, strict=True)
                    except (ValueError, TypeError, SyntaxError) as e:
                        last_error = e
                        continue

                # If no union type worked, raise the last error
                if last_error:
                    raise last_error
                else:
                    raise ValueError(f"No valid conversion found for Union type {param_type}")

            # Use existing type conversion logic from update_parameter
            elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                return param_type(string_value)  # Convert string → enum
            elif ParameterFormManager._is_list_of_enums_static(param_type):
                # Handle List[Enum] types (like List[VariableComponents])
                enum_type = ParameterFormManager._get_enum_from_list_static(param_type)
                if enum_type:
                    # Convert string value to enum, then wrap in list
                    enum_value = enum_type(string_value)
                    return [enum_value]
            elif param_type == float:
                return float(string_value)
            elif param_type == int:
                return int(string_value)
            elif param_type == bool:
                # Convert string → bool
                return string_value.lower() in ('true', '1', 'yes', 'on')
            elif param_type in (list, tuple, dict):
                # Use ast.literal_eval for complex types like [1,2,3], (1,2), {"a":1}
                return ast.literal_eval(string_value)
            elif get_origin(param_type) in (list, tuple, dict):
                # Handle generic types like List[float], Tuple[int, int], Dict[str, int]
                # Use ast.literal_eval since List("[1]") doesn't work, but ast.literal_eval("[1]") does
                return ast.literal_eval(string_value)
            elif param_type is Any:
                # No type hints available - try ast.literal_eval for Python literals
                try:
                    return ast.literal_eval(string_value)
                except (ValueError, SyntaxError):
                    # If literal_eval fails, return as string
                    return string_value
            else:
                # For everything else, try calling the type directly
                return param_type(string_value)

        except (ValueError, TypeError, SyntaxError) as e:
            if strict:
                # Provide specific error message for user
                raise ValueError(f"Cannot convert '{string_value}' to {param_type.__name__}: {e}")
            else:
                # Silent failure - return None (existing behavior)
                return None

    def _create_nested_managers_for_testing(self):
        """Create nested managers without building widgets (for testing)."""
        for param_name, param_type in self.parameter_types.items():
            current_value = self.parameters[param_name]

            # Handle nested dataclasses
            if dataclasses.is_dataclass(param_type):
                # Analyze nested dataclass
                nested_param_info = SignatureAnalyzer.analyze(param_type)

                # Get current values from nested dataclass instance
                nested_parameters = {}
                nested_parameter_types = {}

                for nested_name, nested_info in nested_param_info.items():
                    if current_value:
                        # For lazy dataclasses, preserve None values for placeholder behavior
                        if hasattr(current_value, '_resolve_field_value'):
                            nested_current_value = object.__getattribute__(current_value, nested_name) if hasattr(current_value, nested_name) else nested_info.default_value
                        else:
                            nested_current_value = getattr(current_value, nested_name, nested_info.default_value)
                    else:
                        nested_current_value = nested_info.default_value
                    nested_parameters[nested_name] = nested_current_value
                    nested_parameter_types[nested_name] = nested_info.param_type

                # Create nested form manager with hierarchical underscore notation and parameter info
                nested_field_id = f"{self.field_id}_{param_name}"
                nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id, nested_param_info)

                # Store reference to nested form manager for updates
                if not hasattr(self, 'nested_managers'):
                    self.nested_managers = {}
                self.nested_managers[param_name] = nested_form_manager
