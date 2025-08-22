"""
Shared service layer for parameter form managers.

This module provides a framework-agnostic service layer that eliminates the
architectural dependency between PyQt and Textual implementations by providing
shared business logic and data management.
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Any, Type, Optional, List, Tuple

from openhcs.core.config import LazyDefaultPlaceholderService
from openhcs.ui.shared.parameter_form_constants import CONSTANTS
from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils
from openhcs.ui.shared.ui_utils import debug_param, format_field_id, format_param_name, format_reset_button_id


@dataclass
class ParameterInfo:
    """
    Information about a parameter for form generation.
    
    Attributes:
        name: Parameter name
        type: Parameter type
        current_value: Current parameter value
        default_value: Default parameter value
        description: Parameter description
        is_required: Whether the parameter is required
        is_nested: Whether the parameter is a nested dataclass
        is_optional: Whether the parameter is Optional[T]
    """
    name: str
    type: Type
    current_value: Any
    default_value: Any = None
    description: Optional[str] = None
    is_required: bool = True
    is_nested: bool = False
    is_optional: bool = False


@dataclass
class FormStructure:
    """
    Structure information for a parameter form.
    
    Attributes:
        field_id: Unique identifier for the form
        parameters: List of parameter information
        nested_forms: Dictionary of nested form structures
        has_optional_dataclasses: Whether form has optional dataclass parameters
    """
    field_id: str
    parameters: List[ParameterInfo]
    nested_forms: Dict[str, 'FormStructure']
    has_optional_dataclasses: bool = False


class ParameterFormService:
    """
    Framework-agnostic service for parameter form business logic.
    
    This service provides shared functionality for both PyQt and Textual
    parameter form managers, eliminating the need for cross-framework
    dependencies and providing a clean separation of concerns.
    """
    
    def __init__(self):
        """
        Initialize the parameter form service.
        """
        self._type_utils = ParameterTypeUtils()
    
    def analyze_parameters(self, parameters: Dict[str, Any], parameter_types: Dict[str, Type],
                          field_id: str, parameter_info: Optional[Dict] = None,
                          parent_dataclass_type: Optional[Type] = None) -> FormStructure:
        """
        Analyze parameters and create form structure.

        This method analyzes the parameters and their types to create a complete
        form structure that can be used by any UI framework.

        Args:
            parameters: Dictionary of parameter names to current values
            parameter_types: Dictionary of parameter names to types
            field_id: Unique identifier for the form
            parameter_info: Optional parameter information dictionary
            parent_dataclass_type: Optional parent dataclass type for context

        Returns:
            Complete form structure information
        """
        debug_param("analyze_parameters", f"field_id={field_id}, parameter_count={len(parameters)}")
        
        param_infos = []
        nested_forms = {}
        has_optional_dataclasses = False
        
        for param_name, param_type in parameter_types.items():
            current_value = parameters.get(param_name)
            
            # Create parameter info
            param_info = self._create_parameter_info(
                param_name, param_type, current_value, parameter_info
            )
            param_infos.append(param_info)
            
            # Check for nested dataclasses
            if param_info.is_nested:
                nested_field_id = f"nested_{format_field_id(field_id, param_name)}"
                nested_structure = self._analyze_nested_dataclass(
                    param_name, param_type, current_value, nested_field_id, parent_dataclass_type
                )
                nested_forms[param_name] = nested_structure
            
            # Check for optional dataclasses
            if param_info.is_optional and param_info.is_nested:
                has_optional_dataclasses = True
        
        return FormStructure(
            field_id=field_id,
            parameters=param_infos,
            nested_forms=nested_forms,
            has_optional_dataclasses=has_optional_dataclasses
        )
    
    def convert_value_to_type(self, value: Any, param_type: Type, param_name: str) -> Any:
        """
        Convert a value to the appropriate type for a parameter.
        
        This method provides centralized type conversion logic that can be
        used by any UI framework.
        
        Args:
            value: The value to convert
            param_type: The target parameter type
            param_name: The parameter name (for debugging)
            
        Returns:
            The converted value
        """
        debug_param("convert_value", f"param={param_name}, input_type={type(value).__name__}, target_type={param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}")
        
        if value is None:
            return None
        
        # Handle string "None" literal
        if isinstance(value, str) and value == CONSTANTS.NONE_STRING_LITERAL:
            return None
        
        # Handle enum types
        if self._type_utils.is_enum_type(param_type):
            return param_type(value)
        
        # Handle list of enums
        if self._type_utils.is_list_of_enums(param_type):
            # If value is already a list (from checkbox group widget), return as-is
            if isinstance(value, list):
                return value
            enum_type = self._type_utils.get_enum_from_list_type(param_type)
            if enum_type:
                return [enum_type(value)]
        
        # Handle basic types
        if param_type == bool and isinstance(value, str):
            return self._type_utils.convert_string_to_bool(value)
        if param_type in (int, float) and isinstance(value, str):
            if value == CONSTANTS.EMPTY_STRING:
                return None
            try:
                return param_type(value)
            except (ValueError, TypeError):
                return None

        # Handle empty strings in lazy context - convert to None for all parameter types
        # This is critical for lazy dataclass behavior where None triggers placeholder resolution
        if isinstance(value, str) and value == CONSTANTS.EMPTY_STRING:
            # Check if we're in a lazy context by examining if this is being called
            # during a reset operation or widget clearing in a lazy dataclass context
            # For now, always convert empty strings to None - this preserves lazy behavior
            # and concrete contexts should not be passing empty strings for non-string types
            return None

        # Handle string types - also convert empty strings to None for consistency
        if param_type == str and isinstance(value, str) and value == CONSTANTS.EMPTY_STRING:
            return None

        return value
    
    def get_parameter_display_info(self, param_name: str, param_type: Type,
                                 description: Optional[str] = None) -> Dict[str, str]:
        """
        Get display information for a parameter.
        
        Args:
            param_name: The parameter name
            param_type: The parameter type
            description: Optional parameter description
            
        Returns:
            Dictionary with display information
        """
        return {
            'display_name': format_param_name(param_name),
            'field_label': f"{format_param_name(param_name)}:",
            'checkbox_label': f"Enable {format_param_name(param_name)}",
            'group_title': format_param_name(param_name),
            'description': description or f"Parameter: {format_param_name(param_name)}",
            'tooltip': f"{format_param_name(param_name)} ({param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)})"
        }
    
    def generate_field_ids(self, base_field_id: str, param_name: str) -> Dict[str, str]:
        """
        Generate all field IDs for a parameter.
        
        Args:
            base_field_id: The base field ID
            param_name: The parameter name
            
        Returns:
            Dictionary with all generated field IDs
        """
        widget_id = format_field_id(base_field_id, param_name)

        return {
            'widget_id': widget_id,
            'reset_button_id': format_reset_button_id(widget_id),
            'optional_checkbox_id': f"{base_field_id}_{param_name}_enabled",
            'nested_field_id': f"nested_{base_field_id}_{param_name}",
            'nested_static_id': f"nested_static_{param_name}"
        }
    
    def should_use_concrete_values(self, current_value: Any, is_global_editing: bool = False) -> bool:
        """
        Determine whether to use concrete values for a dataclass parameter.
        
        Args:
            current_value: The current parameter value
            is_global_editing: Whether in global configuration editing mode
            
        Returns:
            True if concrete values should be used
        """
        if current_value is None:
            return False
        
        if is_global_editing:
            return True
        
        # If current_value is a concrete dataclass instance, use its values
        if self._type_utils.is_concrete_dataclass(current_value):
            return True
        
        # For lazy dataclasses, return True so we can extract raw values from them
        if self._type_utils.is_lazy_dataclass(current_value):
            return True
        
        return False
    
    def extract_nested_parameters(self, dataclass_instance: Any, dataclass_type: Type,
                                parent_dataclass_type: Optional[Type] = None) -> Tuple[Dict[str, Any], Dict[str, Type]]:
        """
        Extract parameters and types from a dataclass instance.

        This method always preserves concrete field values when a dataclass instance exists,
        regardless of parent context. Placeholder behavior is handled at the widget level,
        not by discarding concrete values during parameter extraction.
        """
        if not dataclasses.is_dataclass(dataclass_type):
            return {}, {}

        parameters = {}
        parameter_types = {}

        for field in dataclasses.fields(dataclass_type):
            # Always extract actual field values when dataclass instance exists
            # This preserves concrete user-entered values in nested lazy dataclass forms
            if dataclass_instance is not None:
                current_value = self._get_field_value(dataclass_instance, field)
            else:
                current_value = None  # Only use None when no instance exists

            parameters[field.name] = current_value
            parameter_types[field.name] = field.type

        return parameters, parameter_types



    def _get_field_value(self, dataclass_instance: Any, field: Any) -> Any:
        """Extract a single field value from a dataclass instance."""
        if dataclass_instance is None:
            return field.default

        field_name = field.name

        if self._type_utils.has_resolve_field_value(dataclass_instance):
            # Lazy dataclass - get raw value
            return object.__getattribute__(dataclass_instance, field_name) if hasattr(dataclass_instance, field_name) else field.default
        else:
            # Concrete dataclass - get attribute value
            return getattr(dataclass_instance, field_name, field.default)

    def _create_parameter_info(self, param_name: str, param_type: Type, current_value: Any,
                             parameter_info: Optional[Dict] = None) -> ParameterInfo:
        """Create parameter information object."""
        # Check if it's an optional dataclass
        is_optional = self._type_utils.is_optional_dataclass(param_type)
        if is_optional:
            inner_type = self._type_utils.get_optional_inner_type(param_type)
            is_nested = dataclasses.is_dataclass(inner_type)
        else:
            is_nested = dataclasses.is_dataclass(param_type)
        
        # Get description from parameter info
        description = None
        if parameter_info and param_name in parameter_info:
            info_obj = parameter_info[param_name]
            description = getattr(info_obj, 'description', None)
        
        return ParameterInfo(
            name=param_name,
            type=param_type,
            current_value=current_value,
            description=description,
            is_nested=is_nested,
            is_optional=is_optional
        )
    
    def _analyze_nested_dataclass(self, param_name: str, param_type: Type, current_value: Any,
                                nested_field_id: str, parent_dataclass_type: Type = None) -> FormStructure:
        """Analyze a nested dataclass parameter."""
        # Get the actual dataclass type
        if self._type_utils.is_optional_dataclass(param_type):
            dataclass_type = self._type_utils.get_optional_inner_type(param_type)
        else:
            dataclass_type = param_type

        # Extract nested parameters using parent context
        nested_params, nested_types = self.extract_nested_parameters(
            current_value, dataclass_type, parent_dataclass_type
        )

        # Recursively analyze nested structure
        return self.analyze_parameters(nested_params, nested_types, nested_field_id)

    def get_placeholder_text(self, param_name: str, dataclass_type: Type,
                           placeholder_prefix: str = "Pipeline default") -> Optional[str]:
        """
        Get placeholder text using existing OpenHCS infrastructure.

        Args:
            param_name: Name of the parameter to get placeholder for
            dataclass_type: The specific dataclass type (GlobalPipelineConfig or PipelineConfig)
            placeholder_prefix: Prefix for the placeholder text

        The editing mode is automatically derived from the dataclass type's lazy resolution capabilities:
        - Has lazy resolution (PipelineConfig) → orchestrator config editing
        - No lazy resolution (GlobalPipelineConfig) → global config editing
        """
        # Automatically derive editing mode from dataclass type capabilities
        is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)

    def reset_nested_managers(self, nested_managers: Dict[str, Any],
                            dataclass_type: Type, current_config: Any) -> None:
        """Reset all nested managers - fail loud, no defensive programming."""
        for nested_manager in nested_managers.values():
            # All nested managers must have reset_all_parameters method
            nested_manager.reset_all_parameters()



    def get_reset_value_for_parameter(self, param_name: str, param_type: Type,
                                    dataclass_type: Type, is_global_config_editing: Optional[bool] = None) -> Any:
        """
        Get appropriate reset value using existing OpenHCS patterns.

        Args:
            param_name: Name of the parameter to reset
            param_type: Type of the parameter (int, str, bool, etc.)
            dataclass_type: The specific dataclass type
            is_global_config_editing: Whether we're in global config editing mode (auto-detected if None)

        Returns:
            - For global config editing: Actual default values
            - For lazy config editing: None to show placeholder text
        """
        # Context-driven behavior: Use the editing context to determine reset behavior
        # This follows the architectural principle that behavior is determined by context
        # of usage rather than intrinsic properties of the dataclass.

        # Context-driven behavior: Use explicit context when provided
        # Auto-detect editing mode if not explicitly provided
        if is_global_config_editing is None:
            # Fallback: Use existing lazy resolution detection for backward compatibility
            is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)

        # Context-driven behavior: Reset behavior depends on editing context
        if is_global_config_editing:
            # Global config editing: Reset to actual default values
            # Users expect to see concrete defaults when editing global configuration
            return self._get_actual_dataclass_field_default(param_name, dataclass_type)
        else:
            # CRITICAL FIX: For lazy config editing, always return None
            # This ensures reset shows inheritance chain values (like compiler resolution)
            # instead of concrete values from thread-local context
            return None

    def _get_actual_dataclass_field_default(self, param_name: str, dataclass_type: Type) -> Any:
        """
        Get the actual default value for a dataclass field by creating a default instance.

        This ensures we get the real defaults (like num_workers=16) instead of hardcoded primitives.

        Raises:
            ValueError: If dataclass creation fails or field doesn't exist
        """
        try:
            # Create a default instance of the dataclass to get actual field defaults
            default_instance = dataclass_type()
            if not hasattr(default_instance, param_name):
                raise ValueError(f"Field '{param_name}' not found in dataclass {dataclass_type.__name__}")
            return getattr(default_instance, param_name)
        except Exception as e:
            raise ValueError(
                f"Failed to get default value for field '{param_name}' in dataclass {dataclass_type.__name__}: {e}"
            ) from e
