"""
Abstract base class for parameter form managers.

This module defines the common interface and shared behavior for both PyQt and Textual
parameter form implementations, establishing contracts and providing shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional, List
from dataclasses import dataclass

from openhcs.ui.shared.parameter_form_constants import CONSTANTS
from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils
from openhcs.ui.shared.ui_utils import debug_param


@dataclass
class ParameterFormConfig:
    """
    Configuration for parameter form managers.

    This dataclass encapsulates all configuration options for parameter form
    managers, providing a clean interface for customizing form behavior.

    Attributes:
        field_id: Unique identifier for the form
        parameter_info: Optional parameter information dictionary
        is_global_config_editing: Whether editing global configuration
        global_config_type: Type of global configuration being edited
        placeholder_prefix: Prefix for placeholder text
        use_scroll_area: Whether to use scroll area (PyQt only)
        enable_debug: Whether to enable debug logging
        debug_target_params: Set of parameters to debug
        framework: UI framework ('pyqt6' or 'textual')
        color_scheme: Optional color scheme for PyQt
        function_target: Optional function target for docstring fallback
    """
    field_id: str
    parameter_info: Optional[Dict] = None
    is_global_config_editing: bool = False
    global_config_type: Optional[Type] = None
    placeholder_prefix: str = CONSTANTS.DEFAULT_PLACEHOLDER_PREFIX
    use_scroll_area: bool = True
    enable_debug: bool = False
    debug_target_params: Optional[set] = None
    framework: str = CONSTANTS.TEXTUAL_FRAMEWORK
    color_scheme: Optional[Any] = None
    function_target: Optional[Any] = None

    @classmethod
    def for_pyqt(cls, field_id: str, **kwargs) -> 'ParameterFormConfig':
        """Create configuration for PyQt parameter form manager."""
        return cls(field_id=field_id, framework=CONSTANTS.PYQT6_FRAMEWORK, **kwargs)

    @classmethod
    def for_textual(cls, field_id: str, **kwargs) -> 'ParameterFormConfig':
        """Create configuration for Textual parameter form manager."""
        return cls(field_id=field_id, framework=CONSTANTS.TEXTUAL_FRAMEWORK, **kwargs)

    def with_debug(self, enabled: bool = True, target_params: Optional[set] = None) -> 'ParameterFormConfig':
        """Return a copy with debug settings configured."""
        import copy
        config = copy.deepcopy(self)
        config.enable_debug = enabled
        if target_params is not None:
            config.debug_target_params = target_params
        return config

    def with_global_config(self, global_config_type: Type, editing: bool = True) -> 'ParameterFormConfig':
        """Return a copy with global configuration settings."""
        import copy
        config = copy.deepcopy(self)
        config.is_global_config_editing = editing
        config.global_config_type = global_config_type
        return config


class ParameterFormManagerBase(ABC):
    """
    Abstract base class for parameter form managers.
    
    This class defines the common interface and shared behavior for both PyQt
    and Textual parameter form implementations. It provides:
    
    - Common initialization patterns
    - Shared utility access
    - Abstract methods that must be implemented
    - Common parameter management operations
    - Debug logging infrastructure
    
    Subclasses must implement the abstract methods to provide framework-specific
    widget creation and form building functionality.
    """
    
    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, Type], 
                 config: ParameterFormConfig):
        """
        Initialize the parameter form manager with common setup.
        
        Args:
            parameters: Dictionary of parameter names to current values
            parameter_types: Dictionary of parameter names to types
            config: Configuration object for the form manager
        """
        # Store core data
        self.parameters = parameters.copy()
        self.parameter_types = parameter_types
        self.config = config
        
        # Initialize shared utilities
        self.type_utils = ParameterTypeUtils()
        
        # Track nested managers and widgets
        self.nested_managers = {}
        self.widgets = {}
        
        # Log initialization
        self.debugger.log_form_manager_operation("form_manager_initialized", {
            "field_id": config.field_id,
            "parameter_count": len(parameters),
            "has_nested_params": any(ParameterTypeUtils.has_dataclass_fields(t) for t in parameter_types.values())
        })
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def build_form(self) -> Any:
        """
        Build the complete form UI.
        
        This method must be implemented by subclasses to create the framework-specific
        form UI containing all parameter widgets.
        
        Returns:
            The framework-specific form widget/container
        """
        pass
    
    @abstractmethod
    def create_parameter_widget(self, param_name: str, param_type: Type, current_value: Any) -> Any:
        """
        Create a widget for a single parameter.
        
        This method must be implemented by subclasses to create framework-specific
        widgets for individual parameters.
        
        Args:
            param_name: The parameter name
            param_type: The parameter type
            current_value: The current parameter value
            
        Returns:
            The framework-specific widget
        """
        pass
    
    @abstractmethod
    def create_nested_form(self, param_name: str, param_type: Type, current_value: Any) -> Any:
        """
        Create a nested form for dataclass parameters.
        
        This method must be implemented by subclasses to create framework-specific
        nested forms for dataclass parameters.
        
        Args:
            param_name: The parameter name
            param_type: The dataclass type
            current_value: The current dataclass value
            
        Returns:
            The framework-specific nested form widget/container
        """
        pass
    
    @abstractmethod
    def update_widget_value(self, widget: Any, value: Any) -> None:
        """
        Update a widget's value.
        
        This method must be implemented by subclasses to update framework-specific
        widget values.
        
        Args:
            widget: The framework-specific widget
            value: The new value to set
        """
        pass
    
    @abstractmethod
    def get_widget_value(self, widget: Any) -> Any:
        """
        Get a widget's current value.
        
        This method must be implemented by subclasses to retrieve values from
        framework-specific widgets.
        
        Args:
            widget: The framework-specific widget
            
        Returns:
            The current widget value
        """
        pass
    
    # Shared concrete methods
    
    def update_parameter(self, param_name: str, value: Any) -> None:
        """
        Update a parameter value with type conversion and nested handling.

        This method provides common parameter update logic that handles type
        conversion, nested parameters, and debug logging. It updates both the
        internal data model and the corresponding widget.

        Args:
            param_name: The parameter name to update
            value: The new value
        """
        self.debugger.log_parameter_update(param_name, value, "update_parameter")

        # Handle nested parameters
        if self._is_nested_parameter(param_name):
            self._update_nested_parameter(param_name, value)
            return

        # Handle regular parameters
        if param_name in self.parameters:
            # Convert value to appropriate type
            converted_value = self._convert_value_to_type(value, param_name)

            # Update parameter in data model
            old_value = self.parameters[param_name]
            self.parameters[param_name] = converted_value

            self.debugger.log_parameter_update(param_name, converted_value, "parameter_stored")

            # Update corresponding widget if it exists
            if param_name in self.widgets:
                self.update_widget_value(self.widgets[param_name], converted_value)

    def reset_all_parameters(self, defaults: Dict[str, Any] = None) -> None:
        """
        Reset all parameters to their default values.

        Args:
            defaults: Optional dictionary of default values to use
        """
        self.debugger.log_form_manager_operation("reset_all_parameters", {
            "parameter_count": len(self.parameters),
            "has_custom_defaults": defaults is not None
        })

        for param_name in self.parameters.keys():
            if defaults and param_name in defaults:
                default_value = defaults[param_name]
            else:
                default_value = self._get_default_value_for_parameter(param_name)

            self.reset_parameter(param_name, default_value)
    
    def reset_parameter(self, param_name: str, default_value: Any = None) -> None:
        """
        Reset a parameter to its default value.
        
        Args:
            param_name: The parameter name to reset
            default_value: Optional default value (uses type default if None)
        """
        if default_value is None:
            default_value = self._get_default_value_for_parameter(param_name)
        
        old_value = self.parameters.get(param_name)
        self.debugger.log_reset_operation(param_name, old_value, default_value)
        
        self.update_parameter(param_name, default_value)
    
    def get_current_values(self) -> Dict[str, Any]:
        """
        Get all current parameter values.
        
        Returns:
            Dictionary of parameter names to current values
        """
        return self.parameters.copy()
    
    def get_parameter_info(self, param_name: str) -> Optional[Any]:
        """
        Get parameter information for a parameter.
        
        Args:
            param_name: The parameter name
            
        Returns:
            Parameter info object or None
        """
        if self.config.parameter_info:
            return self.config.parameter_info.get(param_name)
        return None
    
    # Protected helper methods
    
    def _is_nested_parameter(self, param_name: str) -> bool:
        """Check if a parameter name represents a nested parameter."""
        return CONSTANTS.FIELD_ID_SEPARATOR in param_name
    
    def _update_nested_parameter(self, param_name: str, value: Any) -> None:
        """Update a nested parameter by delegating to the appropriate nested manager."""
        parts = param_name.split(CONSTANTS.FIELD_ID_SEPARATOR)
        
        # Find the nested manager
        for i in range(1, len(parts)):
            potential_nested = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[:i])
            if potential_nested in self.nested_managers:
                nested_field = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[i:])
                self.debugger.log_nested_update(potential_nested, nested_field, value)
                self.nested_managers[potential_nested].update_parameter(nested_field, value)
                return
    
    def _convert_value_to_type(self, value: Any, param_name: str) -> Any:
        """Convert a value to the appropriate type for a parameter."""
        if param_name not in self.parameter_types or value is None:
            return value
        
        param_type = self.parameter_types[param_name]
        
        # Handle string "None" literal
        if isinstance(value, str) and value == CONSTANTS.NONE_STRING_LITERAL:
            return None
        
        # Handle enum types
        if ParameterTypeUtils.is_enum_type(param_type):
            return param_type(value)
        
        # Handle list of enums
        if ParameterTypeUtils.is_list_of_enums(param_type):
            # If value is already a list (from checkbox group widget), return as-is
            if isinstance(value, list):
                return value
            enum_type = ParameterTypeUtils.get_enum_from_list_type(param_type)
            if enum_type:
                return [enum_type(value)]
        
        # Handle basic types
        if param_type == bool and isinstance(value, str):
            return ParameterTypeUtils.convert_string_to_bool(value)
        
        if param_type in (int, float) and isinstance(value, str):
            if value == CONSTANTS.EMPTY_STRING:
                return None
            try:
                return param_type(value)
            except (ValueError, TypeError):
                return None
        
        return value
    
    def _get_default_value_for_parameter(self, param_name: str) -> Any:
        """Get the default value for a parameter."""
        # This is a simplified implementation - subclasses may override
        param_type = self.parameter_types.get(param_name)
        
        if param_type == bool:
            return False
        elif param_type == int:
            return 0
        elif param_type == float:
            return 0.0
        elif param_type == str:
            return CONSTANTS.EMPTY_STRING
        else:
            return None
