"""
Field ID generation utility for parameter form managers.

This module provides centralized field ID generation methods to eliminate repeated
string formatting logic across both PyQt and Textual parameter form implementations.
"""

from openhcs.ui.shared.parameter_form_constants import CONSTANTS


class FieldIdGenerator:
    """
    Utility class for generating consistent field IDs across parameter form managers.
    
    This class provides static methods for creating field IDs using standardized patterns
    that are shared between PyQt and Textual implementations. All methods use the
    centralized constants to ensure consistency.
    
    Field ID Patterns:
    - Regular fields: "{parent_id}_{param_name}"
    - Nested static fields: "nested_static_{param_name}"
    - Widget IDs: "{field_id}_{param_name}"
    - Reset button IDs: "reset_{widget_id}"
    - Optional checkbox IDs: "{field_id}_{param_name}_enabled"
    """
    
    @staticmethod
    def nested_field_id(parent_id: str, param_name: str) -> str:
        """
        Generate a nested field ID by combining parent ID and parameter name.
        
        This is the standard pattern for creating hierarchical field IDs in nested
        dataclass forms where the parent form's field_id is combined with the
        parameter name using the standard separator.
        
        Args:
            parent_id: The parent form's field ID
            param_name: The parameter name for the nested field
            
        Returns:
            Formatted field ID: "{parent_id}_{param_name}"
            
        Example:
            >>> FieldIdGenerator.nested_field_id("config", "path_planning")
            "config_path_planning"
        """
        return f"{parent_id}{CONSTANTS.FIELD_ID_SEPARATOR}{param_name}"
    
    @staticmethod
    def nested_static_field_id(param_name: str) -> str:
        """
        Generate a nested static field ID for placeholder-only forms.
        
        This pattern is used for nested forms that should not use thread-local
        resolution but instead use static placeholder generation. These forms
        are identified by the "nested_static_" prefix.
        
        Args:
            param_name: The parameter name for the static nested field
            
        Returns:
            Formatted field ID: "nested_static_{param_name}"
            
        Example:
            >>> FieldIdGenerator.nested_static_field_id("path_planning")
            "nested_static_path_planning"
        """
        return f"{CONSTANTS.NESTED_STATIC_PREFIX}{param_name}"
    
    @staticmethod
    def widget_id(field_id: str, param_name: str) -> str:
        """
        Generate a widget ID for individual parameter widgets.
        
        This is the standard pattern for creating widget IDs that uniquely identify
        individual input widgets within a form. The widget ID is used for event
        handling and widget lookup.
        
        Args:
            field_id: The form's field ID
            param_name: The parameter name for the widget
            
        Returns:
            Formatted widget ID: "{field_id}_{param_name}"
            
        Example:
            >>> FieldIdGenerator.widget_id("config", "num_workers")
            "config_num_workers"
        """
        return f"{field_id}{CONSTANTS.FIELD_ID_SEPARATOR}{param_name}"
    
    @staticmethod
    def reset_button_id(widget_id: str) -> str:
        """
        Generate a reset button ID for a parameter widget.
        
        This creates a standardized ID for reset buttons associated with parameter
        widgets. The reset button ID is derived from the widget ID with the
        standard reset prefix.
        
        Args:
            widget_id: The widget ID for the associated parameter widget
            
        Returns:
            Formatted reset button ID: "reset_{widget_id}"
            
        Example:
            >>> FieldIdGenerator.reset_button_id("config_num_workers")
            "reset_config_num_workers"
        """
        return f"{CONSTANTS.RESET_BUTTON_PREFIX}{widget_id}"
    
    @staticmethod
    def optional_checkbox_id(field_id: str, param_name: str) -> str:
        """
        Generate a checkbox ID for Optional[dataclass] parameters.
        
        This creates a standardized ID for checkboxes that enable/disable
        optional dataclass parameters. The checkbox controls whether the
        optional parameter is None or has a value.
        
        Args:
            field_id: The form's field ID
            param_name: The parameter name for the optional dataclass
            
        Returns:
            Formatted checkbox ID: "{field_id}_{param_name}_enabled"
            
        Example:
            >>> FieldIdGenerator.optional_checkbox_id("config", "advanced_settings")
            "config_advanced_settings_enabled"
        """
        return f"{field_id}{CONSTANTS.FIELD_ID_SEPARATOR}{param_name}{CONSTANTS.ENABLED_SUFFIX}"
    
    @staticmethod
    def is_nested_static_field(field_id: str) -> bool:
        """
        Check if a field ID represents a nested static field.
        
        This utility method determines whether a field ID was generated using
        the nested_static_field_id pattern, which indicates the form should
        use static placeholder generation instead of thread-local resolution.
        
        Args:
            field_id: The field ID to check
            
        Returns:
            True if the field ID starts with the nested static prefix
            
        Example:
            >>> FieldIdGenerator.is_nested_static_field("nested_static_path_planning")
            True
            >>> FieldIdGenerator.is_nested_static_field("config_path_planning")
            False
        """
        return field_id.startswith(CONSTANTS.NESTED_STATIC_PREFIX)
    
    @staticmethod
    def extract_param_name_from_nested_static(field_id: str) -> str:
        """
        Extract the parameter name from a nested static field ID.
        
        This utility method extracts the original parameter name from a field ID
        that was generated using the nested_static_field_id pattern.
        
        Args:
            field_id: The nested static field ID
            
        Returns:
            The parameter name without the nested static prefix
            
        Raises:
            ValueError: If the field ID is not a nested static field ID
            
        Example:
            >>> FieldIdGenerator.extract_param_name_from_nested_static("nested_static_path_planning")
            "path_planning"
        """
        if not FieldIdGenerator.is_nested_static_field(field_id):
            raise ValueError(f"Field ID '{field_id}' is not a nested static field ID")
        
        return field_id[len(CONSTANTS.NESTED_STATIC_PREFIX):]
    
    @staticmethod
    def create_lazy_class_name(base_class_name: str, prefix: str = None) -> str:
        """
        Generate a lazy dataclass name using standardized naming patterns.
        
        This creates consistent names for dynamically generated lazy dataclass
        types, using either the global context or static lazy prefixes.
        
        Args:
            base_class_name: The name of the base dataclass
            prefix: Optional prefix (defaults to MIXED_CLASS_PREFIX)
            
        Returns:
            Formatted lazy class name
            
        Example:
            >>> FieldIdGenerator.create_lazy_class_name("PathPlanningConfig")
            "MixedPathPlanningConfig"
            >>> FieldIdGenerator.create_lazy_class_name("PathPlanningConfig", "GlobalContextLazy")
            "GlobalContextLazyPathPlanningConfig"
        """
        if prefix is None:
            prefix = CONSTANTS.MIXED_CLASS_PREFIX
        
        return f"{prefix}{base_class_name}"
