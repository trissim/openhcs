"""
Parameter name formatting utility for parameter form managers.

This module provides centralized parameter name formatting methods to eliminate
repeated string manipulation patterns and ensure consistent display formatting
across both PyQt and Textual parameter form implementations.
"""

from openhcs.ui.shared.parameter_form_constants import CONSTANTS


class ParameterNameFormatter:
    """
    Utility class for consistent parameter name formatting across parameter form managers.
    
    This class provides static methods for converting parameter names (typically in
    snake_case) to various display formats used throughout the UI. All formatting
    follows consistent patterns and uses centralized constants.
    
    Formatting Patterns:
    - Display names: "param_name" -> "Param Name"
    - Checkbox labels: "param_name" -> "Enable Param Name"
    - Field labels: "param_name" -> "Param Name:"
    - Group titles: "param_name" -> "Param Name"
    """
    
    @staticmethod
    def to_display_name(param_name: str) -> str:
        """
        Convert parameter name to display format.
        
        This converts snake_case parameter names to Title Case display names
        by replacing underscores with spaces and capitalizing each word.
        
        Args:
            param_name: The parameter name in snake_case
            
        Returns:
            Formatted display name in Title Case
            
        Example:
            >>> ParameterNameFormatter.to_display_name("output_dir_suffix")
            "Output Dir Suffix"
            >>> ParameterNameFormatter.to_display_name("num_workers")
            "Num Workers"
        """
        return param_name.replace(CONSTANTS.FIELD_ID_SEPARATOR, CONSTANTS.UNDERSCORE_REPLACEMENT).title()
    
    @staticmethod
    def to_checkbox_label(param_name: str) -> str:
        """
        Convert parameter name to checkbox label format.
        
        This creates checkbox labels for optional parameters by prefixing
        the display name with "Enable " to indicate the checkbox controls
        whether the optional parameter is active.
        
        Args:
            param_name: The parameter name in snake_case
            
        Returns:
            Formatted checkbox label with "Enable " prefix
            
        Example:
            >>> ParameterNameFormatter.to_checkbox_label("advanced_settings")
            "Enable Advanced Settings"
            >>> ParameterNameFormatter.to_checkbox_label("path_planning")
            "Enable Path Planning"
        """
        display_name = ParameterNameFormatter.to_display_name(param_name)
        return f"{CONSTANTS.ENABLE_CHECKBOX_PREFIX}{display_name}"
    
    @staticmethod
    def to_field_label(param_name: str) -> str:
        """
        Convert parameter name to field label format.
        
        This creates field labels for form inputs by converting to display
        format and adding a colon suffix to indicate it's a form label.
        
        Args:
            param_name: The parameter name in snake_case
            
        Returns:
            Formatted field label with colon suffix
            
        Example:
            >>> ParameterNameFormatter.to_field_label("num_workers")
            "Num Workers:"
            >>> ParameterNameFormatter.to_field_label("output_format")
            "Output Format:"
        """
        display_name = ParameterNameFormatter.to_display_name(param_name)
        return f"{display_name}{CONSTANTS.FIELD_LABEL_SUFFIX}"
    
    @staticmethod
    def to_group_title(param_name: str) -> str:
        """
        Convert parameter name to group title format.
        
        This creates group titles for nested dataclass sections. Currently
        identical to display name format but separated for potential future
        customization of group title styling.
        
        Args:
            param_name: The parameter name in snake_case
            
        Returns:
            Formatted group title
            
        Example:
            >>> ParameterNameFormatter.to_group_title("path_planning")
            "Path Planning"
            >>> ParameterNameFormatter.to_group_title("advanced_config")
            "Advanced Config"
        """
        return ParameterNameFormatter.to_display_name(param_name)
    
    @staticmethod
    def to_parameter_description(param_name: str, custom_description: str = None) -> str:
        """
        Generate parameter description with fallback to default format.
        
        This creates parameter descriptions for help text and tooltips. If a
        custom description is provided, it's used as-is. Otherwise, a default
        description is generated using the parameter name.
        
        Args:
            param_name: The parameter name in snake_case
            custom_description: Optional custom description
            
        Returns:
            Parameter description text
            
        Example:
            >>> ParameterNameFormatter.to_parameter_description("num_workers")
            "Parameter: num workers"
            >>> ParameterNameFormatter.to_parameter_description("num_workers", "Number of worker threads")
            "Number of worker threads"
        """
        if custom_description:
            return custom_description
        
        # Generate default description
        display_name = param_name.replace(CONSTANTS.FIELD_ID_SEPARATOR, CONSTANTS.UNDERSCORE_REPLACEMENT)
        return f"{CONSTANTS.PARAMETER_DESCRIPTION_PREFIX}{display_name}"
    
    @staticmethod
    def to_tooltip_text(param_name: str, description: str = None, param_type: type = None) -> str:
        """
        Generate comprehensive tooltip text for parameters.
        
        This creates detailed tooltip text that includes the parameter description
        and optionally the parameter type information for enhanced user guidance.
        
        Args:
            param_name: The parameter name in snake_case
            description: Optional parameter description
            param_type: Optional parameter type for type information
            
        Returns:
            Formatted tooltip text
            
        Example:
            >>> ParameterNameFormatter.to_tooltip_text("num_workers", "Number of worker threads", int)
            "Number of worker threads (type: int)"
            >>> ParameterNameFormatter.to_tooltip_text("output_dir")
            "Parameter: output dir"
        """
        base_description = ParameterNameFormatter.to_parameter_description(param_name, description)
        
        if param_type:
            type_name = getattr(param_type, '__name__', str(param_type))
            return f"{base_description} (type: {type_name})"
        
        return base_description
    
    @staticmethod
    def to_debug_identifier(param_name: str, context: str = None) -> str:
        """
        Generate debug identifier for logging and debugging.
        
        This creates consistent debug identifiers for parameters that can be
        used in logging messages and debug output. Optionally includes context
        information for more detailed debugging.
        
        Args:
            param_name: The parameter name in snake_case
            context: Optional context information (e.g., "nested", "reset")
            
        Returns:
            Formatted debug identifier
            
        Example:
            >>> ParameterNameFormatter.to_debug_identifier("output_dir_suffix")
            "output_dir_suffix"
            >>> ParameterNameFormatter.to_debug_identifier("output_dir_suffix", "nested")
            "nested.output_dir_suffix"
        """
        if context:
            return f"{context}{CONSTANTS.DOT_SEPARATOR}{param_name}"
        return param_name
    
    @staticmethod
    def extract_param_name_from_widget_id(widget_id: str, field_id: str) -> str:
        """
        Extract parameter name from widget ID.
        
        This reverses the widget ID generation process to extract the original
        parameter name from a widget ID, given the form's field ID.
        
        Args:
            widget_id: The widget ID to extract from
            field_id: The form's field ID used to generate the widget ID
            
        Returns:
            The extracted parameter name
            
        Raises:
            ValueError: If the widget ID doesn't match the expected pattern
            
        Example:
            >>> ParameterNameFormatter.extract_param_name_from_widget_id("config_num_workers", "config")
            "num_workers"
        """
        expected_prefix = f"{field_id}{CONSTANTS.FIELD_ID_SEPARATOR}"
        
        if not widget_id.startswith(expected_prefix):
            raise ValueError(f"Widget ID '{widget_id}' doesn't match expected pattern for field ID '{field_id}'")
        
        return widget_id[len(expected_prefix):]
    
    @staticmethod
    def is_debug_target_param(param_name: str) -> bool:
        """
        Check if a parameter is a debug target.
        
        This determines whether a parameter should receive enhanced debug logging
        based on the centralized debug target configuration.
        
        Args:
            param_name: The parameter name to check
            
        Returns:
            True if the parameter is a debug target
            
        Example:
            >>> ParameterNameFormatter.is_debug_target_param("output_dir_suffix")
            True
            >>> ParameterNameFormatter.is_debug_target_param("num_workers")
            False
        """
        return param_name in CONSTANTS.DEBUG_TARGET_PARAMS
    
    @staticmethod
    def format_debug_message(message_template: str, *args, **kwargs) -> str:
        """
        Format debug message with consistent debug styling.
        
        This applies consistent debug message formatting with the standard
        debug prefix and suffix patterns used throughout the system.
        
        Args:
            message_template: The message template with format placeholders
            *args: Positional arguments for string formatting
            **kwargs: Keyword arguments for string formatting
            
        Returns:
            Formatted debug message with debug styling
            
        Example:
            >>> ParameterNameFormatter.format_debug_message("param={}, value={}", "test", 42)
            "*** param=test, value=42 ***"
        """
        formatted_message = message_template.format(*args, **kwargs)
        return f"{CONSTANTS.DEBUG_PREFIX}{formatted_message}{CONSTANTS.DEBUG_SUFFIX}"
