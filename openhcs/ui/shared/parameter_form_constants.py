"""
Parameter form constants for eliminating magic strings throughout the parameter form managers.

This module centralizes all hardcoded strings used in both PyQt and Textual parameter form
implementations to improve maintainability and reduce duplication.
"""

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ParameterFormConstants:
    """
    Centralized constants for parameter form implementations.

    This dataclass eliminates magic strings throughout the parameter form system
    by providing a single source of truth for all hardcoded values used in both
    PyQt and Textual implementations.

    Categories:
    - UI text and formatting patterns
    - Widget identification and naming
    - Framework-specific constants
    - Debug and validation settings
    """
    
    # Field ID generation patterns
    FIELD_ID_SEPARATOR: str = "_"
    NESTED_STATIC_PREFIX: str = "nested_static_"
    MIXED_CLASS_PREFIX: str = "Mixed"
    ENABLE_CHECKBOX_PREFIX: str = "Enable "
    RESET_BUTTON_PREFIX: str = "reset_"
    
    # Placeholder text patterns
    DEFAULT_PLACEHOLDER_PREFIX: str = "Pipeline default"
    PLACEHOLDER_WITH_COLON: str = "Pipeline default:"
    PLACEHOLDER_NONE_TEXT: str = "(none)"
    
    # Widget text constants
    RESET_BUTTON_TEXT: str = "Reset"
    NONE_STRING_LITERAL: str = "None"
    EMPTY_STRING: str = ""
    
    # Parameter name formatting
    UNDERSCORE_REPLACEMENT: str = " "
    FIELD_LABEL_SUFFIX: str = ":"
    PARAMETER_DESCRIPTION_PREFIX: str = "Parameter: "
    
    # Debug and logging constants
    DEBUG_PREFIX: str = "*** "
    DEBUG_SUFFIX: str = " ***"
    DEBUG_TARGET_PARAMS: FrozenSet[str] = frozenset({
        "output_dir_suffix", 
        "path_planning"
    })
    
    # Nested manager patterns
    NESTED_MANAGERS_ATTR: str = "nested_managers"
    OPTIONAL_CHECKBOXES_ATTR: str = "optional_checkboxes"
    ENABLED_SUFFIX: str = "_enabled"
    
    # Type checking strings
    DATACLASS_FIELDS_ATTR: str = "__dataclass_fields__"
    RESOLVE_FIELD_VALUE_ATTR: str = "_resolve_field_value"
    BASES_ATTR: str = "__bases__"
    VALUE_ATTR: str = "value"
    
    # Boolean conversion strings
    TRUE_STRINGS: FrozenSet[str] = frozenset({
        "true", "1", "yes", "on"
    })
    
    # CSS and styling constants
    PARAM_LABEL_CLASS: str = "param-label clickable"
    PLACEHOLDER_STYLE_PROPERTY: str = "is_placeholder_state"
    
    # Layout and sizing constants
    AUTO_SIZE: str = "auto"
    FLEXIBLE_WIDTH: str = "1fr"
    LEFT_ALIGN: str = "left"
    
    # Error and validation messages
    UNKNOWN_CONFIG_KEY_MSG: str = "Unknown config key: {}"
    INVALID_CONFIG_KEYS_MSG: str = "Invalid config keys: {}"
    CONFIG_MUST_BE_DICT_MSG: str = "Config must be a non-empty dictionary"
    NO_WIDGET_CREATOR_MSG: str = "No widget creator registered for type: {}"
    CONVERSION_ERROR_MSG: str = "Cannot convert '{}' to {}: {}"
    NO_VALID_CONVERSION_MSG: str = "No valid conversion found for Union type {}"
    
    # File path and naming patterns
    GLOBAL_CONTEXT_LAZY_PREFIX: str = "GlobalContextLazy"
    STATIC_LAZY_PREFIX: str = "StaticLazy"
    
    # Special field analysis types
    DIFFERENT_VALUES_TYPE: str = "different"
    
    # Tooltip and help text patterns
    TOOLTIP_SEPARATOR: str = ": "
    HELP_PARAMETER_PREFIX: str = "Parameter: "
    
    # Widget state constants
    COLLAPSED_STATE: bool = True
    EXPANDED_STATE: bool = False
    COMPACT_WIDGET: bool = True
    
    # Margin and spacing constants (for Textual layouts)
    NO_MARGIN: tuple = (0, 0, 0, 0)
    LEFT_MARGIN_ONLY: tuple = (0, 0, 0, 1)
    
    # Thread-local and context constants
    CURRENT_PIPELINE_CONFIG_ATTR: str = "value"
    
    # Method and attribute name constants
    SET_PATH_METHOD: str = "set_path"
    GET_VALUE_METHOD: str = "get_value"
    SET_VALUE_METHOD: str = "setValue"  # PyQt6 uses camelCase
    BLOCK_SIGNALS_METHOD: str = "blockSignals"
    SET_TEXT_METHOD: str = "setText"
    SET_CHECKED_METHOD: str = "setChecked"
    
    # Logging and debug message templates
    NESTED_DEBUG_MSG: str = "*** NESTED DEBUG *** param_name={}, parent_nested_name={}"
    NESTED_UPDATE_MSG: str = "*** NESTED UPDATE *** Updating {}.{} = {}"
    RESET_DEBUG_MSG: str = "*** RESET DEBUG *** param_name={}, parent_nested_name={}"
    FALLBACK_DEBUG_MSG: str = "*** FALLBACK DEBUG *** Checking fallback for {}"
    TEXTUAL_UPDATE_DEBUG_MSG: str = "*** TEXTUAL UPDATE DEBUG *** {} update_parameter called with: {} (type: {})"
    
    # Path and hierarchy constants
    DOT_SEPARATOR: str = "."
    PATH_PLANNING_FIELD: str = "path_planning"
    OUTPUT_DIR_SUFFIX_FIELD: str = "output_dir_suffix"
    
    # Widget creation and registry constants
    INPUT_TYPE_INTEGER: str = "integer"
    INPUT_TYPE_NUMBER: str = "number"
    INPUT_TYPE_TEXT: str = "text"
    
    # Configuration context constants
    GLOBAL_CONFIG_EDITING_CONTEXT: str = "global_config_editing"
    LAZY_CONTEXT: str = "lazy_context"
    
    # Framework identification constants
    PYQT6_FRAMEWORK: str = "pyqt6"
    TEXTUAL_FRAMEWORK: str = "textual"


# Create a singleton instance for easy access throughout the codebase
CONSTANTS = ParameterFormConstants()
