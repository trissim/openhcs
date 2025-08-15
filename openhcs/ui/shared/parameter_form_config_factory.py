"""
Configuration factory for parameter form managers.

This module provides factory methods and configuration builders for creating
parameter form configurations with common patterns and presets.
"""

from typing import Dict, Any, Type, Optional, Set
from dataclasses import dataclass, field

from openhcs.ui.shared.parameter_form_base import ParameterFormConfig
from openhcs.ui.shared.parameter_form_constants import CONSTANTS


class ParameterFormConfigBuilder:
    """
    Builder pattern for creating parameter form configurations.
    
    This class provides a fluent interface for building parameter form
    configurations with method chaining and validation.
    """
    
    def __init__(self, field_id: str):
        """Initialize builder with required field ID."""
        self._field_id = field_id
        self._parameter_info = None
        self._is_global_config_editing = False
        self._global_config_type = None
        self._placeholder_prefix = CONSTANTS.DEFAULT_PLACEHOLDER_PREFIX
        self._use_scroll_area = True
        self._enable_debug = False
        self._debug_target_params = None
        self._framework = CONSTANTS.TEXTUAL_FRAMEWORK
        self._color_scheme = None
        self._function_target = None
    
    def with_parameter_info(self, parameter_info: Dict[str, Any]) -> 'ParameterFormConfigBuilder':
        """Set parameter information dictionary."""
        self._parameter_info = parameter_info
        return self
    
    def with_global_config(self, global_config_type: Type, editing: bool = True) -> 'ParameterFormConfigBuilder':
        """Configure for global configuration editing."""
        self._is_global_config_editing = editing
        self._global_config_type = global_config_type
        return self
    
    def with_placeholder_prefix(self, prefix: str) -> 'ParameterFormConfigBuilder':
        """Set custom placeholder prefix."""
        self._placeholder_prefix = prefix
        return self
    
    def with_scroll_area(self, use_scroll: bool = True) -> 'ParameterFormConfigBuilder':
        """Configure scroll area usage (PyQt only)."""
        self._use_scroll_area = use_scroll
        return self
    
    def with_debug(self, enabled: bool = True, target_params: Optional[Set[str]] = None) -> 'ParameterFormConfigBuilder':
        """Configure debug logging."""
        self._enable_debug = enabled
        self._debug_target_params = target_params
        return self
    
    def for_pyqt(self, color_scheme: Any = None, function_target: Any = None) -> 'ParameterFormConfigBuilder':
        """Configure for PyQt framework."""
        self._framework = CONSTANTS.PYQT6_FRAMEWORK
        self._color_scheme = color_scheme
        self._function_target = function_target
        return self
    
    def for_textual(self) -> 'ParameterFormConfigBuilder':
        """Configure for Textual framework."""
        self._framework = CONSTANTS.TEXTUAL_FRAMEWORK
        return self
    
    def build(self) -> ParameterFormConfig:
        """Build the final configuration."""
        return ParameterFormConfig(
            field_id=self._field_id,
            parameter_info=self._parameter_info,
            is_global_config_editing=self._is_global_config_editing,
            global_config_type=self._global_config_type,
            placeholder_prefix=self._placeholder_prefix,
            use_scroll_area=self._use_scroll_area,
            enable_debug=self._enable_debug,
            debug_target_params=self._debug_target_params,
            framework=self._framework,
            color_scheme=self._color_scheme,
            function_target=self._function_target
        )


class ParameterFormConfigFactory:
    """
    Factory for creating common parameter form configurations.
    
    This class provides static methods for creating parameter form configurations
    with common patterns and presets, reducing boilerplate code.
    """
    
    @staticmethod
    def create_basic_config(field_id: str, framework: str = CONSTANTS.TEXTUAL_FRAMEWORK) -> ParameterFormConfig:
        """
        Create a basic parameter form configuration.
        
        Args:
            field_id: Unique identifier for the form
            framework: UI framework ('pyqt6' or 'textual')
            
        Returns:
            Basic parameter form configuration
        """
        return ParameterFormConfig(field_id=field_id, framework=framework)
    
    @staticmethod
    def create_debug_config(field_id: str, target_params: Set[str], 
                          framework: str = CONSTANTS.TEXTUAL_FRAMEWORK) -> ParameterFormConfig:
        """
        Create a parameter form configuration with debug logging enabled.
        
        Args:
            field_id: Unique identifier for the form
            target_params: Set of parameters to debug
            framework: UI framework ('pyqt6' or 'textual')
            
        Returns:
            Debug-enabled parameter form configuration
        """
        return ParameterFormConfig(
            field_id=field_id,
            framework=framework,
            enable_debug=True,
            debug_target_params=target_params
        )
    
    @staticmethod
    def create_global_config(field_id: str, global_config_type: Type,
                           framework: str = CONSTANTS.TEXTUAL_FRAMEWORK) -> ParameterFormConfig:
        """
        Create a parameter form configuration for global configuration editing.
        
        Args:
            field_id: Unique identifier for the form
            global_config_type: Type of global configuration being edited
            framework: UI framework ('pyqt6' or 'textual')
            
        Returns:
            Global configuration parameter form configuration
        """
        return ParameterFormConfig(
            field_id=field_id,
            framework=framework,
            is_global_config_editing=True,
            global_config_type=global_config_type
        )
    
    @staticmethod
    def create_nested_config(field_id: str, parent_config: ParameterFormConfig) -> ParameterFormConfig:
        """
        Create a parameter form configuration for nested forms.
        
        Args:
            field_id: Unique identifier for the nested form
            parent_config: Configuration from the parent form
            
        Returns:
            Nested parameter form configuration inheriting parent settings
        """
        return ParameterFormConfig(
            field_id=field_id,
            framework=parent_config.framework,
            is_global_config_editing=parent_config.is_global_config_editing,
            global_config_type=parent_config.global_config_type,
            placeholder_prefix=parent_config.placeholder_prefix,
            use_scroll_area=False,  # Nested forms typically don't use scroll areas
            enable_debug=parent_config.enable_debug,
            debug_target_params=parent_config.debug_target_params,
            color_scheme=parent_config.color_scheme,
            function_target=parent_config.function_target
        )
    
    @staticmethod
    def create_pyqt_config(field_id: str, color_scheme: Any = None, 
                          function_target: Any = None, use_scroll_area: bool = True) -> ParameterFormConfig:
        """
        Create a parameter form configuration optimized for PyQt.
        
        Args:
            field_id: Unique identifier for the form
            color_scheme: Optional PyQt color scheme
            function_target: Optional function target for docstring fallback
            use_scroll_area: Whether to use scroll area
            
        Returns:
            PyQt-optimized parameter form configuration
        """
        return ParameterFormConfig(
            field_id=field_id,
            framework=CONSTANTS.PYQT6_FRAMEWORK,
            color_scheme=color_scheme,
            function_target=function_target,
            use_scroll_area=use_scroll_area
        )
    
    @staticmethod
    def create_textual_config(field_id: str, parameter_info: Optional[Dict] = None) -> ParameterFormConfig:
        """
        Create a parameter form configuration optimized for Textual.
        
        Args:
            field_id: Unique identifier for the form
            parameter_info: Optional parameter information dictionary
            
        Returns:
            Textual-optimized parameter form configuration
        """
        return ParameterFormConfig(
            field_id=field_id,
            framework=CONSTANTS.TEXTUAL_FRAMEWORK,
            parameter_info=parameter_info,
            use_scroll_area=False  # Textual doesn't use scroll areas
        )


# Convenience functions for common patterns

def builder(field_id: str) -> ParameterFormConfigBuilder:
    """Create a new configuration builder."""
    return ParameterFormConfigBuilder(field_id)


def basic_config(field_id: str, framework: str = CONSTANTS.TEXTUAL_FRAMEWORK) -> ParameterFormConfig:
    """Create a basic configuration."""
    return ParameterFormConfigFactory.create_basic_config(field_id, framework)


def debug_config(field_id: str, target_params: Set[str], 
                framework: str = CONSTANTS.TEXTUAL_FRAMEWORK) -> ParameterFormConfig:
    """Create a debug configuration."""
    return ParameterFormConfigFactory.create_debug_config(field_id, target_params, framework)


def pyqt_config(field_id: str, **kwargs) -> ParameterFormConfig:
    """Create a PyQt configuration."""
    return ParameterFormConfigFactory.create_pyqt_config(field_id, **kwargs)


def textual_config(field_id: str, **kwargs) -> ParameterFormConfig:
    """Create a Textual configuration."""
    return ParameterFormConfigFactory.create_textual_config(field_id, **kwargs)
