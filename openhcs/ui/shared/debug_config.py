"""
Debug configuration system for parameter form managers.

This module provides a clean, configurable debug logging system that can be
enabled/disabled and targeted to specific parameters, eliminating debug pollution
in production code while maintaining powerful debugging capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Set, Optional, Any, Dict
from contextlib import contextmanager

from openhcs.ui.shared.parameter_form_constants import CONSTANTS
from openhcs.ui.shared.parameter_name_formatter import ParameterNameFormatter


@dataclass
class DebugConfig:
    """
    Configuration for parameter form debug logging.
    
    This dataclass encapsulates all debug configuration options, allowing
    fine-grained control over what gets logged and how debug messages are
    formatted.
    
    Attributes:
        enabled: Whether debug logging is enabled globally
        target_params: Set of parameter names to debug (empty = debug all)
        log_prefix: Prefix for all debug log messages
        log_level: Logging level for debug messages
        include_stack_trace: Whether to include stack traces in debug logs
        max_value_length: Maximum length for logged parameter values
    """
    enabled: bool = False
    target_params: Set[str] = field(default_factory=lambda: set(CONSTANTS.DEBUG_TARGET_PARAMS))
    log_prefix: str = "PARAM_FORM_DEBUG"
    log_level: int = logging.INFO
    include_stack_trace: bool = False
    max_value_length: int = 200
    
    def should_debug_param(self, param_name: str) -> bool:
        """
        Check if a specific parameter should be debugged.
        
        Args:
            param_name: The parameter name to check
            
        Returns:
            True if debugging is enabled and parameter is targeted
        """
        if not self.enabled:
            return False
        
        # If no specific targets, debug all parameters
        if not self.target_params:
            return True
        
        return param_name in self.target_params
    
    def add_target_param(self, param_name: str) -> None:
        """Add a parameter to the debug target list."""
        self.target_params.add(param_name)
    
    def remove_target_param(self, param_name: str) -> None:
        """Remove a parameter from the debug target list."""
        self.target_params.discard(param_name)
    
    def clear_target_params(self) -> None:
        """Clear all target parameters (debug all when enabled)."""
        self.target_params.clear()


class ParameterFormDebugger:
    """
    Centralized debug logging system for parameter form managers.
    
    This class provides a clean interface for debug logging that can be
    easily enabled/disabled and configured without polluting production code
    with debug statements.
    """
    
    def __init__(self, config: DebugConfig = None, logger_name: str = None):
        """
        Initialize the debugger with configuration.
        
        Args:
            config: Debug configuration (creates default if None)
            logger_name: Logger name (uses module name if None)
        """
        self.config = config or DebugConfig()
        self.logger = logging.getLogger(logger_name or __name__)
    
    def log_parameter_update(self, param_name: str, value: Any, context: str = None) -> None:
        """
        Log a parameter update operation.
        
        Args:
            param_name: The parameter being updated
            value: The new value
            context: Optional context information
        """
        if not self.config.should_debug_param(param_name):
            return
        
        formatted_value = self._format_value(value)
        context_str = f" [{context}]" if context else ""
        
        message = f"{self.config.log_prefix} - Parameter Update{context_str}: {param_name} = {formatted_value}"
        self._log_message(message)
    
    def log_nested_update(self, parent_name: str, nested_name: str, value: Any) -> None:
        """
        Log a nested parameter update operation.
        
        Args:
            parent_name: The parent parameter name
            nested_name: The nested parameter name
            value: The new value
        """
        full_param_name = f"{parent_name}.{nested_name}"
        
        if not self.config.should_debug_param(parent_name) and not self.config.should_debug_param(nested_name):
            return
        
        formatted_value = self._format_value(value)
        message = f"{self.config.log_prefix} - Nested Update: {full_param_name} = {formatted_value}"
        self._log_message(message)
    
    def log_reset_operation(self, param_name: str, old_value: Any, new_value: Any) -> None:
        """
        Log a parameter reset operation.
        
        Args:
            param_name: The parameter being reset
            old_value: The previous value
            new_value: The reset value
        """
        if not self.config.should_debug_param(param_name):
            return
        
        old_formatted = self._format_value(old_value)
        new_formatted = self._format_value(new_value)
        
        message = f"{self.config.log_prefix} - Reset: {param_name} {old_formatted} -> {new_formatted}"
        self._log_message(message)
    
    def log_widget_creation(self, param_name: str, widget_type: str, widget_id: str) -> None:
        """
        Log widget creation for a parameter.
        
        Args:
            param_name: The parameter name
            widget_type: The type of widget created
            widget_id: The widget ID
        """
        if not self.config.should_debug_param(param_name):
            return
        
        message = f"{self.config.log_prefix} - Widget Created: {param_name} -> {widget_type} (id: {widget_id})"
        self._log_message(message)
    
    def log_form_manager_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
        """
        Log general form manager operations.
        
        Args:
            operation: The operation being performed
            details: Optional details dictionary
        """
        if not self.config.enabled:
            return
        
        details_str = ""
        if details:
            formatted_details = {k: self._format_value(v) for k, v in details.items()}
            details_str = f" - {formatted_details}"
        
        message = f"{self.config.log_prefix} - Operation: {operation}{details_str}"
        self._log_message(message)
    
    def log_error(self, param_name: str, error: Exception, context: str = None) -> None:
        """
        Log an error related to parameter processing.
        
        Args:
            param_name: The parameter where the error occurred
            error: The exception that occurred
            context: Optional context information
        """
        if not self.config.should_debug_param(param_name):
            return
        
        context_str = f" [{context}]" if context else ""
        message = f"{self.config.log_prefix} - Error{context_str}: {param_name} - {error}"
        
        self.logger.error(message)
        
        if self.config.include_stack_trace:
            self.logger.exception("Stack trace:")
    
    def log_validation_result(self, param_name: str, is_valid: bool, validation_message: str = None) -> None:
        """
        Log parameter validation results.
        
        Args:
            param_name: The parameter being validated
            is_valid: Whether validation passed
            validation_message: Optional validation message
        """
        if not self.config.should_debug_param(param_name):
            return
        
        status = "VALID" if is_valid else "INVALID"
        message = f"{self.config.log_prefix} - Validation: {param_name} -> {status}"
        
        if validation_message:
            message += f" ({validation_message})"
        
        self._log_message(message)
    
    @contextmanager
    def debug_session(self, session_name: str):
        """
        Context manager for debug sessions with automatic cleanup.
        
        Args:
            session_name: Name of the debug session
        """
        if self.config.enabled:
            self._log_message(f"{self.config.log_prefix} - Session START: {session_name}")
        
        try:
            yield self
        except Exception as e:
            if self.config.enabled:
                self._log_message(f"{self.config.log_prefix} - Session ERROR: {session_name} - {e}")
            raise
        finally:
            if self.config.enabled:
                self._log_message(f"{self.config.log_prefix} - Session END: {session_name}")
    
    def _format_value(self, value: Any) -> str:
        """
        Format a value for logging with length limits.
        
        Args:
            value: The value to format
            
        Returns:
            Formatted string representation
        """
        if value is None:
            return "None"
        
        # Handle special cases
        if hasattr(value, '__dataclass_fields__'):
            return f"<dataclass {type(value).__name__}>"
        
        if hasattr(value, '_resolve_field_value'):
            return f"<lazy_dataclass {type(value).__name__}>"
        
        # Convert to string and truncate if needed
        str_value = str(value)
        if len(str_value) > self.config.max_value_length:
            truncated = str_value[:self.config.max_value_length - 3]
            return f"{truncated}..."
        
        return str_value
    
    def _log_message(self, message: str) -> None:
        """
        Log a message at the configured level.
        
        Args:
            message: The message to log
        """
        self.logger.log(self.config.log_level, message)


# Global debug configuration and debugger instances
DEFAULT_DEBUG_CONFIG = DebugConfig()
DEFAULT_DEBUGGER = ParameterFormDebugger(DEFAULT_DEBUG_CONFIG)


def get_debugger(config: DebugConfig = None) -> ParameterFormDebugger:
    """
    Get a debugger instance with optional custom configuration.
    
    Args:
        config: Optional custom debug configuration
        
    Returns:
        ParameterFormDebugger instance
    """
    if config is None:
        return DEFAULT_DEBUGGER
    
    return ParameterFormDebugger(config)


def enable_debug(target_params: Set[str] = None) -> None:
    """
    Enable debug logging globally.
    
    Args:
        target_params: Optional set of parameters to target (None = all)
    """
    DEFAULT_DEBUG_CONFIG.enabled = True
    if target_params is not None:
        DEFAULT_DEBUG_CONFIG.target_params = target_params


def disable_debug() -> None:
    """Disable debug logging globally."""
    DEFAULT_DEBUG_CONFIG.enabled = False


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled globally."""
    return DEFAULT_DEBUG_CONFIG.enabled
