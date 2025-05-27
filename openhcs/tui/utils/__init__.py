"""
Utility modules for hybrid TUI.

Provides pure functions for:
- Dialog operations (error dialogs, file dialogs)
- Static analysis (parameter introspection, schema replacement)
- File operations (load/save patterns, steps, pipelines)
"""

from .dialogs import show_error_dialog, prompt_for_path_dialog
from .static_analysis import (
    get_abstractstep_parameters,
    get_function_signature,
    get_function_registry_by_backend
)
from .file_operations import (
    load_func_pattern,
    save_func_pattern,
    load_step_file,
    save_step_file
)
from .safe_formatting import (
    safe_format,
    safe_text,
    SafeLabel,
    safe_error_label,
    safe_info_label,
    safe_status_label,
    error_building_component,
    unsupported_type_label,
    field_label
)

__all__ = [
    'show_error_dialog',
    'prompt_for_path_dialog',
    'get_abstractstep_parameters',
    'get_function_signature',
    'get_function_registry_by_backend',
    'load_func_pattern',
    'save_func_pattern',
    'load_step_file',
    'save_step_file',
    'safe_format',
    'safe_text',
    'SafeLabel',
    'safe_error_label',
    'safe_info_label',
    'safe_status_label',
    'error_building_component',
    'unsupported_type_label',
    'field_label'
]
