"""Custom function registration system for OpenHCS.

This module enables users to define custom processing functions via code editor
and have them automatically registered in the function registry. Custom functions
are persisted to disk and auto-loaded on startup.

Core Components:
    - CustomFunctionManager: Manages custom function lifecycle (register, load, delete)
    - ValidationError: Exception raised for invalid custom function code
    - get_default_template: Returns default numpy template for custom functions
    - get_analysis_template: Returns template for analysis functions with @artifact_outputs

Example (Processing):
    >>> from openhcs.processing.custom_functions import CustomFunctionManager
    >>> manager = CustomFunctionManager()
    >>> code = '''
    ... from openhcs.core.memory import numpy
    ...
    ... @numpy
    ... def my_function(image, scale=1.0):
    ...     return image * scale
    ... '''
    >>> funcs = manager.register_from_code(code, persist=True)

Example (Analysis with special outputs):
    >>> from openhcs.processing.custom_functions import get_analysis_template
    >>> template = get_analysis_template()  # Shows @artifact_outputs pattern
"""

import threading

from openhcs.processing.custom_functions.manager import CustomFunctionManager
from openhcs.processing.custom_functions.templates import (
    AVAILABLE_MEMORY_TYPES,
    AVAILABLE_TEMPLATE_CATEGORIES,
    get_analysis_template,
    get_default_template,
    get_multi_output_template,
    get_template_for_memory_type,
)
from openhcs.processing.custom_functions.validation import ValidationError

_CUSTOM_FUNCTION_LOAD_STATE = threading.local()

__all__ = [
    "CustomFunctionManager",
    "ValidationError",
    "get_default_template",
    "get_template_for_memory_type",
    "get_analysis_template",
    "get_multi_output_template",
    "AVAILABLE_MEMORY_TYPES",
    "AVAILABLE_TEMPLATE_CATEGORIES",
]


def __getattr__(name: str):
    loading_names = getattr(_CUSTOM_FUNCTION_LOAD_STATE, "names", frozenset())
    if name.startswith("_") or name in loading_names:
        raise AttributeError(
            f"module 'openhcs.processing.custom_functions' has no attribute '{name}'"
        )

    _load_custom_function_export(name)
    try:
        return globals()[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'openhcs.processing.custom_functions' has no attribute '{name}'"
        ) from exc


def _load_custom_function_export(name: str) -> None:
    if name in globals():
        return
    manager = CustomFunctionManager()
    file_path = manager.storage_dir / f"{name}.py"
    if not file_path.exists():
        return

    previous_names = getattr(_CUSTOM_FUNCTION_LOAD_STATE, "names", frozenset())
    _CUSTOM_FUNCTION_LOAD_STATE.names = previous_names | {name}
    try:
        manager.load_custom_function(name, publish_only_if_missing=True)
    finally:
        _CUSTOM_FUNCTION_LOAD_STATE.names = previous_names
