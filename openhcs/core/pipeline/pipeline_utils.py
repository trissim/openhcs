# openhcs/core/pipeline/pipeline_utils.py
"""Utility functions for the OpenHCS pipeline system."""
from typing import Any, Callable, List, Optional, Tuple

def get_core_callable(func_pattern: Any) -> Optional[Callable[..., Any]]:
    """
    Extracts the first effective Python callable from a func_pattern.
    A func_pattern can be a direct callable, a (callable, kwargs) tuple,
    or a list (chain) where the first element is one of these types.
    """
    if callable(func_pattern) and not isinstance(func_pattern, type):
        # It's a direct callable (and not an uninstantiated class)
        return func_pattern
    elif isinstance(func_pattern, tuple) and func_pattern and \
         callable(func_pattern[0]) and \
         not isinstance(func_pattern[0], type):
        # It's a (callable, kwargs) tuple, ensure first element is a callable function
        return func_pattern[0]
    elif isinstance(func_pattern, list) and func_pattern:
        # It's a list (chain), recursively call for the first item
        return get_core_callable(func_pattern[0])
    return None

