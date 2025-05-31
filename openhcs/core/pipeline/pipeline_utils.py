# openhcs/core/pipeline/pipeline_utils.py
"""Utility functions for the OpenHCS pipeline system."""
import re
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

def to_snake_case(name: str) -> str:
    """
    Converts a string from CamelCase, PascalCase, or mixedCase to snake_case.
    Examples: "MySpecialKey" -> "my_special_key", "HTTPRequest" -> "http_request"
    """
    if not name:
        return ""
    # Add underscore before uppercase letters that are preceded by a lowercase or digit
    s_intermediate = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Add underscore before an uppercase letter that is followed by a lowercase letter,
    # effectively breaking up words like "MyWord" into "My_Word" before lowercasing.
    # This also helps with acronyms at the beginning, e.g. "HTTPRequest" -> "HTTP_Request"
    s_final = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s_intermediate)
    # Re-apply the first rule to catch cases like "WordWord" -> "Word_Word" from s_final
    s_final_recheck = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s_final)
    return s_final_recheck.lower()