"""
Static analysis utilities to replace schema dependencies.

Provides pure functions for parameter introspection and type analysis,
replacing the schema-based approach with direct inspection of classes and functions.
"""

import inspect
import logging
from typing import Any, Dict, List, Callable, Optional, get_type_hints, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def get_abstractstep_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Extract AbstractStep.__init__ parameters via introspection.
    
    Returns:
        Dict mapping parameter names to their metadata:
        {
            'param_name': {
                'type': annotation,
                'default': default_value,
                'required': bool,
                'kind': parameter_kind
            }
        }
    """
    try:
        from openhcs.core.steps.abstract import AbstractStep
        
        sig = inspect.signature(AbstractStep.__init__)
        type_hints = get_type_hints(AbstractStep.__init__)
        params = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            params[name] = {
                'type': type_hints.get(name, param.annotation),
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty,
                'kind': param.kind.name
            }
        
        logger.debug(f"Extracted {len(params)} AbstractStep parameters")
        return params
        
    except Exception as e:
        logger.error(f"Failed to extract AbstractStep parameters: {e}")
        return {}

def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    Extract function signature for parameter form generation.
    
    Args:
        func: Function to analyze
        
    Returns:
        Dict containing parameter metadata and return type
    """
    try:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        params = {}
        for name, param in sig.parameters.items():
            params[name] = {
                'type': type_hints.get(name, param.annotation),
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty,
                'kind': param.kind.name
            }
        
        return {
            'name': getattr(func, '__name__', str(func)),
            'parameters': params,
            'return_type': type_hints.get('return', sig.return_annotation),
            'docstring': inspect.getdoc(func)
        }
        
    except Exception as e:
        logger.warning(f"Failed to extract signature for {func}: {e}")
        return {
            'name': getattr(func, '__name__', str(func)),
            'parameters': {},
            'error': str(e)
        }

def get_function_registry_by_backend() -> Dict[str, List[Callable]]:
    """
    Get FUNC_REGISTRY organized by backend for UI display.
    
    Returns:
        Dict mapping backend names to lists of functions
    """
    try:
        from openhcs.processing.func_registry import FUNC_REGISTRY
        return dict(FUNC_REGISTRY)
    except Exception as e:
        logger.error(f"Failed to access FUNC_REGISTRY: {e}")
        return {}

def analyze_function_pattern(pattern: Any) -> Dict[str, Any]:
    """
    Analyze a function pattern to extract metadata for UI display.
    
    Args:
        pattern: Function pattern (Callable, tuple, list, or dict)
        
    Returns:
        Dict containing pattern analysis results
    """
    try:
        if callable(pattern):
            return {
                'type': 'single_function',
                'functions': [get_function_signature(pattern)]
            }
        elif isinstance(pattern, tuple) and len(pattern) == 2:
            func, kwargs = pattern
            return {
                'type': 'function_with_kwargs',
                'functions': [get_function_signature(func)],
                'kwargs': kwargs
            }
        elif isinstance(pattern, list):
            functions = []
            for item in pattern:
                if callable(item):
                    functions.append(get_function_signature(item))
                elif isinstance(item, tuple):
                    func, _ = item
                    functions.append(get_function_signature(func))
            return {
                'type': 'function_list',
                'functions': functions
            }
        elif isinstance(pattern, dict):
            result = {'type': 'function_dict', 'keys': {}}
            for key, value in pattern.items():
                result['keys'][key] = analyze_function_pattern(value)
            return result
        else:
            return {'type': 'unknown', 'pattern': str(pattern)}
            
    except Exception as e:
        logger.error(f"Failed to analyze pattern {pattern}: {e}")
        return {'type': 'error', 'error': str(e)}

def get_type_widget_mapping() -> Dict[str, str]:
    """
    Get mapping of Python types to appropriate UI widget types.
    
    Returns:
        Dict mapping type annotations to widget type names
    """
    return {
        'str': 'text',
        'int': 'number',
        'float': 'number', 
        'bool': 'checkbox',
        'Path': 'path',
        'Optional[str]': 'text',
        'Optional[bool]': 'checkbox',
        'Optional[List[str]]': 'list',
        'Union[str, Path]': 'path',
        'List[str]': 'list'
    }
