# File: openhcs/textual_tui/widgets/shared/signature_analyzer.py

import inspect
import dataclasses
from typing import Any, Dict, Callable, get_type_hints, NamedTuple, Union

class ParameterInfo(NamedTuple):
    """Information about a parameter."""
    name: str
    param_type: type
    default_value: Any
    is_required: bool

class SignatureAnalyzer:
    """Universal analyzer for extracting parameter information from any target."""
    
    @staticmethod
    def analyze(target: Union[Callable, type]) -> Dict[str, ParameterInfo]:
        """Extract parameter information from any target: function, constructor, or dataclass.
        
        Args:
            target: Function, constructor, or dataclass type
            
        Returns:
            Dict mapping parameter names to ParameterInfo
        """
        if not target:
            return {}
        
        # Dispatch based on target type
        if dataclasses.is_dataclass(target):
            return SignatureAnalyzer._analyze_dataclass(target)
        else:
            return SignatureAnalyzer._analyze_callable(target)
    
    @staticmethod
    def _analyze_callable(callable_obj: Callable) -> Dict[str, ParameterInfo]:
        """Extract parameter information from callable signature."""
        try:
            sig = inspect.signature(callable_obj)
            type_hints = get_type_hints(callable_obj)
            parameters = {}

            param_list = list(sig.parameters.items())

            for i, (param_name, param) in enumerate(param_list):
                # Skip self, cls, kwargs - parent can filter more if needed
                if param_name in ('self', 'cls', 'kwargs'):
                    continue

                # Skip the first parameter (after self/cls) - this is always the image/tensor
                # that gets passed automatically by the processing system
                if i == 0 or (i == 1 and param_list[0][0] in ('self', 'cls')):
                    continue
                
                param_type = type_hints.get(param_name, str)
                default_value = param.default if param.default != inspect.Parameter.empty else None
                is_required = param.default == inspect.Parameter.empty
                
                parameters[param_name] = ParameterInfo(
                    name=param_name,
                    param_type=param_type,
                    default_value=default_value,
                    is_required=is_required
                )
            
            return parameters
            
        except Exception:
            # Return empty dict on error
            return {}
    
    @staticmethod
    def _analyze_dataclass(dataclass_type: type) -> Dict[str, ParameterInfo]:
        """Extract parameter information from dataclass fields."""
        try:
            type_hints = get_type_hints(dataclass_type)
            parameters = {}
            
            for field in dataclasses.fields(dataclass_type):
                param_type = type_hints.get(field.name, str)
                
                # Get default value
                if field.default != dataclasses.MISSING:
                    default_value = field.default
                    is_required = False
                elif field.default_factory != dataclasses.MISSING:
                    default_value = field.default_factory()
                    is_required = False
                else:
                    default_value = None
                    is_required = True
                
                parameters[field.name] = ParameterInfo(
                    name=field.name,
                    param_type=param_type,
                    default_value=default_value,
                    is_required=is_required
                )
            
            return parameters
            
        except Exception:
            # Return empty dict on error
            return {}
