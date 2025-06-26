"""Unified parameter analysis interface for all parameter sources in OpenHCS TUI.

This module provides a single, consistent interface for analyzing parameters from:
- Functions and methods
- Dataclasses and their fields
- Nested dataclass structures
- Any callable or type with parameters

Replaces the fragmented approach of SignatureAnalyzer vs FieldIntrospector.
"""

import inspect
import dataclasses
from typing import Dict, Union, Callable, Type, Any, Optional
from dataclasses import dataclass

from .signature_analyzer import SignatureAnalyzer, ParameterInfo, DocstringExtractor


@dataclass
class UnifiedParameterInfo:
    """Unified parameter information that works for all parameter sources."""
    name: str
    param_type: Type
    default_value: Any
    is_required: bool
    description: Optional[str] = None
    source_type: str = "unknown"  # "function", "dataclass", "nested"
    
    @classmethod
    def from_parameter_info(cls, param_info: ParameterInfo, source_type: str = "function") -> "UnifiedParameterInfo":
        """Convert from existing ParameterInfo to unified format."""
        return cls(
            name=param_info.name,
            param_type=param_info.param_type,
            default_value=param_info.default_value,
            is_required=param_info.is_required,
            description=param_info.description,
            source_type=source_type
        )


class UnifiedParameterAnalyzer:
    """Single interface for analyzing parameters from any source.
    
    This class provides a unified way to extract parameter information
    from functions, dataclasses, and other parameter sources, ensuring
    consistent behavior across the entire application.
    """
    
    @staticmethod
    def analyze(target: Union[Callable, Type, object]) -> Dict[str, UnifiedParameterInfo]:
        """Analyze parameters from any source.
        
        Args:
            target: Function, method, dataclass type, or instance to analyze
            
        Returns:
            Dictionary mapping parameter names to UnifiedParameterInfo objects
            
        Examples:
            # Function analysis
            param_info = UnifiedParameterAnalyzer.analyze(my_function)
            
            # Dataclass analysis
            param_info = UnifiedParameterAnalyzer.analyze(MyDataclass)
            
            # Instance analysis
            param_info = UnifiedParameterAnalyzer.analyze(my_instance)
        """
        if target is None:
            return {}
            
        # Determine the type of target and route to appropriate analyzer
        if inspect.isfunction(target) or inspect.ismethod(target):
            return UnifiedParameterAnalyzer._analyze_callable(target)
        elif inspect.isclass(target):
            if dataclasses.is_dataclass(target):
                return UnifiedParameterAnalyzer._analyze_dataclass_type(target)
            else:
                # Try to analyze constructor
                return UnifiedParameterAnalyzer._analyze_callable(target.__init__)
        elif dataclasses.is_dataclass(target):
            # Instance of dataclass
            return UnifiedParameterAnalyzer._analyze_dataclass_instance(target)
        else:
            # Try to analyze as callable
            if callable(target):
                return UnifiedParameterAnalyzer._analyze_callable(target)
            else:
                return {}
    
    @staticmethod
    def _analyze_callable(callable_obj: Callable) -> Dict[str, UnifiedParameterInfo]:
        """Analyze a callable (function, method, etc.)."""
        try:
            # Use existing SignatureAnalyzer for callables
            param_info_dict = SignatureAnalyzer.analyze(callable_obj)
            
            # Convert to unified format
            unified_params = {}
            for name, param_info in param_info_dict.items():
                unified_params[name] = UnifiedParameterInfo.from_parameter_info(
                    param_info, 
                    source_type="function"
                )
            
            return unified_params
            
        except Exception:
            return {}
    
    @staticmethod
    def _analyze_dataclass_type(dataclass_type: Type) -> Dict[str, UnifiedParameterInfo]:
        """Analyze a dataclass type."""
        try:
            # Extract docstring information
            docstring_info = DocstringExtractor.extract(dataclass_type)
            
            # Get field information
            fields = dataclasses.fields(dataclass_type)
            unified_params = {}
            
            for field in fields:
                # Get field description from docstring
                field_description = docstring_info.parameters.get(field.name)
                
                # Determine if field is required
                is_required = field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING
                
                # Get default value
                if field.default != dataclasses.MISSING:
                    default_value = field.default
                elif field.default_factory != dataclasses.MISSING:
                    default_value = field.default_factory()
                else:
                    default_value = None
                
                unified_params[field.name] = UnifiedParameterInfo(
                    name=field.name,
                    param_type=field.type,
                    default_value=default_value,
                    is_required=is_required,
                    description=field_description,
                    source_type="dataclass"
                )
            
            return unified_params
            
        except Exception:
            return {}
    
    @staticmethod
    def _analyze_dataclass_instance(instance: object) -> Dict[str, UnifiedParameterInfo]:
        """Analyze a dataclass instance."""
        try:
            # Get the type and analyze it
            dataclass_type = type(instance)
            unified_params = UnifiedParameterAnalyzer._analyze_dataclass_type(dataclass_type)
            
            # Update default values with current instance values
            for name, param_info in unified_params.items():
                if hasattr(instance, name):
                    current_value = getattr(instance, name)
                    # Create new UnifiedParameterInfo with current value as default
                    unified_params[name] = UnifiedParameterInfo(
                        name=param_info.name,
                        param_type=param_info.param_type,
                        default_value=current_value,
                        is_required=param_info.is_required,
                        description=param_info.description,
                        source_type="dataclass_instance"
                    )
            
            return unified_params
            
        except Exception:
            return {}
    
    @staticmethod
    def analyze_nested(target: Union[Callable, Type, object], parent_info: Dict[str, UnifiedParameterInfo] = None) -> Dict[str, UnifiedParameterInfo]:
        """Analyze parameters with nested dataclass support.
        
        This method provides enhanced analysis that can handle nested dataclasses
        and maintain parent context information.
        
        Args:
            target: The target to analyze
            parent_info: Optional parent parameter information for context
            
        Returns:
            Dictionary of unified parameter information with nested support
        """
        base_params = UnifiedParameterAnalyzer.analyze(target)
        
        # For each parameter, check if it's a nested dataclass
        enhanced_params = {}
        for name, param_info in base_params.items():
            enhanced_params[name] = param_info
            
            # If this parameter is a dataclass, mark it as having nested structure
            if dataclasses.is_dataclass(param_info.param_type):
                # Update source type to indicate nesting capability
                enhanced_params[name] = UnifiedParameterInfo(
                    name=param_info.name,
                    param_type=param_info.param_type,
                    default_value=param_info.default_value,
                    is_required=param_info.is_required,
                    description=param_info.description,
                    source_type=f"{param_info.source_type}_nested"
                )
        
        return enhanced_params


# Backward compatibility aliases
# These allow existing code to continue working while migration happens
ParameterAnalyzer = UnifiedParameterAnalyzer
analyze_parameters = UnifiedParameterAnalyzer.analyze
