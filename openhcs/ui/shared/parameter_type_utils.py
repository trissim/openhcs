"""
Parameter type utilities for parameter form managers.

This module provides centralized type checking and resolution methods to eliminate
code duplication between PyQt and Textual parameter form implementations.
"""

import dataclasses
from typing import Dict, Optional, Type, Union, get_origin, get_args
from enum import Enum

from openhcs.ui.shared.parameter_form_constants import CONSTANTS


class ParameterTypeUtils:
    """
    Utility class for parameter type checking and resolution.
    
    This class provides static methods for common type operations used throughout
    the parameter form system, including Optional type handling, dataclass detection,
    and Union type resolution.
    """
    
    @staticmethod
    def is_optional_dataclass(param_type: Type) -> bool:
        """
        Check if parameter type is Optional[dataclass].
        
        This method determines whether a type annotation represents an optional
        dataclass parameter (Union[DataclassType, None]).
        
        Args:
            param_type: The type to check
            
        Returns:
            True if the type is Optional[dataclass], False otherwise
            
        Example:
            >>> from typing import Optional
            >>> @dataclass
            ... class Config: pass
            >>> ParameterTypeUtils.is_optional_dataclass(Optional[Config])
            True
            >>> ParameterTypeUtils.is_optional_dataclass(Config)
            False
        """
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            # Check if it's Optional (Union with None)
            if len(args) == 2 and type(None) in args:
                non_none_type = next(arg for arg in args if arg is not type(None))
                return dataclasses.is_dataclass(non_none_type)
        return False
    
    @staticmethod
    def get_optional_inner_type(param_type: Type) -> Type:
        """
        Extract the inner type from Optional[T].
        
        This method extracts the non-None type from an Optional type annotation.
        
        Args:
            param_type: The Optional type to extract from
            
        Returns:
            The inner type (T from Optional[T])
            
        Raises:
            ValueError: If the type is not Optional
            
        Example:
            >>> from typing import Optional
            >>> ParameterTypeUtils.get_optional_inner_type(Optional[str])
            <class 'str'>
        """
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                return next(arg for arg in args if arg is not type(None))
        
        raise ValueError(f"Type {param_type} is not Optional")
    
    @staticmethod
    def get_dataclass_type_for_param(param_name: str, parameter_types: Dict[str, Type]) -> Optional[Type]:
        """
        Get the dataclass type for a parameter, handling Optional types.
        
        This method retrieves the dataclass type for a parameter, automatically
        unwrapping Optional types to get the underlying dataclass.
        
        Args:
            param_name: The parameter name to look up
            parameter_types: Dictionary mapping parameter names to types
            
        Returns:
            The dataclass type, or None if parameter not found or not a dataclass
            
        Example:
            >>> types = {"config": Optional[MyConfig]}
            >>> ParameterTypeUtils.get_dataclass_type_for_param("config", types)
            <class 'MyConfig'>
        """
        if param_name not in parameter_types:
            return None
        
        param_type = parameter_types[param_name]
        
        # Handle Optional[dataclass] types
        if ParameterTypeUtils.is_optional_dataclass(param_type):
            return ParameterTypeUtils.get_optional_inner_type(param_type)
        
        # Handle direct dataclass types
        if dataclasses.is_dataclass(param_type):
            return param_type
        
        return None
    
    @staticmethod
    def resolve_union_type(param_type: Type) -> Type:
        """
        Resolve Union types to their primary type.
        
        This method handles Union types by extracting the primary (non-None) type.
        For Optional types, it returns the inner type. For other Union types,
        it returns the first non-None type.
        
        Args:
            param_type: The Union type to resolve
            
        Returns:
            The resolved primary type
            
        Example:
            >>> from typing import Union, Optional
            >>> ParameterTypeUtils.resolve_union_type(Optional[str])
            <class 'str'>
            >>> ParameterTypeUtils.resolve_union_type(Union[int, str])
            <class 'int'>
        """
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            # Filter out None type and return the first remaining type
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                return non_none_types[0]
        
        return param_type
    
    @staticmethod
    def is_enum_type(param_type: Type) -> bool:
        """
        Check if a type is an Enum type.
        
        Args:
            param_type: The type to check
            
        Returns:
            True if the type is an Enum, False otherwise
        """
        return (hasattr(param_type, CONSTANTS.BASES_ATTR) and 
                Enum in getattr(param_type, CONSTANTS.BASES_ATTR))
    
    @staticmethod
    def is_list_of_enums(param_type: Type) -> bool:
        """
        Check if parameter type is List[Enum].
        
        Args:
            param_type: The type to check
            
        Returns:
            True if the type is List[Enum], False otherwise
        """
        try:
            # Check if it's a generic type (like List[Something])
            if hasattr(param_type, '__origin__') and hasattr(param_type, '__args__'):
                origin = getattr(param_type, '__origin__')
                if origin is list:
                    args = getattr(param_type, '__args__')
                    if args:
                        inner_type = args[0]
                        return ParameterTypeUtils.is_enum_type(inner_type)
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_enum_from_list_type(param_type: Type) -> Optional[Type]:
        """
        Extract enum type from List[Enum] type.
        
        Args:
            param_type: The List[Enum] type
            
        Returns:
            The Enum type, or None if not a List[Enum]
        """
        try:
            if hasattr(param_type, '__origin__') and hasattr(param_type, '__args__'):
                origin = getattr(param_type, '__origin__')
                if origin is list:
                    args = getattr(param_type, '__args__')
                    if args and ParameterTypeUtils.is_enum_type(args[0]):
                        return args[0]
            return None
        except Exception:
            return None
    
    @staticmethod
    def has_dataclass_fields(obj: any) -> bool:
        """
        Check if an object has dataclass fields.
        
        Args:
            obj: The object to check
            
        Returns:
            True if the object has __dataclass_fields__ attribute
        """
        return hasattr(obj, CONSTANTS.DATACLASS_FIELDS_ATTR)
    
    @staticmethod
    def has_resolve_field_value(obj: any) -> bool:
        """
        Check if an object has the _resolve_field_value method (lazy dataclass).
        
        Args:
            obj: The object to check
            
        Returns:
            True if the object has _resolve_field_value attribute
        """
        return hasattr(obj, CONSTANTS.RESOLVE_FIELD_VALUE_ATTR)
    
    @staticmethod
    def is_concrete_dataclass(obj: any) -> bool:
        """
        Check if an object is a concrete (non-lazy) dataclass.
        
        Args:
            obj: The object to check
            
        Returns:
            True if the object is a concrete dataclass
        """
        return (ParameterTypeUtils.has_dataclass_fields(obj) and 
                not ParameterTypeUtils.has_resolve_field_value(obj))
    
    @staticmethod
    def is_lazy_dataclass(obj: any) -> bool:
        """
        Check if an object is a lazy dataclass.
        
        Args:
            obj: The object to check
            
        Returns:
            True if the object is a lazy dataclass
        """
        return ParameterTypeUtils.has_resolve_field_value(obj)
    
    @staticmethod
    def extract_value_attribute(obj: any) -> any:
        """
        Extract the value attribute from an object if it exists.
        
        This is commonly used for enum values and other wrapped types.
        
        Args:
            obj: The object to extract value from
            
        Returns:
            The value attribute if it exists, otherwise the original object
        """
        if hasattr(obj, CONSTANTS.VALUE_ATTR):
            return getattr(obj, CONSTANTS.VALUE_ATTR)
        return obj
    
    @staticmethod
    def convert_string_to_bool(value: str) -> bool:
        """
        Convert string to boolean using standard true/false patterns.
        
        Args:
            value: The string value to convert
            
        Returns:
            True if the string represents a true value, False otherwise
        """
        return value.lower() in CONSTANTS.TRUE_STRINGS
