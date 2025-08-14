"""
Metaprogramming system for dynamic interface generation based on component enums.

This module provides a metaprogramming framework that dynamically generates interface
classes based on the contents of component enums, enabling truly generic component
processing without hardcoded method names or component assumptions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Generic, Optional, Set, Callable, Union
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Enum)


class MethodSignature:
    """Represents a dynamically generated method signature."""
    
    def __init__(self, name: str, return_type: Type = Any, **kwargs):
        self.name = name
        self.return_type = return_type
        self.parameters = kwargs
    
    def __repr__(self):
        params = ", ".join(f"{k}: {v.__name__ if hasattr(v, '__name__') else v}" 
                          for k, v in self.parameters.items())
        return f"{self.name}({params}) -> {self.return_type.__name__ if hasattr(self.return_type, '__name__') else self.return_type}"


class ComponentMethodRegistry:
    """Registry for tracking dynamically generated methods."""
    
    def __init__(self):
        self._methods: Dict[str, Dict[str, MethodSignature]] = {}
    
    def register_method(self, interface_name: str, method: MethodSignature):
        """Register a method for an interface."""
        if interface_name not in self._methods:
            self._methods[interface_name] = {}
        self._methods[interface_name][method.name] = method
        logger.debug(f"Registered method {method.name} for interface {interface_name}")
    
    def get_methods(self, interface_name: str) -> Dict[str, MethodSignature]:
        """Get all methods for an interface."""
        return self._methods.get(interface_name, {})
    
    def has_method(self, interface_name: str, method_name: str) -> bool:
        """Check if an interface has a specific method."""
        return (interface_name in self._methods and 
                method_name in self._methods[interface_name])


# Global method registry
_method_registry = ComponentMethodRegistry()


class DynamicInterfaceMeta(type):
    """
    Metaclass that dynamically generates interface methods based on component enums.
    
    This metaclass inspects the component enum during class creation and generates
    abstract methods for each component, enabling truly generic component processing.
    """
    
    def __new__(mcs, name, bases, namespace, component_enum=None, method_patterns=None, **kwargs):
        """
        Create a new interface class with dynamically generated methods.
        
        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace
            component_enum: Enum class to generate methods from
            method_patterns: List of method patterns to generate
            **kwargs: Additional arguments
        """
        # Default method patterns if not specified
        if method_patterns is None:
            method_patterns = ['process', 'validate', 'get_keys']
        
        # Generate methods if component_enum is provided
        if component_enum is not None:
            logger.info(f"Generating dynamic interface {name} for enum {component_enum.__name__}")
            mcs._generate_methods(namespace, component_enum, method_patterns, name)
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register the interface
        if component_enum is not None:
            cls._component_enum = component_enum
            cls._method_patterns = method_patterns
            logger.info(f"Created dynamic interface {name} with {len(_method_registry.get_methods(name))} methods")
        
        return cls
    
    @staticmethod
    def _generate_methods(namespace: Dict[str, Any], component_enum: Type[Enum], 
                         method_patterns: list, interface_name: str):
        """Generate abstract methods for each component and pattern combination."""
        for component in component_enum:
            component_name = component.value
            
            for pattern in method_patterns:
                method_name = f"{pattern}_{component_name}"
                
                # Create method signature
                signature = MethodSignature(
                    name=method_name,
                    return_type=Any,
                    context=Any,
                    **{f"{component_name}_value": str}
                )
                
                # Register the method
                _method_registry.register_method(interface_name, signature)
                
                # Create abstract method
                def create_abstract_method(method_name=method_name):
                    @abstractmethod
                    def abstract_method(self, context: Any, **kwargs) -> Any:
                        """Dynamically generated abstract method."""
                        raise NotImplementedError(f"Method {method_name} must be implemented")
                    
                    abstract_method.__name__ = method_name
                    abstract_method.__qualname__ = f"{interface_name}.{method_name}"
                    return abstract_method
                
                # Add method to namespace
                namespace[method_name] = create_abstract_method()
                logger.debug(f"Generated abstract method {method_name} for {interface_name}")


class ComponentProcessorInterface(metaclass=DynamicInterfaceMeta):
    """
    Base interface for component processors with dynamically generated methods.
    
    This class uses the DynamicInterfaceMeta metaclass to automatically generate
    abstract methods based on the component enum, providing a truly generic
    interface that adapts to any component configuration.
    """
    
    def __init__(self, component_enum: Type[T]):
        """
        Initialize the processor interface.
        
        Args:
            component_enum: The component enum to process
        """
        self.component_enum = component_enum
        self._validate_implementation()
    
    def _validate_implementation(self):
        """Validate that all required methods are implemented."""
        interface_name = self.__class__.__name__
        required_methods = _method_registry.get_methods(interface_name)
        
        for method_name, signature in required_methods.items():
            if not hasattr(self, method_name):
                raise NotImplementedError(
                    f"Class {self.__class__.__name__} must implement method {method_name}"
                )
            
            method = getattr(self, method_name)
            if not callable(method):
                raise TypeError(
                    f"Attribute {method_name} in {self.__class__.__name__} must be callable"
                )
        
        logger.debug(f"Validated implementation of {interface_name} with {len(required_methods)} methods")
    
    def get_available_methods(self) -> Dict[str, MethodSignature]:
        """Get all available methods for this interface."""
        return _method_registry.get_methods(self.__class__.__name__)
    
    def has_method_for_component(self, component: T, pattern: str) -> bool:
        """Check if a method exists for a specific component and pattern."""
        method_name = f"{pattern}_{component.value}"
        return _method_registry.has_method(self.__class__.__name__, method_name)
    
    def call_component_method(self, component: T, pattern: str, context: Any, **kwargs) -> Any:
        """Dynamically call a component-specific method."""
        method_name = f"{pattern}_{component.value}"
        
        if not hasattr(self, method_name):
            raise AttributeError(
                f"Method {method_name} not found in {self.__class__.__name__}"
            )
        
        method = getattr(self, method_name)
        return method(context, **kwargs)


class InterfaceGenerator:
    """
    Factory for creating component-specific interfaces dynamically.
    
    This class provides a high-level API for generating interfaces based on
    component enums, with caching and type safety features.
    """
    
    def __init__(self):
        self._interface_cache: Dict[str, Type] = {}
    
    def create_interface(self, 
                        component_enum: Type[T], 
                        interface_name: Optional[str] = None,
                        method_patterns: Optional[list] = None,
                        base_classes: Optional[tuple] = None) -> Type[ComponentProcessorInterface]:
        """
        Create a component-specific interface class.
        
        Args:
            component_enum: The component enum to generate interface for
            interface_name: Optional custom interface name
            method_patterns: Optional custom method patterns
            base_classes: Optional additional base classes
            
        Returns:
            Dynamically generated interface class
        """
        # Generate interface name if not provided
        if interface_name is None:
            interface_name = f"{component_enum.__name__}ProcessorInterface"
        
        # Check cache
        cache_key = f"{interface_name}_{id(component_enum)}"
        if cache_key in self._interface_cache:
            logger.debug(f"Returning cached interface {interface_name}")
            return self._interface_cache[cache_key]
        
        # Set default base classes
        if base_classes is None:
            base_classes = (ComponentProcessorInterface,)
        
        # Create the interface class dynamically
        interface_class = DynamicInterfaceMeta(
            interface_name,
            base_classes,
            {},
            component_enum=component_enum,
            method_patterns=method_patterns
        )
        
        # Cache the interface
        self._interface_cache[cache_key] = interface_class
        
        logger.info(f"Created interface {interface_name} for {component_enum.__name__}")
        return interface_class
    
    def get_cached_interface(self, interface_name: str) -> Optional[Type]:
        """Get a cached interface by name."""
        for key, interface in self._interface_cache.items():
            if key.startswith(interface_name):
                return interface
        return None
    
    def clear_cache(self):
        """Clear the interface cache."""
        self._interface_cache.clear()
        logger.debug("Cleared interface cache")


# Global interface generator instance
interface_generator = InterfaceGenerator()
