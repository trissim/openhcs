"""
Metaprogramming system for dynamic parser interface generation.

This module applies metaprogramming to the parser system, generating parser interfaces
dynamically based on VariableComponents enum contents. This eliminates hardcoded
assumptions about component names and makes the parser system truly generic.
"""

import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Type, TypeVar, Optional, Union, Tuple
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Enum)


class ParserMethodRegistry:
    """Registry for tracking dynamically generated parser methods."""
    
    def __init__(self):
        self._methods: Dict[str, Dict[str, str]] = {}
        self._component_enums: Dict[str, Type[Enum]] = {}
    
    def register_parser_interface(self, interface_name: str, component_enum: Type[Enum]):
        """Register a parser interface with its component enum."""
        self._component_enums[interface_name] = component_enum
        self._methods[interface_name] = {}
        
        # Generate method names for each component
        for component in component_enum:
            component_name = component.value
            
            # Generate parse method name: parse_well, parse_site, etc.
            parse_method = f"parse_{component_name}"
            self._methods[interface_name][parse_method] = f"Parse {component_name} from filename"
            
            # Generate construct method name: construct_with_well, construct_with_site, etc.
            construct_method = f"construct_with_{component_name}"
            self._methods[interface_name][construct_method] = f"Construct filename with {component_name}"
        
        logger.debug(f"Registered parser interface {interface_name} with {len(self._methods[interface_name])} methods")
    
    def get_methods(self, interface_name: str) -> Dict[str, str]:
        """Get all methods for a parser interface."""
        return self._methods.get(interface_name, {})
    
    def get_component_enum(self, interface_name: str) -> Optional[Type[Enum]]:
        """Get the component enum for a parser interface."""
        return self._component_enums.get(interface_name)


# Global parser method registry
_parser_registry = ParserMethodRegistry()


class DynamicParserMeta(ABCMeta):
    """
    Metaclass that dynamically generates parser interface methods based on component enums.
    
    This metaclass creates component-specific parsing and construction methods, enabling
    truly generic parser interfaces that adapt to any component configuration.
    """
    
    def __new__(mcs, name, bases, namespace, component_enum=None, **kwargs):
        """
        Create a new parser interface class with dynamically generated methods.
        
        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace
            component_enum: Enum class to generate methods from
            **kwargs: Additional arguments
        """
        # Generate methods if component_enum is provided
        if component_enum is not None:
            logger.info(f"Generating dynamic parser interface {name} for enum {component_enum.__name__}")
            mcs._generate_parser_methods(namespace, component_enum, name)
            
            # Register the interface
            _parser_registry.register_parser_interface(name, component_enum)
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Store metadata on the class
        if component_enum is not None:
            cls._component_enum = component_enum
            cls.FILENAME_COMPONENTS = [component.value for component in component_enum] + ['extension']
            logger.info(f"Created dynamic parser interface {name} with {len(_parser_registry.get_methods(name))} methods")
        
        return cls
    
    @staticmethod
    def _generate_parser_methods(namespace: Dict[str, Any], component_enum: Type[Enum], interface_name: str):
        """Generate abstract parser methods for each component."""
        
        # Generate generic parse_filename method that returns all components
        def create_parse_filename_method():
            @abstractmethod
            def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
                """
                Parse a filename to extract all components.
                
                Returns a dictionary with keys matching component enum values plus 'extension'.
                """
                raise NotImplementedError("parse_filename must be implemented")
            return parse_filename
        
        namespace['parse_filename'] = create_parse_filename_method()
        
        # Generate generic construct_filename method with **kwargs for all components
        def create_construct_filename_method():
            @abstractmethod
            def construct_filename(self, extension: str = '.tif', **component_values) -> str:
                """
                Construct a filename from component values.
                
                Args:
                    extension: File extension
                    **component_values: Component values as keyword arguments
                    
                Returns:
                    Constructed filename string
                """
                raise NotImplementedError("construct_filename must be implemented")
            return construct_filename
        
        namespace['construct_filename'] = create_construct_filename_method()
        
        # Generate component-specific validation methods
        for component in component_enum:
            component_name = component.value
            
            # Generate validate_{component} method
            def create_validate_method(comp_name=component_name):
                @abstractmethod
                def validate_component(self, value: Any) -> bool:
                    f"""Validate {comp_name} component value."""
                    raise NotImplementedError(f"validate_{comp_name} must be implemented")
                
                validate_component.__name__ = f"validate_{comp_name}"
                validate_component.__qualname__ = f"{interface_name}.validate_{comp_name}"
                return validate_component
            
            namespace[f"validate_{component_name}"] = create_validate_method()
            
            # Generate extract_{component} method for component-specific extraction
            def create_extract_method(comp_name=component_name):
                @abstractmethod
                def extract_component(self, filename: str) -> Optional[Any]:
                    f"""Extract {comp_name} component from filename."""
                    raise NotImplementedError(f"extract_{comp_name} must be implemented")
                
                extract_component.__name__ = f"extract_{comp_name}"
                extract_component.__qualname__ = f"{interface_name}.extract_{comp_name}"
                return extract_component
            
            namespace[f"extract_{component_name}"] = create_extract_method()


class GenericFilenameParser(ABC):
    """
    Generic base class for filename parsers with dynamically generated methods.

    This class provides the foundation for truly generic parser interfaces that
    adapt to any component configuration without hardcoded assumptions.
    """

    def __init__(self, component_enum: Type[T]):
        """
        Initialize the generic parser.

        Args:
            component_enum: The component enum this parser handles
        """
        self.component_enum = component_enum
        self.FILENAME_COMPONENTS = [component.value for component in component_enum] + ['extension']
        self.PLACEHOLDER_PATTERN = '{iii}'
        self._generate_dynamic_methods()

    def _generate_dynamic_methods(self):
        """
        Generate dynamic validation and extraction methods for each component.

        Creates methods that can be properly pickled by resolving them before serialization.
        """
        for component in self.component_enum:
            component_name = component.value

            # Create validator and extractor methods
            validator = self._create_generic_validator(component)
            extractor = self._create_generic_extractor(component)

            # Set methods on instance for direct access
            setattr(self, f"validate_{component_name}", validator)
            setattr(self, f"extract_{component_name}", extractor)

    def _create_generic_validator(self, component: Enum):
        """
        Create a generic validator for a component based on enum metadata.

        This approach uses the component enum itself to determine validation rules,
        making it truly generic and adaptable to any component configuration.
        """
        component_name = component.value

        # Define validation rules based on component enum metadata
        # This is generic and doesn't hardcode specific component names
        def validate_component(value: Any) -> bool:
            """Generic validation for any component value."""
            if value is None:
                return True  # Allow None values (placeholders)

            # Generic validation based on value type and placeholder patterns
            if isinstance(value, str):
                # String values: allow non-empty strings or placeholder patterns
                return len(value) > 0 or '{' in value
            elif isinstance(value, int):
                # Integer values: allow positive integers
                return value >= 0
            else:
                # Other types: allow any value (extensible for future component types)
                return True

        return validate_component

    def _create_generic_extractor(self, component: Enum):
        """
        Create a generic extractor for a component based on enum metadata.

        This approach uses the component enum to create extractors that work
        with any component configuration without hardcoded assumptions.
        """
        component_name = component.value

        def extract_component(filename: str) -> Optional[Any]:
            """Generic extraction for any component using parse_filename."""
            parsed = self.parse_filename(filename)
            if parsed and component_name in parsed:
                return parsed[component_name]
            return None

        return extract_component

    @classmethod
    @abstractmethod
    def can_parse(cls, filename: str) -> bool:
        """Check if this parser can parse the given filename."""
        pass

    @abstractmethod
    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """Extract coordinates from component identifier (typically well)."""
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse a filename to extract all components."""
        pass

    @abstractmethod
    def construct_filename(self, extension: str = '.tif', **component_values) -> str:
        """Construct a filename from component values."""
        pass

    def __getstate__(self):
        """
        Custom pickling method to handle dynamic functions.

        Removes dynamic methods before pickling since they can't be serialized,
        but preserves the component_enum so they can be regenerated.
        """
        state = self.__dict__.copy()

        # Remove dynamic methods that can't be pickled
        dynamic_methods = []
        for component in self.component_enum:
            component_name = component.value
            validate_method = f"validate_{component_name}"
            extract_method = f"extract_{component_name}"

            if validate_method in state:
                dynamic_methods.append(validate_method)
                del state[validate_method]
            if extract_method in state:
                dynamic_methods.append(extract_method)
                del state[extract_method]

        # Store the list of removed methods for restoration
        state['_removed_dynamic_methods'] = dynamic_methods
        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to regenerate dynamic functions.

        Restores the object state and regenerates the dynamic methods
        that were removed during pickling.
        """
        # Restore the object state
        self.__dict__.update(state)

        # Remove the temporary list
        if '_removed_dynamic_methods' in self.__dict__:
            del self.__dict__['_removed_dynamic_methods']

        # Regenerate dynamic methods
        self._generate_dynamic_methods()
    
    def get_component_names(self) -> list:
        """Get all component names for this parser."""
        return [component.value for component in self.component_enum]

    def validate_component_by_name(self, component_name: str, value: Any) -> bool:
        """
        Validate a component value using the dynamic validation methods.

        Args:
            component_name: Name of the component to validate
            value: Value to validate

        Returns:
            True if the value is valid for the component
        """
        validate_method_name = f"validate_{component_name}"
        if hasattr(self, validate_method_name):
            validate_method = getattr(self, validate_method_name)
            return validate_method(value)
        else:
            # Fallback: allow any value for unknown components
            return True

    def extract_component_by_name(self, filename: str, component_name: str) -> Optional[Any]:
        """
        Extract a specific component from filename using dynamic extraction methods.

        Args:
            filename: Filename to parse
            component_name: Name of component to extract

        Returns:
            Component value or None if extraction fails

        Raises:
            AttributeError: If no extraction method exists for the component
        """
        extract_method_name = f"extract_{component_name}"
        if hasattr(self, extract_method_name):
            extract_method = getattr(self, extract_method_name)
            return extract_method(filename)
        else:
            raise AttributeError(
                f"No extraction method '{extract_method_name}' found. "
                f"Component '{component_name}' may not be in the component enum."
            )
    
    def validate_component_dict(self, components: Dict[str, Any]) -> bool:
        """
        Validate that a component dictionary contains all required components.
        
        Args:
            components: Dictionary of component values
            
        Returns:
            True if all required components are present and valid
        """
        required_components = set(self.get_component_names())
        provided_components = set(components.keys()) - {'extension'}
        
        # Check if all required components are provided
        if not required_components.issubset(provided_components):
            missing = required_components - provided_components
            logger.warning(f"Missing required components: {missing}")
            return False
        
        # Validate each component using the generic validation system
        for component_name, value in components.items():
            if component_name == 'extension':
                continue

            if not self.validate_component_by_name(component_name, value):
                logger.warning(f"Invalid value for {component_name}: {value}")
                return False
        
        return True
    



class ParserInterfaceGenerator:
    """
    Factory for creating component-specific parser interfaces dynamically.
    
    This class provides a high-level API for generating parser interfaces based on
    component enums, with caching and backward compatibility features.
    """
    
    def __init__(self):
        self._interface_cache: Dict[str, Type] = {}
    
    def create_parser_interface(self, 
                               component_enum: Type[T], 
                               interface_name: Optional[str] = None,
                               base_classes: Optional[tuple] = None) -> Type[GenericFilenameParser]:
        """
        Create a component-specific parser interface class.
        
        Args:
            component_enum: The component enum to generate interface for
            interface_name: Optional custom interface name
            base_classes: Optional additional base classes
            
        Returns:
            Dynamically generated parser interface class
        """
        # Generate interface name if not provided
        if interface_name is None:
            interface_name = f"{component_enum.__name__}FilenameParser"
        
        # Check cache
        cache_key = f"{interface_name}_{id(component_enum)}"
        if cache_key in self._interface_cache:
            logger.debug(f"Returning cached parser interface {interface_name}")
            return self._interface_cache[cache_key]
        
        # Set default base classes
        if base_classes is None:
            base_classes = (GenericFilenameParser,)
        
        # Create the interface class dynamically
        interface_class = DynamicParserMeta(
            interface_name,
            base_classes,
            {},
            component_enum=component_enum
        )
        
        # Cache the interface
        self._interface_cache[cache_key] = interface_class
        
        logger.info(f"Created parser interface {interface_name} for {component_enum.__name__}")
        return interface_class
    
    def get_cached_interface(self, interface_name: str) -> Optional[Type]:
        """Get a cached parser interface by name."""
        for key, interface in self._interface_cache.items():
            if key.startswith(interface_name):
                return interface
        return None
    
    def clear_cache(self):
        """Clear the parser interface cache."""
        self._interface_cache.clear()
        logger.debug("Cleared parser interface cache")


# Global parser interface generator instance
parser_interface_generator = ParserInterfaceGenerator()
