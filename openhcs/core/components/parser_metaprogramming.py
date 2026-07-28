"""
Metaprogramming system for dynamic parser interface generation.

This module applies metaprogramming to the parser system, generating parser interfaces
dynamically based on VariableComponents enum contents. This eliminates hardcoded
assumptions about component names and makes the parser system truly generic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Type, TypeVar, Optional, Tuple, TypeAlias
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Enum)
FilenameParseValue: TypeAlias = str | int | float | bool | None
ComponentValidator: TypeAlias = Callable[[FilenameParseValue], bool]
ComponentExtractor: TypeAlias = Callable[[str], FilenameParseValue]


class FilenameParseResult(dict[str, FilenameParseValue]):
    """Nominal carrier for parsed filename component values."""


class MissingFilenameComponentError(ValueError):
    """A filename parser cannot construct a name without one component."""

    def __init__(self, component_name: str, *, empty: bool = False):
        self.component_name = component_name
        detail = "cannot be empty" if empty else "is required"
        super().__init__(f"Filename component {component_name!r} {detail}.")


def require_filename_component(
    component_values: Mapping[str, object],
    component_name: str,
) -> object:
    """Return a declared filename component or raise for an incomplete contract."""
    if component_name not in component_values:
        raise MissingFilenameComponentError(component_name)
    value = component_values[component_name]
    if value is None or value == "":
        raise MissingFilenameComponentError(component_name, empty=True)
    return value


def optional_filename_component(
    component_values: Mapping[str, object],
    component_name: str,
) -> object | None:
    """Return a declared optional filename component, or None when omitted."""
    if component_name not in component_values:
        return None
    value = component_values[component_name]
    if value is None or value == "":
        return None
    return value


def format_filename_component(value: object, padding: int = 0) -> str:
    """Format a declared filename component without inventing missing values."""
    if isinstance(value, str):
        return value
    return f"{int(value):0{padding}d}" if padding else str(int(value))


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
        Generate validation and extraction authorities for each component.

        The methods are stored in explicit maps instead of instance attributes so
        parser behavior is keyed by declared component identity, not reflection.
        """
        self._component_validators: dict[str, ComponentValidator] = {}
        self._component_extractors: dict[str, ComponentExtractor] = {}
        for component in self.component_enum:
            component_name = component.value
            self._component_validators[component_name] = self._create_generic_validator(component)
            self._component_extractors[component_name] = self._create_generic_extractor(component)

    def _create_generic_validator(self, component: Enum) -> ComponentValidator:
        """
        Create a generic validator for a component based on enum metadata.

        This approach uses the component enum itself to determine validation rules,
        making it truly generic and adaptable to any component configuration.
        """
        # Define validation rules based on component enum metadata
        # This is generic and doesn't hardcode specific component names
        def validate_component(value: FilenameParseValue) -> bool:
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
            return isinstance(value, (float, bool))

        return validate_component

    def _create_generic_extractor(self, component: Enum) -> ComponentExtractor:
        """
        Create a generic extractor for a component based on enum metadata.

        This approach uses the component enum to create extractors that work
        with any component configuration without hardcoded assumptions.
        """
        component_name = component.value

        def extract_component(filename: str) -> FilenameParseValue:
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
    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
        """Parse a filename to extract all components."""
        pass

    @abstractmethod
    def construct_filename(self, extension: str = '.tif', **component_values) -> str:
        """Construct a filename from component values."""
        pass

    def __getstate__(self):
        """
        Custom pickling method to handle dynamic functions.

        Removes generated callables before pickling since they can't be serialized,
        but preserves the component_enum so they can be regenerated.
        """
        state = self.__dict__.copy()
        state.pop("_component_validators", None)
        state.pop("_component_extractors", None)
        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to regenerate dynamic functions.

        Restores the object state and regenerates the dynamic methods
        that were removed during pickling.
        """
        # Restore the object state
        self.__dict__.update(state)

        # Regenerate component authorities
        self._generate_dynamic_methods()
    
    def get_component_names(self) -> list:
        """Get all component names for this parser."""
        return [component.value for component in self.component_enum]

    def validate_component_by_name(self, component_name: str, value: FilenameParseValue) -> bool:
        """
        Validate a component value using the dynamic validation methods.

        Args:
            component_name: Name of the component to validate
            value: Value to validate

        Returns:
            True if the value is valid for the component
        """
        return self._component_validators[component_name](value)

    def extract_component_by_name(self, filename: str, component_name: str) -> FilenameParseValue:
        """
        Extract a specific component from filename using dynamic extraction methods.

        Args:
            filename: Filename to parse
            component_name: Name of component to extract

        Returns:
            Component value or None if extraction fails

        Raises:
            KeyError: If no extraction authority exists for the component
        """
        return self._component_extractors[component_name](filename)
    
    def validate_component_dict(self, components: FilenameParseResult) -> bool:
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
