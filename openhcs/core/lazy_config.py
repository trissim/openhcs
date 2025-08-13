"""
Generic lazy dataclass factory using flexible resolution.

This module provides a truly generic lazy loading abstraction that works with any dataclass
using dataclass field introspection for delayed object creation, eliminating hardcoded
configuration types and maintaining zero knowledge of specific configuration types.
Supports both static resolution (from class) and dynamic resolution (from instance).
Creates complete lazy dataclasses with bound methods - no mixin inheritance needed.
"""

# Standard library imports
import logging
import re
# No ABC needed - using simple functions instead of strategy pattern
from dataclasses import dataclass, fields, is_dataclass, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LazyConfigConstants:
    """Centralized constants for lazy configuration system."""

    # Field paths for thread-local resolution
    MATERIALIZATION_DEFAULTS_PATH: str = "materialization_defaults"
    THREAD_LOCAL_VALUE_ATTR: str = "value"

    # Class names for backward compatibility
    PIPELINE_CONFIG_NAME: str = "PipelineConfig"
    LAZY_STEP_MATERIALIZATION_CONFIG_NAME: str = "LazyStepMaterializationConfig"

    # Method names for dynamic binding
    RESOLVE_FIELD_VALUE_METHOD: str = "_resolve_field_value"
    GET_ATTRIBUTE_METHOD: str = "__getattribute__"
    TO_BASE_CONFIG_METHOD: str = "to_base_config"
    WITH_DEFAULTS_METHOD: str = "with_defaults"
    WITH_OVERRIDES_METHOD: str = "with_overrides"

    # Debug message templates
    LAZY_FIELD_DEBUG_TEMPLATE: str = "LAZY FIELD CREATION: {field_name} - original={original_type}, has_default={has_default}, final={final_type}"
    THREAD_LOCAL_FIELD_DEBUG_TEMPLATE: str = "THREAD-LOCAL LAZY FIELD: {field_name} - original={original_type}, has_default={has_default}, final={final_type}"

    # Class name generation
    LAZY_CLASS_NAME_PREFIX: str = "Lazy"


# Create constants instance for use throughout module
CONSTANTS = LazyConfigConstants()

# Delayed imports to avoid circular dependencies
def _get_config_imports() -> Tuple[Any, Callable[[Any], None], Type, Type]:
    """Get config imports with delayed loading to avoid circular dependencies."""
    from openhcs.core.config import (
        _current_pipeline_config,
        set_current_pipeline_config,
        GlobalPipelineConfig,
        StepMaterializationConfig
    )
    return _current_pipeline_config, set_current_pipeline_config, GlobalPipelineConfig, StepMaterializationConfig


# No strategy pattern needed - just use instance provider functions directly


class FieldPathNavigator:
    """Utility for navigating dot-separated field paths in object hierarchies."""

    @staticmethod
    def navigate_to_instance(current_pipeline_config: Any, field_path: Optional[str] = None) -> Optional[Any]:
        """
        Navigate to instance using explicit field path.

        Args:
            current_pipeline_config: Thread-local storage object
            field_path: Dot-separated path to navigate (None = root)

        Returns:
            Instance at the specified field path, or None if not found
        """
        if not (hasattr(current_pipeline_config, CONSTANTS.THREAD_LOCAL_VALUE_ATTR) and current_pipeline_config.value):
            return None

        instance = current_pipeline_config.value

        if field_path is None:
            # Root instance - return the GlobalPipelineConfig directly
            return instance

        # Navigate dot-separated path
        for field in field_path.split('.'):
            if instance is None:
                return None
            instance = getattr(instance, field, None)

        return instance


class LazyMethodFactory:
    """Factory for creating methods that are bound to lazy dataclasses."""

    @staticmethod
    def create_resolve_field_value_method(resolver: Callable[[str], Any]) -> Callable[[Any, str], Any]:
        """Create _resolve_field_value method for lazy dataclass."""
        def _resolve_field_value(self: Any, field_name: str) -> Any:
            """Resolve field value using configured resolution strategy."""
            return resolver(field_name)
        return _resolve_field_value

    @staticmethod
    def create_getattribute_method() -> Callable[[Any, str], Any]:
        """Create __getattribute__ method for lazy dataclass."""
        def __getattribute__(self: Any, name: str) -> Any:
            """Lazy resolution using configured strategy - ALL fields are lazy."""
            value = object.__getattribute__(self, name)
            if value is None and name in [f.name for f in fields(self.__class__)]:
                return self._resolve_field_value(name)
            return value
        return __getattribute__

    @staticmethod
    def create_to_base_config_method(base_class: Type) -> Callable[[Any], Any]:
        """Create to_base_config method for lazy dataclass."""
        def to_base_config(self: Any) -> Any:
            """Convert lazy config to base config by resolving all fields."""
            resolved_values = {
                field_obj.name: getattr(self, field_obj.name)
                for field_obj in fields(self)
            }
            return base_class(**resolved_values)
        return to_base_config

    @staticmethod
    def create_class_methods() -> Dict[str, Any]:
        """Create class methods for lazy dataclass."""
        return {
            CONSTANTS.WITH_DEFAULTS_METHOD: classmethod(lambda cls: cls()),
            CONSTANTS.WITH_OVERRIDES_METHOD: classmethod(lambda cls, **kwargs: cls(**kwargs))
        }


class LazyDataclassFactory:
    """Generic factory for creating lazy dataclasses with flexible resolution."""

    @staticmethod
    def _introspect_dataclass_fields(base_class: Type, debug_template: str) -> List[Tuple[str, Type, None]]:
        """
        Unified field introspection logic for lazy dataclass creation.

        Analyzes dataclass fields to determine appropriate types for lazy loading,
        preserving original types for fields with defaults while making fields
        without defaults Optional for lazy resolution.

        Args:
            base_class: The dataclass to introspect
            debug_template: Template string for debug logging

        Returns:
            List of (field_name, field_type, default_value) tuples for make_dataclass
        """
        from dataclasses import MISSING

        base_fields = fields(base_class)
        lazy_field_definitions = []

        for field in base_fields:
            # Check if field already has Optional type
            origin = getattr(field.type, '__origin__', None)
            is_already_optional = (origin is Union and
                                 type(None) in getattr(field.type, '__args__', ()))

            # Check if field has default value or factory
            has_default = (field.default is not MISSING or
                         field.default_factory is not MISSING)

            if is_already_optional or not has_default:
                # Field is already Optional or has no default - make it Optional for lazy loading
                field_type = Union[field.type, type(None)] if not is_already_optional else field.type
            else:
                # Field has default - preserve original type (don't make Optional)
                field_type = field.type

            lazy_field_definitions.append((field.name, field_type, None))

            # Debug logging with provided template
            logger.info(debug_template.format(
                field_name=field.name,
                original_type=field.type,
                has_default=has_default,
                final_type=field_type
            ))

        return lazy_field_definitions

    @staticmethod
    def _create_lazy_dataclass_unified(
        base_class: Type,
        instance_provider: Callable[[], Any],
        lazy_class_name: str,
        debug_template: str
    ) -> Type:
        """
        Unified lazy dataclass creation workflow.

        This is the core method that all other factory methods delegate to,
        using a simple instance provider function for resolution.
        """
        if not is_dataclass(base_class):
            raise ValueError(f"{base_class} must be a dataclass")

        # Create resolver directly from instance provider - no strategy pattern needed
        def resolver(field_name: str) -> Any:
            """Resolve field value from instance provided by instance_provider."""
            instance = instance_provider()
            return getattr(instance, field_name)

        # Introspect fields using unified logic
        lazy_field_definitions = LazyDataclassFactory._introspect_dataclass_fields(
            base_class, debug_template
        )

        # Create dataclass
        lazy_class = make_dataclass(lazy_class_name, lazy_field_definitions, frozen=True)

        # Bind methods using factory
        LazyDataclassFactory._bind_methods_to_class(lazy_class, base_class, resolver)

        return lazy_class

    @staticmethod
    def _bind_methods_to_class(lazy_class: Type, base_class: Type, resolver: Callable[[str], Any]) -> None:
        """Bind all necessary methods to the lazy dataclass using method factories."""
        # Create and bind instance methods
        setattr(lazy_class, CONSTANTS.RESOLVE_FIELD_VALUE_METHOD,
                LazyMethodFactory.create_resolve_field_value_method(resolver))
        setattr(lazy_class, CONSTANTS.GET_ATTRIBUTE_METHOD,
                LazyMethodFactory.create_getattribute_method())
        setattr(lazy_class, CONSTANTS.TO_BASE_CONFIG_METHOD,
                LazyMethodFactory.create_to_base_config_method(base_class))

        # Create and bind class methods
        class_methods = LazyMethodFactory.create_class_methods()
        for method_name, method_impl in class_methods.items():
            setattr(lazy_class, method_name, method_impl)

    @staticmethod
    def create_lazy_dataclass(
        defaults_source: Union[Type, Any],
        lazy_class_name: str
    ) -> Type:
        """Create lazy version of any dataclass with flexible resolution."""
        # Determine base class from defaults_source
        base_class = defaults_source if isinstance(defaults_source, type) else type(defaults_source)

        # Create instance provider for static resolution
        if isinstance(defaults_source, type):
            # Static resolution: instantiate class for each field access
            instance_provider = lambda: defaults_source()
        else:
            # Dynamic resolution: use instance values directly
            instance_provider = lambda: defaults_source

        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class=base_class,
            instance_provider=instance_provider,
            lazy_class_name=lazy_class_name,
            debug_template=CONSTANTS.LAZY_FIELD_DEBUG_TEMPLATE
        )

    @staticmethod
    def make_lazy_thread_local(
        base_class: Type,
        field_path: str = None,
        lazy_class_name: str = None
    ) -> Type:
        """
        Create lazy dataclass that resolves from thread-local instance using explicit field paths.

        This unified approach eliminates algorithmic field name conversion bugs by using
        explicit dot-separated paths to navigate the thread-local configuration structure.

        Args:
            base_class: The dataclass type to make lazy
            field_path: Dot-separated path to instance (None = root)
                       Examples: None, "materialization_defaults", "foo.bar.baz"
            lazy_class_name: Optional name for the generated lazy class

        Returns:
            Generated lazy dataclass with explicit thread-local resolution

        Examples:
            # Root thread-local instance
            PipelineConfig = make_lazy_thread_local(
                GlobalPipelineConfig,
                field_path=None
            )

            # Nested field from thread-local instance
            LazyStepMaterializationConfig = make_lazy_thread_local(
                StepMaterializationConfig,
                field_path="materialization_defaults"
            )
        """
        # Generate class name if not provided
        if lazy_class_name is None:
            lazy_class_name = f"{CONSTANTS.LAZY_CLASS_NAME_PREFIX}{base_class.__name__}"

        # Create instance provider for thread-local resolution
        def thread_local_instance_provider() -> Any:
            """Get instance from thread-local storage using field path."""
            _current_pipeline_config, _, _, _ = _get_config_imports()
            return FieldPathNavigator.navigate_to_instance(_current_pipeline_config, field_path)

        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class=base_class,
            instance_provider=thread_local_instance_provider,
            lazy_class_name=lazy_class_name,
            debug_template=CONSTANTS.THREAD_LOCAL_FIELD_DEBUG_TEMPLATE
        )

    # Deprecated methods removed - use make_lazy_thread_local() with explicit field_path


# Widget-level utility functions for clean thread-local storage management
def ensure_pipeline_config_context(orchestrator_global_config: Any) -> None:
    """Ensure proper thread-local storage setup for configuration editing."""
    _, set_current_pipeline_config, _, _ = _get_config_imports()
    set_current_pipeline_config(orchestrator_global_config)


def create_pipeline_config_for_editing(orchestrator_global_config: Any) -> Any:
    """Create PipelineConfig for editing with proper thread-local context."""
    # Ensure thread-local storage is set
    ensure_pipeline_config_context(orchestrator_global_config)

    # Create PipelineConfig with all fields as None for placeholder behavior
    return PipelineConfig()  # All fields None - will show as placeholders


def _add_to_base_config_method(lazy_class: Type, base_class: Type) -> None:
    """Add to_base_config method to lazy dataclass for orchestrator integration."""
    def to_base_config(self):
        """Convert lazy config to base config, resolving None values to current defaults."""
        # Get all field values, resolving None values through lazy loading
        resolved_values = {}
        for field in fields(self):
            value = getattr(self, field.name)  # This triggers lazy resolution for None values
            resolved_values[field.name] = value

        return base_class(**resolved_values)

    # Bind the method to the lazy class
    lazy_class.to_base_config = to_base_config






# Generate lazy configuration classes using unified thread-local resolution
_, _, GlobalPipelineConfig, StepMaterializationConfig = _get_config_imports()

# Use the new unified thread-local resolver for PipelineConfig
# field_path=None means it resolves from the root GlobalPipelineConfig
PipelineConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=GlobalPipelineConfig,
    field_path=None,  # Root instance - gets _current_pipeline_config.value directly
    lazy_class_name=CONSTANTS.PIPELINE_CONFIG_NAME
)

# Add to_base_config method for orchestrator integration
_add_to_base_config_method(PipelineConfig, GlobalPipelineConfig)

# Use the new unified thread-local resolver for step materialization config
# field_path="materialization_defaults" means it resolves from GlobalPipelineConfig.materialization_defaults
LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=StepMaterializationConfig,
    field_path=CONSTANTS.MATERIALIZATION_DEFAULTS_PATH,  # Gets _current_pipeline_config.value.materialization_defaults
    lazy_class_name=CONSTANTS.LAZY_STEP_MATERIALIZATION_CONFIG_NAME
)


