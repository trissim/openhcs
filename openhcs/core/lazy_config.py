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
from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any, Type, Union

logger = logging.getLogger(__name__)

# Delayed imports to avoid circular dependencies
def _get_config_imports():
    """Get config imports with delayed loading to avoid circular dependencies."""
    from openhcs.core.config import (
        _current_pipeline_config,
        set_current_pipeline_config,
        GlobalPipelineConfig,
        StepMaterializationConfig
    )
    return _current_pipeline_config, set_current_pipeline_config, GlobalPipelineConfig, StepMaterializationConfig


class LazyDataclassFactory:
    """Generic factory for creating lazy dataclasses with flexible resolution."""

    @staticmethod
    def create_lazy_dataclass(
        defaults_source: Union[Type, Any],
        lazy_class_name: str
    ) -> Type:
        """Create lazy version of any dataclass with flexible resolution."""
        # Determine base class and resolution strategy
        base_class, resolver = LazyDataclassFactory._get_base_class_and_resolver(defaults_source)
        
        if not is_dataclass(base_class):
            raise ValueError(f"{base_class} must be a dataclass")

        # Introspect base class fields and preserve types for fields with defaults
        base_fields = fields(base_class)
        lazy_field_definitions = []

        for field in base_fields:
            # Check if field already has Optional type
            origin = getattr(field.type, '__origin__', None)
            is_already_optional = (origin is Union and
                                 type(None) in getattr(field.type, '__args__', ()))

            # Check if field has default value or factory
            from dataclasses import MISSING
            has_default = (field.default is not MISSING or
                         field.default_factory is not MISSING)

            if is_already_optional or not has_default:
                # Field is already Optional or has no default - make it Optional for lazy loading
                field_type = Union[field.type, type(None)] if not is_already_optional else field.type
            else:
                # Field has default - preserve original type (don't make Optional)
                field_type = field.type

            lazy_field_definitions.append((field.name, field_type, None))

            # DEBUG: Log field type decisions
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"LAZY FIELD CREATION: {field.name} - original={field.type}, has_default={has_default}, final={field_type}")

        # Create new dataclass with all fields lazy - no base classes needed
        lazy_class = make_dataclass(
            lazy_class_name,
            lazy_field_definitions,
            frozen=True
        )

        # Bind resolution methods directly to the created class
        LazyDataclassFactory._bind_resolution_methods(lazy_class, base_class, resolver)

        return lazy_class

    @staticmethod
    def _get_base_class_and_resolver(defaults_source: Union[Type, Any]) -> tuple[Type, callable]:
        """Determine base class and resolution strategy from defaults_source."""
        if isinstance(defaults_source, type):
            # Static resolution: instantiate class for each field access
            base_class = defaults_source
            resolver = lambda field_name: getattr(defaults_source(), field_name)
        else:
            # Dynamic resolution: use instance values directly
            base_class = type(defaults_source)
            resolver = lambda field_name: getattr(defaults_source, field_name)
        
        return base_class, resolver

    @staticmethod
    def _bind_resolution_methods(lazy_class: Type, base_class: Type, resolver: callable) -> None:
        """Bind resolution methods directly to the lazy class."""

        def _resolve_field_value(self, field_name: str) -> Any:
            """Resolve field value using configured resolution strategy."""
            return resolver(field_name)

        def __getattribute__(self, name: str) -> Any:
            """Lazy resolution using configured strategy - ALL fields are lazy."""
            value = object.__getattribute__(self, name)
            if value is None and name in [f.name for f in fields(self.__class__)]:
                return self._resolve_field_value(name)
            return value

        def to_base_config(self) -> Any:
            """Convert lazy config to base config by resolving all fields."""
            resolved_values = {
                field_obj.name: getattr(self, field_obj.name)
                for field_obj in fields(self)
            }
            return base_class(**resolved_values)

        # Bind methods directly to class using setattr
        setattr(lazy_class, '_resolve_field_value', _resolve_field_value)
        setattr(lazy_class, '__getattribute__', __getattribute__)
        setattr(lazy_class, 'to_base_config', to_base_config)
        setattr(lazy_class, 'with_defaults', classmethod(lambda cls: cls()))
        setattr(lazy_class, 'with_overrides', classmethod(lambda cls, **kwargs: cls(**kwargs)))

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
        if not is_dataclass(base_class):
            raise ValueError(f"{base_class} must be a dataclass")

        # Generate class name if not provided
        if lazy_class_name is None:
            lazy_class_name = f"Lazy{base_class.__name__}"

        # Create unified thread-local resolver using explicit field paths
        def unified_thread_local_resolver(field_name_to_resolve: str) -> Any:
            """Resolve field value from thread-local storage using explicit field path."""
            try:
                # Get config imports with delayed loading
                _current_pipeline_config, _, _, _ = _get_config_imports()

                # Get thread-local instance using explicit field path
                thread_local_instance = LazyDataclassFactory._get_thread_local_instance(
                    _current_pipeline_config, field_path
                )

                if thread_local_instance is not None:
                    return getattr(thread_local_instance, field_name_to_resolve)

            except (AttributeError, ImportError):
                pass

            # Fallback to static resolution if thread-local storage unavailable
            static_instance = base_class()
            return getattr(static_instance, field_name_to_resolve)

        # Introspect base class fields and preserve types for fields with defaults
        base_fields = fields(base_class)
        lazy_field_definitions = []

        for field in base_fields:
            # Check if field already has Optional type
            origin = getattr(field.type, '__origin__', None)
            is_already_optional = (origin is Union and
                                 type(None) in getattr(field.type, '__args__', ()))

            # Check if field has default value or factory
            from dataclasses import MISSING
            has_default = (field.default is not MISSING or
                         field.default_factory is not MISSING)

            if is_already_optional or not has_default:
                # Field is already Optional or has no default - make it Optional for lazy loading
                field_type = Union[field.type, type(None)] if not is_already_optional else field.type
            else:
                # Field has default - preserve original type (don't make Optional)
                field_type = field.type

            lazy_field_definitions.append((field.name, field_type, None))

            # DEBUG: Log field type decisions
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"THREAD-LOCAL LAZY FIELD: {field.name} - original={field.type}, has_default={has_default}, final={field_type}")

        # Create new dataclass with all fields lazy
        lazy_class = make_dataclass(
            lazy_class_name,
            lazy_field_definitions,
            frozen=True
        )

        # Bind resolution methods using the unified thread-local resolver
        LazyDataclassFactory._bind_resolution_methods(lazy_class, base_class, unified_thread_local_resolver)

        return lazy_class

    @staticmethod
    def _get_thread_local_instance(current_pipeline_config, field_path: str = None) -> Any:
        """
        Get thread-local instance using explicit field path navigation.

        Args:
            current_pipeline_config: Thread-local storage object
            field_path: Dot-separated path to navigate (None = root)

        Returns:
            Instance at the specified field path, or None if not found
        """
        if not (hasattr(current_pipeline_config, 'value') and current_pipeline_config.value):
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

    # Old problematic methods removed - replaced by unified field path system

    @staticmethod
    def create_lazy_dataclass_with_generic_thread_local_resolver(
        base_class: Type,
        lazy_class_name: str = None
    ) -> Type:
        """
        Backward compatibility alias for the unified system.

        This method is deprecated. Use make_lazy_thread_local() with explicit field_path instead.
        For StepMaterializationConfig, use field_path="materialization_defaults".
        For GlobalPipelineConfig, use field_path=None.
        """
        # Determine field path based on class name for backward compatibility
        class_name = base_class.__name__

        if class_name == 'GlobalPipelineConfig':
            field_path = None
        elif class_name == 'StepMaterializationConfig':
            field_path = "materialization_defaults"
        else:
            # For other classes, try to guess the field path
            # This is a temporary measure during the transition
            field_path = None
            logger.warning(f"Using deprecated method for {class_name}. Please migrate to make_lazy_thread_local() with explicit field_path.")

        return LazyDataclassFactory.make_lazy_thread_local(
            base_class=base_class,
            field_path=field_path,
            lazy_class_name=lazy_class_name
        )

    # Explicit fallback method removed - use make_lazy_thread_local() with appropriate field_path instead


# Widget-level utility functions for clean thread-local storage management
def ensure_pipeline_config_context(orchestrator_global_config):
    """Ensure proper thread-local storage setup for configuration editing."""
    _, set_current_pipeline_config, _, _ = _get_config_imports()
    set_current_pipeline_config(orchestrator_global_config)


def create_pipeline_config_for_editing(orchestrator_global_config):
    """Create PipelineConfig for editing with proper thread-local context."""
    # Ensure thread-local storage is set
    ensure_pipeline_config_context(orchestrator_global_config)

    # Create PipelineConfig with all fields as None for placeholder behavior
    return PipelineConfig()  # All fields None - will show as placeholders


def _add_to_base_config_method(lazy_class, base_class):
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
    lazy_class_name="PipelineConfig"
)

# Add to_base_config method for orchestrator integration
_add_to_base_config_method(PipelineConfig, GlobalPipelineConfig)

# Use the new unified thread-local resolver for step materialization config
# field_path="materialization_defaults" means it resolves from GlobalPipelineConfig.materialization_defaults
LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=StepMaterializationConfig,
    field_path="materialization_defaults",  # Gets _current_pipeline_config.value.materialization_defaults
    lazy_class_name="LazyStepMaterializationConfig"
)


