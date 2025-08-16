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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from openhcs.core.config import get_base_type_for_lazy

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

# Generic imports for lazy configuration system
def _get_generic_config_imports():
    """Get generic config imports with delayed loading to avoid circular dependencies."""
    from openhcs.core.config import get_current_global_config, set_current_global_config
    return get_current_global_config, set_current_global_config


# No strategy pattern needed - just use instance provider functions directly


class FieldPathNavigator:
    """Utility for navigating dot-separated field paths in object hierarchies."""

    @staticmethod
    def navigate_to_instance(current_global_config: Any, field_path: Optional[str] = None) -> Optional[Any]:
        """
        Navigate to instance using explicit field path.

        Args:
            current_global_config: Thread-local storage object or global config instance
            field_path: Dot-separated path to navigate (None = root)

        Returns:
            Instance at the specified field path, or None if not found
        """
        # Handle both thread-local storage objects and direct config instances
        if hasattr(current_global_config, CONSTANTS.THREAD_LOCAL_VALUE_ATTR):
            if not current_global_config.value:
                return None
            instance = current_global_config.value
        else:
            # Direct config instance
            instance = current_global_config

        if field_path is None:
            # Root instance - return the global config directly
            return instance

        # Navigate dot-separated path
        for field in field_path.split('.'):
            if instance is None:
                return None
            instance = getattr(instance, field, None)

        return instance


@dataclass(frozen=True)
class ResolutionConfig:
    """Declarative configuration for recursive lazy resolution."""
    instance_provider: Callable[[], Any]
    fallback_chain: List[Callable[[str], Any]]

    def resolve_field(self, field_name: str) -> Any:
        """Resolve field through primary instance and fallback chain."""
        return self._try_primary(field_name) or self._try_fallbacks(field_name)

    def _try_primary(self, field_name: str) -> Any:
        """Attempt resolution from primary instance."""
        try:
            instance = self.instance_provider()
            if instance and hasattr(instance, field_name):
                value = object.__getattribute__(instance, field_name)
                return value if value is not None else None
        except (AttributeError, Exception):
            pass
        return None

    def _try_fallbacks(self, field_name: str) -> Any:
        """Attempt resolution through fallback chain."""
        for fallback in self.fallback_chain:
            try:
                value = fallback(field_name)
                if value is not None:
                    return value
            except (AttributeError, Exception):
                continue
        return None


# Functional fallback strategies
def create_static_defaults_fallback(base_class: Type) -> Callable[[str], Any]:
    """Create fallback that resolves to static dataclass defaults."""
    default_instance = base_class()
    return lambda field_name: getattr(default_instance, field_name, None)


def create_instance_fallback(instance_provider: Callable[[], Any]) -> Callable[[str], Any]:
    """Create fallback that resolves from specific instance."""
    return lambda field_name: (
        getattr(instance_provider(), field_name, None)
        if (instance := instance_provider()) else None
    )


@dataclass(frozen=True)
class LazyMethodBindings:
    """Declarative method bindings for lazy dataclasses."""

    @staticmethod
    def create_resolver(resolution_config: ResolutionConfig) -> Callable[[Any, str], Any]:
        """Create field resolver method."""
        return lambda self, field_name: resolution_config.resolve_field(field_name)

    @staticmethod
    def create_getattribute() -> Callable[[Any, str], Any]:
        """Create lazy __getattribute__ method."""
        def __getattribute__(self: Any, name: str) -> Any:
            value = object.__getattribute__(self, name)
            return (self._resolve_field_value(name)
                   if value is None and name in {f.name for f in fields(self.__class__)}
                   else value)
        return __getattribute__

    @staticmethod
    def create_to_base_config(base_class: Type) -> Callable[[Any], Any]:
        """Create base config converter method."""
        return lambda self: base_class(**{
            f.name: getattr(self, f.name) for f in fields(self)
        })

    @staticmethod
    def create_class_methods() -> Dict[str, Any]:
        """Create class-level utility methods."""
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
        debug_template: str,
        use_recursive_resolution: bool = False,
        fallback_chain: Optional[List[Callable[[str], Any]]] = None
    ) -> Type:
        """Create lazy dataclass with declarative configuration."""
        if not is_dataclass(base_class):
            raise ValueError(f"{base_class} must be a dataclass")

        # Create resolution configuration
        if use_recursive_resolution:
            resolution_config = ResolutionConfig(
                instance_provider=instance_provider,
                fallback_chain=fallback_chain or [create_static_defaults_fallback(base_class)]
            )
        else:
            # For non-recursive resolution, create a safe fallback that handles None instance provider
            def safe_instance_fallback(field_name: str) -> Any:
                try:
                    instance = instance_provider()
                    return getattr(instance, field_name, None) if instance is not None else None
                except (AttributeError, Exception):
                    return None

            resolution_config = ResolutionConfig(
                instance_provider=instance_provider,
                fallback_chain=[safe_instance_fallback, create_static_defaults_fallback(base_class)]
            )

        # Create lazy dataclass with introspected fields
        lazy_class = make_dataclass(
            lazy_class_name,
            LazyDataclassFactory._introspect_dataclass_fields(base_class, debug_template),
            frozen=True
        )

        # Bind methods declaratively
        LazyDataclassFactory._bind_methods_to_class(lazy_class, base_class, resolution_config)
        return lazy_class

    @staticmethod
    def _bind_methods_to_class(lazy_class: Type, base_class: Type, resolution_config: ResolutionConfig) -> None:
        """Bind methods to lazy dataclass using declarative configuration."""
        method_bindings = {
            CONSTANTS.RESOLVE_FIELD_VALUE_METHOD: LazyMethodBindings.create_resolver(resolution_config),
            CONSTANTS.GET_ATTRIBUTE_METHOD: LazyMethodBindings.create_getattribute(),
            CONSTANTS.TO_BASE_CONFIG_METHOD: LazyMethodBindings.create_to_base_config(base_class),
            **LazyMethodBindings.create_class_methods()
        }

        for method_name, method_impl in method_bindings.items():
            setattr(lazy_class, method_name, method_impl)

    @staticmethod
    def create_lazy_dataclass(
        defaults_source: Union[Type, Any],
        lazy_class_name: str,
        use_recursive_resolution: bool = False,
        fallback_chain: Optional[List[Callable[[str], Any]]] = None
    ) -> Type:
        """Create lazy dataclass with functional configuration."""
        base_class = defaults_source if isinstance(defaults_source, type) else type(defaults_source)
        instance_provider = (lambda: defaults_source()) if isinstance(defaults_source, type) else (lambda: defaults_source)

        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class, instance_provider, lazy_class_name,
            CONSTANTS.LAZY_FIELD_DEBUG_TEMPLATE, use_recursive_resolution, fallback_chain
        )

    @staticmethod
    def make_lazy_thread_local(
        base_class: Type,
        global_config_type: Type,
        field_path: str = None,
        lazy_class_name: str = None,
        use_recursive_resolution: bool = False
    ) -> Type:
        """
        Create lazy dataclass that resolves from thread-local instance using explicit field paths.

        This unified approach eliminates algorithmic field name conversion bugs by using
        explicit dot-separated paths to navigate the thread-local configuration structure.

        Args:
            base_class: The dataclass type to make lazy (the target type for lazy resolution)
            global_config_type: The global config type used for thread-local storage context
                              (e.g., GlobalPipelineConfig, GlobalAppConfig)
            field_path: Dot-separated path to instance (None = root)
                       Examples: None, "materialization_defaults", "foo.bar.baz"
            lazy_class_name: Optional name for the generated lazy class
            use_recursive_resolution: Whether to use recursive resolution for None values

        Returns:
            Generated lazy dataclass with explicit thread-local resolution

        Note:
            base_class and global_config_type serve different purposes:
            - base_class: The type being made lazy (what the lazy class represents)
            - global_config_type: The type used for thread-local context (where values come from)

            They are often the same (e.g., both GlobalPipelineConfig) but can differ when
            creating lazy versions of nested config types that resolve from a different
            global context (e.g., base_class=StepMaterializationConfig,
            global_config_type=GlobalPipelineConfig).

        Examples:
            # Root thread-local instance with recursive resolution
            PipelineConfig = make_lazy_thread_local(
                GlobalPipelineConfig,
                field_path=None,
                use_recursive_resolution=True
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

        # Global config type is now a required parameter

        # Create instance provider for thread-local resolution
        def thread_local_instance_provider() -> Any:
            """Get instance from thread-local storage using field path."""
            get_current_global_config, _ = _get_generic_config_imports()

            current_config = get_current_global_config(global_config_type)
            if current_config is not None:
                return FieldPathNavigator.navigate_to_instance(current_config, field_path)

            return None

        # Configure fallback chain for recursive resolution
        fallback_chain = [create_static_defaults_fallback(base_class)] if use_recursive_resolution else None

        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class, thread_local_instance_provider, lazy_class_name,
            CONSTANTS.THREAD_LOCAL_FIELD_DEBUG_TEMPLATE, use_recursive_resolution, fallback_chain
        )

    # Deprecated methods removed - use make_lazy_thread_local() with explicit field_path


# Generic utility functions for clean thread-local storage management
def ensure_global_config_context(global_config_type: Type, global_config_instance: Any) -> None:
    """Ensure proper thread-local storage setup for any global config type."""
    _, set_current_global_config = _get_generic_config_imports()
    set_current_global_config(global_config_type, global_config_instance)


def resolve_lazy_configurations_for_serialization(data: Any) -> Any:
    """
    Recursively resolve all lazy dataclass configurations to concrete values for serialization.

    Traverses any data structure and converts lazy dataclass instances to their concrete
    base configurations, making the data structure safe for pickling and inter-process
    communication.

    Args:
        data: Any data structure that may contain lazy dataclass configurations

    Returns:
        Data structure with all lazy configurations resolved to concrete values

    Note:
        Uses existing OpenHCS lazy dataclass infrastructure:
        - get_base_type_for_lazy() to detect lazy dataclass types
        - to_base_config() method to resolve lazy configs to concrete configs
    """
    # Resolve the object itself if it's a lazy dataclass
    resolved_data = (data.to_base_config()
                    if get_base_type_for_lazy(type(data)) is not None
                    else data)

    # Recursively process nested structures based on type
    if is_dataclass(resolved_data) and not isinstance(resolved_data, type):
        # Process dataclass fields recursively
        resolved_fields = {
            field_obj.name: resolve_lazy_configurations_for_serialization(getattr(resolved_data, field_obj.name))
            for field_obj in fields(resolved_data)
        }
        return type(resolved_data)(**resolved_fields)

    elif isinstance(resolved_data, dict):
        # Process dictionary values recursively
        return {
            key: resolve_lazy_configurations_for_serialization(value)
            for key, value in resolved_data.items()
        }

    elif isinstance(resolved_data, (list, tuple)):
        # Process sequence elements recursively
        resolved_items = [resolve_lazy_configurations_for_serialization(item) for item in resolved_data]
        return type(resolved_data)(resolved_items)

    else:
        # Primitive type or unknown structure - return as-is
        return resolved_data


# Generic dataclass editing with configurable value preservation
T = TypeVar('T')


def create_dataclass_for_editing(
    dataclass_type: Type[T],
    source_config: Any,
    preserve_values: bool = False,
    context_provider: Optional[Callable[[Any], None]] = None
) -> T:
    """
    Create any dataclass for editing with configurable value preservation.

    This generic function works with any dataclass type, not just PipelineConfig.

    Args:
        dataclass_type: The dataclass type to create (e.g., PipelineConfig, ZarrConfig)
        source_config: Instance to use for context and optionally field values
        preserve_values:
            - True: Preserve actual field values (direct editing)
            - False: Use None values for placeholders (hierarchical editing)
        context_provider: Optional function to set up context (e.g., thread-local storage)

    Returns:
        Instance of dataclass_type with appropriate field initialization

    Examples:
        # Edit any dataclass with preserved values
        editable_zarr = create_dataclass_for_editing(ZarrConfig, zarr_config, preserve_values=True)

        # Create dataclass with placeholders
        placeholder_vfs = create_dataclass_for_editing(VFSConfig, vfs_config, preserve_values=False)
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")

    # Set up context if provider is given (e.g., thread-local storage)
    if context_provider:
        context_provider(source_config)

    # Initialize field values based on editing mode
    field_values = {}
    for field_obj in fields(dataclass_type):
        if preserve_values:
            # Direct editing: preserve actual field values
            field_values[field_obj.name] = getattr(source_config, field_obj.name)
        else:
            # Hierarchical editing: use None for placeholder behavior
            field_values[field_obj.name] = None

    return dataclass_type(**field_values)


def create_config_for_editing(
    global_config_type: Type,
    global_config_instance: Any,
    preserve_values: bool = False,
    placeholder_prefix: str = "Default"
) -> Any:
    """
    Create editable config for any global dataclass type.

    This is the generic version that works with any global config type.

    Args:
        global_config_type: The global config type (e.g., GlobalPipelineConfig, GlobalAppConfig)
        global_config_instance: Instance to use for context and optionally field values
        preserve_values: Whether to preserve actual values or use placeholders
        placeholder_prefix: Prefix for placeholder text (e.g., "Pipeline default", "App default")

    Returns:
        Lazy config instance suitable for editing
    """
    return create_dataclass_for_editing(
        global_config_type,
        global_config_instance,
        preserve_values=preserve_values,
        context_provider=lambda config: ensure_global_config_context(global_config_type, config)
    )





def rebuild_lazy_config_with_new_global_reference(
    existing_lazy_config: Any,
    new_global_config: Any,
    global_config_type: Optional[Type] = None
) -> Any:
    """
    Rebuild lazy config to reference new global config while preserving field states.

    This function preserves the exact field state of the existing lazy config:
    - Fields that are None (using lazy resolution) remain None
    - Fields that have been explicitly set retain their concrete values
    - Nested dataclass fields are recursively rebuilt to reference new global config
    - The underlying global config reference is updated for None field resolution

    Args:
        existing_lazy_config: Current lazy config instance
        new_global_config: New global config to reference for lazy resolution
        global_config_type: Type of the global config (defaults to type of new_global_config)

    Returns:
        New lazy config instance with preserved field states and updated global reference
    """
    if existing_lazy_config is None:
        return None

    # Determine global config type
    if global_config_type is None:
        global_config_type = type(new_global_config)

    # Set new global config in thread-local storage
    ensure_global_config_context(global_config_type, new_global_config)

    # Extract current field values without triggering lazy resolution
    current_field_values = {}
    for field_obj in fields(existing_lazy_config):
        # Use object.__getattribute__ to get raw stored value (None or concrete value)
        raw_value = object.__getattribute__(existing_lazy_config, field_obj.name)

        # If the field is a concrete nested dataclass, rebuild it with new global reference
        if raw_value is not None and hasattr(raw_value, '__dataclass_fields__'):
            # This is a concrete nested dataclass - get the corresponding field from new global config
            try:
                new_nested_value = getattr(new_global_config, field_obj.name)
                current_field_values[field_obj.name] = new_nested_value
            except AttributeError:
                # Field doesn't exist in new global config, keep original value
                current_field_values[field_obj.name] = raw_value
        else:
            # Regular field (None or non-dataclass value) - preserve as-is
            current_field_values[field_obj.name] = raw_value

    # Create new lazy config instance with preserved field values
    # This maintains the exact state: None values stay None, concrete values stay concrete
    # Nested dataclasses are updated to reference new global config
    lazy_class_type = type(existing_lazy_config)
    return lazy_class_type(**current_field_values)












# This module is now completely generic and contains no pipeline-specific logic.
# Pipeline-specific lazy classes are created in openhcs.core.pipeline_config module.


