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
import threading
from contextlib import contextmanager
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


# Context stack system removed - using simple thread-local storage approach


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
            # Use object.__getattribute__ to avoid triggering lazy resolution during navigation
            try:
                instance = object.__getattribute__(instance, field)
            except AttributeError:
                return None

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
        instance = self.instance_provider()
        return _get_raw_field_value(instance, field_name) if instance else None

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
def _get_raw_field_value(obj: Any, field_name: str) -> Any:
    """
    Get raw field value bypassing lazy property getters to prevent infinite recursion.

    Uses object.__getattribute__() to access stored values directly without triggering
    lazy resolution, which would create circular dependencies in the resolution chain.

    Args:
        obj: Object to get field from
        field_name: Name of field to access

    Returns:
        Raw field value or None if field doesn't exist

    Raises:
        AttributeError: If field doesn't exist (fail-loud behavior)
    """
    try:
        return object.__getattribute__(obj, field_name)
    except AttributeError:
        return None


def create_static_defaults_fallback(base_class: Type) -> Callable[[str], Any]:
    """Create fallback that resolves to static dataclass defaults."""
    default_instance = base_class()
    return lambda field_name: _get_raw_field_value(default_instance, field_name)


def create_instance_fallback(instance_provider: Callable[[], Any]) -> Callable[[str], Any]:
    """Create fallback that resolves from specific instance."""
    def instance_fallback(field_name: str) -> Any:
        instance = instance_provider()
        return _get_raw_field_value(instance, field_name) if instance else None
    return instance_fallback


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
            if value is None and name in {f.name for f in fields(self.__class__)}:
                # Check if this field has a lazy dataclass type
                field_obj = next((f for f in fields(self.__class__) if f.name == name), None)
                if field_obj:
                    field_type = field_obj.type
                    # Handle Optional[LazyType] by unwrapping
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        args = getattr(field_type, '__args__', ())
                        if len(args) == 2 and type(None) in args:
                            field_type = args[0] if args[1] is type(None) else args[1]

                    # Check if field type is a lazy dataclass (has _resolve_field_value method)
                    if hasattr(field_type, '_resolve_field_value') or (
                        hasattr(field_type, '__name__') and field_type.__name__.startswith('Lazy')
                    ):
                        # Create instance of lazy nested class
                        return field_type()

                # Fall back to standard resolution for non-lazy fields
                return self._resolve_field_value(name)
            else:
                return value
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
    def _create_unified_lazy_class(
        base_class: Type,
        global_config_type: Type,
        field_name: str,
        lazy_class_name: str,
        parent_field_path: str = None,
        parent_instance_provider: Optional[Callable[[], Any]] = None
    ) -> Type:
        """Create lazy class with automatic field-level hierarchy discovery and context propagation."""

        # CRITICAL FIX: Construct proper field path from field_name and parent_field_path
        # In recursive resolution context, field_name is just the local field name (e.g., "materialization_defaults")
        # but we need the full path from global config root for proper hierarchy discovery
        full_field_path = f"{parent_field_path}.{field_name}" if parent_field_path else field_name

        # Create context provider that uses parent instance if available
        def nested_context_provider():
            if parent_instance_provider:
                return parent_instance_provider()

            # Fall back to global config
            get_current_global_config, _ = _get_generic_config_imports()
            return get_current_global_config(global_config_type)

        return LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
            base_class=base_class,
            global_config_type=global_config_type,
            field_path=full_field_path,  # Use constructed full path, not just field_name
            lazy_class_name=lazy_class_name,
            context_provider=nested_context_provider  # Pass context provider
        )

    @staticmethod
    def _introspect_dataclass_fields(
        base_class: Type,
        debug_template: str,
        global_config_type: Type = None,
        parent_field_path: str = None,
        parent_instance_provider: Optional[Callable[[], Any]] = None
    ) -> List[Tuple[str, Type, None]]:
        """
        Unified field introspection logic for lazy dataclass creation.

        Analyzes dataclass fields to determine appropriate types for lazy loading,
        preserving original types for fields with defaults while making fields
        without defaults Optional for lazy resolution. Converts nested dataclass
        fields to their lazy equivalents with context propagation.

        Args:
            base_class: The dataclass to introspect
            debug_template: Template string for debug logging
            global_config_type: Global config type for creating lazy nested types
            parent_field_path: Field path prefix for nested lazy types
            parent_instance_provider: Optional parent context provider for nested lazy types

        Returns:
            List of (field_name, field_type, default_value) tuples for make_dataclass
        """
        from dataclasses import MISSING, is_dataclass

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

            # Check if field type is a dataclass that should be made lazy
            field_type = field.type
            if is_dataclass(field.type) and global_config_type is not None:
                # Create lazy version with automatic inheritance detection and context propagation
                lazy_nested_type = LazyDataclassFactory._create_unified_lazy_class(
                    base_class=field.type,
                    global_config_type=global_config_type,
                    field_name=field.name,
                    lazy_class_name=f"Lazy{field.type.__name__}",
                    parent_field_path=parent_field_path,
                    parent_instance_provider=parent_instance_provider
                )
                field_type = lazy_nested_type
                logger.debug(f"Created lazy class for {field.name}: {field.type} -> {lazy_nested_type}")

            if is_already_optional or not has_default:
                # Field is already Optional or has no default - make it Optional for lazy loading
                final_field_type = Union[field_type, type(None)] if not is_already_optional else field_type
            else:
                # Field has default - preserve type (don't make Optional)
                final_field_type = field_type

            lazy_field_definitions.append((field.name, final_field_type, None))

            # Debug logging with provided template (reduced to DEBUG level to reduce log pollution)
            logger.debug(debug_template.format(
                field_name=field.name,
                original_type=field.type,
                has_default=has_default,
                final_type=final_field_type
            ))

        return lazy_field_definitions

    @staticmethod
    def _create_lazy_dataclass_unified(
        base_class: Type,
        instance_provider: Callable[[], Any],
        lazy_class_name: str,
        debug_template: str,
        use_recursive_resolution: bool = False,
        fallback_chain: Optional[List[Callable[[str], Any]]] = None,
        global_config_type: Type = None,
        parent_field_path: str = None,
        parent_instance_provider: Optional[Callable[[], Any]] = None
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
                instance = instance_provider()
                return _get_raw_field_value(instance, field_name) if instance else None

            resolution_config = ResolutionConfig(
                instance_provider=instance_provider,
                fallback_chain=[safe_instance_fallback, create_static_defaults_fallback(base_class)]
            )

        # Create lazy dataclass with introspected fields
        lazy_class = make_dataclass(
            lazy_class_name,
            LazyDataclassFactory._introspect_dataclass_fields(
                base_class, debug_template, global_config_type, parent_field_path, parent_instance_provider
            ),
            frozen=True
        )

        # Bind methods declaratively
        LazyDataclassFactory._bind_methods_to_class(lazy_class, base_class, resolution_config)

        # Automatically register the lazy dataclass with the type registry
        from openhcs.core.config import register_lazy_type_mapping
        register_lazy_type_mapping(lazy_class, base_class)

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
        DEPRECATED: Use make_lazy_with_field_level_auto_hierarchy() instead.

        This method is deprecated in favor of the superior auto-hierarchy approach
        which provides the same functionality plus sophisticated field-level inheritance.

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
            # Root instance with auto-hierarchy resolution
            PipelineConfig = make_lazy_with_field_level_auto_hierarchy(
                GlobalPipelineConfig,
                field_path=None
            )

            # Nested field with auto-hierarchy resolution
            LazyStepMaterializationConfig = make_lazy_with_field_level_auto_hierarchy(
                StepMaterializationConfig,
                field_path="materialization_defaults"
            )
        """
        import warnings
        warnings.warn(
            "make_lazy_thread_local() is deprecated. Use make_lazy_with_field_level_auto_hierarchy() instead.",
            DeprecationWarning,
            stacklevel=2
        )

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
            CONSTANTS.THREAD_LOCAL_FIELD_DEBUG_TEMPLATE, use_recursive_resolution, fallback_chain,
            global_config_type, field_path
        )

    @staticmethod
    def make_lazy_with_field_level_auto_hierarchy(
        base_class: Type,
        global_config_type: Type,
        field_path: str = None,
        lazy_class_name: str = None,
        context_provider: Optional[Callable[[], Any]] = None
    ) -> Type:
        """
        Create lazy dataclass with automatically discovered field-level hierarchy resolution.

        Preserves sophisticated field-level inheritance while using automatic type introspection
        to discover hierarchy relationships, eliminating the need for manual configuration.
        Now supports context-aware resolution for sibling inheritance within instances.

        Args:
            base_class: The dataclass type to make lazy
            global_config_type: The global config type for thread-local resolution
            field_path: Optional field path for the current instance
            lazy_class_name: Optional name for the generated lazy class
            context_provider: Optional function that provides the resolution context.
                             If None, uses global config. If provided, uses the returned instance.

        Returns:
            Generated lazy dataclass with field-level auto-hierarchy resolution
        """
        # Generate class name if not provided
        if lazy_class_name is None:
            lazy_class_name = f"Lazy{base_class.__name__}"

        # Create field-level hierarchy provider with context support
        field_level_provider = create_field_level_hierarchy_provider(
            base_class=base_class,
            global_config_type=global_config_type,
            current_field_path=field_path,
            context_provider=context_provider
        )

        # Use field-level provider with optional static defaults fallback
        # Static fallbacks should only be used in specific contexts (tests, serialization, etc.)
        # In normal app operation, thread-local storage should always be available
        def context_aware_static_fallback(field_name: str) -> Any:
            """Static fallback that warns when used in contexts where thread-local storage should exist."""
            import logging
            logger = logging.getLogger(__name__)

            # Check if we're in a context where thread-local storage should exist
            get_current_global_config, _ = _get_generic_config_imports()
            current_context = get_current_global_config(global_config_type)

            if current_context is None:
                # Check if we're in a PyQt app context where this shouldn't happen
                try:
                    from PyQt6.QtWidgets import QApplication
                    app_instance = QApplication.instance()
                    if app_instance and hasattr(app_instance, 'global_config'):
                        logger.warning(
                            f"🚨 ARCHITECTURE WARNING: Static fallback used for {base_class.__name__}.{field_name} "
                            f"in PyQt app context where thread-local storage should be available. "
                            f"This indicates a context management bug."
                        )
                except ImportError:
                    pass  # PyQt not available, this is expected in some contexts

            # Use static default
            default_instance = base_class()
            return _get_raw_field_value(default_instance, field_name)

        fallback_chain = [context_aware_static_fallback]

        # Create parent instance provider for context propagation
        def parent_instance_provider_for_nested():
            if context_provider:
                return context_provider()
            return None

        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class=base_class,
            instance_provider=field_level_provider,
            lazy_class_name=lazy_class_name,
            debug_template=f"Field-level auto-hierarchy resolution for {base_class.__name__}",
            use_recursive_resolution=False,
            fallback_chain=fallback_chain,
            global_config_type=global_config_type,
            parent_field_path=field_path,
            parent_instance_provider=parent_instance_provider_for_nested
        )

    # Deprecated methods removed - use make_lazy_with_field_level_auto_hierarchy() for all use cases


def create_field_level_hierarchy_provider(
    base_class: Type,
    global_config_type: Type,
    current_field_path: str = None,
    context_provider: Optional[Callable[[], Any]] = None
):
    """
    Create field-level hierarchy provider that preserves sophisticated inheritance logic.

    This maintains the current field-by-field inheritance behavior while using
    automatic hierarchy discovery to replace manual fallback chain configuration.
    Now supports context-aware resolution for sibling inheritance within instances.

    Args:
        base_class: The dataclass type to create hierarchy provider for
        global_config_type: The global config type for thread-local resolution
        current_field_path: Optional field path for the current instance
        context_provider: Optional function that provides the resolution context.
                         If None, uses global config. If provided, uses the returned instance.

    Returns:
        Provider function that creates instances with field-level hierarchy resolution
    """
    from openhcs.core.field_path_detection import FieldPathDetector
    from dataclasses import fields

    # Auto-discover hierarchy paths
    all_field_paths = FieldPathDetector.find_all_field_paths_for_type(
        global_config_type, base_class
    )
    parent_types = FieldPathDetector.find_inheritance_relationships(base_class)
    sibling_paths = []
    for parent_type in parent_types:
        sibling_paths.extend(
            FieldPathDetector.find_all_field_paths_for_type(global_config_type, parent_type)
        )

    # Determine field classifications for sophisticated inheritance
    if parent_types:
        parent_dataclass = parent_types[0]  # Primary parent
        parent_fields = frozenset(f.name for f in fields(parent_dataclass))
        child_fields = frozenset(f.name for f in fields(base_class))
        inherited_fields = parent_fields & child_fields
        own_fields = child_fields - parent_fields
    else:
        inherited_fields = frozenset()
        own_fields = frozenset(f.name for f in fields(base_class))

    def field_level_provider():
        """Provider that implements field-level inheritance logic with auto-discovered hierarchy."""
        if context_provider:
            current_config = context_provider()
        else:
            get_current_global_config, _ = _get_generic_config_imports()
            current_config = get_current_global_config(global_config_type)

        # Get actual global config from app (if PyQt6 app is running)
        actual_global_config = None
        try:
            from PyQt6.QtWidgets import QApplication
            app_instance = QApplication.instance()
            if app_instance and hasattr(app_instance, 'global_config'):
                actual_global_config = app_instance.global_config
                is_pipeline_context = current_config is not actual_global_config
            else:
                # No PyQt6 app running or no global_config - assume pipeline context
                is_pipeline_context = True
        except ImportError:
            # PyQt6 not available - assume pipeline context
            is_pipeline_context = True



        class FieldLevelInheritanceConfig:
            def __init__(self):
                # Build complete hierarchy path list with inheritance logic
                hierarchy_paths = self._build_hierarchy_paths(
                    current_field_path, all_field_paths, sibling_paths
                )

                # Process all fields using unified resolution
                for field_name in inherited_fields | own_fields:
                    is_inherited = field_name in inherited_fields
                    field_value = self._resolve_field_through_hierarchy(
                        field_name, current_config, actual_global_config, hierarchy_paths, is_inherited
                    )
                    setattr(self, field_name, field_value)

            def _build_hierarchy_paths(self, current_path, same_type_paths, parent_paths):
                """Build hierarchy with current and global contexts."""
                hierarchy = []

                # Current context paths
                if current_path:
                    hierarchy.append(('current', current_path))
                hierarchy.extend(('current', path) for path in parent_paths)

                # Global context paths (if different from current)
                if is_pipeline_context:
                    if current_path:
                        hierarchy.append(('global', current_path))
                    hierarchy.extend(('global', path) for path in parent_paths)
                else:
                    hierarchy.extend(('current', path) for path in same_type_paths if path != current_path)

                return hierarchy

            def _resolve_field_through_hierarchy(self, field_name, current_config, actual_global_config, hierarchy_paths, is_inherited):
                """Resolve field through hierarchy with inheritance-aware logic."""
                # Special case: if no hierarchy paths (e.g., top-level config), resolve directly from current context
                if not hierarchy_paths:
                    # First try current config (might be masked)
                    if current_config:
                        value = _get_raw_field_value(current_config, field_name)
                        if value is not None:
                            return value

                    # If current config is None or masked, try actual global config
                    if actual_global_config:
                        value = _get_raw_field_value(actual_global_config, field_name)
                        if value is not None:
                            return value

                for context_type, path in hierarchy_paths:
                    config = current_config if context_type == 'current' else actual_global_config
                    instance = FieldPathNavigator.navigate_to_instance(config, path)

                    if instance:
                        value = _get_raw_field_value(instance, field_name)

                        if is_inherited:
                            if value is not None and value != "":
                                return value
                        else:
                            if value is not None:
                                return value

                return None

        return FieldLevelInheritanceConfig()

    return field_level_provider


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

        # If the field is a concrete nested dataclass, recursively rebuild it
        if raw_value is not None and hasattr(raw_value, '__dataclass_fields__'):
            # This is a concrete nested dataclass - recursively rebuild to preserve concrete values
            try:
                rebuilt_nested_value = rebuild_lazy_config_with_new_global_reference(
                    raw_value, new_global_config, global_config_type
                )
                current_field_values[field_obj.name] = rebuilt_nested_value
            except Exception:
                # If recursive rebuild fails, keep original value
                current_field_values[field_obj.name] = raw_value
        else:
            # Regular field (None or non-dataclass value) - preserve as-is
            current_field_values[field_obj.name] = raw_value

    # Create new lazy config instance with preserved field values
    # This maintains the exact state: None values stay None, concrete values stay concrete
    # Nested dataclasses are updated to reference new global config
    lazy_class_type = type(existing_lazy_config)
    return lazy_class_type(**current_field_values)

