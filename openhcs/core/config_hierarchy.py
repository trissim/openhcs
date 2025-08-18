"""
Hierarchical configuration registry for OpenHCS N-level lazy configuration hierarchies.

This module provides the core infrastructure for managing arbitrary-depth configuration
hierarchies with automatic field path detection and lazy class generation.
"""

import dataclasses
import threading
from typing import Type, Optional, Dict, List, Any, Callable
from dataclasses import dataclass

from openhcs.core.hierarchy_introspection import FieldPathDetector


@dataclass(frozen=True)
class ConfigHierarchyNode:
    """Represents a node in the configuration hierarchy."""
    config_type: Type
    parent_node: Optional['ConfigHierarchyNode']
    field_path: Optional[str]  # Auto-detected from parent type
    lazy_class: Optional[Type] = None  # Generated lazy class

    @property
    def is_root(self) -> bool:
        """Check if this is a root node (no parent)."""
        return self.parent_node is None

    @property
    def depth(self) -> int:
        """Get the depth of this node in the hierarchy (root = 0)."""
        if self.is_root:
            return 0
        return self.parent_node.depth + 1

    def get_resolution_chain(self) -> List['ConfigHierarchyNode']:
        """Get the resolution chain from this node to the root."""
        chain = []
        current = self
        while current is not None:
            chain.append(current)
            current = current.parent_node
        return chain


class ConfigHierarchyRegistry:
    """Registry for configuration type hierarchies."""

    def __init__(self):
        self._hierarchies: Dict[Type, ConfigHierarchyNode] = {}
        self._lazy_classes: Dict[Type, Type] = {}
        self._lock = threading.Lock()

    def register_hierarchy_auto(self, root_type: Type, child_types: List[Type]) -> None:
        """
        Register hierarchy with automatic field path detection.

        This method automatically discovers parent-child relationships using
        type introspection and builds the hierarchy graph.

        Args:
            root_type: The root configuration type (e.g., GlobalPipelineConfig)
            child_types: List of child configuration types to include in hierarchy
        """
        with self._lock:
            # Create root node
            root_node = ConfigHierarchyNode(
                config_type=root_type,
                parent_node=None,
                field_path=None
            )
            self._hierarchies[root_type] = root_node

            # Process child types and build hierarchy
            for child_type in child_types:
                self._register_child_type(child_type, root_node)

    def _register_child_type(self, child_type: Type, root_node: ConfigHierarchyNode) -> None:
        """Register a child type by finding its parent in the hierarchy."""
        # Try to find parent by checking field paths
        parent_node = self._find_parent_for_type(child_type, root_node)

        if parent_node is None:
            # If no parent found, make it a direct child of root
            parent_node = root_node

        # Detect field path from parent to child
        field_path = FieldPathDetector.find_field_path_for_type(
            parent_node.config_type, child_type
        )

        # Create child node
        child_node = ConfigHierarchyNode(
            config_type=child_type,
            parent_node=parent_node,
            field_path=field_path
        )

        self._hierarchies[child_type] = child_node

    def _find_parent_for_type(self, child_type: Type, root_node: ConfigHierarchyNode) -> Optional[ConfigHierarchyNode]:
        """Find the most appropriate parent for a child type."""
        # Check all registered nodes to find the best parent
        best_parent = None

        for node in self._hierarchies.values():
            field_path = FieldPathDetector.find_field_path_for_type(
                node.config_type, child_type
            )
            if field_path is not None:
                # Found a potential parent
                if best_parent is None or node.depth > best_parent.depth:
                    # Prefer deeper nodes (more specific parents)
                    best_parent = node

        return best_parent

    def get_hierarchy_node(self, config_type: Type) -> Optional[ConfigHierarchyNode]:
        """Get the hierarchy node for a configuration type."""
        return self._hierarchies.get(config_type)

    def get_resolution_chain(self, config_type: Type) -> List[ConfigHierarchyNode]:
        """Get parent→child resolution chain for a configuration type."""
        node = self.get_hierarchy_node(config_type)
        if node is None:
            return []
        return list(reversed(node.get_resolution_chain()))  # Root to child order

    def create_lazy_classes_for_hierarchy(self, root_type: Type) -> Dict[Type, Type]:
        """Generate all lazy classes for a hierarchy."""
        from openhcs.core.lazy_config import LazyDataclassFactory

        with self._lock:
            root_node = self._hierarchies.get(root_type)
            if root_node is None:
                raise ValueError(f"Root type {root_type} not registered in hierarchy")

            lazy_classes = {}

            # Generate lazy classes for all nodes in the hierarchy
            for config_type, node in self._hierarchies.items():
                if node.parent_node is None:
                    # Root node - create lazy class with recursive resolution
                    lazy_class = LazyDataclassFactory.make_lazy_thread_local(
                        base_class=config_type,
                        global_config_type=root_type,
                        field_path=None,
                        lazy_class_name=f"Lazy{config_type.__name__}",
                        use_recursive_resolution=True
                    )
                else:
                    # Child node - create lazy class with field path
                    lazy_class = LazyDataclassFactory.make_lazy_thread_local(
                        base_class=config_type,
                        global_config_type=root_type,
                        field_path=node.field_path,
                        lazy_class_name=f"Lazy{config_type.__name__}"
                    )

                lazy_classes[config_type] = lazy_class

                # Update node with lazy class
                updated_node = ConfigHierarchyNode(
                    config_type=node.config_type,
                    parent_node=node.parent_node,
                    field_path=node.field_path,
                    lazy_class=lazy_class
                )
                self._hierarchies[config_type] = updated_node

            return lazy_classes


class HierarchicalContextManager:
    """Manages thread-local contexts for configuration hierarchies."""

    def __init__(self, registry: ConfigHierarchyRegistry):
        self.registry = registry
        self._contexts: Dict[Type, threading.local] = {}
        self._lock = threading.Lock()

    def set_context_for_hierarchy(self, root_type: Type, root_instance: Any) -> None:
        """Set context for entire hierarchy."""
        from openhcs.core.config import set_current_global_config

        with self._lock:
            # Set thread-local context for the root type
            set_current_global_config(root_type, root_instance)

    def propagate_changes_down(self, changed_type: Type, new_instance: Any) -> None:
        """Propagate changes down the hierarchy (top-down only)."""
        from openhcs.core.lazy_config import rebuild_lazy_config_with_new_global_reference
        from openhcs.core.config import set_current_global_config

        # Update thread-local context for the changed type
        set_current_global_config(changed_type, new_instance)

        # Find all child types that need to be updated
        child_types = self._find_child_types(changed_type)

        # Propagate changes to each child type
        for child_type in child_types:
            self._propagate_to_child_type(child_type, changed_type, new_instance)

    def _find_child_types(self, parent_type: Type) -> List[Type]:
        """Find all child types that depend on the parent type."""
        child_types = []

        for config_type, node in self.registry._hierarchies.items():
            if node.parent_node and node.parent_node.config_type == parent_type:
                child_types.append(config_type)
                # Recursively find children of children
                child_types.extend(self._find_child_types(config_type))

        return child_types

    def _propagate_to_child_type(self, child_type: Type, root_type: Type, new_root_instance: Any) -> None:
        """Propagate changes to a specific child type."""
        # This is where we would update any existing lazy instances of the child type
        # For now, the propagation happens automatically through thread-local context
        # when lazy instances are accessed, so no explicit action is needed
        pass


class HierarchicalPropagationSystem:
    """Advanced propagation system for N-level configuration hierarchies."""

    def __init__(self, registry: ConfigHierarchyRegistry, context_manager: HierarchicalContextManager):
        self.registry = registry
        self.context_manager = context_manager
        self._propagation_callbacks: Dict[Type, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    def register_propagation_callback(self, config_type: Type, callback: Callable[[Any], None]) -> None:
        """Register a callback to be called when a config type changes."""
        with self._lock:
            if config_type not in self._propagation_callbacks:
                self._propagation_callbacks[config_type] = []
            self._propagation_callbacks[config_type].append(callback)

    def propagate_change(self, changed_type: Type, new_instance: Any, preserve_concrete_values: bool = True) -> None:
        """
        Propagate configuration changes through the hierarchy.

        This method implements top-down propagation that:
        1. Updates the thread-local context for the changed type
        2. Preserves concrete field values in child configurations
        3. Updates lazy resolution for None fields
        4. Calls registered callbacks for affected types

        Args:
            changed_type: The configuration type that changed
            new_instance: The new configuration instance
            preserve_concrete_values: Whether to preserve concrete values in child configs
        """
        from openhcs.core.lazy_config import rebuild_lazy_config_with_new_global_reference

        with self._lock:
            # Update thread-local context
            self.context_manager.set_context_for_hierarchy(changed_type, new_instance)

            # Find the root type for this hierarchy
            root_type = self._find_root_type(changed_type)

            if root_type is None:
                # This type is not part of a registered hierarchy
                return

            # Get all types in the hierarchy
            hierarchy_types = list(self.registry._hierarchies.keys())

            # Propagate to all types in the hierarchy
            for config_type in hierarchy_types:
                node = self.registry.get_hierarchy_node(config_type)
                if node and self._is_affected_by_change(node, changed_type):
                    # Call propagation callbacks for this type
                    self._call_propagation_callbacks(config_type, new_instance)

    def _find_root_type(self, config_type: Type) -> Optional[Type]:
        """Find the root type for a configuration type."""
        node = self.registry.get_hierarchy_node(config_type)
        if node is None:
            return None

        # Walk up to find the root
        while node.parent_node is not None:
            node = node.parent_node

        return node.config_type

    def _is_affected_by_change(self, node: ConfigHierarchyNode, changed_type: Type) -> bool:
        """Check if a node is affected by a change to the given type."""
        # A node is affected if the changed type is in its resolution chain
        resolution_chain = node.get_resolution_chain()
        for chain_node in resolution_chain:
            if chain_node.config_type == changed_type:
                return True
        return False

    def _call_propagation_callbacks(self, config_type: Type, new_instance: Any) -> None:
        """Call all registered callbacks for a configuration type."""
        callbacks = self._propagation_callbacks.get(config_type, [])
        for callback in callbacks:
            try:
                callback(new_instance)
            except Exception as e:
                # Log error but continue with other callbacks
                import logging
                logging.error(f"Error in propagation callback for {config_type}: {e}")

    def rebuild_lazy_instances(self, config_type: Type, existing_instances: List[Any]) -> List[Any]:
        """
        Rebuild existing lazy instances to use updated global context.

        This method preserves concrete field values while updating lazy resolution
        for None fields to use the new global context.

        Args:
            config_type: The configuration type to rebuild
            existing_instances: List of existing lazy instances

        Returns:
            List of rebuilt lazy instances with preserved concrete values
        """
        from openhcs.core.lazy_config import rebuild_lazy_config_with_new_global_reference

        # Find the root type for this hierarchy
        root_type = self._find_root_type(config_type)
        if root_type is None:
            return existing_instances

        # Get the current global config
        from openhcs.core.config import get_current_global_config
        current_global_config = get_current_global_config(root_type)

        if current_global_config is None:
            return existing_instances

        # Rebuild each instance
        rebuilt_instances = []
        for instance in existing_instances:
            try:
                rebuilt_instance = rebuild_lazy_config_with_new_global_reference(
                    instance, current_global_config, root_type
                )
                rebuilt_instances.append(rebuilt_instance)
            except Exception as e:
                # If rebuild fails, keep the original instance
                import logging
                logging.warning(f"Failed to rebuild lazy instance of {config_type}: {e}")
                rebuilt_instances.append(instance)

        return rebuilt_instances


class HierarchyBuilder:
    """Declarative API for building configuration hierarchies."""

    def __init__(self):
        self.registry = ConfigHierarchyRegistry()
        self.context_manager = HierarchicalContextManager(self.registry)
        self.propagation_system = HierarchicalPropagationSystem(self.registry, self.context_manager)
        self._root_type = None
        self._child_types = []

    def set_root(self, root_type: Type) -> 'HierarchyBuilder':
        """Set the root configuration type for the hierarchy."""
        self._root_type = root_type
        return self

    def add_child(self, child_type: Type) -> 'HierarchyBuilder':
        """Add a child configuration type to the hierarchy."""
        self._child_types.append(child_type)
        return self

    def add_children(self, *child_types: Type) -> 'HierarchyBuilder':
        """Add multiple child configuration types to the hierarchy."""
        self._child_types.extend(child_types)
        return self

    def build(self) -> 'ConfigHierarchy':
        """Build the configuration hierarchy."""
        if self._root_type is None:
            raise ValueError("Root type must be set before building hierarchy")

        # Register the hierarchy
        self.registry.register_hierarchy_auto(self._root_type, self._child_types)

        # Create lazy classes for all types
        lazy_classes = self.registry.create_lazy_classes_for_hierarchy(self._root_type)

        return ConfigHierarchy(
            registry=self.registry,
            context_manager=self.context_manager,
            propagation_system=self.propagation_system,
            lazy_classes=lazy_classes,
            root_type=self._root_type
        )


class ConfigHierarchy:
    """Represents a built configuration hierarchy with all lazy classes."""

    def __init__(
        self,
        registry: ConfigHierarchyRegistry,
        context_manager: HierarchicalContextManager,
        propagation_system: HierarchicalPropagationSystem,
        lazy_classes: Dict[Type, Type],
        root_type: Type
    ):
        self.registry = registry
        self.context_manager = context_manager
        self.propagation_system = propagation_system
        self.lazy_classes = lazy_classes
        self.root_type = root_type

    def get_lazy_class(self, config_type: Type) -> Type:
        """Get the lazy class for a configuration type."""
        lazy_class = self.lazy_classes.get(config_type)
        if lazy_class is None:
            raise ValueError(f"No lazy class found for {config_type}")
        return lazy_class

    def set_global_config(self, config_instance: Any) -> None:
        """Set the global configuration instance."""
        self.context_manager.set_context_for_hierarchy(self.root_type, config_instance)

    def propagate_changes(self, changed_type: Type, new_instance: Any) -> None:
        """Propagate configuration changes through the hierarchy."""
        self.propagation_system.propagate_change(changed_type, new_instance)

    def get_all_lazy_classes(self) -> Dict[Type, Type]:
        """Get all lazy classes in the hierarchy."""
        return self.lazy_classes.copy()


# Decorator API for automatic hierarchy creation
def auto_hierarchy(parent_type: Type, child_types: List[Type] = None):
    """
    Decorator for automatic hierarchy creation.

    This decorator automatically creates a hierarchy for the decorated class
    and provides lazy class generation.

    Args:
        parent_type: The parent configuration type
        child_types: Optional list of child types (auto-detected if not provided)

    Usage:
        @auto_hierarchy(parent_type=GlobalPipelineConfig)
        class MyStepConfig:
            pass

        # The decorator will automatically create lazy classes and hierarchy
    """
    def decorator(cls):
        # Store hierarchy metadata on the class
        cls._hierarchy_parent = parent_type
        cls._hierarchy_children = child_types or []

        # Create hierarchy builder
        builder = HierarchyBuilder()
        builder.set_root(parent_type)

        if child_types:
            builder.add_children(*child_types)
        else:
            # Auto-detect child types by scanning for dataclass fields
            builder.add_child(cls)

        # Build hierarchy and store on class
        cls._hierarchy = builder.build()

        # Add convenience methods to the class
        @classmethod
        def get_lazy_class(cls_inner):
            return cls._hierarchy.get_lazy_class(cls_inner)

        @classmethod
        def set_global_config(cls_inner, config_instance):
            cls._hierarchy.set_global_config(config_instance)

        cls.get_lazy_class = get_lazy_class
        cls.set_global_config = set_global_config

        return cls

    return decorator


# Convenience functions for common hierarchy patterns
def create_openhcs_hierarchy() -> ConfigHierarchy:
    """Create the standard OpenHCS configuration hierarchy."""
    from openhcs.core.config import (
        GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig,
        StepMaterializationConfig, AnalysisConsolidationConfig,
        PlateMetadataConfig, FunctionRegistryConfig
    )

    return (HierarchyBuilder()
            .set_root(GlobalPipelineConfig)
            .add_children(
                PathPlanningConfig, VFSConfig, ZarrConfig, StepMaterializationConfig,
                AnalysisConsolidationConfig, PlateMetadataConfig, FunctionRegistryConfig
            )
            .build())


def create_simple_hierarchy(root_type: Type, *child_types: Type) -> ConfigHierarchy:
    """Create a simple hierarchy with one root and multiple children."""
    return (HierarchyBuilder()
            .set_root(root_type)
            .add_children(*child_types)
            .build())


def create_step_materialization_hierarchy() -> ConfigHierarchy:
    """
    Create the 3-level step materialization hierarchy using the generic N-level system.

    This demonstrates the N-level hierarchy system working for a specific 3-level case:
    - Level 1 (Global): GlobalPipelineConfig.materialization_defaults
    - Level 2 (Orchestrator): PipelineConfig.materialization_defaults
    - Level 3 (Step): Individual step LazyStepMaterializationConfig instances

    The resolution chain: Step → Orchestrator → Global

    Returns:
        ConfigHierarchy instance with 3-level step materialization structure
    """
    from openhcs.core.config import GlobalPipelineConfig, StepMaterializationConfig

    # Use the HierarchyBuilder to properly create the hierarchy
    # This will automatically detect that StepMaterializationConfig is found
    # in GlobalPipelineConfig.materialization_defaults
    builder = HierarchyBuilder()
    hierarchy = (builder
                 .set_root(GlobalPipelineConfig)
                 .add_child(StepMaterializationConfig)
                 .build())

    return hierarchy





class HierarchicalResolutionProvider:
    """Provides N-level resolution chains for configuration hierarchies."""

    def __init__(self, registry: ConfigHierarchyRegistry):
        self.registry = registry

    def create_resolution_chain(self, config_type: Type) -> List[Callable[[str], Any]]:
        """
        Create resolution chain for arbitrary depth hierarchies.

        Returns a list of resolution functions that implement the pattern:
        Context Stack → Level N → Level N-1 → ... → Level 1 → Global defaults

        Args:
            config_type: The configuration type to create resolution chain for

        Returns:
            List of resolution functions in order of precedence
        """
        from openhcs.core.config import get_current_global_config
        from openhcs.core.lazy_config import FieldPathNavigator, create_static_defaults_fallback, get_context_stack

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
            return object.__getattribute__(obj, field_name) if hasattr(obj, field_name) else None

        # Get the hierarchy chain for this config type
        hierarchy_chain = self.registry.get_resolution_chain(config_type)

        resolution_functions = []

        # Add context stack resolver as first priority (3-level hierarchy support)
        def context_stack_resolver(field_name: str) -> Any:
            """Resolve field from context stack (Step → Orchestrator → Global)."""
            context_stack = get_context_stack(config_type)
            # Traverse stack from top (most specific) to bottom (most general)
            for context in reversed(context_stack):
                value = _get_raw_field_value(context, field_name)
                if value is not None:
                    return value
            return None

        resolution_functions.append(context_stack_resolver)

        if not hierarchy_chain:
            # No hierarchy found, use context stack + static defaults only
            resolution_functions.append(create_static_defaults_fallback(config_type))
            return resolution_functions

        # Create resolution functions for each level in the hierarchy
        for i, node in enumerate(hierarchy_chain):
            if node.is_root:
                # Root level - resolve from thread-local global config
                def root_resolver(field_name: str, root_type=node.config_type) -> Any:
                    current_config = get_current_global_config(root_type)
                    return _get_raw_field_value(current_config, field_name) if current_config else None

                resolution_functions.append(root_resolver)
            else:
                # Child level - resolve through field path navigation
                def child_resolver(field_name: str, node=node) -> Any:
                    # Get the root config from thread-local storage
                    root_type = hierarchy_chain[0].config_type
                    current_config = get_current_global_config(root_type)

                    if current_config is not None:
                        # Navigate to the specific config instance for this level
                        config_instance = FieldPathNavigator.navigate_to_instance(
                            current_config, node.field_path
                        )
                        return _get_raw_field_value(config_instance, field_name) if config_instance else None
                    return None

                resolution_functions.append(child_resolver)

        # Add static defaults as final fallback
        resolution_functions.append(create_static_defaults_fallback(config_type))

        return resolution_functions

    def resolve_field_value(self, config_type: Type, field_name: str) -> Any:
        """
        Resolve a field value using the N-level resolution chain.

        Args:
            config_type: The configuration type
            field_name: The field name to resolve

        Returns:
            The resolved field value, or None if not found
        """
        resolution_chain = self.create_resolution_chain(config_type)

        # Try each resolution function in order
        for resolver in resolution_chain:
            try:
                value = resolver(field_name)
                if value is not None:
                    return value
            except (AttributeError, Exception):
                # Continue to next resolver on any error
                continue

        return None

    def create_hierarchical_lazy_class(self, config_type: Type) -> Type:
        """
        Create a lazy class with hierarchical resolution for any config type.

        This method creates a lazy class that uses the N-level resolution chain
        instead of being hardcoded to a specific hierarchy depth.

        Args:
            config_type: The configuration type to make lazy

        Returns:
            Generated lazy class with hierarchical resolution
        """
        from openhcs.core.lazy_config import LazyDataclassFactory, ResolutionConfig

        # Get the hierarchy chain
        hierarchy_chain = self.registry.get_resolution_chain(config_type)

        if not hierarchy_chain:
            # No hierarchy, create simple lazy class with static defaults
            return LazyDataclassFactory.create_lazy_dataclass(
                config_type,
                f"Hierarchical{config_type.__name__}"
            )

        # Create resolution configuration with hierarchical chain
        resolution_functions = self.create_resolution_chain(config_type)

        # Create instance provider (not used for hierarchical resolution)
        def dummy_instance_provider():
            return None

        # Create lazy class with hierarchical resolution
        return LazyDataclassFactory._create_lazy_dataclass_unified(
            base_class=config_type,
            instance_provider=dummy_instance_provider,
            lazy_class_name=f"Hierarchical{config_type.__name__}",
            debug_template="HIERARCHICAL LAZY FIELD: {field_name} - original={original_type}, has_default={has_default}, final={final_type}",
            use_recursive_resolution=True,
            fallback_chain=resolution_functions
        )