"""
Pipeline-specific configuration classes and utilities.

This module contains all pipeline-specific logic that was previously mixed
into the generic lazy configuration system. Now uses the new generic hierarchy
system while maintaining backward compatibility.
"""

from typing import Any, Type, Optional
from dataclasses import fields
from openhcs.core.config import (
    GlobalPipelineConfig, StepMaterializationConfig,
    set_current_global_config, register_lazy_type_mapping
)
from openhcs.core.lazy_config import (
    LazyDataclassFactory, create_config_for_editing,
    ensure_global_config_context, CONSTANTS
)
from openhcs.core.config_hierarchy import create_openhcs_hierarchy


def set_current_pipeline_config(config: GlobalPipelineConfig) -> None:
    """Set the current pipeline config for LazyStepMaterializationConfig defaults."""
    set_current_global_config(GlobalPipelineConfig, config)


def ensure_pipeline_config_context(orchestrator_global_config: Any) -> None:
    """Ensure proper thread-local storage setup for pipeline configuration editing."""
    ensure_global_config_context(GlobalPipelineConfig, orchestrator_global_config)


def create_pipeline_config_for_editing(
    source_config: Any,
    preserve_values: bool = False
) -> Any:
    """
    Create PipelineConfig for editing - pipeline-specific wrapper.

    Args:
        source_config: Instance to use for context and optionally field values
        preserve_values:
            - True: Preserve actual field values (direct editing)
            - False: Use None values for placeholders (hierarchical editing)

    Returns:
        PipelineConfig instance with appropriate field initialization
    """
    return create_config_for_editing(
        GlobalPipelineConfig,
        source_config,
        preserve_values=preserve_values,
        placeholder_prefix="Pipeline default"
    )


def create_editing_config_from_existing_lazy_config(
    existing_lazy_config: Any,
    global_config: Any
) -> Any:
    """
    Create an editing config from existing lazy config with user-set values preserved as actual field values.

    This function is used when reopening orchestrator config editing to ensure that:
    - User-set values appear as actual field values (not placeholders)
    - Unset fields remain None for placeholder behavior
    - Thread-local context is properly set up

    Args:
        existing_lazy_config: Existing lazy config with user customizations
        global_config: Global config for thread-local context setup

    Returns:
        New lazy config suitable for editing with preserved user values
    """
    if existing_lazy_config is None:
        return None

    # Set up thread-local context with updated global config
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.lazy_config import ensure_global_config_context
    ensure_global_config_context(GlobalPipelineConfig, global_config)

    # Extract field values, preserving user-set values as concrete values
    field_values = {}
    for field_obj in fields(existing_lazy_config):
        # Get raw stored value without triggering lazy resolution
        raw_value = object.__getattribute__(existing_lazy_config, field_obj.name)

        if raw_value is not None:
            # User has explicitly set this field - preserve as concrete value
            # This includes nested dataclasses that have been modified
            field_values[field_obj.name] = raw_value
        else:
            # Field is None - keep as None for placeholder behavior
            field_values[field_obj.name] = None

    return PipelineConfig(**field_values)




# Create OpenHCS hierarchy and generate lazy configuration classes
_openhcs_hierarchy = create_openhcs_hierarchy()

# Generate pipeline-specific lazy configuration classes using the new generic system
PipelineConfig = LazyDataclassFactory.make_lazy_hierarchical(
    GlobalPipelineConfig,
    _openhcs_hierarchy.registry
)

# Create 3-level step materialization hierarchy
from openhcs.core.config_hierarchy import create_step_materialization_hierarchy
_step_hierarchy = create_step_materialization_hierarchy()

# Generate step-level lazy class that uses 3-level hierarchy
LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_hierarchical(
    StepMaterializationConfig,
    _step_hierarchy.registry
)


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


# Add to_base_config method for orchestrator integration
_add_to_base_config_method(PipelineConfig, GlobalPipelineConfig)

# Register type mappings for the placeholder service
register_lazy_type_mapping(PipelineConfig, GlobalPipelineConfig)
register_lazy_type_mapping(LazyStepMaterializationConfig, StepMaterializationConfig)
