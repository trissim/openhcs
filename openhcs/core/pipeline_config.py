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
    # Create lazy PipelineConfig instance for editing with placeholder behavior
    from openhcs.core.lazy_config import create_dataclass_for_editing
    return create_dataclass_for_editing(
        PipelineConfig,  # Use lazy PipelineConfig type, not GlobalPipelineConfig
        source_config,
        preserve_values=preserve_values,
        context_provider=lambda config: ensure_pipeline_config_context(config)
    )


def create_editing_config_from_existing_lazy_config(
    existing_lazy_config: Any,
    global_config: Optional[Any] = None  # NEW: Optional parameter
) -> Any:
    """
    Create an editing config from existing lazy config with user-set values preserved as actual field values.

    This function is used when reopening orchestrator config editing to ensure that:
    - User-set values appear as actual field values (not placeholders)
    - Unset fields remain None for placeholder behavior
    - Thread-local context is properly set up

    Args:
        existing_lazy_config: Existing lazy config with user customizations
        global_config: Optional global config for thread-local context setup.
                      If None, uses existing thread-local context (caller-responsible pattern).

    Returns:
        New lazy config suitable for editing with preserved user values
    """
    if existing_lazy_config is None:
        return None

    # Set up thread-local context only if global_config is provided
    if global_config is not None:
        # Legacy behavior - caller provides global_config
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.lazy_config import ensure_global_config_context
        ensure_global_config_context(GlobalPipelineConfig, global_config)
    # Otherwise use existing thread-local context (caller-responsible pattern)

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

    return type(existing_lazy_config)(**field_values)






# Generate pipeline-specific lazy configuration classes using thread-local resolution
PipelineConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=GlobalPipelineConfig,
    global_config_type=GlobalPipelineConfig,
    field_path=None,
    lazy_class_name="PipelineConfig",
    use_recursive_resolution=True
)

# Generate step-level lazy class using thread-local resolution
# No recursive resolution to avoid circular dependencies with PathPlanningConfig inheritance

# Automatically detect field path instead of hardcoding
from openhcs.core.hierarchy_introspection import FieldPathDetector
_step_materialization_field_path = FieldPathDetector.find_field_path_for_type(
    GlobalPipelineConfig, StepMaterializationConfig
)

LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=StepMaterializationConfig,
    global_config_type=GlobalPipelineConfig,
    field_path=_step_materialization_field_path,
    lazy_class_name="LazyStepMaterializationConfig",
    use_recursive_resolution=False  # Disable to prevent recursion with inherited fields
)


def _add_to_base_config_method(lazy_class: Type, base_class: Type) -> None:
    """Add to_base_config method to lazy dataclass for orchestrator integration."""
    def to_base_config(self):
        """Convert lazy config to base config, resolving None values to current defaults."""
        resolved_values = {}
        for field in fields(self):
            # Get raw value first to avoid triggering lazy resolution on nested lazy dataclasses
            raw_value = object.__getattribute__(self, field.name)

            if raw_value is not None:
                # User has set this field - use the raw value
                # If it's a nested lazy dataclass, convert it to base config recursively
                if hasattr(raw_value, 'to_base_config'):
                    resolved_values[field.name] = raw_value.to_base_config()
                else:
                    resolved_values[field.name] = raw_value
            else:
                # Field is None - resolve lazily (safe because it goes to concrete global config)
                resolved_values[field.name] = getattr(self, field.name)

        return base_class(**resolved_values)

    # Bind the method to the lazy class
    lazy_class.to_base_config = to_base_config


# Add to_base_config method for orchestrator integration
_add_to_base_config_method(PipelineConfig, GlobalPipelineConfig)

# Register type mappings for the placeholder service
register_lazy_type_mapping(PipelineConfig, GlobalPipelineConfig)
register_lazy_type_mapping(LazyStepMaterializationConfig, StepMaterializationConfig)
