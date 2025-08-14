"""
Pipeline-specific configuration classes and utilities.

This module contains all pipeline-specific logic that was previously mixed
into the generic lazy configuration system.
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
    """Set the current pipeline config for MaterializationPathConfig defaults."""
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


# Generate pipeline-specific lazy configuration classes
PipelineConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=GlobalPipelineConfig,
    global_config_type=GlobalPipelineConfig,
    field_path=None,  # Root instance
    lazy_class_name=CONSTANTS.PIPELINE_CONFIG_NAME,
    use_recursive_resolution=True
)

LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_thread_local(
    base_class=StepMaterializationConfig,
    global_config_type=GlobalPipelineConfig,
    field_path=CONSTANTS.MATERIALIZATION_DEFAULTS_PATH,
    lazy_class_name=CONSTANTS.LAZY_STEP_MATERIALIZATION_CONFIG_NAME
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
