"""
Factory functions for creating streaming configuration classes.

This module contains OpenHCS-specific utilities for generating streaming configs
with minimal boilerplate. Keeps openhcs/core/config.py purely declarative.
"""

from typing import Optional, List, Type, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from abc import ABC

if TYPE_CHECKING:
    from openhcs.core.config import (
        GlobalPipelineConfig,
        PipelineConfig,
        StreamingConfig,
    )


def create_streaming_config(
    viewer_name: str,
    port: int,
    backend,  # Backend enum
    display_config_class,
    visualizer_module: str,
    visualizer_class_name: str,
    extra_fields: dict = None,
    preview_label: str = None,
    abbreviation: str = None,
) -> Type["StreamingConfig"]:
    """
    Factory to create streaming config classes with minimal boilerplate.

    Eliminates duplication between streaming configs by auto-generating classes
    from declarative specifications. Adding a new streaming backend requires only
    5-10 lines instead of ~50 lines of boilerplate.

    Args:
        viewer_name: Viewer identifier ('napari', 'fiji', etc.)
        port: Default port number
        backend: Backend enum value
        display_config_class: Display config class to inherit from
        visualizer_module: Module path for visualizer class
        visualizer_class_name: Name of visualizer class
        extra_fields: Optional dict of {field_name: (type, default_value)}
        preview_label: Short label for list item previews (e.g., "NAP", "FIJI")
        abbreviation: Short abbreviation for config class name in grouped previews (e.g., "nap", "fiji")

    Returns:
        Dynamically created streaming config class

    Example:
        >>> NapariStreamingConfig = create_streaming_config(
        ...     viewer_name='napari',
        ...     port=5555,
        ...     backend=Backend.NAPARI_STREAM,
        ...     display_config_class=NapariDisplayConfig,
        ...     visualizer_module='openhcs.runtime.napari_stream_visualizer',
        ...     visualizer_class_name='NapariStreamVisualizer'
        ... )
    """
    # Import here to avoid circular dependencies
    from openhcs.core.config import StreamingConfig

    # Get the global_pipeline_config decorator from config module
    # It's created by @auto_create_decorator on GlobalPipelineConfig
    import openhcs.core.config as config_module

    global_pipeline_config = getattr(config_module, "global_pipeline_config", None)
    if global_pipeline_config is None:
        raise RuntimeError(
            "global_pipeline_config decorator not found. Import openhcs.core.config first."
        )

    # Build class namespace with methods
    def _get_streaming_kwargs(self, context):
        kwargs = {
            "port": self.port,
            "host": self.host,
            "transport_mode": self.transport_mode,
            "display_config": self,
        }
        # Add extra fields to kwargs
        if extra_fields:
            for field_name in extra_fields:
                kwargs[field_name] = getattr(self, field_name)
        if context:
            kwargs["microscope_handler"] = context.microscope_handler
            kwargs["plate_path"] = context.plate_path
        return kwargs

    def _create_visualizer(self, filemanager, visualizer_config):
        # Lazy import to avoid circular dependencies
        module = __import__(visualizer_module, fromlist=[visualizer_class_name])
        visualizer_class = getattr(module, visualizer_class_name)
        return visualizer_class(
            filemanager,
            visualizer_config,
            viewer_title=f"OpenHCS {viewer_name.title()} Visualization",
            persistent=self.persistent,
            port=self.port,
            display_config=self,
            transport_mode=self.transport_mode,
        )

    # Compute registry key early (snake_case class name)
    import re
    cls_name = f"{viewer_name.title()}StreamingConfig"
    registry_key = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()

    # Build class dict with properties using lambdas
    class_dict = {
        "port": port,
        "backend": property(lambda self: backend),
        "viewer_type": property(lambda self: viewer_name),
        "streaming_config_key": property(lambda self: registry_key),
        "display_name": property(lambda self: viewer_name.replace("_", " ").title()),
        "step_plan_output_key": property(lambda self: f"{viewer_name}_streaming_paths"),
        "get_streaming_kwargs": _get_streaming_kwargs,
        "create_visualizer": _create_visualizer,
        "__annotations__": {"port": int},
        "__module__": "openhcs.core.config",  # Make it appear as if defined in config.py
    }

    # Add extra fields
    if extra_fields:
        for field_name, (field_type, default_val) in extra_fields.items():
            class_dict[field_name] = default_val
            class_dict["__annotations__"][field_name] = field_type

    # Set the registry key (snake_case class name) for AutoRegisterMeta
    class_dict["_streaming_config_key"] = registry_key

    new_class = type(cls_name, (StreamingConfig, display_config_class), class_dict)

    # Apply decorators
    new_class = dataclass(frozen=True)(new_class)

    # Build decorator kwargs
    decorator_kwargs = {}
    if preview_label is not None:
        decorator_kwargs["preview_label"] = preview_label
    if abbreviation is not None:
        decorator_kwargs["abbreviation"] = abbreviation

    if decorator_kwargs:
        new_class = global_pipeline_config(**decorator_kwargs)(new_class)
    else:
        new_class = global_pipeline_config(new_class)

    return new_class


def build_component_order():
    """
    Build canonical streaming component order from real filename components.

    Returns:
        List of component names in canonical order for layer/window naming
    """
    from openhcs.constants import AllComponents

    filename_component_names = [ac.value for ac in AllComponents]

    component_order = []
    seen = set()
    for name in filename_component_names:
        if name not in seen:
            component_order.append(name)
            seen.add(name)

    return component_order


def get_all_streaming_ports(
    config: "Union[GlobalPipelineConfig, PipelineConfig]" = None,
    num_ports_per_type: int = 10,
) -> List[int]:
    """Get all streaming ports for all registered streaming config types.

    Extracts actual configured ports from the provided config (GlobalPipelineConfig
    or PipelineConfig). This ensures the scanner finds viewers launched with custom
    ports from the orchestrator.

    Args:
        config: GlobalPipelineConfig or PipelineConfig to extract ports from.
                If None, uses current GlobalPipelineConfig from context.
        num_ports_per_type: Number of ports to allocate per streaming type (default: 10)

    Returns:
        List of all streaming ports across all types
    """
    from openhcs.constants.constants import DEFAULT_EXECUTION_SERVER_PORT
    from openhcs.core.config import StreamingConfig, GlobalPipelineConfig
    from openhcs.config_framework.global_config import get_current_global_config

    # Start with execution server port
    ports = [DEFAULT_EXECUTION_SERVER_PORT]

    # Get config to extract ports from
    if config is None:
        config = get_current_global_config(GlobalPipelineConfig)
        if config is None:
            # No config available - return just execution server port
            return ports

    # Extract all streaming config fields from the config
    # Works for both GlobalPipelineConfig and PipelineConfig
    import dataclasses
    import typing
    import inspect

    # Get global config to extract default ports from concrete streaming configs
    global_config = get_current_global_config(GlobalPipelineConfig)

    for field in dataclasses.fields(config):
        # Check if the field TYPE is a StreamingConfig subclass (not the value!)
        # This is critical for PipelineConfig where all fields default to None
        field_type = field.type

        # Handle Optional[StreamingConfig] types
        if (
            hasattr(typing, "get_origin")
            and typing.get_origin(field_type) is typing.Union
        ):
            # Extract the non-None type from Optional[T]
            args = typing.get_args(field_type)
            field_type = next((arg for arg in args if arg is not type(None)), None)

        # Check if field_type is a StreamingConfig subclass
        if (
            field_type is not None
            and inspect.isclass(field_type)
            and issubclass(field_type, StreamingConfig)
        ):
            # Get the port from the field value or global config
            field_value = getattr(config, field.name)

            if field_value is not None:
                # Concrete config - use the instance's port
                port = field_value.port
            elif global_config is not None:
                # Lazy config (PipelineConfig) - get port from global config's concrete instance
                global_field_value = getattr(global_config, field.name, None)
                if global_field_value is not None:
                    port = global_field_value.port
                else:
                    port = None
            else:
                port = None

            # Skip streaming configs that aren't configured
            # (None = not configured, don't scan this port)
            if port is None:
                continue

            # Generate port range for this streaming type
            ports.extend([port + i for i in range(num_ports_per_type)])

    return ports
