"""
Public API for openhcs.

This module provides a stable public API for openhcs, importing selectively
to avoid side-effects. It consolidates user-facing functions and classes from
ez.core and ez.functions.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from openhcs.io.virtual_path import VirtualPath
from openhcs.schemas.config_schemas import PipelineConfig

logger = logging.getLogger(__name__)


def initialize() -> None:
    """
    Initialize openhcs.

    This function initializes all registries and must be called before
    using any other functions in this module.
    """
    # Import registry initialization functions
    # Note: position_generator_registry and image_assembler_registry are now initialized on import
    from openhcs.registries.handlers import initialize_handlers
    from openhcs.registries.step_registry import initialize_pipeline_steps
    from openhcs.registries.storage_providers import \
        initialize_storage_providers

    # Initialize registries
    # Note: position_generator_registry and image_assembler_registry are already initialized
    initialize_storage_providers()
    initialize_pipeline_steps()
    initialize_handlers()

    logger.info("Initialized openhcs")


def create_config(
    input_dir: Union[str, VirtualPath],
    output_dir: Optional[Union[str, VirtualPath]] = None,
    backend_type: str = "ashlar",
    use_gpu: bool = False,
    num_workers: int = 1,
    well_filter: Optional[List[str]] = None,
    **kwargs
) -> PipelineConfig:
    """
    Create a pipeline configuration.

    Args:
        input_dir: Input directory containing images to process
        output_dir: Output directory for processed images (default: input_dir + "_out")
        backend_type: Type of backend to use (default: "ashlar")
        use_gpu: Whether to use GPU acceleration if available (default: False)
        num_workers: Number of worker processes to use (default: 1)
        well_filter: List of wells to process (default: None, process all wells)
        **kwargs: Additional configuration parameters

    Returns:
        PipelineConfig object
    """
    # Import here to avoid circular imports
    from openhcs.schemas.config_schemas import BackendConfig, MISTConfig

    # Create backend config
    backend_config = BackendConfig(type=backend_type)

    # Configure MIST backend if selected
    if backend_type == "mist":
        backend_config.mist = MISTConfig(use_gpu=use_gpu)

    # Create pipeline config
    config = PipelineConfig(
        backend=backend_config,
        num_workers=num_workers,
        well_filter=well_filter,
        **kwargs
    )

    # Convert input_dir to VirtualPath if it's a string
    if isinstance(input_dir, str):
        from openhcs.io.virtual_path import PhysicalPath
        input_dir = PhysicalPath(input_dir)

    # Set output_dir if not provided
    if output_dir is None:
        output_dir = input_dir.parent() / f"{input_dir.name()}{config.out_dir_suffix}"
    elif isinstance(output_dir, str):
        from openhcs.io.virtual_path import PhysicalPath
        output_dir = PhysicalPath(output_dir)

    # Add input_dir and output_dir to config
    config.input_dir = input_dir
    config.output_dir = output_dir

    return config


def run_pipeline(
    config: Union[PipelineConfig, Dict[str, Any]],
    input_dir: Optional[Union[str, VirtualPath]] = None,
    output_dir: Optional[Union[str, VirtualPath]] = None
) -> Dict[str, Any]:
    """
    Run a pipeline with the given configuration.

    Args:
        config: Pipeline configuration
        input_dir: Input directory (overrides config.input_dir if provided)
        output_dir: Output directory (overrides config.output_dir if provided)

    Returns:
        Dictionary of results
    """
    # Ensure openhcs is initialized
    initialize()

    # Convert config to PipelineConfig if it's a dictionary
    if isinstance(config, dict):
        config = create_config(**config)

    # Override input_dir and output_dir if provided
    if input_dir is not None:
        if isinstance(input_dir, str):
            from openhcs.io.virtual_path import PhysicalPath
            input_dir = PhysicalPath(input_dir)
        config.input_dir = input_dir

    if output_dir is not None:
        if isinstance(output_dir, str):
            from openhcs.io.virtual_path import PhysicalPath
            output_dir = PhysicalPath(output_dir)
        config.output_dir = output_dir

    # Import here to avoid circular imports
    from openhcs.core.pipeline_executor import PipelineExecutor

    # Create and run pipeline
    executor = PipelineExecutor(config)
    results = executor.run()

    return results


def stitch_images(
    input_dir: Union[str, VirtualPath],
    output_dir: Optional[Union[str, VirtualPath]] = None,
    backend_type: str = "ashlar",
    use_gpu: bool = False,
    num_workers: int = 1,
    well_filter: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Stitch images in the input directory.

    This is a convenience function that creates a config and runs a pipeline.

    Args:
        input_dir: Input directory containing images to stitch
        output_dir: Output directory for stitched images (default: input_dir + "_out")
        backend_type: Type of backend to use (default: "ashlar")
        use_gpu: Whether to use GPU acceleration if available (default: False)
        num_workers: Number of worker processes to use (default: 1)
        well_filter: List of wells to process (default: None, process all wells)
        **kwargs: Additional configuration parameters

    Returns:
        Dictionary of results
    """
    # Create config
    config = create_config(
        input_dir=input_dir,
        output_dir=output_dir,
        backend_type=backend_type,
        use_gpu=use_gpu,
        num_workers=num_workers,
        well_filter=well_filter,
        **kwargs
    )

    # Run pipeline
    return run_pipeline(config)


# Types are already imported at the top of the file