"""
Core implementation of the EZ module.

This module provides the OpenHCS class, which is a simplified interface
for common stitching workflows.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

from openhcs.core import AutoPipelineFactory
from openhcs.core.pipeline_orchestrator import PipelineOrchestrator


class OpenHCS:
    """
    Simplified interface for microscopy image stitching.

    This class provides an easy-to-use interface for common stitching workflows,
    hiding the complexity of pipelines and orchestrators.
    """

    def __init__(self,
                 input_path: Union[str, Path, 'VirtualPath'],
                 output_path: Optional[Union[str, Path, 'VirtualPath']] = None,
                 normalize: bool = True,
                 flatten_z: Optional[bool] = None,
                 z_method: str = "max",
                 channel_weights: Optional[List[float]] = None,
                 well_filter: Optional[List[str]] = None,
                 storage_mode: Literal["legacy", "memory", "zarr"] = "legacy", # Added
                 storage_root: Optional[Union[str, Path, 'VirtualPath']] = None): # Added
        """
        Initialize with minimal required parameters.

        Args:
            input_path: Path to the plate folder
            output_path: Path for output (default: input_path + "_stitched")
            normalize: Whether to apply normalization
            flatten_z: Whether to flatten Z-stacks (auto-detected if None)
            z_method: Method for Z-flattening ("max", "mean", "focus", etc.)
            channel_weights: Weights for channel compositing (auto-detected if None)
            well_filter: List of wells to process (processes all if None)
            storage_mode: Mode for intermediate storage ('legacy', 'memory', 'zarr').
            storage_root: Root directory for disk-based storage modes (e.g., 'zarr').
        """
        # 🔒 Clause 17 — VFS Boundary
        # Convert paths to VirtualPath objects
        from openhcs.io.virtual_path import VirtualPath
        from openhcs.io.virtual_path_factory import VirtualPathFactory

        # Convert input_path to VirtualPath if it's not already
        if not isinstance(input_path, VirtualPath):
            self.input_path = VirtualPathFactory.from_path(input_path)
        else:
            self.input_path = input_path

        # Auto-generate output path if not provided
        if output_path is None:
            self.output_path = self.input_path.parent() / f"{self.input_path.name()}_stitched"
        else:
            # Convert output_path to VirtualPath if it's not already
            if not isinstance(output_path, VirtualPath):
                self.output_path = VirtualPathFactory.from_path(output_path)
            else:
                self.output_path = output_path

        # Store basic configuration
        self.normalize = normalize
        self.z_method = z_method
        self.well_filter = well_filter
        self.storage_mode = storage_mode # Added
        
        # Convert storage_root to VirtualPath if provided and not already a VirtualPath
        if storage_root:
            if not isinstance(storage_root, VirtualPath):
                self.storage_root = VirtualPathFactory.from_path(storage_root)
            else:
                self.storage_root = storage_root
        else:
            self.storage_root = None

        # Create orchestrator, passing storage config, and explicitly initialize it
        self.orchestrator = PipelineOrchestrator(
            plate_path=self.input_path,
            storage_mode=self.storage_mode, # Pass config
            storage_root=self.storage_root   # Pass config
        ).initialize()  # Explicitly initialize the orchestrator

        # Set parameters (no auto-detection for now)
        # Default to False if flatten_z is None (no auto-detection)
        self.flatten_z = False if flatten_z is None else flatten_z
        self.channel_weights = channel_weights

        # Create factory with current configuration
        self._create_factory()

    def _create_factory(self):
        """Create AutoPipelineFactory with current configuration."""
        self.factory = AutoPipelineFactory(
            input_dir=self.orchestrator.workspace_path,
            output_dir=self.output_path,
            normalize=self.normalize,
            flatten_z=self.flatten_z,
            z_method=self.z_method,
            channel_weights=self.channel_weights,
            well_filter=self.well_filter
        )

    def set_options(self, **kwargs):
        """
        Update configuration options.

        Args:
            **kwargs: Configuration options to update

        Returns:
            self: For method chaining
        """
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown option: {key}")

        # Recreate factory with updated configuration
        self._create_factory()

        return self

    def stitch(self):
        """
        Run the complete stitching process with current settings.

        Returns:
            Path: Path to the output directory
        """
        # Create pipelines
        pipelines = self.factory.create_pipelines()

        # Run pipelines
        self.orchestrator.run(pipelines=pipelines)

        return self.output_path
