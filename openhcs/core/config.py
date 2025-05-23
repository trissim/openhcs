"""
Global configuration dataclasses for OpenHCS.

This module defines the primary configuration objects used throughout the application,
such as VFSConfig, PathPlanningConfig, and the overarching GlobalPipelineConfig.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import os # For a potentially more dynamic default for num_workers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union, Dict, Any
from openhcs.constants import Microscope

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VFSConfig:
    """Configuration for Virtual File System (VFS) related operations."""
    default_intermediate_backend: Literal["memory", "disk", "zarr"] = "memory"
    """Default backend for storing intermediate step results that are not explicitly materialized."""
    
    default_materialization_backend: Literal["disk", "zarr"] = "disk"
    """Default backend for explicitly materialized outputs (e.g., final results, user-requested saves)."""
    
    persistent_storage_root_path: Optional[str] = None
    """
    Optional root path for persistent storage backends like 'disk' or 'zarr'.
    If None, paths might be relative to a workspace or require full specification.
    Example: "/mnt/hcs_data_root" or "./.openhcs_data"
    """

@dataclass(frozen=True)
class PathPlanningConfig:
    """Configuration for pipeline path planning, defining directory suffixes."""
    output_dir_suffix: str = "_outputs"
    """Default suffix for general step output directories."""
    
    positions_dir_suffix: str = "_positions"
    """Suffix for directories containing position generation results."""
    
    stitched_dir_suffix: str = "_stitched"
    """Suffix for directories containing stitched image results."""


@dataclass(frozen=True)
class GlobalPipelineConfig:
    """
    Root configuration object for an OpenHCS pipeline session.
    This object is intended to be instantiated at application startup and treated as immutable.
    """
    num_workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    """Number of worker processes/threads for parallelizable tasks."""
    
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
    """Configuration for path planning (directory suffixes)."""
    
    vfs: VFSConfig = field(default_factory=VFSConfig)
    """Configuration for Virtual File System behavior."""

    microscope: Microscope = Microscope.AUTO
    """Default microscope type for auto-detection."""

    # Future extension point:
    # logging_config: Optional[Dict[str, Any]] = None # For configuring logging levels, handlers
    # plugin_settings: Dict[str, Any] = field(default_factory=dict) # For plugin-specific settings

# --- Default Configuration Provider ---

# Pre-instantiate default sub-configs for clarity if they have many fields or complex defaults.
# For simple cases, direct instantiation in get_default_global_config is also fine.
_DEFAULT_PATH_PLANNING_CONFIG = PathPlanningConfig()
_DEFAULT_VFS_CONFIG = VFSConfig(
    # Example: Set a default persistent_storage_root_path if desired for out-of-the-box behavior
    # persistent_storage_root_path="./openhcs_output_data" 
)

def get_default_global_config() -> GlobalPipelineConfig:
    """
    Provides a default instance of GlobalPipelineConfig.
    
    This function is called if no specific configuration is provided to the
    PipelineOrchestrator, ensuring the application can run with sensible defaults.
    """
    logger.info("Initializing with default GlobalPipelineConfig.")
    return GlobalPipelineConfig(
        # num_workers is already handled by field(default_factory) in GlobalPipelineConfig
        path_planning=_DEFAULT_PATH_PLANNING_CONFIG,
        vfs=_DEFAULT_VFS_CONFIG
    )
