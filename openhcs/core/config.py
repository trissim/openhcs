"""
Configuration classes for openhcs.

This module contains dataclasses for configuration of different components.
"""

import copy  # Import copy module at the top level
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openhcs.constants.constants import (DEFAULT_MARGIN_RATIO,
                                            DEFAULT_MAX_SHIFT,
                                            DEFAULT_NUM_WORKERS,
                                            DEFAULT_OUT_DIR_SUFFIX,
                                            DEFAULT_PIXEL_SIZE,
                                            DEFAULT_POSITIONS_DIR_SUFFIX,
                                            DEFAULT_STITCHED_DIR_SUFFIX,
                                            DEFAULT_TILE_OVERLAP)

logger = logging.getLogger(__name__)


@dataclass
class MISTConfig:
    """Configuration for the MIST backend."""
    use_gpu: bool = False
    """Whether to use GPU acceleration if available."""

    requires_gpu: bool = False
    """Whether GPU is required (will raise NotImplementedError if unavailable)."""

    mist_params: Dict[str, Any] = field(default_factory=dict)
    """Additional MIST-specific parameters."""


@dataclass
class BackendConfig:
    """Configuration for compute backends."""
    type: str = "ashlar"
    """Type of backend to use ('ashlar', 'mist', 'composite', etc.)."""

    mapping: Optional[Dict[str, str]] = None
    """
    Mapping of method names to backend types for composite backends.
    Example: {"generate_positions": "ashlar", "stitch": "mist"}
    """

    params: Optional[Dict[str, Any]] = None
    """Additional parameters for the backend."""

    # MIST-specific configuration
    mist: MISTConfig = field(default_factory=MISTConfig)
    """MIST-specific configuration."""


@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    tile_overlap: float = DEFAULT_TILE_OVERLAP
    max_shift: int = DEFAULT_MAX_SHIFT
    margin_ratio: float = DEFAULT_MARGIN_RATIO
    pixel_size: float = DEFAULT_PIXEL_SIZE
    variable_components: List[str] = field(default_factory=lambda: DEFAULT_VARIABLE_COMPONENTS)
    recursive_pattern_search: bool = DEFAULT_RECURSIVE_PATTERN_SEARCH


# FocusAnalyzerConfig has been removed in favor of direct parameters to FocusAnalyzer


# Constant for num_workers default is now imported from constants.pipeline

@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Directory configuration
    out_dir_suffix: str = DEFAULT_OUT_DIR_SUFFIX  # Default suffix for processing steps
    positions_dir_suffix: str = DEFAULT_POSITIONS_DIR_SUFFIX  # Suffix for position generation step
    stitched_dir_suffix: str = DEFAULT_STITCHED_DIR_SUFFIX  # Suffix for stitching step

    # Processing configuration
    num_workers: int = DEFAULT_NUM_WORKERS  # Use constant default value

    # Stitching configuration
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)

    # Microscope configuration
    force_parser: Optional[str] = None  # Force a specific parser type (e.g., "OperaPhenix")

    # Backend configurations
    position_generator_backend: Optional[BackendConfig] = field(default_factory=lambda: BackendConfig(type="ashlar"))
    """Configuration for the position generator backend (e.g., Ashlar, MIST)."""

    image_assembler_backend: Optional[BackendConfig] = field(default_factory=lambda: BackendConfig(type="cpu"))
    """Configuration for the image assembler backend (e.g., CPU, GPU)."""

    # Old 'backend' field is removed.
    # Consumers must now specify 'position_generator_backend' and/or 'image_assembler_backend'.

    # Intermediate storage configuration
    storage_mode: Literal["legacy", "memory", "zarr"] = "legacy"
    """Mode for storing intermediate pipeline results ('legacy', 'memory', 'zarr').
    'legacy': Default, uses existing in-memory dict within Pipeline.
    'memory': Uses MemoryStorageBackend (persists .npy on completion).
    'zarr': Uses ZarrStorageBackend (persists to disk immediately).
    """
    storage_root: Optional[Path] = None
    """Root directory for storage backends that require it (e.g., Zarr).
    Must be provided when using storage modes that require a physical location (e.g., 'zarr').
    If None when required, an explicit error will be raised during validation.
    For 'memory' mode, this is not required.
    """

    def copy(self):
        """
        Create a deep copy of this configuration object.

        Uses the copy module imported at the top level for deterministic behavior.
        """
        return copy.deepcopy(self)
