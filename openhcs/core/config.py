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
from enum import Enum
from openhcs.constants import Microscope
from openhcs.constants.constants import Backend

logger = logging.getLogger(__name__)


class ZarrCompressor(Enum):
    """Available compression algorithms for zarr storage."""
    BLOSC = "blosc"
    ZLIB = "zlib"
    LZ4 = "lz4"
    ZSTD = "zstd"
    NONE = "none"

    def create_compressor(self, compression_level: int, shuffle: bool = True) -> Optional[Any]:
        """Create the actual zarr compressor instance.

        Args:
            compression_level: Compression level (1-22 for ZSTD, 1-9 for others)
            shuffle: Enable byte shuffling for better compression (blosc only)

        Returns:
            Configured zarr compressor instance or None for no compression
        """
        import zarr

        match self:
            case ZarrCompressor.NONE:
                return None
            case ZarrCompressor.BLOSC:
                return zarr.Blosc(cname='lz4', clevel=compression_level, shuffle=shuffle)
            case ZarrCompressor.ZLIB:
                return zarr.Zlib(level=compression_level)
            case ZarrCompressor.LZ4:
                return zarr.LZ4(acceleration=compression_level)
            case ZarrCompressor.ZSTD:
                return zarr.Zstd(level=compression_level)


class ZarrChunkStrategy(Enum):
    """Chunking strategies for zarr arrays."""
    SINGLE = "single"  # Single chunk per array (optimal for batch I/O)
    AUTO = "auto"      # Let zarr decide chunk size
    CUSTOM = "custom"  # User-defined chunk sizes


class MaterializationBackend(Enum):
    """Available backends for materialization (persistent storage only)."""
    ZARR = "zarr"
    DISK = "disk"

@dataclass(frozen=True)
class ZarrConfig:
    """Configuration for Zarr storage backend."""
    store_name: str = "images.zarr"
    """Name of the zarr store file."""

    compressor: ZarrCompressor = ZarrCompressor.LZ4
    """Compression algorithm to use."""

    compression_level: int = 9
    """Compression level (1-9 for LZ4, higher = more compression)."""

    shuffle: bool = True
    """Enable byte shuffling for better compression (blosc only)."""

    chunk_strategy: ZarrChunkStrategy = ZarrChunkStrategy.SINGLE
    """Chunking strategy for zarr arrays."""

    ome_zarr_metadata: bool = True
    """Generate OME-ZARR compatible metadata and structure."""

    write_plate_metadata: bool = True
    """Write plate-level metadata for HCS viewing (required for napari ome-zarr)."""


@dataclass(frozen=True)
class VFSConfig:
    """Configuration for Virtual File System (VFS) related operations."""
    intermediate_backend: Backend = Backend.MEMORY
    """Backend for storing intermediate step results that are not explicitly materialized."""

    materialization_backend: MaterializationBackend = MaterializationBackend.DISK
    """Backend for explicitly materialized outputs (e.g., final results, user-requested saves)."""

@dataclass(frozen=True)
class AnalysisConsolidationConfig:
    """Configuration for automatic analysis results consolidation."""
    enabled: bool = True
    """Whether to automatically run analysis consolidation after pipeline completion."""

    metaxpress_style: bool = True
    """Whether to generate MetaXpress-compatible output format with headers."""

    well_pattern: str = r"([A-Z]\d{2})"
    """Regex pattern for extracting well IDs from filenames."""

    file_extensions: tuple[str, ...] = (".csv",)
    """File extensions to include in consolidation."""

    exclude_patterns: tuple[str, ...] = (r".*consolidated.*", r".*metaxpress.*", r".*summary.*")
    """Filename patterns to exclude from consolidation."""

    output_filename: str = "metaxpress_style_summary.csv"
    """Name of the consolidated output file."""


@dataclass(frozen=True)
class PlateMetadataConfig:
    """Configuration for plate metadata in MetaXpress-style output."""
    barcode: Optional[str] = None
    """Plate barcode. If None, will be auto-generated from plate name."""

    plate_name: Optional[str] = None
    """Plate name. If None, will be derived from plate path."""

    plate_id: Optional[str] = None
    """Plate ID. If None, will be auto-generated."""

    description: Optional[str] = None
    """Experiment description. If None, will be auto-generated."""

    acquisition_user: str = "OpenHCS"
    """User who acquired the data."""

    z_step: str = "1"
    """Z-step information for MetaXpress compatibility."""


@dataclass(frozen=True)
class PathPlanningConfig:
    """Configuration for pipeline path planning, defining directory suffixes."""
    output_dir_suffix: str = "_outputs"
    """Default suffix for general step output directories."""

    global_output_folder: Optional[str] = None
    """
    Optional global output folder where all plate workspaces and outputs will be created.
    If specified, plate workspaces will be created as {global_output_folder}/{plate_name}_workspace/
    and outputs as {global_output_folder}/{plate_name}_workspace_outputs/.
    If None, uses the current behavior (workspace and outputs in same directory as plate).
    Example: "/data/results" or "/mnt/hcs_output"
    """

    materialization_results_path: str = "results"
    """
    Path for materialized analysis results (CSV, JSON files from special outputs).
    Can be relative to plate folder or absolute path.
    Default: "results" creates a results/ folder in the plate directory.
    Examples: "results", "./analysis", "/data/analysis_results", "../shared_results"
    """


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

    zarr: ZarrConfig = field(default_factory=ZarrConfig)
    """Configuration for Zarr storage backend."""

    analysis_consolidation: AnalysisConsolidationConfig = field(default_factory=AnalysisConsolidationConfig)
    """Configuration for automatic analysis results consolidation."""

    plate_metadata: PlateMetadataConfig = field(default_factory=PlateMetadataConfig)
    """Configuration for plate metadata in consolidated outputs."""



    microscope: Microscope = Microscope.AUTO
    """Default microscope type for auto-detection."""
    
    use_threading: bool = field(default_factory=lambda: os.getenv('OPENHCS_USE_THREADING', 'false').lower() == 'true')
    """Use ThreadPoolExecutor instead of ProcessPoolExecutor for debugging. Reads from OPENHCS_USE_THREADING environment variable."""

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
_DEFAULT_ZARR_CONFIG = ZarrConfig()
_DEFAULT_ANALYSIS_CONSOLIDATION_CONFIG = AnalysisConsolidationConfig()
_DEFAULT_PLATE_METADATA_CONFIG = PlateMetadataConfig()

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
        vfs=_DEFAULT_VFS_CONFIG,
        zarr=_DEFAULT_ZARR_CONFIG,
        analysis_consolidation=_DEFAULT_ANALYSIS_CONSOLIDATION_CONFIG,
        plate_metadata=_DEFAULT_PLATE_METADATA_CONFIG
    )
