"""
Global configuration dataclasses for OpenHCS.

This module defines the primary configuration objects used throughout the application,
such as VFSConfig, PathPlanningConfig, and the overarching GlobalPipelineConfig.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import os # For a potentially more dynamic default for num_workers
import threading
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union, Dict, Any, List
from enum import Enum
from openhcs.constants import Microscope
from openhcs.constants.constants import Backend

# Import TilingLayout for TUI configuration
try:
    from textual_window import TilingLayout
except ImportError:
    # Fallback for when textual-window is not available
    from enum import Enum
    class TilingLayout(Enum):
        FLOATING = "floating"
        MASTER_DETAIL = "master_detail"

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


class WellFilterMode(Enum):
    """Well filtering modes for selective materialization."""
    INCLUDE = "include"  # Materialize only specified wells
    EXCLUDE = "exclude"  # Materialize all wells except specified ones

@dataclass(frozen=True)
class ZarrConfig:
    """Configuration for Zarr storage backend."""
    store_name: str = "images"
    """Name of the zarr store directory."""

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
    """Write plate-level metadata for HCS viewing (required for OME-ZARR viewers like napari)."""


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
    """
    Configuration for pipeline path planning and directory structure.

    This class handles path construction concerns including plate root directories,
    output directory suffixes, and subdirectory organization. It does not handle
    analysis results location, which is controlled at the pipeline level.
    """
    output_dir_suffix: str = "_outputs"
    """Default suffix for general step output directories."""

    global_output_folder: Optional[Path] = None
    """
    Optional global output folder where all plate workspaces and outputs will be created.
    If specified, plate workspaces will be created as {global_output_folder}/{plate_name}_workspace/
    and outputs as {global_output_folder}/{plate_name}_workspace_outputs/.
    If None, uses the current behavior (workspace and outputs in same directory as plate).
    Example: "/data/results" or "/mnt/hcs_output"
    """

    sub_dir: str = "images"
    """
    Subdirectory within plate folder for storing processed data.
    Automatically adds .zarr suffix when using zarr backend.
    Examples: "images", "processed", "data/images"
    """


@dataclass(frozen=True)
class StepMaterializationConfig(PathPlanningConfig):
    """
    Configuration for per-step materialization - configurable in UI.

    This dataclass appears in the UI like any other configuration, allowing users
    to set pipeline-level defaults for step materialization behavior. All step
    materialization instances will inherit these defaults unless explicitly overridden.

    Inherits from PathPlanningConfig to ensure all required path planning fields
    (like global_output_folder) are available for the lazy loading system.

    Well Filtering Options:
    - well_filter=1 materializes first well only (enables quick checkpointing)
    - well_filter=None materializes all wells
    - well_filter=["A01", "B03"] materializes only specified wells
    - well_filter="A01:A12" materializes well range
    - well_filter=5 materializes first 5 wells processed
    - well_filter_mode controls include/exclude behavior
    """

    # Well filtering defaults
    well_filter: Optional[Union[List[str], str, int]] = 1
    """
    Well filtering for selective step materialization:
    - 1: Materialize first well only (default - enables quick checkpointing)
    - None: Materialize all wells
    - List[str]: Specific well IDs ["A01", "B03", "D12"]
    - str: Pattern/range "A01:A12", "row:A", "col:01-06"
    - int: Maximum number of wells (first N processed)
    """

    well_filter_mode: WellFilterMode = WellFilterMode.INCLUDE
    """
    Well filtering mode for step materialization:
    - INCLUDE: Materialize only wells matching the filter
    - EXCLUDE: Materialize all wells except those matching the filter
    """

    # Override PathPlanningConfig defaults to prevent collisions
    output_dir_suffix: str = ""  # Uses same output plate path as main pipeline
    sub_dir: str = "checkpoints"  # vs global "images"


# Thread-local storage for current pipeline config
_current_pipeline_config = threading.local()

def set_current_pipeline_config(config: 'GlobalPipelineConfig'):
    """Set the current pipeline config for MaterializationPathConfig defaults."""
    _current_pipeline_config.value = config

def get_current_materialization_defaults() -> StepMaterializationConfig:
    """Get current step materialization config from pipeline config."""
    if hasattr(_current_pipeline_config, 'value') and _current_pipeline_config.value:
        return _current_pipeline_config.value.materialization_defaults
    # Fallback to default instance if no pipeline config is set
    return StepMaterializationConfig()


class LazyDefaultPlaceholderService:
    """
    Enhanced service supporting factory-created lazy classes with flexible resolution.

    Provides consistent "Pipeline default: {value}" placeholder pattern
    for both static and dynamic lazy configuration classes.
    """

    @staticmethod
    def has_lazy_resolution(dataclass_type: type) -> bool:
        """Check if dataclass has lazy resolution methods (created by factory)."""
        return (hasattr(dataclass_type, '_resolve_field_value') and
                hasattr(dataclass_type, 'to_base_config'))

    @staticmethod
    def get_lazy_resolved_placeholder(
        dataclass_type: type,
        field_name: str,
        app_config: Optional[Any] = None
    ) -> Optional[str]:
        """
        Get placeholder text for lazy-resolved field with flexible resolution.

        Args:
            dataclass_type: The lazy dataclass type (created by factory)
            field_name: Name of the field to resolve
            app_config: Optional app config for dynamic resolution

        Returns:
            "Pipeline default: {value}" format for consistent UI experience.
        """
        if not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type):
            return None

        # For dynamic resolution, create lazy class with current app config
        if app_config:
            from openhcs.core.lazy_config import LazyDataclassFactory
            dynamic_lazy_class = LazyDataclassFactory.create_lazy_dataclass(
                defaults_source=app_config,  # Use the app_config directly
                lazy_class_name=f"Dynamic{dataclass_type.__name__}"
            )
            temp_instance = dynamic_lazy_class()
        else:
            # Use existing lazy class (static resolution)
            temp_instance = dataclass_type()

        resolved_value = getattr(temp_instance, field_name)

        if resolved_value is not None:
            # Format nested dataclasses with key field values
            if hasattr(resolved_value, '__dataclass_fields__'):
                # For nested dataclasses, show key field values instead of generic info
                summary = LazyDefaultPlaceholderService._format_nested_dataclass_summary(resolved_value)
                return f"Pipeline default: {summary}"
            else:
                return f"Pipeline default: {resolved_value}"
        else:
            return "Pipeline default: (none)"

    @staticmethod
    def _format_nested_dataclass_summary(dataclass_instance) -> str:
        """
        Format nested dataclass with all field values for user-friendly placeholders.

        Uses generic dataclass introspection to show all fields with their current values,
        providing a complete and maintainable summary without hardcoded field mappings.
        """
        import dataclasses

        class_name = dataclass_instance.__class__.__name__

        # Get all fields from the dataclass using introspection
        all_fields = [f.name for f in dataclasses.fields(dataclass_instance)]

        # Extract all field values
        field_summaries = []
        for field_name in all_fields:
            try:
                value = getattr(dataclass_instance, field_name)

                # Skip None values to keep summary concise
                if value is None:
                    continue

                # Format different value types appropriately
                if hasattr(value, 'value'):  # Enum
                    formatted_value = value.value
                elif hasattr(value, 'name'):  # Enum with name
                    formatted_value = value.name
                elif isinstance(value, str) and len(value) > 20:  # Long strings
                    formatted_value = f"{value[:17]}..."
                elif dataclasses.is_dataclass(value):  # Nested dataclass
                    formatted_value = f"{value.__class__.__name__}(...)"
                else:
                    formatted_value = str(value)

                field_summaries.append(f"{field_name}={formatted_value}")

            except (AttributeError, Exception):
                # Skip fields that can't be accessed
                continue

        if field_summaries:
            return ", ".join(field_summaries)
        else:
            # Fallback when no non-None fields are found
            return f"{class_name} (default settings)"


# MaterializationPathConfig is now LazyStepMaterializationConfig from lazy_config.py
# Import moved to avoid circular dependency - use lazy import pattern


@dataclass(frozen=True)
class TilingKeybinding:
    """Declarative mapping between key combination and window manager method."""
    key: str
    action: str  # method name that already exists
    description: str


@dataclass(frozen=True)
class TilingKeybindings:
    """Declarative mapping of tiling keybindings to existing methods."""

    # Focus controls
    focus_next: TilingKeybinding = TilingKeybinding("ctrl+j", "focus_next_window", "Focus Next Window")
    focus_prev: TilingKeybinding = TilingKeybinding("ctrl+k", "focus_previous_window", "Focus Previous Window")

    # Layout controls - map to wrapper methods
    horizontal_split: TilingKeybinding = TilingKeybinding("ctrl+shift+h", "set_horizontal_split", "Horizontal Split")
    vertical_split: TilingKeybinding = TilingKeybinding("ctrl+shift+v", "set_vertical_split", "Vertical Split")
    grid_layout: TilingKeybinding = TilingKeybinding("ctrl+shift+g", "set_grid_layout", "Grid Layout")
    master_detail: TilingKeybinding = TilingKeybinding("ctrl+shift+m", "set_master_detail", "Master Detail")
    toggle_floating: TilingKeybinding = TilingKeybinding("ctrl+shift+f", "toggle_floating", "Toggle Floating")

    # Window movement - map to extracted window_manager methods
    move_window_prev: TilingKeybinding = TilingKeybinding("ctrl+shift+left", "move_focused_window_prev", "Move Window Left")
    move_window_next: TilingKeybinding = TilingKeybinding("ctrl+shift+right", "move_focused_window_next", "Move Window Right")
    rotate_left: TilingKeybinding = TilingKeybinding("ctrl+alt+left", "rotate_window_order_left", "Rotate Windows Left")
    rotate_right: TilingKeybinding = TilingKeybinding("ctrl+alt+right", "rotate_window_order_right", "Rotate Windows Right")

    # Gap controls
    gap_increase: TilingKeybinding = TilingKeybinding("ctrl+plus", "gap_increase", "Increase Gap")
    gap_decrease: TilingKeybinding = TilingKeybinding("ctrl+minus", "gap_decrease", "Decrease Gap")

    # Bulk operations
    minimize_all: TilingKeybinding = TilingKeybinding("ctrl+shift+d", "minimize_all_windows", "Minimize All")
    open_all: TilingKeybinding = TilingKeybinding("ctrl+shift+o", "open_all_windows", "Open All")


@dataclass(frozen=True)
class FunctionRegistryConfig:
    """Configuration for function registry behavior across all libraries."""
    enable_scalar_functions: bool = True
    """
    Whether to register functions that return scalars.
    When True: Scalar-returning functions are wrapped as (array, scalar) tuples.
    When False: Scalar-returning functions are filtered out entirely.
    Applies uniformly to all libraries (CuPy, scikit-image, pyclesperanto).
    """


@dataclass(frozen=True)
class TUIConfig:
    """Configuration for OpenHCS Textual User Interface."""
    default_tiling_layout: TilingLayout = TilingLayout.MASTER_DETAIL
    """Default tiling layout for window manager on startup."""

    default_window_gap: int = 1
    """Default gap between windows in tiling mode (in characters)."""

    enable_startup_notification: bool = True
    """Whether to show notification about tiling mode on startup."""

    keybindings: TilingKeybindings = field(default_factory=TilingKeybindings)
    """Declarative mapping of all tiling keybindings."""


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

    materialization_results_path: Path = Path("results")
    """
    Path for materialized analysis results (CSV, JSON files from special outputs).

    This is a pipeline-wide setting that controls where all special output materialization
    functions save their analysis results, regardless of which step produces them.

    Can be relative to plate folder or absolute path.
    Default: "results" creates a results/ folder in the plate directory.
    Examples: "results", "./analysis", "/data/analysis_results", "../shared_results"

    Note: This is separate from per-step image materialization, which is controlled
    by the sub_dir field in each step's materialization_config.
    """

    analysis_consolidation: AnalysisConsolidationConfig = field(default_factory=AnalysisConsolidationConfig)
    """Configuration for automatic analysis results consolidation."""

    plate_metadata: PlateMetadataConfig = field(default_factory=PlateMetadataConfig)
    """Configuration for plate metadata in consolidated outputs."""

    function_registry: FunctionRegistryConfig = field(default_factory=FunctionRegistryConfig)
    """Configuration for function registry behavior."""

    materialization_defaults: StepMaterializationConfig = field(default_factory=StepMaterializationConfig)
    """Default configuration for per-step materialization - configurable in UI."""

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
_DEFAULT_FUNCTION_REGISTRY_CONFIG = FunctionRegistryConfig()
_DEFAULT_MATERIALIZATION_DEFAULTS = StepMaterializationConfig()
_DEFAULT_TUI_CONFIG = TUIConfig()

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
        plate_metadata=_DEFAULT_PLATE_METADATA_CONFIG,
        function_registry=_DEFAULT_FUNCTION_REGISTRY_CONFIG,
        materialization_defaults=_DEFAULT_MATERIALIZATION_DEFAULTS
    )


# Import MaterializationPathConfig directly - circular import solved by moving import to end
from openhcs.core.lazy_config import LazyStepMaterializationConfig as MaterializationPathConfig
