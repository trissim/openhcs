"""
Global configuration dataclasses for OpenHCS.

This module defines the primary configuration objects used throughout the application,
such as VFSConfig, PathPlanningConfig, and the overarching GlobalPipelineConfig.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Annotated, ClassVar
from enum import Enum
from abc import ABC, abstractmethod
from arraybridge.decorators import DtypeConversionConfig
from polystore import config as _polystore_config
from openhcs.constants import (
    AllComponents,
    Microscope,
    SequentialComponents,
    VariableComponents,
    GroupBy,
    DtypeConversion,
)
from openhcs.constants.constants import (
    Backend,
    LiteralDtype,
    get_default_variable_components,
    get_default_group_by,
)
from metaclass_registry import AutoRegisterMeta
from openhcs.constants.input_source import InputSource
from python_introspect import Enableable
from python_introspect.enableable import EnableableMeta
from openhcs.core.runtime_plane_projection import RuntimeSliceInvariantValue
from openhcs.core.vfs_protocol import PlateOutputDirectory, PlateOutputFile
from openhcs.utils.environment import OpenHCSProcessEnvironment

# Import decorator for automatic decorator creation


# Combined metaclass for StreamingConfig to support both Enableable and AutoRegisterMeta
class StreamingConfigMeta(EnableableMeta, AutoRegisterMeta):
    """Combined metaclass supporting Enableable semantics and AutoRegisterMeta registration."""

    pass


from objectstate import auto_create_decorator, abbreviation

# Import platform-aware transport mode default
# This must be imported here to avoid circular imports
import platform

logger = logging.getLogger(__name__)


ZarrCompressor = _polystore_config.ZarrCompressor
ZarrCompressorFactory = _polystore_config.ZarrCompressorFactory
NoZarrCompressorFactory = _polystore_config.NoZarrCompressorFactory
BloscZarrCompressorFactory = _polystore_config.BloscZarrCompressorFactory
ZlibZarrCompressorFactory = _polystore_config.ZlibZarrCompressorFactory
Lz4ZarrCompressorFactory = _polystore_config.Lz4ZarrCompressorFactory
ZstdZarrCompressorFactory = _polystore_config.ZstdZarrCompressorFactory
ZarrChunkStrategy = _polystore_config.ZarrChunkStrategy


class MaterializationBackend(Enum):
    """Available backends for materialization (persistent storage only)."""

    AUTO = "auto"
    ZARR = "zarr"
    DISK = "disk"
    OMERO_LOCAL = "omero_local"


class WellFilterMode(Enum):
    """Well filtering modes for selective materialization."""

    INCLUDE = "include"  # Materialize only specified wells
    EXCLUDE = "exclude"  # Materialize all wells except specified ones


class NormalizationMethod(Enum):
    """Normalization methods for experimental analysis."""

    FOLD_CHANGE = "fold_change"  # value / control_mean
    Z_SCORE = "z_score"  # (value - control_mean) / control_std
    PERCENT_CONTROL = "percent_control"  # (value / control_mean) * 100


class MicroscopeFormat(Enum):
    """Supported microscope formats for experimental analysis."""

    EDDU_CX5 = "EDDU_CX5"  # ThermoFisher CX5 format
    EDDU_METAXPRESS = "EDDU_metaxpress"  # Molecular Devices MetaXpress format


class TransportMode(Enum):
    """ZMQ transport modes for local vs remote communication."""

    IPC = "ipc"  # Inter-process communication (local only, no firewall prompts)
    TCP = "tcp"  # Network sockets (supports remote, triggers firewall)


class MultiprocessingStartMethod(Enum):
    """Process start methods for OpenHCS worker pools."""

    SPAWN = "spawn"
    FORK = "fork"
    FORKSERVER = "forkserver"


@abbreviation("gpc")
@auto_create_decorator
@dataclass(frozen=True)
class GlobalPipelineConfig:
    """
    Root configuration object for an OpenHCS pipeline session.
    This object is intended to be instantiated at application startup and treated as immutable.
    """

    materialization_results_path: Annotated[Path, abbreviation("results_path")] = field(
        default_factory=lambda: Path("results"),
        metadata={"ui_hidden": True},
    )
    """
    Path for materialized analysis results (CSV, JSON files from artifacts).

    This is a pipeline-wide setting that controls where artifact materialization
    functions save their analysis results, regardless of which step produces them.

    Can be relative to plate folder or absolute path.
    Default: "results" creates a results/ folder in the plate directory.
    Examples: "results", "./analysis", "/data/analysis_results", "../shared_results"

    Note: This is separate from per-step image materialization, which is controlled
    by the sub_dir field in each step's step_materialization_config.
    """

    materialize_runtime_artifacts: Annotated[
        bool, abbreviation("mat_artifacts")
    ] = True
    """Persist runtime artifacts such as measurements/tables to configured result files."""

    num_workers: Annotated[int, abbreviation("W")] = 1
    """Number of worker processes/threads for parallelizable tasks."""

    microscope: Annotated[Microscope, abbreviation("scope")] = field(
        default_factory=lambda: Microscope.AUTO,
        metadata={"ui_hidden": True},
    )
    """Default microscope type for auto-detection."""

    use_threading: Annotated[bool, abbreviation("threading")] = field(
        default_factory=OpenHCSProcessEnvironment.use_threading_mode,
        metadata={"ui_hidden": True},
    )
    """Use ThreadPoolExecutor instead of ProcessPoolExecutor for debugging. Reads from OPENHCS_USE_THREADING environment variable."""

    multiprocessing_start_method: Annotated[
        MultiprocessingStartMethod,
        abbreviation("mp_start"),
    ] = field(
        default_factory=lambda: MultiprocessingStartMethod.SPAWN,
        metadata={"ui_hidden": True},
    )
    """Process start method for multiprocessing workers. SPAWN is CUDA-safe; FORK is CPU-only."""

    auto_add_output_plate_to_plate_manager: Annotated[
        bool, abbreviation("auto_add_output_plate")
    ] = False
    """If True, when a plate run completes successfully, the computed output plate root
    (from path planning) is automatically added to Plate Manager as a new orchestrator
    if it is not already present."""

    # Future extension point:
    # logging_config: Optional[Dict[str, Any]] = None # For configuring logging levels, handlers
    # plugin_settings: Dict[str, Any] = field(default_factory=dict) # For plugin-specific settings


@abbreviation("compile_dbg")
@global_pipeline_config(
    inherit_as_none=False,
    always_viewable_fields=["enabled"],
)
@dataclass(frozen=True)
class CompilationDebugConfig(Enableable):
    """Compiler diagnostics inherited by PipelineConfig."""

    compiled_execution_bundle_path: Annotated[
        Optional[PlateOutputFile], abbreviation("bundle")
    ] = None
    """Optional pickle path for the compiled execution bundle."""


# PipelineConfig will be created automatically by the injection system
# (GlobalPipelineConfig → PipelineConfig by removing "Global" prefix)

# Import utilities for dynamic config creation
from openhcs.utils.enum_factory import create_colormap_enum
from openhcs.utils.display_config_factory import (
    create_napari_display_config,
    create_fiji_display_config,
)

# Create colormap enum with minimal set to avoid importing napari (→ dask → GPU libs)
# The lazy=True parameter uses a hardcoded minimal set instead of introspecting napari
NapariColormap = create_colormap_enum(lazy=True)


class NapariDimensionMode(Enum):
    """How to handle different dimensions in napari visualization."""

    SLICE = "slice"  # Show as 2D slice (take middle slice)
    STACK = "stack"  # Show as 3D stack/volume


class NapariVariableSizeHandling(Enum):
    """How to handle images with different sizes in the same layer."""

    SEPARATE_LAYERS = (
        "separate_layers"  # Create separate layers per well (preserves exact data)
    )
    PAD_TO_MAX = "pad_to_max"  # Pad smaller images to match largest (enables stacking)


# Visualization dtype normalization (alias to LiteralDtype - no duplication)
VisualizationDtype = LiteralDtype
VisualizationDtype.__doc__ = (
    """Dtype normalization for visualization streaming (Napari/Fiji stacking)."""
)


# Create NapariDisplayConfig using factory
# Note: Uses lazy colormap enum to avoid importing napari at module level
NapariDisplayConfig = create_napari_display_config(
    colormap_enum=NapariColormap,
    dimension_mode_enum=NapariDimensionMode,
    variable_size_handling_enum=NapariVariableSizeHandling,
    visualization_dtype_enum=VisualizationDtype,
    component_order=AllComponents.ordered_names(),
)

# Apply the global pipeline config decorator with ui_hidden=True
# This config is only inherited by NapariStreamingConfig, so hide it from UI
NapariDisplayConfig = global_pipeline_config(ui_hidden=True)(NapariDisplayConfig)


# ============================================================================
# Fiji Display Configuration
# ============================================================================


class FijiLUT(Enum):
    """Fiji/ImageJ LUT options."""

    GRAYS = "Grays"
    FIRE = "Fire"
    ICE = "Ice"
    SPECTRUM = "Spectrum"
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"


class FijiDimensionMode(Enum):
    """
    How to map OpenHCS dimensions to ImageJ hyperstack dimensions.

    ImageJ hyperstacks have 3 dimensions: Channels (C), Slices (Z), Frames (T).
    Each OpenHCS component (site, channel, z_index, timepoint) can be mapped to one of these.

    - WINDOW: Create separate windows for each value (like Napari SLICE mode)
    - CHANNEL: Map to ImageJ Channel dimension (C)
    - SLICE: Map to ImageJ Slice dimension (Z)
    - FRAME: Map to ImageJ Frame dimension (T)
    """

    WINDOW = "window"  # Separate windows (like Napari SLICE mode)
    CHANNEL = "channel"  # ImageJ Channel dimension (C)
    SLICE = "slice"  # ImageJ Slice dimension (Z)
    FRAME = "frame"  # ImageJ Frame dimension (T)


# Create FijiDisplayConfig using factory (with component-specific fields like Napari)
FijiDisplayConfig = create_fiji_display_config(
    lut_enum=FijiLUT,
    dimension_mode_enum=FijiDimensionMode,
    component_order=AllComponents.ordered_names(),
)

# Apply the global pipeline config decorator with ui_hidden=True
# This config is only inherited by FijiStreamingConfig, so hide it from UI
FijiDisplayConfig = global_pipeline_config(ui_hidden=True)(FijiDisplayConfig)


@abbreviation("wfc")
@global_pipeline_config
@dataclass(frozen=True)
class WellFilterConfig:
    """Base execution-domain filter inherited by specialized well policies.

    At pipeline scope this constrains which wells compile and execute. Nominal
    subclasses reuse the same selection for their own narrower behavior, such
    as main-flow persistence, step checkpoints, or viewer emission.
    """

    well_filter: Annotated[Optional[Union[List[str], str, int]], abbreviation("")] = (
        None
    )
    """Well filter specification: list of wells, pattern string, or non-negative max count. None means all wells; zero selects none."""

    well_filter_mode: Annotated[WellFilterMode, abbreviation("filter_mode")] = (
        WellFilterMode.INCLUDE
    )
    """Whether well_filter is an include list or exclude list."""

    @classmethod
    def well_filter_inheritance_branch(cls) -> type["WellFilterConfig"]:
        """Return the nominal MRO branch that may supply inherited filters."""
        return cls

    @classmethod
    def accepts_well_filter_provenance(cls, source_type: type | None) -> bool:
        """Return whether an inherited filter belongs to this policy branch."""
        if source_type is None:
            return True

        from objectstate import get_base_type_for_lazy

        config_type = get_base_type_for_lazy(cls) or cls
        source_base = get_base_type_for_lazy(source_type) or source_type
        branch_type = cls.well_filter_inheritance_branch()
        branch_base = get_base_type_for_lazy(branch_type) or branch_type
        return issubclass(source_base, config_type) or source_base in branch_base.mro()


@abbreviation("zarr")
@global_pipeline_config(
    inherit_as_none=False,
    field_abbreviations={
        "compressor": "compressor",
        "compression_level": "level",
        "chunk_strategy": "chunks",
    },
)
@dataclass(frozen=True)
class ZarrConfig(_polystore_config.ZarrConfig):
    """OpenHCS registration of PolyStore's Zarr configuration owner.

    OME-ZARR metadata and plate metadata are always enabled for HCS compliance.
    Shuffle filter is always enabled for Blosc compressor (ignored for others).
    """


@abbreviation("vfs")
@global_pipeline_config(always_viewable_fields=["materialization_backend"])
@dataclass(frozen=True)
class VFSConfig:
    """Configuration for Virtual File System (VFS) related operations."""

    read_backend: Annotated[Backend, abbreviation("read")] = Backend.AUTO
    """Backend for reading input data. AUTO uses metadata-based detection for OpenHCS plates."""

    intermediate_backend: Annotated[Backend, abbreviation("intermediate")] = (
        Backend.MEMORY
    )
    """Backend for storing intermediate step results that are not explicitly materialized."""

    materialization_backend: Annotated[
        MaterializationBackend, abbreviation("materialize")
    ] = MaterializationBackend.DISK
    """Backend for explicitly materialized outputs (e.g., final results, user-requested saves)."""


@abbreviation("dtype")
@global_pipeline_config
@dataclass(frozen=True)
class DtypeConfig(
    RuntimeSliceInvariantValue,
    DtypeConversionConfig,
):
    """Configuration for dtype conversion behavior in memory type decorators."""

    default_dtype_conversion: Annotated[DtypeConversion, abbreviation("conv")] = (
        DtypeConversion.NATIVE_OUTPUT
    )
    """Default dtype conversion mode for all decorated functions.
    NATIVE_OUTPUT (no scaling) or PRESERVE_INPUT (scale to input dtype)."""

    @classmethod
    def default_value(cls) -> "DtypeConfig":
        """Return the inheritable OpenHCS callable default."""
        return cls()

    @classmethod
    def annotation_type(cls) -> type["DtypeConfig"]:
        """Expose the inheritable config type on callable signatures."""
        return cls


@abbreviation("proc")
@global_pipeline_config(
    always_viewable_fields=["variable_components", "group_by", "input_source"],
)
@dataclass(frozen=True)
class ProcessingConfig:
    """Independent stack-axis, post-assembly grouping, and main-flow choices."""

    variable_components: Annotated[List[VariableComponents], abbreviation("vars")] = (
        field(default_factory=get_default_variable_components)
    )
    """Components whose values vary along the assembled array axis.

    This field is the stack-axis meaning authority. It is independent of
    ``group_by`` and of whether the callable executes per plane or per stack.
    """

    group_by: Annotated[Optional[GroupBy], abbreviation("group")] = field(
        default_factory=get_default_group_by
    )
    """Component used to partition already assembled values.

    A dictionary function pattern uses the resulting group identity for branch
    selection. This field does not define the assembled stack axis.
    """

    input_source: Annotated[InputSource, abbreviation("source")] = (
        InputSource.PREVIOUS_STEP
    )
    """Main-flow source: previous step or pipeline start.

    Additional named sources are callable artifact inputs satisfied through
    step source bindings or prior artifact producers; they are not another
    ``InputSource`` value.
    """


from openhcs.core import source_bindings as source_binding_configs

SourceBindingsConfig = global_pipeline_config(
    inherit_as_none=False,
    preview_label="SRC",
    abbreviation="src",
)(source_binding_configs.SourceBindingsConfig)

StepSourceBindingsConfig = global_pipeline_config(
    preview_label="STEP_SRC",
    abbreviation="step_src",
)(source_binding_configs.StepSourceBindingsConfig)

source_binding_configs.SourceBindingsConfig = SourceBindingsConfig
source_binding_configs.StepSourceBindingsConfig = StepSourceBindingsConfig


@abbreviation("seq")
@global_pipeline_config
@dataclass(frozen=True)
class SequentialProcessingConfig:
    """Pipeline-level configuration for sequential processing mode.

    Sequential processing changes the orchestrator's execution flow to process
    one combination at a time through all steps, reducing memory usage.
    This is a pipeline-level setting, not per-step.
    """

    sequential_components: Annotated[
        List[SequentialComponents], abbreviation("seq_comp")
    ] = field(default_factory=list)
    """Components to process sequentially (e.g., [SequentialComponents.TIMEPOINT, SequentialComponents.CHANNEL]).

    When set, the orchestrator will process one combination of these components through
    all pipeline steps before moving to the next combination, clearing memory between combinations.
    """


@abbreviation("analysis")
@global_pipeline_config
@dataclass(frozen=True)
class AnalysisConsolidationConfig(Enableable):
    """Configuration for automatic analysis results consolidation.

    enabled controls whether consolidation runs after pipeline completion.
    """

    enabled: Annotated[bool, abbreviation("")] = True   

    metaxpress_style: Annotated[bool, abbreviation("mx_style")] = True
    """Whether to generate MetaXpress-compatible output format with headers."""

    well_pattern: Annotated[str, abbreviation("well_pat")] = r"([A-Z]\d{2})"
    """Regex pattern for extracting well IDs from filenames."""

    file_extensions: Annotated[tuple[str, ...], abbreviation("exts")] = (".csv",)
    """File extensions to include in consolidation."""

    exclude_patterns: Annotated[tuple[str, ...], abbreviation("exclude")] = (
        r".*consolidated.*",
        r".*metaxpress.*",
        r".*summary.*",
    )
    """Filename patterns to exclude from consolidation."""

    output_filename: Annotated[str, abbreviation("out_file")] = (
        "metaxpress_style_summary.csv"
    )
    """Name of the consolidated output file."""

    global_summary_filename: Annotated[str, abbreviation("global_sum")] = (
        "global_metaxpress_summary.csv"
    )
    """Name of the global consolidated summary file combining all plates."""


@abbreviation("plate")
@global_pipeline_config
@dataclass(frozen=True)
class PlateMetadataConfig:
    """Configuration for plate metadata in MetaXpress-style output."""

    barcode: Annotated[Optional[str], abbreviation("barcode")] = None
    """Plate barcode. If None, will be auto-generated from plate name."""

    plate_name: Annotated[Optional[str], abbreviation("name")] = None
    """Plate name. If None, will be derived from plate path."""

    plate_id: Annotated[Optional[str], abbreviation("id")] = None
    """Plate ID. If None, will be auto-generated."""

    description: Annotated[Optional[str], abbreviation("description")] = None
    """Experiment description. If None, will be auto-generated."""

    acquisition_user: Annotated[str, abbreviation("user")] = "OpenHCS"
    """User who acquired the data."""

    z_step: Annotated[str, abbreviation("z_step")] = "1"
    """Z-step information for MetaXpress compatibility."""


@abbreviation("exp")
@global_pipeline_config
@dataclass(frozen=True)
class ExperimentalAnalysisConfig:
    """Configuration for experimental analysis system."""

    config_file_name: Annotated[str, abbreviation("config")] = "config.xlsx"
    """Name of the experimental configuration Excel file."""

    design_sheet_name: Annotated[str, abbreviation("design")] = "drug_curve_map"
    """Name of the sheet containing experimental design."""

    plate_groups_sheet_name: Annotated[str, abbreviation("groups")] = "plate_groups"
    """Name of the sheet containing plate group mappings."""

    normalization_method: Annotated[NormalizationMethod, abbreviation("norm")] = (
        NormalizationMethod.FOLD_CHANGE
    )
    """Normalization method for control-based normalization."""

    export_raw_results: Annotated[bool, abbreviation("raw")] = True
    """Whether to export raw (non-normalized) results."""

    export_heatmaps: Annotated[bool, abbreviation("heatmaps")] = True
    """Whether to generate heatmap visualizations."""

    auto_detect_format: Annotated[bool, abbreviation("auto_format")] = True
    """Whether to automatically detect microscope format."""

    default_format: Annotated[Optional[MicroscopeFormat], abbreviation("format")] = None
    """Default format to use if auto-detection fails."""


@abbreviation("pp")
@global_pipeline_config(
    always_viewable_fields=[
        "well_filter",
        "output_dir_suffix",
        "global_output_folder",
    ],
)
@dataclass(frozen=True)
class PathPlanningConfig(WellFilterConfig):
    """
    Configuration for pipeline path planning and directory structure.

    This class handles path construction concerns including plate root directories,
    output directory suffixes, and subdirectory organization. It does not handle
    analysis results location, which is controlled at the pipeline level.

    Its inherited well filter selects which processed wells seed the automatic
    main-flow output plate. Zero keeps that automatic output runtime-only. This
    policy is independent of step checkpoints, typed artifact materialization,
    and viewer streaming.
    """

    output_dir_suffix: Annotated[str, abbreviation("suffix")] = "_openhcs"
    """Default suffix for general step output directories."""

    global_output_folder: Annotated[
        Optional[PlateOutputDirectory], abbreviation("global_folder")
    ] = None
    """
    Optional global output folder where all plate workspaces and outputs will be created.
    If specified, plate workspaces will be created as {global_output_folder}/{plate_name}_workspace/
    and outputs as {global_output_folder}/{plate_name}_workspace_outputs/.
    If None, uses the current behavior (workspace and outputs in same directory as plate).
    Example: "/data/results" or "/mnt/hcs_output"
    """

    sub_dir: Annotated[str, abbreviation("subdir")] = "images"
    """
    Subdirectory within plate folder for storing processed data.
    Examples: "images", "processed", "data/images"
    """


@abbreviation("step_wf")
@global_pipeline_config(always_viewable_fields=["well_filter"])
@dataclass(frozen=True)
class StepWellFilterConfig(WellFilterConfig):
    """Step-policy filter inheriting the broader pipeline execution domain."""

    # Override defaults for step-level configurations
    # well_filter: Optional[Union[List[str], str, int]] = 1
    pass


@abbreviation("mat")
@global_pipeline_config(preview_label="MAT", always_viewable_fields=["sub_dir"])
@dataclass(frozen=True)
class StepMaterializationConfig(Enableable, StepWellFilterConfig, PathPlanningConfig):
    """
    Configuration for persistent copies of a step's ordinary main-flow result.

    This dataclass appears in the UI like any other configuration, allowing users
    to set pipeline-level defaults for step materialization behavior. All step
    materialization instances will inherit these defaults unless explicitly overridden.

    Typed artifact outputs remain available through runtime dataflow independently.
    Their persistence is owned by artifact output and compiled runtime-artifact
    materialization plans rather than this main-flow checkpoint config.

    Uses multiple inheritance from PathPlanningConfig and StepWellFilterConfig.

    The 'sub_dir' field is conditionally shown in list item previews via always_viewable_fields.
    Since this config is Enableable, the sub_dir will only appear when enabled=True.
    This means disabled materialization configs won't clutter the preview with sub_dir.
    """

    # Override sub_dir for materialization-specific default
    sub_dir: Annotated[str, abbreviation("subdir")] = "checkpoints"
    """Subdirectory for materialized outputs (different from global 'images')."""

    enabled: Annotated[bool, abbreviation("enabled")] = False
    """Whether this materialization config is enabled. When False, config exists but materialization is disabled."""

    @classmethod
    def well_filter_inheritance_branch(cls) -> type[WellFilterConfig]:
        """Keep checkpoint selection on the step-filter branch, not final output."""
        return StepWellFilterConfig


# Define platform-aware default transport mode at module level
# TCP on Windows (no Unix domain socket support), IPC on Unix/Mac
_DEFAULT_TRANSPORT_MODE = (
    TransportMode.TCP if platform.system() == "Windows" else TransportMode.IPC
)


@abbreviation("stream")
@global_pipeline_config(always_viewable_fields=["well_filter"])
@dataclass(frozen=True)
class StreamingDefaults(Enableable, StepWellFilterConfig):
    """Default configuration for streaming to visualizers.

    An unset viewer well filter inherits broader well scope. A viewer override
    can narrow emission, but it cannot undo loading or processing already chosen
    by the pipeline execution-domain filter.

    The 'persistent' field is conditionally shown in list item previews via
    always_viewable_fields. Since this config is Enableable, the persistent field
    will only appear when enabled=True. This means disabled streaming configs won't
    clutter the preview with persistence info, but enabled ones will show whether
    the viewer persists after pipeline completion.
    """

    persistent: Annotated[bool, abbreviation("persist")] = True
    """Whether viewer stays open after pipeline completion."""

    host: Annotated[str, abbreviation("host")] = "localhost"
    """Host for streaming communication. Use 'localhost' for local, or remote IP for network streaming."""

    transport_mode: Annotated[TransportMode, abbreviation("transport")] = (
        _DEFAULT_TRANSPORT_MODE
    )
    """ZMQ transport mode: Platform-aware default (TCP on Windows, IPC on Unix/Mac)."""

    enabled: Annotated[bool, abbreviation("enabled")] = False
    """Whether this streaming config is enabled. When False, config exists but streaming is disabled."""

    batch_size: Annotated[Optional[int], abbreviation("batch")] = None
    """Number of images to batch before displaying.
    
    None = wait for all images in the current operation, then display once (fastest, default).
    N = show incrementally every N images (provides visual feedback but slower).
    """

    scope_accent_color: Annotated[Optional[str], abbreviation("accent")] = None
    """Exact owner-projected scope accent used to frame a managed viewer."""


@abbreviation("stream_cfg")
@global_pipeline_config(ui_hidden=True)
@dataclass(frozen=True)
class StreamingConfig(StreamingDefaults, ABC, metaclass=StreamingConfigMeta):
    """Abstract base configuration for streaming to visualizers.

    Uses multiple inheritance from StepWellFilterConfig and StreamingDefaults.
    Inherited fields (persistent, host, port, transport_mode) are automatically set to None
    by @global_pipeline_config(inherit_as_none=True), enabling polymorphic access without
    type-specific attribute names.
    """

    # AutoRegisterMeta configuration - subclasses auto-register by snake_case class name
    __registry__: ClassVar[dict[str, type["StreamingConfig"]]]
    __registry_key__ = "_streaming_config_key"
    __key_extractor__ = (
        lambda class_name, cls: __import__("re")
        .sub(r"(?<!^)(?=[A-Z])", "_", class_name)
        .lower()
    )
    _streaming_config_key = (
        None  # Will be set by AutoRegisterMeta to snake_case class name
    )

    @property
    @abstractmethod
    def port(self) -> int:
        """Port for streaming communication. Each streamer type has its own default."""
        pass

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Backend enum for this streaming type."""
        pass

    @property
    @abstractmethod
    def viewer_type(self) -> str:
        """Viewer type identifier (e.g., 'napari', 'fiji') for queue tracking and logging."""
        pass

    @property
    @abstractmethod
    def streaming_config_key(self) -> str:
        """ObjectState/registry field key for this streaming config."""
        pass

    @property
    @abstractmethod
    def step_plan_output_key(self) -> str:
        """Key to use in step_plan for this config's output paths."""
        pass

    @property
    @abstractmethod
    def viewer_title(self) -> str:
        """Title shown by this streaming viewer."""
        pass

    @classmethod
    @abstractmethod
    def port_from_config(cls, config) -> Optional[int]:
        """Read this streaming config type's port from an ObjectState-backed config."""
        pass

    @abstractmethod
    def viewer_surface(self, source):
        """Return viewer transport/display/source surface for this source identity."""
        pass

    @abstractmethod
    def streaming_viewer_surface(self, context):
        """Return viewer transport/display/source surface for a processing context."""
        pass

    @abstractmethod
    def create_visualizer(
        self,
        filemanager,
        visualizer_config=None,
        transport_config=None,
    ):
        """Create and return the appropriate visualizer for this streaming config."""
        pass

    @classmethod
    def supported_config_keys(cls) -> tuple[str, ...]:
        """Registered ObjectState field keys for streaming configs."""
        return tuple(sorted(cls.__registry__))

    @classmethod
    def config_type_for_key(cls, config_key: str) -> type["StreamingConfig"]:
        """Return the registered streaming config type for an ObjectState field key."""
        return cls.__registry__[config_key]

    @classmethod
    def display_name_for_config_key(cls, config_key: str) -> str:
        """Return viewer display text for an ObjectState streaming config field key."""
        return cls.config_type_for_key(config_key)().display_name


from openhcs.core.streaming_config_declarations import (
    FIJI_STREAMING_CONFIG_SPEC,
    NAPARI_STREAMING_CONFIG_SPEC,
    StreamingViewerConfigSpec,
)
from openhcs.core.streaming_config_factory import StreamingConfigBehaviorMixin


@abbreviation("nap")
@global_pipeline_config(preview_label="NAP")
@dataclass(frozen=True)
class NapariStreamingConfig(
    StreamingConfigBehaviorMixin,
    StreamingConfig,
    NapariDisplayConfig,
):
    """Streaming configuration for Napari."""

    _streaming_config_key: ClassVar[str] = NAPARI_STREAMING_CONFIG_SPEC.registry_key
    streaming_spec: ClassVar[StreamingViewerConfigSpec] = NAPARI_STREAMING_CONFIG_SPEC
    port: int = 5555
    """Napari viewer transport port; choose a free local port when streaming is enabled."""


@abbreviation("fiji")
@global_pipeline_config(preview_label="FIJI")
@dataclass(frozen=True)
class FijiStreamingConfig(
    StreamingConfigBehaviorMixin,
    StreamingConfig,
    FijiDisplayConfig,
):
    """Streaming configuration for Fiji."""

    _streaming_config_key: ClassVar[str] = FIJI_STREAMING_CONFIG_SPEC.registry_key
    streaming_spec: ClassVar[StreamingViewerConfigSpec] = FIJI_STREAMING_CONFIG_SPEC
    port: int = 5565
    """Fiji viewer transport port; choose a free local port when streaming is enabled."""

    fiji_executable_path: Optional[Path] = None
    """Optional path to the Fiji executable; None uses normal executable discovery."""

# Inject all accumulated fields at the end of module loading.
# Important: use the same lazy_factory module that registered injections
# (objectstate.lazy_factory). Importing via openhcs.config_framework.lazy_factory
# would create a second module with its own empty registry.
from objectstate.lazy_factory import _inject_all_pending_fields

_inject_all_pending_fields()


def runtime_config_parameter(
    parameter: inspect.Parameter,
) -> inspect.Parameter | None:
    """Resolve a callable parameter owned by an exact PipelineConfig field."""
    from dataclasses import fields as dataclass_fields
    from objectstate.lazy_factory import LazyDataclass

    matching_fields = tuple(
        config_field
        for config_field in dataclass_fields(PipelineConfig)
        if config_field.name == parameter.name
        and isinstance(config_field.type, type)
        and issubclass(config_field.type, LazyDataclass)
    )
    if not matching_fields:
        return None
    if len(matching_fields) != 1:
        raise TypeError(
            f"PipelineConfig declares multiple runtime config fields named "
            f"{parameter.name!r}."
        )
    config_type = matching_fields[0].type
    if not isinstance(parameter.annotation, type) or not issubclass(
        config_type,
        parameter.annotation,
    ):
        return None
    return parameter.replace(annotation=config_type, default=config_type())

SourceBindingsConfig = source_binding_configs.SourceBindingsConfig
StepSourceBindingsConfig = source_binding_configs.StepSourceBindingsConfig
LazySourceBindingsConfig = source_binding_configs.LazySourceBindingsConfig
LazyStepSourceBindingsConfig = source_binding_configs.LazyStepSourceBindingsConfig


# ============================================================================
# Streaming Port Utilities
# ============================================================================

# Import streaming port utility from factory module
from openhcs.core.streaming_config_factory import get_all_streaming_ports


# ============================================================================
# Configuration Framework Initialization
# ============================================================================

# Initialize configuration framework with OpenHCS types
from openhcs.config_framework import set_base_config_type

set_base_config_type(GlobalPipelineConfig)

from objectstate import config_context
from objectstate.lazy_factory import resolve_lazy_configurations_for_serialization

with config_context(PipelineConfig()):
    source_binding_configs.EMPTY_SOURCE_BINDINGS = (
        resolve_lazy_configurations_for_serialization(
            LazyStepSourceBindingsConfig(),
        )
    )

# Note: We use the framework's default MRO-based priority function.
# More derived classes automatically get higher priority through MRO depth.
# No custom priority function needed - the framework handles it generically.

logger.debug("Configuration framework initialized with OpenHCS types")

# PERFORMANCE OPTIMIZATION: Cache warming is now done asynchronously in GUI startup
# to avoid blocking imports. For non-GUI contexts (CLI, subprocess), cache warming
# happens on-demand when config windows are first opened.

# NOTE: Step editor cache warming is done in openhcs.core.steps.__init__ to avoid circular imports
