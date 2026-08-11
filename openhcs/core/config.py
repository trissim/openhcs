"""
Global configuration dataclasses for OpenHCS.

This module defines the primary configuration objects used throughout the application,
such as VFSConfig, PathPlanningConfig, and the overarching GlobalPipelineConfig.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Annotated, ClassVar, Callable
from enum import Enum
from abc import ABC, abstractmethod
from arraybridge.decorators import DtypeConversion, DtypeConversionConfig
from polystore import config as _polystore_config
from openhcs.constants import (
    AllComponents,
    Microscope,
    SequentialComponents,
    VariableComponents,
    GroupBy,
)
from openhcs.constants.constants import (
    Backend,
    get_default_variable_components,
    get_default_group_by,
)
from metaclass_registry import AutoRegisterMeta
from openhcs.constants.input_source import InputSource
from python_introspect import AnnotatedDataclassValidationMixin, Enableable
from python_introspect.enableable import EnableableMeta
from polystore.streaming.viewer_transport import ViewerDisplayConfigABC
from zmqruntime.config import (
    NonBlankString,
    PositiveFloat,
    PositiveInteger,
    TcpPort,
    TransportMode,
)
from zmqruntime.viewer_protocol import ViewerDisplayConfigWireField, ViewerWireValue
from zmqruntime.transport import get_default_transport_mode

from openhcs.core.runtime_plane_projection import RuntimeSliceInvariantValue
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.core.vfs_protocol import PlateOutputDirectory, PlateOutputFile
from openhcs.utils.environment import OpenHCSProcessEnvironment

# Import decorator for automatic decorator creation


# Combined metaclass for StreamingConfig to support both Enableable and AutoRegisterMeta
class StreamingConfigMeta(EnableableMeta, AutoRegisterMeta):
    """Combined metaclass supporting Enableable semantics and AutoRegisterMeta registration."""

    pass


from objectstate import auto_create_decorator, abbreviation

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
    """Control-normalization declarations with member-owned calculations."""

    FOLD_CHANGE = (
        "fold_change",
        lambda value, control_mean, _control_std: value / control_mean
        if control_mean
        else None,
    )
    Z_SCORE = (
        "z_score",
        lambda value, control_mean, control_std: (value - control_mean) / control_std
        if control_std
        else None,
    )
    PERCENT_CONTROL = (
        "percent_control",
        lambda value, control_mean, _control_std: (value / control_mean) * 100
        if control_mean
        else None,
    )

    def __new__(
        cls,
        serialized_value: str,
        operation: Callable[[float, float, float], float | None],
    ) -> "NormalizationMethod":
        member = object.__new__(cls)
        member._value_ = serialized_value
        member._operation = operation
        return member

    def normalize(
        self,
        value: float,
        *,
        control_mean: float,
        control_std: float,
    ) -> float | None:
        """Normalize one value against its control reference."""
        return self._operation(value, control_mean, control_std)


class MultiprocessingStartMethod(Enum):
    """Process start methods for OpenHCS worker pools."""

    SPAWN = "spawn"
    FORK = "fork"
    FORKSERVER = "forkserver"


@abbreviation("gpc")
@auto_create_decorator
@dataclass(frozen=True)
class GlobalPipelineConfig(AnnotatedDataclassValidationMixin):
    """
    Root configuration object for an OpenHCS pipeline session.
    This object is intended to be instantiated at application startup and treated as immutable.
    """

    materialization_results_path: Annotated[Path, abbreviation("results_path")] = field(
        default_factory=lambda: Path("results"),
    )
    """
    Directory for materialized named analysis artifacts such as CSV and JSON files.

    A relative path is resolved inside the compiled output plate root; an
    absolute path is used unchanged. This pipeline-wide destination is separate
    from ordinary image outputs and per-step main-flow checkpoints.
    """

    materialize_runtime_artifacts: Annotated[
        bool, abbreviation("mat_artifacts")
    ] = True
    """Persist named runtime artifacts through the compiled artifact-output plan.

    When disabled, measurements, tables, labels, and other named outputs remain
    available to downstream steps but are not saved unless their own artifact
    declaration explicitly requests persistence. This setting does not control
    ordinary main-flow image checkpoints.
    """

    num_workers: Annotated[PositiveInteger, abbreviation("W")] = 1
    """Maximum worker count used for parallel execution and GPU allocation.

    A value of one executes one schedulable work item at a time. Actual
    concurrency may be lower when a pipeline, backend, or available device
    imposes a tighter limit.
    """

    microscope: Annotated[Microscope, abbreviation("scope")] = field(
        default_factory=lambda: Microscope.AUTO,
    )
    """Microscope/source-layout handler used to ingest the plate.

    ``AUTO`` selects a registered handler from source evidence. Choose an exact
    handler to override detection, or ``SOURCE_BINDINGS`` for a generic image
    folder described by ``source_bindings_config``.
    """

    use_threading: Annotated[bool, abbreviation("threading")] = field(
        default_factory=OpenHCSProcessEnvironment.use_threading_mode,
    )
    """Use a shared-process thread pool instead of worker processes.

    Threading is useful for debugging and workloads that release the GIL;
    process workers provide isolation for the normal execution path. The
    ``OPENHCS_USE_THREADING`` environment variable supplies the startup default.
    """

    multiprocessing_start_method: Annotated[
        MultiprocessingStartMethod,
        abbreviation("mp_start"),
    ] = field(
        default_factory=lambda: MultiprocessingStartMethod.SPAWN,
    )
    """Operating-system start method used when process workers are enabled.

    ``SPAWN`` starts clean interpreters and is safe for CUDA. ``FORK`` inherits
    process state and is restricted to CPU use; ``FORKSERVER`` creates workers
    through a clean server process where the platform supports it.
    """

    auto_add_output_plate_to_plate_manager: Annotated[
        bool, abbreviation("auto_add_output_plate")
    ] = False
    """If True, when a plate run completes successfully, the computed output plate root
    (from path planning) is automatically added to Plate Manager as a new orchestrator
    if it is not already present."""

    # Future extension point:
    # logging_config: Optional[Dict[str, Any]] = None # For configuring logging levels, handlers
    # plugin_settings: Dict[str, Any] = field(default_factory=dict) # For plugin-specific settings


# PipelineConfig will be created automatically by the injection system
# (GlobalPipelineConfig → PipelineConfig by removing "Global" prefix)

class NapariDimensionMode(Enum):
    """How component values are placed in Napari image layers."""

    LAYER = "layer"  # Create a separate Napari layer for each value
    STACK = "stack"  # Stack values along an axis in one Napari layer


class NapariVariableSizeHandling(Enum):
    """How to handle images with different sizes in the same layer."""

    SEPARATE_LAYERS = (
        "separate_layers"  # Create separate layers per well (preserves exact data)
    )
    PAD_TO_MAX = "pad_to_max"  # Pad smaller images to match largest (enables stacking)


@dataclass(frozen=True)
class NapariDisplayConfig(
    ViewerDisplayConfigABC,
    AnnotatedDataclassValidationMixin,
):
    """Map streamed OpenHCS dimensions and intensity data onto Napari layers."""

    COMPONENT_ORDER: ClassVar[tuple[str, ...]] = AllComponents.ordered_names()

    colormap: NonBlankString = field(
        default="gray",
        metadata={
            "description": (
                "Name of a colormap registered in the installed Napari viewer. "
                "Napari validates this extensible registry name when displaying "
                "the layer."
            )
        },
    )
    variable_size_handling: NapariVariableSizeHandling = field(
        default=NapariVariableSizeHandling.PAD_TO_MAX,
        metadata={
            "description": (
                "How Napari handles streamed images with different spatial "
                "dimensions: preserve each shape in separate layers or pad "
                "smaller images to the largest shape before stacking."
            )
        },
    )
    site_mode: NapariDimensionMode = field(
        default=NapariDimensionMode.STACK,
        metadata={
            "description": "Whether site values are stacked or use separate layers."
        },
    )
    channel_mode: NapariDimensionMode = field(
        default=NapariDimensionMode.STACK,
        metadata={
            "description": "Whether channel values are stacked or use separate layers."
        },
    )
    z_index_mode: NapariDimensionMode = field(
        default=NapariDimensionMode.STACK,
        metadata={
            "description": "Whether z-index values are stacked or use separate layers."
        },
    )
    timepoint_mode: NapariDimensionMode = field(
        default=NapariDimensionMode.STACK,
        metadata={
            "description": (
                "Whether timepoint values are stacked or use separate layers."
            )
        },
    )
    well_mode: NapariDimensionMode = field(
        default=NapariDimensionMode.STACK,
        metadata={
            "description": "Whether well values are stacked or use separate layers."
        },
    )

    def component_modes(self) -> dict[str, str]:
        """Project typed component fields onto the viewer wire vocabulary."""

        return {
            AllComponents.SITE.value: self.site_mode.value,
            AllComponents.CHANNEL.value: self.channel_mode.value,
            AllComponents.Z_INDEX.value: self.z_index_mode.value,
            AllComponents.TIMEPOINT.value: self.timepoint_mode.value,
            AllComponents.WELL.value: self.well_mode.value,
        }

    def display_payload_extra(self) -> dict[str, str]:
        """Project Napari-specific display settings onto the wire payload."""

        from polystore.napari_stream import NapariDisplayWireField

        return {
            NapariDisplayWireField.COLORMAP.value: self.colormap,
            NapariDisplayWireField.VARIABLE_SIZE_HANDLING.value: (
                self.variable_size_handling.value
            ),
        }

    @classmethod
    def from_display_payload(
        cls,
        payload: Mapping[str, ViewerWireValue],
    ) -> "NapariDisplayConfig":
        """Rehydrate the typed config at the Napari viewer wire boundary."""

        from polystore.napari_stream import NapariDisplayWireField

        component_modes = payload[ViewerDisplayConfigWireField.COMPONENT_MODES.value]
        if not isinstance(component_modes, Mapping):
            raise TypeError("Napari component_modes must be a mapping.")
        if set(component_modes) != set(cls.COMPONENT_ORDER):
            raise ValueError(
                "Napari component_modes must exactly match "
                f"{cls.COMPONENT_ORDER!r}; got {tuple(component_modes)!r}."
            )
        colormap = str(payload[NapariDisplayWireField.COLORMAP.value]).strip()
        if not colormap:
            raise ValueError("Napari colormap name must not be blank.")
        return cls(
            colormap=colormap,
            variable_size_handling=NapariVariableSizeHandling(
                str(payload[NapariDisplayWireField.VARIABLE_SIZE_HANDLING.value])
            ),
            site_mode=NapariDimensionMode(
                str(component_modes[AllComponents.SITE.value])
            ),
            channel_mode=NapariDimensionMode(
                str(component_modes[AllComponents.CHANNEL.value])
            ),
            z_index_mode=NapariDimensionMode(
                str(component_modes[AllComponents.Z_INDEX.value])
            ),
            timepoint_mode=NapariDimensionMode(
                str(component_modes[AllComponents.TIMEPOINT.value])
            ),
            well_mode=NapariDimensionMode(
                str(component_modes[AllComponents.WELL.value])
            ),
        )

# Apply the global pipeline config decorator with ui_hidden=True
# This config is only inherited by NapariStreamingConfig, so hide it from UI
NapariDisplayConfig = global_pipeline_config(ui_hidden=True)(NapariDisplayConfig)


# ============================================================================
# Fiji Display Configuration
# ============================================================================


class FijiDimensionMode(Enum):
    """
    How to map OpenHCS dimensions to ImageJ hyperstack dimensions.

    ImageJ hyperstacks have 3 dimensions: Channels (C), Slices (Z), Frames (T).
    Each OpenHCS component (site, channel, z_index, timepoint) can be mapped to one of these.

    - WINDOW: Create separate windows for each value (like Napari LAYER mode)
    - CHANNEL: Map to ImageJ Channel dimension (C)
    - SLICE: Map to ImageJ Slice dimension (Z)
    - FRAME: Map to ImageJ Frame dimension (T)
    """

    WINDOW = "window"  # Separate windows (like Napari LAYER mode)
    CHANNEL = "channel"  # ImageJ Channel dimension (C)
    SLICE = "slice"  # ImageJ Slice dimension (Z)
    FRAME = "frame"  # ImageJ Frame dimension (T)


@dataclass(frozen=True)
class FijiDisplayConfig(
    ViewerDisplayConfigABC,
    AnnotatedDataclassValidationMixin,
):
    """Map streamed OpenHCS dimensions and intensity data onto Fiji hyperstacks."""

    COMPONENT_ORDER: ClassVar[tuple[str, ...]] = AllComponents.ordered_names()

    lut: NonBlankString = field(
        default="Grays",
        metadata={
            "description": (
                "Name of a lookup table available to the installed Fiji/ImageJ "
                "runtime, including plugin-provided LUTs."
            )
        },
    )
    auto_contrast: bool = field(
        default=True,
        metadata={
            "description": (
                "Automatically set Fiji display limits from the streamed image data."
            )
        },
    )
    site_mode: FijiDimensionMode = field(
        default=FijiDimensionMode.FRAME,
        metadata={
            "description": (
                "ImageJ dimension used for site values, or WINDOW for separate windows."
            )
        },
    )
    channel_mode: FijiDimensionMode = field(
        default=FijiDimensionMode.CHANNEL,
        metadata={
            "description": (
                "ImageJ dimension used for channel values, or WINDOW for separate windows."
            )
        },
    )
    z_index_mode: FijiDimensionMode = field(
        default=FijiDimensionMode.SLICE,
        metadata={
            "description": (
                "ImageJ dimension used for z-index values, or WINDOW for separate windows."
            )
        },
    )
    timepoint_mode: FijiDimensionMode = field(
        default=FijiDimensionMode.FRAME,
        metadata={
            "description": (
                "ImageJ dimension used for timepoint values, or WINDOW for separate windows."
            )
        },
    )
    well_mode: FijiDimensionMode = field(
        default=FijiDimensionMode.FRAME,
        metadata={
            "description": (
                "ImageJ dimension used for well values, or WINDOW for separate windows."
            )
        },
    )

    def component_modes(self) -> dict[str, str]:
        """Project typed component fields onto the viewer wire vocabulary."""

        return {
            AllComponents.SITE.value: self.site_mode.value,
            AllComponents.CHANNEL.value: self.channel_mode.value,
            AllComponents.Z_INDEX.value: self.z_index_mode.value,
            AllComponents.TIMEPOINT.value: self.timepoint_mode.value,
            AllComponents.WELL.value: self.well_mode.value,
        }

    def display_payload_extra(self) -> dict[str, str | bool]:
        """Project Fiji-specific display settings onto the wire payload."""

        from polystore.fiji_stream import FijiDisplayWireField

        return {
            FijiDisplayWireField.LUT.value: self.lut,
            FijiDisplayWireField.AUTO_CONTRAST.value: self.auto_contrast,
        }

    @classmethod
    def from_display_payload(
        cls,
        payload: Mapping[str, ViewerWireValue],
    ) -> "FijiDisplayConfig":
        """Rehydrate the typed config at the Fiji viewer wire boundary."""

        from polystore.fiji_stream import FijiDisplayWireField

        component_modes = payload[ViewerDisplayConfigWireField.COMPONENT_MODES.value]
        if not isinstance(component_modes, Mapping):
            raise TypeError("Fiji component_modes must be a mapping.")
        if set(component_modes) != set(cls.COMPONENT_ORDER):
            raise ValueError(
                "Fiji component_modes must exactly match "
                f"{cls.COMPONENT_ORDER!r}; got {tuple(component_modes)!r}."
            )
        lut = str(payload[FijiDisplayWireField.LUT.value]).strip()
        if not lut:
            raise ValueError("Fiji LUT name must not be blank.")
        auto_contrast = payload[FijiDisplayWireField.AUTO_CONTRAST.value]
        if not isinstance(auto_contrast, bool):
            raise TypeError(
                "Fiji auto_contrast must be bool, "
                f"got {type(auto_contrast).__name__}."
            )
        return cls(
            lut=lut,
            auto_contrast=auto_contrast,
            site_mode=FijiDimensionMode(
                str(component_modes[AllComponents.SITE.value])
            ),
            channel_mode=FijiDimensionMode(
                str(component_modes[AllComponents.CHANNEL.value])
            ),
            z_index_mode=FijiDimensionMode(
                str(component_modes[AllComponents.Z_INDEX.value])
            ),
            timepoint_mode=FijiDimensionMode(
                str(component_modes[AllComponents.TIMEPOINT.value])
            ),
            well_mode=FijiDimensionMode(
                str(component_modes[AllComponents.WELL.value])
            ),
        )

# Apply the global pipeline config decorator with ui_hidden=True
# This config is only inherited by FijiStreamingConfig, so hide it from UI
FijiDisplayConfig = global_pipeline_config(ui_hidden=True)(FijiDisplayConfig)


@abbreviation("wfc")
@global_pipeline_config
@dataclass(frozen=True)
class WellFilterConfig(AnnotatedDataclassValidationMixin):
    """Base execution-domain filter inherited by specialized well policies.

    At pipeline scope this constrains which wells compile and execute. Nominal
    subclasses reuse the same selection for their own narrower behavior, such
    as main-flow persistence, step checkpoints, or viewer emission.
    """

    well_filter: Annotated[Optional[Union[List[str], str, int]], abbreviation("")] = (
        None
    )
    """Well selection matched against the plate's ordered available wells.

    Use a list for exact IDs; a string for one ID, comma-separated IDs,
    ``row:A``, ``col:01-06``, or an inclusive ``A01:A12`` range; or a
    non-negative integer for the first N available wells. ``None`` bypasses
    filtering. Zero matches no wells, so include mode selects none and exclude
    mode selects all.
    """

    well_filter_mode: Annotated[WellFilterMode, abbreviation("filter_mode")] = (
        WellFilterMode.INCLUDE
    )
    """Apply ``well_filter`` as the wells to include or the wells to exclude.

    Include mode rejects unknown explicit wells. Exclude mode ignores unknown
    explicit wells and preserves every unmatched available well.
    """

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
class ZarrConfig(AnnotatedDataclassValidationMixin, _polystore_config.ZarrConfig):
    """OpenHCS registration of PolyStore's Zarr configuration owner.

    OME-ZARR metadata and plate metadata are always enabled for HCS compliance.
    Shuffle filter is always enabled for Blosc compressor (ignored for others).
    """


@abbreviation("vfs")
@global_pipeline_config(always_viewable_fields=["materialization_backend"])
@dataclass(frozen=True)
class VFSConfig(AnnotatedDataclassValidationMixin):
    """Choose storage backends independently for input, runtime, and saved data."""

    read_backend: Annotated[Backend, abbreviation("read")] = Backend.AUTO
    """Backend used to open pipeline input data.

    ``AUTO`` selects from source metadata and layout. An explicit backend forces
    that reader and should only be used when the source representation is known.
    """

    intermediate_backend: Annotated[Backend, abbreviation("intermediate")] = (
        Backend.MEMORY
    )
    """Backend for ordinary main-flow values passed between pipeline steps.

    These values are runtime data, not promised output files; choose memory for
    speed or a disk-backed backend when intermediate arrays exceed available RAM.
    """

    materialization_backend: Annotated[
        MaterializationBackend, abbreviation("materialize")
    ] = MaterializationBackend.DISK
    """Persistent backend used by compiled main-flow and artifact materialization plans.

    This setting selects how requested outputs are stored; it does not itself
    decide which steps or artifacts are materialized.
    """


@abbreviation("dtype")
@global_pipeline_config
@dataclass(frozen=True)
class DtypeConfig(
    AnnotatedDataclassValidationMixin,
    RuntimeSliceInvariantValue,
    DtypeConversionConfig,
):
    """Default output dtype policy for memory-type-decorated processing functions."""

    default_dtype_conversion: Annotated[DtypeConversion, abbreviation("conv")] = (
        DtypeConversion.NATIVE_OUTPUT
    )
    """Fallback dtype conversion for decorated callables without an override.

    ``NATIVE_OUTPUT`` keeps the callable's natural output dtype and values.
    ``PRESERVE_INPUT`` rescales and casts the result to the input dtype. A
    callable- or step-level declaration can override this pipeline default.
    """

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
class ProcessingConfig(AnnotatedDataclassValidationMixin):
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
class SequentialProcessingConfig(AnnotatedDataclassValidationMixin):
    """Pipeline-level configuration for sequential processing mode.

    Sequential processing changes the orchestrator's execution flow to process
    one combination at a time through all steps, reducing memory usage.
    This is a pipeline-level setting, not per-step.
    """

    sequential_components: Annotated[
        List[SequentialComponents], abbreviation("seq_comp")
    ] = field(default_factory=list)
    """Plate components whose value combinations run through the whole pipeline in turn.

    After one combination completes all steps, its runtime data can be released
    before the next combination begins. This bounds peak memory but may reduce
    parallelism. A sequential component cannot also be a step's assembled
    ``variable_components`` axis.
    """


@abbreviation("analysis")
@global_pipeline_config
@dataclass(frozen=True)
class AnalysisConsolidationConfig(AnnotatedDataclassValidationMixin, Enableable):
    """Combine materialized per-well analysis tables after plate execution."""

    enabled: Annotated[bool, abbreviation("")] = True
    """Run table discovery and summary generation after a plate finishes."""

    metaxpress_style: Annotated[bool, abbreviation("mx_style")] = True
    """Write MetaXpress-compatible metadata headers and grouped column ordering.

    When false, the consolidated result is a plain CSV with ``Well`` first and
    remaining columns sorted by name.
    """

    file_extensions: Annotated[tuple[str, ...], abbreviation("exts")] = (".csv",)
    """Exact filename suffixes considered when discovering analysis tables."""

    exclude_patterns: Annotated[tuple[str, ...], abbreviation("exclude")] = (
        r".*consolidated.*",
        r".*metaxpress.*",
        r".*summary.*",
    )
    """Regular expressions matched against filenames after extension filtering.

    Matching files are skipped; the defaults prevent prior summaries from being
    recursively consolidated into a new summary.
    """

    output_filename: Annotated[str, abbreviation("out_file")] = (
        "metaxpress_style_summary.csv"
    )
    """Filename for the summary produced from one plate's included analysis tables."""

    global_summary_filename: Annotated[str, abbreviation("global_sum")] = (
        "global_metaxpress_summary.csv"
    )
    """Filename for the optional summary that combines completed plate summaries."""


@abbreviation("plate")
@global_pipeline_config
@dataclass(frozen=True)
class PlateMetadataConfig(AnnotatedDataclassValidationMixin):
    """Metadata written into MetaXpress-compatible consolidated result headers."""

    barcode: Annotated[Optional[str], abbreviation("barcode")] = None
    """Barcode written to the summary header; ``None`` derives one from the results directory."""

    plate_name: Annotated[Optional[str], abbreviation("name")] = None
    """Plate name written to the summary header; ``None`` uses the results directory name."""

    plate_id: Annotated[Optional[str], abbreviation("id")] = None
    """Plate identifier written to the summary header; ``None`` derives a numeric value from the results path for the current process."""

    description: Annotated[Optional[str], abbreviation("description")] = None
    """Experiment description written to the summary header; ``None`` reports the number of analysed wells."""

    acquisition_user: Annotated[str, abbreviation("user")] = "OpenHCS"
    """Acquisition-user text written to the MetaXpress-compatible header."""

    z_step: Annotated[PositiveFloat, abbreviation("z_step")] = 1.0
    """Positive Z-plane spacing recorded in the MetaXpress-compatible header."""


@dataclass(frozen=True)
class ExperimentalAnalysisConfig(AnnotatedDataclassValidationMixin):
    """Standalone configuration for the experimental-analysis engine."""

    config_file_name: Annotated[str, abbreviation("config")] = "config.xlsx"
    """Name of the experimental configuration Excel file."""

    results_file_name: Annotated[str, abbreviation("results")] = (
        "metaxpress_style_summary.csv"
    )
    """Name of the consolidated microscope results file."""

    compiled_results_file_name: Annotated[str, abbreviation("output")] = (
        "compiled_results_normalized.xlsx"
    )
    """Name of the normalized analysis workbook written by the directory workflow."""

    raw_results_file_name: Annotated[str, abbreviation("raw_output")] = (
        "compiled_results_raw.xlsx"
    )
    """Name of the non-normalized analysis workbook written when raw export is enabled."""

    heatmap_file_name: Annotated[str, abbreviation("heatmap")] = "heatmaps.xlsx"
    """Name of the heatmap workbook written when heatmap export is enabled."""

    design_sheet_name: Annotated[str, abbreviation("design")] = "drug_curve_map"
    """Name of the sheet containing experimental design."""

    plate_groups_sheet_name: Annotated[str, abbreviation("groups")] = "plate_groups"
    """Name of the sheet containing plate group mappings."""

    normalization_method: Annotated[NormalizationMethod, abbreviation("norm")] = (
        NormalizationMethod.FOLD_CHANGE
    )
    """Replicate-local control transformation: value/control mean, control
    z-score, or percentage of control mean. A zero required denominator produces
    a missing normalized value rather than an infinite result."""

    export_raw_results: Annotated[bool, abbreviation("raw")] = True
    """Whether to export raw (non-normalized) results."""

    export_heatmaps: Annotated[bool, abbreviation("heatmaps")] = True
    """Whether to generate heatmap visualizations."""

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
    """Non-empty suffix appended to the input plate name to form the output plate root."""

    global_output_folder: Annotated[
        Optional[PlateOutputDirectory], abbreviation("global_folder")
    ] = None
    """
    Optional parent directory for output plate roots.

    ``None`` creates the output root beside the input plate. An explicit path is
    resolved to an absolute directory during compilation, then combined with the
    input plate name and ``output_dir_suffix``.
    """

    sub_dir: Annotated[str, abbreviation("subdir")] = "images"
    """
    Relative directory inside the output plate root for ordinary processed images.

    This does not choose the analysis-artifact results directory or a step
    checkpoint directory; those are owned by their respective declarations.
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
    """Relative directory inside the output plate root for this step checkpoint."""

    enabled: Annotated[bool, abbreviation("enabled")] = False
    """Persist this step's ordinary main-flow result when enabled.

    Named artifact outputs have their own persistence plan and are unaffected by
    this switch.
    """

    @classmethod
    def well_filter_inheritance_branch(cls) -> type[WellFilterConfig]:
        """Keep checkpoint selection on the step-filter branch, not final output."""
        return StepWellFilterConfig


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
    """Keep the managed viewer open after the pipeline stops sending data."""

    host: Annotated[NonBlankString, abbreviation("host")] = "localhost"
    """Viewer host used by the selected transport.

    Use ``localhost`` for a viewer on this machine. A remote host requires TCP;
    IPC transports remain local regardless of this value.
    """

    transport_mode: Annotated[TransportMode, abbreviation("transport")] = (
        get_default_transport_mode()
    )
    """Transport used for viewer data and control messages.

    The platform default is TCP on Windows and IPC on Unix-like systems. Choose
    TCP for remote viewers; IPC is local-only.
    """

    enabled: Annotated[bool, abbreviation("enabled")] = False
    """Emit this step's eligible outputs to the configured viewer when enabled."""

    scope_accent_color: Annotated[Optional[str], abbreviation("accent")] = field(
        default=None,
        metadata={"ui_hidden": True},
    )
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
    def port(self) -> TcpPort:
        """Port for streaming communication. Each streamer type has its own default."""
        pass

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Backend enum for this streaming type."""
        pass

    @property
    @abstractmethod
    def viewer_type(self) -> ViewerType:
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
    """Per-step Napari transport, scope filter, lifetime, and display policy."""

    _streaming_config_key: ClassVar[str] = NAPARI_STREAMING_CONFIG_SPEC.registry_key
    streaming_spec: ClassVar[StreamingViewerConfigSpec] = NAPARI_STREAMING_CONFIG_SPEC
    port: TcpPort = 5555
    """Napari viewer transport port; choose a free local port when streaming is enabled."""


@abbreviation("fiji")
@global_pipeline_config(preview_label="FIJI")
@dataclass(frozen=True)
class FijiStreamingConfig(
    StreamingConfigBehaviorMixin,
    StreamingConfig,
    FijiDisplayConfig,
):
    """Per-step Fiji transport, scope filter, lifetime, and display policy."""

    _streaming_config_key: ClassVar[str] = FIJI_STREAMING_CONFIG_SPEC.registry_key
    streaming_spec: ClassVar[StreamingViewerConfigSpec] = FIJI_STREAMING_CONFIG_SPEC
    port: TcpPort = 5565
    """Fiji viewer transport port; choose a free local port when streaming is enabled."""


@abbreviation("compile_dbg")
@global_pipeline_config(
    inherit_as_none=False,
    always_viewable_fields=["enabled"],
)
@dataclass(frozen=True)
class CompilationDebugConfig(AnnotatedDataclassValidationMixin, Enableable):
    """Optional persistence of compiler-owned diagnostic artifacts."""

    compiled_execution_bundle_path: Annotated[
        Optional[PlateOutputFile], abbreviation("bundle")
    ] = None
    """Destination for a pickle of the immutable compiled execution bundle.

    No bundle is written when this path is ``None``. The pickle is diagnostic
    state for reproducing or inspecting compilation, not a normal pipeline result.
    """


# Inject all accumulated fields at the end of module loading.
# Use the ObjectState owner that registered the pending field declarations.
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
from objectstate import set_base_config_type

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
