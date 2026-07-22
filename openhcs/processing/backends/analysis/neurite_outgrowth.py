"""MetaXpress-style 2D neurite outgrowth analysis.

The public controls mirror the documented MetaXpress Neurite Outgrowth module.
The opinionated implementation composes the existing CellProfiler-compatible
segmentation and skeleton leaves behind one simple callable boundary.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import heapq
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skan import Skeleton, summarize

from openhcs.core.artifacts import (
    ArtifactMeasurementSubjectRelation,
    ArtifactSpec,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectMeasurementSubjectRelation,
    SpatialGraphArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_label_domains import (
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_object_labels import (
    object_label_dense_array,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_spatial_graph import (
    SpatialGraph,
    SpatialGraphEdge,
    SpatialGraphNode,
)
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
    SpatialGraphROIOptions,
    SWCOptions,
)

from .count_cells_simple import (
    MetaXpressWavelengthSettings,
    segment_metaxpress_round_objects,
)
from .metaxpress_utils import HiddenPixelSize, local_background_response
from ..cellprofiler.feature_enhancement import (
    EnhanceMethod,
    NeuriteMethod,
    OperationMethod,
    enhance_or_suppress_features,
)
from ..cellprofiler.medial_axis import medialaxis
from ..cellprofiler.primary_objects import identify_primary_objects
from ..cellprofiler.secondary import SecondaryMethod, identify_secondary_objects
from ..cellprofiler.skeleton import measure_object_skeleton
from ..cellprofiler.thresholding import (
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    threshold,
)


class NeuriteIllumination(str, Enum):
    """Documented neurite-image illumination modes."""

    FLUORESCENCE = "fluorescence"
    TRANSMISSION = "transmission"


@dataclass(frozen=True)
class CellProfilerNeuriteEngineProfile:
    """Authoritative CP settings shared by compact and modular workflows."""

    body_min_diameter_px: int = 12
    compact_body_min_diameter_px: int = 10
    body_max_diameter_px: int = 100
    adaptive_window_size_px: int = 64
    tubeness_smoothing_px: float = 1.5
    threshold_correction_factor: float = 0.85
    threshold_smoothing_px: float = 1.0
    secondary_regularization_factor: float = 0.05

    def body_detection_kwargs(
        self,
        *,
        adaptive_window_size: int | None = None,
        exclude_size: bool = True,
        min_diameter: int | None = None,
    ) -> dict[str, object]:
        return {
            "min_diameter": (
                self.body_min_diameter_px if min_diameter is None else min_diameter
            ),
            "max_diameter": self.body_max_diameter_px,
            "exclude_size": exclude_size,
            "exclude_border_objects": False,
            "threshold_scope": CellProfilerThresholdScope.ADAPTIVE,
            "threshold_method": CellProfilerThresholdMethod.OTSU,
            "adaptive_window_size": (
                self.adaptive_window_size_px
                if adaptive_window_size is None
                else adaptive_window_size
            ),
        }

    def compact_body_detection_kwargs(
        self,
        *,
        adaptive_window_size: int,
    ) -> dict[str, object]:
        """Candidate settings for MetaXpress predicates and CP propagation."""

        return self.body_detection_kwargs(
            adaptive_window_size=adaptive_window_size,
            exclude_size=False,
            min_diameter=self.compact_body_min_diameter_px,
        )

    def enhancement_kwargs(
        self,
        *,
        smoothing_value: float | None = None,
    ) -> dict[str, object]:
        return {
            "method": OperationMethod.ENHANCE,
            "enhance_method": EnhanceMethod.NEURITES,
            "neurite_method": NeuriteMethod.TUBENESS,
            "smoothing_value": (
                self.tubeness_smoothing_px
                if smoothing_value is None
                else smoothing_value
            ),
            "neurite_rescale": True,
        }

    def threshold_kwargs(
        self,
        *,
        window_size: int | None = None,
        smoothing: float | None = None,
    ) -> dict[str, object]:
        return {
            "threshold_scope": CellProfilerThresholdScope.ADAPTIVE,
            "threshold_method": CellProfilerThresholdMethod.OTSU,
            "threshold_correction_factor": self.threshold_correction_factor,
            "window_size": (
                self.adaptive_window_size_px if window_size is None else window_size
            ),
            "smoothing": (
                self.threshold_smoothing_px if smoothing is None else smoothing
            ),
        }

    def secondary_kwargs(
        self,
        *,
        adaptive_window_size: int | None = None,
    ) -> dict[str, object]:
        return {
            "method": SecondaryMethod.PROPAGATION,
            "threshold_scope": CellProfilerThresholdScope.ADAPTIVE,
            "threshold_method": CellProfilerThresholdMethod.OTSU,
            "threshold_correction_factor": self.threshold_correction_factor,
            "adaptive_window_size": (
                self.adaptive_window_size_px
                if adaptive_window_size is None
                else adaptive_window_size
            ),
            "regularization_factor": self.secondary_regularization_factor,
            "fill_holes": True,
            "discard_edge_objects": False,
        }


CELLPROFILER_NEURITE_ENGINE_PROFILE = CellProfilerNeuriteEngineProfile()


@dataclass(frozen=True)
class MetaXpressCellBodySettings:
    """Documented cell-body controls for Neurite Outgrowth."""

    approximate_max_width: float = 30.0
    """Approximate maximum short-axis width in micrometers."""

    minimum_area: float = 50.0
    """Minimum cell-body area in square micrometers."""

    intensity_above_local_background: float = 100.0
    """Minimum absolute intensity difference from local background."""

    channel_index: int | None = None
    """Optional body channel; omitted means the neurite channel."""

    def validate(self) -> None:
        if (
            not np.isfinite(self.approximate_max_width)
            or self.approximate_max_width <= 0
        ):
            raise ValueError("cell_body.approximate_max_width must be > 0")
        if not np.isfinite(self.minimum_area) or self.minimum_area <= 0:
            raise ValueError("cell_body.minimum_area must be > 0")
        if (
            not np.isfinite(self.intensity_above_local_background)
            or self.intensity_above_local_background < 0
        ):
            raise ValueError("cell_body.intensity_above_local_background must be >= 0")
        if self.channel_index is not None and (
            isinstance(self.channel_index, bool)
            or not isinstance(self.channel_index, (int, np.integer))
            or self.channel_index < 0
        ):
            raise ValueError("cell_body.channel_index must be a non-negative integer")


@dataclass(frozen=True)
class MetaXpressOutgrowthSettings:
    """Documented outgrowth controls for Neurite Outgrowth."""

    maximum_width: float = 4.0
    """Maximum outgrowth width in micrometers."""

    intensity_above_local_background: float = 50.0
    """Minimum absolute intensity difference from local background."""

    minimum_cell_growth_to_log_as_significant: float = 10.0
    """Scoring-only total outgrowth threshold in micrometers."""

    def validate(self) -> None:
        if not np.isfinite(self.maximum_width) or self.maximum_width <= 0:
            raise ValueError("outgrowth.maximum_width must be > 0")
        if (
            not np.isfinite(self.intensity_above_local_background)
            or self.intensity_above_local_background < 0
        ):
            raise ValueError("outgrowth.intensity_above_local_background must be >= 0")
        if (
            not np.isfinite(self.minimum_cell_growth_to_log_as_significant)
            or self.minimum_cell_growth_to_log_as_significant < 0
        ):
            raise ValueError(
                "outgrowth.minimum_cell_growth_to_log_as_significant must be >= 0"
            )


@dataclass(frozen=True)
class MetaXpressNuclearSettings(MetaXpressWavelengthSettings):
    """Optional documented nuclear-stain controls."""

    channel_index: int = 1


@dataclass(frozen=True)
class NeuriteOutgrowthSummary:
    """MetaXpress-style image-level neurite measurements."""

    neurite_channel_index: int
    cell_body_channel_index: int
    nuclear_channel_index: int
    number_of_cells: int
    total_outgrowth_um: float
    mean_outgrowth_per_cell_um: float
    total_processes: int
    mean_processes_per_cell: float
    total_branches: int
    mean_branches_per_cell: float
    total_cell_body_area_um2: float
    mean_cell_body_area_um2: float
    straightness: float
    cells_significant_growth: int
    percent_cells_significant_growth: float
    mean_outgrowth_average_intensity: float
    resolved_crossovers: int


@dataclass(frozen=True)
class NeuriteOutgrowthCellResult:
    """MetaXpress-style cell-by-cell neurite measurements."""

    slice_index: int
    cell: int
    total_outgrowth_um: float
    processes: int
    mean_process_length_um: float
    median_process_length_um: float
    max_process_length_um: float
    branches: int
    straightness: float
    cell_body_area_um2: float
    mean_outgrowth_intensity: float
    significant_growth: bool


NEURITE_SUMMARY_OUTPUT = ArtifactSpec.output(
    "neurite_outgrowth_summary",
    MeasurementsArtifactType,
    materialization=MaterializationSpec(CsvOptions()),
    relations=(ArtifactMeasurementSubjectRelation(),),
)
CELL_BODIES_OUTPUT = ArtifactSpec.output(
    "cell_bodies",
    ObjectLabelsArtifactType,
    materialization=MaterializationSpec(ROIOptions()),
)
NEURITE_CELLS_OUTPUT = ArtifactSpec.output(
    "neurite_outgrowth_cells",
    MeasurementsArtifactType,
    materialization=MaterializationSpec(CsvOptions()),
    relations=(
        ObjectMeasurementSubjectRelation(
            source=CELL_BODIES_OUTPUT.ref(),
            id_field="cell",
        ),
    ),
)
NEURITE_LABELS_OUTPUT = ArtifactSpec.output(
    "neurite_outgrowth",
    ObjectLabelsArtifactType,
    materialization=MaterializationSpec(ROIOptions()),
)
UNIFIED_NEURONS_OUTPUT = ArtifactSpec.output(
    "neurons",
    ObjectLabelsArtifactType,
    materialization=MaterializationSpec(ROIOptions()),
)
NUCLEI_OUTPUT = ArtifactSpec.output(
    "nuclei",
    ObjectLabelsArtifactType,
    materialization=MaterializationSpec(ROIOptions()),
)
NEURITE_MORPHOLOGY_OUTPUT = ArtifactSpec.output(
    "neurite_morphology",
    SpatialGraphArtifactType,
    materialization=MaterializationSpec(
        SWCOptions(),
        SpatialGraphROIOptions(),
    ),
)


@dataclass(frozen=True)
class _TopologyResult:
    path_owners: np.ndarray
    path_distances: np.ndarray
    path_lengths: np.ndarray
    path_euclidean_lengths: np.ndarray
    path_coordinates: tuple[np.ndarray, ...]
    path_endpoint_groups: tuple[tuple[int, int], ...]
    path_branch_types: np.ndarray
    endpoint_group_coordinates: Mapping[int, tuple[float, float]]
    transitions: Mapping[int, tuple[int, ...]]
    root_paths_by_cell: Mapping[int, tuple[int, ...]]
    branch_owner: Mapping[int, int]
    crossing_nodes: frozenset[int]


def _raw_processing_leaf(func):
    """Resolve a composed leaf's runtime body through its callable contract."""

    return CallableContract.from_callable(func).resolve_raw_runtime_callable()


@numpy
@artifact_outputs(
    NEURITE_SUMMARY_OUTPUT,
    NEURITE_CELLS_OUTPUT,
    CELL_BODIES_OUTPUT,
    NEURITE_LABELS_OUTPUT,
    UNIFIED_NEURONS_OUTPUT,
    NUCLEI_OUTPUT,
    NEURITE_MORPHOLOGY_OUTPUT,
)
@artifact_inputs("pixel_size")
def neurite_outgrowth_metaxpress(
    image,
    neurite_channel_index: int = 0,
    illumination: NeuriteIllumination = NeuriteIllumination.FLUORESCENCE,
    cell_body: MetaXpressCellBodySettings = MetaXpressCellBodySettings(),
    outgrowth: MetaXpressOutgrowthSettings = MetaXpressOutgrowthSettings(),
    use_nuclear_stain: bool = False,
    nuclear_stain: MetaXpressNuclearSettings = MetaXpressNuclearSettings(),
    pixel_size: HiddenPixelSize = HiddenPixelSize(1.0),
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    DataclassMeasurementColumnarRows,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SpatialGraph,
]:
    """Measure cell bodies and attached neurites in one 2D channel stack.

    The user-facing controls follow the MetaXpress Neurite Outgrowth module:
    neurite image and illumination; optional cell-body channel, maximum width,
    minimum area, and local-background intensity; outgrowth maximum width,
    local-background intensity, and scoring threshold; plus an optional nuclear
    wavelength with minimum/maximum width and local-background intensity.

    This implementation is deliberately 2D. ``image`` must have shape
    ``(C, Y, X)`` and should be produced by a step whose variable component is
    ``CHANNEL``. Outgrowth detection is independent of the significant-growth
    threshold. CellProfiler-compatible primary-object, tubeness, adaptive Otsu,
    medial-axis, and seed-relative skeleton measurements provide the opinionated
    engine; disconnected traces are omitted from the rendered ownership mask.

    Args:
        neurite_channel_index: Zero-based channel containing neurite outgrowth.
        illumination: Fluorescence or transmission contrast model used for
            foreground detection.
        cell_body: Optional channel plus width, area, and local-background
            thresholds for cell bodies.
        outgrowth: Width, intensity, and significant-growth thresholds for
            attached neurites.
        use_nuclear_stain: Whether to segment a nuclear channel and return nuclei.
        nuclear_stain: Nuclear channel and size/intensity thresholds used when
            nuclear staining is enabled.

    Returns:
        The unchanged image, image- and cell-level measurement rows, and
        channel-aligned cell-body, thin outgrowth, unified-neuron, and nuclear
        masks, followed by a rooted spatial morphology forest. The unified
        layer assigns each body and its owned outgrowth the same integer
        identity, while the graph preserves branch geometry and metrics for
        direct table, ROI-path, and SWC inspection.
    """

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(
            f"Expected a 2D channel stack with shape (C, Y, X), got "
            f"shape {image_array.shape}"
        )
    if not 0 <= neurite_channel_index < image_array.shape[0]:
        raise ValueError("neurite_channel_index is outside the input stack")

    illumination = NeuriteIllumination(illumination)
    cell_body.validate()
    outgrowth.validate()
    pixel_size_um = float(pixel_size)
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size must be a finite value > 0")

    body_channel_index = (
        neurite_channel_index
        if cell_body.channel_index is None
        else int(cell_body.channel_index)
    )
    if not 0 <= body_channel_index < image_array.shape[0]:
        raise ValueError("cell_body.channel_index is outside the input stack")

    nuclei_labels = np.zeros(image_array.shape[1:], dtype=np.int32)
    if use_nuclear_stain:
        nuclear_stain.validate("nuclear_stain")
        if not 0 <= nuclear_stain.channel_index < image_array.shape[0]:
            raise ValueError("nuclear_stain.channel_index is outside the input stack")
        if nuclear_stain.channel_index == neurite_channel_index:
            raise ValueError(
                "nuclear_stain.channel_index must differ from neurite_channel_index"
            )
        nuclei_labels = segment_metaxpress_round_objects(
            image_array[nuclear_stain.channel_index],
            nuclear_stain,
            pixel_size_um,
        )

    bright_objects = illumination == NeuriteIllumination.FLUORESCENCE
    nuclear_seeded_signal_body_mode = (
        use_nuclear_stain and body_channel_index == neurite_channel_index
    )
    body_detection_channel_index = (
        int(nuclear_stain.channel_index)
        if nuclear_seeded_signal_body_mode
        else body_channel_index
    )
    body_image = image_array[body_detection_channel_index]
    if nuclear_seeded_signal_body_mode:
        seed_payload_template = _identify_cell_bodies_cellprofiler(
            body_image,
            cell_body,
            pixel_size_um,
            bright_objects=bright_objects,
        )
        cell_body_payload = object_label_value_with_dense_labels(
            seed_payload_template,
            nuclei_labels,
            domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
        )
    else:
        cell_body_payload = _identify_cell_bodies_cellprofiler(
            body_image,
            cell_body,
            pixel_size_um,
            bright_objects=bright_objects,
            nuclei_labels=nuclei_labels if use_nuclear_stain else None,
        )
    cell_body_labels = object_label_dense_array(
        cell_body_payload,
        dtype=np.int32,
    )

    neurite_image = image_array[neurite_channel_index]
    outgrowth_binary, outgrowth_skeleton, outgrowth_response = (
        _identify_neurites_cellprofiler(
            neurite_image,
            cell_body,
            outgrowth,
            pixel_size_um,
            bright_objects=bright_objects,
        )
    )
    outgrowth_width_px = outgrowth.maximum_width / pixel_size_um
    topology = _analyze_topology(
        outgrowth_skeleton,
        cell_body_labels,
        pixel_size_um,
        outgrowth_width_px,
    )
    owner_skeleton = _render_owned_skeleton(
        outgrowth_skeleton.shape,
        topology,
    )
    owner_outgrowth = _expand_skeleton_ownership(
        owner_skeleton,
        outgrowth_binary,
        outgrowth_width_px,
    )
    unified_neuron_labels = _identify_unified_neurons_cellprofiler(
        neurite_image,
        cell_body_payload,
        body_width_px=cell_body.approximate_max_width / pixel_size_um,
        bright_objects=bright_objects,
    )
    nuclear_seed_mode = use_nuclear_stain and body_detection_channel_index == int(
        nuclear_stain.channel_index
    )
    if nuclear_seed_mode:
        owner_skeleton = _adopt_secondary_owned_skeleton(
            outgrowth_skeleton,
            owner_skeleton,
            unified_neuron_labels,
        )
        cell_body_payload = _qualify_nuclear_cell_bodies(
            cell_body_payload,
            cell_body_labels,
            unified_neuron_labels,
        )
        cell_body_labels = object_label_dense_array(
            cell_body_payload,
            dtype=np.int32,
        )
        topology = _analyze_topology(
            outgrowth_skeleton,
            cell_body_labels,
            pixel_size_um,
            outgrowth_width_px,
        )
        owner_skeleton = _render_owned_skeleton(
            outgrowth_skeleton.shape,
            topology,
        )
        unified_neuron_labels = _identify_unified_neurons_cellprofiler(
            neurite_image,
            cell_body_payload,
            body_width_px=cell_body.approximate_max_width / pixel_size_um,
            bright_objects=bright_objects,
        )
        owner_skeleton = _adopt_secondary_owned_skeleton(
            outgrowth_skeleton,
            owner_skeleton,
            unified_neuron_labels,
        )
        if nuclear_seeded_signal_body_mode:
            signal_cell_bodies = _derive_signal_cell_bodies(
                cell_body_labels,
                unified_neuron_labels,
                neurite_image,
                cell_body,
                pixel_size_um,
                bright_objects=bright_objects,
            )
            keep_signal_body = (
                np.bincount(
                    signal_cell_bodies.ravel(),
                    minlength=int(cell_body_labels.max()) + 1,
                )
                > 0
            )
            keep_signal_body[0] = False
            cell_body_labels = _relabel(signal_cell_bodies, keep_signal_body)
            owner_skeleton = _relabel(owner_skeleton, keep_signal_body)
            unified_neuron_labels = _relabel(
                unified_neuron_labels,
                keep_signal_body,
            )
            cell_body_payload = object_label_value_with_dense_labels(
                cell_body_payload,
                cell_body_labels,
                domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
            )
            unified_neuron_labels = _identify_unified_neurons_cellprofiler(
                neurite_image,
                cell_body_payload,
                body_width_px=cell_body.approximate_max_width / pixel_size_um,
                bright_objects=bright_objects,
            )
            owner_skeleton = _adopt_secondary_owned_skeleton(
                outgrowth_skeleton,
                owner_skeleton,
                unified_neuron_labels,
            )
    owner_skeleton = _repair_signal_supported_skeleton(
        owner_skeleton,
        outgrowth_response,
        unified_neuron_labels,
        cell_body_labels,
        minimum_response=outgrowth.intensity_above_local_background,
    )
    neurite_skeleton = owner_skeleton.copy()
    neurite_skeleton[cell_body_labels > 0] = 0
    topology = _analyze_topology(
        neurite_skeleton > 0,
        cell_body_labels,
        pixel_size_um,
        outgrowth_width_px,
        assigned_path_labels=neurite_skeleton,
    )
    neurite_skeleton = _render_owned_skeleton(
        neurite_skeleton.shape,
        topology,
    )
    soma_attached_skeleton = _prune_soma_detached_skeleton(
        neurite_skeleton,
        cell_body_labels,
        attachment_distance=max(1.0, outgrowth_width_px / 2.0 + 0.5),
    )
    if not np.array_equal(soma_attached_skeleton, neurite_skeleton):
        neurite_skeleton = soma_attached_skeleton
        topology = _analyze_topology(
            neurite_skeleton > 0,
            cell_body_labels,
            pixel_size_um,
            outgrowth_width_px,
            assigned_path_labels=neurite_skeleton,
        )
        neurite_skeleton = _render_owned_skeleton(
            neurite_skeleton.shape,
            topology,
        )
    _, cp_measurement_rows = _raw_processing_leaf(measure_object_skeleton)(
        neurite_skeleton,
        seed_labels=cell_body_payload,
        fill_small_holes=True,
        maximum_hole_size=10,
    )
    cp_measurements = {
        int(row["object_label"]): row for row in cp_measurement_rows.row_mappings()
    }
    keep_measured_skeleton = np.zeros(int(cell_body_labels.max()) + 1, dtype=bool)
    for owner, measurements in cp_measurements.items():
        keep_measured_skeleton[owner] = measurements["total_skeleton_length"] > 0
    measured_neurite_skeleton = np.where(
        keep_measured_skeleton[neurite_skeleton],
        neurite_skeleton,
        0,
    ).astype(np.int32, copy=False)
    if not np.array_equal(measured_neurite_skeleton, neurite_skeleton):
        neurite_skeleton = measured_neurite_skeleton
        topology = _analyze_topology(
            neurite_skeleton > 0,
            cell_body_labels,
            pixel_size_um,
            outgrowth_width_px,
            assigned_path_labels=neurite_skeleton,
        )
        neurite_skeleton = _render_owned_skeleton(
            neurite_skeleton.shape,
            topology,
        )
    owner_outgrowth = _expand_skeleton_ownership(
        neurite_skeleton,
        outgrowth_binary,
        outgrowth_width_px,
    )
    owner_outgrowth[cell_body_labels > 0] = 0

    cell_results = _build_cell_results(
        cell_body_labels,
        owner_outgrowth,
        neurite_image,
        topology,
        outgrowth.minimum_cell_growth_to_log_as_significant,
        pixel_size_um,
        slice_index=body_channel_index,
        cp_measurements=cp_measurements,
    )
    summary = _build_summary(
        cell_results,
        neurite_channel_index=neurite_channel_index,
        cell_body_channel_index=body_channel_index,
        nuclear_channel_index=(
            nuclear_stain.channel_index if use_nuclear_stain else -1
        ),
        resolved_crossovers=len(topology.crossing_nodes),
        mean_outgrowth_average_intensity=(
            float(np.mean(neurite_image[owner_outgrowth > 0]))
            if np.any(owner_outgrowth > 0)
            else 0.0
        ),
    )

    cell_body_stack = np.zeros(image_array.shape, dtype=np.int32)
    cell_body_stack[body_channel_index] = cell_body_labels
    neurite_stack = np.zeros(image_array.shape, dtype=np.int32)
    neurite_stack[neurite_channel_index] = neurite_skeleton
    unified_neuron_stack = np.zeros(image_array.shape, dtype=np.int32)
    unified_neuron_stack[neurite_channel_index] = unified_neuron_labels
    nuclei_stack = np.zeros(image_array.shape, dtype=np.int32)
    if use_nuclear_stain:
        nuclei_stack[nuclear_stain.channel_index] = nuclei_labels
    neurite_morphology = _build_neurite_morphology_graph(
        topology,
        cell_body_labels,
        pixel_size_um=pixel_size_um,
        outgrowth_width_px=outgrowth_width_px,
        assigned_path_labels=neurite_skeleton,
    )
    neurite_morphology = neurite_morphology.replace_fields(
        source_plane_index=neurite_channel_index
    )

    return (
        image,
        DataclassMeasurementColumnarRows(
            (summary,),
            row_type=NeuriteOutgrowthSummary,
        ),
        DataclassMeasurementColumnarRows(
            tuple(cell_results),
            row_type=NeuriteOutgrowthCellResult,
        ),
        cell_body_stack,
        neurite_stack,
        unified_neuron_stack,
        nuclei_stack,
        neurite_morphology,
    )


def _cellprofiler_foreground_image(
    image: np.ndarray,
    *,
    bright_objects: bool,
) -> np.ndarray:
    """Present both illumination modes as bright foreground to CP leaves."""

    image_array = np.asarray(image)
    if bright_objects:
        return image_array
    return np.max(image_array) - image_array


def _cellprofiler_adaptive_window(
    reference_width_px: float,
    image_shape: Sequence[int],
) -> int:
    """Choose a stable CP threshold neighborhood from the declared body scale."""

    minimum_window = max(16, int(np.ceil(2.0 * reference_width_px)))
    scale_window = 1 << (minimum_window - 1).bit_length()
    maximum_window = max(1, min(int(size) for size in image_shape[:2]) // 2)
    return min(scale_window, maximum_window)


def _identify_cell_bodies_cellprofiler(
    image: np.ndarray,
    settings: MetaXpressCellBodySettings,
    pixel_size_um: float,
    *,
    bright_objects: bool,
    nuclei_labels: np.ndarray | None = None,
):
    """Detect with CP IPO, then apply the MetaXpress-owned body predicates."""

    maximum_width_px = settings.approximate_max_width / pixel_size_um
    minimum_area_px = settings.minimum_area / pixel_size_um**2
    _, _, detected_payload = _raw_processing_leaf(identify_primary_objects)(
        _cellprofiler_foreground_image(image, bright_objects=bright_objects),
        **CELLPROFILER_NEURITE_ENGINE_PROFILE.compact_body_detection_kwargs(
            adaptive_window_size=_cellprofiler_adaptive_window(
                maximum_width_px,
                image.shape,
            ),
        ),
    )
    detected_labels = object_label_dense_array(detected_payload, dtype=np.int32)
    response = local_background_response(
        image,
        object_width_px=maximum_width_px,
        bright_objects=bright_objects,
    )
    nuclear_supported = _nuclear_supported_body_candidates(
        detected_labels,
        response,
        nuclei_labels,
        maximum_width_px=maximum_width_px,
        intensity_threshold=settings.intensity_above_local_background,
    )
    minimum_equivalent_diameter_px = 2.0 * np.sqrt(minimum_area_px / np.pi)
    keep = np.zeros(int(detected_labels.max()) + 1, dtype=bool)
    for region in regionprops(detected_labels):
        region_mask = detected_labels == region.label
        region_response = response[region_mask]
        maximum_inscribed_diameter_px = (
            2.0 * float(np.max(ndi.distance_transform_edt(region_mask))) - 1.0
        )
        if (
            region.area >= minimum_area_px
            and maximum_inscribed_diameter_px >= minimum_equivalent_diameter_px
            and region.axis_minor_length <= maximum_width_px
            and region_response.size
            and float(np.mean(region_response))
            >= settings.intensity_above_local_background
            and (nuclear_supported is None or nuclear_supported[region.label])
        ):
            keep[region.label] = True
    filtered_labels = _relabel(detected_labels, keep)
    return object_label_value_with_dense_labels(
        detected_payload,
        filtered_labels,
        domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
    )


def _nuclear_supported_body_candidates(
    detected_labels: np.ndarray,
    response: np.ndarray,
    nuclei_labels: np.ndarray | None,
    *,
    maximum_width_px: float,
    intensity_threshold: float,
) -> np.ndarray | None:
    """Map valid nuclear seeds onto nearby, locally width-bounded CP bodies."""

    if nuclei_labels is None:
        return None
    supported = np.zeros(int(detected_labels.max()) + 1, dtype=bool)
    if not np.any(detected_labels):
        return supported

    distance, nearest = ndi.distance_transform_edt(
        detected_labels == 0,
        return_indices=True,
    )
    foreground_distance = ndi.distance_transform_edt(
        response >= intensity_threshold,
    )
    maximum_seed_distance = maximum_width_px / 2.0
    for nucleus in regionprops(nuclei_labels):
        centroid = tuple(
            int(np.clip(round(value), 0, detected_labels.shape[axis] - 1))
            for axis, value in enumerate(nucleus.centroid)
        )
        if distance[centroid] > maximum_seed_distance:
            continue
        foreground_radius = float(foreground_distance[centroid])
        if foreground_radius <= 0:
            continue
        local_foreground_width = 2.0 * foreground_radius - 1.0
        if local_foreground_width > maximum_width_px:
            continue
        nearest_position = tuple(indices[centroid] for indices in nearest)
        candidate = int(detected_labels[nearest_position])
        if candidate > 0:
            supported[candidate] = True
    return supported


def _identify_neurites_cellprofiler(
    image: np.ndarray,
    cell_body: MetaXpressCellBodySettings,
    settings: MetaXpressOutgrowthSettings,
    pixel_size_um: float,
    *,
    bright_objects: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the public mask, CP medial axis, and local signal evidence."""

    outgrowth_width_px = settings.maximum_width / pixel_size_um
    body_width_px = cell_body.approximate_max_width / pixel_size_um
    cp_image = _cellprofiler_foreground_image(
        image,
        bright_objects=bright_objects,
    )
    enhanced = _raw_processing_leaf(enhance_or_suppress_features)(
        cp_image,
        **CELLPROFILER_NEURITE_ENGINE_PROFILE.enhancement_kwargs(
            smoothing_value=max(0.5, 0.375 * outgrowth_width_px),
        ),
    )
    cp_mask_payload, _ = _raw_processing_leaf(threshold)(
        enhanced,
        **CELLPROFILER_NEURITE_ENGINE_PROFILE.threshold_kwargs(
            window_size=_cellprofiler_adaptive_window(body_width_px, image.shape),
            smoothing=max(0.0, 0.25 * outgrowth_width_px),
        ),
    )
    cp_mask = np.asarray(image_payload_data(cp_mask_payload)) > 0
    response = local_background_response(
        image,
        object_width_px=outgrowth_width_px,
        bright_objects=bright_objects,
    )
    outgrowth_mask = cp_mask & (response >= settings.intensity_above_local_background)
    skeleton_payload = _raw_processing_leaf(medialaxis)(
        outgrowth_mask.astype(np.float32, copy=False)
    )
    skeleton = np.asarray(image_payload_data(skeleton_payload)) > 0
    return outgrowth_mask, skeleton, response


def _identify_unified_neurons_cellprofiler(
    image: np.ndarray,
    cell_body_payload,
    *,
    body_width_px: float,
    bright_objects: bool,
) -> np.ndarray:
    """Return the same seed-propagated final labels as the modular CP workflow."""

    *_, unified_payload = _raw_processing_leaf(identify_secondary_objects)(
        _cellprofiler_foreground_image(image, bright_objects=bright_objects),
        primary_labels=cell_body_payload,
        **CELLPROFILER_NEURITE_ENGINE_PROFILE.secondary_kwargs(
            adaptive_window_size=_cellprofiler_adaptive_window(
                body_width_px,
                image.shape,
            ),
        ),
    )
    return object_label_dense_array(unified_payload, dtype=np.int32)


def _adopt_secondary_owned_skeleton(
    skeleton: np.ndarray,
    owner_skeleton: np.ndarray,
    unified_neuron_labels: np.ndarray,
) -> np.ndarray:
    """Adopt a whole skeleton component when CP gives it one neuron identity."""

    adopted = np.asarray(owner_skeleton, dtype=np.int32).copy()
    components, component_count = ndi.label(
        np.asarray(skeleton, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
    )
    for component in range(1, component_count + 1):
        component_mask = components == component
        owners = np.unique(
            np.concatenate(
                (
                    adopted[component_mask],
                    unified_neuron_labels[component_mask],
                )
            )
        )
        owners = owners[owners > 0]
        if owners.size == 1:
            adopted[component_mask] = int(owners[0])
    return adopted


def _qualify_nuclear_cell_bodies(
    cell_body_payload,
    cell_body_labels: np.ndarray,
    unified_neuron_labels: np.ndarray,
):
    """Keep DAPI candidates that expand into declared neuronal cytoplasm."""

    label_count = int(cell_body_labels.max())
    body_areas = np.bincount(
        cell_body_labels.ravel(),
        minlength=label_count + 1,
    )
    secondary_areas = np.bincount(
        unified_neuron_labels.ravel(),
        minlength=label_count + 1,
    )
    keep = np.zeros(label_count + 1, dtype=bool)
    keep[1:] = secondary_areas[1:] > body_areas[1:]
    return object_label_value_with_dense_labels(
        cell_body_payload,
        _relabel(cell_body_labels, keep),
        domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
    )


def _derive_signal_cell_bodies(
    nuclear_seed_labels: np.ndarray,
    unified_neuron_labels: np.ndarray,
    neurite_image: np.ndarray,
    settings: MetaXpressCellBodySettings,
    pixel_size_um: float,
    *,
    bright_objects: bool,
) -> np.ndarray:
    """Fill bounded neuronal-cytoplasm bodies around qualified nuclear seeds."""

    seeds = np.asarray(nuclear_seed_labels, dtype=np.int32)
    unified = np.asarray(unified_neuron_labels, dtype=np.int32)
    if seeds.shape != unified.shape or seeds.shape != neurite_image.shape:
        raise ValueError(
            "nuclear seeds, unified neurons, and neurite image must share a shape"
        )
    maximum_radius_px = settings.approximate_max_width / (2.0 * pixel_size_um)
    minimum_area_px = settings.minimum_area / pixel_size_um**2
    response = local_background_response(
        neurite_image,
        object_width_px=settings.approximate_max_width / pixel_size_um,
        bright_objects=bright_objects,
    )
    body_foreground = response >= settings.intensity_above_local_background
    foreground_distance = ndi.distance_transform_edt(body_foreground)
    bodies = np.zeros(seeds.shape, dtype=np.int32)
    connectivity = np.ones((3, 3), dtype=bool)
    for owner in range(1, int(seeds.max()) + 1):
        seed = seeds == owner
        if not np.any(seed):
            continue
        seed_coordinates = np.argwhere(seed)
        seed_centroid = tuple(
            int(np.clip(round(value), 0, seeds.shape[axis] - 1))
            for axis, value in enumerate(seed_coordinates.mean(axis=0))
        )
        local_foreground_width = 2.0 * float(foreground_distance[seed_centroid]) - 1.0
        if local_foreground_width > settings.approximate_max_width / pixel_size_um:
            continue
        distance_from_seed = ndi.distance_transform_edt(~seed)
        candidate = (
            (unified == owner)
            & (distance_from_seed <= maximum_radius_px)
            & body_foreground
        )
        components, component_count = ndi.label(candidate, structure=connectivity)
        if component_count == 0:
            continue
        component = min(
            range(1, component_count + 1),
            key=lambda value: (
                float(np.min(distance_from_seed[components == value])),
                value,
            ),
        )
        body = ndi.binary_fill_holes(components == component)
        if np.count_nonzero(body) < minimum_area_px:
            continue
        bodies[body] = owner
    return bodies


def _repair_signal_supported_skeleton(
    labels: np.ndarray,
    signal_response: np.ndarray,
    owner_regions: np.ndarray,
    cell_body_labels: np.ndarray,
    *,
    minimum_response: float,
) -> np.ndarray:
    """Connect owned fragments only through continuous same-owner image evidence.

    Each owner starts at a point inside its cell body. A deterministic
    multi-source least-cost search may traverse that body, already accepted
    skeleton pixels, or pixels that both exceed the declared neurite-response
    threshold and belong to the owner's propagated region. Unsupported or
    foreign-owner fragments are removed instead of receiving inferred chords.
    """

    repaired = np.asarray(labels, dtype=np.int32).copy()
    response = np.asarray(signal_response, dtype=float)
    regions = np.asarray(owner_regions, dtype=np.int32)
    bodies = np.asarray(cell_body_labels, dtype=np.int32)
    if not repaired.shape == response.shape == regions.shape == bodies.shape:
        raise ValueError(
            "labels, signal_response, owner_regions, and cell_body_labels must "
            "have the same shape"
        )
    if not np.isfinite(minimum_response) or minimum_response < 0:
        raise ValueError("minimum_response must be finite and >= 0")

    connectivity = np.ones((3, 3), dtype=bool)
    owners = sorted(int(value) for value in np.unique(repaired) if value > 0)
    for owner in owners:
        body_mask = bodies == owner
        original_owner = (repaired == owner) & ~body_mask
        repaired[(repaired == owner) & body_mask] = 0
        if not np.any(original_owner):
            continue
        soma_coordinate = tuple(
            int(round(value)) for value in _in_body_soma_coordinate(bodies, owner)
        )
        repaired[soma_coordinate] = owner
        occupied_by_other_owner = (repaired > 0) & (repaired != owner)
        signal_support = (response >= minimum_response) & (regions == owner)
        allowed = (
            signal_support | original_owner | body_mask
        ) & ~occupied_by_other_owner

        while True:
            components, _ = ndi.label(
                repaired == owner,
                structure=connectivity,
            )
            root_component = int(components[soma_coordinate])
            connected = components == root_component
            targets = original_owner & ~connected
            if not np.any(targets):
                break
            path = _least_cost_supported_path(
                allowed,
                response,
                connected,
                targets,
                preferred=(original_owner | body_mask),
                minimum_response=minimum_response,
            )
            if path is None:
                break
            repaired[tuple(path.T)] = owner

        components, _ = ndi.label(
            repaired == owner,
            structure=connectivity,
        )
        root_component = int(components[soma_coordinate])
        repaired[(repaired == owner) & (components != root_component)] = 0
    return repaired


def _prune_soma_detached_skeleton(
    labels: np.ndarray,
    cell_body_labels: np.ndarray,
    *,
    attachment_distance: float,
) -> np.ndarray:
    """Keep only external skeleton components that reach their owning soma."""

    skeleton = np.asarray(labels, dtype=np.int32).copy()
    bodies = np.asarray(cell_body_labels, dtype=np.int32)
    if skeleton.shape != bodies.shape:
        raise ValueError("labels and cell_body_labels must have the same shape")
    if not np.isfinite(attachment_distance) or attachment_distance < 0:
        raise ValueError("attachment_distance must be finite and >= 0")

    connectivity = np.ones((3, 3), dtype=bool)
    for owner in sorted(int(value) for value in np.unique(skeleton) if value > 0):
        soma_distance = ndi.distance_transform_edt(bodies != owner)
        components, component_count = ndi.label(
            skeleton == owner,
            structure=connectivity,
        )
        for component in range(1, component_count + 1):
            component_mask = components == component
            if float(np.min(soma_distance[component_mask])) > attachment_distance:
                skeleton[component_mask] = 0
    return skeleton


def _least_cost_supported_path(
    allowed: np.ndarray,
    signal_response: np.ndarray,
    starts: np.ndarray,
    targets: np.ndarray,
    *,
    preferred: np.ndarray,
    minimum_response: float,
) -> np.ndarray | None:
    """Return one deterministic 8-connected path over accepted support pixels."""

    shape = allowed.shape
    distances = np.full(shape, np.inf, dtype=float)
    predecessor_rows = np.full(shape, -1, dtype=np.int32)
    predecessor_columns = np.full(shape, -1, dtype=np.int32)
    pixel_cost = np.full(shape, np.inf, dtype=float)
    if minimum_response > 0:
        supported_response = np.maximum(signal_response, minimum_response)
        pixel_cost[allowed] = 1.0 + (minimum_response / supported_response[allowed])
    else:
        pixel_cost[allowed] = 1.0
    pixel_cost[preferred & allowed] = 0.5

    queue: list[tuple[float, int, int]] = []
    for row, column in np.argwhere(starts & allowed):
        row_index = int(row)
        column_index = int(column)
        distances[row_index, column_index] = 0.0
        heapq.heappush(queue, (0.0, row_index, column_index))

    neighbor_offsets = (
        (-1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2.0)),
    )
    destination: tuple[int, int] | None = None
    while queue:
        distance, row, column = heapq.heappop(queue)
        if distance != distances[row, column]:
            continue
        if targets[row, column]:
            destination = (row, column)
            break
        for row_offset, column_offset, step_length in neighbor_offsets:
            next_row = row + row_offset
            next_column = column + column_offset
            if not (0 <= next_row < shape[0] and 0 <= next_column < shape[1]):
                continue
            if not allowed[next_row, next_column]:
                continue
            step_cost = (
                0.5
                * (pixel_cost[row, column] + pixel_cost[next_row, next_column])
                * step_length
            )
            candidate = distance + step_cost
            if candidate >= distances[next_row, next_column]:
                continue
            distances[next_row, next_column] = candidate
            predecessor_rows[next_row, next_column] = row
            predecessor_columns[next_row, next_column] = column
            heapq.heappush(queue, (candidate, next_row, next_column))

    if destination is None:
        return None
    path = [destination]
    while not starts[path[-1]]:
        row, column = path[-1]
        predecessor = (
            int(predecessor_rows[row, column]),
            int(predecessor_columns[row, column]),
        )
        if predecessor[0] < 0:
            raise RuntimeError("Signal-supported path has no predecessor to a start")
        path.append(predecessor)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _relabel(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    mapping = np.zeros(len(keep), dtype=np.int32)
    kept_labels = np.flatnonzero(keep)
    mapping[kept_labels] = np.arange(1, len(kept_labels) + 1, dtype=np.int32)
    return mapping[labels]


def _analyze_topology(
    skeleton: np.ndarray,
    cell_body_labels: np.ndarray,
    pixel_size_um: float,
    outgrowth_width_px: float,
    *,
    assigned_path_labels: np.ndarray | None = None,
) -> _TopologyResult:
    if (
        assigned_path_labels is not None
        and assigned_path_labels.shape != skeleton.shape
    ):
        raise ValueError(
            "assigned_path_labels must have the same shape as the skeleton"
        )
    if not skeleton.any():
        return _empty_topology()

    skeleton_graph = Skeleton(skeleton, spacing=pixel_size_um)
    branch_table = summarize(skeleton_graph, separator="_").reset_index(drop=True)
    path_count = skeleton_graph.n_paths
    if path_count == 0:
        return _empty_topology()

    path_coordinates = tuple(
        np.asarray(skeleton_graph.path_coordinates(path_index), dtype=int)
        for path_index in range(path_count)
    )
    path_lengths = branch_table["branch_distance"].to_numpy(dtype=float)
    path_euclidean_lengths = branch_table["euclidean_distance"].to_numpy(dtype=float)
    path_branch_types = branch_table["branch_type"].to_numpy(dtype=np.int32)
    node_incidence: dict[int, list[int]] = defaultdict(list)
    node_endpoints: dict[int, list[tuple[int, int]]] = defaultdict(list)
    node_is_source: dict[tuple[int, int], bool] = {}
    node_coordinates: dict[int, np.ndarray] = {}
    for path_index, row in branch_table.iterrows():
        source = int(row["node_id_src"])
        destination = int(row["node_id_dst"])
        node_incidence[source].append(path_index)
        node_incidence[destination].append(path_index)
        node_endpoints[source].append((path_index, 0))
        node_endpoints[destination].append((path_index, 1))
        node_is_source[(source, path_index)] = True
        node_is_source[(destination, path_index)] = False
        node_coordinates[source] = np.array(
            [row["image_coord_src_0"], row["image_coord_src_1"]], dtype=float
        )
        node_coordinates[destination] = np.array(
            [row["image_coord_dst_0"], row["image_coord_dst_1"]], dtype=float
        )

    transitions: dict[int, set[int]] = {
        path_index: set() for path_index in range(path_count)
    }
    branch_nodes: set[int] = set()
    crossing_nodes: set[int] = set()
    path_endpoint_groups = [[0, 0] for _ in range(path_count)]
    endpoint_group_coordinates: dict[int, tuple[float, float]] = {}
    next_endpoint_group = 1
    lookahead = max(2, int(np.ceil(outgrowth_width_px)))
    for node in sorted(node_incidence):
        incident_paths = node_incidence[node]
        unique_paths = sorted(set(incident_paths))
        crossing_pairs = _crossing_pairs(
            unique_paths,
            path_coordinates,
            node_is_source,
            node,
            lookahead,
        )
        if crossing_pairs is not None:
            crossing_nodes.add(node)
            endpoint_groups = tuple(
                tuple(
                    endpoint for endpoint in node_endpoints[node] if endpoint[0] in pair
                )
                for pair in crossing_pairs
            )
        else:
            if len(unique_paths) >= 3:
                branch_nodes.add(node)
            endpoint_groups = (tuple(node_endpoints[node]),)

        for endpoint_group in endpoint_groups:
            group_id = next_endpoint_group
            next_endpoint_group += 1
            group_paths = sorted({path_index for path_index, _ in endpoint_group})
            for path_index, endpoint_index in endpoint_group:
                path_endpoint_groups[path_index][endpoint_index] = group_id
            coordinate_path, coordinate_endpoint = min(endpoint_group)
            coordinate = path_coordinates[coordinate_path][
                0 if coordinate_endpoint == 0 else -1
            ]
            endpoint_group_coordinates[group_id] = tuple(
                float(value) for value in coordinate
            )
            for first, second in combinations(group_paths, 2):
                transitions[first].add(second)
                transitions[second].add(first)

    expanded_bodies = expand_labels(
        cell_body_labels,
        distance=max(1, int(np.ceil(outgrowth_width_px)) + 2),
    )
    root_labels_by_path: dict[int, tuple[int, ...]] = {}
    for path_index, coordinates in enumerate(path_coordinates):
        labels = expanded_bodies[tuple(coordinates.T)]
        labels = labels[labels > 0]
        if labels.size:
            counts = Counter(int(label) for label in labels)
            root_labels_by_path[path_index] = tuple(
                label for label, _ in counts.most_common()
            )

    path_owners, path_distances = _propagate_path_owners(
        path_lengths,
        transitions,
        root_labels_by_path,
    )
    if assigned_path_labels is not None:
        for path_index, coordinates in enumerate(path_coordinates):
            labels = assigned_path_labels[tuple(coordinates.T)]
            labels = labels[labels > 0]
            if labels.size:
                path_owners[path_index] = Counter(
                    int(label) for label in labels
                ).most_common(1)[0][0]
    roots_by_cell: dict[int, list[int]] = defaultdict(list)
    for path_index, labels in root_labels_by_path.items():
        owner = int(path_owners[path_index])
        if owner > 0 and owner in labels:
            roots_by_cell[owner].append(path_index)

    branch_owner: dict[int, int] = {}
    for node in branch_nodes:
        owned_paths = [
            int(path_owners[path_index])
            for path_index in node_incidence[node]
            if path_owners[path_index] > 0
        ]
        if owned_paths:
            branch_owner[node] = Counter(owned_paths).most_common(1)[0][0]
    branch_owner = _merge_nearby_owned_nodes(
        branch_owner,
        node_coordinates,
        radius=max(1.0, outgrowth_width_px),
    )

    used_crossings = {
        node
        for node in crossing_nodes
        if any(path_owners[path_index] > 0 for path_index in node_incidence[node])
    }
    return _TopologyResult(
        path_owners=path_owners,
        path_distances=path_distances,
        path_lengths=path_lengths,
        path_euclidean_lengths=path_euclidean_lengths,
        path_coordinates=path_coordinates,
        path_endpoint_groups=tuple(
            tuple(endpoint_groups) for endpoint_groups in path_endpoint_groups
        ),
        path_branch_types=path_branch_types,
        endpoint_group_coordinates=endpoint_group_coordinates,
        transitions={key: tuple(sorted(value)) for key, value in transitions.items()},
        root_paths_by_cell={
            cell: tuple(sorted(set(paths))) for cell, paths in roots_by_cell.items()
        },
        branch_owner=branch_owner,
        crossing_nodes=frozenset(used_crossings),
    )


def _merge_nearby_owned_nodes(
    node_owner: Mapping[int, int],
    node_coordinates: Mapping[int, np.ndarray],
    *,
    radius: float,
) -> dict[int, int]:
    """Collapse multi-pixel junction neighborhoods into one branch event."""

    remaining = set(node_owner)
    merged: dict[int, int] = {}
    while remaining:
        seed = min(remaining)
        owner = node_owner[seed]
        cluster = {seed}
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            current = frontier.pop()
            nearby = {
                candidate
                for candidate in remaining
                if node_owner[candidate] == owner
                and np.linalg.norm(
                    node_coordinates[current] - node_coordinates[candidate]
                )
                <= radius
            }
            remaining.difference_update(nearby)
            cluster.update(nearby)
            frontier.extend(nearby)
        merged[min(cluster)] = owner
    return merged


def _empty_topology() -> _TopologyResult:
    return _TopologyResult(
        path_owners=np.zeros(0, dtype=np.int32),
        path_distances=np.zeros(0, dtype=float),
        path_lengths=np.zeros(0, dtype=float),
        path_euclidean_lengths=np.zeros(0, dtype=float),
        path_coordinates=(),
        path_endpoint_groups=(),
        path_branch_types=np.zeros(0, dtype=np.int32),
        endpoint_group_coordinates={},
        transitions={},
        root_paths_by_cell={},
        branch_owner={},
        crossing_nodes=frozenset(),
    )


def _crossing_pairs(
    incident_paths: Sequence[int],
    path_coordinates: Sequence[np.ndarray],
    node_is_source: Mapping[tuple[int, int], bool],
    node: int,
    lookahead: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(incident_paths) != 4:
        return None

    directions = []
    for path_index in incident_paths:
        coordinates = path_coordinates[path_index]
        if len(coordinates) < 2:
            return None
        source_side = node_is_source[(node, path_index)]
        node_coordinate = coordinates[0] if source_side else coordinates[-1]
        sample_index = min(lookahead, len(coordinates) - 1)
        sample_coordinate = (
            coordinates[sample_index] if source_side else coordinates[-sample_index - 1]
        )
        vector = sample_coordinate.astype(float) - node_coordinate.astype(float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        directions.append(vector / norm)

    pairings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    scored_pairings = []
    for pairing in pairings:
        opposite_scores = tuple(
            -float(np.dot(directions[first], directions[second]))
            for first, second in pairing
        )
        scored_pairings.append((min(opposite_scores), sum(opposite_scores), pairing))

    minimum_score, _, best_pairing = max(scored_pairings)
    if minimum_score < np.cos(np.deg2rad(30.0)):
        return None
    return tuple(
        (incident_paths[first], incident_paths[second])
        for first, second in best_pairing
    )  # type: ignore[return-value]


def _propagate_path_owners(
    path_lengths: np.ndarray,
    transitions: Mapping[int, Iterable[int]],
    root_labels_by_path: Mapping[int, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    path_owners = np.zeros(len(path_lengths), dtype=np.int32)
    distances = np.full(len(path_lengths), np.inf, dtype=float)
    queue: list[tuple[float, int, int]] = []
    for path_index, labels in root_labels_by_path.items():
        for label in labels:
            heapq.heappush(queue, (0.0, int(label), path_index))

    while queue:
        distance, owner, path_index = heapq.heappop(queue)
        if distance > distances[path_index]:
            continue
        if distance == distances[path_index] and path_owners[path_index] <= owner:
            continue
        distances[path_index] = distance
        path_owners[path_index] = owner
        for neighbor in transitions[path_index]:
            neighbor_distance = distance + 0.5 * (
                path_lengths[path_index] + path_lengths[neighbor]
            )
            if neighbor_distance <= distances[neighbor]:
                heapq.heappush(queue, (neighbor_distance, owner, neighbor))
    return path_owners, distances


def _build_neurite_morphology_graph(
    topology: _TopologyResult,
    cell_body_labels: np.ndarray,
    *,
    pixel_size_um: float,
    outgrowth_width_px: float,
    assigned_path_labels: np.ndarray | None = None,
) -> SpatialGraph:
    """Project owned Skan paths into deterministic soma-rooted forests."""

    if assigned_path_labels is not None and assigned_path_labels.shape != (
        cell_body_labels.shape
    ):
        raise ValueError(
            "assigned_path_labels must have the same shape as cell_body_labels"
        )
    paths_by_owner: dict[int, list[int]] = defaultdict(list)
    for path_index, coordinates in enumerate(topology.path_coordinates):
        owner = int(topology.path_owners[path_index])
        if assigned_path_labels is not None:
            labels = assigned_path_labels[tuple(coordinates.T)]
            labels = labels[labels > 0]
            owner = (
                Counter(int(label) for label in labels).most_common(1)[0][0]
                if labels.size
                else 0
            )
        if owner > 0:
            paths_by_owner[owner].append(path_index)

    graph_nodes: list[SpatialGraphNode] = []
    graph_edges: list[SpatialGraphEdge] = []
    next_node_id = 1
    next_edge_id = 1
    process_radius_um = max(
        pixel_size_um / 2.0,
        outgrowth_width_px * pixel_size_um / 2.0,
    )

    for owner in sorted(paths_by_owner):
        owner_paths = tuple(sorted(paths_by_owner[owner]))
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for path_index in owner_paths:
            first_group, second_group = topology.path_endpoint_groups[path_index]
            adjacency[first_group].append((path_index, second_group))
            adjacency[second_group].append((path_index, first_group))
        for incident_paths in adjacency.values():
            incident_paths.sort()

        body_area_um2 = float(np.count_nonzero(cell_body_labels == owner)) * (
            pixel_size_um**2
        )
        soma_distance = ndi.distance_transform_edt(cell_body_labels != owner)
        soma_coordinate = _in_body_soma_coordinate(cell_body_labels, owner)
        soma_radius_um = max(
            process_radius_um,
            float(np.sqrt(body_area_um2 / np.pi)),
        )
        primary_root_group = min(
            adjacency,
            key=lambda group_id: (
                sum(
                    (
                        topology.endpoint_group_coordinates[group_id][axis]
                        - soma_coordinate[axis]
                    )
                    ** 2
                    for axis in range(2)
                ),
                group_id,
            ),
        )
        remaining_groups = set(adjacency)

        while remaining_groups:
            component_seed = min(remaining_groups)
            component_groups: set[int] = set()
            component_paths: set[int] = set()
            frontier = [component_seed]
            while frontier:
                group_id = frontier.pop()
                if group_id in component_groups:
                    continue
                component_groups.add(group_id)
                for path_index, neighbor_group in adjacency[group_id]:
                    component_paths.add(path_index)
                    if neighbor_group not in component_groups:
                        frontier.append(neighbor_group)
            remaining_groups.difference_update(component_groups)

            root_group = min(
                component_groups,
                key=lambda group_id: (
                    sum(
                        (
                            topology.endpoint_group_coordinates[group_id][axis]
                            - soma_coordinate[axis]
                        )
                        ** 2
                        for axis in range(2)
                    ),
                    group_id,
                ),
            )
            is_soma_component = root_group == primary_root_group
            root_coordinate = topology.endpoint_group_coordinates[root_group]
            root_index = tuple(
                int(np.clip(round(value), 0, cell_body_labels.shape[axis] - 1))
                for axis, value in enumerate(root_coordinate)
            )
            root_inside_soma = cell_body_labels[root_index] == owner
            soma_attachment_distance = max(
                1.0,
                outgrowth_width_px / 2.0 + 0.5,
            )
            root_touches_soma = any(
                float(
                    np.min(
                        soma_distance[tuple(topology.path_coordinates[path_index].T)]
                    )
                )
                <= soma_attachment_distance
                for path_index in component_paths
            )
            soma_connected = root_inside_soma or root_touches_soma
            if root_inside_soma and is_soma_component:
                root_role = "soma_root"
            elif root_touches_soma:
                root_role = "soma_attachment_root"
            else:
                root_role = "disconnected_root"
            root_node = SpatialGraphNode.from_features(
                node_id=next_node_id,
                coordinates=root_coordinate,
                radius=(
                    soma_radius_um if root_role == "soma_root" else process_radius_um
                ),
                features={
                    "label": owner,
                    "neuron_label": owner,
                    "node_role": root_role,
                },
            )
            next_node_id += 1
            graph_nodes.append(root_node)
            nodes_by_group: dict[int, SpatialGraphNode] = {root_group: root_node}
            for group_id in sorted(component_groups):
                if group_id == root_group:
                    continue
                node = SpatialGraphNode.from_features(
                    node_id=next_node_id,
                    coordinates=topology.endpoint_group_coordinates[group_id],
                    radius=process_radius_um,
                    features={
                        "label": owner,
                        "neuron_label": owner,
                        "node_role": "neurite",
                    },
                )
                next_node_id += 1
                graph_nodes.append(node)
                nodes_by_group[group_id] = node

            node_distances = {root_group: 0.0}
            visited_groups = {root_group}
            emitted_paths: set[int] = set()
            candidate_edges: list[tuple[float, int, int, int]] = []

            def enqueue_from(group_id: int) -> None:
                for path_index, neighbor_group in adjacency[group_id]:
                    if path_index not in component_paths:
                        continue
                    candidate_distance = node_distances[group_id] + float(
                        topology.path_lengths[path_index]
                    )
                    heapq.heappush(
                        candidate_edges,
                        (
                            candidate_distance,
                            path_index,
                            group_id,
                            neighbor_group,
                        ),
                    )

            enqueue_from(root_group)
            while candidate_edges:
                (
                    target_distance,
                    path_index,
                    source_group,
                    target_group,
                ) = heapq.heappop(candidate_edges)
                if path_index in emitted_paths:
                    continue

                endpoint_groups = topology.path_endpoint_groups[path_index]
                coordinates = topology.path_coordinates[path_index]
                if endpoint_groups != (source_group, target_group):
                    coordinates = coordinates[::-1]
                target_is_cycle_break = target_group in visited_groups
                if target_is_cycle_break:
                    target_node = SpatialGraphNode.from_features(
                        node_id=next_node_id,
                        coordinates=coordinates[-1],
                        radius=process_radius_um,
                        features={
                            "label": owner,
                            "neuron_label": owner,
                            "node_role": "cycle_break",
                        },
                    )
                    next_node_id += 1
                    graph_nodes.append(target_node)
                else:
                    visited_groups.add(target_group)
                    node_distances[target_group] = target_distance
                    target_node = nodes_by_group[target_group]
                branch_distance_um = float(topology.path_lengths[path_index])
                euclidean_distance_um = float(
                    topology.path_euclidean_lengths[path_index]
                )
                graph_edges.append(
                    SpatialGraphEdge.from_features(
                        edge_id=next_edge_id,
                        source=nodes_by_group[source_group],
                        target=target_node,
                        coordinates=coordinates,
                        features={
                            "label": owner,
                            "neuron_label": owner,
                            "branch_distance_um": branch_distance_um,
                            "euclidean_distance_um": euclidean_distance_um,
                            "tortuosity": (
                                max(
                                    1.0,
                                    branch_distance_um / euclidean_distance_um,
                                )
                                if euclidean_distance_um > 0
                                else 0.0
                            ),
                            "distance_from_soma_um": (
                                node_distances[source_group]
                                if soma_connected
                                else float("nan")
                            ),
                            "branch_type": int(topology.path_branch_types[path_index]),
                        },
                    )
                )
                next_edge_id += 1
                emitted_paths.add(path_index)
                if not target_is_cycle_break:
                    enqueue_from(target_group)

    graph = SpatialGraph(
        name=NEURITE_MORPHOLOGY_OUTPUT.name,
        nodes=tuple(graph_nodes),
        edges=tuple(graph_edges),
        coordinate_spacing=(pixel_size_um, pixel_size_um),
    )
    graph.require_directed_forest()
    return graph


def _in_body_soma_coordinate(
    cell_body_labels: np.ndarray,
    owner: int,
) -> tuple[float, float]:
    owner_coordinates = np.argwhere(cell_body_labels == owner)
    if not len(owner_coordinates):
        raise ValueError(
            f"Neurite topology owner {owner} has no corresponding cell body."
        )
    centroid = owner_coordinates.mean(axis=0)
    soma_index = int(np.argmin(np.sum((owner_coordinates - centroid) ** 2, axis=1)))
    return tuple(float(value) for value in owner_coordinates[soma_index])


def _render_owned_skeleton(
    shape: tuple[int, int],
    topology: _TopologyResult,
) -> np.ndarray:
    owner_skeleton = np.zeros(shape, dtype=np.int32)
    for path_index, coordinates in enumerate(topology.path_coordinates):
        owner = int(topology.path_owners[path_index])
        if owner > 0:
            empty = owner_skeleton[tuple(coordinates.T)] == 0
            owner_skeleton[tuple(coordinates[empty].T)] = owner
    return owner_skeleton


def _expand_skeleton_ownership(
    owner_skeleton: np.ndarray,
    outgrowth_binary: np.ndarray,
    outgrowth_width_px: float,
) -> np.ndarray:
    if not owner_skeleton.any():
        return np.zeros(owner_skeleton.shape, dtype=np.int32)
    distance, nearest = ndi.distance_transform_edt(
        owner_skeleton == 0,
        return_indices=True,
    )
    nearest_owner = owner_skeleton[tuple(nearest)]
    radius = max(1.0, outgrowth_width_px / 2.0 + 0.5)
    return np.where(
        outgrowth_binary & (distance <= radius),
        nearest_owner,
        0,
    ).astype(np.int32, copy=False)


def _build_cell_results(
    cell_body_labels: np.ndarray,
    owner_outgrowth: np.ndarray,
    neurite_image: np.ndarray,
    topology: _TopologyResult,
    significant_growth_threshold_um: float,
    pixel_size_um: float,
    *,
    slice_index: int,
    cp_measurements: Mapping[int, Mapping[str, object]],
) -> list[NeuriteOutgrowthCellResult]:
    cell_count = int(cell_body_labels.max())
    body_areas = np.bincount(cell_body_labels.ravel(), minlength=cell_count + 1).astype(
        float
    )
    body_areas *= pixel_size_um**2
    results = []

    for cell in range(1, cell_count + 1):
        cp_measurement = cp_measurements[cell]
        path_indexes = np.flatnonzero(topology.path_owners == cell)
        total_outgrowth = float(cp_measurement["total_skeleton_length"]) * pixel_size_um
        roots = topology.root_paths_by_cell.get(cell, ())
        geometric_process_lengths = _measure_process_lengths(cell, roots, topology)
        geometric_total = float(np.sum(geometric_process_lengths))
        if geometric_total > 0 and total_outgrowth > 0:
            process_lengths = [
                length * total_outgrowth / geometric_total
                for length in geometric_process_lengths
            ]
        else:
            process_lengths = []
        process_count = int(cp_measurement["number_trunks"])
        curve_length = float(np.sum(topology.path_lengths[path_indexes]))
        euclidean_length = float(np.sum(topology.path_euclidean_lengths[path_indexes]))
        straightness = euclidean_length / curve_length if curve_length else 0.0
        outgrowth_pixels = owner_outgrowth == cell
        mean_intensity = (
            float(np.mean(neurite_image[outgrowth_pixels]))
            if outgrowth_pixels.any()
            else 0.0
        )
        results.append(
            NeuriteOutgrowthCellResult(
                slice_index=slice_index,
                cell=cell,
                total_outgrowth_um=total_outgrowth,
                processes=process_count,
                mean_process_length_um=(
                    total_outgrowth / process_count if process_count else 0.0
                ),
                median_process_length_um=(
                    float(np.median(process_lengths)) if process_lengths else 0.0
                ),
                max_process_length_um=(
                    float(np.max(process_lengths)) if process_lengths else 0.0
                ),
                branches=int(cp_measurement["number_non_trunk_branches"]),
                straightness=straightness,
                cell_body_area_um2=float(body_areas[cell]),
                mean_outgrowth_intensity=mean_intensity,
                significant_growth=(total_outgrowth > significant_growth_threshold_um),
            )
        )
    return results


def _measure_process_lengths(
    cell: int,
    roots: Sequence[int],
    topology: _TopologyResult,
) -> list[float]:
    if not roots:
        return []
    root_owner = np.full(len(topology.path_lengths), -1, dtype=int)
    distances = np.full(len(topology.path_lengths), np.inf, dtype=float)
    queue: list[tuple[float, int, int]] = []
    for root_number, path_index in enumerate(sorted(set(roots))):
        heapq.heappush(queue, (0.0, root_number, path_index))

    while queue:
        distance, process, path_index = heapq.heappop(queue)
        if topology.path_owners[path_index] != cell:
            continue
        if distance > distances[path_index]:
            continue
        if distance == distances[path_index] and root_owner[path_index] <= process:
            continue
        distances[path_index] = distance
        root_owner[path_index] = process
        for neighbor in topology.transitions[path_index]:
            if topology.path_owners[neighbor] != cell:
                continue
            next_distance = distance + 0.5 * (
                topology.path_lengths[path_index] + topology.path_lengths[neighbor]
            )
            if next_distance <= distances[neighbor]:
                heapq.heappush(queue, (next_distance, process, neighbor))

    process_lengths = np.zeros(len(set(roots)), dtype=float)
    for path_index, process in enumerate(root_owner):
        if process >= 0:
            process_lengths[process] += topology.path_lengths[path_index]
    return [float(length) for length in process_lengths if length > 0]


def _build_summary(
    cell_results: Sequence[NeuriteOutgrowthCellResult],
    *,
    neurite_channel_index: int,
    cell_body_channel_index: int,
    nuclear_channel_index: int,
    resolved_crossovers: int,
    mean_outgrowth_average_intensity: float,
) -> NeuriteOutgrowthSummary:
    cell_count = len(cell_results)
    total_outgrowth = float(sum(row.total_outgrowth_um for row in cell_results))
    total_processes = int(sum(row.processes for row in cell_results))
    total_branches = int(sum(row.branches for row in cell_results))
    total_body_area = float(sum(row.cell_body_area_um2 for row in cell_results))
    significant_count = sum(row.significant_growth for row in cell_results)
    return NeuriteOutgrowthSummary(
        neurite_channel_index=neurite_channel_index,
        cell_body_channel_index=cell_body_channel_index,
        nuclear_channel_index=nuclear_channel_index,
        number_of_cells=cell_count,
        total_outgrowth_um=total_outgrowth,
        mean_outgrowth_per_cell_um=(
            total_outgrowth / cell_count if cell_count else 0.0
        ),
        total_processes=total_processes,
        mean_processes_per_cell=(total_processes / cell_count if cell_count else 0.0),
        total_branches=total_branches,
        mean_branches_per_cell=(total_branches / cell_count if cell_count else 0.0),
        total_cell_body_area_um2=total_body_area,
        mean_cell_body_area_um2=(total_body_area / cell_count if cell_count else 0.0),
        straightness=(
            float(np.mean([row.straightness for row in cell_results]))
            if cell_results
            else 0.0
        ),
        cells_significant_growth=int(significant_count),
        percent_cells_significant_growth=(
            100.0 * significant_count / cell_count if cell_count else 0.0
        ),
        mean_outgrowth_average_intensity=mean_outgrowth_average_intensity,
        resolved_crossovers=resolved_crossovers,
    )
