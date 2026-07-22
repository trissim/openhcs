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
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
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


@dataclass(frozen=True)
class _TopologyResult:
    path_owners: np.ndarray
    path_distances: np.ndarray
    path_lengths: np.ndarray
    path_euclidean_lengths: np.ndarray
    path_coordinates: tuple[np.ndarray, ...]
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
        masks. The unified layer assigns each body and its owned outgrowth the
        same integer identity so the final biological result is directly
        inspectable rather than inferred across independent layers.
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
    body_image = image_array[body_channel_index]
    cell_body_payload = _identify_cell_bodies_cellprofiler(
        body_image,
        cell_body,
        pixel_size_um,
        bright_objects=bright_objects,
    )
    cell_body_labels = object_label_dense_array(
        cell_body_payload,
        dtype=np.int32,
    )

    neurite_image = image_array[neurite_channel_index]
    outgrowth_binary, outgrowth_skeleton = _identify_neurites_cellprofiler(
        neurite_image,
        cell_body,
        outgrowth,
        pixel_size_um,
        bright_objects=bright_objects,
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

    _, cp_measurement_rows = _raw_processing_leaf(measure_object_skeleton)(
        outgrowth_skeleton,
        seed_labels=cell_body_payload,
        fill_small_holes=True,
        maximum_hole_size=10,
    )
    cp_measurements = {
        int(row["object_label"]): row for row in cp_measurement_rows.row_mappings()
    }

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
    neurite_stack[neurite_channel_index] = owner_skeleton
    unified_neuron_stack = np.zeros(image_array.shape, dtype=np.int32)
    unified_neuron_stack[neurite_channel_index] = np.where(
        cell_body_labels > 0,
        cell_body_labels,
        owner_outgrowth,
    )
    nuclei_stack = np.zeros(image_array.shape, dtype=np.int32)
    if use_nuclear_stain:
        nuclei_stack[nuclear_stain.channel_index] = nuclei_labels

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
):
    """Detect with CP IPO, then apply the MetaXpress-owned body predicates."""

    maximum_width_px = settings.approximate_max_width / pixel_size_um
    minimum_area_px = settings.minimum_area / pixel_size_um**2
    _, _, detected_payload = _raw_processing_leaf(identify_primary_objects)(
        _cellprofiler_foreground_image(image, bright_objects=bright_objects),
        exclude_size=False,
        exclude_border_objects=False,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        adaptive_window_size=_cellprofiler_adaptive_window(
            maximum_width_px,
            image.shape,
        ),
    )
    detected_labels = object_label_dense_array(detected_payload, dtype=np.int32)
    response = local_background_response(
        image,
        object_width_px=maximum_width_px,
        bright_objects=bright_objects,
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
        ):
            keep[region.label] = True
    filtered_labels = _relabel(detected_labels, keep)
    return detected_payload.with_replacement_labels(filtered_labels)


def _identify_neurites_cellprofiler(
    image: np.ndarray,
    cell_body: MetaXpressCellBodySettings,
    settings: MetaXpressOutgrowthSettings,
    pixel_size_um: float,
    *,
    bright_objects: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the public intensity mask and its CP medial-axis skeleton."""

    outgrowth_width_px = settings.maximum_width / pixel_size_um
    body_width_px = cell_body.approximate_max_width / pixel_size_um
    cp_image = _cellprofiler_foreground_image(
        image,
        bright_objects=bright_objects,
    )
    enhanced = _raw_processing_leaf(enhance_or_suppress_features)(
        cp_image,
        method=OperationMethod.ENHANCE,
        enhance_method=EnhanceMethod.NEURITES,
        neurite_method=NeuriteMethod.TUBENESS,
        smoothing_value=max(0.5, 0.375 * outgrowth_width_px),
        neurite_rescale=True,
    )
    cp_mask_payload, _ = _raw_processing_leaf(threshold)(
        enhanced,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        threshold_correction_factor=0.85,
        window_size=_cellprofiler_adaptive_window(body_width_px, image.shape),
        smoothing=max(0.0, 0.25 * outgrowth_width_px),
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
    return outgrowth_mask, skeleton


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
) -> _TopologyResult:
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
    node_incidence: dict[int, list[int]] = defaultdict(list)
    node_is_source: dict[tuple[int, int], bool] = {}
    node_coordinates: dict[int, np.ndarray] = {}
    for path_index, row in branch_table.iterrows():
        source = int(row["node_id_src"])
        destination = int(row["node_id_dst"])
        node_incidence[source].append(path_index)
        node_incidence[destination].append(path_index)
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
    lookahead = max(2, int(np.ceil(outgrowth_width_px)))
    for node, incident_paths in node_incidence.items():
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
            pairs: Iterable[tuple[int, int]] = crossing_pairs
        else:
            if len(unique_paths) >= 3:
                branch_nodes.add(node)
            pairs = combinations(unique_paths, 2)
        for first, second in pairs:
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
