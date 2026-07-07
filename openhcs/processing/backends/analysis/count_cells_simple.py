"""
Simple cell counting using thresholding and connected component labeling.

This module provides a compact NumPy/scikit-image cell counter intended for
basic bright- or dark-object workflows. The implementation deliberately keeps
the segmentation model simple: threshold each 2D slice, label connected
components, optionally split oversized components, then apply geometric filters
to the final candidate objects.
"""

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import MaterializationSpec, CsvOptions, ROIOptions

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Tuple, List, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import regionprops
from skimage.segmentation import watershed


class ThresholdMethod(str, Enum):
    """Thresholding methods for cell detection."""
    OTSU = "otsu"
    LI = "li"
    YEN = "yen"
    PERCENTILE = "percentile"
    MANUAL = "manual"


class Foreground(str, Enum):
    """Foreground type for thresholding."""
    BRIGHT = "bright"
    DARK = "dark"


class SimpleColocalizationMethod(str, Enum):
    """Colocalization matching methods for simple dual-channel counting."""
    OVERLAP = "overlap"
    CENTROID_DISTANCE = "centroid_distance"


@dataclass(frozen=True)
class SimpleCellCountResult:
    """Canonical result schema for simple single-channel counting."""
    slice_index: int
    cell_count: int

    @classmethod
    def csv_fields(cls) -> List[str]:
        """Return CSV field order from the dataclass declaration."""
        return [field.name for field in fields(cls)]


@dataclass(frozen=True)
class DualChannelCountResult:
    """Canonical result schema for simple dual-channel colocalization."""
    channel_1_index: int
    channel_2_index: int
    channel_1_count: int
    channel_2_count: int
    colocalized_count: int
    channel_1_only_count: int
    channel_2_only_count: int
    channel_1_colocalized_percent: float
    channel_2_colocalized_percent: float
    colocalization_method: str
    mean_colocalization_distance: float
    mean_overlap_fraction: float

    @classmethod
    def csv_fields(cls) -> List[str]:
        """Return CSV field order from the dataclass declaration."""
        return [field.name for field in fields(cls)]


@dataclass(frozen=True)
class ColocalizationCandidate:
    """Internal one-to-one colocalization match candidate."""
    channel_1_label: int
    channel_2_label: int
    distance: float
    overlap_pixels: int
    overlap_fraction: float


# Make the Enums importable/stable for multiprocessing/ZMQ pickling
import openhcs.processing.backends.analysis.count_cells_simple as _count_cells_simple

ThresholdMethod.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "ThresholdMethod", ThresholdMethod)

Foreground.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "Foreground", Foreground)

SimpleColocalizationMethod.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "SimpleColocalizationMethod", SimpleColocalizationMethod)

SimpleCellCountResult.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "SimpleCellCountResult", SimpleCellCountResult)

DualChannelCountResult.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "DualChannelCountResult", DualChannelCountResult)

ColocalizationCandidate.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "ColocalizationCandidate", ColocalizationCandidate)


@numpy
@special_outputs(
    (
        "cell_counts",
        MaterializationSpec(CsvOptions(fields=SimpleCellCountResult.csv_fields()))
    ),
    ("segmentation_masks", MaterializationSpec(ROIOptions()))
)
def count_cells_simple(
    image,
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold: float = 0.5,
    threshold_percentile: float = 99.0,
    foreground: Foreground = Foreground.BRIGHT,
    min_size: int = 20,
    max_size: int = 100000,
    max_eccentricity: float = 1.0,
    watershed_large_objects: bool = False,
    watershed_min_size: Optional[int] = None,
    watershed_max_size: Optional[int] = None,
    watershed_min_distance: int = 5,
    watershed_footprint_size: int = 3
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """
    Count thresholded objects in a 3D image stack with optional shape cleanup.

    The function processes each 2D plane of ``image`` independently. For every
    slice it computes a threshold, builds a binary foreground mask, labels
    connected components, optionally applies distance-transform watershed only
    to components larger than ``max_size``, and then applies the final object
    acceptance filters. The reported count and ROI mask therefore describe the
    final accepted connected components after all splitting and filtering.

    Filtering order is intentional:

    1. ``min_size`` is used only in the final acceptance pass.
    2. If ``watershed_large_objects`` is enabled, raw connected components
       whose area is greater than ``watershed_min_size`` are eligible for
       splitting. If ``watershed_min_size`` is ``None``, ``max_size`` is used
       as the split trigger for backward compatibility. When
       ``watershed_max_size`` is set, only components with
       ``watershed_min_size < area <= watershed_max_size`` are split.
    3. ``min_size``, ``max_size``, and ``max_eccentricity`` are then applied to
       the resulting candidate regions.

    This means a large merged component can be split into multiple accepted
    cells, as long as each watershed fragment lands within the size and shape
    limits. If watershed cannot find at least two seeds for a large object, that
    object remains a single candidate and is usually rejected by ``max_size``.

    Args:
        image: Input 3D image stack with shape ``(Z, Y, X)``. OpenHCS also uses
            this shape for logical single-plane data, so a single 2D image
            should be supplied as ``image[None, :, :]``. The input image is
            returned unchanged as the primary output.
        threshold_method: Thresholding strategy used to create the foreground
            mask for each slice. ``OTSU``, ``LI``, and ``YEN`` compute an
            automatic threshold from the slice. ``PERCENTILE`` uses
            ``threshold_percentile``. ``MANUAL`` uses ``threshold`` directly,
            with fractional values in ``[0, 1]`` interpreted as a fraction of
            the slice maximum when the image intensity range is larger than
            ``[0, 1]``.
        threshold: Manual threshold value used only when
            ``threshold_method=ThresholdMethod.MANUAL``. Values greater than
            ``1`` are treated as native image-intensity values. Values in
            ``[0, 1]`` are treated as normalized fractions for scaled integer or
            high-dynamic-range arrays.
        threshold_percentile: Percentile in ``[0, 100]`` used only when
            ``threshold_method=ThresholdMethod.PERCENTILE``. For bright
            foregrounds, pixels above this percentile become foreground; for
            dark foregrounds, pixels below it become foreground.
        foreground: Polarity of the objects to count. ``BRIGHT`` counts pixels
            above the computed threshold. ``DARK`` counts pixels below the
            computed threshold.
        min_size: Minimum accepted object area in pixels after optional
            watershed splitting. Objects smaller than this are removed from the
            output mask and not counted.
        max_size: Maximum accepted object area in pixels after optional
            watershed splitting. Objects larger than this are treated as
            artifacts or unresolved merged objects and are removed from the
            output mask.
        max_eccentricity: Maximum accepted object eccentricity after optional
            watershed splitting and size filtering. Eccentricity is the
            scikit-image region shape measure where ``0.0`` is circular and
            ``1.0`` is nearly line-like. The default ``1.0`` preserves the
            previous behavior and effectively disables shape filtering. Lower
            values reject elongated debris, scratches, neurites, or merged
            streak-like objects that otherwise satisfy the size filter.
        watershed_large_objects: If ``True``, apply distance-transform watershed
            to oversized connected components. This is useful when touching
            cells merge into one component that would otherwise be rejected by
            the size filter. It is disabled by default to preserve the original
            simple connected-component behavior.
        watershed_min_size: Optional lower area threshold for watershed
            attempts. If ``None``, the function uses ``max_size`` as the split
            trigger, matching the original behavior. Set this lower than
            ``max_size`` to force more aggressive declumping without also
            lowering the final accepted object size. For example,
            ``watershed_min_size=100`` and ``max_size=300`` means "try to split
            raw components above 100 px, then keep final fragments up to
            300 px."
        watershed_max_size: Optional upper area limit for watershed attempts.
            If ``None``, every component larger than the split trigger is
            eligible for watershed. If set, only components with area greater
            than the split trigger and less than or equal to
            ``watershed_max_size`` are split. Components above this cap are
            left unsplit and then rejected by the final ``max_size`` filter,
            which is useful for preventing huge debris, plate edges, bubbles,
            or illumination artifacts from being over-segmented into plausible
            cell-sized fragments.
        watershed_min_distance: Minimum pixel spacing between local maxima used
            as watershed seeds for oversized objects. Larger values produce
            fewer, more conservative splits; smaller values can split dense or
            noisy objects more aggressively.
        watershed_footprint_size: Side length, in pixels, of the square
            footprint used when finding local maxima for watershed seeds.
            Larger footprints suppress nearby competing maxima and reduce
            over-splitting; smaller footprints allow more seed points.

    Returns:
        Tuple:
          - ``image`` unchanged, preserving OpenHCS primary-output semantics.
          - A list of per-slice dictionaries with ``slice_index`` and
            ``cell_count`` fields. The count is the number of accepted labels
            after optional splitting and all filters.
          - A list of labeled ``int32`` segmentation masks, one per input
            slice. Background is ``0``. Accepted objects are relabeled
            sequentially from ``1`` within each slice.

    Raises:
        ValueError: If size limits, eccentricity limits, or watershed seed
            parameters are outside their supported ranges.
    """
    _validate_simple_segmentation_params(
        min_size=min_size,
        max_size=max_size,
        max_eccentricity=max_eccentricity,
        watershed_min_size=watershed_min_size,
        watershed_max_size=watershed_max_size,
        watershed_min_distance=watershed_min_distance,
        watershed_footprint_size=watershed_footprint_size,
    )

    results = []
    masks = []

    for i, slice_data in enumerate(image):
        labeled_filtered = _segment_simple_slice(
            slice_data,
            threshold_method=threshold_method,
            threshold=threshold,
            threshold_percentile=threshold_percentile,
            foreground=foreground,
            min_size=min_size,
            max_size=max_size,
            max_eccentricity=max_eccentricity,
            watershed_large_objects=watershed_large_objects,
            watershed_min_size=watershed_min_size,
            watershed_max_size=watershed_max_size,
            watershed_min_distance=watershed_min_distance,
            watershed_footprint_size=watershed_footprint_size,
        )
        final_count = int(labeled_filtered.max())

        results.append(
            asdict(SimpleCellCountResult(slice_index=i, cell_count=int(final_count)))
        )

        masks.append(labeled_filtered.astype(np.int32, copy=False))

    return image, results, masks


@numpy
@special_outputs(
    (
        "dual_channel_counts",
        MaterializationSpec(
            CsvOptions(fields=DualChannelCountResult.csv_fields())
        )
    ),
    ("colocalization_masks", MaterializationSpec(ROIOptions()))
)
def count_cells_simple_dual_channel(
    image,
    channel_1_index: int = 0,
    channel_2_index: int = 1,
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold: float = 0.5,
    threshold_percentile: float = 99.0,
    foreground: Foreground = Foreground.BRIGHT,
    min_size: int = 20,
    max_size: int = 100000,
    max_eccentricity: float = 1.0,
    watershed_large_objects: bool = False,
    watershed_min_size: Optional[int] = None,
    watershed_max_size: Optional[int] = None,
    watershed_min_distance: int = 5,
    watershed_footprint_size: int = 3,
    colocalization_method: SimpleColocalizationMethod = (
        SimpleColocalizationMethod.OVERLAP
    ),
    min_overlap_fraction: float = 0.25,
    max_colocalization_distance: float = 5.0,
    return_channel_masks: bool = False
) -> Tuple[np.ndarray, List[dict], List[np.ndarray]]:
    """
    Count two channels with the simple segmenter and summarize colocalization.

    This is the dual-channel companion to ``count_cells_simple``. It expects a
    3D stack whose first axis contains channels, segments exactly two selected
    planes with the same simple threshold/watershed/filter pipeline, and then
    matches objects across the two resulting label masks. The primary output is
    the input image unchanged; quantitative coloc data and ROI masks are
    emitted through special outputs.

    Segmentation behavior is intentionally identical to ``count_cells_simple``:
    each selected channel is thresholded, connected components are labeled,
    optional watershed is applied only to oversized components, and
    ``min_size``, ``max_size``, and ``max_eccentricity`` are applied to the
    final candidate regions. Use the single-channel function first when tuning
    these segmentation parameters, then reuse them here for coloc.

    Colocalization is one-to-one. A detected object in channel 1 can match at
    most one detected object in channel 2, and vice versa. When multiple
    candidate matches are possible, the function greedily keeps the strongest
    non-conflicting pairs: largest overlap for ``OVERLAP`` mode, shortest
    centroid distance for ``CENTROID_DISTANCE`` mode.

    Args:
        image: Input 3D stack with shape ``(C, Y, X)``. The function treats the
            first axis as channel-like planes and does not infer channel names.
            ``channel_1_index`` and ``channel_2_index`` select the two planes to
            compare. The input image is returned unchanged as the primary
            output.
        channel_1_index: First channel plane to segment and compare.
        channel_2_index: Second channel plane to segment and compare. It must
            differ from ``channel_1_index``.
        threshold_method: Thresholding strategy used independently on each
            selected channel. Automatic methods compute one threshold per
            channel plane; ``MANUAL`` uses the same ``threshold`` value for both
            planes.
        threshold: Manual threshold value used only for ``MANUAL`` thresholding.
            Fractional values in ``[0, 1]`` are interpreted as a fraction of
            each selected channel's maximum when that channel has a larger
            native intensity range.
        threshold_percentile: Percentile used only for ``PERCENTILE``
            thresholding, computed independently per selected channel.
        foreground: Whether objects are brighter or darker than the background.
            The same foreground polarity is used for both channels.
        min_size: Minimum accepted object area in pixels after optional
            watershed splitting.
        max_size: Maximum accepted object area in pixels after optional
            watershed splitting. If ``watershed_large_objects=True``, raw
            components larger than this value are candidates for splitting.
        max_eccentricity: Maximum accepted object eccentricity after watershed
            and size filtering. ``1.0`` disables shape filtering.
        watershed_large_objects: Whether to split oversized raw connected
            components before final filtering.
        watershed_min_size: Optional lower area threshold for watershed
            attempts. If ``None``, ``max_size`` is used as the split trigger.
            Set this lower than ``max_size`` to split more aggressively while
            still accepting larger final fragments.
        watershed_max_size: Optional upper area cap for watershed attempts.
            Components above this cap are not split and are rejected by the
            final ``max_size`` filter.
        watershed_min_distance: Minimum seed spacing for watershed of oversized
            objects.
        watershed_footprint_size: Square footprint side length for watershed
            seed detection.
        colocalization_method: ``OVERLAP`` matches objects whose masks overlap
            enough. ``CENTROID_DISTANCE`` matches objects whose centroids are
            within ``max_colocalization_distance`` pixels.
        min_overlap_fraction: Minimum overlap required in ``OVERLAP`` mode.
            The overlap fraction is ``overlap_pixels / min(area_1, area_2)`` so
            a smaller object mostly contained in a larger object can count as
            colocalized.
        max_colocalization_distance: Maximum centroid distance, in pixels, used
            in ``CENTROID_DISTANCE`` mode.
        return_channel_masks: If ``False``, the ROI special output contains
            only a label mask for colocalized pairs. If ``True``, it contains
            channel 1 labels, channel 2 labels, and the colocalization labels in
            that order.

    Returns:
        Tuple:
          - ``image`` unchanged.
          - A one-element list containing a CSV-friendly summary dictionary.
          - A list of ``int32`` label masks. The colocalization mask labels each
            matched pair with a unique integer and leaves background/unmatched
            objects as ``0``.

    Raises:
        ValueError: If the input is not 3D, channel indices are invalid, or
            segmentation/colocalization parameters are outside supported ranges.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D channel stack, got {image.ndim}D")
    if channel_1_index == channel_2_index:
        raise ValueError("channel_1_index and channel_2_index must be different")
    if not 0 <= channel_1_index < image.shape[0]:
        raise ValueError("channel_1_index is outside the input stack")
    if not 0 <= channel_2_index < image.shape[0]:
        raise ValueError("channel_2_index is outside the input stack")
    if not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must be in [0.0, 1.0]")
    if max_colocalization_distance < 0:
        raise ValueError("max_colocalization_distance must be >= 0")

    _validate_simple_segmentation_params(
        min_size=min_size,
        max_size=max_size,
        max_eccentricity=max_eccentricity,
        watershed_min_size=watershed_min_size,
        watershed_max_size=watershed_max_size,
        watershed_min_distance=watershed_min_distance,
        watershed_footprint_size=watershed_footprint_size,
    )

    channel_1_labels = _segment_simple_slice(
        image[channel_1_index],
        threshold_method=threshold_method,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        foreground=foreground,
        min_size=min_size,
        max_size=max_size,
        max_eccentricity=max_eccentricity,
        watershed_large_objects=watershed_large_objects,
        watershed_min_size=watershed_min_size,
        watershed_max_size=watershed_max_size,
        watershed_min_distance=watershed_min_distance,
        watershed_footprint_size=watershed_footprint_size,
    )
    channel_2_labels = _segment_simple_slice(
        image[channel_2_index],
        threshold_method=threshold_method,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        foreground=foreground,
        min_size=min_size,
        max_size=max_size,
        max_eccentricity=max_eccentricity,
        watershed_large_objects=watershed_large_objects,
        watershed_min_size=watershed_min_size,
        watershed_max_size=watershed_max_size,
        watershed_min_distance=watershed_min_distance,
        watershed_footprint_size=watershed_footprint_size,
    )

    pairs = _match_colocalized_labels(
        channel_1_labels,
        channel_2_labels,
        colocalization_method=colocalization_method,
        min_overlap_fraction=min_overlap_fraction,
        max_colocalization_distance=max_colocalization_distance,
    )
    colocalization_mask = _create_colocalization_label_mask(
        channel_1_labels,
        channel_2_labels,
        pairs,
    )

    channel_1_count = int(channel_1_labels.max())
    channel_2_count = int(channel_2_labels.max())
    colocalized_count = len(pairs)
    mean_distance = (
        float(np.mean([pair.distance for pair in pairs])) if pairs else 0.0
    )
    mean_overlap = (
        float(np.mean([pair.overlap_fraction for pair in pairs]))
        if pairs
        else 0.0
    )

    result = DualChannelCountResult(
        channel_1_index=int(channel_1_index),
        channel_2_index=int(channel_2_index),
        channel_1_count=channel_1_count,
        channel_2_count=channel_2_count,
        colocalized_count=colocalized_count,
        channel_1_only_count=channel_1_count - colocalized_count,
        channel_2_only_count=channel_2_count - colocalized_count,
        channel_1_colocalized_percent=(
            100.0 * colocalized_count / channel_1_count
            if channel_1_count
            else 0.0
        ),
        channel_2_colocalized_percent=(
            100.0 * colocalized_count / channel_2_count
            if channel_2_count
            else 0.0
        ),
        colocalization_method=SimpleColocalizationMethod(colocalization_method).value,
        mean_colocalization_distance=mean_distance,
        mean_overlap_fraction=mean_overlap,
    )

    masks = [colocalization_mask.astype(np.int32, copy=False)]
    if return_channel_masks:
        masks = [
            channel_1_labels.astype(np.int32, copy=False),
            channel_2_labels.astype(np.int32, copy=False),
            *masks,
        ]

    return image, [asdict(result)], masks


def _validate_simple_segmentation_params(
    *,
    min_size: int,
    max_size: int,
    max_eccentricity: float,
    watershed_min_size: Optional[int],
    watershed_max_size: Optional[int],
    watershed_min_distance: int,
    watershed_footprint_size: int,
) -> None:
    """Validate shared simple segmentation parameters."""
    if min_size < 0:
        raise ValueError("min_size must be >= 0")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if not 0.0 <= max_eccentricity <= 1.0:
        raise ValueError("max_eccentricity must be in [0.0, 1.0]")
    if watershed_min_size is not None and watershed_min_size < 1:
        raise ValueError("watershed_min_size must be >= 1 when set")
    split_size = max_size if watershed_min_size is None else watershed_min_size
    if watershed_max_size is not None and watershed_max_size <= split_size:
        raise ValueError(
            "watershed_max_size must be greater than the watershed split threshold when set"
        )
    if watershed_min_distance < 1:
        raise ValueError("watershed_min_distance must be >= 1")
    if watershed_footprint_size < 1:
        raise ValueError("watershed_footprint_size must be >= 1")


def _segment_simple_slice(
    slice_data: np.ndarray,
    *,
    threshold_method: ThresholdMethod,
    threshold: float,
    threshold_percentile: float,
    foreground: Foreground,
    min_size: int,
    max_size: int,
    max_eccentricity: float,
    watershed_large_objects: bool,
    watershed_min_size: Optional[int],
    watershed_max_size: Optional[int],
    watershed_min_distance: int,
    watershed_footprint_size: int,
) -> np.ndarray:
    """Segment one 2D plane with the simple counter's canonical logic."""
    threshold_method = ThresholdMethod(threshold_method)
    foreground = Foreground(foreground)
    threshold_value = _compute_threshold(
        slice_data,
        threshold_method=threshold_method,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
    )

    if foreground == Foreground.BRIGHT:
        binary = slice_data > threshold_value
    elif foreground == Foreground.DARK:
        binary = slice_data < threshold_value
    else:
        raise ValueError(
            f"Unknown foreground: {foreground!r} (expected 'bright' or 'dark')"
        )

    labeled, num_objects = ndi.label(binary)

    if watershed_large_objects and num_objects > 0:
        labeled = _watershed_large_objects(
            labeled,
            split_size=max_size if watershed_min_size is None else watershed_min_size,
            watershed_max_size=watershed_max_size,
            min_distance=watershed_min_distance,
            footprint_size=watershed_footprint_size,
        )

    if labeled.max() == 0:
        return np.zeros_like(labeled, dtype=np.int32)

    if max_eccentricity == 1.0:
        return _filter_labels_by_area(labeled, min_size=min_size, max_size=max_size)

    keep_mask = np.zeros(int(labeled.max()) + 1, dtype=bool)
    for region in regionprops(labeled):
        if (
            min_size <= region.area <= max_size
            and region.eccentricity <= max_eccentricity
        ):
            keep_mask[region.label] = True

    return _relabel_by_keep_mask(labeled, keep_mask)


def _filter_labels_by_area(
    labeled: np.ndarray,
    *,
    min_size: int,
    max_size: int,
) -> np.ndarray:
    """Filter labels by pixel area and relabel accepted objects densely."""
    counts = np.bincount(labeled.ravel())
    keep_mask = (counts >= min_size) & (counts <= max_size)
    return _relabel_by_keep_mask(labeled, keep_mask)


def _relabel_by_keep_mask(labeled: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Apply a boolean label keep mask and produce dense int32 labels."""
    if keep_mask.size:
        keep_mask[0] = False

    kept_count = int(np.count_nonzero(keep_mask))
    if kept_count == 0:
        return np.zeros_like(labeled, dtype=np.int32)

    remap = np.zeros(keep_mask.shape[0], dtype=np.int32)
    remap[keep_mask] = np.arange(1, kept_count + 1, dtype=np.int32)
    return remap[labeled]


def _compute_threshold(
    slice_data: np.ndarray,
    *,
    threshold_method: ThresholdMethod,
    threshold: float,
    threshold_percentile: float,
) -> float:
    """Compute one threshold in the native intensity scale of a 2D plane."""
    if threshold_method == ThresholdMethod.OTSU:
        return float(threshold_otsu(slice_data))
    if threshold_method == ThresholdMethod.LI:
        return float(threshold_li(slice_data))
    if threshold_method == ThresholdMethod.YEN:
        return float(threshold_yen(slice_data))
    if threshold_method == ThresholdMethod.PERCENTILE:
        return float(np.percentile(slice_data, threshold_percentile))
    if threshold_method == ThresholdMethod.MANUAL:
        threshold_value = float(threshold)
        if 0.0 <= threshold_value <= 1.0:
            max_val = float(np.max(slice_data))
            if max_val > 1.0:
                threshold_value *= max_val
        return threshold_value
    raise ValueError(f"Unknown threshold_method: {threshold_method!r}")


def _watershed_large_objects(
    labeled: np.ndarray,
    split_size: int,
    watershed_max_size: Optional[int],
    min_distance: int,
    footprint_size: int,
) -> np.ndarray:
    """Split components above split_size and at or below watershed_max_size."""
    counts = np.bincount(labeled.ravel())
    split_mask = counts > split_size
    if watershed_max_size is not None:
        split_mask &= counts <= watershed_max_size
    if split_mask.size:
        split_mask[0] = False

    split_labels = np.flatnonzero(split_mask)
    if split_labels.size == 0:
        return labeled.astype(np.int32, copy=False)

    output = labeled.astype(np.int32, copy=True)
    object_slices = ndi.find_objects(labeled)
    next_label = int(labeled.max()) + 1
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)

    for label_id in split_labels:
        component_slice = object_slices[label_id - 1]
        if component_slice is None:
            continue

        component = labeled[component_slice] == label_id
        distance = ndi.distance_transform_edt(component)
        seeds = peak_local_max(
            distance,
            min_distance=min_distance,
            footprint=footprint,
            labels=component,
        )

        if len(seeds) <= 1:
            continue

        markers = np.zeros_like(component, dtype=np.int32)
        markers[seeds[:, 0], seeds[:, 1]] = np.arange(1, len(seeds) + 1)
        component_splits = watershed(-distance, markers, mask=component)

        output_view = output[component_slice]
        output_view[component] = 0
        for split_label in range(1, int(component_splits.max()) + 1):
            output_view[component_splits == split_label] = next_label
            next_label += 1

    return output


def _match_colocalized_labels(
    channel_1_labels: np.ndarray,
    channel_2_labels: np.ndarray,
    *,
    colocalization_method: SimpleColocalizationMethod,
    min_overlap_fraction: float,
    max_colocalization_distance: float,
) -> List[ColocalizationCandidate]:
    """Return greedy one-to-one colocalized label pairs."""
    colocalization_method = SimpleColocalizationMethod(colocalization_method)
    regions_1 = {region.label: region for region in regionprops(channel_1_labels)}
    regions_2 = {region.label: region for region in regionprops(channel_2_labels)}
    if not regions_1 or not regions_2:
        return []

    if colocalization_method == SimpleColocalizationMethod.OVERLAP:
        candidates = _overlap_colocalization_candidates(
            channel_1_labels,
            channel_2_labels,
            regions_1,
            regions_2,
            min_overlap_fraction,
        )
        candidates.sort(key=lambda item: item.overlap_fraction, reverse=True)
    elif colocalization_method == SimpleColocalizationMethod.CENTROID_DISTANCE:
        candidates = _distance_colocalization_candidates(
            regions_1,
            regions_2,
            max_colocalization_distance,
        )
        candidates.sort(key=lambda item: item.distance)
    else:
        raise ValueError(f"Unknown colocalization_method: {colocalization_method!r}")

    used_1 = set()
    used_2 = set()
    pairs = []
    for candidate in candidates:
        label_1 = candidate.channel_1_label
        label_2 = candidate.channel_2_label
        if label_1 in used_1 or label_2 in used_2:
            continue
        pairs.append(candidate)
        used_1.add(label_1)
        used_2.add(label_2)

    return pairs


def _overlap_colocalization_candidates(
    channel_1_labels: np.ndarray,
    channel_2_labels: np.ndarray,
    regions_1: dict,
    regions_2: dict,
    min_overlap_fraction: float,
) -> List[ColocalizationCandidate]:
    """Build label-pair candidates from actual mask intersections."""
    overlap_mask = (channel_1_labels > 0) & (channel_2_labels > 0)
    if not np.any(overlap_mask):
        return []

    label_pairs, overlap_pixels = np.unique(
        np.stack(
            [
                channel_1_labels[overlap_mask],
                channel_2_labels[overlap_mask],
            ],
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )

    candidates = []
    for (label_1, label_2), overlap in zip(label_pairs, overlap_pixels):
        region_1 = regions_1[int(label_1)]
        region_2 = regions_2[int(label_2)]
        overlap_fraction = float(overlap / min(region_1.area, region_2.area))
        if overlap_fraction < min_overlap_fraction:
            continue
        candidates.append(
            _colocalization_candidate(
                int(label_1),
                int(label_2),
                region_1,
                region_2,
                overlap_pixels=int(overlap),
                overlap_fraction=overlap_fraction,
            )
        )
    return candidates


def _distance_colocalization_candidates(
    regions_1: dict,
    regions_2: dict,
    max_colocalization_distance: float,
) -> List[ColocalizationCandidate]:
    """Build label-pair candidates from centroid distances."""
    candidates = []
    for label_1, region_1 in regions_1.items():
        for label_2, region_2 in regions_2.items():
            candidate = _colocalization_candidate(
                int(label_1),
                int(label_2),
                region_1,
                region_2,
                overlap_pixels=0,
                overlap_fraction=0.0,
            )
            if candidate.distance <= max_colocalization_distance:
                candidates.append(candidate)
    return candidates


def _colocalization_candidate(
    label_1: int,
    label_2: int,
    region_1,
    region_2,
    *,
    overlap_pixels: int,
    overlap_fraction: float,
) -> ColocalizationCandidate:
    """Create one colocalization candidate."""
    centroid_1 = np.array(region_1.centroid, dtype=float)
    centroid_2 = np.array(region_2.centroid, dtype=float)
    return ColocalizationCandidate(
        channel_1_label=label_1,
        channel_2_label=label_2,
        distance=float(np.linalg.norm(centroid_1 - centroid_2)),
        overlap_pixels=overlap_pixels,
        overlap_fraction=float(overlap_fraction),
    )


def _create_colocalization_label_mask(
    channel_1_labels: np.ndarray,
    channel_2_labels: np.ndarray,
    pairs: List[ColocalizationCandidate],
) -> np.ndarray:
    """Create a label mask where each colocalized pair gets one label."""
    coloc_mask = np.zeros_like(channel_1_labels, dtype=np.int32)
    for coloc_label, pair in enumerate(pairs, start=1):
        label_1 = pair.channel_1_label
        label_2 = pair.channel_2_label
        coloc_mask[channel_1_labels == label_1] = coloc_label
        coloc_mask[channel_2_labels == label_2] = coloc_label
    return coloc_mask
