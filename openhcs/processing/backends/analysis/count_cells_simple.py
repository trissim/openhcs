"""
Simple cell counting using thresholding and connected component labeling.

This module provides a compact NumPy/scikit-image cell counter intended for
basic bright- or dark-object workflows, plus MetaXpress-style W1 cell counting
and W2 stained-area scoring. The implementations share connected-component and
watershed infrastructure while exposing controls appropriate to each workflow.
"""

from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import (
    AlignedROIMask,
    AlignedROIMasks,
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
)

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Tuple, List, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import regionprops
from skimage.segmentation import expand_labels, watershed

from .metaxpress_utils import HiddenPixelSize, local_background_response, odd_size


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


@dataclass(frozen=True)
class SimpleCellSegmentationConfig:
    """Canonical settings for one channel of simple cell segmentation."""

    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    """Thresholding strategy used to create the foreground mask."""

    threshold: float = 0.5
    """Manual threshold used when ``threshold_method`` is ``MANUAL``."""

    threshold_percentile: float = 99.0
    """Percentile used when ``threshold_method`` is ``PERCENTILE``."""

    foreground: Foreground = Foreground.BRIGHT
    """Whether objects are brighter or darker than the background."""

    min_size: int = 20
    """Minimum accepted object area in pixels."""

    max_size: int = 100000
    """Maximum accepted object area in pixels."""

    max_eccentricity: float = 1.0
    """Maximum accepted object eccentricity; ``1.0`` disables this filter."""

    watershed_large_objects: bool = False
    """Whether to split oversized connected components before filtering."""

    watershed_min_size: Optional[int] = None
    """Optional lower area threshold for watershed attempts."""

    watershed_max_size: Optional[int] = None
    """Optional upper area threshold for watershed attempts."""

    watershed_min_distance: int = 5
    """Minimum spacing between watershed seed points."""

    watershed_footprint_size: int = 3
    """Square footprint side length used to find watershed seeds."""

    def validate(self) -> None:
        """Validate the complete settings block for one channel."""

        if self.min_size < 0:
            raise ValueError("min_size must be >= 0")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        if not 0.0 <= self.max_eccentricity <= 1.0:
            raise ValueError("max_eccentricity must be in [0.0, 1.0]")
        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in [0.0, 100.0]")
        if self.watershed_min_size is not None and self.watershed_min_size < 1:
            raise ValueError("watershed_min_size must be >= 1 when set")

        split_size = (
            self.max_size
            if self.watershed_min_size is None
            else self.watershed_min_size
        )
        if (
            self.watershed_max_size is not None
            and self.watershed_max_size <= split_size
        ):
            raise ValueError(
                "watershed_max_size must be greater than the watershed split "
                "threshold when set"
            )
        if self.watershed_min_distance < 1:
            raise ValueError("watershed_min_distance must be >= 1")
        if self.watershed_footprint_size < 1:
            raise ValueError("watershed_footprint_size must be >= 1")


class StainedArea(str, Enum):
    """MetaXpress-style cellular compartment used to score W2 staining."""

    NUCLEUS = "nucleus"
    NUCLEUS_AND_CYTOPLASM = "nucleus and cytoplasm"


@dataclass(frozen=True)
class MetaXpressWavelengthSettings:
    """User-facing MetaXpress-style detection settings for one wavelength."""

    channel_index: int = 0
    """Zero-based index of this wavelength in the channel stack."""

    approx_min_width: float = 5.0
    """Approximate minimum short-axis width in micrometers."""

    approx_max_width: float = 30.0
    """Approximate maximum short-axis width in micrometers."""

    intensity_above_local_background: float = 100.0
    """Minimum raw-intensity difference above adaptive local background."""

    def validate(self, name: str) -> None:
        """Validate one wavelength's complete public settings block."""

        if self.channel_index < 0:
            raise ValueError(f"{name}.channel_index must be >= 0")
        if not np.isfinite(self.approx_min_width) or self.approx_min_width <= 0:
            raise ValueError(f"{name}.approx_min_width must be > 0")
        if (
            not np.isfinite(self.approx_max_width)
            or self.approx_max_width < self.approx_min_width
        ):
            raise ValueError(
                f"{name}.approx_max_width must be >= {name}.approx_min_width"
            )
        if (
            not np.isfinite(self.intensity_above_local_background)
            or self.intensity_above_local_background < 0
        ):
            raise ValueError(f"{name}.intensity_above_local_background must be >= 0")


@dataclass(frozen=True)
class MetaXpressW2Settings(MetaXpressWavelengthSettings):
    """MetaXpress-style W2 settings, including the scored cellular area."""

    channel_index: int = 1
    stained_area: StainedArea = StainedArea.NUCLEUS
    """Score W2 staining in nuclei or in expanded nucleus-plus-cytoplasm areas."""

    def validate(self, name: str = "w2") -> None:
        super().validate(name)
        StainedArea(self.stained_area)


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
    """MetaXpress-style W1 cell count and W2 positive/negative scoring summary."""

    w1_channel_index: int
    w2_channel_index: int
    total_cell_count: int
    w2_positive_cell_count: int
    w2_negative_cell_count: int
    w2_positive_percent: float
    w2_stained_area: str
    minimum_stained_area: float
    all_w2_mean_stained_area: float
    positive_w2_mean_stained_area: float

    @classmethod
    def csv_fields(cls) -> List[str]:
        """Return CSV field order from the dataclass declaration."""
        return [field.name for field in fields(cls)]


# Make the Enums importable/stable for multiprocessing/ZMQ pickling
import openhcs.processing.backends.analysis.count_cells_simple as _count_cells_simple

ThresholdMethod.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "ThresholdMethod", ThresholdMethod)

Foreground.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "Foreground", Foreground)

SimpleCellSegmentationConfig.__module__ = _count_cells_simple.__name__
setattr(
    _count_cells_simple,
    "SimpleCellSegmentationConfig",
    SimpleCellSegmentationConfig,
)

StainedArea.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "StainedArea", StainedArea)

MetaXpressWavelengthSettings.__module__ = _count_cells_simple.__name__
setattr(
    _count_cells_simple,
    "MetaXpressWavelengthSettings",
    MetaXpressWavelengthSettings,
)

MetaXpressW2Settings.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "MetaXpressW2Settings", MetaXpressW2Settings)

SimpleCellCountResult.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "SimpleCellCountResult", SimpleCellCountResult)

DualChannelCountResult.__module__ = _count_cells_simple.__name__
setattr(_count_cells_simple, "DualChannelCountResult", DualChannelCountResult)


@numpy
@special_outputs(
    (
        "cell_counts",
        MaterializationSpec(CsvOptions(fields=SimpleCellCountResult.csv_fields())),
    ),
    ("segmentation_masks", MaterializationSpec(ROIOptions())),
)
def count_cells_simple(
    image,
    segmentation_settings: SimpleCellSegmentationConfig = (
        SimpleCellSegmentationConfig()
    ),
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
        segmentation_settings: Complete threshold, foreground, size, shape, and
            watershed settings for every independently processed image plane.

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
    segmentation_settings.validate()

    results = []
    masks = []

    for i, slice_data in enumerate(image):
        labeled_filtered = _segment_simple_slice(
            slice_data,
            segmentation_settings,
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
        MaterializationSpec(CsvOptions(fields=DualChannelCountResult.csv_fields())),
    ),
    ("colocalization_masks", MaterializationSpec(ROIOptions())),
)
@special_inputs("pixel_size")
def count_cells_simple_dual_channel(
    image,
    w1: MetaXpressWavelengthSettings = MetaXpressWavelengthSettings(
        channel_index=0,
    ),
    w2: MetaXpressW2Settings = MetaXpressW2Settings(
        channel_index=1,
    ),
    minimum_stained_area: float = 10.0,
    pixel_size: HiddenPixelSize = HiddenPixelSize(1.0),
) -> Tuple[np.ndarray, List[dict], AlignedROIMasks]:
    """Count W1 nuclei and score W2-positive cells like MetaXpress MWCS.

    W1 is the required all-nuclei wavelength and therefore defines total cell
    count. W2 is an additional stain scored within each W1 cell's selected
    compartment. A cell is W2-positive when its detected stained area is at
    least ``minimum_stained_area``.

    Widths and stained areas are expressed in micrometers and square
    micrometers. OpenHCS injects the plate pixel size at compilation; direct
    calls use the 1.0 micrometer-per-pixel default. Detection uses adaptive
    local-background subtraction. Object-size filters, background window,
    watershed trigger, seed spacing, and seed footprint are all derived from
    the W1/W2 width settings rather than exposed as separate controls.

    Args:
        image: Input stack with shape ``(C, Y, X)``. The input is returned
            unchanged as the primary output.
        w1: W1 nucleus channel, approximate short-axis width range, and minimum
            intensity above local background.
        w2: W2 stain channel with the same detection controls plus the cellular
            compartment to score: nucleus or nucleus and cytoplasm.
        minimum_stained_area: Minimum W2 stained area in square micrometers for
            a W1 cell to be scored W2-positive.
        pixel_size: Plate pixel size in micrometers per pixel. This is injected
            from microscope metadata and hidden from the function editor.

    Returns:
        Tuple:
          - ``image`` unchanged.
          - A one-element list containing MetaXpress-style positive/negative W2
            scoring and stained-area summary fields.
          - Independently aligned W1 nucleus and W2 stain-object ROI masks. W1
            ROI metadata includes each cell's W2-positive classification and
            stained area.

    Raises:
        ValueError: If the stack, wavelength settings, pixel size, or minimum
            stained area is invalid.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D channel stack, got {image.ndim}D")
    w1.validate("w1")
    w2.validate("w2")
    if w1.channel_index == w2.channel_index:
        raise ValueError("w1.channel_index and w2.channel_index must be different")
    if not 0 <= w1.channel_index < image.shape[0]:
        raise ValueError("w1.channel_index is outside the input stack")
    if not 0 <= w2.channel_index < image.shape[0]:
        raise ValueError("w2.channel_index is outside the input stack")
    if minimum_stained_area < 0:
        raise ValueError("minimum_stained_area must be >= 0")

    pixel_size_um = float(pixel_size)
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size must be a finite value > 0")

    w1_labels = segment_metaxpress_round_objects(
        image[w1.channel_index],
        w1,
        pixel_size_um,
    )
    w2_labels = segment_metaxpress_round_objects(
        image[w2.channel_index],
        w2,
        pixel_size_um,
    )
    compartments = _build_w2_compartments(
        w1_labels,
        w1,
        w2,
        pixel_size_um,
    )
    stained_areas = _measure_stained_area_by_cell(
        compartments,
        w2_labels > 0,
        int(w1_labels.max()),
        pixel_size_um,
    )
    positive_cells = stained_areas >= float(minimum_stained_area)
    if positive_cells.size:
        positive_cells[0] = False

    total_cell_count = int(w1_labels.max())
    positive_count = int(np.count_nonzero(positive_cells))
    negative_count = total_cell_count - positive_count
    cell_areas = stained_areas[1:]
    positive_areas = stained_areas[positive_cells]

    result = DualChannelCountResult(
        w1_channel_index=int(w1.channel_index),
        w2_channel_index=int(w2.channel_index),
        total_cell_count=total_cell_count,
        w2_positive_cell_count=positive_count,
        w2_negative_cell_count=negative_count,
        w2_positive_percent=(
            100.0 * positive_count / total_cell_count if total_cell_count else 0.0
        ),
        w2_stained_area=StainedArea(w2.stained_area).value,
        minimum_stained_area=float(minimum_stained_area),
        all_w2_mean_stained_area=(
            float(np.mean(cell_areas)) if cell_areas.size else 0.0
        ),
        positive_w2_mean_stained_area=(
            float(np.mean(positive_areas)) if positive_areas.size else 0.0
        ),
    )

    w1_label_metadata = {
        label: {
            "w2_positive": bool(positive_cells[label]),
            "w2_stained_area_um2": float(stained_areas[label]),
        }
        for label in range(1, total_cell_count + 1)
    }
    masks = AlignedROIMasks(
        (
            AlignedROIMask(
                mask=w1_labels.astype(np.int32, copy=False),
                source_index=int(w1.channel_index),
                role="w1_nuclei",
                label_metadata=w1_label_metadata,
            ),
            AlignedROIMask(
                mask=w2_labels.astype(np.int32, copy=False),
                source_index=int(w2.channel_index),
                role="w2_stain",
            ),
        )
    )

    return image, [asdict(result)], masks


def segment_metaxpress_round_objects(
    slice_data: np.ndarray,
    settings: MetaXpressWavelengthSettings,
    pixel_size_um: float,
) -> np.ndarray:
    """Segment bright round objects using shared MetaXpress-style controls."""

    min_width_px = settings.approx_min_width / pixel_size_um
    max_width_px = settings.approx_max_width / pixel_size_um
    intensity_above_background = local_background_response(
        slice_data,
        object_width_px=max_width_px,
        bright_objects=True,
    )
    binary = intensity_above_background >= settings.intensity_above_local_background

    max_object_area = max(1, int(np.ceil(np.pi * (max_width_px / 2.0) ** 2)))
    seed_spacing = max(1, int(round(min_width_px / 2.0)))
    seed_footprint = odd_size(max(1.0, min_width_px / 2.0))
    labeled = _label_binary_components(
        binary,
        watershed_large_objects=True,
        watershed_split_size=max_object_area,
        watershed_max_size=None,
        watershed_min_distance=seed_spacing,
        watershed_footprint_size=seed_footprint,
    )

    keep_mask = np.zeros(int(labeled.max()) + 1, dtype=bool)
    for region in regionprops(labeled):
        if min_width_px <= region.axis_minor_length <= max_width_px:
            keep_mask[region.label] = True
    return _relabel_by_keep_mask(labeled, keep_mask)


def _build_w2_compartments(
    w1_labels: np.ndarray,
    w1: MetaXpressWavelengthSettings,
    w2: MetaXpressW2Settings,
    pixel_size_um: float,
) -> np.ndarray:
    """Build non-overlapping W1-cell compartments used for W2 scoring."""

    stained_area = StainedArea(w2.stained_area)
    if stained_area == StainedArea.NUCLEUS:
        return w1_labels.astype(np.int32, copy=False)

    expansion_um = max(
        0.0,
        (w2.approx_max_width - w1.approx_max_width) / 2.0,
    )
    expansion_px = expansion_um / pixel_size_um
    return expand_labels(w1_labels, distance=expansion_px).astype(
        np.int32,
        copy=False,
    )


def _measure_stained_area_by_cell(
    compartments: np.ndarray,
    stained_mask: np.ndarray,
    cell_count: int,
    pixel_size_um: float,
) -> np.ndarray:
    """Measure W2 stained area in square micrometers for every W1 cell."""

    stained_cell_labels = compartments[stained_mask & (compartments > 0)]
    stained_pixels = np.bincount(
        stained_cell_labels,
        minlength=cell_count + 1,
    ).astype(float, copy=False)
    return stained_pixels * pixel_size_um**2


def _segment_simple_slice(
    slice_data: np.ndarray,
    settings: SimpleCellSegmentationConfig,
) -> np.ndarray:
    """Segment one 2D plane with the simple counter's canonical logic."""
    threshold_method = ThresholdMethod(settings.threshold_method)
    foreground = Foreground(settings.foreground)
    threshold_value = _compute_threshold(
        slice_data,
        settings,
    )

    if foreground == Foreground.BRIGHT:
        binary = slice_data > threshold_value
    elif foreground == Foreground.DARK:
        binary = slice_data < threshold_value
    else:
        raise ValueError(
            f"Unknown foreground: {foreground!r} (expected 'bright' or 'dark')"
        )

    labeled = _label_binary_components(
        binary,
        watershed_large_objects=settings.watershed_large_objects,
        watershed_split_size=(
            settings.max_size
            if settings.watershed_min_size is None
            else settings.watershed_min_size
        ),
        watershed_max_size=settings.watershed_max_size,
        watershed_min_distance=settings.watershed_min_distance,
        watershed_footprint_size=settings.watershed_footprint_size,
    )

    if labeled.max() == 0:
        return np.zeros_like(labeled, dtype=np.int32)

    if settings.max_eccentricity == 1.0:
        return _filter_labels_by_area(
            labeled,
            min_size=settings.min_size,
            max_size=settings.max_size,
        )

    keep_mask = np.zeros(int(labeled.max()) + 1, dtype=bool)
    for region in regionprops(labeled):
        if (
            settings.min_size <= region.area <= settings.max_size
            and region.eccentricity <= settings.max_eccentricity
        ):
            keep_mask[region.label] = True

    return _relabel_by_keep_mask(labeled, keep_mask)


def _label_binary_components(
    binary: np.ndarray,
    *,
    watershed_large_objects: bool,
    watershed_split_size: int,
    watershed_max_size: Optional[int],
    watershed_min_distance: int,
    watershed_footprint_size: int,
) -> np.ndarray:
    """Label a binary mask and optionally split large connected components."""

    labeled, num_objects = ndi.label(binary)
    if watershed_large_objects and num_objects > 0:
        labeled = _watershed_large_objects(
            labeled,
            split_size=watershed_split_size,
            watershed_max_size=watershed_max_size,
            min_distance=watershed_min_distance,
            footprint_size=watershed_footprint_size,
        )
    return labeled.astype(np.int32, copy=False)


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
    settings: SimpleCellSegmentationConfig,
) -> float:
    """Compute one threshold in the native intensity scale of a 2D plane."""
    threshold_method = ThresholdMethod(settings.threshold_method)
    if threshold_method == ThresholdMethod.OTSU:
        return float(threshold_otsu(slice_data))
    if threshold_method == ThresholdMethod.LI:
        return float(threshold_li(slice_data))
    if threshold_method == ThresholdMethod.YEN:
        return float(threshold_yen(slice_data))
    if threshold_method == ThresholdMethod.PERCENTILE:
        return float(np.percentile(slice_data, settings.threshold_percentile))
    if threshold_method == ThresholdMethod.MANUAL:
        threshold_value = float(settings.threshold)
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
