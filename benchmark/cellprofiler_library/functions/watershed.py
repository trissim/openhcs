"""
Converted from CellProfiler: Watershed
Original: watershed
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
    build_structuring_element,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    CELLPROFILER_WATERSHED_BASIC_DEFAULTS,
    CellProfiler4DistanceMarkerBackendStrategy,
    CellProfiler4DistanceInitialWatershedStrategy,
    CellProfiler4InitialWatershedStrategy,
    CellProfiler4MarkerInitialWatershedStrategy,
    CellProfiler4WatershedRuntimeStrategy,
    ConnectedComponentsWatershedSeedStrategy,
    DistanceWatershedMethodStrategy,
    IntensityWatershedDeclumpStrategy,
    IntensityWatershedMethodStrategy,
    LibraryWatershedRuntimeStrategy,
    LocalWatershedSeedStrategy,
    MarkerWatershedMethodStrategy,
    NoneWatershedDeclumpStrategy,
    RegionalWatershedSeedStrategy,
    ShapeWatershedDeclumpStrategy,
    WatershedBasicDefaults,
    WatershedComputationImages,
    WatershedDeclumpMethod,
    WatershedDeclumpStrategy,
    WatershedInputKeyword,
    WatershedInputs,
    WatershedMethod,
    WatershedMethodStrategy,
    WatershedParameters,
    WatershedRuntimeFamily,
    WatershedRuntimeStrategy,
    WatershedSeedMethod,
    WatershedSeedStrategy,
    WatershedSegmentationSurface,
    coerce_watershed_declump_method,
    coerce_watershed_method,
    coerce_watershed_runtime_family,
    coerce_watershed_seed_method,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


@dataclass
class WatershedStats:
    slice_index: int
    object_count: int
    mean_area: float


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics.

    Newer scikit-image raises a queued neighbor's priority to at least the
    source pixel priority. CellProfiler 4.2 did not, and that changes basin
    ownership for tied/near-tied object boundaries by a few pixels.
    """
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed as _cellprofiler_legacy_watershed,
    )

    return _cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
        backend_provider=backend_provider,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("markers", "mask")
@special_outputs(
    ("watershed_stats", csv_materializer(fields=["slice_index", "object_count", "mean_area"])),
    ("labels", segmentation_mask_rois())
)
def watershed(
    image: np.ndarray,
    markers: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    watershed_method: Literal["distance", "intensity", "markers"] = "distance",
    declump_method: Literal["shape", "intensity"] = "shape",
    seed_method: Literal["local", "regional"] = "local",
    use_advanced_settings: bool = True,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    min_intensity: float = 0.0,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: float = 0.0,
    exclude_border: bool = False,
    watershed_line: bool = False,
    gaussian_sigma: float = 0.0,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
    runtime_family: WatershedRuntimeFamily | str = WatershedRuntimeFamily.LIBRARY,
) -> Tuple[np.ndarray, WatershedStats, np.ndarray]:
    """
    Apply watershed segmentation to separate touching objects.
    
    Args:
        image: Input binary or grayscale image (H, W)
        watershed_method: Method for watershed - 'distance' uses distance transform,
                         'intensity' uses intensity image, 'markers' uses marker image
        declump_method: Method for declumping - 'shape' or 'intensity'
        seed_method: Seed detection method - 'local' for local maxima, 'regional' for regional
        max_seeds: Maximum number of seeds (-1 for unlimited)
        downsample: Downsampling factor for speed
        min_distance: Minimum distance between seeds
        min_intensity: Minimum intensity for seeds
        footprint: Footprint size for local maxima detection
        connectivity: Connectivity for watershed (1 or 2)
        compactness: Compactness parameter for watershed
        exclude_border: Whether to exclude objects touching border
        watershed_line: Whether to draw watershed lines between objects
        gaussian_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
        structuring_element: Shape of structuring element for morphological operations
        structuring_element_size: Size of structuring element
    
    Returns:
        Tuple of (original image, watershed statistics, labeled image)
    """
    from skimage.measure import regionprops
    method = coerce_watershed_method(watershed_method)
    resolved_declump_method = coerce_watershed_declump_method(declump_method)
    if not use_advanced_settings:
        defaults = CELLPROFILER_WATERSHED_BASIC_DEFAULTS
        seed_method = defaults.seed_method
        max_seeds = defaults.max_seeds
        min_distance = defaults.min_distance
        min_intensity = defaults.min_intensity
        connectivity = defaults.connectivity
        compactness = defaults.compactness
        watershed_line = defaults.watershed_line
        gaussian_sigma = defaults.gaussian_sigma
    
    if not np.array_equal(image, image.astype(bool)):
        raise ValueError("Watershed expects a thresholded image as input.")
    structuring_element_array = adapt_structuring_element_rank(
        build_structuring_element(
            coerce_cellprofiler_enum(StructuringElement, structuring_element),
            structuring_element_size,
        ),
        image.ndim,
    )
    if structuring_element_array.ndim != image.ndim:
        raise ValueError(
            "Watershed structuring element dimensionality must match the image; "
            f"got structuring element ndim={structuring_element_array.ndim} for "
            f"image ndim={image.ndim}."
        )

    labels = WatershedRuntimeStrategy.for_enum_member(
        coerce_watershed_runtime_family(runtime_family)
    ).labels(
        image,
        markers,
        mask,
        WatershedParameters(
            method=method,
            declump_method=resolved_declump_method,
            seed_method=coerce_watershed_seed_method(seed_method),
            use_advanced_settings=use_advanced_settings,
            max_seeds=max_seeds,
            downsample=downsample,
            min_distance=min_distance,
            min_intensity=min_intensity,
            footprint=footprint,
            connectivity=connectivity,
            compactness=compactness,
            exclude_border=exclude_border,
            watershed_line=watershed_line,
            gaussian_sigma=gaussian_sigma,
            structuring_element=structuring_element_array,
        ),
    )

    props = regionprops(labels)
    object_count = len(props)
    mean_area = np.mean([p.area for p in props]) if props else 0.0

    stats = WatershedStats(
        slice_index=0,
        object_count=object_count,
        mean_area=float(mean_area)
    )

    return image, stats, labels.astype(np.int32)
