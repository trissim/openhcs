"""
CuPy-based cell counting and multi-channel colocalization analysis for OpenHCS.

This module provides comprehensive cell counting capabilities using CuPy and CuCIM,
supporting both single-channel and multi-channel analysis with various detection
methods and colocalization metrics.
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

from skimage.segmentation import watershed as skimage_watershed

from openhcs.core.memory import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.backends.analysis.cell_counting_common import (
    AreaFilter,
    AreaFilterRequest,
    CellCountResult,
    ColocalizationAnalysis,
    ColocalizationMethod,
    DetectionMethod,
    MultiChannelResult,
    ThresholdMethod,
    WatershedThresholdBackend,
    WatershedThresholdMethodStrategy,
    colocalization_analyzer_catalog,
    detection_method_catalog,
)
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    MaterializationSpec,
)
from openhcs.utils.import_utils import optional_import_placeholder

cp = optional_import_placeholder("cupy")

logger = logging.getLogger(__name__)

# Optional imports using established pattern
cupyx_scipy = optional_import_placeholder("cupyx.scipy")
cupyx_spatial_distance = optional_import_placeholder("cupyx.scipy.spatial.distance")
cucim_feature = optional_import_placeholder("cucim.skimage.feature")
cucim_filters = optional_import_placeholder("cucim.skimage.filters")
cucim_segmentation = optional_import_placeholder("cucim.skimage.segmentation")
cucim_morphology = optional_import_placeholder("cucim.skimage.morphology")
cucim_measure = optional_import_placeholder("cucim.skimage.measure")
cucim_exposure = optional_import_placeholder("cucim.skimage.exposure")

ndimage = cupyx_scipy.ndimage
blob_log = cucim_feature.blob_log
blob_dog = cucim_feature.blob_dog
blob_doh = cucim_feature.blob_doh
peak_local_max = cucim_feature.peak_local_max
threshold_otsu = cucim_filters.threshold_otsu
threshold_li = cucim_filters.threshold_li
gaussian = cucim_filters.gaussian
median = cucim_filters.median
clear_border = cucim_segmentation.clear_border
remove_small_objects = cucim_morphology.remove_small_objects
disk = cucim_morphology.disk
label = cucim_measure.label
regionprops = cucim_measure.regionprops

WATERSHED_THRESHOLD_BACKEND = WatershedThresholdBackend(
    otsu=threshold_otsu,
    li=threshold_li,
)


def _watershed_with_cpu_partition(image, markers, *, mask):
    """Run watershed through the CPU fallback and restore the CuPy device."""

    from openhcs.constants.constants import MemoryType

    device_id = MemoryType.CUPY.device_id_of(image)
    if device_id is None:
        raise ValueError("CuPy watershed input does not declare a GPU device.")
    labels = skimage_watershed(
        MemoryType.CUPY.to_numpy(image),
        MemoryType.CUPY.to_numpy(markers),
        mask=MemoryType.CUPY.to_numpy(mask),
    )
    return MemoryType.CUPY.from_numpy(labels, device_id)


def count_cells_single_channel(
    image_stack: cp.ndarray,
    # Detection method and parameters
    detection_method: DetectionMethod = DetectionMethod.BLOB_LOG,  # UI will show radio buttons
    # Blob detection parameters
    min_sigma: float = 1.0,  # Minimum blob size (pixels)
    max_sigma: float = 10.0,  # Maximum blob size (pixels)
    num_sigma: int = 10,  # Number of sigma values to test
    threshold: float = 0.1,  # Detection threshold (0.0-1.0)
    overlap: float = 0.5,  # Maximum overlap between blobs (0.0-1.0)
    # Watershed parameters
    watershed_footprint_size: int = 3,  # Local maxima footprint size
    watershed_min_distance: int = 5,  # Minimum distance between peaks
    watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # UI will show threshold methods
    # Preprocessing parameters
    enable_preprocessing: bool = True,
    gaussian_sigma: float = 1.0,  # Gaussian blur sigma
    median_disk_size: int = 1,  # Median filter disk size
    # Filtering parameters
    min_cell_area: int = 10,  # Minimum cell area (pixels)
    max_cell_area: int = 1000,  # Maximum cell area (pixels)
    remove_border_cells: bool = True,  # Remove cells touching image border
    # Output parameters
    return_segmentation_mask: bool = False,
) -> Tuple[cp.ndarray, List[CellCountResult]]:
    """
    Count cells in single-channel image stack using various detection methods.

    Args:
        image_stack: 3D CuPy array (Z, Y, X) where each Z slice is processed independently
        detection_method: Method for cell detection (see DetectionMethod enum)
        min_sigma: Minimum blob size for blob detection methods
        max_sigma: Maximum blob size for blob detection methods
        num_sigma: Number of sigma values to test for blob detection
        threshold: Detection threshold (method-dependent)
        overlap: Maximum overlap between detected blobs
        watershed_footprint_size: Footprint size for local maxima detection
        watershed_min_distance: Minimum distance between watershed peaks
        watershed_threshold_method: Thresholding method for watershed
        enable_preprocessing: Apply Gaussian and median filtering
        gaussian_sigma: Standard deviation for Gaussian blur
        median_disk_size: Disk size for median filtering
        min_cell_area: Minimum area for valid cells
        max_cell_area: Maximum area for valid cells
        remove_border_cells: Remove cells touching image borders
        return_segmentation_mask: Return segmentation masks in output

    Returns:
        output_stack: Original image stack unchanged (Z, Y, X)
        cell_count_results: List of CellCountResult objects for each slice
        segmentation_masks: (Special output) List of segmentation mask arrays if return_segmentation_mask=True
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")

    results = []
    segmentation_masks = []

    # Store parameters for reproducibility (convert enums to values)
    parameters = {
        "detection_method": detection_method.value,
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "num_sigma": num_sigma,
        "threshold": threshold,
        "overlap": overlap,
        "watershed_footprint_size": watershed_footprint_size,
        "watershed_min_distance": watershed_min_distance,
        "watershed_threshold_method": watershed_threshold_method.value,
        "gaussian_sigma": gaussian_sigma,
        "median_disk_size": median_disk_size,
        "min_cell_area": min_cell_area,
        "max_cell_area": max_cell_area,
        "remove_border_cells": remove_border_cells,
    }

    logging.info(
        f"Processing {image_stack.shape[0]} slices with {detection_method.value} method"
    )

    for z_idx in range(image_stack.shape[0]):
        slice_img = image_stack[z_idx].astype(cp.float64)

        # Apply preprocessing if enabled
        if enable_preprocessing:
            slice_img = _preprocess_image(slice_img, gaussian_sigma, median_disk_size)

        # Detect cells using specified method
        result = _detect_cells_single_method(
            slice_img, z_idx, detection_method.value, parameters
        )

        results.append(result)

        # Create segmentation mask if requested
        if return_segmentation_mask:
            segmentation_mask = _create_segmentation_visualization(
                slice_img,
                result.cell_positions,
                max_sigma,
                result.cell_areas,
                result.binary_mask,
            )
            segmentation_masks.append(segmentation_mask)

    # Always return segmentation masks (empty list if not requested)
    # This ensures consistent return signature for special outputs system
    return image_stack, results, segmentation_masks


@cupy_func
@artifact_outputs(
    (
        "multi_channel_counts",
        MaterializationSpec(
            JsonOptions(filename_suffix=".json", wrap_list=True),
            CsvOptions(filename_suffix="_details.csv"),
            primary=0,
        ),
    )
)
def count_cells_multi_channel(
    image_stack: cp.ndarray,
    chan_1: int,  # Index of first channel (positional arg)
    chan_2: int,  # Index of second channel (positional arg)
    # Detection parameters for channel 1 (all single-channel params available)
    chan_1_method: DetectionMethod = DetectionMethod.BLOB_LOG,  # UI will show radio buttons
    chan_1_min_sigma: float = 1.0,  # Minimum blob size (pixels)
    chan_1_max_sigma: float = 10.0,  # Maximum blob size (pixels)
    chan_1_num_sigma: int = 10,  # Number of sigma values to test
    chan_1_threshold: float = 0.1,  # Detection threshold (0.0-1.0)
    chan_1_overlap: float = 0.5,  # Maximum overlap between blobs (0.0-1.0)
    chan_1_watershed_footprint_size: int = 3,  # Local maxima footprint size
    chan_1_watershed_min_distance: int = 5,  # Minimum distance between peaks
    chan_1_watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # Thresholding method
    chan_1_enable_preprocessing: bool = True,  # Apply preprocessing
    chan_1_gaussian_sigma: float = 1.0,  # Gaussian blur sigma
    chan_1_median_disk_size: int = 1,  # Median filter disk size
    chan_1_min_area: int = 10,  # Minimum cell area (pixels)
    chan_1_max_area: int = 1000,  # Maximum cell area (pixels)
    chan_1_remove_border_cells: bool = True,  # Remove cells touching border
    # Detection parameters for channel 2 (all single-channel params available)
    chan_2_method: DetectionMethod = DetectionMethod.BLOB_LOG,  # UI will show radio buttons
    chan_2_min_sigma: float = 1.0,  # Minimum blob size (pixels)
    chan_2_max_sigma: float = 10.0,  # Maximum blob size (pixels)
    chan_2_num_sigma: int = 10,  # Number of sigma values to test
    chan_2_threshold: float = 0.1,  # Detection threshold (0.0-1.0)
    chan_2_overlap: float = 0.5,  # Maximum overlap between blobs (0.0-1.0)
    chan_2_watershed_footprint_size: int = 3,  # Local maxima footprint size
    chan_2_watershed_min_distance: int = 5,  # Minimum distance between peaks
    chan_2_watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # Thresholding method
    chan_2_enable_preprocessing: bool = True,  # Apply preprocessing
    chan_2_gaussian_sigma: float = 1.0,  # Gaussian blur sigma
    chan_2_median_disk_size: int = 1,  # Median filter disk size
    chan_2_min_area: int = 10,  # Minimum cell area (pixels)
    chan_2_max_area: int = 1000,  # Maximum cell area (pixels)
    chan_2_remove_border_cells: bool = True,  # Remove cells touching border
    # Colocalization parameters
    colocalization_method: ColocalizationMethod = ColocalizationMethod.DISTANCE_BASED,  # UI will show coloc methods
    max_distance: float = 5.0,  # Maximum distance for colocalization (pixels)
    min_overlap_area: float = 0.3,  # Minimum overlap fraction for area-based method
    intensity_threshold: float = 0.5,  # Threshold for intensity-based methods
    # Output parameters
    return_colocalization_map: bool = False,
) -> Tuple[cp.ndarray, List[MultiChannelResult]]:
    """
    Count cells in multi-channel image stack with colocalization analysis.

    Each channel can be processed with independent parameters, providing the same
    flexibility as the single-channel function for each channel separately.

    Args:
        image_stack: 3D CuPy array (Z, Y, X) where Z represents different channels
        chan_1: Index of first channel in the stack (positional)
        chan_2: Index of second channel in the stack (positional)

        # Channel 1 detection parameters (same as single-channel function)
        chan_1_method: Detection method for channel 1 (DetectionMethod enum)
        chan_1_min_sigma: Minimum blob size for channel 1
        chan_1_max_sigma: Maximum blob size for channel 1
        chan_1_num_sigma: Number of sigma values to test for channel 1
        chan_1_threshold: Detection threshold for channel 1 (0.0-1.0)
        chan_1_overlap: Maximum overlap between blobs for channel 1
        chan_1_watershed_footprint_size: Local maxima footprint size for channel 1
        chan_1_watershed_min_distance: Minimum distance between peaks for channel 1
        chan_1_watershed_threshold_method: Thresholding method for channel 1
        chan_1_enable_preprocessing: Apply preprocessing to channel 1
        chan_1_gaussian_sigma: Gaussian blur sigma for channel 1
        chan_1_median_disk_size: Median filter size for channel 1
        chan_1_min_area: Minimum cell area for channel 1
        chan_1_max_area: Maximum cell area for channel 1
        chan_1_remove_border_cells: Remove border cells for channel 1

        # Channel 2 detection parameters (same as single-channel function)
        chan_2_method: Detection method for channel 2 (DetectionMethod enum)
        chan_2_min_sigma: Minimum blob size for channel 2
        chan_2_max_sigma: Maximum blob size for channel 2
        chan_2_num_sigma: Number of sigma values to test for channel 2
        chan_2_threshold: Detection threshold for channel 2 (0.0-1.0)
        chan_2_overlap: Maximum overlap between blobs for channel 2
        chan_2_watershed_footprint_size: Local maxima footprint size for channel 2
        chan_2_watershed_min_distance: Minimum distance between peaks for channel 2
        chan_2_watershed_threshold_method: Thresholding method for channel 2
        chan_2_enable_preprocessing: Apply preprocessing to channel 2
        chan_2_gaussian_sigma: Gaussian blur sigma for channel 2
        chan_2_median_disk_size: Median filter size for channel 2
        chan_2_min_area: Minimum cell area for channel 2
        chan_2_max_area: Maximum cell area for channel 2
        chan_2_remove_border_cells: Remove border cells for channel 2

        # Colocalization parameters
        colocalization_method: Method for determining colocalization (ColocalizationMethod enum)
        max_distance: Maximum distance for distance-based colocalization (pixels)
        min_overlap_area: Minimum overlap fraction for area-based colocalization
        intensity_threshold: Threshold for intensity-based colocalization (0.0-1.0)
        return_colocalization_map: Return colocalization visualization

    Returns:
        output_stack: Original images or colocalization maps
        multi_channel_results: List of MultiChannelResult objects
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")

    if chan_1 >= image_stack.shape[0] or chan_2 >= image_stack.shape[0]:
        raise ValueError(
            f"Channel indices {chan_1}, {chan_2} exceed stack size {image_stack.shape[0]}"
        )

    if chan_1 == chan_2:
        raise ValueError("Channel 1 and Channel 2 must be different")

    # Extract channel images
    chan_1_img = image_stack[chan_1 : chan_1 + 1]  # Keep 3D shape for consistency
    chan_2_img = image_stack[chan_2 : chan_2 + 1]

    # Count cells in each channel separately using the single-channel function
    # Channel 1 parameters (all explicit)
    chan_1_params = {
        "detection_method": chan_1_method,
        "min_sigma": chan_1_min_sigma,
        "max_sigma": chan_1_max_sigma,
        "num_sigma": chan_1_num_sigma,
        "threshold": chan_1_threshold,
        "overlap": chan_1_overlap,
        "watershed_footprint_size": chan_1_watershed_footprint_size,
        "watershed_min_distance": chan_1_watershed_min_distance,
        "watershed_threshold_method": chan_1_watershed_threshold_method,
        "enable_preprocessing": chan_1_enable_preprocessing,
        "gaussian_sigma": chan_1_gaussian_sigma,
        "median_disk_size": chan_1_median_disk_size,
        "min_cell_area": chan_1_min_area,
        "max_cell_area": chan_1_max_area,
        "remove_border_cells": chan_1_remove_border_cells,
        "return_segmentation_mask": False,
    }

    # Channel 2 parameters (all explicit)
    chan_2_params = {
        "detection_method": chan_2_method,
        "min_sigma": chan_2_min_sigma,
        "max_sigma": chan_2_max_sigma,
        "num_sigma": chan_2_num_sigma,
        "threshold": chan_2_threshold,
        "overlap": chan_2_overlap,
        "watershed_footprint_size": chan_2_watershed_footprint_size,
        "watershed_min_distance": chan_2_watershed_min_distance,
        "watershed_threshold_method": chan_2_watershed_threshold_method,
        "enable_preprocessing": chan_2_enable_preprocessing,
        "gaussian_sigma": chan_2_gaussian_sigma,
        "median_disk_size": chan_2_median_disk_size,
        "min_cell_area": chan_2_min_area,
        "max_cell_area": chan_2_max_area,
        "remove_border_cells": chan_2_remove_border_cells,
        "return_segmentation_mask": False,
    }

    # Process each channel
    _, chan_1_results = count_cells_single_channel(chan_1_img, **chan_1_params)
    _, chan_2_results = count_cells_single_channel(chan_2_img, **chan_2_params)

    # Perform colocalization analysis
    multi_results = []
    output_stack = image_stack.copy()

    # Since we're processing single slices from each channel, we only have one result each
    chan_1_result = chan_1_results[0]
    chan_2_result = chan_2_results[0]

    # Analyze colocalization
    coloc_result = _analyze_colocalization(
        chan_1_result,
        chan_2_result,
        colocalization_method.value,
        max_distance,
        min_overlap_area,
        intensity_threshold,
    )

    multi_results.append(coloc_result)

    # Create colocalization visualization if requested
    if return_colocalization_map:
        coloc_map = _create_colocalization_map(
            image_stack[chan_1], image_stack[chan_2], coloc_result
        )
        # Replace one of the channels with the colocalization map
        output_stack = cp.stack([image_stack[chan_1], image_stack[chan_2], coloc_map])

    return output_stack, multi_results


def _preprocess_image(
    image: cp.ndarray, gaussian_sigma: float, median_disk_size: int
) -> cp.ndarray:
    """Apply preprocessing to enhance cell detection."""
    # Gaussian blur to reduce noise
    if gaussian_sigma > 0:
        image = gaussian(image, sigma=gaussian_sigma, preserve_range=True)

    # Median filter to remove salt-and-pepper noise
    if median_disk_size > 0:
        image = median(image, disk(median_disk_size))

    return image


def _detect_cells_single_method(
    image: cp.ndarray, slice_idx: int, method: str, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using specified method."""
    try:
        detector = DETECTION_METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown detection method: {method}")
    return detector(image, slice_idx, params)


def _detect_cells_blob_log(
    image: cp.ndarray, slice_idx: int, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using Laplacian of Gaussian blob detection."""
    blobs = blob_log(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=params["num_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"],
    )

    # Extract positions, areas, and intensities
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        # Estimate area from sigma (blob radius ≈ sigma * sqrt(2))
        radius = sigma * cp.sqrt(2)
        area = cp.pi * radius**2
        areas.append(float(area))

        # Sample intensity at blob center
        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        # Use sigma as confidence measure (larger blobs = higher confidence)
        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    # Filter by area constraints
    filtered_data = (
        AreaFilter()
        .apply(
            AreaFilterRequest.from_measurements(
                positions,
                areas,
                intensities,
                confidences,
                min_area=params["min_cell_area"],
                max_area=params["max_cell_area"],
            )
        )
        .as_measurement_args()
    )

    return CellCountResult.from_measurements(
        slice_idx, "blob_log", *filtered_data, params
    )


def _detect_cells_blob_dog(
    image: cp.ndarray, slice_idx: int, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using Difference of Gaussian blob detection."""
    blobs = blob_dog(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"],
    )

    # Process similar to blob_log
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        radius = sigma * cp.sqrt(2)
        area = cp.pi * radius**2
        areas.append(float(area))

        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    filtered_data = (
        AreaFilter()
        .apply(
            AreaFilterRequest.from_measurements(
                positions,
                areas,
                intensities,
                confidences,
                min_area=params["min_cell_area"],
                max_area=params["max_cell_area"],
            )
        )
        .as_measurement_args()
    )

    return CellCountResult.from_measurements(
        slice_idx, "blob_dog", *filtered_data, params
    )


def _detect_cells_blob_doh(
    image: cp.ndarray, slice_idx: int, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using Determinant of Hessian blob detection."""
    blobs = blob_doh(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=params["num_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"],
    )

    # Process similar to other blob methods
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        radius = sigma * cp.sqrt(2)
        area = cp.pi * radius**2
        areas.append(float(area))

        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    filtered_data = (
        AreaFilter()
        .apply(
            AreaFilterRequest.from_measurements(
                positions,
                areas,
                intensities,
                confidences,
                min_area=params["min_cell_area"],
                max_area=params["max_cell_area"],
            )
        )
        .as_measurement_args()
    )

    return CellCountResult.from_measurements(
        slice_idx, "blob_doh", *filtered_data, params
    )


def _detect_cells_watershed(
    image: cp.ndarray, slice_idx: int, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using watershed segmentation."""
    threshold_val = WatershedThresholdMethodStrategy.for_method_value(
        params["watershed_threshold_method"],
    ).threshold(
        WATERSHED_THRESHOLD_BACKEND,
        image,
        params["watershed_threshold_method"],
    )

    # Create binary mask
    binary = image > threshold_val

    # Remove small objects and border objects
    binary = remove_small_objects(binary, min_size=params["min_cell_area"])
    if params["remove_border_cells"]:
        binary = clear_border(binary)

    # Find local maxima as seeds
    distance = ndimage.distance_transform_edt(binary)
    local_maxima = peak_local_max(
        distance,
        min_distance=params["watershed_min_distance"],
        footprint=cp.ones(
            (params["watershed_footprint_size"], params["watershed_footprint_size"])
        ),
    )

    # Convert coordinates to binary mask
    local_maxima_mask = cp.zeros_like(distance, dtype=bool)
    if len(local_maxima) > 0:
        local_maxima_mask[local_maxima[:, 0], local_maxima[:, 1]] = True

    # Create markers for watershed
    # Convert boolean mask to integer labels for connected components
    markers = label(local_maxima_mask.astype(cp.uint8))

    # Apply watershed
    labels = _watershed_with_cpu_partition(-distance, markers, mask=binary)

    # Extract region properties
    regions = regionprops(labels, intensity_image=image)

    positions = []
    areas = []
    intensities = []
    confidences = []
    valid_labels = []  # Track which labels pass the size filter

    for region in regions:
        # Filter by area
        if params["min_cell_area"] <= region.area <= params["max_cell_area"]:
            # Centroid (note: regionprops returns (row, col) = (y, x))
            y, x = region.centroid
            positions.append((float(x), float(y)))

            areas.append(float(region.area))
            intensities.append(float(region.mean_intensity))

            # Use area as confidence measure (normalized)
            confidence = min(1.0, region.area / params["max_cell_area"])
            confidences.append(confidence)

            # Track this label as valid
            valid_labels.append(region.label)

    # Create filtered binary mask with only cells that passed size filter
    filtered_binary_mask = cp.isin(labels, cp.array(valid_labels))

    return CellCountResult.from_measurements(
        slice_idx,
        "watershed",
        positions,
        areas,
        intensities,
        confidences,
        params,
        binary_mask=filtered_binary_mask,  # Only cells that passed all filters
    )


def _detect_cells_threshold(
    image: cp.ndarray, slice_idx: int, params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using simple thresholding and connected components."""
    # Apply threshold
    binary = image > params["threshold"] * image.max()

    # Remove small objects and border objects
    binary = remove_small_objects(binary, min_size=params["min_cell_area"])
    if params["remove_border_cells"]:
        binary = clear_border(binary)

    # Label connected components
    labels = label(binary)
    regions = regionprops(labels, intensity_image=image)

    positions = []
    areas = []
    intensities = []
    confidences = []
    valid_labels = []  # Track which labels pass the size filter

    for region in regions:
        # Filter by area
        if params["min_cell_area"] <= region.area <= params["max_cell_area"]:
            y, x = region.centroid
            positions.append((float(x), float(y)))

            areas.append(float(region.area))
            intensities.append(float(region.mean_intensity))

            # Use intensity as confidence measure
            confidence = float(region.mean_intensity / image.max())
            confidences.append(confidence)

            # Track this label as valid
            valid_labels.append(region.label)

    # Create filtered binary mask with only cells that passed size filter
    filtered_binary_mask = cp.isin(labels, cp.array(valid_labels))

    return CellCountResult.from_measurements(
        slice_idx,
        "threshold",
        positions,
        areas,
        intensities,
        confidences,
        params,
        binary_mask=filtered_binary_mask,  # Only cells that passed all filters
    )


DetectionMethodHandler = Callable[
    [cp.ndarray, int, Dict[str, Any]],
    CellCountResult,
]


DETECTION_METHODS: dict[str, DetectionMethodHandler] = detection_method_catalog(
    blob_log=_detect_cells_blob_log,
    blob_dog=_detect_cells_blob_dog,
    blob_doh=_detect_cells_blob_doh,
    watershed=_detect_cells_watershed,
    threshold=_detect_cells_threshold,
)


def _analyze_colocalization(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    method: str,
    max_distance: float,
    min_overlap_area: float,
    intensity_threshold: float,
) -> MultiChannelResult:
    """Analyze colocalization between two channels."""
    try:
        analyzer = COLOCALIZATION_ANALYZERS[method]
    except KeyError:
        raise ValueError(f"Unknown colocalization method: {method}")
    return analyzer(
        chan_1_result,
        chan_2_result,
        max_distance,
        min_overlap_area,
        intensity_threshold,
    )


ColocalizationAnalyzer = Callable[
    [CellCountResult, CellCountResult, float, float, float],
    MultiChannelResult,
]


COLOCALIZATION_ANALYZERS: dict[str, ColocalizationAnalyzer] = (
    colocalization_analyzer_catalog(
        distance_based=(
            lambda chan_1_result, chan_2_result, max_distance, _min_overlap_area, _intensity_threshold: ColocalizationAnalysis().distance_based(
                chan_1_result,
                chan_2_result,
                max_distance,
            )
        ),
        overlap_area=(
            lambda chan_1_result, chan_2_result, _max_distance, min_overlap_area, _intensity_threshold: ColocalizationAnalysis().overlap_based(
                chan_1_result,
                chan_2_result,
                min_overlap_area,
            )
        ),
        intensity_correlation=(
            lambda chan_1_result, chan_2_result, _max_distance, _min_overlap_area, intensity_threshold: ColocalizationAnalysis().intensity_based(
                chan_1_result,
                chan_2_result,
                intensity_threshold,
            )
        ),
        manders_coefficients=(
            lambda chan_1_result, chan_2_result, _max_distance, _min_overlap_area, intensity_threshold: ColocalizationAnalysis().manders(
                chan_1_result,
                chan_2_result,
                intensity_threshold,
            )
        ),
    )
)


def _create_segmentation_visualization(
    image: cp.ndarray,
    positions: List[Tuple[float, float]],
    max_sigma: float,
    cell_areas: List[float] = None,
    binary_mask: cp.ndarray = None,
) -> cp.ndarray:
    """Create segmentation visualization using actual binary mask if available."""

    # If we have the actual binary mask from detection, use it directly
    if binary_mask is not None:
        # Convert boolean mask to uint16 to match input image dtype
        # Use max intensity for detected cells, 0 for background
        max_intensity = image.max() if image.max() > 0 else 65535
        return (binary_mask * max_intensity).astype(image.dtype)

    # Fallback to original circular marker approach for blob methods
    visualization = image.copy()

    # Mark detected cells with their actual sizes
    for i, (x, y) in enumerate(positions):
        # Use actual cell area if available, otherwise fall back to max_sigma
        if cell_areas and i < len(cell_areas):
            # Convert area to radius (assuming circular cells)
            radius = cp.sqrt(cell_areas[i] / cp.pi)
        else:
            # Fallback to max_sigma for backward compatibility
            radius = max_sigma * 2

        # Create circular markers with actual cell size
        rr, cc = cp.ogrid[: image.shape[0], : image.shape[1]]
        mask = (rr - y) ** 2 + (cc - x) ** 2 <= radius**2

        # Ensure indices are within bounds
        valid_mask = (
            (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
        )
        mask = mask & valid_mask

        visualization[mask] = visualization.max()  # Bright markers

    return visualization


def _create_colocalization_map(
    chan_1_img: cp.ndarray, chan_2_img: cp.ndarray, coloc_result: MultiChannelResult
) -> cp.ndarray:
    """Create colocalization visualization map."""
    # Create RGB-like visualization
    coloc_map = cp.zeros_like(chan_1_img)

    # Mark colocalized positions
    for x, y in coloc_result.overlap_positions:
        # Create markers for colocalized cells
        rr, cc = cp.ogrid[: chan_1_img.shape[0], : chan_1_img.shape[1]]
        mask = (rr - y) ** 2 + (cc - x) ** 2 <= 25  # 5-pixel radius

        valid_mask = (
            (rr >= 0)
            & (rr < chan_1_img.shape[0])
            & (cc >= 0)
            & (cc < chan_1_img.shape[1])
        )
        mask = mask & valid_mask

        coloc_map[mask] = chan_1_img.max()  # Bright colocalization markers

    return coloc_map
