"""
Converted from CellProfiler: IdentifyPrimaryObjects
Original: IdentifyPrimaryObjects.run

Identifies primary objects (e.g., nuclei) in grayscale images using
thresholding, declumping, and watershed segmentation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy


class UnclumpMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    NONE = "none"


class WatershedMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    PROPAGATE = "propagate"
    NONE = "none"


class FillHolesOption(Enum):
    NEVER = ("never", False, False)
    AFTER_BOTH = ("after_both", True, True)
    AFTER_DECLUMP = ("after_declump", False, True)

    def __new__(
        cls,
        value: str,
        fill_before_declump: bool,
        fill_after_declump: bool,
    ):
        option = object.__new__(cls)
        option._value_ = value
        option.fill_before_declump = fill_before_declump
        option.fill_after_declump = fill_after_declump
        return option


class ExcessObjectHandling(Enum):
    CONTINUE = ("Continue", False)
    ERASE = ("Erase", True)

    def __new__(cls, value: str, erase_excess: bool):
        option = object.__new__(cls)
        option._value_ = value
        option.erase_excess = erase_excess
        return option


class WatershedImageBuilder(ABC, metaclass=AutoRegisterMeta):
    """Build the watershed surface for one closed watershed method."""

    __registry_key__ = "method"
    method: ClassVar[WatershedMethod | None] = None

    @classmethod
    def for_method(cls, method: WatershedMethod) -> "WatershedImageBuilder":
        return cls.__registry__[method]()

    @abstractmethod
    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """Return the image used as the watershed surface."""


class IntensityWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.INTENSITY

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        return 1 - image


class ShapeWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.SHAPE

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi

        return -ndi.distance_transform_edt(binary)


class PropagateWatershedImageBuilder(WatershedImageBuilder):
    method = WatershedMethod.PROPAGATE

    def build(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        return 1 - image


def _fill_labeled_holes(labeled_image: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions without scanning once per object."""
    from scipy import ndimage as ndi

    object_mask = labeled_image > 0
    filled_mask = ndi.binary_fill_holes(object_mask)
    hole_mask = filled_mask & ~object_mask
    if not hole_mask.any():
        return labeled_image

    _, nearest_indices = ndi.distance_transform_edt(
        ~object_mask,
        return_distances=True,
        return_indices=True,
    )
    filled_labels = labeled_image.copy()
    filled_labels[hole_mask] = labeled_image[
        tuple(axis_indices[hole_mask] for axis_indices in nearest_indices)
    ]
    return filled_labels


@dataclass
class PrimaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    median_area: float
    total_area: float
    threshold_used: float


@numpy
def identify_primary_objects(
    image: np.ndarray,
    min_diameter: int = 10,
    max_diameter: int = 40,
    exclude_size: bool = True,
    exclude_border_objects: bool = True,
    unclump_method: UnclumpMethod = UnclumpMethod.INTENSITY,
    watershed_method: WatershedMethod = WatershedMethod.INTENSITY,
    automatic_smoothing: bool = True,
    smoothing_filter_size: int = 10,
    automatic_suppression: bool = True,
    maxima_suppression_size: float = 7.0,
    low_res_maxima: bool = True,
    fill_holes: FillHolesOption = FillHolesOption.AFTER_BOTH,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    maximum_object_count: int = 500,
    limit_erase: ExcessObjectHandling = ExcessObjectHandling.CONTINUE,
) -> Tuple[np.ndarray, PrimaryObjectStats, np.ndarray]:
    """
    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the input image' -> (pipeline-handled)
        'Name the primary objects to be identified' -> (pipeline-handled)
        'Typical diameter of objects, in pixel units (Min,Max)' -> [min_diameter, max_diameter]
        'Discard objects outside the diameter range?' -> exclude_size
        'Discard objects touching the border of the image?' -> exclude_border_objects
        'Method to distinguish clumped objects' -> unclump_method
        'Method to draw dividing lines between clumped objects' -> watershed_method
        'Size of smoothing filter' -> smoothing_filter_size
        'Suppress local maxima that are closer than this minimum allowed distance' -> maxima_suppression_size
        'Speed up by using lower-resolution image to find local maxima?' -> low_res_maxima
        'Fill holes in identified objects?' -> fill_holes
        'Automatically calculate size of smoothing filter for declumping?' -> automatic_smoothing
        'Automatically calculate minimum allowed distance between local maxima?' -> automatic_suppression
        'Handling of objects if excessive number of objects identified' -> limit_erase
        'Maximum number of objects' -> maximum_object_count
        'Threshold correction factor' -> threshold_correction_factor
        'Lower bound on threshold' -> threshold_min
        'Upper bound on threshold' -> threshold_max

    Identify primary objects in a grayscale image.
    
    Args:
        image: Input grayscale image (H, W)
        min_diameter: Minimum object diameter in pixels
        max_diameter: Maximum object diameter in pixels
        exclude_size: Discard objects outside diameter range
        exclude_border_objects: Discard objects touching image border
        unclump_method: Method to distinguish clumped objects
        watershed_method: Method to draw dividing lines between clumped objects
        automatic_smoothing: Auto-calculate smoothing filter size
        smoothing_filter_size: Size of smoothing filter for declumping
        automatic_suppression: Auto-calculate maxima suppression distance
        maxima_suppression_size: Minimum distance between local maxima
        low_res_maxima: Use lower resolution for finding maxima (faster)
        fill_holes: When to fill holes in objects
        threshold_correction_factor: Multiply threshold by this factor
        threshold_min: Minimum threshold value
        threshold_max: Maximum threshold value
        maximum_object_count: Max objects before erasing (if limit_erase=True)
        limit_erase: Erase all objects if count exceeds maximum
    
    Returns:
        Tuple of (original image, object statistics, labeled image)
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_li, gaussian
    from skimage.segmentation import watershed
    from skimage.morphology import binary_erosion, disk, remove_small_holes, remove_small_objects
    from skimage.measure import label, regionprops
    from skimage.feature import peak_local_max
    
    # Normalize image to 0-1 if needed
    if image.max() > 1.0:
        img = image.astype(np.float32) / image.max()
    else:
        img = image.astype(np.float32)
    
    # Calculate threshold using Li method (default in CellProfiler basic mode)
    thresh = threshold_li(img)
    thresh = thresh * threshold_correction_factor
    thresh = max(threshold_min, min(threshold_max, thresh))
    
    # Create binary image
    binary = img > thresh
    
    # Fill holes if requested (before declumping)
    if fill_holes.fill_before_declump:
        max_hole_size = int(np.pi * (max_diameter ** 2) / 4)
        binary = remove_small_holes(binary, area_threshold=max_hole_size)
    
    # Initial labeling
    labeled_image, object_count = ndi.label(binary, structure=np.ones((3, 3), bool))
    
    # Declumping and watershed
    if unclump_method != UnclumpMethod.NONE and watershed_method != WatershedMethod.NONE and object_count > 0:
        # Calculate smoothing filter size
        if automatic_smoothing:
            smooth_size = 2.35 * min_diameter / 3.5
        else:
            smooth_size = smoothing_filter_size
        
        # Calculate maxima suppression size
        if automatic_suppression:
            suppress_size = min_diameter / 1.5
        else:
            suppress_size = maxima_suppression_size
        
        # Smooth image for finding maxima
        if smooth_size > 0:
            sigma = smooth_size / 2.35
            smoothed = gaussian(img, sigma=sigma)
        else:
            smoothed = img
        
        # Find local maxima based on unclump method
        if unclump_method == UnclumpMethod.INTENSITY:
            maxima_image = smoothed
        else:  # SHAPE
            distance = ndi.distance_transform_edt(binary)
            maxima_image = distance
        
        # Find peaks
        min_distance = max(1, int(suppress_size))
        coordinates = peak_local_max(
            maxima_image,
            min_distance=min_distance,
            labels=labeled_image,
            exclude_border=False
        )
        
        # Create markers from peaks
        markers = np.zeros(img.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = i
        
        watershed_image = WatershedImageBuilder.for_method(watershed_method).build(
            img,
            binary,
        )
        
        # Apply watershed
        if markers.max() > 0:
            labeled_image = watershed(
                watershed_image,
                markers=markers,
                mask=binary,
                connectivity=2
            )
            object_count = labeled_image.max()
    
    # Fill holes after declumping if requested
    if fill_holes.fill_after_declump:
        labeled_image = _fill_labeled_holes(labeled_image)
    
    # Filter objects touching border
    if exclude_border_objects and object_count > 0:
        border_labels = set()
        border_labels.update(labeled_image[0, :].flatten())
        border_labels.update(labeled_image[-1, :].flatten())
        border_labels.update(labeled_image[:, 0].flatten())
        border_labels.update(labeled_image[:, -1].flatten())
        border_labels.discard(0)
        
        for lbl in border_labels:
            labeled_image[labeled_image == lbl] = 0
    
    # Filter objects by size
    if exclude_size and object_count > 0:
        min_area = np.pi * (min_diameter ** 2) / 4
        max_area = np.pi * (max_diameter ** 2) / 4
        
        props = regionprops(labeled_image)
        for prop in props:
            if prop.area < min_area or prop.area > max_area:
                labeled_image[labeled_image == prop.label] = 0
    
    # Relabel to ensure consecutive labels
    labeled_image, object_count = label(labeled_image > 0, return_num=True)
    
    # Check object count limit
    if limit_erase.erase_excess and object_count > maximum_object_count:
        labeled_image = np.zeros_like(labeled_image)
        object_count = 0
    
    # Calculate statistics
    if object_count > 0:
        props = regionprops(labeled_image)
        areas = [p.area for p in props]
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
        total_area = float(np.sum(areas))
    else:
        mean_area = 0.0
        median_area = 0.0
        total_area = 0.0
    
    stats = PrimaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        median_area=median_area,
        total_area=total_area,
        threshold_used=float(thresh)
    )
    
    return image, stats, labeled_image.astype(np.int32)
