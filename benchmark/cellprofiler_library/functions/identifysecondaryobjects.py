"""
Converted from CellProfiler: IdentifySecondaryObjects
Original: IdentifySecondaryObjects.run

Identifies secondary objects (e.g., cells) using primary objects (e.g., nuclei)
as seeds, expanding them based on intensity gradients or distance.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy


class SecondaryMethod(Enum):
    PROPAGATION = ("propagation", True)
    WATERSHED_GRADIENT = ("watershed_gradient", True)
    WATERSHED_IMAGE = ("watershed_image", True)
    DISTANCE_N = ("distance_n", False)
    DISTANCE_B = ("distance_b", True)

    def __new__(cls, value: str, requires_threshold: bool):
        method = object.__new__(cls)
        method._value_ = value
        method.requires_threshold = requires_threshold
        return method


class ThresholdMethod(Enum):
    OTSU = "otsu"
    LI = "li"
    MINIMUM = "minimum"
    TRIANGLE = "triangle"


@dataclass
class SecondaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    median_area: float
    total_area: int
    area_coverage_percent: float
    threshold_value: float


def _fill_labeled_holes(labels: np.ndarray) -> np.ndarray:
    """Fill holes in labeled objects."""
    from scipy.ndimage import binary_fill_holes
    
    filled = np.zeros_like(labels)
    for label_id in range(1, labels.max() + 1):
        mask = labels == label_id
        filled_mask = binary_fill_holes(mask)
        filled[filled_mask] = label_id
    return filled


def _propagate_labels(
    image: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    regularization: float
) -> np.ndarray:
    """Propagate labels using intensity-weighted distance.
    
    This is a simplified implementation of the propagation algorithm.
    Uses watershed with modified distance metric.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    
    if labels.max() == 0:
        return labels.copy()
    
    # Compute gradient magnitude for edge detection
    from scipy.ndimage import sobel
    gradient = np.abs(sobel(image, axis=0)) + np.abs(sobel(image, axis=1))
    
    # Combine distance and gradient information
    # Higher regularization = more weight on distance
    distance = distance_transform_edt(labels == 0)
    
    if regularization > 0:
        # Combine gradient and distance
        combined = gradient + regularization * distance
    else:
        combined = gradient
    
    # Use watershed to propagate labels
    result = watershed(combined, markers=labels, mask=mask)
    
    return result


@dataclass(frozen=True)
class SecondaryImageInputs:
    image: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class SecondaryThresholdResult:
    value: float
    mask: np.ndarray


@dataclass(frozen=True)
class SecondaryThresholdRequest:
    image: np.ndarray
    method: SecondaryMethod
    threshold_method: ThresholdMethod
    threshold_correction_factor: float
    threshold_min: float
    threshold_max: float


@dataclass(frozen=True)
class SecondarySegmentationRequest:
    image: np.ndarray
    labels: np.ndarray
    thresholded: np.ndarray
    distance_to_dilate: int
    regularization_factor: float

    @property
    def has_primary_objects(self) -> bool:
        return self.labels.max() > 0

    @property
    def object_mask(self) -> np.ndarray:
        return self.thresholded | (self.labels > 0)


class ThresholdCalculator(ABC, metaclass=AutoRegisterMeta):
    """Threshold strategy for one closed CellProfiler threshold method."""

    __registry_key__ = "method"
    method: ClassVar[ThresholdMethod | None] = None

    @classmethod
    def for_method(cls, method: ThresholdMethod) -> "ThresholdCalculator":
        return cls.__registry__[method]()

    @abstractmethod
    def calculate(self, image: np.ndarray) -> float:
        """Calculate a threshold value for a normalized intensity image."""


class OtsuThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.OTSU

    def calculate(self, image: np.ndarray) -> float:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(image))


class LiThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.LI

    def calculate(self, image: np.ndarray) -> float:
        from skimage.filters import threshold_li

        return float(threshold_li(image))


class MinimumThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.MINIMUM

    def calculate(self, image: np.ndarray) -> float:
        from skimage.filters import threshold_minimum, threshold_otsu

        try:
            return float(threshold_minimum(image))
        except RuntimeError:
            return float(threshold_otsu(image))


class TriangleThresholdCalculator(ThresholdCalculator):
    method = ThresholdMethod.TRIANGLE

    def calculate(self, image: np.ndarray) -> float:
        from skimage.filters import threshold_triangle

        return float(threshold_triangle(image))


class SecondarySegmentationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Segmentation strategy for one closed secondary-object method."""

    __registry_key__ = "method"
    method: ClassVar[SecondaryMethod | None] = None

    @classmethod
    def for_method(cls, method: SecondaryMethod) -> "SecondarySegmentationStrategy":
        return cls.__registry__[method]()

    def segment(self, request: SecondarySegmentationRequest) -> np.ndarray:
        if not request.has_primary_objects:
            return np.zeros_like(request.labels)
        return self._segment_non_empty(request)

    @abstractmethod
    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        """Segment secondary objects when primary labels are present."""


class DistanceOnlySegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_N

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        distances, indices = distance_transform_edt(
            request.labels == 0,
            return_indices=True,
        )
        labels_out = np.zeros_like(request.labels)
        dilate_mask = distances <= request.distance_to_dilate
        labels_out[dilate_mask] = request.labels[
            indices[0][dilate_mask],
            indices[1][dilate_mask],
        ]
        return labels_out


class DistanceMaskedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_B

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        labels_out = _propagate_labels(
            request.image,
            request.labels,
            request.object_mask,
            1.0,
        )
        distances = distance_transform_edt(request.labels == 0)
        labels_out[distances > request.distance_to_dilate] = 0
        labels_out[request.labels > 0] = request.labels[request.labels > 0]
        return labels_out


class PropagationSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.PROPAGATION

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return _propagate_labels(
            request.image,
            request.labels,
            request.object_mask,
            request.regularization_factor,
        )


class GradientWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_GRADIENT

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        from scipy.ndimage import sobel

        sobel_image = np.abs(sobel(request.image, axis=0)) + np.abs(
            sobel(request.image, axis=1)
        )
        return _watershed_secondary_labels(request, sobel_image)


class ImageWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_IMAGE

    def _segment_non_empty(
        self,
        request: SecondarySegmentationRequest,
    ) -> np.ndarray:
        return _watershed_secondary_labels(request, 1.0 - request.image)


def _watershed_secondary_labels(
    request: SecondarySegmentationRequest,
    watershed_image: np.ndarray,
) -> np.ndarray:
    from skimage.segmentation import watershed

    return watershed(
        watershed_image,
        markers=request.labels,
        mask=request.object_mask,
        connectivity=2,
    )


def _normalize_secondary_inputs(
    image: np.ndarray,
    primary_labels: np.ndarray,
) -> SecondaryImageInputs:
    if image.ndim == 3 and image.shape[0] == 2:
        return SecondaryImageInputs(
            image=image[0].astype(np.float64),
            labels=image[1].astype(np.int32),
        )
    return SecondaryImageInputs(
        image=image.astype(np.float64),
        labels=primary_labels.astype(np.int32),
    )


def _normalize_intensity_image(image: np.ndarray) -> np.ndarray:
    if image.max() > image.min():
        return (image - image.min()) / (image.max() - image.min())
    return image


def _threshold_secondary_objects(
    request: SecondaryThresholdRequest,
) -> SecondaryThresholdResult:
    if not request.method.requires_threshold:
        return SecondaryThresholdResult(
            value=0.0,
            mask=np.ones_like(request.image, dtype=bool),
        )

    threshold_value = ThresholdCalculator.for_method(
        request.threshold_method
    ).calculate(request.image)
    threshold_value = threshold_value * request.threshold_correction_factor
    threshold_value = max(
        request.threshold_min,
        min(request.threshold_max, threshold_value),
    )
    return SecondaryThresholdResult(
        value=threshold_value,
        mask=request.image > threshold_value,
    )


def _postprocess_secondary_labels(
    labels: np.ndarray,
    *,
    fill_holes: bool,
    discard_edge_objects: bool,
) -> np.ndarray:
    labels_out = labels
    if fill_holes and labels_out.max() > 0:
        labels_out = _fill_labeled_holes(labels_out)
    if discard_edge_objects and labels_out.max() > 0:
        labels_out = _discard_edge_objects(labels_out)
    return labels_out.astype(np.int32)


def _discard_edge_objects(labels: np.ndarray) -> np.ndarray:
    from skimage.measure import label as relabel

    edge_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    ]))
    labels_out = labels.copy()
    for edge_label in edge_labels:
        if edge_label > 0:
            labels_out[labels_out == edge_label] = 0

    if labels_out.max() == 0:
        return labels_out
    return relabel(labels_out > 0).astype(np.int32)


def _secondary_object_stats(
    labels: np.ndarray,
    *,
    image_shape: tuple[int, int],
    threshold_value: float,
) -> SecondaryObjectStats:
    from skimage.measure import regionprops

    object_count = int(labels.max())
    if object_count > 0:
        areas = [p.area for p in regionprops(labels)]
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
        total_area = int(np.sum(areas))
    else:
        mean_area = 0.0
        median_area = 0.0
        total_area = 0

    height, width = image_shape
    area_coverage = 100.0 * total_area / (height * width) if height * width else 0.0
    return SecondaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        median_area=median_area,
        total_area=total_area,
        area_coverage_percent=area_coverage,
        threshold_value=float(threshold_value),
    )


@numpy
def identify_secondary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray,
    method: SecondaryMethod = SecondaryMethod.PROPAGATION,
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    distance_to_dilate: int = 10,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    discard_edge_objects: bool = False,
) -> Tuple[np.ndarray, SecondaryObjectStats, np.ndarray]:
    """
    Identify secondary objects using primary objects as seeds.
    
    Args:
        image: Input intensity image, shape (2, H, W) where [0] is intensity, [1] is primary labels
               OR shape (H, W) if primary_labels provided separately
        primary_labels: Label image of primary objects (seeds)
        method: Method for identifying secondary objects
        threshold_method: Method for thresholding the image
        threshold_correction_factor: Factor to multiply threshold by
        threshold_min: Minimum threshold value
        threshold_max: Maximum threshold value  
        distance_to_dilate: Pixels to expand for distance methods
        regularization_factor: Lambda for propagation method (0=gradient only, higher=more distance)
        fill_holes: Whether to fill holes in identified objects
        discard_edge_objects: Whether to discard objects touching image border
        
    Returns:
        Tuple of (image, stats, secondary_labels)
    """
    inputs = _normalize_secondary_inputs(image, primary_labels)
    img = _normalize_intensity_image(inputs.image)
    threshold = _threshold_secondary_objects(
        SecondaryThresholdRequest(
            image=img,
            method=method,
            threshold_method=threshold_method,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    )
    labels_out = SecondarySegmentationStrategy.for_method(method).segment(
        SecondarySegmentationRequest(
            image=img,
            labels=inputs.labels,
            thresholded=threshold.mask,
            distance_to_dilate=distance_to_dilate,
            regularization_factor=regularization_factor,
        )
    )
    labels_out = _postprocess_secondary_labels(
        labels_out,
        fill_holes=fill_holes,
        discard_edge_objects=discard_edge_objects,
    )
    stats = _secondary_object_stats(
        labels_out,
        image_shape=img.shape,
        threshold_value=threshold.value,
    )
    
    return img.astype(np.float32), stats, labels_out
