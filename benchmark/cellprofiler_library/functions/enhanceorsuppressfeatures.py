"""Converted from CellProfiler: EnhanceOrSuppressFeatures."""

from enum import Enum

import numpy as np
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.transform

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class OperationMethod(Enum):
    ENHANCE = "Enhance"
    SUPPRESS = "Suppress"


class EnhanceMethod(Enum):
    SPECKLES = "Speckles"
    NEURITES = "Neurites"
    DARK_HOLES = "Dark holes"
    CIRCLES = "Circles"
    TEXTURE = "Texture"
    DIC = "DIC"


class SpeckleAccuracy(Enum):
    FAST = "Fast"
    SLOW = "Slow"


class NeuriteMethod(Enum):
    GRADIENT = "Line structures"
    TUBENESS = "Tubeness"


@numpy(contract=ProcessingContract.PURE_2D)
def enhance_or_suppress_features(
    image: np.ndarray,
    method: OperationMethod = OperationMethod.ENHANCE,
    enhance_method: EnhanceMethod = EnhanceMethod.SPECKLES,
    radius: float = 10.0,
    speckle_accuracy: SpeckleAccuracy = SpeckleAccuracy.FAST,
    neurite_method: NeuriteMethod = NeuriteMethod.GRADIENT,
    neurite_rescale: bool = False,
    dark_hole_radius_min: int = 1,
    dark_hole_radius_max: int = 10,
    smoothing_value: float = 2.0,
    dic_angle: float = 0.0,
    dic_decay: float = 0.95,
) -> np.ndarray:
    """Enhance or suppress image features using independent CP-compatible semantics."""
    method = _coerce_function_enum(OperationMethod, method)
    enhance_method = _coerce_function_enum(EnhanceMethod, enhance_method)
    speckle_accuracy = _coerce_function_enum(SpeckleAccuracy, speckle_accuracy)
    neurite_method = _coerce_function_enum(NeuriteMethod, neurite_method)
    image_data = np.asarray(image_payload_data(image))
    if image_data.dtype != np.float32 and image_data.dtype != np.float64:
        image_data = image_data.astype(np.float32)
    mask = _enhancement_mask(image, image_data)

    if method == OperationMethod.ENHANCE:
        result = _enhance(
            image_data,
            mask,
            enhance_method=enhance_method,
            radius=radius,
            speckle_accuracy=speckle_accuracy,
            neurite_method=neurite_method,
            neurite_rescale=neurite_rescale,
            dark_hole_radius_min=dark_hole_radius_min,
            dark_hole_radius_max=dark_hole_radius_max,
            smoothing_value=smoothing_value,
            dic_angle=dic_angle,
            dic_decay=dic_decay,
        )
    elif method == OperationMethod.SUPPRESS:
        result = _suppress_features(
            image_data,
            mask,
            radius,
        )
    else:
        raise ValueError(f"Unknown filtering method: {method}")

    return image_payload_with_context(
        np.asarray(result, dtype=np.float32),
        mask=image_payload_mask(image),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )


def _enhance(
    image_data: np.ndarray,
    mask: np.ndarray,
    *,
    enhance_method: EnhanceMethod,
    radius: float,
    speckle_accuracy: SpeckleAccuracy,
    neurite_method: NeuriteMethod,
    neurite_rescale: bool,
    dark_hole_radius_min: int,
    dark_hole_radius_max: int,
    smoothing_value: float,
    dic_angle: float,
    dic_decay: float,
) -> np.ndarray:
    if enhance_method == EnhanceMethod.SPECKLES:
        return _enhance_speckles(
            image_data,
            mask,
            radius,
            speckle_accuracy,
        )
    if enhance_method == EnhanceMethod.NEURITES:
        return _enhance_neurites(
            image_data,
            mask,
            smoothing_value,
            radius,
            neurite_method,
            neurite_rescale,
        )
    if enhance_method == EnhanceMethod.DARK_HOLES:
        return _enhance_dark_holes(
            image_data,
            mask,
            dark_hole_radius_min,
            dark_hole_radius_max,
        )
    if enhance_method == EnhanceMethod.CIRCLES:
        return _enhance_circles(
            image_data,
            mask,
            radius,
        )
    if enhance_method == EnhanceMethod.TEXTURE:
        return _enhance_texture(
            image_data,
            mask,
            smoothing_value,
        )
    if enhance_method == EnhanceMethod.DIC:
        return _enhance_dic(
            image_data,
            mask,
            dic_angle,
            dic_decay,
            smoothing_value,
        )
    raise NotImplementedError(f"Unimplemented enhance method: {enhance_method}")


def _enhancement_mask(image: object, image_data: np.ndarray) -> np.ndarray:
    mask = image_payload_mask(image)
    if mask is None:
        return np.ones(image_data.shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _structuring_element(radius: float) -> np.ndarray:
    return skimage.morphology.disk(max(1, int(round(radius))))


def _masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, image, 0)


def _restore_masked_background(
    result: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    output = np.asarray(result, dtype=np.float32).copy()
    output[~mask] = original[~mask]
    return output


def _suppress_features(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    footprint = _structuring_element(radius)
    opened = skimage.morphology.opening(_masked_image(image_data, mask), footprint=footprint)
    return _restore_masked_background(opened, image_data, mask)


def _enhance_speckles(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius: float,
    speckle_accuracy: SpeckleAccuracy,
) -> np.ndarray:
    footprint = _structuring_element(radius)
    masked = _masked_image(image_data, mask)
    if speckle_accuracy is SpeckleAccuracy.FAST and radius > 3:
        opened = scipy.ndimage.maximum_filter(
            scipy.ndimage.minimum_filter(masked, footprint=footprint),
            footprint=footprint,
        )
        result = masked - opened
    else:
        result = skimage.morphology.white_tophat(masked, footprint=footprint)
    return _restore_masked_background(result, image_data, mask)


def _enhance_neurites(
    image_data: np.ndarray,
    mask: np.ndarray,
    smoothing_value: float,
    radius: float,
    neurite_method: NeuriteMethod,
    neurite_rescale: bool,
) -> np.ndarray:
    masked = _masked_image(image_data, mask)
    if neurite_method is NeuriteMethod.TUBENESS:
        smoothed = skimage.filters.gaussian(masked, sigma=smoothing_value)
        result = skimage.filters.sato(
            smoothed,
            sigmas=range(1, max(2, int(round(radius))) + 1),
            black_ridges=False,
        )
    else:
        footprint = _structuring_element(radius)
        result = (
            masked
            + skimage.morphology.white_tophat(masked, footprint=footprint)
            - skimage.morphology.black_tophat(masked, footprint=footprint)
        )
        result = np.clip(result, 0, None)
    if neurite_rescale:
        result = skimage.exposure.rescale_intensity(result, out_range=(0.0, 1.0))
    return _restore_masked_background(result, image_data, mask)


def _enhance_dark_holes(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius_min: int,
    radius_max: int,
) -> np.ndarray:
    masked = _masked_image(image_data, mask)
    radii = range(max(1, radius_min), max(radius_min, radius_max) + 1)
    responses = [
        skimage.morphology.black_tophat(masked, footprint=_structuring_element(radius))
        for radius in radii
    ]
    result = np.maximum.reduce(responses) if responses else np.zeros_like(masked)
    return _restore_masked_background(result, image_data, mask)


def _enhance_circles(
    image_data: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    masked = _masked_image(image_data, mask)
    radius_i = max(1, int(round(radius)))
    result = skimage.transform.hough_circle(masked, [radius_i])[0]
    return _restore_masked_background(result, image_data, mask)


def _enhance_texture(
    image_data: np.ndarray,
    mask: np.ndarray,
    smoothing_value: float,
) -> np.ndarray:
    masked = _masked_image(image_data, mask).astype(float)
    mean = scipy.ndimage.gaussian_filter(masked, smoothing_value)
    mean_squared = scipy.ndimage.gaussian_filter(masked * masked, smoothing_value)
    result = np.maximum(mean_squared - mean * mean, 0)
    return _restore_masked_background(result, image_data, mask)


def _enhance_dic(
    image_data: np.ndarray,
    mask: np.ndarray,
    angle: float,
    decay: float,
    smoothing_value: float,
) -> np.ndarray:
    smoothed = scipy.ndimage.gaussian_filter(_masked_image(image_data, mask), smoothing_value)
    radians = np.deg2rad(angle)
    shift = np.array((np.sin(radians), np.cos(radians))) * max(decay, 0)
    coords = np.indices(smoothed.shape, dtype=float)
    forward = scipy.ndimage.map_coordinates(
        smoothed,
        coords + shift.reshape(2, 1, 1),
        order=1,
        mode="nearest",
    )
    backward = scipy.ndimage.map_coordinates(
        smoothed,
        coords - shift.reshape(2, 1, 1),
        order=1,
        mode="nearest",
    )
    result = np.maximum(forward - backward, 0)
    return _restore_masked_background(result, image_data, mask)
