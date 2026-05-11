"""
Converted from CellProfiler: MeasureTexture
Original: MeasureTexture module

Measures Haralick texture features from grayscale images.
These features quantify the degree and nature of textures within images
and objects to characterize roughness and smoothness.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_schemas import (
    DataclassCompanionSchema,
    DataclassFieldInsertion,
)
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


# Haralick feature names
F_HARALICK = [
    "AngularSecondMoment", "Contrast", "Correlation", "Variance",
    "InverseDifferenceMoment", "SumAverage", "SumVariance", "SumEntropy",
    "Entropy", "DifferenceVariance", "DifferenceEntropy", "InfoMeas1", "InfoMeas2"
]

N_DIRECTIONS_2D = 4


@dataclass
class TextureMeasurement:
    """Texture measurement results for a single slice/image."""
    slice_index: int
    scale: int
    direction: int
    gray_levels: int
    angular_second_moment: float
    contrast: float
    correlation: float
    variance: float
    inverse_difference_moment: float
    sum_average: float
    sum_variance: float
    sum_entropy: float
    entropy: float
    difference_variance: float
    difference_entropy: float
    info_meas1: float
    info_meas2: float
    source_image_name: str | None = None


ObjectTextureMeasurement = DataclassCompanionSchema(
    source_type=TextureMeasurement,
    companion_name="ObjectTextureMeasurement",
    insertions=(
        DataclassFieldInsertion("object_label", int, after_field="slice_index"),
    ),
    module_name=__name__,
    doc="Texture measurement results per object.",
).materialize()


def _normalize_gray_levels(gray_levels: int) -> int:
    return max(2, min(256, int(gray_levels)))


def _texture_scales(scale: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(scale, (tuple, list)):
        return tuple(int(value) for value in scale)
    return (int(scale),)


def _cellprofiler_pixel_data(image: np.ndarray, gray_levels: int) -> np.ndarray:
    """Quantize image data the same way CellProfiler MeasureTexture does."""
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte

    pixel_data = image.copy() if image.dtype == np.uint8 else img_as_ubyte(image)
    if gray_levels != 256:
        pixel_data = rescale_intensity(
            pixel_data,
            in_range=(0, 255),
            out_range=(0, gray_levels - 1),
        ).astype(np.uint8)
    return pixel_data


def _zero_feature_matrix() -> np.ndarray:
    return np.zeros((N_DIRECTIONS_2D, len(F_HARALICK)), dtype=float)


def _clean_feature_vector(features: np.ndarray) -> np.ndarray:
    clean = np.asarray(features, dtype=float).copy()
    clean[~np.isfinite(clean)] = 0
    return clean


def _haralick_feature_matrix(
    pixel_data: np.ndarray,
    *,
    scale: int,
    ignore_zeros: bool,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Return CP-compatible Haralick rows using an explicit backend."""
    from openhcs.processing.backends.cellprofiler.texture import (
        HaralickTextureBackendStrategy,
    )

    pixel_data = np.asarray(pixel_data)
    if not _haralick_has_valid_domain(
        pixel_data,
        scale=scale,
        ignore_zeros=ignore_zeros,
    ):
        return _zero_feature_matrix()

    backend = HaralickTextureBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )
    return np.asarray(
        backend.haralick_features(
            pixel_data,
            scale=scale,
            ignore_zeros=ignore_zeros,
        ),
        dtype=float,
    )


def _haralick_has_valid_domain(
    pixel_data: np.ndarray,
    *,
    scale: int,
    ignore_zeros: bool,
) -> bool:
    if pixel_data.ndim != 2:
        raise ValueError(
            "MeasureTexture expects a 2D image plane. Stack dispatch must be "
            "handled by the OpenHCS processing contract."
        )
    if scale < 1:
        raise ValueError(f"MeasureTexture scale must be positive, got {scale}.")
    if pixel_data.shape[0] <= scale or pixel_data.shape[1] <= scale:
        return False
    if not ignore_zeros:
        return True
    nonzero = pixel_data != 0
    return _has_nonzero_haralick_pairs(nonzero, scale)


def _has_nonzero_haralick_pairs(nonzero: np.ndarray, scale: int) -> bool:
    return (
        np.any(nonzero[:, :-scale] & nonzero[:, scale:])
        and np.any(nonzero[:-scale, :-scale] & nonzero[scale:, scale:])
        and np.any(nonzero[:-scale, :] & nonzero[scale:, :])
        and np.any(nonzero[:-scale, scale:] & nonzero[scale:, :-scale])
    )


def _feature_row(feature_matrix: np.ndarray, direction: int) -> np.ndarray:
    if direction >= feature_matrix.shape[0]:
        return np.zeros((len(F_HARALICK),), dtype=float)
    return _clean_feature_vector(feature_matrix[direction, :])


def _texture_measurement(
    *,
    scale: int,
    direction: int,
    gray_levels: int,
    features: np.ndarray,
) -> TextureMeasurement:
    return TextureMeasurement(
        slice_index=0,
        scale=scale,
        direction=direction,
        gray_levels=gray_levels,
        angular_second_moment=float(features[0]),
        contrast=float(features[1]),
        correlation=float(features[2]),
        variance=float(features[3]),
        inverse_difference_moment=float(features[4]),
        sum_average=float(features[5]),
        sum_variance=float(features[6]),
        sum_entropy=float(features[7]),
        entropy=float(features[8]),
        difference_variance=float(features[9]),
        difference_entropy=float(features[10]),
        info_meas1=float(features[11]),
        info_meas2=float(features[12]),
    )


def _object_texture_measurement(
    *,
    object_label: int,
    scale: int,
    direction: int,
    gray_levels: int,
    features: np.ndarray,
) -> ObjectTextureMeasurement:
    return ObjectTextureMeasurement(
        slice_index=0,
        object_label=object_label,
        scale=scale,
        direction=direction,
        gray_levels=gray_levels,
        angular_second_moment=float(features[0]),
        contrast=float(features[1]),
        correlation=float(features[2]),
        variance=float(features[3]),
        inverse_difference_moment=float(features[4]),
        sum_average=float(features[5]),
        sum_variance=float(features[6]),
        sum_entropy=float(features[7]),
        entropy=float(features[8]),
        difference_variance=float(features[9]),
        difference_entropy=float(features[10]),
        info_meas1=float(features[11]),
        info_meas2=float(features[12]),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("texture_measurements", csv_materializer(
    fields=["slice_index", "scale", "direction", "gray_levels",
            "angular_second_moment", "contrast", "correlation", "variance",
            "inverse_difference_moment", "sum_average", "sum_variance", 
            "sum_entropy", "entropy", "difference_variance", "difference_entropy",
            "info_meas1", "info_meas2", "source_image_name"],
    analysis_type="texture"
)))
def measure_texture(
    image: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, List[TextureMeasurement]]:
    """
    Measure Haralick texture features on a grayscale image.
    
    Computes 13 Haralick texture features derived from the gray-level
    co-occurrence matrix (GLCM) at the specified scale.
    
    Args:
        image: Input grayscale image (H, W), values in [0, 1]
        scale: Distance in pixels for GLCM computation (default: 3)
        gray_levels: Number of gray levels for quantization (2-256, default: 256)
    
    Returns:
        Tuple of (original image, list of TextureMeasurement for each direction)
    """
    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = _cellprofiler_pixel_data(image, gray_levels)

    measurements = []
    for texture_scale in _texture_scales(scale):
        feature_matrix = _haralick_feature_matrix(
            pixel_data,
            scale=texture_scale,
            ignore_zeros=False,
            backend_provider=haralick_backend_provider,
        )

        for direction in range(N_DIRECTIONS_2D):
            measurements.append(
                _texture_measurement(
                    scale=texture_scale,
                    direction=direction,
                    gray_levels=gray_levels,
                    features=_feature_row(feature_matrix, direction),
                )
            )

    return image, measurements


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("object_texture_measurements", csv_materializer(
    fields=["slice_index", "object_label", "scale", "direction", "gray_levels",
            "angular_second_moment", "contrast", "correlation", "variance",
            "inverse_difference_moment", "sum_average", "sum_variance", 
            "sum_entropy", "entropy", "difference_variance", "difference_entropy",
            "info_meas1", "info_meas2", "source_image_name"],
    analysis_type="object_texture"
)))
def measure_texture_objects(
    image: np.ndarray,
    labels: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    texture_crop_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, List[ObjectTextureMeasurement]]:
    """
    Measure Haralick texture features for each labeled object.
    
    Computes 13 Haralick texture features for each object in the label image,
    derived from the gray-level co-occurrence matrix (GLCM) at the specified scale.
    
    Args:
        image: Input grayscale image (H, W), values in [0, 1]
        labels: Label image with integer object labels (H, W)
        scale: Distance in pixels for GLCM computation (default: 3)
        gray_levels: Number of gray levels for quantization (2-256, default: 256)
    
    Returns:
        Tuple of (original image, list of ObjectTextureMeasurement for each object/direction)
    """
    from openhcs.processing.backends.cellprofiler.texture import (
        ObjectTextureCropBackendStrategy,
    )

    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = _cellprofiler_pixel_data(image, gray_levels)
    crop_backend = ObjectTextureCropBackendStrategy.for_callable(
        measure_texture_objects,
        backend_provider=texture_crop_backend_provider,
    )

    measurements = []

    object_labels, intensity_crops = crop_backend.object_intensity_crops(
        pixel_data,
        labels,
    )
    if object_labels.size == 0:
        return image, measurements

    for object_label, label_data in zip(object_labels, intensity_crops, strict=True):
        for texture_scale in _texture_scales(scale):
            feature_matrix = _haralick_feature_matrix(
                label_data,
                scale=texture_scale,
                ignore_zeros=True,
                backend_provider=haralick_backend_provider,
            )

            for direction in range(N_DIRECTIONS_2D):
                measurements.append(
                    _object_texture_measurement(
                        object_label=int(object_label),
                        scale=texture_scale,
                        direction=direction,
                        gray_levels=gray_levels,
                        features=_feature_row(feature_matrix, direction),
                    )
                )

    return image, measurements


def _prepare_measure_texture() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    measure_texture.__wrapped__(image)


def _prepare_measure_texture_objects() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_texture_objects.__wrapped__(image, labels)


measure_texture.__openhcs_prepare__ = _prepare_measure_texture
measure_texture_objects.__openhcs_prepare__ = _prepare_measure_texture_objects
