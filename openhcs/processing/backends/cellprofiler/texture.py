"""Texture-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import numpy as np
from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    pure_2d_batch_executor,
    special_inputs,
    special_outputs,
)
from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest
from openhcs.core.runtime_semantics import dense_object_label_id_domain
from openhcs.core.runtime_values import (
    DenseObjectLabelPlaneDomainStackRequest,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelSourcePlaneProjectionRequest,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


F_HARALICK = [
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "Variance",
    "InverseDifferenceMoment",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeas1",
    "InfoMeas2",
]

N_DIRECTIONS_2D = 4
ObjectIntensityCrops = tuple[np.ndarray, tuple[np.ndarray, ...]]
TextureLabelSource: TypeAlias = ObjectLabelValue | np.ndarray | None
ObjectTextureResult: TypeAlias = tuple[np.ndarray, list["ObjectTextureMeasurement"]]


@dataclass(frozen=True, slots=True)
class HaralickFeatureColumn:
    """Descriptor exposing one Haralick vector coordinate as a row column."""

    feature_index: int

    def __get__(
        self,
        instance: "HaralickFeatureColumns | None",
        owner: type["HaralickFeatureColumns"] | None = None,
    ) -> float | "HaralickFeatureColumn":
        del owner
        if instance is None:
            return self
        return float(instance.features.values[self.feature_index])


class TextureAxisColumns:
    """Output-column aliases owned by the texture axis carrier."""

    slice_index: ClassVar[AliasProperty[int]] = AliasProperty("axis.slice_index")
    scale: ClassVar[AliasProperty[int]] = AliasProperty("axis.scale")
    direction: ClassVar[AliasProperty[int]] = AliasProperty("axis.direction")
    gray_levels: ClassVar[AliasProperty[int]] = AliasProperty("axis.gray_levels")


class HaralickFeatureColumns:
    """Output-column aliases owned by the Haralick feature vector."""

    angular_second_moment: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(0)
    contrast: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(1)
    correlation: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(2)
    variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(3)
    inverse_difference_moment: ClassVar[HaralickFeatureColumn] = (
        HaralickFeatureColumn(4)
    )
    sum_average: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(5)
    sum_variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(6)
    sum_entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(7)
    entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(8)
    difference_variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(9)
    difference_entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(10)
    info_meas1: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(11)
    info_meas2: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(12)


@dataclass
class TextureMeasurement(TextureAxisColumns, HaralickFeatureColumns):
    """Texture measurement results for a single slice/image."""

    axis: "TextureMeasurementAxis"
    features: "HaralickFeatureVector"
    source_image_name: str | None = None


@dataclass
class ObjectTextureMeasurement(TextureAxisColumns, HaralickFeatureColumns):
    """Texture measurement results per object."""

    object_label: int
    axis: "TextureMeasurementAxis"
    features: "HaralickFeatureVector"
    source_image_name: str | None = None


class ObjectTextureCropBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Extract masked object intensity crops for texture measurement."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityCrops:
        """Return positive object labels and CP-style masked intensity crops."""


class HaralickTextureBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Compute CP-compatible 2-D Haralick feature matrices."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        """Return one Haralick feature row per 2-D direction."""


class NumbaNumpyObjectTextureCropBackendStrategy(ObjectTextureCropBackendStrategy):
    """Numba-accelerated NumPy backend for object texture crop extraction."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        labels = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]], dtype=np.int64)
        self.object_intensity_crops(image, labels)

    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityCrops:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "Numba texture crop backend currently supports 2-D NumPy planes."
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "Texture image and labels must have identical shapes; got "
                f"{image_array.shape!r} and {labels_array.shape!r}."
            )
        object_labels, boxes = _object_bounding_boxes_numba(
            np.ascontiguousarray(labels_array, dtype=np.int64)
        )
        crops: list[np.ndarray] = []
        for index, object_label in enumerate(object_labels):
            y0, y1, x0, x1 = boxes[index]
            label_crop = labels_array[y0:y1, x0:x1]
            intensity_crop = np.asarray(image_array[y0:y1, x0:x1]).copy()
            intensity_crop[label_crop != object_label] = 0
            crops.append(intensity_crop)
        return object_labels.astype(np.int64, copy=False), tuple(crops)


class NumbaNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Numba implementation of mahotas' default 2-D Haralick semantics."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(25, dtype=np.int64).reshape((5, 5))
        self.haralick_features(image, scale=1, ignore_zeros=False)

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        pixel_array = np.ascontiguousarray(pixel_data)
        if pixel_array.ndim != 2:
            raise ValueError("Haralick texture backend expects a 2-D image plane.")
        if scale < 1:
            raise ValueError(f"Haralick texture scale must be positive, got {scale}.")
        if pixel_array.shape[0] <= scale or pixel_array.shape[1] <= scale:
            return np.zeros((4, 13), dtype=np.float64)
        return _haralick_2d_features_numba(
            pixel_array.astype(np.int64, copy=False),
            int(scale),
            bool(ignore_zeros),
        )


class NativeNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Explicit mahotas backend used as the native reference implementation."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        import mahotas.features as mahotas_features

        return np.asarray(
            mahotas_features.haralick(
                np.asarray(pixel_data),
                distance=scale,
                ignore_zeros=ignore_zeros,
            ),
            dtype=np.float64,
        )


def _normalize_gray_levels(gray_levels: int) -> int:
    return max(2, min(256, int(gray_levels)))


def _texture_scales(scale: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(scale, (tuple, list)):
        return tuple(int(value) for value in scale)
    return (int(scale),)


@dataclass(frozen=True, slots=True)
class CellProfilerTexturePixelDataRequest:
    """Quantize image data the same way CellProfiler MeasureTexture does."""

    image: np.ndarray
    gray_levels: int

    def pixel_data(self) -> np.ndarray:
        from skimage.exposure import rescale_intensity
        from skimage.util import img_as_ubyte

        pixel_data = (
            self.image.copy()
            if self.image.dtype == np.uint8
            else img_as_ubyte(self.image)
        )
        if self.gray_levels != 256:
            pixel_data = rescale_intensity(
                pixel_data,
                in_range=(0, 255),
                out_range=(0, self.gray_levels - 1),
            ).astype(np.uint8)
        return pixel_data


def _zero_feature_matrix() -> np.ndarray:
    return np.zeros((N_DIRECTIONS_2D, len(F_HARALICK)), dtype=float)


def _clean_feature_vector(features: np.ndarray) -> np.ndarray:
    clean = np.asarray(features, dtype=float).copy()
    clean[~np.isfinite(clean)] = 0
    return clean


@dataclass(frozen=True, slots=True)
class HaralickFeatureMatrixRequest:
    """Request for CP-compatible Haralick rows using an explicit backend."""

    pixel_data: np.ndarray
    scale: int
    ignore_zeros: bool
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION

    def feature_matrix(self) -> np.ndarray:
        pixel_data = np.asarray(self.pixel_data)
        if not _haralick_has_valid_domain(
            pixel_data,
            scale=self.scale,
            ignore_zeros=self.ignore_zeros,
        ):
            return _zero_feature_matrix()

        backend = HaralickTextureBackendStrategy.for_memory_type(
            backend_provider=self.backend_provider,
        )
        return np.asarray(
            backend.haralick_features(
                pixel_data,
                scale=self.scale,
                ignore_zeros=self.ignore_zeros,
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


@dataclass(frozen=True, slots=True)
class TextureMeasurementAxis:
    """Scale, direction, and gray-level coordinates for one Haralick row."""

    slice_index: int
    scale: int
    direction: int
    gray_levels: int

    def object_key(self, object_label: int) -> tuple[int, int, int, int, int]:
        return (
            object_label,
            self.slice_index,
            self.scale,
            self.direction,
            self.gray_levels,
        )


@dataclass(frozen=True, slots=True)
class HaralickFeatureVector:
    """Cleaned Haralick feature row with constructors for output records."""

    values: np.ndarray

    @classmethod
    def from_matrix(
        cls,
        feature_matrix: np.ndarray,
        direction: int,
    ) -> "HaralickFeatureVector":
        if direction >= feature_matrix.shape[0]:
            return cls.zeros()
        return cls(_clean_feature_vector(feature_matrix[direction, :]))

    @classmethod
    def zeros(cls) -> "HaralickFeatureVector":
        return cls(np.zeros((len(F_HARALICK),), dtype=float))

    def image_measurement(self, axis: TextureMeasurementAxis) -> TextureMeasurement:
        return TextureMeasurement(
            axis=axis,
            features=self,
        )

    def object_measurement(
        self,
        axis: TextureMeasurementAxis,
        *,
        object_label: int,
    ) -> ObjectTextureMeasurement:
        return ObjectTextureMeasurement(
            object_label=object_label,
            axis=axis,
            features=self,
        )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "texture_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "scale",
                "direction",
                "gray_levels",
                "angular_second_moment",
                "contrast",
                "correlation",
                "variance",
                "inverse_difference_moment",
                "sum_average",
                "sum_variance",
                "sum_entropy",
                "entropy",
                "difference_variance",
                "difference_entropy",
                "info_meas1",
                "info_meas2",
                "source_image_name",
            ],
            analysis_type="texture",
        ),
    )
)
def measure_texture(
    image: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, list[TextureMeasurement]]:
    """Measure Haralick texture features on a grayscale image."""
    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image,
        gray_levels=gray_levels,
    ).pixel_data()

    measurements = []
    for texture_scale in _texture_scales(scale):
        feature_matrix = HaralickFeatureMatrixRequest(
            pixel_data=pixel_data,
            scale=texture_scale,
            ignore_zeros=False,
            backend_provider=haralick_backend_provider,
        ).feature_matrix()

        for direction in range(N_DIRECTIONS_2D):
            axis = TextureMeasurementAxis(
                slice_index=0,
                scale=texture_scale,
                direction=direction,
                gray_levels=gray_levels,
            )
            measurements.append(
                HaralickFeatureVector.from_matrix(
                    feature_matrix,
                    direction,
                ).image_measurement(
                    axis,
                )
            )

    return image, measurements


@dataclass(frozen=True, slots=True)
class ObjectTextureMeasurementCompletionRequest:
    """Fill missing per-object texture rows for the declared label domain."""

    measurements: tuple[ObjectTextureMeasurement, ...]
    labels: TextureLabelSource
    scale: int | tuple[int, ...] | list[int]
    gray_levels: int

    def complete(self) -> list[ObjectTextureMeasurement]:
        object_domain = dense_object_label_id_domain(self.labels)
        if not object_domain:
            return list(self.measurements)

        by_key = {
            measurement.axis.object_key(measurement.object_label): measurement
            for measurement in self.measurements
        }
        complete: list[ObjectTextureMeasurement] = []
        zero_features = HaralickFeatureVector.zeros()
        axes = self.axes
        for object_label in object_domain:
            for axis in axes:
                key = axis.object_key(object_label)
                if key in by_key:
                    complete.append(by_key[key])
                    continue
                complete.append(
                    zero_features.object_measurement(
                        axis,
                        object_label=object_label,
                    )
                )
        return complete

    @property
    def axes(self) -> tuple[TextureMeasurementAxis, ...]:
        axes = tuple(
            dict.fromkeys(
                measurement.axis
                for measurement in self.measurements
            )
        )
        if axes:
            return axes
        return tuple(
            TextureMeasurementAxis(
                slice_index=0,
                scale=texture_scale,
                direction=direction,
                gray_levels=self.gray_levels,
            )
            for texture_scale in _texture_scales(self.scale)
            for direction in range(N_DIRECTIONS_2D)
        )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "object_texture_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "object_label",
                "scale",
                "direction",
                "gray_levels",
                "angular_second_moment",
                "contrast",
                "correlation",
                "variance",
                "inverse_difference_moment",
                "sum_average",
                "sum_variance",
                "sum_entropy",
                "entropy",
                "difference_variance",
                "difference_entropy",
                "info_meas1",
                "info_meas2",
                "source_image_name",
            ],
            analysis_type="object_texture",
        ),
    )
)
def measure_texture_objects(
    image: np.ndarray,
    labels: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    texture_crop_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int = 0,
) -> tuple[np.ndarray, list[ObjectTextureMeasurement]]:
    """Measure Haralick texture features for each labeled object."""
    image_array = np.asarray(image)
    original_labels = labels
    if image_array.ndim == 2 and slice_index == 0:
        plane_domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            labels,
            dtype=np.int32,
        ).stack()
        if plane_domain_stack is not None:
            measurements: list[ObjectTextureMeasurement] = []
            for plane_index in range(plane_domain_stack.plane_count):
                _image, plane_measurements = measure_texture_objects.__wrapped__(
                    image,
                    plane_domain_stack.plane(plane_index),
                    scale=scale,
                    gray_levels=gray_levels,
                    texture_crop_backend_provider=texture_crop_backend_provider,
                    haralick_backend_provider=haralick_backend_provider,
                    slice_index=plane_index,
                )
                measurements.extend(plane_measurements)
            return image, measurements

    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image,
        gray_levels=gray_levels,
    ).pixel_data()
    crop_backend = ObjectTextureCropBackendStrategy.for_callable(
        measure_texture_objects,
        backend_provider=texture_crop_backend_provider,
    )

    measurements = []
    label_projection = TextureLabelSliceProjection.from_source(
        labels,
        np.asarray(image),
        slice_index,
    )
    labels_2d = label_projection.labels_2d()
    if labels_2d is not None:
        labels = labels_2d
    object_labels, intensity_crops = crop_backend.object_intensity_crops(
        pixel_data,
        labels,
    )
    if object_labels.size == 0:
        return image, ObjectTextureMeasurementCompletionRequest(
            measurements=tuple(measurements),
            labels=original_labels,
            scale=scale,
            gray_levels=gray_levels,
        ).complete()

    for object_label, label_data in zip(object_labels, intensity_crops, strict=True):
        for texture_scale in _texture_scales(scale):
            feature_matrix = HaralickFeatureMatrixRequest(
                pixel_data=label_data,
                scale=texture_scale,
                ignore_zeros=True,
                backend_provider=haralick_backend_provider,
            ).feature_matrix()

            for direction in range(N_DIRECTIONS_2D):
                axis = TextureMeasurementAxis(
                    slice_index=slice_index,
                    scale=texture_scale,
                    direction=direction,
                    gray_levels=gray_levels,
                )
                measurements.append(
                    HaralickFeatureVector.from_matrix(
                        feature_matrix,
                        direction,
                    ).object_measurement(
                        axis,
                        object_label=int(object_label),
                    )
                )

    return image, ObjectTextureMeasurementCompletionRequest(
        measurements=tuple(measurements),
        labels=original_labels,
        scale=scale,
        gray_levels=gray_levels,
    ).complete()


def measure_texture_objects_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[ObjectTextureResult]:
    """Measure per-slice object texture with labels projected to each image plane."""
    kwargs = request.kwargs
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = None
    label_array = TextureDenseLabelArray.from_value(labels)
    results: list[ObjectTextureResult] = []
    for slice_index, slice_2d in enumerate(request.slices_2d):
        slice_kwargs = kwargs
        label_projection = TextureLabelSliceProjection(
            source=labels,
            dense_labels=label_array,
            slice_array=np.asarray(slice_2d),
            slice_index=slice_index,
        )
        labels_2d = label_projection.labels_2d()
        if labels_2d is not None:
            slice_kwargs = dict(kwargs)
            slice_kwargs["labels"] = label_projection.projected_payload(labels_2d)
        results.append(request.execute_one_with_kwargs(slice_index, slice_kwargs))
    return results


class TextureDenseLabelArray:
    """Dense label coercion for MeasureTexture object inputs."""

    @classmethod
    def from_value(cls, labels: TextureLabelSource) -> np.ndarray | None:
        if labels is None:
            return None
        if isinstance(labels, ObjectLabelValue):
            return object_label_dense_array(labels, dtype=np.int32)
        return np.asarray(labels)


@dataclass(frozen=True, slots=True)
class TextureLabelSliceProjection:
    """Project object labels onto the image plane being texture-measured."""

    source: TextureLabelSource
    dense_labels: np.ndarray | None
    slice_array: np.ndarray
    slice_index: int

    @classmethod
    def from_source(
        cls,
        source: TextureLabelSource,
        slice_array: np.ndarray,
        slice_index: int,
    ) -> "TextureLabelSliceProjection":
        return cls(
            source=source,
            dense_labels=TextureDenseLabelArray.from_value(source),
            slice_array=slice_array,
            slice_index=slice_index,
        )

    def labels_2d(self) -> np.ndarray | None:
        if self.dense_labels is None or self.slice_array.ndim != 2:
            return None
        selected = self.dense_labels
        while (
            selected.ndim > 2
            and selected.shape[-2:] == self.slice_array.shape
            and selected.shape[0] > 0
        ):
            selected = selected[min(self.slice_index, selected.shape[0] - 1)]
        if selected.ndim == 2 and selected.shape == self.slice_array.shape:
            return np.asarray(selected, dtype=np.int32)
        return None

    def projected_payload(self, labels_2d: np.ndarray) -> TextureLabelSource:
        return ObjectLabelMeasurementPayloadStrategy.for_source(
            self.source
        ).materialize(
            self.source,
            ObjectLabelSourcePlaneProjectionRequest(labels_2d, self.slice_index),
        )


pure_2d_batch_executor(measure_texture_objects_batch)(measure_texture_objects)


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


@njit(cache=True)
def _max_value_2d_numba(values: np.ndarray) -> int:
    height, width = values.shape
    max_value = 0
    for y in range(height):
        for x in range(width):
            value = values[y, x]
            if value > max_value:
                max_value = value
    return max_value


@njit(cache=True)
def _object_bounding_boxes_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = labels.shape
    max_label = _max_value_2d_numba(labels)

    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.full(max_label + 1, -1, dtype=np.int64)
    max_x = np.full(max_label + 1, -1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y > max_y[label]:
                max_y[label] = y
            if x > max_x[label]:
                max_x[label] = x

    object_count = 0
    for label in range(1, max_label + 1):
        if max_y[label] >= 0:
            object_count += 1

    object_labels = np.empty(object_count, dtype=np.int64)
    boxes = np.empty((object_count, 4), dtype=np.int64)
    index = 0
    for label in range(1, max_label + 1):
        if max_y[label] < 0:
            continue
        object_labels[index] = label
        boxes[index, 0] = min_y[label]
        boxes[index, 1] = max_y[label] + 1
        boxes[index, 2] = min_x[label]
        boxes[index, 3] = max_x[label] + 1
        index += 1
    return object_labels, boxes


@njit(cache=True)
def _haralick_2d_features_numba(
    image: np.ndarray,
    distance: int,
    ignore_zeros: bool,
) -> np.ndarray:
    height, width = image.shape
    max_value = _max_value_2d_numba(image)

    gray_count = max_value + 1
    features = np.zeros((4, 13), dtype=np.float64)
    deltas_y = np.array((0, 1, 1, 1), dtype=np.int64)
    deltas_x = np.array((1, 1, 0, -1), dtype=np.int64)

    for direction in range(4):
        cmat = np.zeros((gray_count, gray_count), dtype=np.float64)
        dy = deltas_y[direction] * distance
        dx = deltas_x[direction] * distance
        for y in range(height):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for x in range(width):
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                a = image[y, x]
                b = image[yy, xx]
                if ignore_zeros and (a == 0 or b == 0):
                    continue
                cmat[a, b] += 1.0
                cmat[b, a] += 1.0

        total = cmat.sum()
        if total == 0.0:
            continue
        features[direction, :] = _haralick_features_from_cmat_numba(cmat, total)
    return features


@njit(cache=True)
def _haralick_features_from_cmat_numba(
    cmat: np.ndarray,
    total: float,
) -> np.ndarray:
    gray_count = cmat.shape[0]
    feats = np.zeros(13, dtype=np.float64)
    px = np.zeros(gray_count, dtype=np.float64)
    py = np.zeros(gray_count, dtype=np.float64)
    px_plus_y = np.zeros(gray_count * 2, dtype=np.float64)
    px_minus_y = np.zeros(gray_count, dtype=np.float64)

    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            px[j] += p
            py[i] += p
            px_plus_y[i + j] += p
            diff = i - j
            if diff < 0:
                diff = -diff
            px_minus_y[diff] += p
            feats[0] += p * p
            feats[1] += diff * diff * p
            feats[4] += p / (1.0 + diff * diff)

    ux = 0.0
    uy = 0.0
    for k in range(gray_count):
        ux += px[k] * k
        uy += py[k] * k

    vx = 0.0
    vy = 0.0
    for k in range(gray_count):
        vx += px[k] * k * k
        vy += py[k] * k * k
    vx -= ux * ux
    vy -= uy * uy

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    if sx == 0.0 or sy == 0.0:
        feats[2] = 1.0
    else:
        ijp = 0.0
        for i in range(gray_count):
            for j in range(gray_count):
                ijp += i * j * (cmat[i, j] / total)
        feats[2] = (ijp - ux * uy) / (sx * sy)

    feats[3] = vx
    sum_average = 0.0
    sum_second = 0.0
    for k in range(gray_count * 2):
        sum_average += k * px_plus_y[k]
        sum_second += k * k * px_plus_y[k]
    feats[5] = sum_average
    feats[7] = _entropy_numba(px_plus_y)
    feats[6] = sum_second - sum_average * sum_average
    feats[8] = _entropy_matrix_numba(cmat, total)

    mean_minus = 0.0
    for k in range(gray_count):
        mean_minus += px_minus_y[k]
    mean_minus /= gray_count
    variance_minus = 0.0
    for k in range(gray_count):
        delta = px_minus_y[k] - mean_minus
        variance_minus += delta * delta
    feats[9] = variance_minus / gray_count
    feats[10] = _entropy_numba(px_minus_y)

    hx = _entropy_numba(px)
    hy = _entropy_numba(py)
    hxy1 = 0.0
    hxy2 = 0.0
    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            cross = py[i] * px[j]
            if cross > 0.0 and p > 0.0:
                hxy1 -= p * np.log2(cross)
            if cross > 0.0:
                hxy2 -= cross * np.log2(cross)

    if hx >= hy:
        max_h = hx
    else:
        max_h = hy
    if max_h == 0.0:
        feats[11] = feats[8] - hxy1
    else:
        feats[11] = (feats[8] - hxy1) / max_h
    info2 = 1.0 - np.exp(-2.0 * (hxy2 - feats[8]))
    if info2 < 0.0:
        info2 = 0.0
    feats[12] = np.sqrt(info2)
    return feats


@njit(cache=True)
def _entropy_numba(values: np.ndarray) -> float:
    result = 0.0
    for value in values:
        if value > 0.0:
            result -= value * np.log2(value)
    return result


@njit(cache=True)
def _entropy_matrix_numba(cmat: np.ndarray, total: float) -> float:
    result = 0.0
    height, width = cmat.shape
    for y in range(height):
        for x in range(width):
            p = cmat[y, x] / total
            if p > 0.0:
                result -= p * np.log2(p)
    return result


__all__ = public_names_from_objects(
    CellProfilerTexturePixelDataRequest,
    HaralickFeatureMatrixRequest,
    HaralickTextureBackendStrategy,
    NativeNumpyHaralickTextureBackendStrategy,
    NumbaNumpyHaralickTextureBackendStrategy,
    NumbaNumpyObjectTextureCropBackendStrategy,
    ObjectTextureCropBackendStrategy,
    ObjectTextureMeasurement,
    TextureMeasurement,
    measure_texture,
    measure_texture_objects,
    extra_names=(
        "F_HARALICK",
        "N_DIRECTIONS_2D",
        "ObjectIntensityCrops",
    ),
)
