"""Texture-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Annotated, ClassVar

from metaclass_registry import AutoRegisterMeta
from numba import njit
import numpy as np

from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    PerObjectMeasurementExecutionModule,
    ScopedMeasurementModule,
    SourceQualifiedWideMeasurementRowsModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily, setting_values
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    SettingToKeywordBinding,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class MeasureTextureObjectMeasurementRowPolicy(
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy
):
    """Complete texture rows against the exact projected nominal label domain."""

    row_identity = MeasurementObjectRowIdentity.ROW_SEQUENCE

    def row_identity_axis_fields(
        self,
        axis_fields: Sequence[str],
        *,
        label_payload: RuntimeCallableArgument | None = None,
    ) -> tuple[str, ...]:
        """Use every field of the nominal texture axis as row identity."""
        del label_payload
        return tuple(
            dict.fromkeys(
                (
                    *axis_fields,
                    *(field.name for field in fields(TextureMeasurementAxis)),
                )
            )
        )


def _measure_texture_scope(value: str) -> CellProfilerMeasurementTargetScope:
    """Parse CellProfiler's texture target-scope literals."""

    return cellprofiler_enum_from_literal(
        CellProfilerMeasurementTargetScope,
        value,
        aliases={
            "images": CellProfilerMeasurementTargetScope.IMAGE,
            "objects": CellProfilerMeasurementTargetScope.OBJECT,
        },
    )


class MeasureTextureModule(
    LabelsObjectInputPolicy,
    MeasureTextureObjectMeasurementRowPolicy,
    PerObjectMeasurementExecutionModule,
    ScopedMeasurementModule,
    SourceQualifiedWideMeasurementRowsModule,
):
    module_name = "MeasureTexture"
    function_name = "measure_texture"
    validated = True
    function_variants = ("measure_texture_objects",)
    confidence = 1.0
    measurement_category_prefixes = (("texture",),)
    measurement_scope_binding = SettingToKeywordBinding(
        SettingNameFamily(
            "Measure images or objects?",
            aliases=("Measure whole images or objects?",),
        ),
        "measurement_scope",
        _measure_texture_scope,
    )
    measurement_scope_default = CellProfilerMeasurementTargetScope.IMAGE
    ignored_settings = (
        "Hidden",
        "Angles to measure",
        "Measure Gabor features?",
        "Number of angles to compute for Gabor",
    )

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Haralick feature families emitted by MeasureTexture."""

        ANGULAR_SECOND_MOMENT = "AngularSecondMoment"
        CONTRAST = "Contrast"
        CORRELATION = "Correlation"
        VARIANCE = "Variance"
        INVERSE_DIFFERENCE_MOMENT = "InverseDifferenceMoment"
        SUM_AVERAGE = "SumAverage"
        SUM_VARIANCE = "SumVariance"
        SUM_ENTROPY = "SumEntropy"
        ENTROPY = "Entropy"
        DIFFERENCE_VARIANCE = "DifferenceVariance"
        DIFFERENCE_ENTROPY = "DifferenceEntropy"
        INFO_MEAS1 = ("InfoMeas1", (), (), (), "info_meas1")
        INFO_MEAS2 = ("InfoMeas2", (), (), (), "info_meas2")

    texture_scale_setting = "Texture scale to measure"
    gray_levels_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Enter how many gray levels to measure the texture at"
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            gray_levels_setting,
            "gray_levels",
            parse_cellprofiler_int,
        ),
    )

    @classmethod
    def haralick_feature_prefixes(cls) -> tuple[str, ...]:
        del cls
        return tuple(
            f"{normalize_runtime_identifier(feature_name)}_"
            for feature_name in F_HARALICK
        )

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        bound = cls._bind_declared_settings(module, binder=binder)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        texture_scales = setting_values(module, cls.texture_scale_setting)
        if texture_scales:
            parsed_scales = tuple(
                (parse_cellprofiler_int(value) for value in texture_scales)
            )
            kwargs["scale"] = (
                parsed_scales[0] if len(parsed_scales) == 1 else parsed_scales
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.texture_scale_setting), None
            )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module, BoundModuleSettings(kwargs, unmapped_kwargs)
            ),
        )


F_HARALICK = [feature.value for feature in MeasureTextureModule.MeasurementFeature]
N_DIRECTIONS_2D = 4
TextureScale = Annotated[
    int | tuple[int, ...] | list[int],
    "Pixel offsets at which gray-level co-occurrences are measured.",
]
ObjectIntensityCrops = tuple[np.ndarray, tuple[np.ndarray, ...]]


class ObjectTextureCropBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Extract masked object intensity crops for texture measurement."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def object_intensity_crops(
        self, image: np.ndarray, labels: np.ndarray
    ) -> ObjectIntensityCrops:
        """Return positive object labels and CP-style masked intensity crops."""


class HaralickTextureBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Compute CP-compatible 2-D Haralick feature matrices."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_features(
        self, pixel_data: np.ndarray, *, scale: int, ignore_zeros: bool
    ) -> np.ndarray:
        """Return one Haralick feature row per 2-D direction."""


class NumbaNumpyObjectTextureCropBackendStrategy(ObjectTextureCropBackendStrategy):
    """Numba-accelerated NumPy backend for object texture crop extraction."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        labels = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]], dtype=np.int64)
        self.object_intensity_crops(image, labels)

    def object_intensity_crops(
        self, image: np.ndarray, labels: np.ndarray
    ) -> ObjectIntensityCrops:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "Numba texture crop backend currently supports 2-D NumPy planes."
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                f"Texture image and labels must have identical shapes; got {image_array.shape!r} and {labels_array.shape!r}."
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
        return (object_labels.astype(np.int64, copy=False), tuple(crops))


class NumbaNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Numba implementation of mahotas' default 2-D Haralick semantics."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(25, dtype=np.int64).reshape((5, 5))
        self.haralick_features(image, scale=1, ignore_zeros=False)

    def haralick_features(
        self, pixel_data: np.ndarray, *, scale: int, ignore_zeros: bool
    ) -> np.ndarray:
        import mahotas.features.texture as mahotas_texture

        pixel_array = np.ascontiguousarray(pixel_data)
        if pixel_array.ndim != 2:
            raise ValueError("Haralick texture backend expects a 2-D image plane.")
        if scale < 1:
            raise ValueError(f"Haralick texture scale must be positive, got {scale}.")
        if pixel_array.shape[0] <= scale or pixel_array.shape[1] <= scale:
            return np.zeros((4, 13), dtype=np.float64)
        cooccurrence_matrices = _haralick_2d_cooccurrence_matrices_numba(
            pixel_array.astype(np.int64, copy=False), int(scale)
        )
        return np.asarray(
            mahotas_texture.haralick_features(
                cooccurrence_matrices,
                ignore_zeros=ignore_zeros,
            ),
            dtype=np.float64,
        )


class NativeNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Explicit mahotas backend used as the native reference implementation."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE

    def haralick_features(
        self, pixel_data: np.ndarray, *, scale: int, ignore_zeros: bool
    ) -> np.ndarray:
        import mahotas.features as mahotas_features

        return np.asarray(
            mahotas_features.haralick(
                np.asarray(pixel_data), distance=scale, ignore_zeros=ignore_zeros
            ),
            dtype=np.float64,
        )


def _normalize_gray_levels(gray_levels: int) -> int:
    return max(2, min(256, int(gray_levels)))


def _texture_scales(scale: TextureScale) -> tuple[int, ...]:
    if isinstance(scale, (tuple, list)):
        return tuple((int(value) for value in scale))
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
                pixel_data, in_range=(0, 255), out_range=(0, self.gray_levels - 1)
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
            pixel_data, scale=self.scale, ignore_zeros=self.ignore_zeros
        ):
            return _zero_feature_matrix()
        backend = HaralickTextureBackendStrategy.for_memory_type(
            backend_provider=self.backend_provider
        )
        return np.asarray(
            backend.haralick_features(
                pixel_data, scale=self.scale, ignore_zeros=self.ignore_zeros
            ),
            dtype=float,
        )


def _haralick_has_valid_domain(
    pixel_data: np.ndarray, *, scale: int, ignore_zeros: bool
) -> bool:
    if pixel_data.ndim != 2:
        raise ValueError(
            "MeasureTexture expects a 2D image plane. Stack dispatch must be handled by the OpenHCS processing contract."
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
    """Cleaned Haralick feature row in declared feature-member order."""

    values: np.ndarray

    @classmethod
    def from_matrix(
        cls, feature_matrix: np.ndarray, direction: int
    ) -> "HaralickFeatureVector":
        if direction >= feature_matrix.shape[0]:
            return cls.zeros()
        return cls(_clean_feature_vector(feature_matrix[direction, :]))

    @classmethod
    def zeros(cls) -> "HaralickFeatureVector":
        return cls(np.zeros((len(F_HARALICK),), dtype=float))


def _texture_measurement_fields(*, object_scope: bool) -> tuple[FieldSpec, ...]:
    """Return the exact producer schema from nominal axis and feature declarations."""
    axis_fields = FieldSpec.from_dataclass_type(TextureMeasurementAxis)
    return (
        axis_fields[0],
        *(
            (FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),)
            if object_scope
            else ()
        ),
        *axis_fields[1:],
        *(
            FieldSpec(feature.measurement_row_field_name, float)
            for feature in MeasureTextureModule.MeasurementFeature
        ),
    )


def _texture_measurement_rows(
    axes: Sequence[TextureMeasurementAxis],
    feature_vectors: Sequence[HaralickFeatureVector],
    *,
    object_labels: Sequence[int] | None = None,
) -> MeasurementProjectedColumnarRows:
    """Build exact columnar texture rows without an inferred row schema."""
    if len(axes) != len(feature_vectors):
        raise ValueError("Texture measurement axes and feature vectors must align.")
    if object_labels is not None and len(object_labels) != len(axes):
        raise ValueError("Texture object labels and measurement axes must align.")

    object_scope = object_labels is not None
    axis_fields = FieldSpec.from_dataclass_type(TextureMeasurementAxis)
    feature_fields = tuple(
        FieldSpec(feature.measurement_row_field_name, float)
        for feature in MeasureTextureModule.MeasurementFeature
    )
    columns: dict[str, Sequence[object]] = {
        axis_fields[0].name: tuple(getattr(axis, axis_fields[0].name) for axis in axes)
    }
    if object_labels is not None:
        columns[MeasurementRowAxisField.OBJECT_LABEL.value] = tuple(
            int(object_label) for object_label in object_labels
        )
    columns.update(
        {
            field_spec.name: tuple(getattr(axis, field_spec.name) for axis in axes)
            for field_spec in axis_fields[1:]
        }
    )
    columns.update(
        {
            field_spec.name: tuple(
                float(feature_vector.values[feature_index])
                for feature_vector in feature_vectors
            )
            for feature_index, field_spec in enumerate(feature_fields)
        }
    )
    return MeasurementProjectedColumnarRows(
        MappingProxyType(columns),
        fields=_texture_measurement_fields(object_scope=object_scope),
        object_row_identity=(
            MeasurementObjectRowIdentity.ROW_SEQUENCE if object_scope else None
        ),
    )


@numpy(contract=ProcessingContract.PURE_2D)
def measure_texture(
    image: np.ndarray,
    scale: TextureScale = 3,
    gray_levels: int = 256,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, MeasurementProjectedColumnarRows]:
    """Measure Haralick texture features on a grayscale image."""
    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image, gray_levels=gray_levels
    ).pixel_data()
    return (
        image,
        _image_texture_measurements(
            pixel_data,
            scale=scale,
            gray_levels=gray_levels,
            haralick_backend_provider=haralick_backend_provider,
        ),
    )


def _image_texture_measurements(
    pixel_data: np.ndarray,
    *,
    scale: TextureScale,
    gray_levels: int,
    haralick_backend_provider: BackendProviderInput,
) -> MeasurementProjectedColumnarRows:
    """Return image-scoped texture rows from already-quantized pixels."""
    axes: list[TextureMeasurementAxis] = []
    feature_vectors: list[HaralickFeatureVector] = []
    for texture_scale in _texture_scales(scale):
        feature_matrix = HaralickFeatureMatrixRequest(
            pixel_data=pixel_data,
            scale=texture_scale,
            ignore_zeros=False,
            backend_provider=haralick_backend_provider,
        ).feature_matrix()
        for direction in range(N_DIRECTIONS_2D):
            axes.append(
                TextureMeasurementAxis(
                    slice_index=0,
                    scale=texture_scale,
                    direction=direction,
                    gray_levels=gray_levels,
                )
            )
            feature_vectors.append(
                HaralickFeatureVector.from_matrix(feature_matrix, direction)
            )
    return _texture_measurement_rows(axes, feature_vectors)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def measure_texture_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    measurement_scope: CellProfilerMeasurementTargetScope = (
        CellProfilerMeasurementTargetScope.OBJECT
    ),
    scale: TextureScale = 3,
    gray_levels: int = 256,
    texture_crop_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int = 0,
) -> tuple[np.ndarray, ColumnarRows]:
    """Measure Haralick texture for the image, labeled objects, or both.

    Args:
        labels: Objects within which texture features are measured.
        slice_index: Zero-based image-plane index recorded with object measurements.
    """
    target_scope = coerce_cellprofiler_measurement_target_scope(
        measurement_scope,
        CellProfilerMeasurementTargetScope.OBJECT,
    ).measurement_scope_selection
    image_array = np.asarray(image)
    if not isinstance(labels, ObjectLabelValue):
        raise TypeError(
            "MeasureTexture objects requires a runtime-projected ObjectLabelValue."
        )
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if image_array.ndim != 2 or label_array.ndim != 2:
        raise ValueError(
            "MeasureTexture objects requires runtime-projected 2-D image and label planes."
        )
    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image, gray_levels=gray_levels
    ).pixel_data()
    image_measurements = (
        _image_texture_measurements(
            pixel_data,
            scale=scale,
            gray_levels=gray_levels,
            haralick_backend_provider=haralick_backend_provider,
        )
        if target_scope.includes(MeasurementScope.IMAGE)
        else None
    )
    if not target_scope.includes(MeasurementScope.OBJECT):
        if image_measurements is None:
            raise ValueError("MeasureTexture requires at least one measurement scope.")
        return (image, image_measurements)
    crop_backend = ObjectTextureCropBackendStrategy.for_callable(
        measure_texture_objects, backend_provider=texture_crop_backend_provider
    )
    axes: list[TextureMeasurementAxis] = []
    feature_vectors: list[HaralickFeatureVector] = []
    measured_object_labels: list[int] = []
    object_labels, intensity_crops = crop_backend.object_intensity_crops(
        pixel_data, label_array
    )
    for object_label, intensity_crop in zip(
        object_labels,
        intensity_crops,
        strict=True,
    ):
        for texture_scale in _texture_scales(scale):
            feature_matrix = HaralickFeatureMatrixRequest(
                pixel_data=intensity_crop,
                scale=texture_scale,
                ignore_zeros=True,
                backend_provider=haralick_backend_provider,
            ).feature_matrix()
            for direction in range(N_DIRECTIONS_2D):
                axes.append(
                    TextureMeasurementAxis(
                        slice_index=slice_index,
                        scale=texture_scale,
                        direction=direction,
                        gray_levels=gray_levels,
                    )
                )
                feature_vectors.append(
                    HaralickFeatureVector.from_matrix(feature_matrix, direction)
                )
                measured_object_labels.append(int(object_label))
    object_measurements = _texture_measurement_rows(
        axes,
        feature_vectors,
        object_labels=measured_object_labels,
    )
    if image_measurements is None:
        return (image, object_measurements)
    return (
        image,
        ConcatenatedColumnarRows((image_measurements, object_measurements)),
    )


def _prepare_measure_texture() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    measure_texture.__wrapped__(image)


def _prepare_measure_texture_objects() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_texture_objects.__wrapped__(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
    )


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
def _object_bounding_boxes_numba(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return (object_labels, boxes)


@njit(cache=True)
def _haralick_2d_cooccurrence_matrices_numba(
    image: np.ndarray, distance: int
) -> np.ndarray:
    height, width = image.shape
    max_value = _max_value_2d_numba(image)
    gray_count = max_value + 1
    matrices = np.zeros((4, gray_count, gray_count), dtype=np.int32)
    deltas_y = np.array((0, 1, 1, 1), dtype=np.int64)
    deltas_x = np.array((1, 1, 0, -1), dtype=np.int64)
    for direction in range(4):
        cmat = matrices[direction]
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
                cmat[a, b] += 1
                cmat[b, a] += 1
    return matrices


__all__ = public_names_from_objects(
    CellProfilerTexturePixelDataRequest,
    HaralickFeatureMatrixRequest,
    HaralickTextureBackendStrategy,
    NativeNumpyHaralickTextureBackendStrategy,
    NumbaNumpyHaralickTextureBackendStrategy,
    NumbaNumpyObjectTextureCropBackendStrategy,
    ObjectTextureCropBackendStrategy,
    measure_texture,
    measure_texture_objects,
    extra_names=("F_HARALICK", "N_DIRECTIONS_2D", "ObjectIntensityCrops"),
)
