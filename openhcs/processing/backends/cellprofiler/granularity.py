"""Granularity numerics for CellProfiler-compatible texture measurement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import logging
import os
import re
from threading import Lock
import time
from types import MappingProxyType
from typing import ClassVar

import cv2
from metaclass_registry import AutoRegisterMeta
from numba import njit
import numpy as np

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.equivalence.cells import (
    runtime_cell_signature_counters_equivalent,
)
from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey
from openhcs.core.equivalence.measurement_features import (
    RuntimeMeasurementIndexedDescriptorEquivalence,
)
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_batch_contracts import (
    RuntimeBatchInvocationRequest,
    measurement_image_batch_executor,
)
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    RuntimeMeasurementFeature,
    RuntimeMeasurementIndexedDescriptorDeclaration,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
    PerObjectMeasurementExecutionModule,
    SourceQualifiedWideMeasurementRowsModule,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    DenseColumnarObjectMeasurementRowsMixin,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.cellprofiler._backend import (
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    BackendProviderInput,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

GRANULARITY_SPECTRUM_LENGTH = 16


@dataclass(frozen=True, slots=True)
class GranularitySpectrumDescriptor:
    """Nominal identity of one CellProfiler granularity spectrum index."""

    spectrum_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.spectrum_index, int):
            raise TypeError("Granularity spectrum index must be an integer.")
        if not 1 <= self.spectrum_index <= GRANULARITY_SPECTRUM_LENGTH:
            raise ValueError(
                "Granularity spectrum index must be between 1 and "
                f"{GRANULARITY_SPECTRUM_LENGTH}, got {self.spectrum_index}."
            )


class GranularitySpectrumDescriptorDeclaration(
    RuntimeMeasurementIndexedDescriptorDeclaration,
    RuntimeMeasurementIndexedDescriptorEquivalence,
):
    """Parse and render exact MeasureGranularity spectrum identities."""

    declaration_key = "cellprofiler_granularity_spectrum"
    feature_category = "Granularity"
    _row_field_pattern = re.compile(r"gs([1-9]|1[0-6])\Z")
    _feature_name_pattern = re.compile(r"Granularity_([1-9]|1[0-6])\Z", re.I)

    @classmethod
    def from_measurement_row_field_name(
        cls,
        field_name: str,
    ) -> GranularitySpectrumDescriptor | None:
        """Return the descriptor declared by one raw producer row field."""
        match = cls._row_field_pattern.fullmatch(str(field_name))
        if match is None:
            return None
        return GranularitySpectrumDescriptor(int(match.group(1)))

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> GranularitySpectrumDescriptor | None:
        """Parse one unqualified native CellProfiler granularity feature name."""
        match = cls._feature_name_pattern.fullmatch(str(feature_name))
        if match is None:
            return None
        return GranularitySpectrumDescriptor(int(match.group(1)))

    @classmethod
    def feature_name(cls, descriptor: object) -> str:
        """Render one unqualified native CellProfiler granularity feature name."""
        if not isinstance(descriptor, GranularitySpectrumDescriptor):
            raise TypeError(
                f"{cls.__name__}.feature_name requires GranularitySpectrumDescriptor."
            )
        return f"{cls.feature_category}_{descriptor.spectrum_index}"

    @classmethod
    def source_qualified_feature_name(
        cls,
        descriptor: GranularitySpectrumDescriptor,
        *,
        source_image_name: str,
    ) -> str:
        """Render one exact source-qualified producer feature identity."""
        if not source_image_name:
            raise ValueError("Granularity source image name cannot be empty.")
        return f"{cls.feature_name(descriptor)}_{source_image_name}"

    @classmethod
    def indexed_suffix_token_width(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Keep the native index before source identity, not as an export suffix."""
        del feature_tokens
        return None

    @classmethod
    def descriptor_values_equivalent(
        cls,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Use ordinary numeric equivalence for granularity descriptor values."""
        del cls, descriptor, key
        return runtime_cell_signature_counters_equivalent(left, right, policy)


class MeasureGranularityObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin, CellProfilerObjectMeasurementRowPolicy
):
    """Granularity rows are emitted over the complete dense object domain."""


class MeasureGranularityModule(
    LabelsObjectInputPolicy,
    MeasureGranularityObjectMeasurementRowPolicy,
    PerObjectMeasurementExecutionModule,
    SourceQualifiedWideMeasurementRowsModule,
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
):
    module_name = "MeasureGranularity"
    function_name = "measure_granularity"
    validated = True
    function_variants = ("measure_granularity_objects",)
    confidence = 1.0
    measurement_category_prefixes = (("granularity",),)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Indexed granularity spectrum family emitted by MeasureGranularity."""

        SPECTRUM = (
            GranularitySpectrumDescriptorDeclaration.feature_category,
            (),
            (),
            (GranularitySpectrumDescriptorDeclaration,),
            "gs",
        )

    subsample_size_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Subsampling factor for granularity measurements"
    )
    background_subsample_size_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Subsampling factor for background reduction"
    )
    element_radius_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Radius of structuring element"
    )
    spectrum_length_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Range of the granular spectrum"
    )
    ignored_settings = ("Measure within objects?", "image_count", "object_count")
    scalar_setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            subsample_size_setting,
            "subsample_size",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            background_subsample_size_setting,
            "background_subsample_size",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            element_radius_setting,
            "element_radius",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            spectrum_length_setting,
            "spectrum_length",
            parse_cellprofiler_int,
        ),
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        scalar_setting_bindings
    )

    @classmethod
    def source_qualified_feature_name(
        cls,
        field_name: str,
        source_image_name: str,
    ) -> str:
        """Project one raw spectrum field to its exact native producer identity."""
        descriptor = (
            GranularitySpectrumDescriptorDeclaration.from_measurement_row_field_name(
                field_name
            )
        )
        if descriptor is None:
            raise ValueError(
                f"{cls.__name__} has no declared granularity spectrum identity "
                f"for raw field {field_name!r}."
            )
        return GranularitySpectrumDescriptorDeclaration.source_qualified_feature_name(
            descriptor,
            source_image_name=source_image_name,
        )

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        bound = cls._bind_declared_settings(module, binder=binder)
        for binding in cls.scalar_setting_bindings:
            values = setting_values(module, binding.setting_name)
            if not values:
                continue
            parsed_values = tuple(
                (
                    (
                        binder.parse_value(
                            setting_names(binding.setting_name)[0],
                            value,
                        )
                        if binding.parse is None
                        else binding.parse(value)
                    )
                    for value in values
                )
            )
            first_value = parsed_values[0]
            if any((value != first_value for value in parsed_values[1:])):
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has per-row "
                    f"{setting_names(binding.setting_name)[0]!r} values "
                    f"{parsed_values!r}; OpenHCS currently binds one granularity "
                    "setting set per module."
                )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(dict(bound.kwargs), dict(bound.unmapped_kwargs)),
        )


_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def profile_enabled() -> bool:
    """Return whether per-function granularity runtime profiling is enabled."""
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeProfiler:
    """Shared CellProfiler runtime-profile emitter bound to a module logger."""

    logger: logging.Logger

    def enabled(self) -> bool:
        return profile_enabled()

    def log(self, label: str, seconds: float, **fields: object) -> None:
        RuntimeProfileLogger.log(self.logger, label, seconds, **fields)


runtime_profiler = CellProfilerRuntimeProfiler(logger)


def log_profile(label: str, seconds: float, **fields: object) -> None:
    """Emit one granularity runtime profile event when enabled."""
    runtime_profiler.log(label, seconds, **fields)


@dataclass
class GranularityMeasurement:
    """Granularity spectrum measurements for an image."""

    slice_index: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


@dataclass
class ObjectGranularityMeasurement:
    """Granularity spectrum measurements per object."""

    slice_index: int
    object_id: int
    gs1: float
    gs2: float
    gs3: float
    gs4: float
    gs5: float
    gs6: float
    gs7: float
    gs8: float
    gs9: float
    gs10: float
    gs11: float
    gs12: float
    gs13: float
    gs14: float
    gs15: float
    gs16: float


def _granularity_measurement(gs_values: list[float]) -> GranularityMeasurement:
    while len(gs_values) < GRANULARITY_SPECTRUM_LENGTH:
        gs_values.append(0.0)
    return GranularityMeasurement(
        slice_index=0,
        gs1=gs_values[0],
        gs2=gs_values[1],
        gs3=gs_values[2],
        gs4=gs_values[3],
        gs5=gs_values[4],
        gs6=gs_values[5],
        gs7=gs_values[6],
        gs8=gs_values[7],
        gs9=gs_values[8],
        gs10=gs_values[9],
        gs11=gs_values[10],
        gs12=gs_values[11],
        gs13=gs_values[12],
        gs14=gs_values[13],
        gs15=gs_values[14],
        gs16=gs_values[15],
    )


def _object_granularity_measurement(
    object_id: int, gs: np.ndarray
) -> ObjectGranularityMeasurement:
    return ObjectGranularityMeasurement(
        slice_index=0,
        object_id=int(object_id),
        gs1=gs[0],
        gs2=gs[1],
        gs3=gs[2],
        gs4=gs[3],
        gs5=gs[4],
        gs6=gs[5],
        gs7=gs[6],
        gs8=gs[7],
        gs9=gs[8],
        gs10=gs[9],
        gs11=gs[10],
        gs12=gs[11],
        gs13=gs[12],
        gs14=gs[13],
        gs15=gs[14],
        gs16=gs[15],
    )


def object_granularity_measurement_value_fields() -> tuple[str, ...]:
    """Return granularity spectrum fields from the row declaration."""
    return tuple(
        field.name
        for field in dataclass_fields(ObjectGranularityMeasurement)
        if field.name not in {"slice_index", "object_id"}
    )


@dataclass(frozen=True, slots=True)
class ObjectGranularityMeasurementRows(ObjectMeasurementColumnarRows):
    """Columnar object granularity rows over the emitted label-id domain."""

    fields: ClassVar[tuple[FieldSpec, ...]] = FieldSpec.from_dataclass_type(
        ObjectGranularityMeasurement
    )
    object_ids: np.ndarray
    gs_values: np.ndarray
    slice_index: int = 0
    _columns: Mapping[str, np.ndarray] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object_ids = np.asarray(self.object_ids, dtype=np.int32)
        gs_values = np.asarray(self.gs_values, dtype=np.float64)
        if gs_values.ndim != 2:
            raise ValueError("Object granularity values must be a 2-D array.")
        if object_ids.size != gs_values.shape[0]:
            raise ValueError(
                "Object granularity rows require one spectrum per object ID."
            )
        columns: dict[str, np.ndarray] = {
            "slice_index": np.full(
                object_ids.size, int(self.slice_index), dtype=np.int32
            ),
            "object_id": object_ids,
        }
        for column_index, field_name in enumerate(
            object_granularity_measurement_value_fields()
        ):
            columns[field_name] = (
                gs_values[:, column_index]
                if column_index < gs_values.shape[1]
                else np.zeros(object_ids.size, dtype=np.float64)
            )
        object.__setattr__(self, "object_ids", object_ids)
        object.__setattr__(self, "gs_values", gs_values)
        object.__setattr__(self, "_columns", MappingProxyType(columns))
        self.validate_fields()

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns

    def __len__(self) -> int:
        return int(self.object_ids.size)

    def __iter__(self):
        for row_index in range(len(self)):
            yield self[row_index]

    def __getitem__(self, row_index: int) -> ObjectGranularityMeasurement:
        return _object_granularity_measurement(
            int(self.object_ids[row_index]),
            self.gs_values[row_index],
        )


@dataclass(frozen=True, slots=True)
class GranularityImageSeries:
    """Background-corrected image and reconstruction series."""

    pixels: np.ndarray
    new_shape: np.ndarray
    reconstructions: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class GranularityBatchSettings:
    """Granularity settings that determine the reusable reconstruction series."""

    subsample_size: float
    background_subsample_size: float
    element_radius: int
    spectrum_length: int

    @classmethod
    def from_request(
        cls, request: RuntimeBatchInvocationRequest
    ) -> "GranularityBatchSettings":
        return cls(
            subsample_size=float(request.kwargs["subsample_size"]),
            background_subsample_size=float(
                request.kwargs["background_subsample_size"]
            ),
            element_radius=int(request.kwargs["element_radius"]),
            spectrum_length=int(request.kwargs["spectrum_length"]),
        )


@dataclass(frozen=True, slots=True)
class GranularityBatchInvocation:
    """One object-granularity request prepared for shared image-series execution."""

    index: int
    request: RuntimeBatchInvocationRequest
    image: np.ndarray
    labels: np.ndarray
    object_range: np.ndarray
    settings: GranularityBatchSettings

    @classmethod
    def from_request(
        cls,
        index: int,
        request: RuntimeBatchInvocationRequest,
    ) -> "GranularityBatchInvocation | None":
        if "labels" not in request.kwargs:
            return None
        image = np.asarray(image_payload_data(request.image))
        labels = object_label_dense_array(request.kwargs["labels"], dtype=np.int32)
        object_range = np.unique(labels[labels > 0]).astype(np.int32, copy=False)
        return cls(
            index=index,
            request=request,
            image=image,
            labels=labels,
            object_range=object_range,
            settings=GranularityBatchSettings.from_request(request),
        )

    def series_key(
        self,
    ) -> tuple[
        str,
        tuple[int, ...],
        bytes,
        GranularityBatchSettings,
    ]:
        dtype, shape, digest = granularity_image_content_key(self.image)
        return (dtype, shape, digest, self.settings)

    def output(self, series: GranularityImageSeries) -> tuple[object, object]:
        if self.object_range.size == 0:
            return (
                self.request.image,
                ObjectGranularityMeasurementRows(
                    np.empty(0, dtype=np.int32),
                    np.empty((0, GRANULARITY_SPECTRUM_LENGTH), dtype=np.float64),
                ),
            )
        gs_per_object = object_granularity_values(
            self.image,
            self.labels,
            self.object_range,
            series,
            subsample_size=self.settings.subsample_size,
            spectrum_length=self.settings.spectrum_length,
        )
        return (
            self.request.image,
            ObjectGranularityMeasurementRows(self.object_range, gs_per_object),
        )


@dataclass(frozen=True, slots=True)
class GranularityImageSeriesRequest:
    """Request for reusable background-corrected granularity reconstructions."""

    image: np.ndarray
    subsample_size: float
    background_subsample_size: float
    element_radius: int
    spectrum_length: int
    profile_function: str

    def log_profile(self, label: str, seconds: float, **fields: object) -> None:
        log_profile(label, seconds, function=self.profile_function, **fields)

    def series(self) -> GranularityImageSeries:
        image_array = np.asarray(self.image)
        phase_started_at = time.perf_counter()
        dtype, shape, digest = granularity_image_content_key(image_array)
        self.log_profile(
            "granularity_series_key", time.perf_counter() - phase_started_at
        )
        key = (
            dtype,
            shape,
            digest,
            float(self.subsample_size),
            float(self.background_subsample_size),
            int(self.element_radius),
            int(self.spectrum_length),
        )
        with GRANULARITY_IMAGE_SERIES_CACHE_LOCK:
            entry = GRANULARITY_IMAGE_SERIES_CACHE.get(key)
            if entry is not None:
                GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
                self.log_profile("granularity_series_cache_hit", 0.0)
                return entry
        phase_started_at = time.perf_counter()
        pixels, new_shape = background_corrected_pixels(
            image_array,
            self.subsample_size,
            self.background_subsample_size,
            self.element_radius,
        )
        self.log_profile(
            "granularity_background_correct",
            time.perf_counter() - phase_started_at,
            shape=tuple((int(value) for value in pixels.shape)),
        )
        phase_started_at = time.perf_counter()
        reconstructions = granularity_reconstruction_series(
            pixels, self.spectrum_length
        )
        self.log_profile(
            "granularity_reconstruction_series",
            time.perf_counter() - phase_started_at,
            reconstructions=len(reconstructions),
        )
        series = GranularityImageSeries(
            pixels=pixels, new_shape=new_shape, reconstructions=reconstructions
        )
        with GRANULARITY_IMAGE_SERIES_CACHE_LOCK:
            GRANULARITY_IMAGE_SERIES_CACHE[key] = series
            GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
            while (
                len(GRANULARITY_IMAGE_SERIES_CACHE)
                > GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES
            ):
                GRANULARITY_IMAGE_SERIES_CACHE.popitem(last=False)
        return series


GRANULARITY_IMAGE_SERIES_CACHE: dict[
    tuple[str, tuple[int, ...], bytes, float, float, int, int],
    GranularityImageSeries,
] = OrderedDict()
GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES = 16
GRANULARITY_IMAGE_SERIES_CACHE_LOCK = Lock()


def granularity_array_content_key(
    array: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact content key for a granularity array."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha1(contiguous.view(np.uint8)).digest()
    return (
        str(contiguous.dtype),
        tuple((int(value) for value in contiguous.shape)),
        digest,
    )


def granularity_image_content_key(
    image: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact value key for deterministic granularity series reuse."""
    return granularity_array_content_key(image)


class GranularityReconstructionBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Exact grayscale reconstruction backend used by MeasureGranularity."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def reconstruct_radius_one(
        self,
        seed: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Return reconstruction-by-dilation for the fixed CP granularity footprint."""

    def reconstruct_series(
        self,
        pixels: np.ndarray,
        spectrum_length: int,
    ) -> tuple[np.ndarray, ...]:
        """Return the erosion/reconstruction images for one granularity spectrum."""
        from skimage import morphology

        ero = pixels.copy()
        footprint = morphology.disk(1, dtype=np.uint8)
        reconstructions = []
        erosion_seconds = 0.0
        reconstruction_seconds = 0.0
        for index in range(int(spectrum_length)):
            phase_started_at = time.perf_counter()
            ero = granularity_grey_erosion(ero, footprint)
            erosion_seconds += time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            reconstruction = self.reconstruct_radius_one(ero, pixels)
            reconstruction_seconds += time.perf_counter() - phase_started_at
            log_profile(
                "granularity_reconstruction_iteration",
                time.perf_counter() - phase_started_at,
                function="measure_granularity_objects",
                iteration=index + 1,
                shape=tuple((int(value) for value in pixels.shape)),
            )
            reconstructions.append(reconstruction)
        log_profile(
            "granularity_reconstruction_erosion_total",
            erosion_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        log_profile(
            "granularity_reconstruction_dilation_total",
            reconstruction_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        return tuple(reconstructions)


class NativeGranularityReconstructionBackendStrategy(
    GranularityReconstructionBackendStrategy
):
    """Reference skimage reconstruction backend for granularity."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def reconstruct_radius_one(
        self,
        seed: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        from skimage import morphology

        footprint = morphology.disk(1, dtype=bool)
        return morphology.reconstruction(seed, mask, footprint=footprint)


class NumbaGranularityReconstructionBackendStrategy(
    GranularityReconstructionBackendStrategy
):
    """Single-thread exact radius-one FIFO reconstruction for granularity spectra."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        for dtype in (np.float32, np.float64):
            seed = np.zeros((8, 8), dtype=dtype)
            mask = np.zeros((8, 8), dtype=dtype)
            mask[2:6, 2:6] = 1.0
            seed[3:5, 3:5] = 1.0
            self.reconstruct_radius_one(seed, mask)

    def reconstruct_radius_one(
        self,
        seed: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        seed_array = np.asarray(seed)
        mask_array = np.asarray(mask)
        if seed_array.ndim != 2 or mask_array.ndim != 2:
            raise ValueError("Granularity reconstruction requires 2-D arrays.")
        if seed_array.shape != mask_array.shape:
            raise ValueError(
                "Granularity reconstruction seed and mask shapes must match, "
                f"got {seed_array.shape!r} and {mask_array.shape!r}."
            )
        return _granularity_reconstruction_radius_one_numba(
            np.ascontiguousarray(seed_array),
            np.ascontiguousarray(mask_array),
        )

    def reconstruct_series(
        self,
        pixels: np.ndarray,
        spectrum_length: int,
    ) -> tuple[np.ndarray, ...]:
        from skimage import morphology

        pixels_array = np.ascontiguousarray(np.asarray(pixels))
        if pixels_array.ndim != 2:
            raise ValueError("Granularity reconstruction requires 2-D arrays.")
        ero = pixels_array.copy()
        footprint = morphology.disk(1, dtype=np.uint8)
        queue_rows = np.empty(pixels_array.size, dtype=np.int64)
        queue_cols = np.empty(pixels_array.size, dtype=np.int64)
        queued = np.empty(pixels_array.shape, dtype=np.bool_)
        reconstructions = []
        erosion_seconds = 0.0
        reconstruction_seconds = 0.0
        for index in range(int(spectrum_length)):
            phase_started_at = time.perf_counter()
            ero = granularity_grey_erosion(ero, footprint)
            erosion_seconds += time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            reconstruction = _granularity_reconstruction_radius_one_numba_with_queue(
                np.ascontiguousarray(ero),
                pixels_array,
                queue_rows,
                queue_cols,
                queued,
            )
            reconstruction_seconds += time.perf_counter() - phase_started_at
            log_profile(
                "granularity_reconstruction_iteration",
                time.perf_counter() - phase_started_at,
                function="measure_granularity_objects",
                iteration=index + 1,
                shape=tuple((int(value) for value in pixels_array.shape)),
            )
            reconstructions.append(reconstruction)
        log_profile(
            "granularity_reconstruction_erosion_total",
            erosion_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        log_profile(
            "granularity_reconstruction_dilation_total",
            reconstruction_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        return tuple(reconstructions)


class OpenCVGranularityReconstructionBackendStrategy(
    GranularityReconstructionBackendStrategy
):
    """Single-thread exact radius-one reconstruction via geodesic OpenCV dilation."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.OPENCV,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.OPENCV
    is_default_backend = False

    def reconstruct_radius_one(
        self,
        seed: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        seed_array = np.asarray(seed)
        mask_array = np.asarray(mask)
        if seed_array.ndim != 2 or mask_array.ndim != 2:
            raise ValueError("Granularity reconstruction requires 2-D arrays.")
        if seed_array.shape != mask_array.shape:
            raise ValueError(
                "Granularity reconstruction seed and mask shapes must match, "
                f"got {seed_array.shape!r} and {mask_array.shape!r}."
            )
        return _granularity_reconstruction_radius_one_opencv(
            np.ascontiguousarray(seed_array),
            np.ascontiguousarray(mask_array),
        )

    def reconstruct_series(
        self,
        pixels: np.ndarray,
        spectrum_length: int,
    ) -> tuple[np.ndarray, ...]:
        from skimage import morphology

        pixels_array = np.ascontiguousarray(np.asarray(pixels))
        if pixels_array.ndim != 2:
            raise ValueError("Granularity reconstruction requires 2-D arrays.")
        ero = pixels_array.copy()
        footprint = morphology.disk(1, dtype=np.uint8)
        reconstructions = []
        erosion_seconds = 0.0
        reconstruction_seconds = 0.0
        for index in range(int(spectrum_length)):
            phase_started_at = time.perf_counter()
            ero = granularity_grey_erosion(ero, footprint)
            erosion_seconds += time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            reconstruction = _granularity_reconstruction_radius_one_opencv(
                np.ascontiguousarray(ero),
                pixels_array,
            )
            reconstruction_seconds += time.perf_counter() - phase_started_at
            log_profile(
                "granularity_reconstruction_iteration",
                time.perf_counter() - phase_started_at,
                function="measure_granularity_objects",
                iteration=index + 1,
                shape=tuple((int(value) for value in pixels_array.shape)),
            )
            reconstructions.append(reconstruction)
        log_profile(
            "granularity_reconstruction_erosion_total",
            erosion_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        log_profile(
            "granularity_reconstruction_dilation_total",
            reconstruction_seconds,
            function="measure_granularity_objects",
            reconstructions=len(reconstructions),
        )
        return tuple(reconstructions)


def granularity_reconstruction_backend(
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> GranularityReconstructionBackendStrategy:
    """Return the declared granularity reconstruction backend."""
    return GranularityReconstructionBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def granularity_reconstruction_series(
    pixels: np.ndarray, spectrum_length: int
) -> tuple[np.ndarray, ...]:
    """Compute the erosion/reconstruction images shared across object sets."""
    return granularity_reconstruction_backend().reconstruct_series(
        pixels,
        spectrum_length,
    )


@njit(cache=True)
def _granularity_reconstruction_radius_one_numba(
    seed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Exact 4-connected grayscale reconstruction-by-dilation."""
    queue_rows = np.empty(seed.size, dtype=np.int64)
    queue_cols = np.empty(seed.size, dtype=np.int64)
    queued = np.empty(seed.shape, dtype=np.bool_)
    return _granularity_reconstruction_radius_one_numba_with_queue(
        seed,
        mask,
        queue_rows,
        queue_cols,
        queued,
    )


def _granularity_reconstruction_radius_one_opencv(
    seed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Exact geodesic reconstruction by repeated radius-one dilation."""
    from skimage import morphology

    footprint = morphology.disk(1, dtype=np.uint8)
    current = np.ascontiguousarray(seed).copy()
    while True:
        next_image = cv2.dilate(
            current,
            footprint,
            borderType=cv2.BORDER_REPLICATE,
        )
        np.minimum(next_image, mask, out=next_image)
        if np.array_equal(next_image, current):
            return next_image
        current = next_image


@njit(cache=True)
def _granularity_reconstruction_radius_one_numba_with_queue(
    seed: np.ndarray,
    mask: np.ndarray,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    queued: np.ndarray,
) -> np.ndarray:
    """Exact 4-connected grayscale reconstruction-by-dilation using caller scratch."""
    height, width = seed.shape
    output = seed.copy()
    queued.fill(False)

    head = 0
    tail = 0
    queue_count = 0
    queue_capacity = queue_rows.size

    for row in range(height):
        for column in range(width):
            value = output[row, column]
            if row > 0 and output[row - 1, column] > value:
                value = output[row - 1, column]
            if column > 0 and output[row, column - 1] > value:
                value = output[row, column - 1]
            mask_value = mask[row, column]
            if value > mask_value:
                value = mask_value
            output[row, column] = value

    for row in range(height - 1, -1, -1):
        for column in range(width - 1, -1, -1):
            value = output[row, column]
            if row + 1 < height and output[row + 1, column] > value:
                value = output[row + 1, column]
            if column + 1 < width and output[row, column + 1] > value:
                value = output[row, column + 1]
            mask_value = mask[row, column]
            if value > mask_value:
                value = mask_value
            output[row, column] = value

            if _granularity_reconstruction_can_raise_any_neighbor(
                output,
                mask,
                value,
                row,
                column,
            ):
                tail, queue_count = _granularity_reconstruction_enqueue(
                    queue_rows,
                    queue_cols,
                    queued,
                    row,
                    column,
                    tail,
                    queue_count,
                    queue_capacity,
                )

    while queue_count > 0:
        row = queue_rows[head]
        column = queue_cols[head]
        queued[row, column] = False
        head += 1
        if head == queue_capacity:
            head = 0
        queue_count -= 1

        value = output[row, column]
        if row > 0:
            tail, queue_count = _granularity_reconstruction_update_neighbor(
                output,
                mask,
                row - 1,
                column,
                value,
                queue_rows,
                queue_cols,
                queued,
                tail,
                queue_count,
                queue_capacity,
            )
        if row + 1 < height:
            tail, queue_count = _granularity_reconstruction_update_neighbor(
                output,
                mask,
                row + 1,
                column,
                value,
                queue_rows,
                queue_cols,
                queued,
                tail,
                queue_count,
                queue_capacity,
            )
        if column > 0:
            tail, queue_count = _granularity_reconstruction_update_neighbor(
                output,
                mask,
                row,
                column - 1,
                value,
                queue_rows,
                queue_cols,
                queued,
                tail,
                queue_count,
                queue_capacity,
            )
        if column + 1 < width:
            tail, queue_count = _granularity_reconstruction_update_neighbor(
                output,
                mask,
                row,
                column + 1,
                value,
                queue_rows,
                queue_cols,
                queued,
                tail,
                queue_count,
                queue_capacity,
            )
    return output


@njit(cache=True)
def _granularity_reconstruction_can_raise_any_neighbor(
    output: np.ndarray,
    mask: np.ndarray,
    value: float,
    row: int,
    column: int,
) -> bool:
    height, width = output.shape
    if (
        row > 0
        and output[row - 1, column] < value
        and output[row - 1, column] < mask[row - 1, column]
    ):
        return True
    if (
        row + 1 < height
        and output[row + 1, column] < value
        and output[row + 1, column] < mask[row + 1, column]
    ):
        return True
    if (
        column > 0
        and output[row, column - 1] < value
        and output[row, column - 1] < mask[row, column - 1]
    ):
        return True
    return (
        column + 1 < width
        and output[row, column + 1] < value
        and output[row, column + 1] < mask[row, column + 1]
    )


@njit(cache=True)
def _granularity_reconstruction_update_neighbor(
    output: np.ndarray,
    mask: np.ndarray,
    row: int,
    column: int,
    source_value: float,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    queued: np.ndarray,
    tail: int,
    queue_count: int,
    queue_capacity: int,
) -> tuple[int, int]:
    candidate = source_value
    mask_value = mask[row, column]
    if candidate > mask_value:
        candidate = mask_value
    if candidate <= output[row, column]:
        return tail, queue_count
    output[row, column] = candidate
    return _granularity_reconstruction_enqueue(
        queue_rows,
        queue_cols,
        queued,
        row,
        column,
        tail,
        queue_count,
        queue_capacity,
    )


@njit(cache=True)
def _granularity_reconstruction_enqueue(
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    queued: np.ndarray,
    row: int,
    column: int,
    tail: int,
    queue_count: int,
    queue_capacity: int,
) -> tuple[int, int]:
    if queued[row, column]:
        return tail, queue_count
    queued[row, column] = True
    queue_rows[tail] = row
    queue_cols[tail] = column
    tail += 1
    if tail == queue_capacity:
        tail = 0
    queue_count += 1
    return tail, queue_count


def background_corrected_pixels(
    image: np.ndarray,
    subsample_size: float,
    background_subsample_size: float,
    element_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-style background-subtracted granularity pixels."""
    from skimage import morphology

    image = np.asarray(image)
    orig_shape = image.shape
    if subsample_size < 1:
        new_shape = np.asarray(orig_shape, dtype=np.float64) * float(subsample_size)
        pixels = resample_from_cp_grid(image, new_shape, 1.0 / float(subsample_size))
    else:
        pixels = image.copy()
        new_shape = np.asarray(orig_shape, dtype=np.float64)
    if background_subsample_size < 1:
        back_shape = new_shape * float(background_subsample_size)
        back_pixels = resample_from_cp_grid(
            pixels, back_shape, 1.0 / float(background_subsample_size)
        )
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape
    footprint = morphology.disk(int(element_radius), dtype=np.uint8)
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[...] = back_pixels
    back_pixels = granularity_grey_erosion(back_pixels_mask, footprint)
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[...] = back_pixels
    back_pixels = granularity_grey_dilation(back_pixels_mask, footprint)
    if background_subsample_size < 1:
        back_pixels = resample_between_cp_grids(back_pixels, back_shape, new_shape)
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0
    return (pixels, new_shape)


def granularity_grey_erosion(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Return CP/skimage-equivalent greyscale erosion for granularity footprints."""
    return cv2.erode(
        np.asarray(image),
        np.asarray(footprint, dtype=np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )


def granularity_grey_dilation(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Return CP/skimage-equivalent greyscale dilation for granularity footprints."""
    return cv2.dilate(
        np.asarray(image),
        np.asarray(footprint, dtype=np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )


@dataclass(frozen=True, slots=True)
class GranularityLabelPixels:
    """Foreground label pixels used to compute object granularity means."""

    object_ids: np.ndarray
    flat_offsets: np.ndarray
    row_offsets: np.ndarray
    column_offsets: np.ndarray
    object_indexes: np.ndarray
    object_counts: np.ndarray

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        object_ids: np.ndarray,
    ) -> "GranularityLabelPixels":
        label_array = np.asarray(labels, dtype=np.int32)
        object_ids = np.asarray(object_ids, dtype=np.int32)
        if object_ids.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return cls(object_ids, empty, empty, empty, empty, empty)
        label_to_index = np.full(int(object_ids.max()) + 1, -1, dtype=np.int64)
        label_to_index[object_ids] = np.arange(object_ids.size, dtype=np.int64)
        flat_labels = label_array.ravel()
        candidate_offsets = np.flatnonzero(
            (flat_labels > 0) & (flat_labels <= int(object_ids.max()))
        ).astype(np.int64, copy=False)
        object_indexes = label_to_index[flat_labels[candidate_offsets]]
        present_mask = object_indexes >= 0
        flat_offsets = candidate_offsets[present_mask]
        object_indexes = object_indexes[present_mask].astype(np.int64, copy=False)
        row_offsets = (flat_offsets // label_array.shape[1]).astype(
            np.int64, copy=False
        )
        column_offsets = (flat_offsets % label_array.shape[1]).astype(
            np.int64,
            copy=False,
        )
        object_counts = np.bincount(
            object_indexes,
            minlength=int(object_ids.size),
        ).astype(np.float64, copy=False)
        return cls(
            object_ids=object_ids,
            flat_offsets=flat_offsets,
            row_offsets=row_offsets,
            column_offsets=column_offsets,
            object_indexes=object_indexes,
            object_counts=object_counts,
        )

    def means_from_image(self, image: np.ndarray) -> np.ndarray:
        """Return per-object means from an image aligned to the label grid."""
        return _granularity_label_pixel_means(
            np.asarray(image).ravel(),
            self.flat_offsets,
            self.object_indexes,
            self.object_counts,
        )

    def means_from_resampled_image(
        self,
        image: np.ndarray,
        logical_shape: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        """Return per-object means after CP coordinate-grid resampling."""
        sampled_values = self.resampled_values(image, logical_shape, original_shape)
        return _granularity_label_values_means(
            sampled_values,
            self.object_indexes,
            self.object_counts,
        )

    def resampled_values(
        self,
        image: np.ndarray,
        logical_shape: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        from scipy import ndimage as ndi

        row_coords = self.row_offsets.astype(np.float64)
        column_coords = self.column_offsets.astype(np.float64)
        row_coords *= (
            float(logical_shape[0] - 1) / float(original_shape[0] - 1)
            if original_shape[0] > 1
            else 0.0
        )
        column_coords *= (
            float(logical_shape[1] - 1) / float(original_shape[1] - 1)
            if original_shape[1] > 1
            else 0.0
        )
        return ndi.map_coordinates(image, (row_coords, column_coords), order=1)


@njit(cache=True)
def _granularity_label_pixel_means(
    image_values: np.ndarray,
    flat_offsets: np.ndarray,
    object_indexes: np.ndarray,
    object_counts: np.ndarray,
) -> np.ndarray:
    sums = np.zeros(object_counts.size, dtype=np.float64)
    for pixel_index in range(flat_offsets.size):
        sums[object_indexes[pixel_index]] += float(
            image_values[flat_offsets[pixel_index]]
        )
    means = np.empty(object_counts.size, dtype=np.float64)
    for object_index in range(object_counts.size):
        if object_counts[object_index] > 0.0:
            means[object_index] = sums[object_index] / object_counts[object_index]
        else:
            means[object_index] = np.nan
    return means


@njit(cache=True)
def _granularity_label_values_means(
    values: np.ndarray,
    object_indexes: np.ndarray,
    object_counts: np.ndarray,
) -> np.ndarray:
    sums = np.zeros(object_counts.size, dtype=np.float64)
    for value_index in range(values.size):
        sums[object_indexes[value_index]] += float(values[value_index])
    means = np.empty(object_counts.size, dtype=np.float64)
    for object_index in range(object_counts.size):
        if object_counts[object_index] > 0.0:
            means[object_index] = sums[object_index] / object_counts[object_index]
        else:
            means[object_index] = np.nan
    return means


def object_granularity_values(
    image: np.ndarray,
    labels: np.ndarray,
    object_range: np.ndarray,
    series: GranularityImageSeries,
    *,
    subsample_size: float,
    spectrum_length: int,
) -> np.ndarray:
    """Return CP granularity spectrum values for each object id."""
    image = np.asarray(image)
    labels = np.asarray(labels)
    if image.ndim != 2 or labels.ndim != 2 or image.shape != labels.shape:
        raise ValueError(
            "MeasureGranularity object measurements require one 2-D image plane "
            "and one same-shaped 2-D object-label plane; got "
            f"image shape {image.shape!r} and labels shape {labels.shape!r}."
        )
    orig_shape = image.shape
    new_shape = series.new_shape
    label_pixels = GranularityLabelPixels.from_labels(labels, object_range)
    current_means = label_pixels.means_from_image(image)
    start_means = np.maximum(current_means, np.finfo(float).eps)
    gs_per_object = np.zeros((int(object_range.size), 16))
    for gs_idx, rec in enumerate(series.reconstructions[: int(spectrum_length)]):
        prev_means = current_means.copy()
        if subsample_size < 1:
            new_means = label_pixels.means_from_resampled_image(
                rec,
                new_shape,
                orig_shape,
            )
        else:
            new_means = label_pixels.means_from_image(rec)
        gs_values = (prev_means - new_means) * 100 / start_means
        gs_per_object[:, gs_idx] = gs_values
        current_means = new_means
    return gs_per_object


def resample_to_original_shape_cp(
    image: np.ndarray,
    logical_shape: np.ndarray,
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Restore a CP-resampled image to the original grid."""
    from scipy import ndimage as ndi

    row_coords, col_coords = np.mgrid[
        0 : original_shape[0], 0 : original_shape[1]
    ].astype(float)
    row_coords *= (
        float(logical_shape[0] - 1) / float(original_shape[0] - 1)
        if original_shape[0] > 1
        else 0.0
    )
    col_coords *= (
        float(logical_shape[1] - 1) / float(original_shape[1] - 1)
        if original_shape[1] > 1
        else 0.0
    )
    return ndi.map_coordinates(image, (row_coords, col_coords), order=1)


def resample_from_cp_grid(
    image: np.ndarray,
    logical_shape: np.ndarray,
    coordinate_scale: float,
) -> np.ndarray:
    """Sample an image with CellProfiler's ``numpy.mgrid`` coordinate grid."""
    from scipy import ndimage as ndi

    row_coords, col_coords = np.mgrid[
        0 : logical_shape[0], 0 : logical_shape[1]
    ].astype(float)
    row_coords *= float(coordinate_scale)
    col_coords *= float(coordinate_scale)
    return ndi.map_coordinates(image, (row_coords, col_coords), order=1)


def resample_between_cp_grids(
    image: np.ndarray,
    source_logical_shape: np.ndarray,
    target_logical_shape: np.ndarray,
) -> np.ndarray:
    """Sample one CP logical grid onto another CP logical grid."""
    from scipy import ndimage as ndi

    row_coords, col_coords = np.mgrid[
        0 : target_logical_shape[0], 0 : target_logical_shape[1]
    ].astype(float)
    row_coords *= (
        float(source_logical_shape[0] - 1) / float(target_logical_shape[0] - 1)
        if target_logical_shape[0] > 1
        else 0.0
    )
    col_coords *= (
        float(source_logical_shape[1] - 1) / float(target_logical_shape[1] - 1)
        if target_logical_shape[1] > 1
        else 0.0
    )
    return ndi.map_coordinates(image, (row_coords, col_coords), order=1)


@numpy(contract=ProcessingContract.PURE_2D)
def measure_granularity(
    image: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure granularity spectrum of an image."""
    series = GranularityImageSeriesRequest(
        image=image,
        subsample_size=subsample_size,
        background_subsample_size=background_subsample_size,
        element_radius=element_radius,
        spectrum_length=spectrum_length,
        profile_function="measure_granularity",
    ).series()
    pixels = series.pixels
    startmean = max(np.mean(pixels), np.finfo(float).eps)
    currentmean = startmean
    gs_values = []
    for index, reconstruction in enumerate(series.reconstructions):
        prevmean = currentmean
        currentmean = np.mean(reconstruction)
        gs = (prevmean - currentmean) * 100 / startmean
        gs_values.append(gs)
    return (
        image,
        DataclassMeasurementColumnarRows(
            (_granularity_measurement(gs_values),),
            row_type=GranularityMeasurement,
        ),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def measure_granularity_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> tuple[np.ndarray, ObjectGranularityMeasurementRows]:
    """Measure granularity spectrum within labeled objects.

    Args:
        labels: Object-label plane defining the regions for which separate
            granularity spectra are measured.
    """
    labels = object_label_dense_array(labels, dtype=np.int32)
    object_range = np.unique(labels[labels > 0]).astype(np.int32, copy=False)
    if object_range.size == 0:
        return (
            image,
            ObjectGranularityMeasurementRows(
                np.empty(0, dtype=np.int32),
                np.empty((0, GRANULARITY_SPECTRUM_LENGTH), dtype=np.float64),
            ),
        )
    series = GranularityImageSeriesRequest(
        image=image,
        subsample_size=subsample_size,
        background_subsample_size=background_subsample_size,
        element_radius=element_radius,
        spectrum_length=spectrum_length,
        profile_function="measure_granularity_objects",
    ).series()
    gs_per_object = object_granularity_values(
        image,
        labels,
        object_range,
        series,
        subsample_size=subsample_size,
        spectrum_length=spectrum_length,
    )
    return (image, ObjectGranularityMeasurementRows(object_range, gs_per_object))


def measure_granularity_objects_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest], object
    ],
) -> list[object]:
    """Share each image reconstruction series across object-label invocations."""
    if len(requests) <= 1:
        return [execute_request(func, request) for request in requests]

    invocations = tuple(
        GranularityBatchInvocation.from_request(index, request)
        for index, request in enumerate(requests)
    )
    if any(invocation is None for invocation in invocations):
        return [execute_request(func, request) for request in requests]
    concrete_invocations = tuple(
        invocation for invocation in invocations if invocation is not None
    )
    invocations_by_series: dict[
        tuple[str, tuple[int, ...], bytes, GranularityBatchSettings],
        list[GranularityBatchInvocation],
    ] = {}
    for invocation in concrete_invocations:
        key = invocation.series_key()
        if key not in invocations_by_series:
            invocations_by_series[key] = []
        invocations_by_series[key].append(invocation)

    def execute_group(
        group: list[GranularityBatchInvocation],
    ) -> list[tuple[int, object]]:
        first = group[0]
        series = GranularityImageSeriesRequest(
            image=first.image,
            subsample_size=first.settings.subsample_size,
            background_subsample_size=first.settings.background_subsample_size,
            element_radius=first.settings.element_radius,
            spectrum_length=first.settings.spectrum_length,
            profile_function="measure_granularity_objects",
        ).series()
        return [(invocation.index, invocation.output(series)) for invocation in group]

    grouped_outputs: list[tuple[int, object]] = []
    groups = tuple(invocations_by_series.values())
    for group in groups:
        grouped_outputs.extend(execute_group(group))
    ordered_outputs = {index: output for index, output in grouped_outputs}
    return [ordered_outputs[index] for index in range(len(requests))]


measurement_image_batch_executor(measure_granularity_objects_batch)(
    measure_granularity_objects
)


@processing_prepare(measure_granularity, measure_granularity_objects)
def _prepare_granularity_backend() -> None:
    """Compile Numba kernels used by the granularity backend before execution."""
    GranularityReconstructionBackendStrategy.prepare_registered_family()
    for dtype in (np.float32, np.float64):
        image = np.linspace(0.0, 1.0, 64 * 64, dtype=dtype).reshape((64, 64))
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[8:24, 8:24] = 1
        labels[32:56, 32:56] = 2
        measure_granularity.__wrapped__(
            image,
            subsample_size=1.0,
            background_subsample_size=0.25,
            element_radius=10,
            spectrum_length=5,
        )
        measure_granularity_objects.__wrapped__(
            image,
            labels,
            subsample_size=1.0,
            background_subsample_size=0.25,
            element_radius=10,
            spectrum_length=5,
        )


__all__ = [
    "GRANULARITY_IMAGE_SERIES_CACHE",
    "GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES",
    "GRANULARITY_SPECTRUM_LENGTH",
    "GranularityImageSeries",
    "GranularityImageSeriesRequest",
    "GranularityMeasurement",
    "GranularitySpectrumDescriptor",
    "GranularitySpectrumDescriptorDeclaration",
    "ObjectGranularityMeasurement",
    "ObjectGranularityMeasurementRows",
    "OpenCVGranularityReconstructionBackendStrategy",
    "background_corrected_pixels",
    "granularity_image_content_key",
    "granularity_reconstruction_series",
    "log_profile",
    "measure_granularity",
    "measure_granularity_objects",
    "object_granularity_values",
]
