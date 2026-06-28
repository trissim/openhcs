"""Granularity numerics for CellProfiler-compatible texture measurement."""

from __future__ import annotations

from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)

from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageMeasurementInputModule,
    ModuleSettingsSourceModule,
    ObjectMeasurementInputModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
    ObjectMeasurementRowsModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    DenseEmittedObjectMeasurementRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)

class MeasureGranularityObjectMeasurementRowPolicy(
    DenseEmittedObjectMeasurementRowsMixin,
    CellProfilerObjectMeasurementRowPolicy,
):
    """Granularity rows are emitted over the complete dense object domain."""


class MeasureGranularityModule(
    LabelsObjectInputPolicy,
    PerObjectMeasurementExecutionModule,
    ObjectMeasurementRowsModule,
    MeasureGranularityObjectMeasurementRowPolicy,
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
):
    module_name = 'MeasureGranularity'
    function_name = 'measure_granularity'
    validated = True
    function_variants = ('measure_granularity_objects',)
    contract = 'unknown'
    confidence = 1.0
    object_measurement_setting = SettingNameFamily(
        "Select objects to measure",
        aliases=("Select an object to measure",),
    )
    ignored_settings = (
        "Measure within objects?",
        "image_count",
        "object_count",
    )
    scalar_settings: ClassVar[Mapping[str, tuple[str, Callable[[str], Any]]]] = {
        "Subsampling factor for granularity measurements": (
            "subsample_size",
            parse_cellprofiler_float,
        ),
        "Subsampling factor for background reduction": (
            "background_subsample_size",
            parse_cellprofiler_float,
        ),
        "Radius of structuring element": (
            "element_radius",
            parse_cellprofiler_int,
        ),
        "Range of the granular spectrum": (
            "spectrum_length",
            parse_cellprofiler_int,
        ),
    }

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        default_function_name: str | None = None,
    ) -> "ResolvedModuleFunction":
        object_values = setting_values(module, cls.object_measurement_setting)
        if any(split_symbol_names(value) for value in object_values):
            return super().resolve_function(
                module,
                default_function_name=cls.function_variants[0],
            )
        return super().resolve_function(
            module,
            default_function_name=default_function_name,
        )

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        bound = cls._bind_generic_settings(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name, (parameter_name, parse) in cls.scalar_settings.items():
            values = setting_values(module, setting_name)
            if not values:
                continue
            parsed_values = tuple(parse(value) for value in values)
            first_value = parsed_values[0]
            if any(value != first_value for value in parsed_values[1:]):
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has per-row "
                    f"{setting_name!r} values {parsed_values!r}; OpenHCS "
                    "currently binds one granularity setting set per module."
                )
            kwargs[parameter_name] = first_value
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        return cls.measurement_artifact_contract_from_declared_settings(
            assembler,
            builder,
            module,
        )



from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
import os
import time

import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi

from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)

GRANULARITY_FIELDS = [
    "slice_index",
    "gs1",
    "gs2",
    "gs3",
    "gs4",
    "gs5",
    "gs6",
    "gs7",
    "gs8",
    "gs9",
    "gs10",
    "gs11",
    "gs12",
    "gs13",
    "gs14",
    "gs15",
    "gs16",
]


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
    while len(gs_values) < 16:
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
    object_id: int,
    gs: np.ndarray,
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


@dataclass(frozen=True, slots=True)
class GranularityImageSeries:
    """Background-corrected image and reconstruction series."""

    pixels: np.ndarray
    new_shape: np.ndarray
    reconstructions: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class GranularityImageSeriesCacheEntry:
    """Cache entry for one exact granularity image-series key."""

    series: GranularityImageSeries


@dataclass(frozen=True, slots=True)
class GranularityLabelPixels:
    """Compact foreground pixel coordinates for one label/object domain."""

    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    pixel_object_indices: np.ndarray
    object_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class GranularityLabelPixelsCacheEntry:
    """Cache entry for compact granularity label pixels."""

    pixels: GranularityLabelPixels


@dataclass(frozen=True, slots=True)
class GranularityLabelPixelsRequest:
    """Request for reusable compact label pixels."""

    labels: np.ndarray
    object_range: np.ndarray
    profile_function: str

    def log_profile(self, label: str, seconds: float, **fields: object) -> None:
        log_profile(label, seconds, function=self.profile_function, **fields)

    def pixels(self) -> GranularityLabelPixels:
        labels_array = np.ascontiguousarray(self.labels, dtype=np.int32)
        object_range_array = np.ascontiguousarray(self.object_range, dtype=np.int32)
        phase_started_at = time.perf_counter()
        key = (
            *granularity_array_content_key(labels_array),
            *granularity_array_content_key(object_range_array),
        )
        self.log_profile(
            "granularity_label_pixels_key",
            time.perf_counter() - phase_started_at,
            objects=int(object_range_array.size),
        )
        entry = GRANULARITY_LABEL_PIXELS_CACHE.get(key)
        if entry is not None:
            GRANULARITY_LABEL_PIXELS_CACHE.move_to_end(key)
            self.log_profile(
                "granularity_label_pixels_cache_hit",
                0.0,
                objects=int(object_range_array.size),
            )
            return entry.pixels

        phase_started_at = time.perf_counter()
        label_to_index = label_to_index_lookup_numba(object_range_array)
        pixel_rows, pixel_cols, pixel_object_indices, object_counts = (
            compact_label_pixels_from_lookup_numba(labels_array, label_to_index)
        )
        pixels = GranularityLabelPixels(
            pixel_rows=pixel_rows,
            pixel_cols=pixel_cols,
            pixel_object_indices=pixel_object_indices,
            object_counts=object_counts,
        )
        self.log_profile(
            "granularity_label_pixels_compact",
            time.perf_counter() - phase_started_at,
            objects=int(object_range_array.size),
            pixels=int(pixel_rows.size),
        )
        GRANULARITY_LABEL_PIXELS_CACHE[key] = GranularityLabelPixelsCacheEntry(
            pixels=pixels,
        )
        GRANULARITY_LABEL_PIXELS_CACHE.move_to_end(key)
        while len(GRANULARITY_LABEL_PIXELS_CACHE) > GRANULARITY_LABEL_PIXELS_CACHE_MAX_ENTRIES:
            GRANULARITY_LABEL_PIXELS_CACHE.popitem(last=False)
        return pixels


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
            "granularity_series_key",
            time.perf_counter() - phase_started_at,
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
        entry = GRANULARITY_IMAGE_SERIES_CACHE.get(key)
        if entry is not None:
            GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
            self.log_profile(
                "granularity_series_cache_hit",
                0.0,
            )
            return entry.series

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
            shape=tuple(int(value) for value in pixels.shape),
        )
        phase_started_at = time.perf_counter()
        reconstructions = granularity_reconstruction_series(
            pixels,
            self.spectrum_length,
        )
        self.log_profile(
            "granularity_reconstruction_series",
            time.perf_counter() - phase_started_at,
            reconstructions=len(reconstructions),
        )
        series = GranularityImageSeries(
            pixels=pixels,
            new_shape=new_shape,
            reconstructions=reconstructions,
        )
        GRANULARITY_IMAGE_SERIES_CACHE[key] = GranularityImageSeriesCacheEntry(
            series=series,
        )
        GRANULARITY_IMAGE_SERIES_CACHE.move_to_end(key)
        while len(GRANULARITY_IMAGE_SERIES_CACHE) > GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES:
            GRANULARITY_IMAGE_SERIES_CACHE.popitem(last=False)
        return series


GRANULARITY_IMAGE_SERIES_CACHE: dict[
    tuple[str, tuple[int, ...], bytes, float, float, int, int],
    GranularityImageSeriesCacheEntry,
] = OrderedDict()
GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES = 16
GRANULARITY_LABEL_PIXELS_CACHE: dict[
    tuple[
        str,
        tuple[int, ...],
        bytes,
        str,
        tuple[int, ...],
        bytes,
    ],
    GranularityLabelPixelsCacheEntry,
] = OrderedDict()
GRANULARITY_LABEL_PIXELS_CACHE_MAX_ENTRIES = 16


def granularity_array_content_key(
    array: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact content key for a granularity array."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha1(contiguous.view(np.uint8)).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def granularity_image_content_key(
    image: np.ndarray,
) -> tuple[str, tuple[int, ...], bytes]:
    """Return an exact value key for deterministic granularity series reuse."""
    return granularity_array_content_key(image)


def granularity_reconstruction_series(
    pixels: np.ndarray,
    spectrum_length: int,
) -> tuple[np.ndarray, ...]:
    """Compute the erosion/reconstruction images shared across object sets."""
    ero = pixels.copy()
    cross_offsets = disk_offsets(1)
    reconstructions = []
    erosion_seconds = 0.0
    reconstruction_seconds = 0.0
    for index in range(int(spectrum_length)):
        phase_started_at = time.perf_counter()
        ero = gray_erosion_offsets_reflect_numba(ero, cross_offsets)
        erosion_seconds += time.perf_counter() - phase_started_at
        phase_started_at = time.perf_counter()
        reconstruction = reconstruct_dilation_cross_numba(ero, pixels)
        reconstruction_seconds += time.perf_counter() - phase_started_at
        log_profile(
            "granularity_reconstruction_iteration",
            time.perf_counter() - phase_started_at,
            function="measure_granularity_objects",
            iteration=index + 1,
            shape=tuple(int(value) for value in pixels.shape),
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


def background_corrected_pixels(
    image: np.ndarray,
    subsample_size: float,
    background_subsample_size: float,
    element_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CP-style background-subtracted granularity pixels."""
    image = np.asarray(image, dtype=np.float64)
    orig_shape = image.shape

    if subsample_size < 1:
        new_shape = (np.asarray(orig_shape) * subsample_size).astype(np.int64)
        new_shape = np.maximum(new_shape, 1)
        pixels = resample_bilinear_numba(
            image,
            int(new_shape[0]),
            int(new_shape[1]),
            1.0 / float(subsample_size),
            1.0 / float(subsample_size),
        )
    else:
        pixels = image.copy()
        new_shape = np.asarray(orig_shape, dtype=np.int64)

    if background_subsample_size < 1:
        back_shape = (new_shape * background_subsample_size).astype(np.int64)
        back_shape = np.maximum(back_shape, 1)
        back_pixels = resample_bilinear_numba(
            pixels,
            int(back_shape[0]),
            int(back_shape[1]),
            1.0 / float(background_subsample_size),
            1.0 / float(background_subsample_size),
        )
    else:
        back_pixels = pixels.copy()
        back_shape = new_shape

    back_pixels = GranularityDiskOpening(int(element_radius)).apply(back_pixels)

    if background_subsample_size < 1:
        row_scale = (
            float(back_shape[0] - 1) / float(new_shape[0] - 1)
            if new_shape[0] > 1
            else 0.0
        )
        col_scale = (
            float(back_shape[1] - 1) / float(new_shape[1] - 1)
            if new_shape[1] > 1
            else 0.0
        )
        back_pixels = resample_bilinear_numba(
            back_pixels,
            int(new_shape[0]),
            int(new_shape[1]),
            row_scale,
            col_scale,
        )

    pixels = pixels - back_pixels
    clip_negative_inplace_numba(pixels)
    return pixels, new_shape


def object_granularity_values(
    image: np.ndarray,
    labels: np.ndarray,
    object_range: np.ndarray,
    series: GranularityImageSeries,
    *,
    subsample_size: float,
    spectrum_length: int,
    label_pixels: GranularityLabelPixels | None = None,
) -> np.ndarray:
    """Return CP granularity spectrum values for each object id."""
    orig_shape = image.shape
    pixels = series.pixels
    new_shape = series.new_shape
    if label_pixels is None:
        label_pixels = GranularityLabelPixelsRequest(
            labels=labels,
            object_range=object_range,
            profile_function="measure_granularity_objects",
        ).pixels()
    current_means = mean_by_compact_label_pixels_numba(
        np.asarray(image),
        label_pixels.pixel_rows,
        label_pixels.pixel_cols,
        label_pixels.pixel_object_indices,
        label_pixels.object_counts,
    )
    start_means = np.maximum(current_means, np.finfo(float).eps)
    gs_per_object = np.zeros((int(object_range.size), 16))
    for gs_idx, rec in enumerate(series.reconstructions[: int(spectrum_length)]):
        prev_means = current_means.copy()
        if subsample_size < 1:
            row_scale = (
                float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                if orig_shape[0] > 1
                else 0.0
            )
            col_scale = (
                float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                if orig_shape[1] > 1
                else 0.0
            )
            new_means = mean_by_compact_label_pixels_from_resampled_numba(
                np.asarray(rec),
                label_pixels.pixel_rows,
                label_pixels.pixel_cols,
                label_pixels.pixel_object_indices,
                label_pixels.object_counts,
                row_scale,
                col_scale,
            )
        else:
            new_means = mean_by_compact_label_pixels_numba(
                np.asarray(rec),
                label_pixels.pixel_rows,
                label_pixels.pixel_cols,
                label_pixels.pixel_object_indices,
                label_pixels.object_counts,
            )
        gs_values = (prev_means - new_means) * 100 / start_means
        if gs_idx > 0:
            np.maximum(gs_values, 0.0, out=gs_values)
        gs_per_object[:, gs_idx] = gs_values
        current_means = new_means
    return gs_per_object


@lru_cache(maxsize=None)
def disk_offsets(radius: int) -> np.ndarray:
    """Return integer offsets for skimage.morphology.disk(radius)."""
    radius = int(radius)
    coords = []
    radius_sq = radius * radius
    for row in range(-radius, radius + 1):
        for col in range(-radius, radius + 1):
            if row * row + col * col <= radius_sq:
                coords.append((row, col))
    return np.asarray(coords, dtype=np.int32)


@lru_cache(maxsize=None)
def disk_footprint(radius: int) -> np.ndarray:
    """Return a boolean disk footprint matching ``disk_offsets``."""
    radius = int(radius)
    offsets = disk_offsets(radius)
    diameter = radius * 2 + 1
    footprint = np.zeros((diameter, diameter), dtype=bool)
    for row_offset, col_offset in offsets:
        footprint[int(row_offset) + radius, int(col_offset) + radius] = True
    return footprint


@dataclass(frozen=True, slots=True)
class GranularityDiskOpening:
    """Exact reflected grayscale opening for granularity background correction."""

    radius: int

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.footprint_fits(image):
            offsets = disk_offsets(self.radius)
            eroded = gray_erosion_offsets_reflect_numba(image, offsets)
            return gray_dilation_offsets_reflect_numba(eroded, offsets)
        footprint = disk_footprint(self.radius)
        eroded = ndi.grey_erosion(image, footprint=footprint, mode="reflect")
        return ndi.grey_dilation(eroded, footprint=footprint, mode="reflect")

    def footprint_fits(self, image: np.ndarray) -> bool:
        diameter = int(self.radius) * 2 + 1
        return image.ndim == 2 and min(image.shape) >= diameter


@njit(cache=True)
def resample_bilinear_numba(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    result = np.empty((output_height, output_width), dtype=np.float64)
    for row in range(output_height):
        sample_row = row * row_scale
        for col in range(output_width):
            result[row, col] = bilinear_sample_numba(
                image,
                sample_row,
                col * col_scale,
            )
    return result


@njit(cache=True)
def clip_negative_inplace_numba(image: np.ndarray) -> None:
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            if image[row, col] < 0.0:
                image[row, col] = 0.0


@njit(cache=True, parallel=True)
def gray_erosion_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value < best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True, parallel=True)
def gray_dilation_offsets_reflect_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    result = np.empty((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            best = -np.inf
            for offset_index in range(offsets.shape[0]):
                sample_row = reflect_index(row + int(offsets[offset_index, 0]), height)
                sample_col = reflect_index(col + int(offsets[offset_index, 1]), width)
                value = float(image[sample_row, sample_col])
                if value > best:
                    best = value
            result[row, col] = best
    return result


@njit(cache=True)
def reflect_index(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index - 1
        else:
            index = 2 * size - index - 1
    return index


@njit(cache=True)
def reconstruct_dilation_cross_numba(
    seed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Morphological reconstruction by dilation with disk(1) connectivity."""
    height, width = seed.shape
    total_pixels = height * width
    queue_rows = np.empty(total_pixels, dtype=np.int64)
    queue_cols = np.empty(total_pixels, dtype=np.int64)
    queued = np.zeros((height, width), dtype=np.bool_)
    result = seed.copy()

    for row in range(height):
        for col in range(width):
            value = result[row, col]
            if row > 0 and result[row - 1, col] > value:
                value = result[row - 1, col]
            if col > 0 and result[row, col - 1] > value:
                value = result[row, col - 1]
            mask_value = mask[row, col]
            if value > mask_value:
                value = mask_value
            result[row, col] = value

    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            value = result[row, col]
            if row + 1 < height and result[row + 1, col] > value:
                value = result[row + 1, col]
            if col + 1 < width and result[row, col + 1] > value:
                value = result[row, col + 1]
            mask_value = mask[row, col]
            if value > mask_value:
                value = mask_value
            result[row, col] = value

    total_pixels = height * width
    head = 0
    tail = 0
    queue_count = 0

    for row in range(height):
        for col in range(width):
            value = result[row, col]
            if row > 0:
                tail, queue_count = enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row - 1,
                    col,
                    value,
                )
            if row + 1 < height:
                tail, queue_count = enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row + 1,
                    col,
                    value,
                )
            if col > 0:
                tail, queue_count = enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row,
                    col - 1,
                    value,
                )
            if col + 1 < width:
                tail, queue_count = enqueue_reconstruct_neighbor(
                    result,
                    mask,
                    queued,
                    queue_rows,
                    queue_cols,
                    tail,
                    queue_count,
                    row,
                    col + 1,
                    value,
                )

    while queue_count > 0:
        row = queue_rows[head]
        col = queue_cols[head]
        head += 1
        if head == total_pixels:
            head = 0
        queue_count -= 1
        queued[row, col] = False
        value = result[row, col]
        if row > 0:
            tail, queue_count = propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row - 1,
                col,
                value,
            )
        if row + 1 < height:
            tail, queue_count = propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row + 1,
                col,
                value,
            )
        if col > 0:
            tail, queue_count = propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row,
                col - 1,
                value,
            )
        if col + 1 < width:
            tail, queue_count = propagate_reconstruct_neighbor(
                result,
                mask,
                queued,
                queue_rows,
                queue_cols,
                tail,
                queue_count,
                row,
                col + 1,
                value,
            )

    return result


@njit(cache=True)
def enqueue_reconstruct_neighbor(
    result: np.ndarray,
    mask: np.ndarray,
    queued: np.ndarray,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    tail: int,
    queue_count: int,
    row: int,
    col: int,
    source_value: float,
) -> tuple[int, int]:
    if (
        result[row, col] < source_value
        and result[row, col] < mask[row, col]
        and not queued[row, col]
    ):
        queue_rows[tail] = row
        queue_cols[tail] = col
        queued[row, col] = True
        tail += 1
        if tail == queue_rows.shape[0]:
            tail = 0
        queue_count += 1
    return tail, queue_count


@njit(cache=True)
def propagate_reconstruct_neighbor(
    result: np.ndarray,
    mask: np.ndarray,
    queued: np.ndarray,
    queue_rows: np.ndarray,
    queue_cols: np.ndarray,
    tail: int,
    queue_count: int,
    row: int,
    col: int,
    source_value: float,
) -> tuple[int, int]:
    if result[row, col] >= source_value or result[row, col] >= mask[row, col]:
        return tail, queue_count
    new_value = source_value
    if new_value > mask[row, col]:
        new_value = mask[row, col]
    if new_value <= result[row, col]:
        return tail, queue_count
    result[row, col] = new_value
    if not queued[row, col]:
        queue_rows[tail] = row
        queue_cols[tail] = col
        queued[row, col] = True
        tail += 1
        if tail == queue_rows.shape[0]:
            tail = 0
        queue_count += 1
    return tail, queue_count


@njit(cache=True)
def label_to_index_lookup_numba(object_ids: np.ndarray) -> np.ndarray:
    """Return dense label-id to object-index lookup for repeated scans."""
    max_label = 0
    for index in range(len(object_ids)):
        object_id = int(object_ids[index])
        if object_id > max_label:
            max_label = object_id
    label_to_index = np.full(max_label + 1, -1, dtype=np.int64)
    for index in range(len(object_ids)):
        object_id = int(object_ids[index])
        if object_id > 0:
            label_to_index[object_id] = index
    return label_to_index


@njit(cache=True)
def compact_label_pixels_from_lookup_numba(
    labels: np.ndarray,
    label_to_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return foreground label pixels once so spectrum iterations skip background."""
    object_count = 0
    for label_id in range(len(label_to_index)):
        index = int(label_to_index[label_id])
        if index >= object_count:
            object_count = index + 1

    max_label = len(label_to_index) - 1
    height, width = labels.shape
    foreground_count = 0
    object_counts = np.zeros(object_count, dtype=np.int64)
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index < 0:
                continue
            foreground_count += 1
            object_counts[index] += 1

    pixel_rows = np.empty(foreground_count, dtype=np.int64)
    pixel_cols = np.empty(foreground_count, dtype=np.int64)
    pixel_object_indices = np.empty(foreground_count, dtype=np.int64)
    out_index = 0
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            index = int(label_to_index[label_id])
            if index < 0:
                continue
            pixel_rows[out_index] = row
            pixel_cols[out_index] = col
            pixel_object_indices[out_index] = index
            out_index += 1
    return pixel_rows, pixel_cols, pixel_object_indices, object_counts


@njit(cache=True)
def mean_by_compact_label_pixels_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    pixel_object_indices: np.ndarray,
    object_counts: np.ndarray,
) -> np.ndarray:
    """Return object means by scanning only cached foreground label pixels."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    for pixel_index in range(len(pixel_object_indices)):
        object_index = int(pixel_object_indices[pixel_index])
        sums[object_index] += float(
            image[int(pixel_rows[pixel_index]), int(pixel_cols[pixel_index])]
        )

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def mean_by_compact_label_pixels_from_resampled_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    pixel_object_indices: np.ndarray,
    object_counts: np.ndarray,
    row_scale: float,
    col_scale: float,
) -> np.ndarray:
    """Mean resampled values using cached foreground label pixels only."""
    object_count = len(object_counts)
    means = np.empty(object_count, dtype=np.float64)
    sums = np.zeros(object_count, dtype=np.float64)
    for pixel_index in range(len(pixel_object_indices)):
        object_index = int(pixel_object_indices[pixel_index])
        sums[object_index] += bilinear_sample_numba(
            image,
            float(pixel_rows[pixel_index]) * row_scale,
            float(pixel_cols[pixel_index]) * col_scale,
        )

    for index in range(object_count):
        count = object_counts[index]
        means[index] = sums[index] / count if count > 0 else np.nan
    return means


@njit(cache=True)
def bilinear_sample_numba(
    image: np.ndarray,
    row_coord: float,
    col_coord: float,
) -> float:
    height, width = image.shape
    row0 = int(np.floor(row_coord))
    col0 = int(np.floor(col_coord))
    if row0 < 0:
        row0 = 0
    if col0 < 0:
        col0 = 0
    if row0 >= height:
        row0 = height - 1
    if col0 >= width:
        col0 = width - 1
    row1 = row0 + 1
    col1 = col0 + 1
    if row1 >= height:
        row1 = row0
    if col1 >= width:
        col1 = col0

    row_weight = row_coord - row0
    col_weight = col_coord - col0
    top = (
        float(image[row0, col0]) * (1.0 - col_weight)
        + float(image[row0, col1]) * col_weight
    )
    bottom = (
        float(image[row1, col0]) * (1.0 - col_weight)
        + float(image[row1, col1]) * col_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "granularity_measurements",
        csv_materializer(
            fields=GRANULARITY_FIELDS,
            analysis_type="granularity",
        ),
    )
)
def measure_granularity(
    image: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> tuple[np.ndarray, GranularityMeasurement]:
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
        if index > 0 and gs < 0.0:
            gs = 0.0
        gs_values.append(gs)

    return image, _granularity_measurement(gs_values)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "object_granularity_measurements",
        csv_materializer(
            fields=["slice_index", "object_id", *GRANULARITY_FIELDS[1:]],
            analysis_type="object_granularity",
        ),
    )
)
def measure_granularity_objects(
    image: np.ndarray,
    labels: np.ndarray,
    subsample_size: float = 0.25,
    background_subsample_size: float = 0.25,
    element_radius: int = 10,
    spectrum_length: int = 16,
) -> tuple[np.ndarray, list[ObjectGranularityMeasurement]]:
    """Measure granularity spectrum within labeled objects."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    object_range = np.unique(labels[labels > 0]).astype(np.int32, copy=False)
    if object_range.size == 0:
        return image, []

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
    return image, [
        _object_granularity_measurement(int(object_id), gs_per_object[index])
        for index, object_id in enumerate(object_range)
    ]


def _prepare_granularity_backend() -> None:
    """Compile Numba kernels used by the granularity backend before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
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


measure_granularity.__openhcs_prepare__ = _prepare_granularity_backend
measure_granularity_objects.__openhcs_prepare__ = _prepare_granularity_backend


__all__ = [
    "GRANULARITY_IMAGE_SERIES_CACHE",
    "GRANULARITY_IMAGE_SERIES_CACHE_MAX_ENTRIES",
    "GRANULARITY_LABEL_PIXELS_CACHE",
    "GRANULARITY_LABEL_PIXELS_CACHE_MAX_ENTRIES",
    "GRANULARITY_FIELDS",
    "GranularityImageSeries",
    "GranularityImageSeriesCacheEntry",
    "GranularityImageSeriesRequest",
    "GranularityLabelPixels",
    "GranularityLabelPixelsCacheEntry",
    "GranularityLabelPixelsRequest",
    "GranularityMeasurement",
    "ObjectGranularityMeasurement",
    "background_corrected_pixels",
    "bilinear_sample_numba",
    "compact_label_pixels_from_lookup_numba",
    "disk_offsets",
    "granularity_image_content_key",
    "granularity_reconstruction_series",
    "gray_dilation_offsets_reflect_numba",
    "gray_erosion_offsets_reflect_numba",
    "label_to_index_lookup_numba",
    "log_profile",
    "measure_granularity",
    "measure_granularity_objects",
    "mean_by_compact_label_pixels_from_resampled_numba",
    "mean_by_compact_label_pixels_numba",
    "object_granularity_values",
    "reconstruct_dilation_cross_numba",
]
