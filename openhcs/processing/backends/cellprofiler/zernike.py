"""Zernike backend strategies for CellProfiler-compatible measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_INTENSITY_DEBUG_TRACE_DIR_ENV = "OPENHCS_ZERNIKE_INTENSITY_DEBUG_TRACE_DIR"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True)
class _ZernikeLabelGeometry:
    centers: np.ndarray
    radii: np.ndarray
    y_coords: np.ndarray
    x_coords: np.ndarray
    label_values: np.ndarray
    raw_label_values: np.ndarray


@dataclass(frozen=True)
class ZernikeIntensityDebugTrace:
    """Object-indexed Zernike state emitted only when debug tracing is enabled."""

    backend_provider: CellProfilerBackendProvider
    image_shape: tuple[int, ...]
    image_dtype: str
    image_digest: bytes
    labels_shape: tuple[int, ...]
    labels_dtype: str
    labels_digest: bytes
    max_order: int
    object_ids: np.ndarray
    zernike_numbers: tuple[tuple[int, int], ...]
    centers: np.ndarray
    radii: np.ndarray
    areas: np.ndarray
    y_coords: np.ndarray
    x_coords: np.ndarray
    label_values: np.ndarray
    pixel_values: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray

    @classmethod
    def from_intensity_measurement(
        cls,
        *,
        backend_provider: CellProfilerBackendProvider,
        image: np.ndarray,
        labels: np.ndarray,
        max_order: int,
        object_ids: np.ndarray,
        zernike_numbers: tuple[tuple[int, int], ...],
        centers: np.ndarray,
        radii: np.ndarray,
        areas: np.ndarray,
        y_coords: np.ndarray,
        x_coords: np.ndarray,
        label_values: np.ndarray,
        pixel_values: np.ndarray,
        magnitudes: np.ndarray,
        phases: np.ndarray,
    ) -> "ZernikeIntensityDebugTrace":
        image_key = _array_content_key(image)
        labels_key = _array_content_key(labels)
        return cls(
            backend_provider=backend_provider,
            image_shape=tuple(int(value) for value in image.shape),
            image_dtype=image_key[0],
            image_digest=image_key[2],
            labels_shape=tuple(int(value) for value in labels.shape),
            labels_dtype=labels_key[0],
            labels_digest=labels_key[2],
            max_order=int(max_order),
            object_ids=np.ascontiguousarray(object_ids, dtype=np.int32),
            zernike_numbers=zernike_numbers,
            centers=np.ascontiguousarray(centers, dtype=np.float64),
            radii=np.ascontiguousarray(radii, dtype=np.float64),
            areas=np.ascontiguousarray(areas, dtype=np.float64),
            y_coords=np.ascontiguousarray(y_coords, dtype=np.int64),
            x_coords=np.ascontiguousarray(x_coords, dtype=np.int64),
            label_values=np.ascontiguousarray(label_values, dtype=np.int32),
            pixel_values=np.ascontiguousarray(pixel_values, dtype=np.float64),
            magnitudes=np.ascontiguousarray(magnitudes, dtype=np.float64),
            phases=np.ascontiguousarray(phases, dtype=np.float64),
        )

    def write_if_enabled(self) -> Path | None:
        trace_dir_text = os.environ.get(_INTENSITY_DEBUG_TRACE_DIR_ENV)
        if trace_dir_text is None or not trace_dir_text.strip():
            return None
        trace_dir = Path(trace_dir_text)
        trace_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"zernike_intensity_{os.getpid()}_{time.time_ns()}_"
            f"{self.backend_provider.value}_{self.object_ids.size}_"
            f"{self.max_order}_{self.image_digest.hex()}_"
            f"{self.labels_digest.hex()}.pkl"
        )
        path = trace_dir / filename
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path


_ZERNIKE_LABEL_GEOMETRY_CACHE: OrderedDict[
    tuple[str, tuple[int, ...], bytes, str, tuple[int, ...], bytes],
    _ZernikeLabelGeometry,
] = OrderedDict()
_ZERNIKE_LABEL_GEOMETRY_CACHE_MAX_ENTRIES = 16


class ShapeZernikeBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shape Zernike moment backends keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
        """Return Zernike indexes and dense-label moment values."""

    @abstractmethod
    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
        """Return intensity-weighted Zernike magnitudes and phases."""


class CentrosomeNumpyShapeZernikeBackendStrategy(ShapeZernikeBackendStrategy):
    """Centrosome-backed NumPy implementation matching legacy semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
        import centrosome.zernike

        labels_array = np.asarray(labels)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = centrosome.zernike.get_zernike_indexes(
            int(max_order) + 1
        )
        zernike_numbers = tuple(
            (int(n), int(m))
            for n, m in zernike_numbers_array
        )
        zernike_values = centrosome.zernike.zernike(
            zernike_numbers_array,
            labels_array,
            measured_label_ids,
        )
        return zernike_numbers, np.asarray(zernike_values)

    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
        import centrosome.cpmorphology
        import centrosome.zernike
        from scipy import ndimage as ndi

        image_array = np.asarray(image, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int32)
        object_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = centrosome.zernike.get_zernike_indexes(
            int(max_order) + 1
        )
        zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
        if object_ids.size == 0 or zernike_numbers_array.size == 0:
            return (
                zernike_numbers,
                np.zeros((object_ids.size, len(zernike_numbers)), dtype=float),
                np.zeros((object_ids.size, len(zernike_numbers)), dtype=float),
            )

        centers, radii = centrosome.cpmorphology.minimum_enclosing_circle(
            labels_array,
            object_ids,
        )
        y_coords, x_coords = np.nonzero(labels_array > 0)
        if y_coords.size:
            label_values = labels_array[y_coords, x_coords]
            valid = (
                (label_values > 0)
                & (label_values <= object_ids.size)
                & np.isfinite(radii[label_values - 1])
                & (radii[label_values - 1] > 0)
            )
            y_coords = y_coords[valid]
            x_coords = x_coords[valid]
            label_values = label_values[valid]

        if not y_coords.size:
            return (
                zernike_numbers,
                np.full((object_ids.size, len(zernike_numbers)), np.nan),
                np.full((object_ids.size, len(zernike_numbers)), np.nan),
            )

        label_indexes = label_values - 1
        yx = (
            np.column_stack((y_coords, x_coords)).astype(np.float64)
            - centers[label_indexes]
        ) / radii[label_indexes, np.newaxis]
        polynomials = centrosome.zernike.construct_zernike_polynomials(
            yx[:, 1],
            yx[:, 0],
            zernike_numbers_array,
        )
        areas = ndi.sum(
            np.ones(label_values.shape, dtype=np.int32),
            labels=label_values,
            index=object_ids,
        )
        pixel_values = image_array[y_coords, x_coords]
        magnitudes = np.empty((object_ids.size, zernike_numbers_array.shape[0]))
        phases = np.empty((object_ids.size, zernike_numbers_array.shape[0]))
        for zernike_index in range(zernike_numbers_array.shape[0]):
            real_sum = ndi.sum(
                pixel_values * polynomials[:, zernike_index].real,
                labels=label_values,
                index=object_ids,
            )
            imag_sum = ndi.sum(
                pixel_values * polynomials[:, zernike_index].imag,
                labels=label_values,
                index=object_ids,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                magnitudes[:, zernike_index] = (
                    np.sqrt(real_sum * real_sum + imag_sum * imag_sum) / areas
                )
            phases[:, zernike_index] = np.arctan2(real_sum, imag_sum)

        ZernikeIntensityDebugTrace.from_intensity_measurement(
            backend_provider=self.backend_provider,
            image=image_array,
            labels=labels_array,
            max_order=max_order,
            object_ids=object_ids,
            zernike_numbers=zernike_numbers,
            centers=centers,
            radii=radii,
            areas=areas,
            y_coords=y_coords,
            x_coords=x_coords,
            label_values=label_values,
            pixel_values=pixel_values,
            magnitudes=magnitudes,
            phases=phases,
        ).write_if_enabled()

        return zernike_numbers, magnitudes, phases


class LegacyFastNumpyShapeZernikeBackendStrategy(ShapeZernikeBackendStrategy):
    """Mixed legacy-fast Zernike backend with explicit centrosome exact leaves."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.LEGACY_FAST,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = True

    def shape_zernike_moments(
        self,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
        labels_array = np.asarray(labels, dtype=np.int32)
        if labels_array.size == 0 or int(labels_array.max()) <= 0:
            zernike_numbers_array = _zernike_indexes_array(int(max_order))
            zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
            return zernike_numbers, np.zeros((0, len(zernike_numbers)), dtype=float)

        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = _zernike_indexes_array(int(max_order))
        zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
        if zernike_numbers_array.size == 0:
            return zernike_numbers, np.zeros((measured_label_ids.size, 0), dtype=float)

        geometry = _zernike_label_geometry(
            labels_array,
            measured_label_ids,
        )
        if geometry.y_coords.size == 0:
            return zernike_numbers, np.zeros(
                (measured_label_ids.size, zernike_numbers_array.shape[0]),
                dtype=float,
            )

        centers = geometry.centers
        radii = geometry.radii
        y_coords = geometry.y_coords
        x_coords = geometry.x_coords
        label_values = geometry.label_values
        valid = (
            (label_values > 0)
            & (label_values <= measured_label_ids.size)
            & np.isfinite(radii[label_values - 1])
            & (radii[label_values - 1] > 0)
        )
        y_coords = np.ascontiguousarray(y_coords[valid], dtype=np.float64)
        x_coords = np.ascontiguousarray(x_coords[valid], dtype=np.float64)
        label_values = np.ascontiguousarray(label_values[valid], dtype=np.int32)
        if y_coords.size == 0:
            return zernike_numbers, np.zeros(
                (measured_label_ids.size, zernike_numbers_array.shape[0]),
                dtype=float,
            )

        coefficients, exponents, term_counts = _zernike_radial_terms(
            zernike_numbers_array
        )
        denominators = np.ascontiguousarray(np.pi * radii * radii, dtype=np.float64)
        score_started_at = time.perf_counter()
        values = _score_zernike_moments_direct_numba(
            np.ascontiguousarray(label_values, dtype=np.int32),
            np.ascontiguousarray(y_coords, dtype=np.float64),
            np.ascontiguousarray(x_coords, dtype=np.float64),
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(zernike_numbers_array, dtype=np.int64),
            coefficients,
            exponents,
            term_counts,
            denominators,
            int(measured_label_ids.size),
        )
        _log_profile(
            "zernike_shape_score",
            time.perf_counter() - score_started_at,
            objects=int(measured_label_ids.size),
            pixels=int(y_coords.size),
            orders=zernike_numbers_array.shape[0],
        )
        return zernike_numbers, values

    def intensity_zernike_moments(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        measured_labels: np.ndarray,
        *,
        max_order: int,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
        image_array = np.asarray(image, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int32)
        measured_label_ids = np.asarray(measured_labels, dtype=np.int32)
        zernike_numbers_array = _zernike_indexes_array(int(max_order))
        zernike_numbers = tuple((int(n), int(m)) for n, m in zernike_numbers_array)
        if measured_label_ids.size == 0 or zernike_numbers_array.size == 0:
            return (
                zernike_numbers,
                np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
                np.zeros((measured_label_ids.size, len(zernike_numbers)), dtype=float),
            )

        geometry = _zernike_label_geometry(
            labels_array,
            measured_label_ids,
        )
        if geometry.y_coords.size == 0:
            return (
                zernike_numbers,
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
            )

        centers = geometry.centers
        radii = geometry.radii
        y_coords = geometry.y_coords.astype(np.int64, copy=False)
        x_coords = geometry.x_coords.astype(np.int64, copy=False)
        label_values = geometry.label_values
        raw_label_values = geometry.raw_label_values
        valid = (
            (label_values > 0)
            & (label_values <= measured_label_ids.size)
            & np.isfinite(radii[label_values - 1])
            & (radii[label_values - 1] > 0)
        )
        y_coords = np.ascontiguousarray(y_coords[valid], dtype=np.int64)
        x_coords = np.ascontiguousarray(x_coords[valid], dtype=np.int64)
        label_values = np.ascontiguousarray(label_values[valid], dtype=np.int32)
        raw_label_values = np.ascontiguousarray(
            raw_label_values[valid],
            dtype=np.int32,
        )
        if y_coords.size == 0:
            return (
                zernike_numbers,
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
                np.full((measured_label_ids.size, len(zernike_numbers)), np.nan),
            )

        coefficients, exponents, term_counts = _zernike_radial_terms(
            zernike_numbers_array
        )
        score_started_at = time.perf_counter()
        magnitudes, phases = _score_intensity_zernike_moments_direct_numba(
            np.ascontiguousarray(image_array, dtype=np.float64),
            label_values,
            raw_label_values,
            y_coords,
            x_coords,
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(zernike_numbers_array, dtype=np.int64),
            coefficients,
            exponents,
            term_counts,
            int(measured_label_ids.size),
        )
        _log_profile(
            "zernike_intensity_score",
            time.perf_counter() - score_started_at,
            objects=int(measured_label_ids.size),
            pixels=int(y_coords.size),
            orders=zernike_numbers_array.shape[0],
        )
        ZernikeIntensityDebugTrace.from_intensity_measurement(
            backend_provider=self.backend_provider,
            image=image_array,
            labels=labels_array,
            max_order=max_order,
            object_ids=measured_label_ids,
            zernike_numbers=zernike_numbers,
            centers=centers,
            radii=radii,
            areas=np.bincount(
                raw_label_values,
                minlength=measured_label_ids.size + 1,
            )[1:].astype(np.float64),
            y_coords=y_coords,
            x_coords=x_coords,
            label_values=label_values,
            pixel_values=image_array[y_coords, x_coords],
            magnitudes=magnitudes,
            phases=phases,
        ).write_if_enabled()
        return zernike_numbers, magnitudes, phases


def shape_zernike_moments(
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    max_order: int,
    backend_provider: BackendProviderInput | None = None,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Return shape Zernike moments through the selected backend."""
    return ShapeZernikeBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).shape_zernike_moments(
        labels,
        measured_labels,
        max_order=max_order,
    )


def intensity_zernike_moments(
    image: np.ndarray,
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    max_order: int,
    backend_provider: BackendProviderInput | None = None,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """Return intensity-weighted Zernike moments through the selected backend."""
    return ShapeZernikeBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).intensity_zernike_moments(
        image,
        labels,
        measured_labels,
        max_order=max_order,
    )


def _zernike_label_geometry(
    labels: np.ndarray,
    object_ids: np.ndarray,
) -> _ZernikeLabelGeometry:
    """Return exact cached label geometry shared by shape/intensity Zernikes."""
    import centrosome.cpmorphology

    total_started_at = time.perf_counter()
    labels_array = np.ascontiguousarray(labels, dtype=np.int32)
    object_ids_array = np.ascontiguousarray(object_ids, dtype=np.int32)
    key_started_at = time.perf_counter()
    key = (*_array_content_key(labels_array), *_array_content_key(object_ids_array))
    _log_profile(
        "zernike_geometry_key",
        time.perf_counter() - key_started_at,
        objects=object_ids_array.size,
    )
    entry = _ZERNIKE_LABEL_GEOMETRY_CACHE.get(key)
    if entry is not None:
        _ZERNIKE_LABEL_GEOMETRY_CACHE.move_to_end(key)
        _log_profile(
            "zernike_geometry_cache_hit",
            time.perf_counter() - total_started_at,
            objects=object_ids_array.size,
        )
        return entry

    circle_started_at = time.perf_counter()
    centers, radii = centrosome.cpmorphology.minimum_enclosing_circle(
        labels_array,
        object_ids_array,
    )
    _log_profile(
        "zernike_geometry_min_enclosing_circle",
        time.perf_counter() - circle_started_at,
        objects=object_ids_array.size,
    )
    compact_started_at = time.perf_counter()
    y_coords, x_coords = np.nonzero(labels_array > 0)
    label_to_row = np.zeros(int(labels_array.max(initial=0)) + 1, dtype=np.int32)
    valid_object_ids = object_ids_array[
        (object_ids_array > 0) & (object_ids_array < label_to_row.size)
    ]
    label_to_row[valid_object_ids] = np.arange(
        1,
        valid_object_ids.size + 1,
        dtype=np.int32,
    )
    label_values = label_to_row[labels_array[y_coords, x_coords]]
    raw_label_values = np.ascontiguousarray(
        labels_array[y_coords, x_coords],
        dtype=np.int32,
    )
    _log_profile(
        "zernike_geometry_compact_pixels",
        time.perf_counter() - compact_started_at,
        pixels=label_values.size,
    )
    geometry = _ZernikeLabelGeometry(
        centers=np.ascontiguousarray(centers, dtype=np.float64),
        radii=np.ascontiguousarray(radii, dtype=np.float64),
        y_coords=np.ascontiguousarray(y_coords, dtype=np.float64),
        x_coords=np.ascontiguousarray(x_coords, dtype=np.float64),
        label_values=np.ascontiguousarray(label_values, dtype=np.int32),
        raw_label_values=raw_label_values,
    )
    _ZERNIKE_LABEL_GEOMETRY_CACHE[key] = geometry
    _ZERNIKE_LABEL_GEOMETRY_CACHE.move_to_end(key)
    while len(_ZERNIKE_LABEL_GEOMETRY_CACHE) > _ZERNIKE_LABEL_GEOMETRY_CACHE_MAX_ENTRIES:
        _ZERNIKE_LABEL_GEOMETRY_CACHE.popitem(last=False)
    _log_profile(
        "zernike_geometry_total",
        time.perf_counter() - total_started_at,
        objects=object_ids_array.size,
        pixels=label_values.size,
    )
    return geometry


def _array_content_key(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return str(contiguous.dtype), tuple(int(value) for value in contiguous.shape), digest


def _zernike_indexes_array(max_order: int) -> np.ndarray:
    indexes: list[tuple[int, int]] = []
    for n_value in range(0, int(max_order) + 1):
        for m_value in range(n_value % 2, n_value + 1, 2):
            indexes.append((n_value, m_value))
    return np.asarray(indexes, dtype=np.int64)


__all__ = [
    "CentrosomeNumpyShapeZernikeBackendStrategy",
    "LegacyFastNumpyShapeZernikeBackendStrategy",
    "ShapeZernikeBackendStrategy",
    "ZernikeIntensityDebugTrace",
    "intensity_zernike_moments",
    "shape_zernike_moments",
]


@njit(cache=True)
def _score_zernike_moments_direct_numba(
    label_values: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    zernike_numbers: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    term_counts: np.ndarray,
    denominators: np.ndarray,
    object_count: int,
) -> np.ndarray:
    zernike_count = zernike_numbers.shape[0]
    real_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    imag_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    max_order = 0
    for zernike_index in range(zernike_count):
        n_value = int(zernike_numbers[zernike_index, 0])
        m_value = abs(int(zernike_numbers[zernike_index, 1]))
        if n_value > max_order:
            max_order = n_value
        if m_value > max_order:
            max_order = m_value
    rho_powers = np.empty(max_order + 1, dtype=np.float64)
    cos_by_m = np.empty(max_order + 1, dtype=np.float64)
    sin_by_m = np.empty(max_order + 1, dtype=np.float64)
    for pixel_index in range(label_values.size):
        object_index = label_values[pixel_index] - 1
        if object_index < 0 or object_index >= object_count:
            continue
        radius = radii[object_index]
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        normalized_y = (y_coords[pixel_index] - centers[object_index, 0]) / radius
        normalized_x = (x_coords[pixel_index] - centers[object_index, 1]) / radius
        rho_squared = normalized_x * normalized_x + normalized_y * normalized_y
        if rho_squared > 1.0:
            continue
        rho = np.sqrt(rho_squared)
        rho_powers[0] = 1.0
        for order in range(1, max_order + 1):
            rho_powers[order] = rho_powers[order - 1] * rho

        cos_by_m[0] = 1.0
        sin_by_m[0] = 0.0
        if max_order > 0:
            if rho > 0.0:
                cos_theta = normalized_y / rho
                sin_theta = normalized_x / rho
            else:
                cos_theta = 1.0
                sin_theta = 0.0
            cos_by_m[1] = cos_theta
            sin_by_m[1] = sin_theta
            for order in range(2, max_order + 1):
                cos_by_m[order] = (
                    cos_by_m[order - 1] * cos_theta
                    - sin_by_m[order - 1] * sin_theta
                )
                sin_by_m[order] = (
                    sin_by_m[order - 1] * cos_theta
                    + cos_by_m[order - 1] * sin_theta
                )
        for zernike_index in range(zernike_count):
            radial = 0.0
            for term_index in range(term_counts[zernike_index]):
                radial += (
                    coefficients[zernike_index, term_index]
                    * rho_powers[exponents[zernike_index, term_index]]
                )
            m = abs(zernike_numbers[zernike_index, 1])
            real_sums[object_index, zernike_index] += radial * cos_by_m[m]
            imag_sums[object_index, zernike_index] += radial * sin_by_m[m]

    output = np.empty((object_count, zernike_count), dtype=np.float64)
    for object_index in range(object_count):
        denominator = denominators[object_index]
        if not np.isfinite(denominator) or denominator <= 0.0:
            for zernike_index in range(zernike_count):
                output[object_index, zernike_index] = np.nan
            continue
        for zernike_index in range(zernike_count):
            real_value = real_sums[object_index, zernike_index]
            imag_value = imag_sums[object_index, zernike_index]
            output[object_index, zernike_index] = (
                np.sqrt(real_value * real_value + imag_value * imag_value)
                / denominator
            )
    return output


@njit(cache=True)
def _score_intensity_zernike_moments_direct_numba(
    image: np.ndarray,
    label_values: np.ndarray,
    raw_label_values: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    zernike_numbers: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    term_counts: np.ndarray,
    object_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    zernike_count = zernike_numbers.shape[0]
    real_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    imag_sums = np.zeros((object_count, zernike_count), dtype=np.float64)
    areas = np.zeros(object_count, dtype=np.float64)
    max_order = 0
    for zernike_index in range(zernike_count):
        n_value = int(zernike_numbers[zernike_index, 0])
        m_value = abs(int(zernike_numbers[zernike_index, 1]))
        if n_value > max_order:
            max_order = n_value
        if m_value > max_order:
            max_order = m_value
    rho_powers = np.empty(max_order + 1, dtype=np.float64)
    cos_by_m = np.empty(max_order + 1, dtype=np.float64)
    sin_by_m = np.empty(max_order + 1, dtype=np.float64)
    for pixel_index in range(label_values.size):
        object_index = label_values[pixel_index] - 1
        if object_index < 0 or object_index >= object_count:
            continue
        raw_label_value = raw_label_values[pixel_index]
        raw_object_index = raw_label_value - 1
        if raw_object_index < 0 or raw_object_index >= object_count:
            continue
        radius = radii[object_index]
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        y = y_coords[pixel_index]
        x = x_coords[pixel_index]
        areas[raw_object_index] += 1.0
        normalized_y = (y - centers[object_index, 0]) / radius
        normalized_x = (x - centers[object_index, 1]) / radius
        rho_squared = normalized_x * normalized_x + normalized_y * normalized_y
        if rho_squared > 1.0:
            continue
        rho = np.sqrt(rho_squared)
        rho_powers[0] = 1.0
        for order in range(1, max_order + 1):
            rho_powers[order] = rho_powers[order - 1] * rho

        cos_by_m[0] = 1.0
        sin_by_m[0] = 0.0
        if max_order > 0:
            if rho > 0.0:
                cos_theta = normalized_y / rho
                sin_theta = normalized_x / rho
            else:
                cos_theta = 1.0
                sin_theta = 0.0
            cos_by_m[1] = cos_theta
            sin_by_m[1] = sin_theta
            for order in range(2, max_order + 1):
                cos_by_m[order] = (
                    cos_by_m[order - 1] * cos_theta
                    - sin_by_m[order - 1] * sin_theta
                )
                sin_by_m[order] = (
                    sin_by_m[order - 1] * cos_theta
                    + cos_by_m[order - 1] * sin_theta
                )
        pixel_value = image[y, x]
        for zernike_index in range(zernike_count):
            radial = 0.0
            for term_index in range(term_counts[zernike_index]):
                radial += (
                    coefficients[zernike_index, term_index]
                    * rho_powers[exponents[zernike_index, term_index]]
                )
            m = abs(zernike_numbers[zernike_index, 1])
            real_sums[raw_object_index, zernike_index] += (
                pixel_value * radial * cos_by_m[m]
            )
            imag_sums[raw_object_index, zernike_index] += (
                pixel_value * radial * sin_by_m[m]
            )

    magnitudes = np.empty((object_count, zernike_count), dtype=np.float64)
    phases = np.empty((object_count, zernike_count), dtype=np.float64)
    for object_index in range(object_count):
        area = areas[object_index]
        for zernike_index in range(zernike_count):
            real_value = real_sums[object_index, zernike_index]
            imag_value = imag_sums[object_index, zernike_index]
            if area <= 0.0:
                magnitudes[object_index, zernike_index] = np.nan
            else:
                magnitudes[object_index, zernike_index] = (
                    np.sqrt(real_value * real_value + imag_value * imag_value) / area
                )
            phases[object_index, zernike_index] = np.arctan2(
                real_value,
                imag_value,
            )

    return magnitudes, phases


def _zernike_radial_terms(
    zernike_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial-polynomial coefficients in Numba-friendly dense arrays."""
    numbers = np.asarray(zernike_numbers, dtype=np.int64)
    max_terms = 1
    for n_value, m_value in numbers:
        max_terms = max(max_terms, (int(n_value) - abs(int(m_value))) // 2 + 1)

    coefficients = np.zeros((numbers.shape[0], max_terms), dtype=np.float64)
    exponents = np.zeros((numbers.shape[0], max_terms), dtype=np.int64)
    term_counts = np.zeros(numbers.shape[0], dtype=np.int64)
    for zernike_index, (n_value, m_value) in enumerate(numbers):
        n = int(n_value)
        m = abs(int(m_value))
        term_count = (n - m) // 2 + 1
        term_counts[zernike_index] = term_count
        for s in range(term_count):
            coefficients[zernike_index, s] = (
                (-1.0 if s % 2 else 1.0)
                * float(math.factorial(n - s))
                / (
                    float(math.factorial(s))
                    * float(math.factorial((n + m) // 2 - s))
                    * float(math.factorial((n - m) // 2 - s))
                )
            )
            exponents[zernike_index, s] = n - 2 * s
    return (
        np.ascontiguousarray(coefficients),
        np.ascontiguousarray(exponents),
        np.ascontiguousarray(term_counts),
    )
