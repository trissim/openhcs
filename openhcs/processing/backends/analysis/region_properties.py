"""Reusable dense-label region-property backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType


class AnalysisBackendProvider(str, Enum):
    """Typed provider identifiers for reusable OpenHCS analysis primitives."""

    NUMBA = "numba"


DEFAULT_ANALYSIS_BACKEND_PROVIDER = AnalysisBackendProvider.NUMBA
_BACKEND_KEY_SEPARATOR = ":"
BackendProviderInput = AnalysisBackendProvider
BackendStrategyT = TypeVar(
    "BackendStrategyT",
    bound="AnalysisBackendStrategyMixin",
)


def _normalize_memory_type(memory_type: MemoryType | str = MemoryType.NUMPY) -> MemoryType:
    return memory_type if isinstance(memory_type, MemoryType) else MemoryType(str(memory_type))


def _normalize_backend_provider(
    backend_provider: BackendProviderInput = DEFAULT_ANALYSIS_BACKEND_PROVIDER,
) -> AnalysisBackendProvider:
    if not isinstance(backend_provider, AnalysisBackendProvider):
        raise TypeError("Analysis backend provider must be an AnalysisBackendProvider enum value")
    return backend_provider


def analysis_backend_key(
    memory_type: MemoryType | str = MemoryType.NUMPY,
    backend_provider: BackendProviderInput = DEFAULT_ANALYSIS_BACKEND_PROVIDER,
) -> str:
    """Return the registry key for one reusable analysis backend implementation."""
    provider = _normalize_backend_provider(backend_provider)
    return _normalize_memory_type(memory_type).value + _BACKEND_KEY_SEPARATOR + provider.value


class AnalysisBackendStrategyMixin:
    """Backend lookup for reusable OpenHCS analysis primitives."""

    backend_key: ClassVar[str | None] = None
    memory_type: ClassVar[MemoryType | None] = None
    backend_provider: ClassVar[AnalysisBackendProvider] = DEFAULT_ANALYSIS_BACKEND_PROVIDER
    is_default_backend: ClassVar[bool] = False

    @classmethod
    def for_memory_type(
        cls: type[BackendStrategyT],
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput | None = None,
    ) -> BackendStrategyT:
        return cls._resolve_backend_class(memory_type, backend_provider)()

    @classmethod
    def available_backend_providers(
        cls,
        memory_type: MemoryType | str | None = None,
    ) -> tuple[AnalysisBackendProvider, ...]:
        resolved = None if memory_type is None else _normalize_memory_type(memory_type)
        providers: list[AnalysisBackendProvider] = []
        for strategy_cls in getattr(cls, "__registry__", {}).values():
            if resolved is not None and strategy_cls.memory_type is not resolved:
                continue
            providers.append(_normalize_backend_provider(strategy_cls.backend_provider))
        return tuple(sorted(set(providers), key=lambda provider: provider.value))

    @classmethod
    def _resolve_backend_class(
        cls: type[BackendStrategyT],
        memory_type: MemoryType | str,
        backend_provider: BackendProviderInput | None,
    ) -> type[BackendStrategyT]:
        resolved = _normalize_memory_type(memory_type)
        registry: dict[str, type[BackendStrategyT]] = getattr(cls, "__registry__", {})
        if backend_provider is not None:
            provider = _normalize_backend_provider(backend_provider)
            key = analysis_backend_key(resolved, provider)
            if key not in registry:
                raise NotImplementedError(
                    f"No {cls.__name__} backend is registered for memory type "
                    f"{resolved.value!r} and provider {provider.value!r}. Registered "
                    f"providers for this memory type: "
                    f"{cls.available_backend_providers(resolved)!r}."
                )
            return registry[key]

        matches = [
            strategy_cls
            for strategy_cls in registry.values()
            if strategy_cls.memory_type is resolved
            and bool(strategy_cls.is_default_backend)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise NotImplementedError(
                f"No default {cls.__name__} backend is registered for memory type "
                f"{resolved.value!r}. Registered providers for this memory type: "
                f"{cls.available_backend_providers(resolved)!r}."
            )
        raise RuntimeError(
            f"Multiple default {cls.__name__} backends are registered for memory "
            f"type {resolved.value!r}: "
            f"{tuple(strategy.__name__ for strategy in matches)!r}."
        )


@dataclass(frozen=True, slots=True)
class DenseLabelRegionProperties:
    """Dense per-label properties indexed by positive label order."""

    label: np.ndarray
    area: np.ndarray
    perimeter: np.ndarray
    bbox_min_y: np.ndarray
    bbox_min_x: np.ndarray
    bbox_max_y: np.ndarray
    bbox_max_x: np.ndarray
    bbox_area: np.ndarray
    centroid_y: np.ndarray
    centroid_x: np.ndarray
    equivalent_diameter: np.ndarray
    extent: np.ndarray
    major_axis_length: np.ndarray
    minor_axis_length: np.ndarray
    eccentricity: np.ndarray
    orientation: np.ndarray
    euler_number: np.ndarray
    moments: np.ndarray
    moments_central: np.ndarray
    moments_normalized: np.ndarray
    moments_hu: np.ndarray
    inertia_tensor: np.ndarray
    inertia_tensor_eigvals: np.ndarray

    def as_regionprops_table_subset(self) -> dict[str, np.ndarray]:
        """Return keys matching skimage.regionprops_table for covered fields."""
        props: dict[str, np.ndarray] = {
            "label": self.label,
            "area": self.area,
            "perimeter": self.perimeter,
            "bbox-0": self.bbox_min_y,
            "bbox-1": self.bbox_min_x,
            "bbox-2": self.bbox_max_y,
            "bbox-3": self.bbox_max_x,
            "bbox_area": self.bbox_area,
            "centroid-0": self.centroid_y,
            "centroid-1": self.centroid_x,
            "equivalent_diameter": self.equivalent_diameter,
            "extent": self.extent,
            "major_axis_length": self.major_axis_length,
            "minor_axis_length": self.minor_axis_length,
            "eccentricity": self.eccentricity,
            "orientation": self.orientation,
            "euler_number": self.euler_number,
        }
        for row in range(4):
            for column in range(4):
                props[f"moments-{row}-{column}"] = self.moments[:, row, column]
                props[f"moments_central-{row}-{column}"] = self.moments_central[
                    :, row, column
                ]
                props[f"moments_normalized-{row}-{column}"] = (
                    self.moments_normalized[:, row, column]
                )
        for index in range(7):
            props[f"moments_hu-{index}"] = self.moments_hu[:, index]
        for row in range(2):
            for column in range(2):
                props[f"inertia_tensor-{row}-{column}"] = self.inertia_tensor[
                    :, row, column
                ]
        for index in range(2):
            props[f"inertia_tensor_eigvals-{index}"] = self.inertia_tensor_eigvals[
                :, index
            ]
        return props


class LabelRegionPropertiesBackendStrategy(
    AnalysisBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Dense label-region properties keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure_2d(self, labels: np.ndarray) -> DenseLabelRegionProperties:
        """Measure reusable 2-D dense-label region properties."""


class NumbaNumpyLabelRegionPropertiesBackendStrategy(
    LabelRegionPropertiesBackendStrategy
):
    """Numba-accelerated NumPy dense-label region properties."""

    backend_key = analysis_backend_key(
        MemoryType.NUMPY,
        AnalysisBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = AnalysisBackendProvider.NUMBA
    is_default_backend = True

    def measure_2d(self, labels: np.ndarray) -> DenseLabelRegionProperties:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError(
                "Numba label-region properties currently support 2-D labels."
            )
        arrays = _dense_label_region_properties_2d_numba(
            np.ascontiguousarray(label_array)
        )
        return DenseLabelRegionProperties(*arrays)


def label_region_properties_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> LabelRegionPropertiesBackendStrategy:
    """Return the selected dense-label region-properties backend."""
    return LabelRegionPropertiesBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


def binary_area_and_perimeter_2d(mask: np.ndarray) -> tuple[float, float]:
    """Return skimage-compatible foreground area and Crookes perimeter."""
    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.ndim != 2:
        raise NotImplementedError(
            "Numba binary area/perimeter currently supports 2-D masks."
        )
    area, perimeter = _binary_area_perimeter_2d_numba(
        np.ascontiguousarray(mask_array)
    )
    return float(area), float(perimeter)


def label_area_and_rounded_perimeter_2d(labels: np.ndarray) -> tuple[float, float]:
    """Return summed label area and CellProfiler-style rounded per-label perimeter."""
    labels_array = np.asarray(labels, dtype=np.int32)
    if labels_array.ndim != 2:
        raise NotImplementedError(
            "Numba label area/perimeter currently supports 2-D labels."
        )
    area, perimeter = _label_area_rounded_perimeter_2d_numba(
        np.ascontiguousarray(labels_array)
    )
    return float(area), float(perimeter)


@njit(cache=True)
def _dense_label_region_properties_2d_numba(labels: np.ndarray):
    height, width = labels.shape
    max_label = 0
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > max_label:
                max_label = label_id

    area_dense = np.zeros(max_label + 1, dtype=np.float64)
    sum_y = np.zeros(max_label + 1, dtype=np.float64)
    sum_x = np.zeros(max_label + 1, dtype=np.float64)
    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)

    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0:
                continue
            area_dense[label_id] += 1.0
            sum_y[label_id] += row
            sum_x[label_id] += col
            if row < min_y[label_id]:
                min_y[label_id] = row
            if col < min_x[label_id]:
                min_x[label_id] = col
            if row + 1 > max_y[label_id]:
                max_y[label_id] = row + 1
            if col + 1 > max_x[label_id]:
                max_x[label_id] = col + 1

    object_count = 0
    for label_id in range(1, max_label + 1):
        if area_dense[label_id] > 0.0:
            object_count += 1

    label = np.empty(object_count, dtype=np.int32)
    area = np.empty(object_count, dtype=np.float64)
    perimeter = np.empty(object_count, dtype=np.float64)
    bbox_min_y = np.empty(object_count, dtype=np.int64)
    bbox_min_x = np.empty(object_count, dtype=np.int64)
    bbox_max_y = np.empty(object_count, dtype=np.int64)
    bbox_max_x = np.empty(object_count, dtype=np.int64)
    bbox_area = np.empty(object_count, dtype=np.float64)
    centroid_y = np.empty(object_count, dtype=np.float64)
    centroid_x = np.empty(object_count, dtype=np.float64)
    equivalent_diameter = np.empty(object_count, dtype=np.float64)
    extent = np.empty(object_count, dtype=np.float64)
    major_axis_length = np.empty(object_count, dtype=np.float64)
    minor_axis_length = np.empty(object_count, dtype=np.float64)
    eccentricity = np.empty(object_count, dtype=np.float64)
    orientation = np.empty(object_count, dtype=np.float64)
    euler_number = np.empty(object_count, dtype=np.int64)
    moments = np.zeros((object_count, 4, 4), dtype=np.float64)
    moments_central = np.zeros((object_count, 4, 4), dtype=np.float64)
    moments_normalized = np.full((object_count, 4, 4), np.nan, dtype=np.float64)
    moments_hu = np.zeros((object_count, 7), dtype=np.float64)
    inertia_tensor = np.zeros((object_count, 2, 2), dtype=np.float64)
    inertia_tensor_eigvals = np.zeros((object_count, 2), dtype=np.float64)
    dense_to_index = np.full(max_label + 1, -1, dtype=np.int64)

    index = 0
    for label_id in range(1, max_label + 1):
        count = area_dense[label_id]
        if count <= 0.0:
            continue
        dense_to_index[label_id] = index
        label[index] = label_id
        area[index] = count
        bbox_min_y[index] = min_y[label_id]
        bbox_min_x[index] = min_x[label_id]
        bbox_max_y[index] = max_y[label_id]
        bbox_max_x[index] = max_x[label_id]
        box_area = (max_y[label_id] - min_y[label_id]) * (
            max_x[label_id] - min_x[label_id]
        )
        bbox_area[index] = float(box_area)
        centroid_y[index] = sum_y[label_id] / count
        centroid_x[index] = sum_x[label_id] / count
        equivalent_diameter[index] = np.sqrt(4.0 * count / np.pi)
        extent[index] = count / float(box_area) if box_area > 0 else 0.0
        index += 1

    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0:
                continue
            index = dense_to_index[label_id]
            local_y = float(row - bbox_min_y[index])
            local_x = float(col - bbox_min_x[index])
            centered_y = float(row) - centroid_y[index]
            centered_x = float(col) - centroid_x[index]
            powers_y = np.empty(4, dtype=np.float64)
            powers_x = np.empty(4, dtype=np.float64)
            powers_cy = np.empty(4, dtype=np.float64)
            powers_cx = np.empty(4, dtype=np.float64)
            powers_y[0] = 1.0
            powers_x[0] = 1.0
            powers_cy[0] = 1.0
            powers_cx[0] = 1.0
            for power in range(1, 4):
                powers_y[power] = powers_y[power - 1] * local_y
                powers_x[power] = powers_x[power - 1] * local_x
                powers_cy[power] = powers_cy[power - 1] * centered_y
                powers_cx[power] = powers_cx[power - 1] * centered_x
            for p in range(4):
                for q in range(4):
                    moments[index, p, q] += powers_y[p] * powers_x[q]
                    moments_central[index, p, q] += powers_cy[p] * powers_cx[q]

    for index in range(object_count):
        m00 = moments_central[index, 0, 0]
        if m00 <= 0.0:
            continue
        moments_central[index, 1, 1] = _label_mu11_skimage_order_2d(
            labels,
            label[index],
            bbox_min_y[index],
            bbox_min_x[index],
            bbox_max_y[index],
            bbox_max_x[index],
            centroid_y[index],
            centroid_x[index],
        )
        for p in range(4):
            for q in range(4):
                if p + q >= 2:
                    moments_normalized[index, p, q] = moments_central[
                        index, p, q
                    ] / (m00 ** (1.0 + (p + q) / 2.0))
        nu20 = moments_normalized[index, 2, 0]
        nu02 = moments_normalized[index, 0, 2]
        nu11 = moments_normalized[index, 1, 1]
        nu30 = moments_normalized[index, 3, 0]
        nu12 = moments_normalized[index, 1, 2]
        nu21 = moments_normalized[index, 2, 1]
        nu03 = moments_normalized[index, 0, 3]
        moments_hu[index, 0] = nu20 + nu02
        moments_hu[index, 1] = (nu20 - nu02) ** 2 + 4.0 * nu11**2
        moments_hu[index, 2] = (nu30 - 3.0 * nu12) ** 2 + (
            3.0 * nu21 - nu03
        ) ** 2
        moments_hu[index, 3] = (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2
        moments_hu[index, 4] = (nu30 - 3.0 * nu12) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3.0 * (nu21 + nu03) ** 2
        ) + (3.0 * nu21 - nu03) * (nu21 + nu03) * (
            3.0 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )
        moments_hu[index, 5] = (nu20 - nu02) * (
            (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        ) + 4.0 * nu11 * (nu30 + nu12) * (nu21 + nu03)
        moments_hu[index, 6] = (3.0 * nu21 - nu03) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3.0 * (nu21 + nu03) ** 2
        ) - (nu30 - 3.0 * nu12) * (nu21 + nu03) * (
            3.0 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )

        mu20 = moments_central[index, 2, 0] / m00
        mu02 = moments_central[index, 0, 2] / m00
        mu11 = moments_central[index, 1, 1] / m00
        inertia_tensor[index, 0, 0] = mu02
        inertia_tensor[index, 0, 1] = -mu11
        inertia_tensor[index, 1, 0] = -mu11
        inertia_tensor[index, 1, 1] = mu20
        trace_half = 0.5 * (mu20 + mu02)
        delta = np.sqrt(0.25 * (mu20 - mu02) ** 2 + mu11**2)
        eig0 = trace_half + delta
        eig1 = trace_half - delta
        if eig0 < eig1:
            tmp = eig0
            eig0 = eig1
            eig1 = tmp
        inertia_tensor_eigvals[index, 0] = eig0
        inertia_tensor_eigvals[index, 1] = eig1
        major_axis_length[index] = 4.0 * np.sqrt(max(eig0, 0.0))
        minor_axis_length[index] = 4.0 * np.sqrt(max(eig1, 0.0))
        eccentricity[index] = (
            np.sqrt(max(0.0, 1.0 - eig1 / eig0)) if eig0 > 0.0 else 0.0
        )
        orientation_a, orientation_b, orientation_c = (
            _label_orientation_tensor_components_2d(
                labels,
                label[index],
                bbox_min_y[index],
                bbox_min_x[index],
                bbox_max_y[index],
                bbox_max_x[index],
                m00,
            )
        )
        if orientation_a - orientation_c == 0.0:
            if orientation_b < 0.0:
                orientation[index] = -0.25 * np.pi
            else:
                orientation[index] = 0.25 * np.pi
        else:
            orientation[index] = 0.5 * np.arctan2(
                -2.0 * orientation_b,
                orientation_c - orientation_a,
            )
        perimeter[index] = _label_perimeter_2d(
            labels,
            label[index],
            bbox_min_y[index],
            bbox_min_x[index],
            bbox_max_y[index],
            bbox_max_x[index],
        )
        euler_number[index] = _label_euler_number_2d(
            labels,
            label[index],
            bbox_min_y[index],
            bbox_min_x[index],
            bbox_max_y[index],
            bbox_max_x[index],
        )

    return (
        label,
        area,
        perimeter,
        bbox_min_y,
        bbox_min_x,
        bbox_max_y,
        bbox_max_x,
        bbox_area,
        centroid_y,
        centroid_x,
        equivalent_diameter,
        extent,
        major_axis_length,
        minor_axis_length,
        eccentricity,
        orientation,
        euler_number,
        moments,
        moments_central,
        moments_normalized,
        moments_hu,
        inertia_tensor,
        inertia_tensor_eigvals,
    )


@njit(cache=True)
def _label_perimeter_2d(
    labels: np.ndarray,
    label_id: int,
    min_y: int,
    min_x: int,
    max_y: int,
    max_x: int,
) -> float:
    """Return skimage-compatible 4-neighborhood Crookes perimeter."""
    weights = np.zeros(50, dtype=np.float64)
    weights[5] = 1.0
    weights[7] = 1.0
    weights[15] = 1.0
    weights[17] = 1.0
    weights[25] = 1.0
    weights[27] = 1.0
    weights[21] = np.sqrt(2.0)
    weights[33] = np.sqrt(2.0)
    weights[13] = (1.0 + np.sqrt(2.0)) / 2.0
    weights[23] = (1.0 + np.sqrt(2.0)) / 2.0

    total = 0.0
    for row in range(min_y, max_y):
        for col in range(min_x, max_x):
            config = 0
            if _label_border_pixel_4(labels, label_id, row - 1, col - 1):
                config += 10
            if _label_border_pixel_4(labels, label_id, row - 1, col):
                config += 2
            if _label_border_pixel_4(labels, label_id, row - 1, col + 1):
                config += 10
            if _label_border_pixel_4(labels, label_id, row, col - 1):
                config += 2
            if _label_border_pixel_4(labels, label_id, row, col):
                config += 1
            if _label_border_pixel_4(labels, label_id, row, col + 1):
                config += 2
            if _label_border_pixel_4(labels, label_id, row + 1, col - 1):
                config += 10
            if _label_border_pixel_4(labels, label_id, row + 1, col):
                config += 2
            if _label_border_pixel_4(labels, label_id, row + 1, col + 1):
                config += 10
            total += weights[config]
    return total


@njit(cache=True)
def _label_border_pixel_4(
    labels: np.ndarray,
    label_id: int,
    row: int,
    col: int,
) -> bool:
    if not _label_pixel_at(labels, label_id, row, col):
        return False
    return not (
        _label_pixel_at(labels, label_id, row - 1, col)
        and _label_pixel_at(labels, label_id, row + 1, col)
        and _label_pixel_at(labels, label_id, row, col - 1)
        and _label_pixel_at(labels, label_id, row, col + 1)
    )


@njit(cache=True)
def _label_euler_number_2d(
    labels: np.ndarray,
    label_id: int,
    min_y: int,
    min_x: int,
    max_y: int,
    max_x: int,
) -> int:
    """Return skimage-compatible 2-D Euler number for one label crop."""
    coefs = np.array(
        (0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0),
        dtype=np.int64,
    )
    total = 0
    for row in range(min_y - 1, max_y + 1):
        for col in range(min_x - 1, max_x + 1):
            config = 0
            if _label_pixel_at(labels, label_id, row, col):
                config += 1
            if _label_pixel_at(labels, label_id, row, col + 1):
                config += 4
            if _label_pixel_at(labels, label_id, row + 1, col):
                config += 2
            if _label_pixel_at(labels, label_id, row + 1, col + 1):
                config += 8
            total += coefs[config]
    return int(total)


@njit(cache=True)
def _label_mu11_skimage_order_2d(
    labels: np.ndarray,
    label_id: int,
    min_y: int,
    min_x: int,
    max_y: int,
    max_x: int,
    centroid_y: float,
    centroid_x: float,
) -> float:
    total = 0.0
    for col in range(min_x, max_x):
        column_total = 0.0
        for row in range(min_y, max_y):
            if int(labels[row, col]) == int(label_id):
                column_total += float(row) - centroid_y
        total += column_total * (float(col) - centroid_x)
    return total


@njit(cache=True)
def _label_orientation_tensor_components_2d(
    labels: np.ndarray,
    label_id: int,
    min_y: int,
    min_x: int,
    max_y: int,
    max_x: int,
    m00: float,
) -> tuple[float, float, float]:
    """Return skimage-0.18 orientation tensor components (a, b, c)."""
    local_height = max_y - min_y
    local_width = max_x - min_x

    sum_y = 0.0
    sum_x = 0.0
    for row in range(min_y, max_y):
        for col in range(min_x, max_x):
            if int(labels[row, col]) == int(label_id):
                sum_y += float(row - min_y)
                sum_x += float(col - min_x)
    centroid_y = sum_y / m00
    centroid_x = sum_x / m00

    # skimage 0.18 computes central moments by applying one dot reduction per
    # axis to the binary crop. The reduction order is observable for symmetric
    # objects because orientation tie-breaking depends on signed zeros and ulps.
    reduced_y = np.zeros((3, local_width), dtype=np.float64)
    for col in range(local_width):
        for row in range(local_height):
            if int(labels[min_y + row, min_x + col]) != int(label_id):
                continue
            delta_y = float(row) - centroid_y
            reduced_y[0, col] += 1.0
            reduced_y[1, col] += delta_y
            reduced_y[2, col] += delta_y * delta_y

    powers_x = np.empty((local_width, 3), dtype=np.float64)
    for col in range(local_width):
        delta_x = float(col) - centroid_x
        powers_x[col, 0] = 1.0
        powers_x[col, 1] = delta_x
        powers_x[col, 2] = delta_x * delta_x
    central = np.dot(reduced_y, powers_x)
    mu20 = central[2, 0]
    mu02 = central[0, 2]
    mu11 = central[1, 1]

    return mu02 / m00, -mu11 / m00, mu20 / m00


@njit(cache=True)
def _label_pixel_at(
    labels: np.ndarray,
    label_id: int,
    row: int,
    col: int,
) -> bool:
    return (
        row >= 0
        and col >= 0
        and row < labels.shape[0]
        and col < labels.shape[1]
        and int(labels[row, col]) == int(label_id)
    )


@njit(cache=True)
def _label_area_rounded_perimeter_2d_numba(
    labels: np.ndarray,
) -> tuple[float, float]:
    height, width = labels.shape
    max_label = 0
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > max_label:
                max_label = label_id
    if max_label <= 0:
        return 0.0, 0.0

    area_dense = np.zeros(max_label + 1, dtype=np.float64)
    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.zeros(max_label + 1, dtype=np.int64)
    max_x = np.zeros(max_label + 1, dtype=np.int64)
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0:
                continue
            area_dense[label_id] += 1.0
            if row < min_y[label_id]:
                min_y[label_id] = row
            if col < min_x[label_id]:
                min_x[label_id] = col
            if row + 1 > max_y[label_id]:
                max_y[label_id] = row + 1
            if col + 1 > max_x[label_id]:
                max_x[label_id] = col + 1

    area = 0.0
    perimeter = 0.0
    for label_id in range(1, max_label + 1):
        if area_dense[label_id] <= 0.0:
            continue
        area += area_dense[label_id]
        perimeter += np.round(
            _label_perimeter_2d(
                labels,
                label_id,
                int(min_y[label_id]),
                int(min_x[label_id]),
                int(max_y[label_id]),
                int(max_x[label_id]),
            )
        )
    return area, perimeter


@njit(cache=True)
def _binary_area_perimeter_2d_numba(mask: np.ndarray) -> tuple[float, float]:
    height, width = mask.shape
    area = 0.0
    min_y = height
    min_x = width
    max_y = 0
    max_x = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            area += 1.0
            if row < min_y:
                min_y = row
            if col < min_x:
                min_x = col
            if row + 1 > max_y:
                max_y = row + 1
            if col + 1 > max_x:
                max_x = col + 1
    if area == 0.0:
        return 0.0, 0.0

    configs = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    config_height, config_width = configs.shape
    for row in range(min_y, max_y):
        local_row = row - min_y
        for col in range(min_x, max_x):
            if not _binary_border_pixel_4(mask, row, col):
                continue
            local_col = col - min_x
            if local_row + 1 < config_height:
                if local_col + 1 < config_width:
                    configs[local_row + 1, local_col + 1] += 10
                configs[local_row + 1, local_col] += 2
                if local_col > 0:
                    configs[local_row + 1, local_col - 1] += 10
            if local_col + 1 < config_width:
                configs[local_row, local_col + 1] += 2
            configs[local_row, local_col] += 1
            if local_col > 0:
                configs[local_row, local_col - 1] += 2
            if local_row > 0:
                if local_col + 1 < config_width:
                    configs[local_row - 1, local_col + 1] += 10
                configs[local_row - 1, local_col] += 2
                if local_col > 0:
                    configs[local_row - 1, local_col - 1] += 10

    perimeter = 0.0
    for row in range(config_height):
        for col in range(config_width):
            perimeter += _perimeter_weight_for_config_numba(int(configs[row, col]))
    return area, perimeter


@njit(cache=True)
def _perimeter_weight_for_config_numba(config: int) -> float:
    if (
        config == 5
        or config == 7
        or config == 15
        or config == 17
        or config == 25
        or config == 27
    ):
        return 1.0
    if config == 21 or config == 33:
        return np.sqrt(2.0)
    if config == 13 or config == 23:
        return (1.0 + np.sqrt(2.0)) / 2.0
    return 0.0


@njit(cache=True)
def _binary_border_pixel_4(mask: np.ndarray, row: int, col: int) -> bool:
    if not _binary_pixel_at(mask, row, col):
        return False
    return not (
        _binary_pixel_at(mask, row - 1, col)
        and _binary_pixel_at(mask, row + 1, col)
        and _binary_pixel_at(mask, row, col - 1)
        and _binary_pixel_at(mask, row, col + 1)
    )


@njit(cache=True)
def _binary_pixel_at(mask: np.ndarray, row: int, col: int) -> bool:
    return (
        row >= 0
        and col >= 0
        and row < mask.shape[0]
        and col < mask.shape[1]
        and bool(mask[row, col])
    )


__all__ = [
    "binary_area_and_perimeter_2d",
    "DenseLabelRegionProperties",
    "label_area_and_rounded_perimeter_2d",
    "LabelRegionPropertiesBackendStrategy",
    "NumbaNumpyLabelRegionPropertiesBackendStrategy",
    "label_region_properties_backend",
]
