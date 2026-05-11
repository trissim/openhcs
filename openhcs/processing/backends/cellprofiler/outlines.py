"""Object outline backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import skimage.color
import skimage.segmentation
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange
from skimage import img_as_float

from openhcs.constants.constants import MemoryType
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.color import coerce_rgb_color
from openhcs.processing.backends.cellprofiler.image_geometry import (
    align_binary_mask_to_shape,
    align_label_plane_to_shape,
    collapse_singleton_plane_stack,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


class ObjectOutlineBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object outline operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def outline(self, labels: np.ndarray) -> np.ndarray:
        """Return a labeled inner outline image."""


class NumbaNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Numba-accelerated NumPy object outline primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def outline(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError("Object outlines currently support 2-D labels.")
        return _outline_numba(np.ascontiguousarray(label_array))


class CentrosomeNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Explicit centrosome provider for NumPy object outlines."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def outline(self, labels: np.ndarray) -> np.ndarray:
        from centrosome.outline import outline

        return outline(labels)


def object_outline_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectOutlineBackendStrategy:
    """Return the selected CellProfiler object outline backend."""
    return ObjectOutlineBackendStrategy.for_memory_type(
        backend_provider=backend_provider,
    )


@njit(cache=True, parallel=True)
def _outline_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    output = np.zeros((height, width), dtype=labels.dtype)
    for y in prange(height):
        for x in range(width):
            center = labels[y, x]
            if center <= 0:
                continue
            min_label = center
            max_label = center
            for dy in range(-1, 2):
                ny = y + dy
                for dx in range(-1, 2):
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        value = 0
                    else:
                        value = labels[ny, nx]
                    if value < min_label:
                        min_label = value
                    if value > max_label:
                        max_label = value
            if max_label != min_label:
                output[y, x] = center
    return output


class LineMode(Enum):
    """Closed CellProfiler outline boundary modes."""

    INNER = ("inner", "Inner")
    OUTER = ("outer", "Outer")
    THICK = ("thick", "Thick")

    @property
    def skimage_mode(self) -> str:
        return self.value[0]


class OutlineDisplayMode(Enum):
    """Closed CellProfiler outline display modes."""

    COLOR = ("color", "Color")
    GRAYSCALE = ("grayscale", "Grayscale")


class MaxType(Enum):
    """Closed CellProfiler grayscale outline intensity modes."""

    MAX_IMAGE = ("max_image", "Max of image")
    MAX_POSSIBLE = ("max_possible", "Max possible")


class OutlineSourceKind(str, Enum):
    """Runtime source kind for one OverlayOutlines row."""

    IMAGE = "image"
    OBJECTS = "objects"


@dataclass(frozen=True, slots=True)
class OverlayOutlineRuntimeRow:
    """One runtime OverlayOutlines row after compiler lowering."""

    source_kind: OutlineSourceKind
    color: tuple[float, float, float]

    @classmethod
    def from_literals(
        cls,
        source_kind: OutlineSourceKind | str,
        color: str | Sequence[float],
    ) -> "OverlayOutlineRuntimeRow":
        return cls(
            source_kind=coerce_cellprofiler_enum(OutlineSourceKind, source_kind),
            color=coerce_rgb_color(color),
        )


@dataclass(frozen=True, slots=True)
class OverlayOutlineExecutionContext:
    """Runtime OverlayOutlines plan shared by plane and single-slice execution."""

    rows: tuple[OverlayOutlineRuntimeRow, ...]
    object_labels: tuple[np.ndarray, ...]
    blank_image: bool
    display_mode: OutlineDisplayMode
    line_mode: LineMode
    max_type: MaxType

    def __post_init__(self) -> None:
        if len(self.object_labels) != self.object_row_count:
            raise ValueError("OverlayOutlines object_labels count must match object rows.")

    @property
    def image_row_count(self) -> int:
        return sum(row.source_kind is OutlineSourceKind.IMAGE for row in self.rows)

    @property
    def object_row_count(self) -> int:
        return sum(row.source_kind is OutlineSourceKind.OBJECTS for row in self.rows)

    @property
    def first_outline_image_index(self) -> int:
        return 0 if self.blank_image else 1

    def plane(self, slice_index: int) -> "OverlayOutlineExecutionContext":
        return type(self)(
            rows=self.rows,
            object_labels=tuple(
                _plane_payload_slice(labels, slice_index)
                for labels in self.object_labels
            ),
            blank_image=self.blank_image,
            display_mode=self.display_mode,
            line_mode=self.line_mode,
            max_type=self.max_type,
        )

    def render(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        if _requires_plane_stack_execution(image_sources, self.object_labels):
            return self.render_plane_stack(image_sources)
        return self.render_single_plane(image_sources)

    def render_plane_stack(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        slice_count = _aligned_plane_slice_count((*image_sources, *self.object_labels))
        return np.stack(
            tuple(
                self.plane(slice_index).render_single_plane(
                    tuple(
                        _plane_payload_slice(source, slice_index)
                        for source in image_sources
                    )
                )
                for slice_index in range(slice_count)
            )
        ).astype(np.float32)

    def render_single_plane(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        output = _base_image(
            image_sources=image_sources,
            object_labels=self.object_labels,
            blank_image=self.blank_image,
            display_mode=self.display_mode,
        )
        outline_intensity = _outline_intensity(output, self.blank_image, self.max_type)
        image_index = self.first_outline_image_index
        object_index = 0
        for row in self.rows:
            if row.source_kind is OutlineSourceKind.IMAGE:
                output = _draw_outline_image(
                    output,
                    image_sources[image_index],
                    row.color,
                    outline_intensity=outline_intensity,
                    display_mode=self.display_mode,
                )
                image_index += 1
                continue
            output = _draw_object_labels(
                output,
                collapse_singleton_plane_stack(self.object_labels[object_index]),
                row.color,
                outline_intensity=outline_intensity,
                display_mode=self.display_mode,
                line_mode=self.line_mode,
            )
            object_index += 1
        if self.display_mode is OutlineDisplayMode.GRAYSCALE and output.ndim == 3:
            return skimage.color.rgb2gray(output).astype(np.float32)
        return output.astype(np.float32)


@numpy(contract=ProcessingContract.FLEXIBLE)
def overlay_outlines(
    image: np.ndarray,
    *,
    blank_image: bool = False,
    display_mode: OutlineDisplayMode | str = OutlineDisplayMode.COLOR,
    line_mode: LineMode | str = LineMode.INNER,
    max_type: MaxType | str = MaxType.MAX_IMAGE,
    outline_source_kinds: Sequence[OutlineSourceKind | str] = (
        OutlineSourceKind.OBJECTS,
    ),
    outline_colors: Sequence[str | Sequence[float]] = ("Red",),
    object_labels: Sequence[np.ndarray] = (),
    dtype_config: Any | None = None,
) -> np.ndarray:
    """Overlay object-derived or image-derived outlines onto one output image."""
    del dtype_config
    context = OverlayOutlineExecutionContext(
        rows=_runtime_rows(outline_source_kinds, outline_colors),
        object_labels=tuple(object_labels),
        blank_image=blank_image,
        display_mode=coerce_cellprofiler_enum(OutlineDisplayMode, display_mode),
        line_mode=coerce_cellprofiler_enum(LineMode, line_mode),
        max_type=coerce_cellprofiler_enum(MaxType, max_type),
    )
    image_sources = _image_sources_from_payload(
        image,
        blank_image=context.blank_image,
        image_row_count=context.image_row_count,
    )

    return context.render(image_sources)


def _runtime_rows(
    source_kinds: Sequence[OutlineSourceKind | str],
    colors: Sequence[str | Sequence[float]],
) -> tuple[OverlayOutlineRuntimeRow, ...]:
    if not source_kinds:
        raise ValueError("OverlayOutlines requires at least one outline row.")
    return tuple(
        OverlayOutlineRuntimeRow.from_literals(
            source_kind,
            _indexed_value(colors, index, default="Red"),
        )
        for index, source_kind in enumerate(source_kinds)
    )


def _image_sources_from_payload(
    image: np.ndarray,
    *,
    blank_image: bool,
    image_row_count: int,
) -> tuple[np.ndarray, ...]:
    expected_count = image_row_count if blank_image else image_row_count + 1
    if expected_count == 0:
        return ()
    if expected_count == 1:
        return (image,)
    if image.ndim < 3 or image.shape[0] != expected_count:
        raise ValueError(
            "OverlayOutlines expected a stack whose first axis contains the "
            f"base image plus outline images; expected {expected_count} planes, "
            f"got shape {getattr(image, 'shape', None)}."
        )
    return tuple(image[index] for index in range(expected_count))


def _requires_plane_stack_execution(
    image_sources: tuple[np.ndarray, ...],
    object_labels: Sequence[np.ndarray],
) -> bool:
    return any(_is_plane_stack_payload(payload) for payload in (*image_sources, *object_labels))


def _aligned_plane_slice_count(payloads: Sequence[np.ndarray]) -> int:
    slice_counts = frozenset(
        _plane_slice_count(payload)
        for payload in payloads
        if _is_plane_stack_payload(payload)
    )
    if not slice_counts:
        return 1
    if len(slice_counts) != 1:
        raise ValueError(
            "OverlayOutlines plane-stack inputs must have aligned slice counts; "
            f"got {sorted(slice_counts)!r}."
        )
    return next(iter(slice_counts))


def _plane_payload_slice(payload: np.ndarray, slice_index: int) -> np.ndarray:
    if _is_plane_stack_payload(payload):
        return payload[slice_index]
    return payload


def _plane_slice_count(payload: np.ndarray) -> int:
    return int(payload.shape[0])


def _is_plane_stack_payload(payload: np.ndarray) -> bool:
    return payload.ndim == 3 and not is_color_image_slice(payload)


def _base_image(
    *,
    image_sources: tuple[np.ndarray, ...],
    object_labels: Sequence[np.ndarray],
    blank_image: bool,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
    if blank_image:
        shape = _blank_shape(image_sources, object_labels)
        if display_mode is OutlineDisplayMode.COLOR:
            return np.zeros((*shape, 3), dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)

    if not image_sources:
        raise ValueError("OverlayOutlines requires a base image outside blank mode.")
    base = img_as_float(image_sources[0])
    if display_mode is OutlineDisplayMode.COLOR:
        if base.ndim == 2:
            return skimage.color.gray2rgb(base).astype(np.float32)
        return base.astype(np.float32)
    if base.ndim == 3:
        return skimage.color.rgb2gray(base).astype(np.float32)
    return base.astype(np.float32)


def _blank_shape(
    image_sources: tuple[np.ndarray, ...],
    object_labels: Sequence[np.ndarray],
) -> tuple[int, ...]:
    if object_labels:
        return tuple(collapse_singleton_plane_stack(object_labels[0]).shape)
    if image_sources:
        return tuple(image_sources[0].shape[:2])
    raise ValueError("OverlayOutlines blank mode requires an outline source.")


def _outline_intensity(
    output: np.ndarray,
    blank_image: bool,
    max_type: MaxType,
) -> float:
    if blank_image or max_type is MaxType.MAX_POSSIBLE:
        return 1.0
    return float(np.max(output))


def _draw_object_labels(
    output: np.ndarray,
    labels: np.ndarray,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
    line_mode: LineMode,
) -> np.ndarray:
    labels_2d = align_label_plane_to_shape(
        object_label_dense_array(labels, dtype=np.int32),
        output.shape[:2],
    )
    outline_color: tuple[float, float, float] | float
    if display_mode is OutlineDisplayMode.COLOR:
        if output.ndim == 2:
            output = skimage.color.gray2rgb(output)
        outline_color = color
    else:
        outline_color = outline_intensity
    boundaries = skimage.segmentation.find_boundaries(
        labels_2d,
        mode=line_mode.skimage_mode,
    )
    if not np.any(boundaries):
        return output
    return skimage.segmentation.mark_boundaries(
        output,
        labels_2d,
        color=outline_color,
        mode=line_mode.skimage_mode,
    )


def _draw_outline_image(
    output: np.ndarray,
    outline_image: np.ndarray,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
    mask = _outline_image_mask(outline_image)
    mask = align_binary_mask_to_shape(mask, output.shape[:2])
    if display_mode is OutlineDisplayMode.COLOR:
        if output.ndim == 2:
            output = skimage.color.gray2rgb(output)
        output[mask] = color
        return output
    output[mask] = outline_intensity
    return output


def _outline_image_mask(outline_image: np.ndarray) -> np.ndarray:
    mask = np.asarray(outline_image) > 0
    if is_color_image_slice(mask):
        return np.any(mask, axis=-1)
    return mask


def _indexed_value(
    values: Sequence[Any],
    index: int,
    *,
    default: Any,
) -> Any:
    if not values:
        return default
    if index < len(values):
        return values[index]
    return values[-1]


__all__ = [
    "CentrosomeNumpyObjectOutlineBackendStrategy",
    "LineMode",
    "MaxType",
    "NumbaNumpyObjectOutlineBackendStrategy",
    "ObjectOutlineBackendStrategy",
    "OutlineDisplayMode",
    "OutlineSourceKind",
    "OverlayOutlineExecutionContext",
    "OverlayOutlineRuntimeRow",
    "object_outline_backend",
    "overlay_outlines",
]
