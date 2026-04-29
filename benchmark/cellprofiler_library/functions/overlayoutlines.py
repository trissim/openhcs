"""Converted from CellProfiler: OverlayOutlines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import skimage.color
import skimage.segmentation
from skimage import img_as_float

from openhcs.core.memory.decorators import numpy
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

from benchmark.cellprofiler_library.color import coerce_rgb_color

EnumT = TypeVar("EnumT", bound=Enum)


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
            source_kind=_coerce_source_kind(source_kind),
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
        display_mode=_coerce_enum(OutlineDisplayMode, display_mode),
        line_mode=_coerce_enum(LineMode, line_mode),
        max_type=_coerce_enum(MaxType, max_type),
    )
    image_sources = _image_sources_from_payload(
        image,
        blank_image=context.blank_image,
        image_row_count=context.image_row_count,
    )

    if _requires_plane_stack_execution(image_sources, context.object_labels):
        return _overlay_plane_stack(
            context=context,
            image_sources=image_sources,
        )
    return _overlay_single_plane(
        context=context,
        image_sources=image_sources,
    )


def _overlay_plane_stack(
    *,
    context: OverlayOutlineExecutionContext,
    image_sources: tuple[np.ndarray, ...],
) -> np.ndarray:
    slice_count = _aligned_plane_slice_count((*image_sources, *context.object_labels))
    return np.stack(
        tuple(
            _overlay_single_plane(
                context=context.plane(slice_index),
                image_sources=tuple(
                    _plane_payload_slice(source, slice_index)
                    for source in image_sources
                ),
            )
            for slice_index in range(slice_count)
        )
    ).astype(np.float32)


def _overlay_single_plane(
    *,
    context: OverlayOutlineExecutionContext,
    image_sources: tuple[np.ndarray, ...],
) -> np.ndarray:
    output = _base_image(
        image_sources=image_sources,
        object_labels=context.object_labels,
        blank_image=context.blank_image,
        display_mode=context.display_mode,
    )
    outline_intensity = _outline_intensity(output, context.blank_image, context.max_type)
    image_index = context.first_outline_image_index
    object_index = 0
    for row in context.rows:
        if row.source_kind is OutlineSourceKind.IMAGE:
            output = _draw_outline_image(
                output,
                image_sources[image_index],
                row.color,
                outline_intensity=outline_intensity,
                display_mode=context.display_mode,
            )
            image_index += 1
            continue
        output = _draw_object_labels(
            output,
            _collapse_singleton_label_stack(context.object_labels[object_index]),
            row.color,
            outline_intensity=outline_intensity,
            display_mode=context.display_mode,
            line_mode=context.line_mode,
        )
        object_index += 1
    if context.display_mode is OutlineDisplayMode.GRAYSCALE and output.ndim == 3:
        return skimage.color.rgb2gray(output).astype(np.float32)
    return output.astype(np.float32)


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
        return tuple(_collapse_singleton_label_stack(object_labels[0]).shape)
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
    labels_2d = _resize_labels(labels.astype(np.int32), output.shape[:2])
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
    marked = np.array(output, copy=True)
    marked[boundaries] = outline_color
    return marked


def _draw_outline_image(
    output: np.ndarray,
    outline_image: np.ndarray,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
    mask = _outline_image_mask(outline_image)
    mask = _resize_mask(mask, output.shape[:2])
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


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == shape:
        return labels
    return _resize_nearest(labels, shape).astype(np.int32)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return _resize_nearest(mask.astype(np.uint8), shape).astype(bool)


def _resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize

    return resize(
        image,
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


def _collapse_singleton_label_stack(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 3 and labels.shape[0] == 1:
        return labels[0]
    return labels


def _coerce_source_kind(value: OutlineSourceKind | str) -> OutlineSourceKind:
    if isinstance(value, OutlineSourceKind):
        return value
    normalized = str(value).strip().lower()
    return OutlineSourceKind(normalized)


def _coerce_enum(enum_type: type[EnumT], value: EnumT | str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    normalized = str(value).strip().lower().replace(" ", "_")
    for member in enum_type:
        if normalized in _enum_member_literals(member):
            return member
    raise ValueError(f"{enum_type.__name__} does not support {value!r}.")


def _enum_member_literals(member: Enum) -> frozenset[str]:
    literals = [member.name]
    if isinstance(member.value, tuple):
        literals.extend(str(value) for value in member.value)
    else:
        literals.append(str(member.value))
    return frozenset(
        str(literal).strip().lower().replace(" ", "_")
        for literal in literals
    )


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
