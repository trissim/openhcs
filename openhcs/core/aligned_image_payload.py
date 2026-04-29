"""Generic aligned image-payload composition for multi-source runtime inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_grayscale_image_slice,
    is_grayscale_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import MEMORY_TYPE_NUMPY, convert_memory, detect_memory_type


class ImagePayloadExecutionMode(Enum):
    """How a runtime executor should interpret a resolved image payload."""

    NATURAL = "natural"
    FULL_STACK = "full_stack"
    ALIGNED_MULTI_IMAGE_STACK = "aligned_multi_image_stack"


@dataclass(frozen=True, slots=True)
class ImagePayloadComposition:
    """Resolved image payload plus its execution mode."""

    payload: Any
    execution_mode: ImagePayloadExecutionMode


@dataclass(frozen=True, slots=True)
class AlignedImageStack:
    """Per-slice multi-image bundles aligned to one OpenHCS stack."""

    slices: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        if not self.slices:
            raise ValueError("AlignedImageStack.slices cannot be empty.")


class ImageBundleLayout(ABC, metaclass=AutoRegisterMeta):
    """Nominal layout for heterogeneous same-slice runtime image bundles."""

    __registry_key__ = "layout_key"
    __skip_if_no_key__ = True
    layout_key: ClassVar[str | None] = None

    @classmethod
    def for_slices(cls, slices: Sequence[Any]) -> "ImageBundleLayout":
        for layout_type in cls.__registry__.values():
            if layout_type.matches(slices):
                return layout_type()
        raise ValueError(
            "OpenHCS image bundles require 2D grayscale or HWC color slices; "
            f"got shapes {[getattr(slice_data, 'shape', None) for slice_data in slices]!r}."
        )

    @classmethod
    @abstractmethod
    def matches(cls, slices: Sequence[Any]) -> bool:
        """Return whether this layout can compose the supplied slices."""

    @abstractmethod
    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        """Stack same-slice runtime images into one callable input bundle."""


class MixedColorImageBundleLayout(ImageBundleLayout):
    """Promote grayscale slices when a bundle mixes grayscale and color images."""

    layout_key = "mixed_color"

    @classmethod
    def matches(cls, slices: Sequence[Any]) -> bool:
        return (
            all(_is_bundle_image_slice(slice_data) for slice_data in slices)
            and any(is_color_image_slice(slice_data) for slice_data in slices)
            and any(is_grayscale_image_slice(slice_data) for slice_data in slices)
        )

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        numpy_slices = tuple(
            _as_numpy_slice(slice_data, gpu_id)
            for slice_data in slices
        )
        spatial_shapes = {tuple(slice_data.shape[:2]) for slice_data in numpy_slices}
        if len(spatial_shapes) != 1:
            raise ValueError(
                "OpenHCS mixed color image bundles require stable spatial shape; "
                f"got {[slice_data.shape for slice_data in numpy_slices]!r}."
            )
        channel_counts = {
            int(slice_data.shape[-1])
            for slice_data in numpy_slices
            if is_color_image_slice(slice_data)
        }
        if len(channel_counts) != 1:
            raise ValueError(
                "OpenHCS mixed color image bundles require stable color channel "
                f"count; got {sorted(channel_counts)!r}."
            )
        channel_count = next(iter(channel_counts))
        stacked = np.stack(
            tuple(
                _promote_slice_to_color(slice_data, channel_count)
                for slice_data in numpy_slices
            )
        )
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return _convert_payload(stacked, MEMORY_TYPE_NUMPY, memory_type, gpu_id)


def compose_aligned_image_payload(
    owner_name: str,
    image_payloads: tuple[Any, ...],
) -> ImagePayloadComposition:
    """Compose one or more image payloads into an executor-ready payload."""
    if not image_payloads:
        raise ValueError(f"{owner_name} cannot compose an empty image input set.")
    if len(image_payloads) == 1:
        return ImagePayloadComposition(
            payload=image_payloads[0],
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )

    payload_slices = tuple(
        payload_slices_for_alignment(payload)
        for payload in image_payloads
    )
    slice_counts = tuple(len(slices) for slices in payload_slices)
    max_slice_count = max(slice_counts)
    invalid_counts = tuple(
        count
        for count in slice_counts
        if count not in {1, max_slice_count}
    )
    if invalid_counts:
        raise ValueError(
            f"{owner_name} cannot align multi-image inputs with incompatible "
            f"slice counts {slice_counts!r}."
        )

    if max_slice_count == 1:
        return ImagePayloadComposition(
            payload=compose_one_image_bundle(
                tuple(slices[0] for slices in payload_slices)
            ),
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        )
    return ImagePayloadComposition(
        payload=AlignedImageStack(
            slices=tuple(
                compose_one_image_bundle(
                    tuple(
                        aligned_payload_slice(slices, slice_index)
                        for slices in payload_slices
                    )
                )
                for slice_index in range(max_slice_count)
            )
        ),
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )


def payload_slices_for_alignment(payload: Any) -> tuple[Any, ...]:
    """Return payload slices used for multi-source alignment."""
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return (payload,)
    if is_color_image_slice(payload):
        return (payload,)
    if is_grayscale_image_stack(payload) or is_color_image_stack(payload):
        memory_type = detect_memory_type(payload)
        return tuple(
            ImageStackLayout.for_stack(payload).unstack(
                array=payload,
                memory_type=memory_type,
                gpu_id=0,
            )
        )
    return (payload,)


def aligned_payload_slice(
    slices: tuple[Any, ...],
    slice_index: int,
) -> Any:
    """Return the payload slice for one aligned execution index."""
    if len(slices) == 1:
        return slices[0]
    return slices[slice_index]


def aligned_image_stack_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    """Slice runtime-array kwargs alongside an aligned image stack."""
    return {
        name: aligned_image_stack_kwarg(value, slice_index, slice_count)
        for name, value in kwargs.items()
    }


def aligned_image_stack_kwarg(
    value: Any,
    slice_index: int,
    slice_count: int,
) -> Any:
    """Slice one runtime-array kwarg when it shares the aligned stack length."""
    if not hasattr(value, "ndim"):
        return value
    slices = payload_slices_for_alignment(value)
    if len(slices) == slice_count:
        return slices[slice_index]
    if len(slices) == 1:
        return slices[0]
    return value


def compose_one_image_bundle(
    image_payloads: tuple[Any, ...],
) -> Any:
    """Stack same-slice image payloads into one multi-image bundle."""
    memory_type = detect_memory_type(image_payloads[0])
    if _is_homogeneous_image_bundle(image_payloads):
        return ImageStackLayout.for_slices(image_payloads).stack(
            slices=image_payloads,
            memory_type=memory_type,
            gpu_id=0,
        )
    return ImageBundleLayout.for_slices(image_payloads).stack(
        slices=image_payloads,
        memory_type=memory_type,
        gpu_id=0,
    )


def payload_slice_count(payload: Any) -> int:
    """Return the number of aligned slices represented by one payload."""
    return len(payload_slices_for_alignment(payload))


def _is_bundle_image_slice(value: Any) -> bool:
    return is_grayscale_image_slice(value) or is_color_image_slice(value)


def _is_homogeneous_image_bundle(slices: Sequence[Any]) -> bool:
    return (
        all(is_grayscale_image_slice(slice_data) for slice_data in slices)
        or all(is_color_image_slice(slice_data) for slice_data in slices)
    )


def _as_numpy_slice(slice_data: Any, gpu_id: int) -> np.ndarray:
    source_type = detect_memory_type(slice_data)
    if source_type == MEMORY_TYPE_NUMPY:
        return slice_data
    return _convert_payload(slice_data, source_type, MEMORY_TYPE_NUMPY, gpu_id)


def _promote_slice_to_color(slice_data: np.ndarray, channel_count: int) -> np.ndarray:
    if is_color_image_slice(slice_data):
        return slice_data
    return np.repeat(slice_data[:, :, np.newaxis], channel_count, axis=2)


def _convert_payload(
    data: Any,
    source_type: str,
    target_type: str,
    gpu_id: int,
) -> Any:
    return convert_memory(
        data=data,
        source_type=source_type,
        target_type=target_type,
        gpu_id=gpu_id,
    )
