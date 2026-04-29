"""Generic aligned image-payload composition for multi-source runtime inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_grayscale_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type


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
    return ImageStackLayout.for_slices(image_payloads).stack(
        slices=image_payloads,
        memory_type=memory_type,
        gpu_id=0,
    )


def payload_slice_count(payload: Any) -> int:
    """Return the number of aligned slices represented by one payload."""
    return len(payload_slices_for_alignment(payload))
