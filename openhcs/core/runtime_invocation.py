"""Generic runtime invocation records shared by dialect adapters."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode

PayloadT = TypeVar("PayloadT")
SliceValueT = TypeVar("SliceValueT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageExecutionContext(ABC):
    """Source provenance and execution mode for an image-like invocation."""

    source_image_name: str | None
    execution_mode: ImagePayloadExecutionMode = ImagePayloadExecutionMode.NATURAL


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRuntimeInputRequest(RuntimeImageExecutionContext):
    """Shared provenance for resolved image-like runtime inputs."""

    image_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageRequest(ResolvedRuntimeInputRequest, Generic[PayloadT]):
    """Resolved image payload and source metadata for one runtime invocation."""

    payload: PayloadT


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeFunctionInvocationRequest(
    ResolvedRuntimeInputRequest,
    Generic[PayloadT],
):
    """Resolved callable inputs for one runtime function invocation."""

    image: PayloadT
    kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeSliceAlignedValues(Generic[SliceValueT]):
    """Non-image payload with one backend-native value per runtime slice."""

    slices: tuple[SliceValueT, ...]

    def __post_init__(self) -> None:
        slices = tuple(self.slices)
        if not slices:
            raise ValueError("RuntimeSliceAlignedValues.slices cannot be empty.")
        object.__setattr__(self, "slices", slices)

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    def value_for_slice(self, slice_index: int) -> SliceValueT:
        return self.slices[slice_index]


def requested_image_execution_mode(
    *,
    force_full_stack: bool,
    execution_mode: ImagePayloadExecutionMode | None,
) -> ImagePayloadExecutionMode:
    """Resolve legacy full-stack forcing into a typed image execution mode."""
    if execution_mode is not None:
        return execution_mode
    if force_full_stack:
        return ImagePayloadExecutionMode.FULL_STACK
    return ImagePayloadExecutionMode.NATURAL
