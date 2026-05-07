"""Generic runtime invocation records shared by dialect adapters."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInvocationOptions(ABC):
    """Typed, non-callable settings that control one runtime invocation."""


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
