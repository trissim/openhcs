"""Generic runtime invocation records shared by dialect adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInvocationOptions(ABC):
    """Typed, non-callable settings that control one runtime invocation."""


class RuntimeOutputBundle(ABC):
    """Nominal multi-output bundle lowered by runtime execution contracts."""

    @abstractmethod
    def as_runtime_tuple(self) -> tuple[object, ...]:
        """Return the positional ABI consumed by runtime output aggregation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageExecutionContext(ABC, metaclass=AutoRegisterMeta):
    """Source provenance and execution mode for an image-like invocation."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None

    source_image_name: str | None
    execution_mode: ImagePayloadExecutionMode = ImagePayloadExecutionMode.NATURAL


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRuntimeInputRequest(RuntimeImageExecutionContext):
    """Shared provenance for resolved image-like runtime inputs."""

    registry_key: ClassVar[str] = "resolved_input"

    image_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageRequest(ResolvedRuntimeInputRequest, Generic[PayloadT]):
    """Resolved image payload and source metadata for one runtime invocation."""

    registry_key: ClassVar[str] = "image_request"

    payload: PayloadT


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeFunctionInvocationRequest(
    ResolvedRuntimeInputRequest,
    Generic[PayloadT],
):
    """Resolved callable inputs for one runtime function invocation."""

    registry_key: ClassVar[str] = "function_invocation"

    image: PayloadT
    kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeBatchInvocationRequest(RuntimeImageExecutionContext):
    """One invocation inside a nominal runtime batch."""

    registry_key: ClassVar[str] = "batch_invocation"

    image: object
    kwargs: Mapping[str, object]
    batch_index: int
    batch_count: int
    semantic_group_key: tuple[Hashable, ...] | None = None


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
