"""Generic runtime invocation records shared by dialect adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import inspect
from types import MappingProxyType
from typing import ClassVar, Generic, Protocol, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues

PayloadT = TypeVar("PayloadT")


class RuntimeParameterDeclaration(Protocol):
    """Nominal callable parameter declaration exposed by compiled bindings."""

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the public callable parameter name."""

    @classmethod
    def parameter(cls) -> inspect.Parameter:
        """Return the injected callable signature parameter."""


class SliceIndexRuntimeParameter:
    """Runtime-supplied pure-2D plane index parameter."""

    @classmethod
    def require_parameter_name(cls) -> str:
        return "slice_index"

    @classmethod
    def parameter(cls) -> inspect.Parameter:
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=int | None,
        )


@dataclass(frozen=True, slots=True)
class RuntimeParameterBinding:
    """Compile-resolved runtime parameter value for one callable invocation."""

    parameter_type: type[RuntimeParameterDeclaration]
    value: object

    @property
    def parameter_name(self) -> str:
        return self.parameter_type.require_parameter_name()


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInvocationOptions(ABC):
    """Typed, non-callable settings that control one runtime invocation."""

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the callable ABI name for invocation-options injection."""
        return "runtime_invocation_options"


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

    func: Callable[..., object]
    image: object
    kwargs: Mapping[str, object]
    batch_index: int
    batch_count: int
    semantic_group_key: tuple[Hashable, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kwargs",
            MappingProxyType(
                {
                    **runtime_callable_defaults(self.func),
                    **dict(self.kwargs),
                }
            ),
        )


@lru_cache(maxsize=1024)
def runtime_callable_defaults(func: Callable[..., object]) -> Mapping[str, object]:
    """Return callable defaults visible to runtime batch executors."""
    try:
        callable_signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Runtime batch function {func!r} must expose an inspectable signature."
        ) from exc
    defaults: dict[str, object] = {}
    for parameter in callable_signature.parameters.values():
        if parameter.default is inspect.Parameter.empty:
            continue
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        defaults[parameter.name] = parameter.default
    return MappingProxyType(defaults)


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
