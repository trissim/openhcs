"""Runtime batch execution contracts shared by compiler and pipeline decorators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

from metaclass_registry import AutoRegisterMeta


F = TypeVar("F", bound=Callable)


class RuntimeBatchExecutionDomain(str, Enum):
    """Nominal domains where one callable can batch equivalent invocations."""

    PURE_2D_SLICES = "pure_2d_slices"
    MEASUREMENT_IMAGES = "measurement_images"


class RuntimeBatchCallableMetadataField(str, Enum):
    """Callable metadata fields owned by runtime batch contracts."""

    EXECUTORS = "__openhcs_runtime_batch_executors__"
    LEGACY_PURE_2D_EXECUTOR = "__openhcs_pure_2d_batch_executor__"

    def read(self, func: Callable, default: Any = None) -> Any:
        """Read this metadata field from a callable namespace."""
        return _callable_namespace(func).get(self.value, default)

    def write(self, func: F, value: Any) -> F:
        """Write this metadata field to a callable and return the callable."""
        setattr(func, self.value, value)
        return func

    def owner_label(self, func: Callable) -> str:
        """Return a compact field label for validation errors."""
        return f"{func}.{self.value}"


RUNTIME_BATCH_EXECUTORS_ATTR = RuntimeBatchCallableMetadataField.EXECUTORS.value
PURE_2D_BATCH_EXECUTOR_ATTR = (
    RuntimeBatchCallableMetadataField.LEGACY_PURE_2D_EXECUTOR.value
)


def _callable_namespace(func: Callable) -> Mapping[str, Any]:
    """Return a callable metadata namespace without probing dynamic attributes."""
    try:
        return vars(func)
    except TypeError:
        return {}


class RuntimeBatchExecutor(ABC, metaclass=AutoRegisterMeta):
    """Nominal callable object for reusable runtime batch execution policies."""

    __registry_key__ = "executor_name"
    __skip_if_no_key__ = True
    executor_name: ClassVar[str | None] = None

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute one runtime batch domain."""


@dataclass(frozen=True, slots=True)
class RuntimePure2DSliceBatchRequest:
    """Nominal request for equivalent pure-2D slice invocations."""

    func: Callable[..., Any]
    slices_2d: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    execute_slice: Callable[
        [Callable[..., Any], Any, Mapping[str, Any], int, int],
        Any,
    ]

    @property
    def slice_count(self) -> int:
        """Number of slice invocations in this batch."""
        return len(self.slices_2d)

    def execute_one(self, slice_index: int) -> Any:
        """Execute one slice through the runtime-owned invocation path."""
        return self.execute_slice(
            self.func,
            self.slices_2d[slice_index],
            self.kwargs,
            slice_index,
            self.slice_count,
        )


@dataclass(frozen=True, slots=True)
class RuntimeBatchCallableFamily:
    """Callable plus its raw processing ancestor for inherited batch contracts."""

    func: Callable[..., Any]
    raw_processing_function: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if self.raw_processing_function is not None and not callable(
            self.raw_processing_function
        ):
            raise TypeError(
                "raw_processing_function must be callable when inheriting runtime "
                "batch executors, got "
                f"{type(self.raw_processing_function).__name__}."
            )

    def executors(self) -> Mapping[RuntimeBatchExecutionDomain, Callable]:
        """Return batch executors declared by the wrapper family."""
        batch_executors = dict(runtime_batch_executors_from_callable(self.func))
        if self.raw_processing_function is not None:
            inherited = runtime_batch_executors_from_callable(
                self.raw_processing_function
            )
            for domain, executor in inherited.items():
                batch_executors.setdefault(domain, executor)
        return MappingProxyType(batch_executors)


class Pure2DSliceBatchExecutor(RuntimeBatchExecutor):
    """Base contract for equivalent pure-2D slice batch execution."""

    @classmethod
    def default_executor(cls) -> "Pure2DSliceBatchExecutor":
        """Return the MRO-selected default pure-2D batch executor."""
        for executor_type in cls._concrete_subclasses():
            return executor_type()
        raise RuntimeError("No concrete Pure2DSliceBatchExecutor is registered.")

    @classmethod
    def _concrete_subclasses(cls) -> tuple[type["Pure2DSliceBatchExecutor"], ...]:
        """Return concrete subclasses in Python MRO discovery order."""
        concrete_types: list[type[Pure2DSliceBatchExecutor]] = []
        for subclass in cls.__subclasses__():
            concrete_types.extend(subclass._concrete_subclasses())
            if not getattr(subclass, "__abstractmethods__", ()):
                concrete_types.append(subclass)
        return tuple(dict.fromkeys(concrete_types))


class SerialPure2DSliceBatchExecutor(Pure2DSliceBatchExecutor):
    """Default single-process, single-thread pure-2D batch executor."""

    executor_name = "serial_pure_2d_slices"

    def __call__(
        self,
        request: RuntimePure2DSliceBatchRequest,
    ) -> list[Any]:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]


def runtime_batch_executors_from_callable(
    func: Callable,
) -> Mapping[RuntimeBatchExecutionDomain, Callable]:
    """Return declared batch executors keyed by runtime batch domain."""
    executors_field = RuntimeBatchCallableMetadataField.EXECUTORS
    legacy_pure_2d_field = RuntimeBatchCallableMetadataField.LEGACY_PURE_2D_EXECUTOR
    declared = executors_field.read(func, {})
    if declared is None:
        declared = {}
    if not isinstance(declared, Mapping):
        raise TypeError(
            f"{executors_field.owner_label(func)} must be a mapping."
        )
    batch_executors: dict[RuntimeBatchExecutionDomain, Callable] = {}
    for raw_domain, executor in declared.items():
        domain = (
            raw_domain
            if isinstance(raw_domain, RuntimeBatchExecutionDomain)
            else RuntimeBatchExecutionDomain(str(raw_domain))
        )
        if not callable(executor):
            raise TypeError(
                f"{executors_field.owner_label(func)}[{domain.value!r}] must "
                f"be callable, got {type(executor).__name__}."
            )
        batch_executors[domain] = executor

    legacy_executor = legacy_pure_2d_field.read(func)
    if legacy_executor is not None:
        if not callable(legacy_executor):
            raise TypeError(
                f"{legacy_pure_2d_field.owner_label(func)} must be callable, "
                f"got {type(legacy_executor).__name__}."
            )
        batch_executors.setdefault(
            RuntimeBatchExecutionDomain.PURE_2D_SLICES,
            legacy_executor,
        )
    return MappingProxyType(batch_executors)


def runtime_batch_executor(
    domain: RuntimeBatchExecutionDomain,
    executor: Callable,
) -> Callable[[F], F]:
    """Declare a callable-owned batch executor for one runtime batch domain."""
    if not isinstance(domain, RuntimeBatchExecutionDomain):
        raise TypeError(
            "runtime_batch_executor domain must be RuntimeBatchExecutionDomain, "
            f"got {type(domain).__name__}."
        )
    if not callable(executor):
        raise TypeError(
            "runtime_batch_executor executor must be callable, "
            f"got {type(executor).__name__}."
        )

    def decorator(func: F) -> F:
        batch_executors = dict(runtime_batch_executors_from_callable(func))
        batch_executors[domain] = executor
        RuntimeBatchCallableMetadataField.EXECUTORS.write(
            func,
            MappingProxyType(batch_executors),
        )
        if domain is RuntimeBatchExecutionDomain.PURE_2D_SLICES:
            RuntimeBatchCallableMetadataField.LEGACY_PURE_2D_EXECUTOR.write(
                func,
                executor,
            )
        return func

    return decorator


def pure_2d_batch_executor(executor: Callable) -> Callable[[F], F]:
    """Declare a batch executor for equivalent pure-2D slice invocations."""
    return runtime_batch_executor(
        RuntimeBatchExecutionDomain.PURE_2D_SLICES,
        executor,
    )


def measurement_image_batch_executor(executor: Callable) -> Callable[[F], F]:
    """Declare a batch executor for equivalent measurement-image invocations."""
    return runtime_batch_executor(
        RuntimeBatchExecutionDomain.MEASUREMENT_IMAGES,
        executor,
    )
