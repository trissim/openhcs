"""Runtime batch execution contracts shared by compiler and pipeline decorators."""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
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

    def write(self, func: F, value: Any) -> F:
        """Write this metadata field to a callable and return the callable."""
        setattr(func, self.value, value)
        return func

    def owner_label(self, func: Callable) -> str:
        """Return a compact field label for validation errors."""
        return f"{func}.{self.value}"


RUNTIME_BATCH_EXECUTORS_ATTR = RuntimeBatchCallableMetadataField.EXECUTORS.value


@dataclass(frozen=True, slots=True)
class RuntimeBatchCallableMetadata:
    """Runtime-batch metadata projection for one callable."""

    namespace: Mapping[str, Any]

    @classmethod
    def from_callable(cls, func: Callable) -> "RuntimeBatchCallableMetadata":
        """Project callable metadata without probing dynamic attributes."""
        try:
            namespace = vars(func)
        except TypeError:
            namespace = {}
        return cls(namespace)

    def value(
        self,
        field: RuntimeBatchCallableMetadataField,
        default: Any = None,
    ) -> Any:
        """Read one runtime-batch metadata field."""
        return self.namespace.get(field.value, default)


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

    executor_priority: ClassVar[int] = 0

    @classmethod
    def default_executor(cls) -> "Pure2DSliceBatchExecutor":
        """Return the highest-priority registered pure-2D batch executor."""
        executor_types = cls._concrete_subclasses()
        if executor_types:
            return max(
                executor_types,
                key=lambda executor_type: executor_type.executor_priority,
            )()
        raise RuntimeError("No concrete Pure2DSliceBatchExecutor is registered.")

    @classmethod
    def _concrete_subclasses(cls) -> tuple[type["Pure2DSliceBatchExecutor"], ...]:
        """Return concrete subclasses in Python MRO discovery order."""
        concrete_types: list[type[Pure2DSliceBatchExecutor]] = []
        for subclass in cls.__subclasses__():
            concrete_types.extend(subclass._concrete_subclasses())
            if not inspect.isabstract(subclass):
                concrete_types.append(subclass)
        return tuple(dict.fromkeys(concrete_types))


class ParallelPure2DSliceBatchExecutor(Pure2DSliceBatchExecutor):
    """Default thread-backed executor for independent pure-2D slice batches."""

    executor_name = "parallel_pure_2d_slices"
    executor_priority = 100

    def __call__(
        self,
        request: RuntimePure2DSliceBatchRequest,
    ) -> list[Any]:
        max_workers = min(request.slice_count, os.cpu_count() or 1)
        if max_workers <= 1:
            return [request.execute_one(0)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(request.execute_one, range(request.slice_count)))


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
    metadata = RuntimeBatchCallableMetadata.from_callable(func)
    declared = metadata.value(executors_field, {})
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
