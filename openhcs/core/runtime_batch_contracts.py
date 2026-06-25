"""Runtime batch execution contracts shared by compiler and pipeline decorators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.runtime_invocation import runtime_callable_defaults
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeSliceProjectionStrategy,
    RuntimeProjectionAxis,
)


F = TypeVar("F", bound=Callable)
RuntimeSliceDataT = TypeVar("RuntimeSliceDataT")
RuntimeSliceResultT = TypeVar("RuntimeSliceResultT")
RuntimeKwargValueT = TypeVar("RuntimeKwargValueT")


class RuntimeBatchExecutionDomain(str, Enum):
    """Nominal domains where one callable can batch equivalent invocations."""

    PURE_2D_SLICES = "pure_2d_slices"
    MEASUREMENT_IMAGES = "measurement_images"


class RuntimeBatchCallableMetadataField(str, Enum):
    """Callable metadata fields owned by runtime batch contracts."""

    EXECUTORS = "__openhcs_runtime_batch_executors__"

    def owner_label(self, func: Callable) -> str:
        """Return a compact field label for validation errors."""
        return f"{func}.{self.value}"


RUNTIME_BATCH_EXECUTORS_ATTR = RuntimeBatchCallableMetadataField.EXECUTORS.value


@dataclass(frozen=True, slots=True)
class RuntimePure2DSliceBatchRequest(
    Generic[RuntimeSliceDataT, RuntimeSliceResultT, RuntimeKwargValueT],
):
    """Nominal request for equivalent pure-2D slice invocations."""

    func: Callable[..., RuntimeSliceResultT]
    slices_2d: tuple[RuntimeSliceDataT, ...]
    kwargs: Mapping[str, RuntimeKwargValueT]
    execute_slice: Callable[
        [
            Callable[..., RuntimeSliceResultT],
            RuntimeSliceDataT,
            Mapping[str, RuntimeKwargValueT],
            int,
            int,
        ],
        RuntimeSliceResultT,
    ]

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

    @property
    def slice_count(self) -> int:
        """Number of slice invocations in this batch."""
        return len(self.slices_2d)

    def execute_one(self, slice_index: int) -> RuntimeSliceResultT:
        """Execute one slice through the runtime-owned invocation path."""
        return self.execute_one_with_kwargs(slice_index, self.kwargs)

    def execute_one_with_kwargs(
        self,
        slice_index: int,
        kwargs: Mapping[str, RuntimeKwargValueT],
    ) -> RuntimeSliceResultT:
        """Execute one slice and stamp the result with its runtime-slice identity."""
        result = self.execute_slice(
            self.func,
            self.slices_2d[slice_index],
            kwargs,
            slice_index,
            self.slice_count,
        )
        return RuntimeSliceProjectionStrategy.strategy_for_value(
            result
        ).identity_projected_value(
            result,
            RuntimeProjectionAxis(slice_index=slice_index, extent=self.slice_count),
        )


class RuntimeBatchExecutor(ABC, metaclass=AutoRegisterMeta):
    """Nominal callable object for reusable runtime batch execution policies."""

    __registry_key__ = "executor_name"
    __skip_if_no_key__ = True
    executor_name: ClassVar[str | None] = None

    @abstractmethod
    def __call__(
        self,
        request: RuntimePure2DSliceBatchRequest[
            RuntimeSliceDataT,
            RuntimeSliceResultT,
            RuntimeKwargValueT,
        ],
    ) -> list[RuntimeSliceResultT]:
        """Execute one runtime batch domain."""

@dataclass(frozen=True, slots=True)
class RuntimeBatchCallableFamily:
    """Callable plus its raw processing ancestor for inherited batch contracts."""

    func: Callable
    raw_processing_function: Callable | None = None

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
                if domain not in batch_executors:
                    batch_executors[domain] = executor
        return MappingProxyType(batch_executors)


class Pure2DSliceBatchExecutor(RuntimeBatchExecutor):
    """Base contract for equivalent pure-2D slice batch execution."""

    @classmethod
    def default_executor(cls) -> "Pure2DSliceBatchExecutor":
        """Return the single-thread default pure-2D batch executor."""
        return SerialPure2DSliceBatchExecutor()


class ParallelPure2DSliceBatchExecutor(Pure2DSliceBatchExecutor):
    """Explicit thread-backed executor for independent pure-2D slice batches."""

    executor_name = "parallel_pure_2d_slices"

    def __call__(
        self,
        request: RuntimePure2DSliceBatchRequest[
            RuntimeSliceDataT,
            RuntimeSliceResultT,
            RuntimeKwargValueT,
        ],
    ) -> list[RuntimeSliceResultT]:
        raise RuntimeError(
            "ParallelPure2DSliceBatchExecutor is disabled for single-thread runtime "
            "benchmarking. Use a process-level batching contract instead."
        )


class SerialPure2DSliceBatchExecutor(Pure2DSliceBatchExecutor):
    """Default single-process, single-thread pure-2D batch executor."""

    executor_name = "serial_pure_2d_slices"

    def __call__(
        self,
        request: RuntimePure2DSliceBatchRequest[
            RuntimeSliceDataT,
            RuntimeSliceResultT,
            RuntimeKwargValueT,
        ],
    ) -> list[RuntimeSliceResultT]:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]


def runtime_batch_executors_from_callable(
    func: Callable,
) -> Mapping[RuntimeBatchExecutionDomain, Callable]:
    """Return declared batch executors keyed by runtime batch domain."""
    executors_field = RuntimeBatchCallableMetadataField.EXECUTORS
    try:
        namespace = vars(func)
    except TypeError:
        declared = {}
    else:
        if executors_field.value in namespace:
            declared = namespace[executors_field.value]
        else:
            declared = {}
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
        try:
            namespace = vars(func)
        except TypeError as exc:
            raise TypeError(
                f"{func!r} cannot carry runtime batch executor metadata."
            ) from exc
        namespace[RuntimeBatchCallableMetadataField.EXECUTORS.value] = (
            MappingProxyType(batch_executors)
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
