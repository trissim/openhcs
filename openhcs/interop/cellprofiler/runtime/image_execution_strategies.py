"""Image execution strategies for CellProfiler runtime payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import time
from typing import ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta

from openhcs.core.callable_contract import CallableContract
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    CellProfilerProcessingContractAuthority,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
        CellProfilerFunctionContractExecutor,
    )


class CellProfilerImageExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal executor mode family for CellProfiler image payload semantics."""

    __registry_key__ = "mode_key"
    __skip_if_no_key__ = True
    mode: ClassVar[ImagePayloadExecutionMode | None] = None
    mode_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_mode(
        cls,
        mode: ImagePayloadExecutionMode,
    ) -> "CellProfilerImageExecutionStrategy":
        return cls.__registry__[mode.value]()

    @abstractmethod
    def execute(
        self,
        executor: CellProfilerFunctionContractExecutor,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> CellProfilerRuntimeValue:
        """Execute one resolved image payload according to its nominal mode."""


class NaturalImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Delegate natural payloads through the callable processing contract."""

    mode = ImagePayloadExecutionMode.NATURAL
    mode_key = ImagePayloadExecutionMode.NATURAL.value

    def execute(
        self,
        executor: CellProfilerFunctionContractExecutor,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> CellProfilerRuntimeValue:
        function_name = CallableContract.from_callable(func).function_name
        contract_started_at = time.perf_counter()
        contract = CellProfilerProcessingContractAuthority.for_callable(func)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_natural_processing_contract",
            time.perf_counter() - contract_started_at,
            function=function_name,
            contract=contract.name,
        )
        execute_started_at = time.perf_counter()
        result = contract.execute(
            executor,
            func,
            image,
            **kwargs,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_natural_contract_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            contract=contract.name,
        )
        return result


class FullStackImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute an already-volumetric payload without per-slice rewriting."""

    mode = ImagePayloadExecutionMode.FULL_STACK
    mode_key = ImagePayloadExecutionMode.FULL_STACK.value

    def execute(
        self,
        executor: CellProfilerFunctionContractExecutor,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> CellProfilerRuntimeValue:
        return executor.execute_full_stack(func, image, **kwargs)


class AlignedMultiImageStackExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute aligned multi-image bundles slice-by-slice as a single payload."""

    mode = ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    mode_key = ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK.value

    def execute(
        self,
        executor: CellProfilerFunctionContractExecutor,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> CellProfilerRuntimeValue:
        return executor._execute_aligned_multi_image_stack(
            func,
            image,
            **dict(kwargs),
        )
