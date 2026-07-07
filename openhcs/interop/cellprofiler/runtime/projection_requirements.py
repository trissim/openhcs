"""Nominal runtime-plane projection requirements for CellProfiler execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import CallableContract
from openhcs.core.registry_strategies import (
    AlwaysMatchesContextMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerFunction
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    CellProfilerProcessingContractAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class CellProfilerRuntimePlaneProjectionCapability(ABC, metaclass=AutoRegisterMeta):
    """Nominal runtime-plane projection capability used by CP invocation binding."""

    __registry_key__ = "capability_key"
    __skip_if_no_key__ = True
    capability_key: ClassVar[str | None] = None

    @classmethod
    def registered_capability_types(
        cls,
    ) -> tuple[type["CellProfilerRuntimePlaneProjectionCapability"], ...]:
        return tuple(cls.__registry__.values())


class RuntimeArtifactValueProjectionCapability(
    CellProfilerRuntimePlaneProjectionCapability
):
    """Projection capability for runtime artifacts entering plane-scoped calls."""


class RuntimeArtifactImageInputProjectionCapability(
    RuntimeArtifactValueProjectionCapability
):
    """Projection capability for image artifacts loaded from runtime stores."""

    capability_key = "runtime_artifact_image_input"


class RuntimeSliceKwargProjectionCapability(RuntimeArtifactValueProjectionCapability):
    """Projection capability for kwargs carrying runtime-slice-aligned values."""

    capability_key = "runtime_slice_kwarg"


class CurrentSourceImagePayloadProjectionCapability(
    CellProfilerRuntimePlaneProjectionCapability
):
    """Projection capability for source-identity matching against current image."""

    capability_key = "current_source_image_payload"


def projection_capabilities_include(
    declared_capabilities: frozenset[
        type[CellProfilerRuntimePlaneProjectionCapability]
    ],
    capability_type: type[CellProfilerRuntimePlaneProjectionCapability],
) -> bool:
    """Return whether declared capabilities include a capability family."""

    return any(
        issubclass(declared_type, capability_type)
        for declared_type in declared_capabilities
    )


@dataclass(frozen=True, slots=True)
class RuntimePlaneProjectionRequirementContext:
    """Callable/execution context for runtime-plane projection requirements."""

    func: CellProfilerFunction
    default_execution_mode: ImagePayloadExecutionMode

    @property
    def uses_natural_image_execution(self) -> bool:
        return self.default_execution_mode is ImagePayloadExecutionMode.NATURAL

    @property
    def callable_accepts_plane_scoped_runtime_values(self) -> bool:
        return (
            CellProfilerProcessingContractAuthority.for_callable(self.func)
            is not ProcessingContract.PURE_3D
        )

    @property
    def uses_full_stack_batch_execution(self) -> bool:
        return self.default_execution_mode is ImagePayloadExecutionMode.FULL_STACK


class RuntimePlaneProjectionRequirement(
    MostDerivedContextStrategyMixin[RuntimePlaneProjectionRequirementContext],
    ABC,
):
    """MRO-selected declaration of runtime-plane projection capabilities."""

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def capabilities_for_context(
        cls,
        context: RuntimePlaneProjectionRequirementContext,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        return cls.for_context(context).projection_capabilities()

    @abstractmethod
    def projection_capabilities(
        self,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        """Return the projection capabilities required by this context."""


class NoRuntimePlaneProjectionRequirement(
    AlwaysMatchesContextMixin[RuntimePlaneProjectionRequirementContext],
    RuntimePlaneProjectionRequirement,
):
    """Default declaration for callables that keep runtime values unprojected."""

    strategy_key = "none"

    def projection_capabilities(
        self,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        return frozenset()


class NaturalRuntimePlaneProjectionRequirement(NoRuntimePlaneProjectionRequirement):
    """Plane-scoped 2D invocation over runtime artifacts and aligned kwargs."""

    strategy_key = "natural_runtime_plane"

    def matches(self, context: RuntimePlaneProjectionRequirementContext) -> bool:
        return (
            context.uses_natural_image_execution
            and context.callable_accepts_plane_scoped_runtime_values
        )

    def projection_capabilities(
        self,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        return frozenset(
            (
                RuntimeArtifactImageInputProjectionCapability,
                RuntimeSliceKwargProjectionCapability,
            )
        )


class FullStackBatchRuntimePlaneProjectionRequirement(
    NoRuntimePlaneProjectionRequirement
):
    """Plane-scoped runtime values for full-stack batched non-3D callables."""

    strategy_key = "full_stack_batch_runtime_plane"

    def matches(self, context: RuntimePlaneProjectionRequirementContext) -> bool:
        return (
            context.uses_full_stack_batch_execution
            and context.callable_accepts_plane_scoped_runtime_values
        )

    def projection_capabilities(
        self,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        return frozenset(
            (
                RuntimeArtifactImageInputProjectionCapability,
                RuntimeSliceKwargProjectionCapability,
            )
        )
