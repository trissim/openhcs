"""Compiler-owned execution artifacts for worker transport."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from openhcs.config_framework.lazy_factory import (
    resolve_lazy_configurations_for_serialization,
)
from openhcs.core.config import GlobalPipelineConfig, MultiprocessingStartMethod
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.worker_start_policy import (
    WorkerStartDecision,
    WorkerStartExecutionFacts,
    resolve_worker_start_context,
)


@dataclass(frozen=True, slots=True)
class CompiledGpuRegistryPlan:
    """Compiled worker-process GPU registry initialization inputs."""

    configured_num_workers: int

    @classmethod
    def from_global_config(
        cls,
        global_config: GlobalPipelineConfig,
    ) -> "CompiledGpuRegistryPlan":
        return cls(configured_num_workers=global_config.num_workers)

    def setup_global_registry(self) -> None:
        """Initialize the worker GPU registry from compiled scalar facts."""
        from openhcs.core.orchestrator.gpu_scheduler import (
            initialize_gpu_registry,
            is_gpu_registry_initialized,
        )

        if is_gpu_registry_initialized():
            return
        initialize_gpu_registry(configured_num_workers=self.configured_num_workers)


@dataclass(frozen=True, slots=True)
class CompiledWorkerStartPlan:
    """Compiled multiprocessing start method decision."""

    requested: MultiprocessingStartMethod
    resolved: MultiprocessingStartMethod
    reason: str
    gpu_enabled: bool
    server_mode: bool

    @classmethod
    def from_requested(
        cls,
        requested: MultiprocessingStartMethod,
        *,
        server_mode: bool,
        execution_facts: WorkerStartExecutionFacts,
    ) -> "CompiledWorkerStartPlan":
        decision = resolve_worker_start_context(
            requested,
            server_mode=server_mode,
            gpu_enabled=execution_facts.gpu_enabled,
        )
        return cls(
            requested=decision.requested,
            resolved=decision.resolved,
            reason=decision.reason,
            gpu_enabled=execution_facts.gpu_enabled,
            server_mode=server_mode,
        )

    @property
    def changed(self) -> bool:
        return self.requested is not self.resolved

    def multiprocessing_context(self):
        import multiprocessing

        return multiprocessing.get_context(self.resolved.value)

    def as_decision(self) -> WorkerStartDecision:
        return WorkerStartDecision(
            requested=self.requested,
            resolved=self.resolved,
            reason=self.reason,
            context=self.multiprocessing_context(),
        )


@dataclass(frozen=True, slots=True)
class CompiledRuntimeEnvironmentPlan:
    """Compiled runtime environment facts consumed by worker execution."""

    worker_start: CompiledWorkerStartPlan
    use_threading: bool
    gpu_registry: CompiledGpuRegistryPlan

    @classmethod
    def from_global_config(
        cls,
        global_config: GlobalPipelineConfig,
        *,
        compiled_contexts: Mapping[str, ProcessingContext] | None,
        server_mode: bool,
    ) -> "CompiledRuntimeEnvironmentPlan":
        execution_facts = WorkerStartExecutionFacts.from_compiled_contexts(
            compiled_contexts
        )
        return cls(
            worker_start=CompiledWorkerStartPlan.from_requested(
                global_config.multiprocessing_start_method,
                server_mode=server_mode,
                execution_facts=execution_facts,
            ),
            use_threading=global_config.use_threading,
            gpu_registry=CompiledGpuRegistryPlan.from_global_config(global_config),
        )

    @property
    def multiprocessing_start_method(self) -> MultiprocessingStartMethod:
        return self.worker_start.resolved

    @property
    def configured_num_workers(self) -> int:
        return self.gpu_registry.configured_num_workers

    def with_execution_shape(
        self,
        *,
        use_threading: bool,
        configured_num_workers: int,
    ) -> "CompiledRuntimeEnvironmentPlan":
        """Return this compiled plan with worker-shape scalars replaced."""

        return replace(
            self,
            use_threading=use_threading,
            gpu_registry=CompiledGpuRegistryPlan(
                configured_num_workers=configured_num_workers
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledExecutionBundle:
    """Compiled execution data split by semantic role.

    ``runtime_contexts`` preserve the rich in-process compiled state used for
    runtime facts and fork inheritance. ``transport_contexts`` are the
    pickle-safe contexts submitted through worker queues.
    """

    pipeline_definition: Sequence[AbstractStep]
    runtime_contexts: Mapping[str, ProcessingContext]
    transport_contexts: Mapping[str, ProcessingContext]
    worker_assignments: Mapping[str, list[str]]
    runtime_environment: CompiledRuntimeEnvironmentPlan

    @classmethod
    def from_runtime_contexts(
        cls,
        *,
        pipeline_definition: Sequence[AbstractStep],
        runtime_contexts: Mapping[str, ProcessingContext],
        worker_assignments: Mapping[str, list[str]],
        runtime_environment: CompiledRuntimeEnvironmentPlan,
    ) -> "CompiledExecutionBundle":
        transport_contexts = resolve_lazy_configurations_for_serialization(
            dict(runtime_contexts.items())
        )
        return cls(
            pipeline_definition=pipeline_definition,
            runtime_contexts=dict(runtime_contexts.items()),
            transport_contexts=transport_contexts,
            worker_assignments={
                key: list(value) for key, value in worker_assignments.items()
            },
            runtime_environment=runtime_environment,
        )

    @classmethod
    def from_unassigned_runtime_contexts(
        cls,
        *,
        pipeline_definition: Sequence[AbstractStep],
        runtime_contexts: Mapping[str, ProcessingContext],
        runtime_environment: CompiledRuntimeEnvironmentPlan,
    ) -> "CompiledExecutionBundle":
        """Create a bundle before worker ownership has been assigned."""

        return cls.from_runtime_contexts(
            pipeline_definition=pipeline_definition,
            runtime_contexts=runtime_contexts,
            worker_assignments={},
            runtime_environment=runtime_environment,
        )

    def as_compilation_result(self) -> dict[str, "CompiledExecutionBundle"]:
        return {
            "execution_bundle": self,
        }
