"""Compiler-owned execution artifacts for worker transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from openhcs.config_framework.lazy_factory import (
    resolve_lazy_configurations_for_serialization,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep


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

    @classmethod
    def from_runtime_contexts(
        cls,
        *,
        pipeline_definition: Sequence[AbstractStep],
        runtime_contexts: Mapping[str, ProcessingContext],
        worker_assignments: Mapping[str, list[str]],
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
        )

    @classmethod
    def from_unassigned_runtime_contexts(
        cls,
        *,
        pipeline_definition: Sequence[AbstractStep],
        runtime_contexts: Mapping[str, ProcessingContext],
    ) -> "CompiledExecutionBundle":
        """Create a bundle before worker ownership has been assigned."""

        return cls.from_runtime_contexts(
            pipeline_definition=pipeline_definition,
            runtime_contexts=runtime_contexts,
            worker_assignments={},
        )

    def as_compilation_result(self) -> dict[str, Any]:
        return {
            "pipeline_definition": self.pipeline_definition,
            "compiled_contexts": self.runtime_contexts,
            "worker_assignments": self.worker_assignments,
            "execution_bundle": self,
        }
