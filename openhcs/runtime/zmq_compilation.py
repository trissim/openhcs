"""Compilation and compile-artifact reuse for ZMQ execution."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Mapping, TYPE_CHECKING

from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.runtime.zmq_progress import (
    ZMQCompileProgressHeartbeat,
    ZMQProgressEmitter,
)

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


logger = logging.getLogger(__name__)


def extract_compiled_axis_ids(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> list[str]:
    """Extract unique multiprocessing axis ids from compiled context keys."""

    axis_ids: list[str] = []
    seen: set[str] = set()
    for context_key in compiled_contexts.keys():
        axis_id = (
            context_key.split("__combo_", 1)[0]
            if "__combo_" in context_key
            else context_key
        )
        if axis_id not in seen:
            seen.add(axis_id)
            axis_ids.append(axis_id)
    return sorted(axis_ids)


def extract_compiled_step_names(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> list[str]:
    """Extract ordered step names from compiled step plans."""

    if not compiled_contexts:
        raise ValueError("Compile artifact missing compiled_contexts")

    ordered_names: list[str] | None = None
    for context_key, context in compiled_contexts.items():
        step_names = [
            plan.step_name
            for _step_index, plan in sorted(context.step_plans.items())
        ]
        if ordered_names is None:
            ordered_names = step_names
            continue
        if step_names != ordered_names:
            raise ValueError(
                "Compiled contexts disagree on step names: "
                f"{context_key} has {step_names}, expected {ordered_names}"
            )

    return [] if ordered_names is None else ordered_names


@dataclass(frozen=True, slots=True)
class ZMQCompilationResult:
    """Compiled execution artifacts needed by worker execution."""

    execution_bundle: CompiledExecutionBundle
    compiled_axis_ids: list[str]
    output_plate_root: str | None = None
    auto_add_output_plate: bool | None = None

    @property
    def worker_assignments(self) -> dict[str, list[str]]:
        return {
            worker_slot: list(axis_ids)
            for worker_slot, axis_ids in self.execution_bundle.worker_assignments.items()
        }


@dataclass(frozen=True, slots=True)
class ZMQCompilationRequest:
    """Compile or reuse a compile artifact for one ZMQ execution."""

    execution_id: str
    plate_id: str
    pipeline_steps: Sequence[AbstractStep]
    orchestrator: "PipelineOrchestrator"
    wells: list[str]
    compile_artifact_id: str | None
    request_signature: str
    debug_replay_signature: str
    retain_compile_artifact: bool
    compiled_artifacts: MutableMapping[str, "ZMQCompileArtifactRecord"]
    progress_emitter: ZMQProgressEmitter
    flush_progress: Callable[[], None]
    immediate_progress_queue: Any
    debug_execution_policy: Any
    compile_heartbeat_interval_seconds: float = 2.0

    def resolve(self) -> ZMQCompilationResult:
        if self.compile_artifact_id is not None:
            return self.reuse_artifact()
        return self.compile_fresh()

    def reuse_artifact(self) -> ZMQCompilationResult:
        artifact = self.compiled_artifacts.get(self.compile_artifact_id)
        if artifact is not None and not self.retain_compile_artifact:
            artifact = self.compiled_artifacts.pop(self.compile_artifact_id)
        if artifact is None:
            raise ValueError(
                f"Missing compile artifact '{self.compile_artifact_id}'. "
                "Re-run compilation before execution."
            )
        expected_signature = (
            self.debug_replay_signature
            if self.retain_compile_artifact
            else self.request_signature
        )
        artifact_signature = artifact.signature_for_retain_policy(
            self.retain_compile_artifact
        )
        if artifact_signature != expected_signature:
            logger.error(
                "[%s] Compile artifact signature mismatch: artifact_id=%s artifact_sig=%s request_sig=%s",
                self.execution_id,
                self.compile_artifact_id,
                artifact_signature[:12],
                expected_signature[:12],
            )
            raise ValueError(
                f"Compile artifact '{self.compile_artifact_id}' does not match execution request"
            )
        if artifact.plate_id != str(self.plate_id):
            raise ValueError(
                f"Compile artifact '{self.compile_artifact_id}' is for plate "
                f"{artifact.plate_id}, not {self.plate_id}"
            )

        execution_bundle = artifact.compilation.execution_bundle
        compiled_contexts = execution_bundle.runtime_contexts
        if not compiled_contexts:
            raise ValueError("Compile artifact missing compiled_contexts")
        worker_assignments = {
            worker_slot: list(axis_ids)
            for worker_slot, axis_ids in execution_bundle.worker_assignments.items()
        }
        compiled_axis_ids = extract_compiled_axis_ids(compiled_contexts)
        compiled_step_names = extract_compiled_step_names(compiled_contexts)
        self.progress_emitter.artifact_init_started(
            compiled_axis_ids=compiled_axis_ids,
            worker_assignments=worker_assignments,
            step_names=compiled_step_names,
        )
        logger.info(
            "[%s] Reused compile artifact %s for plate %s (sig=%s)",
            self.execution_id,
            self.compile_artifact_id,
            self.plate_id,
            expected_signature[:12],
        )
        return ZMQCompilationResult(
            execution_bundle=execution_bundle,
            compiled_axis_ids=compiled_axis_ids,
            output_plate_root=artifact.compilation.output_plate_root,
            auto_add_output_plate=artifact.compilation.auto_add_output_plate,
        )

    def compile_fresh(self) -> ZMQCompilationResult:
        from openhcs.core.progress import set_progress_queue

        set_progress_queue(self.immediate_progress_queue)
        try:
            with ZMQCompileProgressHeartbeat(
                progress_emitter=self.progress_emitter,
                step_count=len(self.pipeline_steps),
                flush_progress=self.flush_progress,
                interval_seconds=self.compile_heartbeat_interval_seconds,
            ):
                compilation = self.orchestrator.compile_pipelines(
                    pipeline_definition=self.pipeline_steps,
                    well_filter=self.wells,
                    is_zmq_execution=True,
                    debug_execution_policy=self.debug_execution_policy,
                )
        finally:
            set_progress_queue(None)

        if not isinstance(compilation, dict) or "execution_bundle" not in compilation:
            raise ValueError("Compilation did not return execution_bundle")
        execution_bundle = compilation["execution_bundle"]
        compiled_contexts = execution_bundle.runtime_contexts
        if not compiled_contexts:
            raise ValueError("Compilation produced no compiled contexts")

        worker_assignments = {
            worker_slot: list(axis_ids)
            for worker_slot, axis_ids in execution_bundle.worker_assignments.items()
        }
        compiled_axis_ids = extract_compiled_axis_ids(compiled_contexts)
        compiled_step_names = extract_compiled_step_names(compiled_contexts)
        self.progress_emitter.compiled_init_started(
            compiled_axis_ids=compiled_axis_ids,
            worker_assignments=worker_assignments,
        )
        self.progress_emitter.compile_succeeded(
            step_count=len(compiled_step_names),
            compiled_axis_ids=compiled_axis_ids,
            worker_assignments=worker_assignments,
        )
        self.flush_progress()
        for axis_id in compiled_axis_ids:
            self.progress_emitter.axis_compile_succeeded(axis_id)
        self.flush_progress()

        first_context = next(iter(compiled_contexts.values()))
        output_plate_root = first_context.output_plate_root
        auto_add_output_plate = bool(first_context.auto_add_output_plate_to_plate_manager)
        logger.info(
            "[%s] Captured auto_add_output_plate=%s output_plate_root=%s",
            self.execution_id,
            auto_add_output_plate,
            output_plate_root,
        )
        return ZMQCompilationResult(
            execution_bundle=execution_bundle,
            compiled_axis_ids=compiled_axis_ids,
            output_plate_root=None if output_plate_root is None else str(output_plate_root),
            auto_add_output_plate=auto_add_output_plate,
        )


@dataclass(frozen=True, slots=True)
class ZMQCompileArtifactRecord:
    """Stored compile-only artifact payload."""

    execution_id: str
    plate_id: str
    request_signature: str
    debug_replay_signature: str
    compilation: ZMQCompilationResult
    created_at: float = field(default_factory=time.time)

    def signature_for_retain_policy(self, retain_compile_artifact: bool) -> str:
        if retain_compile_artifact:
            return self.debug_replay_signature
        return self.request_signature
