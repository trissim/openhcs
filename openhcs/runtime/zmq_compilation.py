"""Compilation and compile-artifact reuse for ZMQ execution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

from zmqruntime.messages import MessageFields

from openhcs.runtime.zmq_progress import ZMQProgressEmitter


logger = logging.getLogger(__name__)


def extract_compiled_axis_ids(compiled_contexts: dict[str, Any]) -> list[str]:
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


@dataclass(frozen=True, slots=True)
class ZMQCompilationResult:
    """Compiled execution artifacts needed by worker execution."""

    execution_bundle: Any
    compiled_contexts: dict[str, Any]
    compiled_pipeline_definition: Any
    worker_assignments: dict[str, list[str]]
    compiled_axis_ids: list[str]
    output_plate_root: str | None = None
    auto_add_output_plate: bool | None = None


@dataclass(frozen=True, slots=True)
class ZMQCompilationRequest:
    """Compile or reuse a compile artifact for one ZMQ execution."""

    execution_id: str
    plate_id: str
    pipeline_steps: list[Any]
    orchestrator: Any
    wells: list[str]
    compile_artifact_id: str | None
    request_signature: str
    debug_replay_signature: str
    retain_compile_artifact: bool
    compiled_artifacts: dict[str, dict[str, Any]]
    progress_emitter: ZMQProgressEmitter
    flush_progress: Callable[[], None]
    immediate_progress_queue: Any

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
        signature_key = (
            "debug_replay_signature"
            if self.retain_compile_artifact
            else "request_signature"
        )
        expected_signature = (
            self.debug_replay_signature
            if self.retain_compile_artifact
            else self.request_signature
        )
        if artifact[signature_key] != expected_signature:
            logger.error(
                "[%s] Compile artifact signature mismatch: artifact_id=%s artifact_sig=%s request_sig=%s",
                self.execution_id,
                self.compile_artifact_id,
                str(artifact[signature_key])[:12],
                expected_signature[:12],
            )
            raise ValueError(
                f"Compile artifact '{self.compile_artifact_id}' does not match execution request"
            )
        if artifact[MessageFields.PLATE_ID] != str(self.plate_id):
            raise ValueError(
                f"Compile artifact '{self.compile_artifact_id}' is for plate "
                f"{artifact[MessageFields.PLATE_ID]}, not {self.plate_id}"
            )

        execution_bundle = artifact["execution_bundle"]
        compiled_contexts = execution_bundle.runtime_contexts
        if compiled_contexts is None:
            raise ValueError("Compile artifact missing compiled_contexts")
        worker_assignments = dict(execution_bundle.worker_assignments)
        compiled_axis_ids = extract_compiled_axis_ids(compiled_contexts)
        self.progress_emitter.artifact_init_started(
            compiled_axis_ids=compiled_axis_ids,
            worker_assignments=worker_assignments,
            step_names=[step.name for step in self.pipeline_steps],
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
            compiled_contexts=compiled_contexts,
            compiled_pipeline_definition=artifact.get("compiled_pipeline_definition"),
            worker_assignments=worker_assignments,
            compiled_axis_ids=compiled_axis_ids,
            output_plate_root=(
                None
                if artifact.get("output_plate_root") is None
                else str(artifact["output_plate_root"])
            ),
            auto_add_output_plate=(
                None
                if artifact.get("auto_add_output_plate") is None
                else bool(artifact["auto_add_output_plate"])
            ),
        )

    def compile_fresh(self) -> ZMQCompilationResult:
        from openhcs.core.progress import set_progress_queue

        set_progress_queue(self.immediate_progress_queue)
        try:
            compilation = self.orchestrator.compile_pipelines(
                pipeline_definition=self.pipeline_steps,
                well_filter=self.wells,
                is_zmq_execution=True,
            )
        finally:
            set_progress_queue(None)

        if not isinstance(compilation, dict) or "execution_bundle" not in compilation:
            raise ValueError("Compilation did not return execution_bundle")
        execution_bundle = compilation["execution_bundle"]
        compiled_contexts = execution_bundle.runtime_contexts
        compiled_pipeline_definition = compilation.get(
            "pipeline_definition", self.pipeline_steps
        )
        if not compiled_contexts:
            raise ValueError("Compilation produced no compiled contexts")

        worker_assignments = compilation["worker_assignments"]
        compiled_axis_ids = extract_compiled_axis_ids(compiled_contexts)
        self.progress_emitter.compiled_init_started(
            compiled_axis_ids=compiled_axis_ids,
            worker_assignments=worker_assignments,
        )
        self.progress_emitter.compile_succeeded(
            step_count=len(self.pipeline_steps),
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
            compiled_contexts=compiled_contexts,
            compiled_pipeline_definition=compiled_pipeline_definition,
            worker_assignments=worker_assignments,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_at": time.time(),
            "request_signature": self.request_signature,
            "debug_replay_signature": self.debug_replay_signature,
            MessageFields.PLATE_ID: str(self.plate_id),
            "execution_bundle": self.compilation.execution_bundle,
            "compiled_pipeline_definition": self.compilation.compiled_pipeline_definition,
            "output_plate_root": self.compilation.output_plate_root,
            "auto_add_output_plate": self.compilation.auto_add_output_plate,
        }

