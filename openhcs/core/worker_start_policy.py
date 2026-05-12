"""Worker process start policy for OpenHCS execution."""

from __future__ import annotations

import multiprocessing
import sys
from dataclasses import dataclass
from typing import Mapping

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import GlobalPipelineConfig, MultiprocessingStartMethod
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.constants import VALID_GPU_MEMORY_TYPES


@dataclass(frozen=True)
class WorkerStartStepFacts:
    """Worker-start-relevant facts from one compiled step plan."""

    input_memory_type: str | None = None
    output_memory_type: str | None = None
    gpu_id: int | None = None

    @classmethod
    def from_compiled_step_plan(
        cls,
        step_plan: CompiledStepPlan,
    ) -> "WorkerStartStepFacts":
        return cls(
            input_memory_type=step_plan.input_memory_type,
            output_memory_type=step_plan.output_memory_type,
            gpu_id=step_plan.gpu_id,
        )

    @property
    def uses_gpu(self) -> bool:
        return (
            self.gpu_id is not None
            or self.input_memory_type in VALID_GPU_MEMORY_TYPES
            or self.output_memory_type in VALID_GPU_MEMORY_TYPES
        )


@dataclass(frozen=True)
class WorkerStartExecutionFacts:
    """Compiled execution facts that constrain worker start policy."""

    gpu_enabled: bool = False

    @classmethod
    def from_compiled_contexts(
        cls,
        compiled_contexts: Mapping[str, ProcessingContext] | None,
    ) -> "WorkerStartExecutionFacts":
        if not compiled_contexts:
            return cls()

        for context in compiled_contexts.values():
            for step_plan in context.step_plans.values():
                if WorkerStartStepFacts.from_compiled_step_plan(step_plan).uses_gpu:
                    return cls(gpu_enabled=True)
        return cls()


@dataclass(frozen=True)
class WorkerStartPlatform:
    """Platform capabilities relevant to multiprocessing start selection."""

    name: str

    @classmethod
    def current(cls) -> "WorkerStartPlatform":
        return cls(name=sys.platform)

    @property
    def is_windows(self) -> bool:
        return self.name.startswith("win")

    @property
    def is_macos(self) -> bool:
        return self.name == "darwin"

    @property
    def is_linux(self) -> bool:
        return self.name.startswith("linux")


@dataclass(frozen=True)
class WorkerStartDecision:
    """Resolved multiprocessing context plus the policy reason."""

    requested: MultiprocessingStartMethod
    resolved: MultiprocessingStartMethod
    reason: str
    context: multiprocessing.context.BaseContext

    @property
    def changed(self) -> bool:
        return self.requested is not self.resolved


def resolve_worker_start_context(
    global_config: GlobalPipelineConfig,
    *,
    server_mode: bool,
    gpu_enabled: bool,
    allow_unsafe_fork: bool = False,
    platform: WorkerStartPlatform | None = None,
) -> WorkerStartDecision:
    """Resolve the worker start method for one execution.

    The resolver owns worker-start safety policy so execution entry points do not
    hardcode their own spawn/fork branches. The returned context should be used
    for every multiprocessing primitive created for the same execution.
    """

    requested = global_config.multiprocessing_start_method
    platform = platform or WorkerStartPlatform.current()

    if platform.is_windows:
        if requested is not MultiprocessingStartMethod.SPAWN:
            raise ValueError(
                f"Windows only supports {MultiprocessingStartMethod.SPAWN.value!r} "
                f"worker start in OpenHCS, got {requested.value!r}."
            )
        return _decision(requested, requested, "windows requires spawn")

    if requested is MultiprocessingStartMethod.SPAWN:
        return _decision(requested, requested, "spawn requested")

    if platform.is_macos and not allow_unsafe_fork:
        return _decision(
            requested,
            MultiprocessingStartMethod.SPAWN,
            "macOS fork-style workers require explicit unsafe override",
        )

    if gpu_enabled and not allow_unsafe_fork:
        return _decision(
            requested,
            MultiprocessingStartMethod.SPAWN,
            "GPU execution requires spawn unless unsafe fork is explicitly allowed",
        )

    if platform.is_linux:
        return _decision(
            requested,
            requested,
            "linux CPU/server execution honors configured worker start method"
            if server_mode
            else "linux CPU execution honors configured worker start method",
        )

    if allow_unsafe_fork:
        return _decision(
            requested,
            requested,
            "unsafe fork override honors configured worker start method",
        )

    return _decision(
        requested,
        MultiprocessingStartMethod.SPAWN,
        f"unknown platform {platform.name!r} falls back to spawn",
    )


def _decision(
    requested: MultiprocessingStartMethod,
    resolved: MultiprocessingStartMethod,
    reason: str,
) -> WorkerStartDecision:
    return WorkerStartDecision(
        requested=requested,
        resolved=resolved,
        reason=reason,
        context=multiprocessing.get_context(resolved.value),
    )
