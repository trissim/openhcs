from __future__ import annotations

import pytest

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.worker_start_policy import (
    WorkerStartPlatform,
    WorkerStartExecutionFacts,
    resolve_worker_start_context,
)


def test_linux_cpu_server_honors_configured_fork() -> None:
    decision = resolve_worker_start_context(
        MultiprocessingStartMethod.FORK,
        server_mode=True,
        gpu_enabled=False,
        platform=WorkerStartPlatform("linux"),
    )

    assert decision.resolved is MultiprocessingStartMethod.FORK
    assert decision.context.get_start_method() == "fork"


def test_linux_gpu_execution_downgrades_fork_to_spawn() -> None:
    decision = resolve_worker_start_context(
        MultiprocessingStartMethod.FORK,
        server_mode=True,
        gpu_enabled=True,
        platform=WorkerStartPlatform("linux"),
    )

    assert decision.requested is MultiprocessingStartMethod.FORK
    assert decision.resolved is MultiprocessingStartMethod.SPAWN
    assert decision.changed
    assert decision.context.get_start_method() == "spawn"


def test_windows_rejects_fork_request() -> None:
    with pytest.raises(ValueError, match="Windows only supports"):
        resolve_worker_start_context(
            MultiprocessingStartMethod.FORK,
            server_mode=True,
            gpu_enabled=False,
            platform=WorkerStartPlatform("win32"),
        )


def test_macos_downgrades_fork_without_unsafe_override() -> None:
    decision = resolve_worker_start_context(
        MultiprocessingStartMethod.FORK,
        server_mode=True,
        gpu_enabled=False,
        platform=WorkerStartPlatform("darwin"),
    )

    assert decision.resolved is MultiprocessingStartMethod.SPAWN
    assert "macOS" in decision.reason


def test_execution_facts_use_compiled_step_plan_semantics() -> None:
    compiled_contexts = {
        "A01": ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name="cpu",
                    step_type="function",
                    axis_id="A01",
                    input_memory_type="numpy",
                    output_memory_type="numpy",
                ),
                1: CompiledStepPlan(
                    step_index=1,
                    step_name="gpu",
                    step_type="function",
                    axis_id="A01",
                    output_memory_type="cupy",
                ),
            },
        )
    }

    assert WorkerStartExecutionFacts.from_compiled_contexts(compiled_contexts).gpu_enabled


def test_execution_facts_treat_numpy_contexts_as_cpu() -> None:
    compiled_contexts = {
        "A01": ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name="cpu",
                    step_type="function",
                    axis_id="A01",
                    input_memory_type="numpy",
                    output_memory_type="numpy",
                )
            },
        )
    }

    assert not WorkerStartExecutionFacts.from_compiled_contexts(
        compiled_contexts
    ).gpu_enabled
