"""Compile framework-local devices from FunctionStep memory declarations."""

from __future__ import annotations

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.framework_device_resolver import resolve_framework_devices


def assign_framework_devices(
    step_plans: dict[int, CompiledStepPlan],
) -> None:
    """Assign the pipeline footprint and project exact bindings to each step."""

    required_memory_types = frozenset(
        memory_type
        for step_plan in step_plans.values()
        for memory_type in step_plan.gpu_memory_types
    )
    pipeline_assignment = resolve_framework_devices(required_memory_types)
    for step_plan in step_plans.values():
        step_plan.device_assignment = pipeline_assignment.select(
            step_plan.gpu_memory_types
        )
