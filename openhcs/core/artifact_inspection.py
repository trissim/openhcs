"""Typed compiled-artifact inspection transport for UI consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
)

if TYPE_CHECKING:
    from openhcs.core.compiled_execution import CompiledExecutionBundle
    from openhcs.core.compiled_step_plan import CompiledStepPlan
    from openhcs.core.function_patterns import CompiledFunctionInvocation


class ArtifactInspectionControlMessageType(str, Enum):
    """OpenHCS control message types owned by compiled artifact inspection."""

    READ_COMPILED = "openhcs_artifact_read_compiled"


@dataclass(frozen=True, slots=True)
class CompiledArtifactInvocationInspection:
    """Exact artifact contracts and plans for one compiled invocation."""

    key: FunctionInvocationKey
    function_name: str
    input_edges: tuple[InvocationArtifactInputEdgePlan, ...]
    output_specs: tuple[ArtifactSpec, ...]
    output_plans: tuple[ArtifactOutputPlan, ...]

    @classmethod
    def from_invocation(
        cls,
        invocation: "CompiledFunctionInvocation",
    ) -> "CompiledArtifactInvocationInspection":
        output_specs = tuple(invocation.contract.artifact_outputs)
        output_plans = tuple(invocation.artifact_output_plans)
        if tuple(spec.ref() for spec in output_specs) != tuple(
            plan.ref() for plan in output_plans
        ):
            raise ValueError(
                f"Compiled invocation {invocation.key!r} output contracts do not "
                "match their compiled output plans."
            )
        return cls(
            key=invocation.key,
            function_name=invocation.contract.function_name,
            input_edges=tuple(invocation.artifact_input_edges),
            output_specs=output_specs,
            output_plans=output_plans,
        )


@dataclass(frozen=True, slots=True)
class CompiledArtifactStepInspection:
    """Compiler-owned artifact state for one context and pipeline step."""

    context_id: str
    axis_id: str
    step_index: int
    step_name: str
    artifact_inputs: tuple[ArtifactInputPlan, ...]
    artifact_outputs: tuple[ArtifactOutputPlan, ...]
    invocations: tuple[CompiledArtifactInvocationInspection, ...]

    @classmethod
    def from_step_plan(
        cls,
        *,
        context_id: str,
        plan: "CompiledStepPlan",
    ) -> "CompiledArtifactStepInspection":
        pattern = plan.compiled_function_pattern
        return cls(
            context_id=context_id,
            axis_id=plan.axis_id,
            step_index=plan.step_index,
            step_name=plan.step_name,
            artifact_inputs=tuple(plan.artifact_inputs.values()),
            artifact_outputs=tuple(plan.artifact_outputs.values()),
            invocations=(
                ()
                if pattern is None
                else tuple(
                    CompiledArtifactInvocationInspection.from_invocation(invocation)
                    for invocation in pattern.iter_invocations()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledArtifactInspection:
    """Pickle-safe projection of one retained compiled execution bundle."""

    compile_artifact_id: str
    plate_id: str
    steps: tuple[CompiledArtifactStepInspection, ...]

    @classmethod
    def from_execution_bundle(
        cls,
        *,
        compile_artifact_id: str,
        plate_id: str,
        bundle: "CompiledExecutionBundle",
    ) -> "CompiledArtifactInspection":
        if not compile_artifact_id:
            raise ValueError("compile_artifact_id cannot be empty.")
        if not plate_id:
            raise ValueError("plate_id cannot be empty.")
        return cls(
            compile_artifact_id=compile_artifact_id,
            plate_id=plate_id,
            steps=tuple(
                CompiledArtifactStepInspection.from_step_plan(
                    context_id=context_id,
                    plan=plan,
                )
                for context_id, context in sorted(bundle.runtime_contexts.items())
                for _step_index, plan in sorted(context.step_plans.items())
            ),
        )

    def steps_for_index(
        self,
        step_index: int,
    ) -> tuple[CompiledArtifactStepInspection, ...]:
        """Return every compiled context projection for one pipeline position."""

        return tuple(step for step in self.steps if step.step_index == step_index)


@dataclass(frozen=True, slots=True)
class CompiledArtifactInspectionRequest:
    """Control request for one retained compile artifact projection."""

    compile_artifact_id: str

    def __post_init__(self) -> None:
        if not self.compile_artifact_id:
            raise ValueError("compile_artifact_id cannot be empty.")


@dataclass(frozen=True, slots=True)
class CompiledArtifactInspectionControlPayload:
    """Wire payload for one compiled-artifact inspection request."""

    compile_artifact_id: str
    message_type: ArtifactInspectionControlMessageType = (
        ArtifactInspectionControlMessageType.READ_COMPILED
    )

    @classmethod
    def from_request(
        cls,
        request: CompiledArtifactInspectionRequest,
    ) -> "CompiledArtifactInspectionControlPayload":
        return cls(compile_artifact_id=request.compile_artifact_id)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompiledArtifactInspectionControlPayload":
        message_type = ArtifactInspectionControlMessageType(str(payload["type"]))
        if message_type is not ArtifactInspectionControlMessageType.READ_COMPILED:
            raise ValueError(
                "Unsupported compiled artifact inspection control type: "
                f"{message_type.value!r}."
            )
        return cls(
            compile_artifact_id=str(payload["compile_artifact_id"]),
            message_type=message_type,
        )

    def to_request(self) -> CompiledArtifactInspectionRequest:
        return CompiledArtifactInspectionRequest(
            compile_artifact_id=self.compile_artifact_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "compile_artifact_id": self.compile_artifact_id,
        }


@dataclass(frozen=True, slots=True)
class CompiledArtifactInspectionResponse:
    """Control response carrying the compiler-owned inspection projection."""

    inspection: CompiledArtifactInspection

    def to_control_response(self) -> dict[str, Any]:
        return {"status": "ok", "inspection": self.inspection}

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompiledArtifactInspectionResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        inspection = payload["inspection"]
        if not isinstance(inspection, CompiledArtifactInspection):
            raise TypeError(
                "Compiled artifact inspection response requires "
                f"CompiledArtifactInspection, got {type(inspection).__name__}."
            )
        return cls(inspection=inspection)
