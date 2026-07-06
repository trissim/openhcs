"""Artifact materialization policy over artifact type declarations."""

from __future__ import annotations

from openhcs.core.artifacts import (
    ArtifactMaterializationPayload,
    ArtifactOutputPlan,
    ArtifactType,
    SpecialArtifactType,
)
from openhcs.core.python_source_literal import PythonSourceLiteral
from openhcs.core.runtime_values import (
    RuntimeValue,
)
from openhcs.processing.materialization import MaterializationSpec


class _NoArtifactMaterialization(ArtifactMaterializationPayload, PythonSourceLiteral):
    """Explicit opt-out for artifact materialization policy resolution."""

    def __repr__(self) -> str:
        return "NO_ARTIFACT_MATERIALIZATION"

    def __reduce__(self):
        return (_no_artifact_materialization, ())

    def source_literal(self) -> str:
        return "NO_ARTIFACT_MATERIALIZATION"

    def source_literal_imports(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            {
                (
                    "openhcs.core.artifact_materialization_policy",
                    "NO_ARTIFACT_MATERIALIZATION",
                )
            }
        )


NO_ARTIFACT_MATERIALIZATION = _NoArtifactMaterialization()


def _no_artifact_materialization() -> _NoArtifactMaterialization:
    return NO_ARTIFACT_MATERIALIZATION


def resolve_artifact_materialization_spec(
    output_plan: ArtifactOutputPlan,
    runtime_value: RuntimeValue,
) -> MaterializationSpec | None:
    """Resolve explicit or default materialization for one planned artifact.

    Existing explicit MaterializationSpec declarations remain authoritative.
    SPECIAL artifacts remain explicit-only for legacy side-channel compatibility.
    Semantic artifact kinds without defaults fail loudly.
    """
    if output_plan.materialization is NO_ARTIFACT_MATERIALIZATION:
        return None

    if output_plan.materialization is not None:
        return output_plan.materialization

    if output_plan.artifact_type is SpecialArtifactType:
        return None

    materialization_spec = output_plan.artifact_type.default_materialization_spec(
        runtime_value.schema
    )
    if materialization_spec is None:
        raise ValueError(
            f"No default materialization registered for artifact "
            f"'{output_plan.name}' of kind {output_plan.artifact_type.value}. "
            "Declare an explicit MaterializationSpec or add an ArtifactType hook."
        )

    return materialization_spec
