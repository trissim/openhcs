"""ArtifactKind materialization policy over existing writer infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema
from openhcs.processing.materialization import (
    MaterializationSpec,
    csv_only,
    json_only,
    segmentation_mask_rois,
)


class _NoArtifactMaterialization:
    """Explicit opt-out for artifact materialization policy resolution."""

    def __repr__(self) -> str:
        return "NO_ARTIFACT_MATERIALIZATION"

    def __reduce__(self):
        return (_no_artifact_materialization, ())


NO_ARTIFACT_MATERIALIZATION = _NoArtifactMaterialization()


def _no_artifact_materialization() -> _NoArtifactMaterialization:
    return NO_ARTIFACT_MATERIALIZATION


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationRule:
    """Default materialization rule for one semantic artifact kind."""

    kind: ArtifactKind
    spec_factory: Callable[[RuntimeValueSchema], MaterializationSpec]

    def __post_init__(self) -> None:
        if not callable(self.spec_factory):
            raise TypeError(
                f"ArtifactMaterializationRule for {self.kind.value} requires "
                "a callable spec_factory."
            )

    def build_spec(self, schema: RuntimeValueSchema) -> MaterializationSpec:
        """Build a concrete MaterializationSpec for a runtime value schema."""
        if schema.kind is not self.kind:
            raise ValueError(
                f"Materialization rule for {self.kind.value} cannot handle "
                f"schema kind {schema.kind.value}."
            )
        return self.spec_factory(schema)


def _csv_spec(schema: RuntimeValueSchema) -> MaterializationSpec:
    fields = [field.name for field in schema.fields] or None
    return csv_only(suffix=".csv", fields=fields)


def _json_spec(_schema: RuntimeValueSchema) -> MaterializationSpec:
    return json_only(suffix=".json")


def _object_label_spec(_schema: RuntimeValueSchema) -> MaterializationSpec:
    return segmentation_mask_rois()


DEFAULT_ARTIFACT_MATERIALIZATION_RULES: dict[
    ArtifactKind,
    ArtifactMaterializationRule,
] = {
    ArtifactKind.MEASUREMENTS: ArtifactMaterializationRule(
        ArtifactKind.MEASUREMENTS,
        _csv_spec,
    ),
    ArtifactKind.RELATIONSHIPS: ArtifactMaterializationRule(
        ArtifactKind.RELATIONSHIPS,
        _csv_spec,
    ),
    ArtifactKind.TABLE: ArtifactMaterializationRule(
        ArtifactKind.TABLE,
        _csv_spec,
    ),
    ArtifactKind.SPATIAL_GRID: ArtifactMaterializationRule(
        ArtifactKind.SPATIAL_GRID,
        _json_spec,
    ),
    ArtifactKind.OBJECT_LABELS: ArtifactMaterializationRule(
        ArtifactKind.OBJECT_LABELS,
        _object_label_spec,
    ),
    ArtifactKind.METADATA: ArtifactMaterializationRule(
        ArtifactKind.METADATA,
        _json_spec,
    ),
}


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

    if output_plan.kind is ArtifactKind.SPECIAL:
        return None

    rule = DEFAULT_ARTIFACT_MATERIALIZATION_RULES.get(output_plan.kind)
    if rule is None:
        raise ValueError(
            f"No default materialization registered for artifact "
            f"'{output_plan.name}' of kind {output_plan.kind.value}. "
            "Declare an explicit MaterializationSpec or add an ArtifactKind rule."
        )

    return rule.build_spec(runtime_value.schema)
