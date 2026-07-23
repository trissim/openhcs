"""Generic runtime artifact identity and nominal value storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self

from openhcs.core.artifacts import ArtifactOutputPlan, ArtifactType
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)

if TYPE_CHECKING:
    from openhcs.core.runtime_image_values import ImagePayloadMetadata


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """Stable runtime identity for one artifact value."""

    name: str
    artifact_type: type[ArtifactType]
    scope: RuntimeExecutionAxisScope
    semantic_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArtifactKey.name cannot be empty.")
        if ArtifactType.coerce(self.artifact_type) is not self.artifact_type:
            raise TypeError(
                "ArtifactKey.artifact_type must be an ArtifactType class; wire values "
                "must be resolved before constructing the key."
            )
        if not isinstance(self.scope, RuntimeExecutionAxisScope):
            raise TypeError(
                "ArtifactKey.scope must be a RuntimeExecutionAxisScope, got "
                f"{type(self.scope).__name__}."
            )


@dataclass(frozen=True, slots=True)
class RuntimeValue:
    """A nominal artifact value validated against one compiled storage plan."""

    key: ArtifactKey
    data: Any
    materialization_source_metadata: "ImagePayloadMetadata | None" = None

    @classmethod
    def from_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
        data: Any,
        *,
        execution_scope: RuntimeExecutionAxisScope,
        materialization_source_metadata: "ImagePayloadMetadata | None" = None,
    ) -> Self:
        """Construct one exact runtime value from already-normalized data."""

        if not isinstance(execution_scope, RuntimeExecutionAxisScope):
            raise TypeError(
                "RuntimeValue execution_scope must be a RuntimeExecutionAxisScope."
            )
        group_key = output_plan.single_group_key
        artifact_scope = execution_scope.for_group_coordinate(
            output_plan.group_component if group_key is not None else None,
            group_key,
        )
        return cls(
            key=ArtifactKey(
                name=output_plan.name,
                artifact_type=output_plan.artifact_type,
                scope=artifact_scope,
                semantic_id=output_plan.artifact_type.runtime_semantic_id(data),
            ),
            data=data,
            materialization_source_metadata=materialization_source_metadata,
        )

    @classmethod
    def compose(
        cls,
        values: Sequence["RuntimeValue"],
        producer_group_scope: ComponentGroupScope | None = None,
    ) -> Any:
        """Compose producer records through their shared artifact-type owner."""

        runtime_values = tuple(values)
        if not runtime_values:
            raise ValueError("Cannot compose an empty runtime value group.")
        if producer_group_scope is not None and not isinstance(
            producer_group_scope, ComponentGroupScope
        ):
            raise TypeError(
                "RuntimeValue.compose producer_group_scope must be a "
                "ComponentGroupScope or None."
            )
        if producer_group_scope is not None:
            if producer_group_scope.is_ungrouped:
                raise ValueError(
                    "Runtime producer-group composition requires a grouped producer "
                    "scope."
                )
            for value in runtime_values:
                value_scope = value.key.scope
                if value_scope.component is not producer_group_scope.component:
                    raise ValueError(
                        "Runtime producer-group composition received a value from "
                        f"{value_scope.component!r}, expected "
                        f"{producer_group_scope.component!r}."
                    )
                if not producer_group_scope.contains_runtime_key(
                    value_scope.value_text
                ):
                    raise ValueError(
                        "Runtime producer-group composition received a value outside "
                        f"the declared scope {producer_group_scope!r}."
                    )
        artifact_type = runtime_values[0].artifact_type
        mismatched_types = tuple(
            value.artifact_type
            for value in runtime_values
            if value.artifact_type is not artifact_type
        )
        if mismatched_types:
            raise TypeError(
                "Grouped runtime values must have one artifact type, got "
                f"{(artifact_type, *mismatched_types)!r}."
            )
        return artifact_type.compose_runtime_values(
            runtime_values,
            producer_group_scope=producer_group_scope,
        )

    @classmethod
    def normalize(
        cls,
        output_plan: ArtifactOutputPlan,
        value: Any,
        *,
        axis_id: str,
        materialization_source_metadata: "ImagePayloadMetadata | None" = None,
    ) -> "RuntimeValue":
        """Normalize a raw artifact return exactly once through its compiled plan."""

        return cls.normalize_for_execution_scope(
            output_plan,
            value,
            execution_scope=RuntimeExecutionAxisScope(axis_id=str(axis_id)),
            materialization_source_metadata=materialization_source_metadata,
        )

    @classmethod
    def normalize_for_execution_scope(
        cls,
        output_plan: ArtifactOutputPlan,
        value: Any,
        *,
        execution_scope: RuntimeExecutionAxisScope,
        materialization_source_metadata: "ImagePayloadMetadata | None" = None,
    ) -> "RuntimeValue":
        """Normalize one artifact against its exact runtime execution identity."""

        if not isinstance(execution_scope, RuntimeExecutionAxisScope):
            raise TypeError(
                "RuntimeValue execution_scope must be a RuntimeExecutionAxisScope."
            )
        expected_scope = execution_scope.for_group_coordinate(
            output_plan.group_component if output_plan.single_group_key is not None else None,
            output_plan.single_group_key,
        )
        if isinstance(value, RuntimeValue):
            validated = value.validated_for_output_plan(
                output_plan,
                axis_id=execution_scope.axis_id,
            )
            if validated.key.scope != expected_scope:
                raise ValueError(
                    f"Artifact {output_plan.name!r} belongs to execution scope "
                    f"{validated.key.scope!r}, not {expected_scope!r}."
                )
            return validated

        normalized = output_plan.normalize_payload(
            value,
            axis_id=execution_scope.axis_id,
        )

        runtime_value = cls.from_output_plan(
            output_plan,
            normalized,
            execution_scope=execution_scope,
            materialization_source_metadata=materialization_source_metadata,
        )
        return runtime_value.validated_for_output_plan(
            output_plan,
            axis_id=execution_scope.axis_id,
        )

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def artifact_type(self) -> type[ArtifactType]:
        return self.key.artifact_type

    def materialization_payload(self) -> object:
        """Return the payload that materializers should receive for this value."""

        return self.artifact_type.materialization_payload(self)

    def validated_for_output_plan(
        self,
        output_plan: ArtifactOutputPlan,
        *,
        axis_id: str,
    ) -> "RuntimeValue":
        """Validate this value against one compiled output without renormalizing it."""

        if self.key.name != output_plan.name:
            raise ValueError(
                f"RuntimeValue name {self.key.name!r} does not match planned "
                f"artifact {output_plan.name!r}."
            )
        if self.artifact_type is not output_plan.artifact_type:
            raise ValueError(
                f"Artifact {output_plan.name!r} expected "
                f"{output_plan.artifact_type.value}, got {self.artifact_type.value}."
            )
        if self.key.scope.axis_id != axis_id:
            raise ValueError(
                f"Artifact {output_plan.name!r} belongs to axis "
                f"{self.key.scope.axis_id!r}, not {axis_id!r}."
            )
        self.artifact_type.validate_runtime_payload(output_plan.name, self.data)
        semantic_id = self.artifact_type.runtime_semantic_id(self.data)
        if semantic_id != self.key.semantic_id:
            return replace(self, key=replace(self.key, semantic_id=semantic_id))
        return self
