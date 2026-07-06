"""Generic returned-output matching for runtime artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)


def artifact_spec_participates_in_main_flow(spec: ArtifactSpec) -> bool:
    """Return whether a returned artifact spec is the main runtime-flow value."""
    return (
        spec.artifact_type.participates_in_main_flow_output
        and spec.sidecar_role is None
    )


@dataclass(frozen=True, slots=True)
class RuntimeReturnedOutputMatcher:
    """Resolve retained output specs against generic callable return values."""

    retained_specs: tuple[ArtifactSpec, ...]
    declared_specs: tuple[ArtifactSpec, ...]
    main_output: Any
    artifact_values: tuple[Any, ...]
    returned_specs: tuple[ArtifactSpec, ...] = ()

    def resolve(self) -> dict[str, Any] | None:
        if not self.retained_specs:
            return {}
        if (
            len(self.retained_specs) == 1
            and self.retained_specs[0].artifact_type is ImageArtifactType
            and self.retained_specs[0].sidecar_role is None
        ):
            return {self.retained_specs[0].name: self.main_output}
        declared_resolution = self.resolve_from_declared_outputs()
        if declared_resolution is not None:
            return declared_resolution

        if len(self.retained_specs) == 1:
            return {self.retained_specs[0].name: self.single_output_value(self.retained_specs[0])}

        positional_resolution = self.resolve_positional_outputs()
        if positional_resolution is not None:
            return positional_resolution

        return self.resolve_from_returned_specs()

    def single_output_value(self, spec: ArtifactSpec) -> Any:
        if spec.artifact_type is ImageArtifactType:
            return self.main_output
        if not self.artifact_values:
            raise ValueError(
                f"Runtime callable did not return a value for output '{spec.name}'."
            )
        if spec.artifact_type is ObjectLabelsArtifactType:
            return self.artifact_values[-1]
        if spec.artifact_type is MeasurementsArtifactType:
            return self.artifact_values[-1]
        return self.artifact_values[0]

    def resolve_from_declared_outputs(self) -> dict[str, Any] | None:
        if not self.declared_specs:
            return None
        return self.resolve_from_candidates(
            self.declared_return_candidates(),
            require_exact_names=True,
        )

    def declared_return_candidates(self) -> tuple[tuple[ArtifactSpec, Any], ...]:
        main_index = self.declared_main_output_index()
        if main_index is None:
            if len(self.declared_specs) < len(self.artifact_values):
                return ()
            return tuple(
                zip(
                    self.declared_specs[: len(self.artifact_values)],
                    self.artifact_values,
                    strict=True,
                )
            )
        artifact_specs = (
            *self.declared_specs[:main_index],
            *self.declared_specs[main_index + 1 :],
        )
        artifact_candidates = tuple(
            zip(artifact_specs, self.artifact_values, strict=True)
        )
        return (
            *artifact_candidates[:main_index],
            (self.declared_specs[main_index], self.main_output),
            *artifact_candidates[main_index:],
        )

    def declared_main_output_index(self) -> int | None:
        if len(self.declared_specs) != len(self.artifact_values) + 1:
            return None
        return self.first_declared_main_output_index()

    def first_declared_main_output_index(self) -> int | None:
        for index, spec in enumerate(self.declared_specs):
            if artifact_spec_participates_in_main_flow(spec):
                return index
        return None

    def resolve_positional_outputs(self) -> dict[str, Any] | None:
        if (
            self.retained_specs[0].artifact_type is ImageArtifactType
            and len(self.retained_specs) == len(self.artifact_values) + 1
        ):
            return {
                self.retained_specs[0].name: self.main_output,
                **{
                    spec.name: value
                    for spec, value in zip(
                        self.retained_specs[1:],
                        self.artifact_values,
                        strict=True,
                    )
                },
            }
        if len(self.retained_specs) != len(self.artifact_values):
            return None
        return {
            spec.name: value
            for spec, value in zip(self.retained_specs, self.artifact_values, strict=True)
        }

    def resolve_from_returned_specs(self) -> dict[str, Any] | None:
        if not self.returned_specs:
            return None
        candidate_specs = self.returned_specs_with_retained_tail(
            self.returned_specs,
            len(self.artifact_values),
        )
        return self.resolve_from_candidates(
            (
                (ArtifactSpec.output("<main>", ImageArtifactType), self.main_output),
                *zip(candidate_specs, self.artifact_values, strict=False),
            ),
            require_exact_names=False,
        )

    def resolve_from_candidates(
        self,
        candidates: tuple[tuple[ArtifactSpec, Any], ...],
        *,
        require_exact_names: bool,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        resolved: dict[str, Any] = {}
        cursor = 0
        for spec in self.retained_specs:
            match_index = self.next_candidate_index(
                candidates,
                spec,
                cursor,
                require_exact_names=require_exact_names,
            )
            if match_index is None:
                return None
            resolved[spec.name] = candidates[match_index][1]
            cursor = match_index + 1
        return resolved

    def next_candidate_index(
        self,
        candidates: tuple[tuple[ArtifactSpec, Any], ...],
        retained_spec: ArtifactSpec,
        cursor: int,
        *,
        require_exact_names: bool,
    ) -> int | None:
        for index in range(cursor, len(candidates)):
            if self.same_artifact_identity(candidates[index][0], retained_spec):
                return index
        if require_exact_names:
            return None
        for index in range(cursor, len(candidates)):
            if self.same_artifact_semantics(candidates[index][0], retained_spec):
                return index
        return None

    @staticmethod
    def same_artifact_identity(left: ArtifactSpec, right: ArtifactSpec) -> bool:
        return (
            left.name == right.name
            and left.artifact_type is right.artifact_type
            and left.sidecar_role is right.sidecar_role
        )

    @staticmethod
    def same_artifact_semantics(left: ArtifactSpec, right: ArtifactSpec) -> bool:
        return left.artifact_type is right.artifact_type and left.sidecar_role is right.sidecar_role

    def returned_specs_with_retained_tail(
        self,
        candidate_specs: tuple[ArtifactSpec, ...],
        artifact_value_count: int,
    ) -> tuple[ArtifactSpec, ...]:
        if len(candidate_specs) >= artifact_value_count:
            return candidate_specs
        remaining_counts: dict[tuple[type[ArtifactType], Any], int] = {}
        for spec in self.retained_specs:
            key = (spec.artifact_type, spec.sidecar_role)
            remaining_counts[key] = remaining_counts.get(key, 0) + 1
        for spec in candidate_specs:
            key = (spec.artifact_type, spec.sidecar_role)
            if key not in remaining_counts:
                continue
            remaining_counts[key] -= 1
            if remaining_counts[key] <= 0:
                remaining_counts.pop(key)
        tail: list[ArtifactSpec] = []
        for spec in self.retained_specs:
            key = (spec.artifact_type, spec.sidecar_role)
            count = remaining_counts.get(key, 0)
            if count <= 0:
                continue
            tail.append(spec)
            remaining_counts[key] = count - 1
        return (*candidate_specs, *tail[: artifact_value_count - len(candidate_specs)])
