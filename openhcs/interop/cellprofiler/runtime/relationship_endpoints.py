"""Relationship endpoint contracts for CellProfiler runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.runtime_semantics import (
    parent_child_relationship_artifact_endpoints,
    parent_child_relationship_artifact_name,
)
from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )


@dataclass(frozen=True, slots=True)
class RelationshipEndpointContract:
    """Nominal parent/child endpoint contract for one relationship artifact."""

    parent: ArtifactSpec
    child: ArtifactSpec


RelationshipEndpointMatch = tuple[ArtifactSpec, ArtifactSpec]


@dataclass(frozen=True, slots=True)
class RelationshipEndpointMatches:
    """Cardinality authority for artifact-name endpoint matches."""

    matches: tuple[RelationshipEndpointMatch, ...]
    module_name: str
    relationship_name: str

    def contract_or_none(self) -> RelationshipEndpointContract | None:
        match self.matches:
            case ((parent, child),):
                return RelationshipEndpointContract(parent, child)
            case ():
                return None
            case _:
                raise ValueError(
                    f"{self.module_name} relationship output "
                    f"'{self.relationship_name}' matches multiple object endpoint pairs."
                )


@dataclass(frozen=True, slots=True)
class TwoInputRelationshipEndpointFallback:
    """Fallback endpoint contract when a module declares exactly two input objects."""

    object_inputs: tuple[ArtifactSpec, ...]
    object_outputs: tuple[ArtifactSpec, ...]

    def contract_or_none(self) -> RelationshipEndpointContract | None:
        match self.object_inputs, self.object_outputs:
            case ((parent, child), ()):
                return RelationshipEndpointContract(parent, child)
            case _:
                return None


@dataclass(frozen=True, slots=True)
class RelationshipEndpointResolver:
    """Resolve declared relationship artifacts to parent/child object endpoints."""

    request: CellProfilerOutputRecordRequest

    @classmethod
    def for_request(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> "RelationshipEndpointResolver":
        return cls(request)

    @property
    def object_inputs(self) -> tuple[ArtifactSpec, ...]:
        return self.request.contract.declared_input_collection().of_kind(
            ArtifactKind.OBJECT_LABELS
        )

    @property
    def object_outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.request.contract.output_collection().of_kind(
            ArtifactKind.OBJECT_LABELS
        )

    @property
    def relationship_outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.request.contract.declared_output_collection().of_kind(
            ArtifactKind.RELATIONSHIPS
        )

    def endpoint_specs(
        self,
        relationship_spec: ArtifactSpec,
    ) -> tuple[ArtifactSpec, ArtifactSpec]:
        contract = self.endpoint_contract(relationship_spec)
        return contract.parent, contract.child

    def endpoint_contract(
        self,
        relationship_spec: ArtifactSpec,
    ) -> RelationshipEndpointContract:
        """Return the endpoint contract for one relationship artifact."""
        matches = self.artifact_name_matches(relationship_spec)
        match_contract = RelationshipEndpointMatches(
            matches,
            self.request.module_name,
            relationship_spec.name,
        ).contract_or_none()
        if match_contract is not None:
            return match_contract
        fallback_contract = TwoInputRelationshipEndpointFallback(
            self.object_inputs,
            self.object_outputs,
        ).contract_or_none()
        if fallback_contract is not None:
            return fallback_contract
        endpoints = parent_child_relationship_artifact_endpoints(
            relationship_spec.name,
            parent_candidates=ArtifactSpecCollection(self.object_inputs).names(),
        )
        if endpoints is not None:
            parent_name, child_name = endpoints
            parent_spec = ArtifactSpecCollection(self.object_inputs).by_name(
                parent_name
            )
            if parent_spec is not None:
                child_spec = ArtifactSpecCollection(
                    (*self.object_outputs, *self.object_inputs),
                ).by_name(child_name)
                return RelationshipEndpointContract(
                    parent_spec,
                    child_spec
                    or ArtifactSpec(
                        child_name,
                        ArtifactKind.OBJECT_LABELS,
                    ),
                )
        module_contract = self.module_relationship_endpoint_contract(relationship_spec)
        if module_contract is not None:
            return module_contract
        raise NotImplementedError(
            f"{self.request.module_name} relationship output "
            f"'{relationship_spec.name}' cannot be mapped to object endpoints from "
            f"inputs={[spec.name for spec in self.object_inputs]} and "
            f"outputs={[spec.name for spec in self.object_outputs]}."
        )

    def artifact_name_matches(
        self,
        relationship_spec: ArtifactSpec,
    ) -> tuple[RelationshipEndpointMatch, ...]:
        candidate_children = (*self.object_inputs, *self.object_outputs)
        return tuple(
            (parent_spec, child_spec)
            for parent_spec in self.object_inputs
            for child_spec in candidate_children
            if parent_spec.name != child_spec.name
            and relationship_spec.name
            == parent_child_relationship_artifact_name(
                parent_spec.name,
                child_spec.name,
            )
        )

    def distance_measurements_apply(
        self,
        relationship_spec: ArtifactSpec,
    ) -> bool:
        """Return whether relationship-distance rows belong to this artifact."""
        module_type = CellProfilerModule.for_module(self.request.module_name)
        if module_type is None:
            return False
        return module_type.relationship_distance_measurements_apply(
            self,
            relationship_spec,
        )

    def module_relationship_endpoint_contract(
        self,
        relationship_spec: ArtifactSpec,
    ) -> RelationshipEndpointContract | None:
        module_type = CellProfilerModule.for_module(self.request.module_name)
        if module_type is None:
            return None
        return module_type.relationship_endpoint_contract(
            self,
            relationship_spec,
        )

    def indexed_object_input_contract(
        self,
        input_indices: tuple[int, int],
    ) -> RelationshipEndpointContract:
        parent_index, child_index = input_indices
        return RelationshipEndpointContract(
            self.object_input_at(parent_index),
            self.object_input_at(child_index),
        )

    def object_input_at(self, index: int) -> ArtifactSpec:
        try:
            return self.object_inputs[index]
        except IndexError as exc:
            raise NotImplementedError(
                f"{self.request.module_name} primary relationship endpoint "
                f"requires object input index {index}, got inputs="
                f"{[spec.name for spec in self.object_inputs]}."
            ) from exc

    def relationship_output_at(self, index: int) -> ArtifactSpec:
        try:
            return self.relationship_outputs[index]
        except IndexError as exc:
            raise NotImplementedError(
                f"{self.request.module_name} primary relationship endpoint "
                f"requires relationship output index {index}, got outputs="
                f"{[spec.name for spec in self.relationship_outputs]}."
            ) from exc
