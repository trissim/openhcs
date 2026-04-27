"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingOrigin,
)
from openhcs.core.runtime_stores import (
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_values import (
    FieldSpec,
    MeasurementTable,
    NamedImage,
    ObjectLabelSet,
    ObjectLabelRepresentation,
    RelationshipEndpoint,
    ObjectRelationship,
    normalize_artifact_value,
)


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeAdapter:
    """CellProfiler-like API backed by typed OpenHCS runtime state.

    The adapter deliberately has no object/image/measurement dictionaries of its
    own. Writes require compiled output plans and a filemanager so the
    RuntimeValueStore record and VFS payload stay aligned with the normal
    FunctionStep runtime boundary.
    """

    runtime_value_store: RuntimeValueStore
    axis_id: str
    artifact_outputs: Mapping[str, ArtifactOutputPlan] = field(default_factory=dict)
    source_binding_plan: CompiledSourceBindingPlan = field(
        default_factory=CompiledSourceBindingPlan.empty
    )
    group_key: str | None = None
    filemanager: Any | None = None
    backend: str = "memory"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_value_store, RuntimeValueStore):
            raise TypeError(
                "CellProfilerRuntimeAdapter.runtime_value_store must be "
                f"RuntimeValueStore, got {type(self.runtime_value_store).__name__}."
            )
        if not self.axis_id:
            raise ValueError("CellProfilerRuntimeAdapter.axis_id cannot be empty.")
        if not self.backend:
            raise ValueError("CellProfilerRuntimeAdapter.backend cannot be empty.")
        if not isinstance(self.source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_plan must be "
                "CompiledSourceBindingPlan, got "
                f"{type(self.source_binding_plan).__name__}."
            )

        outputs = dict(self.artifact_outputs)
        for name, plan in outputs.items():
            if not isinstance(plan, ArtifactOutputPlan):
                raise TypeError(
                    f"artifact_outputs['{name}'] must be ArtifactOutputPlan, "
                    f"got {type(plan).__name__}."
                )
            if name != plan.name:
                raise ValueError(
                    f"artifact_outputs key '{name}' does not match plan name "
                    f"'{plan.name}'."
                )
        object.__setattr__(self, "artifact_outputs", MappingProxyType(outputs))
        if self.group_key is not None:
            object.__setattr__(self, "group_key", str(self.group_key))

    def resolve_source_image(
        self,
        alias: str,
        fallback_image: Any,
    ) -> Any:
        binding = self._require_source_binding(alias)
        return SourceBindingResolver.for_origin(binding.origin).resolve_image(
            SourceBindingResolutionRequest(
                alias=alias,
                binding=binding,
                adapter=self,
                fallback_image=fallback_image,
            )
        )

    def require_resolvable_source_aliases(
        self,
        aliases: tuple[str, ...],
    ) -> None:
        if len(aliases) <= 1:
            return
        unresolved = tuple(
            alias
            for alias in aliases
            if _binding_requires_selector(self._require_source_binding(alias))
        )
        if unresolved:
            raise NotImplementedError(
                "Multiple external CellProfiler image bindings still require a "
                "typed selector-bearing NamesAndTypes/Images plan. Unresolved "
                f"aliases: {list(unresolved)}."
            )

    def has_source_binding(
        self,
        alias: str,
    ) -> bool:
        return self.source_binding_plan.binding_for_alias(alias, self.group_key) is not None

    def _require_source_binding(
        self,
        alias: str,
    ) -> NamedSourceBinding:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        if binding is None:
            raise RuntimeError(
                f"Missing compiled source binding for CellProfiler image alias "
                f"'{alias}' on axis '{self.axis_id}' and group {self.group_key!r}."
            )
        return binding

    def add_image(
        self,
        name: str,
        data: Any,
        *,
        dimensions: tuple[str, ...] = (),
        source_image_name: str | None = None,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            ArtifactKind.IMAGE,
            NamedImage(
                name=name,
                data=data,
                dimensions=dimensions,
                source_image_name=source_image_name,
            ),
        )

    def get_image(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> NamedImage:
        record = self._resolve_one(name, ArtifactKind.IMAGE, group_key=group_key)
        schema = record.value.schema
        return NamedImage(
            name=name,
            data=record.value.data,
            dimensions=schema.dimensions,
            source_image_name=schema.source_image_name,
        )

    def add_objects(
        self,
        name: str,
        labels: Any,
        *,
        source_image_name: str | None = None,
        dimensions: tuple[str, ...] = (),
        representation: ObjectLabelRepresentation = (
            ObjectLabelRepresentation.DENSE_LABELS
        ),
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            ArtifactKind.OBJECT_LABELS,
            ObjectLabelSet(
                name=name,
                labels=labels,
                source_image_name=source_image_name,
                dimensions=dimensions,
                representation=representation,
            ),
        )

    def get_objects(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> ObjectLabelSet:
        record = self._resolve_one(
            name,
            ArtifactKind.OBJECT_LABELS,
            group_key=group_key,
        )
        schema = record.value.schema
        return ObjectLabelSet(
            name=name,
            labels=record.value.data,
            source_image_name=schema.source_image_name,
            dimensions=schema.dimensions,
            representation=(
                schema.label_representation
                or ObjectLabelRepresentation.DENSE_LABELS
            ),
        )

    def add_measurements(
        self,
        name: str,
        rows: Any,
        *,
        object_name: str | None = None,
        fields: tuple[FieldSpec, ...] = (),
        object_id_field: str | None = None,
        source_image_name: str | None = None,
    ) -> StoredRuntimeValue:
        if object_name is not None:
            self._resolve_one(object_name, ArtifactKind.OBJECT_LABELS)
        return self._record_native_value(
            name,
            ArtifactKind.MEASUREMENTS,
            MeasurementTable(
                name=name,
                rows=rows,
                object_name=object_name,
                fields=fields,
                object_id_field=object_id_field,
                source_image_name=source_image_name,
            ),
        )

    def get_measurements(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> MeasurementTable:
        record = self._resolve_one(
            name,
            ArtifactKind.MEASUREMENTS,
            group_key=group_key,
        )
        schema = record.value.schema
        return MeasurementTable(
            name=name,
            rows=record.value.data,
            object_name=schema.object_name,
            fields=schema.fields,
            object_id_field=schema.object_id_field,
            source_image_name=schema.source_image_name,
        )

    def add_relationship(
        self,
        name: str,
        *,
        parent_object_name: str,
        child_object_name: str,
        parent_ids: Any,
        child_ids: Any,
    ) -> StoredRuntimeValue:
        self._resolve_one(parent_object_name, ArtifactKind.OBJECT_LABELS)
        self._resolve_one(child_object_name, ArtifactKind.OBJECT_LABELS)
        return self._record_native_value(
            name,
            ArtifactKind.RELATIONSHIPS,
            ObjectRelationship(
                name=name,
                source=RelationshipEndpoint(
                    parent_object_name,
                    role="parent",
                    id_field="parent_id",
                ),
                target=RelationshipEndpoint(
                    child_object_name,
                    role="child",
                    id_field="child_id",
                ),
                source_ids=parent_ids,
                target_ids=child_ids,
                relationship_type="parent_child",
            ),
        )

    def get_relationship(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> ObjectRelationship:
        record = self._resolve_one(
            name,
            ArtifactKind.RELATIONSHIPS,
            group_key=group_key,
        )
        data = record.value.data
        if not isinstance(data, Mapping):
            raise TypeError(
                f"Relationship '{name}' payload must be mapping-backed, "
                f"got {type(data).__name__}."
            )
        schema = record.value.schema
        relationship = schema.relationship
        if relationship is not None:
            return ObjectRelationship(
                name=name,
                source=relationship.source,
                target=relationship.target,
                source_ids=data[relationship.source.id_field],
                target_ids=data[relationship.target.id_field],
                relationship_type=relationship.relationship_type,
            )
        return ObjectRelationship(
            name=name,
            source=RelationshipEndpoint(
                data["source_object"],
                role="source",
                id_field="source_id",
            ),
            target=RelationshipEndpoint(
                data["target_object"],
                role="target",
                id_field="target_id",
            ),
            source_ids=data["source_id"],
            target_ids=data["target_id"],
        )

    def _record_native_value(
        self,
        name: str,
        expected_kind: ArtifactKind,
        native_value: Any,
    ) -> StoredRuntimeValue:
        plan = self._require_output_plan(name, expected_kind)
        runtime_value = normalize_artifact_value(
            plan,
            native_value,
            axis_id=self.axis_id,
        )
        self._save_payload(runtime_value.data, plan.path)
        return self.runtime_value_store.record(
            runtime_value,
            path=plan.path,
            backend=self.backend,
        )

    def _require_output_plan(
        self,
        name: str,
        expected_kind: ArtifactKind,
    ) -> ArtifactOutputPlan:
        plan = self.artifact_outputs.get(name)
        if plan is None:
            raise RuntimeError(
                f"No compiled output plan for CellProfiler artifact '{name}' "
                f"({expected_kind.value})."
            )
        if plan.kind is not expected_kind:
            raise ValueError(
                f"CellProfiler artifact '{name}' expected output kind "
                f"{expected_kind.value}, got compiled kind {plan.kind.value}."
            )
        return plan

    def _resolve_one(
        self,
        name: str,
        kind: ArtifactKind,
        *,
        group_key: str | None = None,
    ) -> StoredRuntimeValue:
        records = self.runtime_value_store.find(
            name=name,
            kind=kind,
            axis_id=self.axis_id,
            group_key=group_key,
            match_group=group_key is not None,
        )
        if not records:
            raise RuntimeError(
                f"Missing CellProfiler runtime artifact '{name}' "
                f"({kind.value}) on axis '{self.axis_id}'."
            )
        if len(records) > 1:
            raise RuntimeError(
                f"Ambiguous CellProfiler runtime artifact '{name}' "
                f"({kind.value}) on axis '{self.axis_id}': {records!r}."
            )
        return records[0]

    def _save_payload(self, data: Any, path: str) -> None:
        if self.filemanager is None:
            raise RuntimeError(
                "CellProfilerRuntimeAdapter.filemanager is required for writes; "
                "adapter writes must persist through the OpenHCS VFS boundary."
            )
        try:
            save = self.filemanager.save
            ensure_directory = self.filemanager.ensure_directory
        except AttributeError as exc:
            raise TypeError(
                "CellProfilerRuntimeAdapter.filemanager must provide "
                "save() and ensure_directory()."
            ) from exc
        ensure_directory(str(Path(path).parent), self.backend)
        save(data, path, self.backend)


@dataclass(frozen=True, slots=True)
class SourceBindingResolutionRequest:
    """Source-binding resolution inputs for one external image alias."""

    alias: str
    binding: NamedSourceBinding
    adapter: CellProfilerRuntimeAdapter
    fallback_image: Any


class SourceBindingResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving typed source bindings."""

    __registry_key__ = "origin"
    __skip_if_no_key__ = True
    origin: ClassVar[SourceBindingOrigin | None] = None

    @classmethod
    def for_origin(cls, origin: SourceBindingOrigin) -> "SourceBindingResolver":
        return cls.__registry__[origin]()

    @abstractmethod
    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        """Resolve one named source image binding."""


class StepInputSourceBindingResolver(SourceBindingResolver):
    """Resolve named images directly from the current FunctionStep input."""

    origin = SourceBindingOrigin.STEP_INPUT

    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        if _binding_requires_selector(request.binding):
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' declares selector "
                "constraints over step input, but selector-based channel/view "
                "resolution is not wired yet."
            )
        return request.fallback_image


class PipelineStartSourceBindingResolver(SourceBindingResolver):
    """Reject pipeline-start bindings until metadata-backed source planning lands."""

    origin = SourceBindingOrigin.PIPELINE_START

    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        raise NotImplementedError(
            f"CellProfiler source alias '{request.alias}' targets pipeline-start "
            "resolution, but metadata-backed NamesAndTypes/Images planning is not "
            "wired yet."
        )


def _binding_requires_selector(binding: NamedSourceBinding) -> bool:
    selector = binding.selector
    return bool(selector.components or selector.metadata or not selector.inherit_current_scope)
