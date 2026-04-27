"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.memory import detect_memory_type, stack_slices, unstack_slices
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataExtractionRule,
    MetadataSource,
    NamedSourceBinding,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceBindingRuntimeContext,
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
    source_binding_context: SourceBindingRuntimeContext = field(
        default_factory=SourceBindingRuntimeContext.empty
    )
    group_key: str | None = None
    processing_context: Any | None = None
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
        if not isinstance(self.source_binding_context, SourceBindingRuntimeContext):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_context must be "
                "SourceBindingRuntimeContext, got "
                f"{type(self.source_binding_context).__name__}."
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
        for alias in aliases:
            self._require_source_binding(alias)

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
class SourceBindingRequestBase(ABC):
    """Shared nominal fields for source-binding request records."""

    alias: str
    binding: NamedSourceBinding


@dataclass(frozen=True, slots=True)
class SourceBindingResolutionRequest(SourceBindingRequestBase):
    """Source-binding resolution inputs for one external image alias."""

    adapter: CellProfilerRuntimeAdapter
    fallback_image: Any


@dataclass(frozen=True, slots=True)
class SourceFilterMatchRequest:
    """Typed request for one source-filter match evaluation."""

    file_path: str
    clause: SourceFilterClause
    target: str


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
        if not _binding_requires_selector(request.binding):
            return request.fallback_image
        step_input_files = request.adapter.source_binding_context.step_input_files
        if not step_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs step-input "
                "selector resolution, but no step input file universe was "
                "provided to the runtime adapter."
            )
        parsed_candidates = _parse_source_candidates(
            step_input_files,
            request.adapter,
        )
        matched = _match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components={},
        )
        selected_files = _require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=matched,
                source_description="step input",
            )
        )
        return _select_step_input_stack(
            request=request,
            selected_paths=tuple(candidate.path for candidate in selected_files),
        )


class PipelineStartSourceBindingResolver(SourceBindingResolver):
    """Resolve named images from the original pipeline-start source universe."""

    origin = SourceBindingOrigin.PIPELINE_START

    def resolve_image(self, request: SourceBindingResolutionRequest) -> Any:
        pipeline_input_files = request.adapter.source_binding_context.pipeline_input_files
        if not pipeline_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs pipeline-start "
                "selector resolution, but no pipeline-start file universe was "
                "provided to the runtime adapter."
            )
        inherit_components = _inherited_scope_components(
            request.adapter.source_binding_context.step_input_files,
            request.adapter,
        )
        parsed_candidates = _parse_source_candidates(
            pipeline_input_files,
            request.adapter,
        )
        matched = _match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components=inherit_components,
        )
        selected_files = _require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=matched,
                source_description="pipeline start",
            )
        )
        return _load_pipeline_start_stack(
            adapter=request.adapter,
            selected_paths=tuple(candidate.path for candidate in selected_files),
            fallback_image=request.fallback_image,
        )


def _binding_requires_selector(binding: NamedSourceBinding) -> bool:
    selector = binding.selector
    return bool(selector.components or selector.metadata or not selector.inherit_current_scope)


@dataclass(frozen=True, slots=True)
class ParsedSourceCandidate:
    """One parsed file candidate used for source-binding selector resolution."""

    path: str
    filename: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MatchedSourceCandidatesRequest(SourceBindingRequestBase):
    """Typed request for fail-loud source-candidate selection."""

    matched: tuple[ParsedSourceCandidate, ...]
    source_description: str

    @classmethod
    def from_resolution(
        cls,
        request: SourceBindingResolutionRequest,
        *,
        matched: tuple[ParsedSourceCandidate, ...],
        source_description: str,
    ) -> "MatchedSourceCandidatesRequest":
        return cls(
            alias=request.alias,
            binding=request.binding,
            matched=matched,
            source_description=source_description,
        )


def _parse_source_candidates(
    file_paths: tuple[str, ...],
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[ParsedSourceCandidate, ...]:
    parser = _require_processing_context(adapter).microscope_handler.parser
    candidates: list[ParsedSourceCandidate] = []
    for file_path in file_paths:
        resolved_path = _resolved_source_path(file_path, adapter)
        filename = Path(resolved_path).name
        metadata = dict(parser.parse_filename(filename) or {})
        extracted_metadata = _metadata_from_rules(
            resolved_path,
            adapter.source_binding_plan.metadata_rules,
        )
        _merge_metadata(
            metadata,
            extracted_metadata,
            path=resolved_path,
        )
        if not metadata:
            continue
        candidates.append(
            ParsedSourceCandidate(
                path=str(file_path),
                filename=filename,
                metadata=MappingProxyType(dict(metadata)),
            )
        )
    return tuple(candidates)


def _match_candidates(
    *,
    candidates: tuple[ParsedSourceCandidate, ...],
    binding: NamedSourceBinding,
    inherit_components: Mapping[str, str],
) -> tuple[ParsedSourceCandidate, ...]:
    metadata_fields = {selector.field for selector in binding.selector.metadata}
    if metadata_fields:
        unsupported = tuple(
            field
            for field in sorted(metadata_fields)
            if not any(field in candidate.metadata for candidate in candidates)
        )
        if unsupported:
            raise NotImplementedError(
                "Source-binding metadata selectors are only supported when the "
                "native OpenHCS filename parser exposes those fields. Missing "
                f"fields: {list(unsupported)}."
            )

    component_selectors = {
        selector.component.value: selector.value
        for selector in binding.selector.components
    }
    effective_components = (
        {
            **inherit_components,
            **component_selectors,
        }
        if binding.selector.inherit_current_scope
        else component_selectors
    )

    return tuple(
        candidate
        for candidate in candidates
        if _candidate_matches_explicit_components(candidate, component_selectors)
        and _candidate_matches_inherited_scope(candidate, effective_components)
        and _candidate_matches_metadata(candidate, binding.selector.metadata)
    )


def _candidate_matches_explicit_components(
    candidate: ParsedSourceCandidate,
    expected_components: Mapping[str, str],
) -> bool:
    return all(
        candidate.metadata.get(component_name) is not None
        and str(candidate.metadata[component_name]) == value
        for component_name, value in expected_components.items()
    )


def _candidate_matches_inherited_scope(
    candidate: ParsedSourceCandidate,
    inherited_scope: Mapping[str, str],
) -> bool:
    return all(
        candidate.metadata.get(field_name) is None
        or str(candidate.metadata[field_name]) == value
        for field_name, value in inherited_scope.items()
    )


def _candidate_matches_metadata(
    candidate: ParsedSourceCandidate,
    metadata_selectors: tuple[Any, ...],
) -> bool:
    return all(
        candidate.metadata.get(selector.field) is not None
        and str(candidate.metadata[selector.field]) == selector.value
        for selector in metadata_selectors
    )


def _require_matched_candidates(
    request: MatchedSourceCandidatesRequest,
) -> tuple[ParsedSourceCandidate, ...]:
    if request.matched:
        return request.matched
    raise RuntimeError(
        f"CellProfiler source alias '{request.alias}' with selector "
        f"{request.binding.selector!r} matched no files in the "
        f"{request.source_description} source universe."
    )


def _select_step_input_stack(
    *,
    request: SourceBindingResolutionRequest,
    selected_paths: tuple[str, ...],
) -> Any:
    step_input_files = request.adapter.source_binding_context.step_input_files
    indexed_paths = {path: index for index, path in enumerate(step_input_files)}
    selected_indexes = tuple(
        indexed_paths[path]
        for path in step_input_files
        if path in selected_paths
    )
    fallback_image = request.fallback_image
    if not selected_indexes:
        raise RuntimeError(
            f"CellProfiler source alias '{request.alias}' selected no step-input "
            "stack indexes after filename matching."
        )
    if len(step_input_files) == 1:
        return fallback_image
    slices = _unstack_payload(fallback_image)
    selected_slices = [slices[index] for index in selected_indexes]
    return _restack_like_payload(selected_slices, fallback_image)


def _load_pipeline_start_stack(
    *,
    adapter: CellProfilerRuntimeAdapter,
    selected_paths: tuple[str, ...],
    fallback_image: Any,
) -> Any:
    if not selected_paths:
        raise RuntimeError("Pipeline-start source selection cannot load zero paths.")
    context = _require_processing_context(adapter)
    backend = adapter.source_binding_context.pipeline_input_backend
    if backend is None:
        raise RuntimeError(
            "Pipeline-start source resolution requires pipeline_input_backend."
        )
    load_kwargs: dict[str, Any] = {}
    if backend == Backend.ZARR.value:
        load_kwargs["zarr_config"] = context.global_config.zarr_config
    loaded_images = context.filemanager.load_batch(list(selected_paths), backend, **load_kwargs)
    if not loaded_images:
        raise RuntimeError(
            f"Pipeline-start source resolution loaded no images from {list(selected_paths)}."
        )
    return _restack_like_payload(loaded_images, fallback_image)


def _unstack_payload(payload: Any) -> list[Any]:
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return [payload]
    memory_type = detect_memory_type(payload)
    return list(unstack_slices(payload, memory_type, 0, validate_slices=False))


def _restack_like_payload(
    slices: list[Any],
    reference_payload: Any,
) -> Any:
    if not slices:
        raise ValueError("Cannot restack an empty slice list.")
    if len(slices) == 1 and hasattr(reference_payload, "ndim") and reference_payload.ndim == 2:
        return slices[0]
    memory_type = detect_memory_type(reference_payload)
    return stack_slices(slices, memory_type=memory_type, gpu_id=0)


def _inherited_scope_components(
    step_input_files: tuple[str, ...],
    adapter: CellProfilerRuntimeAdapter,
) -> Mapping[str, str]:
    if not step_input_files:
        return {}
    candidates = _parse_source_candidates(step_input_files, adapter)
    if not candidates:
        return {}
    shared: dict[str, str] = {}
    first_metadata = candidates[0].metadata
    for field_name, value in first_metadata.items():
        if value is None:
            continue
        normalized_value = str(value)
        if all(
            candidate.metadata.get(field_name) is not None
            and str(candidate.metadata[field_name]) == normalized_value
            for candidate in candidates[1:]
        ):
            shared[field_name] = normalized_value
    return MappingProxyType(shared)


def _require_processing_context(adapter: CellProfilerRuntimeAdapter) -> Any:
    if adapter.processing_context is None:
        raise RuntimeError(
            "CellProfilerRuntimeAdapter.processing_context is required for "
            "selector-bearing source resolution."
        )
    return adapter.processing_context


def _resolved_source_path(
    file_path: str,
    adapter: CellProfilerRuntimeAdapter,
) -> str:
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    step_input_dir = adapter.source_binding_context.step_input_dir
    if step_input_dir is None:
        return str(path)
    return str(Path(step_input_dir) / path)


def _metadata_from_rules(
    file_path: str,
    metadata_rules: tuple[MetadataExtractionRule, ...],
) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for rule in metadata_rules:
        if not _rule_filters_match(file_path, rule.filters):
            continue
        target = _metadata_source_text(file_path, rule.source)
        match = re.search(rule.pattern, target)
        if match is None:
            continue
        _merge_metadata(
            extracted,
            {
                key: str(value)
                for key, value in match.groupdict().items()
                if value is not None
            },
            path=file_path,
        )
    return extracted


def _metadata_source_text(
    file_path: str,
    source: MetadataSource,
) -> str:
    path = Path(file_path)
    if source is MetadataSource.FOLDER_NAME:
        return str(path.parent)
    return path.name


def _rule_filters_match(
    file_path: str,
    filters: tuple[SourceFilterClause, ...],
) -> bool:
    return all(_filter_clause_matches(file_path, clause) for clause in filters)


def _filter_clause_matches(
    file_path: str,
    clause: SourceFilterClause,
) -> bool:
    target = SourceFilterTargetResolver.for_subject(clause.subject).resolve_text(file_path)
    return SourceFilterMatcher.for_match_type(clause.match_type).matches(
        SourceFilterMatchRequest(
            file_path=file_path,
            clause=clause,
            target=target,
        )
    )


class SourceFilterMatcher(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for typed source-filter match behavior."""

    __registry_key__ = "match_type"
    __skip_if_no_key__ = True
    match_type: ClassVar[SourceFilterMatchType | None] = None

    @classmethod
    def for_match_type(
        cls,
        match_type: SourceFilterMatchType,
    ) -> "SourceFilterMatcher":
        return cls.__registry__[match_type]()

    @abstractmethod
    def matches(self, request: SourceFilterMatchRequest) -> bool:
        """Return whether one file path satisfies the filter clause."""


class ContainsSourceFilterMatcher(SourceFilterMatcher):
    match_type = SourceFilterMatchType.CONTAINS

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return _require_filter_value(request.clause) in request.target


class DoesNotContainSourceFilterMatcher(SourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_CONTAIN

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return _require_filter_value(request.clause) not in request.target


class ContainsRegexSourceFilterMatcher(SourceFilterMatcher):
    match_type = SourceFilterMatchType.CONTAINS_REGEX

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return re.search(_require_filter_value(request.clause), request.target) is not None


class DoesNotContainRegexSourceFilterMatcher(SourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return re.search(_require_filter_value(request.clause), request.target) is None


class IsImageSourceFilterMatcher(SourceFilterMatcher):
    match_type = SourceFilterMatchType.IS_IMAGE

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return _is_image_path(request.file_path)


class SourceFilterTargetResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for source-filter target text resolution."""

    __registry_key__ = "subject"
    __skip_if_no_key__ = True
    subject: ClassVar[SourceFilterSubject | None] = None

    @classmethod
    def for_subject(
        cls,
        subject: SourceFilterSubject,
    ) -> "SourceFilterTargetResolver":
        return cls.__registry__[subject]()

    @abstractmethod
    def resolve_text(self, file_path: str) -> str:
        """Return the subject-specific text inspected by one filter clause."""


class FileSourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.FILE

    def resolve_text(self, file_path: str) -> str:
        return Path(file_path).name


class DirectorySourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.DIRECTORY

    def resolve_text(self, file_path: str) -> str:
        return str(Path(file_path).parent)


class ExtensionSourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.EXTENSION

    def resolve_text(self, file_path: str) -> str:
        return Path(file_path).suffix.lower()


def _is_image_path(file_path: str) -> bool:
    suffix = Path(file_path).suffix.lower()
    return suffix in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _require_filter_value(clause: SourceFilterClause) -> str:
    if clause.value is None:
        raise ValueError(
            "SourceFilterClause.value must be set unless match_type is IS_IMAGE."
        )
    return clause.value


def _merge_metadata(
    target: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    path: str,
) -> None:
    for key, value in additions.items():
        existing = target.get(key)
        normalized_value = str(value)
        if existing is not None and str(existing) != normalized_value:
            raise RuntimeError(
                f"Conflicting metadata field '{key}' while parsing source candidate "
                f"{path!r}: {existing!r} != {normalized_value!r}."
            )
        target[key] = normalized_value
