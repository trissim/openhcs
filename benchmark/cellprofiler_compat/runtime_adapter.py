"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_compat.measurement_lookup import (
    measurement_row_mapping,
    measurement_row_object_name,
    measurement_rows,
)
from openhcs.constants.constants import Backend, FileFormat
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingRuntimeContext,
    SourceBindingOrigin,
)
from openhcs.core.source_matching import (
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    source_filters_match,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeValueStore,
    StoredRuntimeValue,
    replace_runtime_artifact_payload,
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
        request = self._source_resolution_request(
            alias,
            ArtifactKind.IMAGE,
            fallback_image,
        )
        return SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )

    def resolve_source_objects(
        self,
        alias: str,
        fallback_image: Any,
    ) -> ObjectLabelSet:
        request = self._source_resolution_request(
            alias,
            ArtifactKind.OBJECT_LABELS,
            fallback_image,
        )
        labels = SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )
        return ObjectLabelSet(
            name=alias,
            labels=labels,
            source_image_name=alias,
        )

    def _source_resolution_request(
        self,
        alias: str,
        kind: ArtifactKind,
        fallback_image: Any,
    ) -> "SourceBindingResolutionRequest":
        return SourceBindingResolutionRequest(
            alias=alias,
            binding=self._require_source_binding(alias, kind),
            adapter=self,
            fallback_image=fallback_image,
        )

    def require_resolvable_source_aliases(
        self,
        aliases: tuple[str, ...],
    ) -> None:
        for alias in aliases:
            self._require_source_binding(alias, ArtifactKind.IMAGE)

    def has_source_binding(
        self,
        alias: str,
        kind: ArtifactKind | None = None,
    ) -> bool:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        return binding is not None and (
            kind is None or binding.artifact_kind is kind
        )

    def _require_source_binding(
        self,
        alias: str,
        kind: ArtifactKind,
    ) -> NamedSourceBinding:
        binding = self.source_binding_plan.binding_for_alias(alias, self.group_key)
        if binding is None:
            raise RuntimeError(
                f"Missing compiled source binding for CellProfiler "
                f"{kind.value} alias '{alias}' on axis '{self.axis_id}' and "
                f"group {self.group_key!r}."
            )
        if binding.artifact_kind is not kind:
            raise RuntimeError(
                f"CellProfiler source binding '{alias}' is declared as "
                f"{binding.artifact_kind.value}, not {kind.value}."
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
        if object_name is not None and not self.has_source_binding(
            object_name,
            ArtifactKind.OBJECT_LABELS,
        ):
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
        return MeasurementTable.from_runtime_value(record.value)

    def measurement_tables_for_object(
        self,
        object_name: str,
        *,
        group_key: str | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return prior measurement tables whose subject is an object set."""
        records = self.runtime_value_store.find(
            kind=ArtifactKind.MEASUREMENTS,
            axis_id=self.axis_id,
            group_key=group_key,
            match_group=group_key is not None,
        )
        return tuple(
            table
            for record in records
            if _measurement_table_matches_object(
                table := MeasurementTable.from_runtime_value(record.value),
                object_name,
            )
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
        return self.runtime_value_store.replace(
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
        replace_runtime_artifact_payload(
            self.filemanager,
            data,
            RuntimeArtifactLocation(path=path, backend=self.backend),
        )


def _measurement_table_matches_object(
    table: MeasurementTable,
    object_name: str,
) -> bool:
    if table.object_name == object_name:
        return True
    if table.object_name is not None:
        return False
    return any(
        measurement_row_object_name(measurement_row_mapping(row)) == object_name
        for row in measurement_rows((table,))
    )


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
class SourceBindingMatchPlanRequest:
    """Typed request for deriving target metadata from an image-set match plan."""

    alias: str
    plan: SourceBindingMatchPlan
    step_input_candidates: tuple["ParsedSourceCandidate", ...]
    target_candidates: tuple["ParsedSourceCandidate", ...]
    full_pipeline_candidates: tuple["ParsedSourceCandidate", ...]
    source_binding_plan: CompiledSourceBindingPlan
    group_key: str | None


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
            return _natural_step_input_payload(request.fallback_image)
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
        step_input_candidates = _parse_source_candidates(
            request.adapter.source_binding_context.step_input_files,
            request.adapter,
        )
        inherit_components = _pipeline_start_inherited_components(
            request.adapter.source_binding_plan,
            step_input_candidates,
        )
        parsed_candidates = _parse_source_candidates(
            pipeline_input_files,
            request.adapter,
        )
        initially_matched = _match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components=inherit_components,
        )
        matched = _match_image_set_candidates(
            request.alias,
            request.adapter.source_binding_plan.match_plan,
            step_input_candidates,
            initially_matched,
            parsed_candidates,
            source_binding_plan=request.adapter.source_binding_plan,
            group_key=request.adapter.group_key,
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


@dataclass(frozen=True, slots=True)
class PipelineStartSourceLoadRequest:
    """Typed request for loading pipeline-start source payloads."""

    adapter: CellProfilerRuntimeAdapter
    selected_paths: tuple[str, ...]
    backend: str


class PipelineStartSourceFileLoader(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for loading selected pipeline-start source files."""

    __registry_key__ = "loader_key"
    __skip_if_no_key__ = True
    loader_key: ClassVar[str | None] = None

    @classmethod
    def for_paths(
        cls,
        selected_paths: tuple[str, ...],
    ) -> "PipelineStartSourceFileLoader":
        matching_loaders = tuple(
            loader
            for loader in (
                loader_type() for loader_type in cls.__registry__.values()
            )
            if loader.accepts_all(selected_paths)
        )
        if len(matching_loaders) == 1:
            return matching_loaders[0]
        suffixes = sorted({Path(path).suffix.lower() for path in selected_paths})
        if not matching_loaders:
            raise RuntimeError(
                "Pipeline-start source resolution has no registered loader for "
                f"selected source suffixes {suffixes!r}."
            )
        raise RuntimeError(
            "Pipeline-start source resolution has ambiguous registered loaders for "
            f"selected source suffixes {suffixes!r}."
        )

    def accepts_all(self, selected_paths: tuple[str, ...]) -> bool:
        return bool(selected_paths) and all(
            self.accepts_path(path) for path in selected_paths
        )

    @abstractmethod
    def accepts_path(self, path: str) -> bool:
        """Return whether this loader owns one source file path."""

    @abstractmethod
    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        """Load selected source files as stackable image-like payloads."""


class OpenHCSImageSourceFileLoader(PipelineStartSourceFileLoader):
    """Load normal image sources through the OpenHCS VFS filemanager."""

    loader_key = "openhcs_image"

    def accepts_path(self, path: str) -> bool:
        return is_image_path(path)

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        context = _require_processing_context(request.adapter)
        load_kwargs: dict[str, Any] = {}
        if request.backend == Backend.ZARR.value:
            load_kwargs["zarr_config"] = context.global_config.zarr_config
        loaded_images = context.filemanager.load_batch(
            list(request.selected_paths),
            request.backend,
            **load_kwargs,
        )
        return list(loaded_images)


class MatlabMatrixSourceFileLoader(PipelineStartSourceFileLoader):
    """Load CellProfiler MATLAB matrix image sources such as illumination files."""

    loader_key = "matlab_matrix"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".mat"

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        return [self._load_matrix(path) for path in request.selected_paths]

    def _load_matrix(self, path: str) -> Any:
        from scipy.io import loadmat

        payloads = _matlab_numeric_arrays(loadmat(path))
        if not payloads:
            raise RuntimeError(
                f"MATLAB source file {path!r} contains no numeric image arrays."
            )
        if len(payloads) == 1:
            return payloads[0][1]
        image_payloads = tuple(
            payload for name, payload in payloads if name.strip().lower() == "image"
        )
        if len(image_payloads) == 1:
            return image_payloads[0]
        names = tuple(name for name, _payload in payloads)
        raise RuntimeError(
            f"MATLAB source file {path!r} contains multiple numeric arrays "
            f"{names!r}; expected exactly one payload or one 'Image' payload."
        )


class NumpyArraySourceFileLoader(PipelineStartSourceFileLoader):
    """Load NumPy array image sources such as saved illumination functions."""

    loader_key = "numpy_array"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() in FileFormat.NUMPY.value

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[Any]:
        return [self._load_array(path) for path in request.selected_paths]

    def _load_array(self, path: str) -> Any:
        import numpy as np

        payload = np.load(path)
        if not _is_numeric_array_payload(payload):
            raise RuntimeError(
                f"NumPy source file {path!r} does not contain a numeric image array."
            )
        return payload


def _binding_requires_selector(binding: NamedSourceBinding) -> bool:
    selector = binding.selector
    return bool(
        selector.components
        or selector.metadata
        or selector.filters
        or not selector.inherit_current_scope
    )


@dataclass(frozen=True, slots=True)
class ParsedSourceCandidate:
    """One parsed file candidate used for source-binding selector resolution."""

    path: str
    resolved_path: str
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
        metadata = _candidate_metadata(
            file_path,
            resolved_path,
            adapter,
            parser,
        )
        candidates.append(
            ParsedSourceCandidate(
                path=str(file_path),
                resolved_path=str(resolved_path),
                filename=Path(resolved_path).name,
                metadata=MappingProxyType(dict(metadata)),
            )
        )
    return tuple(candidates)


def _candidate_metadata(
    file_path: str,
    resolved_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    context = adapter.source_binding_context
    _merge_candidate_path_metadata(
        metadata,
        resolved_path,
        adapter,
        parser,
        strict=True,
    )
    if Path(file_path) != Path(resolved_path):
        _merge_candidate_path_metadata(
            metadata,
            file_path,
            adapter,
            parser,
            strict=_step_input_source_path(file_path, context) is None,
        )
    virtual_path = _virtual_workspace_path_for_source(resolved_path, context)
    if virtual_path is not None and virtual_path not in {file_path, resolved_path}:
        _merge_candidate_path_metadata(
            metadata,
            virtual_path,
            adapter,
            parser,
            strict=False,
        )
    return metadata


def _merge_candidate_path_metadata(
    metadata: dict[str, Any],
    metadata_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: Any,
    *,
    strict: bool,
) -> None:
    parsed_metadata = parser.parse_filename(Path(metadata_path).name) or {}
    extracted_metadata = metadata_from_rules(
        metadata_path,
        adapter.source_binding_plan.metadata_rules,
    )
    if strict:
        merge_source_metadata(metadata, parsed_metadata, path=metadata_path)
        merge_source_metadata(metadata, extracted_metadata, path=metadata_path)
        return
    _merge_missing_source_metadata(metadata, parsed_metadata)
    _merge_missing_source_metadata(metadata, extracted_metadata)


def _merge_missing_source_metadata(
    metadata: dict[str, Any],
    additions: Mapping[str, Any],
) -> None:
    for key, value in additions.items():
        metadata.setdefault(key, str(value))


def _virtual_workspace_path_for_source(
    resolved_path: str,
    context: SourceBindingRuntimeContext,
) -> str | None:
    for virtual_path, source_path in context.step_input_source_paths.items():
        if Path(source_path) == Path(resolved_path):
            return virtual_path
    return None


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
        and source_filters_match(candidate.resolved_path, binding.selector.filters)
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


def _candidate_matches_image_set_metadata(
    candidate: ParsedSourceCandidate,
    image_set_metadata: Mapping[str, str],
) -> bool:
    return all(
        candidate.metadata.get(field_name) is not None
        and str(candidate.metadata[field_name]) == value
        for field_name, value in image_set_metadata.items()
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
        return _natural_step_input_payload(fallback_image)
    slices = _unstack_payload(fallback_image)
    selected_slices = [slices[index] for index in selected_indexes]
    return _restack_like_payload(selected_slices, fallback_image)


def _natural_step_input_payload(fallback_image: Any) -> Any:
    if not hasattr(fallback_image, "ndim"):
        return fallback_image
    if fallback_image.ndim == 2:
        return fallback_image
    return _restack_like_payload(_unstack_payload(fallback_image), fallback_image)


def _load_pipeline_start_stack(
    *,
    adapter: CellProfilerRuntimeAdapter,
    selected_paths: tuple[str, ...],
    fallback_image: Any,
) -> Any:
    if not selected_paths:
        raise RuntimeError("Pipeline-start source selection cannot load zero paths.")
    backend = adapter.source_binding_context.pipeline_input_backend
    if backend is None:
        raise RuntimeError(
            "Pipeline-start source resolution requires pipeline_input_backend."
        )
    loaded_payloads = PipelineStartSourceFileLoader.for_paths(
        selected_paths,
    ).load_slices(
        PipelineStartSourceLoadRequest(
            adapter=adapter,
            selected_paths=selected_paths,
            backend=backend,
        )
    )
    if not loaded_payloads:
        raise RuntimeError(
            "Pipeline-start source resolution loaded no payloads from "
            f"{list(selected_paths)}."
        )
    return _restack_like_payload(loaded_payloads, fallback_image)


def _matlab_numeric_arrays(
    mat_payload: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, payload)
        for name, payload in mat_payload.items()
        if not name.startswith("__") and _is_numeric_array_payload(payload)
    )


def _is_numeric_array_payload(payload: Any) -> bool:
    dtype = getattr(payload, "dtype", None)
    return (
        hasattr(payload, "ndim")
        and dtype is not None
        and dtype.kind in {"b", "u", "i", "f", "c"}
        and payload.ndim >= 2
    )


def _unstack_payload(payload: Any) -> list[Any]:
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return [payload]
    if is_color_image_slice(payload):
        return [payload]
    memory_type = detect_memory_type(payload)
    return ImageStackLayout.for_stack(payload).unstack(
        array=payload,
        memory_type=memory_type,
        gpu_id=0,
    )


def _restack_like_payload(
    slices: list[Any],
    reference_payload: Any,
) -> Any:
    if not slices:
        raise ValueError("Cannot restack an empty slice list.")
    if len(slices) == 1:
        return slices[0]
    memory_type = detect_memory_type(reference_payload)
    return ImageStackLayout.for_slices(slices).stack(
        slices=slices,
        memory_type=memory_type,
        gpu_id=0,
    )


def _inherited_scope_components(
    candidates: tuple[ParsedSourceCandidate, ...],
) -> Mapping[str, str]:
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


def _pipeline_start_inherited_components(
    source_binding_plan: CompiledSourceBindingPlan,
    step_input_candidates: tuple[ParsedSourceCandidate, ...],
) -> Mapping[str, str]:
    if source_binding_plan.match_plan is not None:
        return MappingProxyType({})
    return _inherited_scope_components(step_input_candidates)


class SourceBindingMatchPlanResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for restricting target candidates to the current image set."""

    __registry_key__ = "method_key"
    __skip_if_no_key__ = True
    method: ClassVar[SourceBindingMatchMethod | None] = None
    method_key: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls,
        method: SourceBindingMatchMethod,
    ) -> "SourceBindingMatchPlanResolver":
        return cls.__registry__[method.value]()

    @abstractmethod
    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        """Return target candidates belonging to the current image set."""


class MetadataSourceBindingMatchPlanResolver(SourceBindingMatchPlanResolver):
    method = SourceBindingMatchMethod.METADATA
    method_key = SourceBindingMatchMethod.METADATA.value

    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        constraints: dict[str, str] = {}
        for dimension in request.plan.dimensions:
            target_field = dimension.field_for_alias(request.alias)
            if target_field is None:
                continue
            match_value = _dimension_match_value(
                dimension=dimension,
                target_alias=request.alias,
                step_input_candidates=request.step_input_candidates,
            )
            if match_value is None:
                continue
            existing = constraints.get(target_field)
            if existing is not None and existing != match_value:
                raise RuntimeError(
                    f"Conflicting image-set match values for alias {request.alias!r} "
                    f"field {target_field!r}: {existing!r} != {match_value!r}."
                )
            constraints[target_field] = match_value
        metadata_constraints = MappingProxyType(constraints)
        return tuple(
            candidate
            for candidate in request.target_candidates
            if _candidate_matches_image_set_metadata(candidate, metadata_constraints)
        )


class OrderSourceBindingMatchPlanResolver(SourceBindingMatchPlanResolver):
    method = SourceBindingMatchMethod.ORDER
    method_key = SourceBindingMatchMethod.ORDER.value

    def match_candidates(
        self,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        current_index = _order_match_index(request)
        if current_index is None:
            scoped_candidates = _target_candidates_in_current_scope(
                request.step_input_candidates,
                request.target_candidates,
            )
            return scoped_candidates or request.target_candidates
        ordered_target_candidates = _ordered_source_candidates(request.target_candidates)
        if current_index >= len(ordered_target_candidates):
            return ()
        return (ordered_target_candidates[current_index],)


def _match_image_set_candidates(
    alias: str,
    match_plan: SourceBindingMatchPlan | None,
    step_input_candidates: tuple[ParsedSourceCandidate, ...],
    target_candidates: tuple[ParsedSourceCandidate, ...],
    full_pipeline_candidates: tuple[ParsedSourceCandidate, ...],
    *,
    source_binding_plan: CompiledSourceBindingPlan,
    group_key: str | None,
) -> tuple[ParsedSourceCandidate, ...]:
    if match_plan is None or not step_input_candidates or not target_candidates:
        return target_candidates
    return SourceBindingMatchPlanResolver.for_method(
        match_plan.method
    ).match_candidates(
        SourceBindingMatchPlanRequest(
            alias=alias,
            plan=match_plan,
            step_input_candidates=step_input_candidates,
            target_candidates=target_candidates,
            full_pipeline_candidates=full_pipeline_candidates,
            source_binding_plan=source_binding_plan,
            group_key=group_key,
        )
    )


def _ordered_source_candidates(
    candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[ParsedSourceCandidate, ...]:
    return tuple(sorted(candidates, key=lambda candidate: candidate.resolved_path))


def _target_candidates_in_current_scope(
    step_input_candidates: tuple[ParsedSourceCandidate, ...],
    target_candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[ParsedSourceCandidate, ...]:
    current_scope = _inherited_scope_components(step_input_candidates)
    if not current_scope:
        return ()
    return tuple(
        candidate
        for candidate in target_candidates
        if _candidate_matches_inherited_scope(candidate, current_scope)
    )


def _order_match_index(
    request: SourceBindingMatchPlanRequest,
) -> int | None:
    indexes = {
        index
        for candidate in request.step_input_candidates
        for index in (_source_alias_order_index(candidate=candidate, request=request),)
        if index is not None
    }
    if not indexes:
        return None
    if len(indexes) != 1:
        raise RuntimeError(
            f"Order-based image-set matching for alias {request.alias!r} found "
            f"conflicting current image-set indexes: {sorted(indexes)}."
        )
    return next(iter(indexes))


def _source_alias_order_index(
    *,
    candidate: ParsedSourceCandidate,
    request: SourceBindingMatchPlanRequest,
) -> int | None:
    matched_indexes: set[int] = set()
    for binding in request.source_binding_plan.bindings_for_group(request.group_key):
        if binding.alias == request.alias:
            continue
        for index, ordered_candidate in enumerate(
            _ordered_binding_candidates(
                binding=binding,
                candidates=request.full_pipeline_candidates,
            )
        ):
            if ordered_candidate.resolved_path == candidate.resolved_path:
                matched_indexes.add(index)
                break
    if not matched_indexes:
        return None
    if len(matched_indexes) != 1:
        raise RuntimeError(
            f"Order-based image-set matching could not uniquely assign source file "
            f"{candidate.resolved_path!r} to one alias order index."
        )
    return next(iter(matched_indexes))


def _ordered_binding_candidates(
    *,
    binding: NamedSourceBinding,
    candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[ParsedSourceCandidate, ...]:
    return _ordered_source_candidates(
        _match_candidates(
            candidates=candidates,
            binding=binding,
            inherit_components={},
        )
    )


def _dimension_match_value(
    *,
    dimension: SourceBindingMatchDimension,
    target_alias: str,
    step_input_candidates: tuple[ParsedSourceCandidate, ...],
) -> str | None:
    candidate_values = {
        value
        for field in dimension.fields
        if field.alias != target_alias
        for value in _shared_candidate_values(
            field,
            step_input_candidates,
        )
    }
    if not candidate_values:
        return None
    if len(candidate_values) > 1:
        raise RuntimeError(
            "Current step input candidates produce conflicting image-set match "
            f"values for alias {target_alias!r}: {sorted(candidate_values)!r}."
        )
    return next(iter(candidate_values))


def _shared_candidate_values(
    field: SourceBindingMatchField,
    step_input_candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[str, ...]:
    values = tuple(
        str(candidate.metadata[field.metadata_field])
        for candidate in step_input_candidates
        if candidate.metadata.get(field.metadata_field) is not None
    )
    if not values:
        return ()
    shared_values = set(values)
    if len(shared_values) != 1:
        raise RuntimeError(
            "Current step input candidates do not share a single image-set match "
            f"value for metadata field {field.metadata_field!r}: {sorted(shared_values)!r}."
        )
    return (values[0],)


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
    source_path = _step_input_source_path(file_path, adapter.source_binding_context)
    if source_path is not None:
        return source_path
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    step_input_dir = adapter.source_binding_context.step_input_dir
    if step_input_dir is None:
        return str(path)
    return str(Path(step_input_dir) / path)


def _step_input_source_path(
    file_path: str,
    context: SourceBindingRuntimeContext,
) -> str | None:
    for key in _source_path_lookup_keys(file_path, context.step_input_dir):
        source_path = context.step_input_source_paths.get(key)
        if source_path is not None:
            return source_path
    return None


def _source_path_lookup_keys(
    file_path: str,
    step_input_dir: str | None,
) -> tuple[str, ...]:
    path = Path(file_path)
    keys = dict.fromkeys((str(file_path), path.as_posix()))
    if path.is_absolute() and step_input_dir is not None:
        try:
            relative_path = path.relative_to(step_input_dir)
        except ValueError:
            pass
        else:
            keys[relative_path.as_posix()] = None
    return tuple(keys)
