"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import cast, ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamProducer
from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.axis_filter import step_axis_allows_config
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_artifact_queries import MeasurementTableUnion
from openhcs.core.runtime_stores import (
    RuntimeArtifactAddress,
    RuntimeArtifactLocation,
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.source_image_provenance import (
    SourceImageIdentity,
    SourceImageProvenanceFields,
)
from openhcs.core.source_matching import (
    with_source_component_metadata,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentity,
    FunctionOutputIdentityAuthority,
    FunctionOutputParserContext,
    FunctionOutputPathAuthority,
)
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityAuthority,
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraAuthority,
    StreamSourceComponentMetadataItems,
)
from openhcs.core.streaming_config_factory import StreamingViewerSurface
from openhcs.processing.materialization.core import (
    BackendKwargs,
    MaterializationSpec,
    MaterializationValue,
    Output,
    RawBackendKwargs,
    ViewerStreamBackendCallKwargs,
    materialization_outputs,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.runtime_artifact_values import RuntimeValue

if TYPE_CHECKING:
    from polystore.filemanager import FileManager
    from openhcs.core.context.processing_context import ProcessingContext

class ArtifactMaterializationRecordReducer(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
):
    """Registered reducer from runtime store records to materialization records."""

    artifact_type: ClassVar[type[ArtifactType] | None] = ArtifactType

    def records_for_exact_scopes(
        self,
        *,
        records: tuple[StoredRuntimeValue, ...],
        output_plan: ArtifactOutputPlan,
        axis_id: str,
        group_key: str | None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Partition records by exact scope before artifact-specific reduction."""

        records_by_scope: dict[
            RuntimeExecutionAxisScope,
            list[StoredRuntimeValue],
        ] = {}
        for record in records:
            records_by_scope.setdefault(record.key.scope, []).append(record)
        return tuple(
            reduced_record
            for scoped_records in records_by_scope.values()
            for reduced_record in self.records_for_group(
                records=tuple(scoped_records),
                output_plan=output_plan,
                axis_id=axis_id,
                group_key=group_key,
            )
        )

    def records_for_group(
        self,
        *,
        records: tuple[StoredRuntimeValue, ...],
        output_plan: ArtifactOutputPlan,
        axis_id: str,
        group_key: str | None,
    ) -> tuple[StoredRuntimeValue, ...]:
        del output_plan, axis_id, group_key
        if len(records) <= 1:
            return records
        record_locations = tuple(
            (
                record.key.semantic_id,
                record.path,
            )
            for record in records
        )
        raise RuntimeError(
            f"Ambiguous RuntimeValueStore records for planned artifact "
            f"materialization '{records[0].key.name}' "
            f"({records[0].key.artifact_type.value}) on axis "
            f"'{records[0].key.scope.axis_id}' group "
            f"{records[0].key.scope.value_text!r}: {record_locations!r}."
        )


class MeasurementArtifactMaterializationRecordReducer(
    ArtifactMaterializationRecordReducer
):
    """Union same-artifact measurement subject records before materialization."""

    artifact_type = MeasurementsArtifactType

    def records_for_group(
        self,
        *,
        records: tuple[StoredRuntimeValue, ...],
        output_plan: ArtifactOutputPlan,
        axis_id: str,
        group_key: str | None,
    ) -> tuple[StoredRuntimeValue, ...]:
        if len(records) <= 1:
            return records
        group_plan = output_plan.for_group(group_key)
        tables = tuple(cast(MeasurementTable, record.value.data) for record in records)
        table = MeasurementTableUnion(output_plan.name, tables).as_artifact_table()
        value = RuntimeValue.normalize_for_execution_scope(
            group_plan,
            table,
            execution_scope=records[0].key.scope,
        )
        return (
            StoredRuntimeValue(
                value=value,
                location=RuntimeArtifactLocation(
                    path=group_plan.path,
                    backend=Backend.MEMORY.value,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationBackendPlan:
    """Persistent backends and streaming payload builders for artifact output."""

    persistent_backend_kwargs: BackendKwargs
    streaming_viewer_surfaces: Mapping[str, StreamingViewerSurface]

    def backends_for(
        self,
        *,
        materialization_spec: MaterializationSpec,
        filemanager: "FileManager",
        stream_output_paths: tuple[str, ...],
    ) -> list[str]:
        return [
            *(
                self.persistent_backend_kwargs
                if materialization_spec.participates_in_persistent_materialization()
                else ()
            ),
            *self.streamable_viewer_surfaces(
                filemanager=filemanager,
                stream_output_paths=stream_output_paths,
            ),
        ]

    def streamable_viewer_surfaces(
        self,
        *,
        filemanager: "FileManager",
        stream_output_paths: tuple[str, ...],
    ) -> Mapping[str, StreamingViewerSurface]:
        return {
            backend: viewer_surface
            for backend, viewer_surface in self.streaming_viewer_surfaces.items()
            if any(
                self.stream_backend_supports_output_path(
                    filemanager,
                    backend,
                    path,
                )
                for path in stream_output_paths
            )
        }

    @staticmethod
    def stream_backend_supports_output_path(
        filemanager: "FileManager",
        backend: str,
        path: str,
    ) -> bool:
        """Return whether one exact viewer backend accepts this output path."""
        return filemanager._get_backend(backend).supports_file_path(path)

    def supports_stream_output_path(
        self,
        filemanager: "FileManager",
        path: str,
    ) -> bool:
        """Return whether an enabled viewer backend accepts this exact output."""
        return any(
            self.stream_backend_supports_output_path(filemanager, backend, path)
            for backend in self.streaming_viewer_surfaces
        )

    def backend_kwargs(
        self,
        *,
        materialization_spec: MaterializationSpec,
        data: MaterializationValue,
        fallback_source_identity: SourceImageIdentity | None,
        producer_identity: StreamProducerIdentity,
        context: "ProcessingContext",
        filemanager: "FileManager",
        images_dir: str,
        stream_output_paths: tuple[str, ...],
    ) -> BackendKwargs:
        result: BackendKwargs = {}
        persistent_backend_kwargs = (
            self.persistent_backend_kwargs
            if materialization_spec.participates_in_persistent_materialization()
            else {}
        )
        for backend, kwargs in persistent_backend_kwargs.items():
            values = dict(kwargs)
            contextual_values = filemanager._get_backend(
                backend
            ).contextual_save_kwargs(images_dir=images_dir)
            conflicts = {
                name
                for name, value in contextual_values.items()
                if name in values and values[name] != value
            }
            if conflicts:
                raise ValueError(
                    "Persistent artifact backend kwargs conflict with backend-owned "
                    f"context values: {sorted(conflicts)!r}."
                )
            values.update(contextual_values)
            result[backend] = RawBackendKwargs(values)

        streamable_viewer_surfaces = self.streamable_viewer_surfaces(
            filemanager=filemanager,
            stream_output_paths=stream_output_paths,
        )
        if not streamable_viewer_surfaces:
            return result

        source_metadata_items = ArtifactStreamSourceMetadataAuthority.metadata_items(
            materialization_spec=materialization_spec,
            data=data,
            fallback_source_identity=fallback_source_identity,
        )
        producer = ViewerStreamProducer.from_identity(producer_identity)
        for backend, viewer_surface in streamable_viewer_surfaces.items():
            stream_backend_kwargs = StreamComponentMessageExtraAuthority.from_context(
                viewer_surface,
                context=context,
                source_metadata_items=source_metadata_items,
            ).viewer_backend_kwargs(
                producer=producer,
                images_dir=images_dir,
            )
            result[backend] = ViewerStreamBackendCallKwargs(
                stream_backend_kwargs,
            )
        return result


class ArtifactStreamSourceMetadataAuthority:
    """Choose source metadata for the viewer request that starts artifact streaming."""

    @staticmethod
    def metadata_items(
        *,
        materialization_spec: MaterializationSpec,
        data: MaterializationValue,
        fallback_source_identity: SourceImageIdentity | None,
    ) -> StreamSourceComponentMetadataItems:
        fallback_source_identity = (
            fallback_source_identity
            or ArtifactStreamSourceMetadataAuthority.payload_source_identity(data)
        )
        emitted_identities = materialization_spec.emitted_source_identities(data)
        if emitted_identities:
            return StreamSourceComponentMetadataItems.from_source_identities(
                emitted_identities,
                fallback_source_identity=fallback_source_identity,
            )
        return StreamSourceComponentMetadataItems.from_values(
            (
                fallback_source_identity.component_metadata
                if fallback_source_identity is not None
                else None,
            )
        )

    @staticmethod
    def payload_source_identity(
        data: MaterializationValue,
    ) -> SourceImageIdentity | None:
        metadata = image_payload_metadata(data)
        source_identity = metadata.source_provenance.scalar_source_identity
        if source_identity.addressable:
            return source_identity
        return None


class ArtifactMaterializationTargetPlan(ABC, metaclass=AutoRegisterMeta):
    """Nominal target policy for artifact materialization destinations."""

    __registry_key__ = "target_key"
    __skip_if_no_key__ = True
    target_key: ClassVar[str | None] = None

    def backend_plan(
        self,
        plan: CompiledStepPlan,
        context: "ProcessingContext",
        materialization: "RuntimeArtifactMaterialization",
    ) -> ArtifactMaterializationBackendPlan:
        streams_artifact = not plan.compiled_function_pattern.publishes_output_to_main_flow(
            materialization.output_plan,
            materialization.record.key.scope.value_text,
        )
        return ArtifactMaterializationBackendPlan(
            persistent_backend_kwargs=self.persistent_backend_kwargs(),
            streaming_viewer_surfaces=(
                {
                    config.backend.value: config.streaming_viewer_surface(context)
                    for config in plan.streaming_configs.values()
                    if step_axis_allows_config(
                        context.step_axis_filters,
                        step_index=plan.step_index,
                        config=config,
                        axis_id=context.axis_id,
                    )
                }
                if streams_artifact
                else {}
            ),
        )

    @abstractmethod
    def persistent_backend_kwargs(self) -> BackendKwargs:
        """Return persistent materialization backends owned by this policy."""


@dataclass(frozen=True, slots=True)
class PersistentArtifactMaterializationTargetPlan(ArtifactMaterializationTargetPlan):
    """Target policy for persistent files plus any enabled viewer streams."""

    target_key = "persistent"
    backend: str

    def persistent_backend_kwargs(self) -> BackendKwargs:
        return {self.backend: RawBackendKwargs()}


class StreamingOnlyArtifactMaterializationTargetPlan(ArtifactMaterializationTargetPlan):
    """Target policy for viewer streams with no persistent artifact files."""

    target_key = "streaming_only"

    def persistent_backend_kwargs(self) -> BackendKwargs:
        return {}


@dataclass(frozen=True, slots=True)
class ArtifactAnalysisOutputDescriptor:
    """Filename plus source identity for an artifact materialization."""

    filename: str
    source_identity: SourceImageIdentity | None
    source_filename: str | None = None


@dataclass(frozen=True, slots=True)
class PlannedArtifactMaterializationPath:
    """Compile-time preview of paths one materialized artifact group may emit."""

    group_key: str | None
    base_path: str
    candidate_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannedArtifactMaterializationPreview:
    """Compile-time materialization path preview for an explicit artifact spec."""

    filename_uses_source_identity: bool
    paths: tuple[PlannedArtifactMaterializationPath, ...]
    runtime_metadata_can_refine_paths: bool


@dataclass(frozen=True, slots=True)
class ArtifactRecordSourceDescriptor:
    """Resolved source stem and stream identity for one artifact record."""

    filename: str
    source_identity: SourceImageIdentity | None

    @property
    def filename_stem(self) -> str:
        return Path(self.filename).stem


class AnalysisOutputDescriptorAuthority:
    """Own analysis artifact filenames and their source component metadata."""

    @classmethod
    def build(
        cls,
        output_key: str,
        plan: CompiledStepPlan,
        context: "ProcessingContext",
        dict_key: str | None = None,
        artifact_path: str | None = None,
        record: StoredRuntimeValue | None = None,
        materialization_spec: MaterializationSpec | None = None,
        output_plan: ArtifactOutputPlan | None = None,
    ) -> ArtifactAnalysisOutputDescriptor:
        """Build analysis output identity from the artifact's source metadata."""
        record_source = None
        if record is not None:
            if materialization_spec is None:
                raise ValueError(
                    "Artifact record descriptor requires a materialization spec."
                )
            record_source = cls.record_source_descriptor(
                context,
                plan,
                record,
                materialization_spec,
                output_plan=output_plan,
            )
        if record_source is not None:
            return ArtifactAnalysisOutputDescriptor(
                filename=(
                    f"{record_source.filename_stem}_{output_key}"
                    f"_step{plan.pipeline_position}.roi.zip"
                ),
                source_identity=record_source.source_identity,
                source_filename=record_source.filename,
            )

        if artifact_path is not None:
            source_identity = cls.source_identity_for_path(
                context,
                artifact_path,
            )
            if source_identity is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    source_identity=source_identity,
                )
        if dict_key is not None and artifact_path is not None:
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{Path(artifact_path).stem}.roi.zip",
                source_identity=None,
            )
        return ArtifactAnalysisOutputDescriptor(
            filename=f"{plan.axis_id}_{output_key}_step{plan.pipeline_position}.roi.zip",
            source_identity=None,
        )

    @staticmethod
    def record_payload_metadata(
        record: StoredRuntimeValue | None,
    ) -> ImagePayloadMetadata | None:
        if record is None:
            return None
        if isinstance(record.value.data, SourceImageProvenanceFields):
            return ImagePayloadMetadata(
                source_provenance=record.value.data.source_provenance,
            )
        return image_payload_metadata(record.value.data)

    @classmethod
    def record_metadata_with_runtime_scope(
        cls,
        record: StoredRuntimeValue,
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        """Attach the artifact key's complete typed execution identity."""

        scope = record.key.scope
        if scope.has_fixed_components:
            component_metadata = scope.fixed_component_metadata(
                metadata.source_component_metadata
            )
        elif scope.component is not None:
            component_metadata = with_source_component_metadata(
                dict(metadata.source_component_metadata or {}),
                scope.component,
                scope.require_value_text(),
            )
        else:
            return metadata
        return metadata.with_source_provenance(
            metadata.source_provenance.with_source_component_metadata(
                component_metadata
            ),
        )

    @classmethod
    def record_source_descriptor(
        cls,
        context: "ProcessingContext",
        plan: CompiledStepPlan,
        record: StoredRuntimeValue,
        materialization_spec: MaterializationSpec,
        *,
        output_plan: ArtifactOutputPlan | None = None,
    ) -> ArtifactRecordSourceDescriptor | None:
        """Return source filename stem and identity for one artifact record."""
        exact_fixed_scope = record.key.scope.has_fixed_components
        if not materialization_spec.uses_source_identity_filename():
            return None
        metadata = cls.record_payload_metadata(record)
        if metadata is None:
            return None
        if (
            output_plan is not None
            and output_plan.materialization_source() is not None
            and output_plan.materialization_source()
            != output_plan.source_context_source()
        ):
            metadata = output_plan.materialization_metadata(record.value)
        metadata = cls.record_metadata_with_runtime_scope(
            record,
            metadata,
        )
        parser_context = FunctionOutputParserContext.from_processing_context(context)
        use_filename_identity = materialization_spec.uses_filename_source_identity(
            record.value.data
        )
        if use_filename_identity or exact_fixed_scope:
            identity = FunctionOutputIdentityAuthority.filename_identity_from_metadata(
                parser_context.parser,
                metadata,
            )
            if identity is None:
                raise ValueError(
                    f"Artifact output {record.key.name!r} requires source-identity "
                    "filename materialization from its exact execution scope, but "
                    "its declared source metadata has no addressable identity."
                )
        else:
            identity = FunctionOutputIdentityAuthority.identity_from_metadata(
                parser_context.parser,
                metadata,
                fallback_identity_path=record.path,
                variable_components=plan.variable_components,
            )
        if identity is not None:
            try:
                filename = Path(
                    FunctionOutputPathAuthority.filename_for_identity(
                        parser_context.parser,
                        identity,
                    )
                ).name
            except ValueError:
                if (
                    exact_fixed_scope
                    or (
                        output_plan is not None
                        and output_plan.materialization_source() is not None
                    )
                ):
                    raise
            else:
                return ArtifactRecordSourceDescriptor(
                    filename=filename,
                    source_identity=cls.record_source_identity_from_metadata(
                        metadata,
                        identity,
                    ),
                )
        source_path = metadata.source_provenance.scalar_source_identity.path
        if source_path is None:
            return None
        return ArtifactRecordSourceDescriptor(
            filename=Path(source_path).name,
            source_identity=cls.record_source_identity_from_metadata(
                metadata,
                identity,
            ),
        )

    @classmethod
    def record_source_identity_from_metadata(
        cls,
        metadata: ImagePayloadMetadata,
        identity: FunctionOutputIdentity | None,
    ) -> SourceImageIdentity | None:
        """Return scalar source identity with parser-resolved component metadata."""
        source_identity = metadata.source_provenance.scalar_source_identity
        if identity is not None:
            if metadata.source_component_metadata is None:
                component_metadata = {}
            else:
                component_metadata = dict(metadata.source_component_metadata)
            identity_metadata = identity.filename_component_metadata()
            for key, value in identity_metadata.items():
                component = AllComponents.from_value(str(key))
                if component is None:
                    component_metadata[str(key)] = value
                    continue
                component_metadata = with_source_component_metadata(
                    component_metadata,
                    component,
                    value,
                )
            source_identity = SourceImageIdentity(
                path=source_identity.path,
                component_metadata=component_metadata,
            )
        if source_identity.addressable:
            return source_identity
        return None

    @classmethod
    def materialization_base_path(
        cls,
        *,
        plan: CompiledStepPlan,
        output_descriptor: ArtifactAnalysisOutputDescriptor,
        output_plan: ArtifactOutputPlan,
    ) -> Path:
        """Return the declared base path for one materialized artifact record."""
        if (
            issubclass(output_plan.artifact_type, ImageArtifactType)
            and output_plan.materialization_uses_source_identity_filename()
            and output_descriptor.source_filename is not None
        ):
            return Path(plan.artifact_images_dir) / output_descriptor.source_filename
        return plan.artifact_analysis_output_dir / output_descriptor.filename

    @staticmethod
    def source_identity_for_path(
        context: "ProcessingContext",
        path: str | Path,
    ) -> SourceImageIdentity | None:
        """Return microscope source identity for a source path when available."""
        component_metadata = FunctionOutputParserContext.from_processing_context(
            context
        ).parse_path_metadata(path)
        if component_metadata is None:
            return None
        return SourceImageIdentity(component_metadata=component_metadata)


def actual_materialization_records(
    *,
    store: RuntimeValueStore,
    plan: CompiledStepPlan,
    output_plan: ArtifactOutputPlan,
) -> tuple[StoredRuntimeValue, ...]:
    """Return actually produced memory records for one planned artifact output."""
    if not output_plan.group_keys:
        raise RuntimeError(
            f"Artifact output plan '{output_plan.name}' has no group keys."
        )

    if output_plan.group_component is not None and tuple(output_plan.group_keys) == (
        None,
    ):
        dynamic_records = tuple(
            record
            for record in store.find(
                name=output_plan.name,
                artifact_type=output_plan.artifact_type,
                axis_id=plan.axis_id,
            )
            if record.key.scope.value_text is not None
            and record.backend == Backend.MEMORY.value
            and store.get(record.key).location == record.location
        )
        if dynamic_records:
            reducer = ArtifactMaterializationRecordReducer.for_artifact_type(
                output_plan.artifact_type
            )
            record_sort_items = []
            dynamic_group_keys = tuple(
                dict.fromkeys(
                    str(record.key.scope.value_text) for record in dynamic_records
                )
            )
            group_order = {
                group_key: index
                for index, group_key in enumerate(sorted(dynamic_group_keys))
            }
            for group_key in dynamic_group_keys:
                records_for_group = tuple(
                    record
                    for record in dynamic_records
                    if str(record.key.scope.value_text) == group_key
                )
                for record in reducer.records_for_exact_scopes(
                    records=records_for_group,
                    output_plan=output_plan,
                    axis_id=plan.axis_id,
                    group_key=group_key,
                ):
                    record_sort_items.append((group_order[group_key], record))
            return tuple(
                record
                for _, record in sorted(record_sort_items, key=lambda item: item[0])
            )

    group_order = {
        group_key: index for index, group_key in enumerate(output_plan.group_keys)
    }
    record_sort_items = []
    missing_group_keys = []
    for group_key in output_plan.group_keys:
        records = tuple(
            record
            for record in store.find(
                name=output_plan.name,
                artifact_type=output_plan.artifact_type,
                axis_id=plan.axis_id,
                group_key=group_key,
                match_group=True,
            )
            if record.backend == Backend.MEMORY.value
            and store.get(record.key).location == record.location
        )
        if not records:
            missing_group_keys.append(group_key)
            continue
        for record in ArtifactMaterializationRecordReducer.for_artifact_type(
            output_plan.artifact_type
        ).records_for_exact_scopes(
            records=records,
            output_plan=output_plan,
            axis_id=plan.axis_id,
            group_key=group_key,
        ):
            record_sort_items.append((group_order[group_key], record))

    if missing_group_keys and not record_sort_items:
        candidates = tuple(
            store.find(
                name=output_plan.name,
                artifact_type=output_plan.artifact_type,
                axis_id=plan.axis_id,
            )
        )
        candidate_locations = tuple(
            (
                candidate.key.scope.value_text,
                candidate.backend,
                candidate.path,
            )
            for candidate in candidates
        )
        identity_records = tuple(
            candidate
            for candidate in candidates
            if candidate.key.scope.value_text is None
            and candidate.backend == Backend.MEMORY.value
            and store.get(candidate.key).location == candidate.location
        )
        if identity_records:
            return identity_records
        raise RuntimeError(
            f"Missing RuntimeValueStore record for planned artifact materialization "
            f"'{output_plan.name}' ({output_plan.artifact_type.value}) on axis "
            f"'{plan.axis_id}' groups {tuple(missing_group_keys)!r}. "
            f"Candidate same-name records: "
            f"{candidate_locations!r}."
        )
    return tuple(
        record for _, record in sorted(record_sort_items, key=lambda item: item[0])
    )
@dataclass(frozen=True, slots=True)
class RuntimeArtifactMaterialization:
    """One exact compiled artifact value and its generic materialization target."""

    output_key: str
    output_plan: ArtifactOutputPlan
    spec: MaterializationSpec
    record: StoredRuntimeValue
    data: MaterializationValue
    base_path: Path
    source_identity: SourceImageIdentity | None

    @classmethod
    def from_record(
        cls,
        *,
        output_key: str,
        output_plan: ArtifactOutputPlan,
        record: StoredRuntimeValue,
        plan: CompiledStepPlan,
        context: "ProcessingContext",
    ) -> "RuntimeArtifactMaterialization":
        """Build one materialization from an exact compiled output record."""
        spec = output_plan.materialization
        if not isinstance(spec, MaterializationSpec):
            raise TypeError(
                f"Artifact output {output_plan.name!r} declares unsupported "
                f"materialization {type(spec).__name__}."
            )
        data = output_plan.materialization_payload(record.value)
        emits_projected_planes = spec.emits_variable_component_planes(data)
        if (
            issubclass(output_plan.artifact_type, ImageArtifactType)
            and output_plan.materialization_uses_source_identity_filename()
            and emits_projected_planes
        ):
            base_path = Path(plan.artifact_images_dir) / output_plan.name
            source_identity = (
                ArtifactStreamSourceMetadataAuthority.payload_source_identity(data)
            )
        else:
            output_descriptor = AnalysisOutputDescriptorAuthority.build(
                output_key,
                plan,
                context,
                record.key.scope.value_text,
                artifact_path=record.path,
                record=record,
                materialization_spec=spec,
                output_plan=output_plan,
            )
            base_path = AnalysisOutputDescriptorAuthority.materialization_base_path(
                plan=plan,
                output_descriptor=output_descriptor,
                output_plan=output_plan,
            )
            source_identity = output_descriptor.source_identity
        return cls(
            output_key=output_key,
            output_plan=output_plan,
            spec=spec,
            record=record,
            data=data,
            base_path=base_path,
            source_identity=source_identity,
        )

    def outputs(
        self,
        plan: CompiledStepPlan,
        context: "ProcessingContext",
    ) -> tuple[Output, ...]:
        """Derive the exact writer outputs for this runtime artifact."""
        return materialization_outputs(
            self.spec,
            self.data,
            str(self.base_path),
            context.filemanager,
            context=context,
            artifact_source_identity=self.source_identity,
            variable_components=self.output_plan.variable_components,
            output_key=self.output_key,
            step_index=plan.step_index,
        )

    def viewer_outputs(
        self,
        plan: CompiledStepPlan,
        context: "ProcessingContext",
    ) -> tuple[Output, ...]:
        """Return exact writer outputs accepted by this step's viewer backends."""
        backend_plan = StreamingOnlyArtifactMaterializationTargetPlan().backend_plan(
            plan,
            context,
            self,
        )
        return tuple(
            output
            for output in self.outputs(plan, context)
            if backend_plan.supports_stream_output_path(
                context.filemanager,
                output.path,
            )
        )


def runtime_artifact_materializations(
    plan: CompiledStepPlan,
    context: "ProcessingContext",
) -> tuple[RuntimeArtifactMaterialization, ...]:
    """Derive actual materializations from compiled outputs and runtime values."""

    materializations: list[RuntimeArtifactMaterialization] = []
    store = context.runtime_value_store
    for output_plan in plan.artifact_outputs.values():
        if output_plan.materialization is None:
            continue
        records = actual_materialization_records(
            store=store,
            plan=plan,
            output_plan=output_plan,
        )
        for record in records:
            materializations.append(
                RuntimeArtifactMaterialization.from_record(
                    output_key=output_plan.name,
                    output_plan=output_plan,
                    record=record,
                    plan=plan,
                    context=context,
                )
            )
    return tuple(materializations)


def observed_runtime_artifact_materializations(
    plan: CompiledStepPlan,
    context: "ProcessingContext",
) -> tuple[RuntimeArtifactMaterialization, ...]:
    """Derive historical materializations from the runtime observation ledger."""
    observed = context.runtime_value_store.observed_values
    return tuple(
        RuntimeArtifactMaterialization.from_record(
            output_key=output_plan.name,
            output_plan=output_plan,
            record=record,
            plan=plan,
            context=context,
        )
        for output_plan in plan.artifact_outputs.values()
        if output_plan.materialization is not None
        for record in observed
        if RuntimeValueStore.address_matches_plan(
            RuntimeArtifactAddress.from_record(record),
            output_plan,
            axis_id=plan.axis_id,
        )
    )


def materialized_artifact_output_paths(
    plan: CompiledStepPlan,
    context: "ProcessingContext",
) -> tuple[Path, ...]:
    """Re-derive exact persistent output paths from runtime materializations."""

    if not plan.runtime_artifact_materialization.has_persistent_target:
        return ()
    return tuple(
        Path(output.path)
        for materialization in runtime_artifact_materializations(plan, context)
        if materialization.spec.participates_in_runtime_export_observation()
        for output in materialization.outputs(plan, context)
    )


def planned_materialization_preview(
    *,
    context: "ProcessingContext",
    plan: CompiledStepPlan,
    output_key: str,
    output_plan: ArtifactOutputPlan,
) -> PlannedArtifactMaterializationPreview | None:
    """Return candidate output paths for an explicitly materialized artifact.

    Runtime payload metadata can still refine ROI and source-identity filenames.
    This preview stays deliberately compile-time: it reports candidates from the
    declared MaterializationSpec and the compiled analysis output directory.
    """
    materialization_spec = output_plan.materialization
    if not isinstance(materialization_spec, MaterializationSpec):
        return None

    path_previews = tuple(
        _planned_materialization_path(
            context=context,
            plan=plan,
            output_key=output_key,
            output_plan=output_plan.for_group(group_key),
            materialization_spec=materialization_spec,
        )
        for group_key in (output_plan.group_keys or (None,))
    )
    return PlannedArtifactMaterializationPreview(
        filename_uses_source_identity=(
            output_plan.materialization_uses_source_identity_filename()
        ),
        paths=path_previews,
        runtime_metadata_can_refine_paths=True,
    )


def _planned_materialization_path(
    *,
    context: "ProcessingContext",
    plan: CompiledStepPlan,
    output_key: str,
    output_plan: ArtifactOutputPlan,
    materialization_spec: MaterializationSpec,
) -> PlannedArtifactMaterializationPath:
    descriptor = AnalysisOutputDescriptorAuthority.build(
        output_key,
        plan,
        context,
        output_plan.single_group_key,
        artifact_path=output_plan.path,
        output_plan=output_plan,
    )
    base_path = str(plan.artifact_analysis_output_dir / descriptor.filename)
    return PlannedArtifactMaterializationPath(
        group_key=output_plan.single_group_key,
        base_path=base_path,
        candidate_paths=materialization_spec.candidate_paths(base_path),
    )


def materialize_artifact_outputs(
    filemanager: "FileManager",
    plan: CompiledStepPlan,
    target_plan: ArtifactMaterializationTargetPlan,
    context: "ProcessingContext",
) -> None:
    """Materialize planned artifact outputs to persistent and streaming backends."""
    from openhcs.processing.materialization import materialize

    images_dir = plan.artifact_images_dir

    for materialization in runtime_artifact_materializations(plan, context):
        backend_plan = target_plan.backend_plan(plan, context, materialization)
        record = materialization.record
        data = materialization.data
        filemanager.ensure_directory(Path(record.path).parent, record.backend)
        stream_output_paths = materialization.spec.candidate_paths(
            str(materialization.base_path)
        )
        backends = backend_plan.backends_for(
            materialization_spec=materialization.spec,
            filemanager=filemanager,
            stream_output_paths=stream_output_paths,
        )
        if not backends:
            continue
        materialize(
            materialization.spec,
            data,
            str(materialization.base_path),
            filemanager,
            backends,
            backend_plan.backend_kwargs(
                materialization_spec=materialization.spec,
                data=data,
                fallback_source_identity=materialization.source_identity,
                producer_identity=(
                    FunctionStepOutputProducerIdentityAuthority.build(
                        FunctionStepOutputProducerIdentityRequest.from_artifact(
                            plan,
                            materialization.output_plan,
                        )
                    )
                ),
                context=context,
                filemanager=filemanager,
                images_dir=images_dir,
                stream_output_paths=stream_output_paths,
            ),
            context=context,
            artifact_source_identity=materialization.source_identity,
            variable_components=materialization.output_plan.variable_components,
            output_key=materialization.output_key,
            step_index=plan.step_index,
        )
