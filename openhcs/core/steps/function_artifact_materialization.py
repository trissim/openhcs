"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamProducer
from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.artifact_materialization_policy import (
    resolve_artifact_materialization_spec,
)
from openhcs.core.runtime_stores import (
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import (
    SourceImageIdentity,
)
from openhcs.core.source_matching import (
    source_component_metadata_value,
    source_metadata_component,
    with_source_component_metadata,
)
from openhcs.core.source_metadata import SourceMetadataRoleView
from openhcs.core.steps.function_output_identity import (
    FunctionOutputComponentIdentityAuthority,
    FunctionOutputIdentity,
    FunctionOutputIdentityAuthority,
    FunctionOutputParserContext,
    FunctionOutputPathAuthority,
)
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityAuthority,
    FunctionStepOutputProducerIdentityRequest,
    step_output_manifest,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraAuthority,
    StreamScopedDisplayConfig,
    StreamSourceComponentMetadataItems,
)
from openhcs.core.streaming_config_factory import StreamingViewerSurface
from openhcs.processing.materialization.core import (
    BackendKwargs,
    MaterializationSpec,
    MaterializationValue,
    RawBackendKwargs,
    ViewerStreamBackendCallKwargs,
)

if TYPE_CHECKING:
    from polystore.filemanager import FileManager
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class ArtifactMaterializationBackendPlan:
    """Persistent backends and streaming payload builders for artifact output."""

    persistent_backend_kwargs: BackendKwargs
    streaming_viewer_surfaces: Mapping[str, StreamingViewerSurface]

    def backends_for(
        self,
        *,
        filemanager: "FileManager",
        stream_output_paths: tuple[str, ...],
    ) -> list[str]:
        return [
            *self.persistent_backend_kwargs,
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
                filemanager._get_backend(backend).supports_file_path(path)
                for path in stream_output_paths
            )
        }

    def backend_kwargs(
        self,
        *,
        source_metadata_items: StreamSourceComponentMetadataItems,
        producer_identity: StreamProducerIdentity,
        context: "ProcessingContext",
        filemanager: "FileManager",
        images_dir: str,
        stream_output_paths: tuple[str, ...],
    ) -> BackendKwargs:
        result: BackendKwargs = {}
        for backend, kwargs in self.persistent_backend_kwargs.items():
            values = dict(kwargs)
            if "images_dir" in values and values["images_dir"] != images_dir:
                raise ValueError(
                    "Persistent artifact backend kwargs already define a different "
                    f"'images_dir': {values['images_dir']!r} != {images_dir!r}."
            )
            values["images_dir"] = images_dir
            result[backend] = RawBackendKwargs(values)
        producer = ViewerStreamProducer.from_identity(producer_identity)
        for backend, viewer_surface in self.streamable_viewer_surfaces(
            filemanager=filemanager,
            stream_output_paths=stream_output_paths,
        ).items():
            component_order = tuple(
                str(component)
                for component in viewer_surface.display_config.COMPONENT_ORDER
            )
            complete_component_order = source_metadata_items.complete_component_order(
                component_order
            )
            if complete_component_order != component_order:
                logger.info(
                    "Artifact viewer stream omitting incomplete source component(s) "
                    "%s from backend %s",
                    tuple(
                        component
                        for component in component_order
                        if component not in complete_component_order
                    ),
                    backend,
                )
                viewer_surface = replace(
                    viewer_surface,
                    display_config=StreamScopedDisplayConfig(
                        base=viewer_surface.display_config,
                        component_order=complete_component_order,
                    ),
                )
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
        plan: FunctionStepExecutionPlan,
        context: "ProcessingContext",
    ) -> ArtifactMaterializationBackendPlan:
        return ArtifactMaterializationBackendPlan(
            persistent_backend_kwargs=self.persistent_backend_kwargs(),
            streaming_viewer_surfaces={
                config.backend.value: config.streaming_viewer_surface(context)
                for config in plan.streaming_configs
            },
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

    filename_stem: str
    source_identity: SourceImageIdentity | None

class AnalysisOutputDescriptorAuthority:
    """Own analysis artifact filenames and their source component metadata."""

    @classmethod
    def build(
        cls,
        output_key: str,
        plan: FunctionStepExecutionPlan,
        context: "ProcessingContext",
        dict_key: str | None = None,
        artifact_path: str | None = None,
        record: StoredRuntimeValue | None = None,
        materialization_spec: MaterializationSpec | None = None,
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
            )
        if record_source is not None:
            return ArtifactAnalysisOutputDescriptor(
                filename=(
                    f"{record_source.filename_stem}_{output_key}"
                    f"_step{plan.pipeline_position}.roi.zip"
                ),
                source_identity=record_source.source_identity,
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

        memory_paths = cls.produced_memory_paths(context, plan)

        if not memory_paths:
            if dict_key is not None and artifact_path is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    source_identity=None,
                )
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{plan.axis_id}_{output_key}_step{plan.pipeline_position}.roi.zip",
                source_identity=None,
            )

        group_by_component = cls.group_by_component(plan)
        if dict_key and group_by_component is not None:
            parser_context = FunctionOutputParserContext.from_processing_context(
                context
            )
            filtered_paths = []
            for path in memory_paths:
                metadata = parser_context.parse_path_metadata(path)
                if (
                    metadata
                    and cls.group_key_matches_metadata(
                        metadata,
                        group_by_component,
                        dict_key,
                    )
                ):
                    filtered_paths.append(path)

            if filtered_paths:
                memory_paths = filtered_paths

        base_filename = Path(memory_paths[0]).stem
        return ArtifactAnalysisOutputDescriptor(
            filename=f"{base_filename}_{output_key}_step{plan.pipeline_position}.roi.zip",
            source_identity=cls.source_identity_for_path(
                context,
                memory_paths[0],
            ),
        )

    @staticmethod
    def record_payload_metadata(
        record: StoredRuntimeValue | None,
    ) -> ImagePayloadMetadata | None:
        if record is None:
            return None
        payload_metadata = image_payload_metadata(record.value.data)
        schema_provenance = record.value.schema.source_provenance
        if not schema_provenance.has_values:
            return payload_metadata
        return replace(
            payload_metadata,
            source_provenance=schema_provenance.with_missing_from(
                payload_metadata.source_provenance,
            ),
        )

    @staticmethod
    def group_by_component(
        plan: FunctionStepExecutionPlan,
    ) -> AllComponents | None:
        if plan.group_by_value is None:
            return None
        return AllComponents.from_value(str(plan.group_by_value))

    @staticmethod
    def group_key_matches_metadata(
        metadata: Mapping[str, object],
        component: AllComponents,
        group_key: str,
    ) -> bool:
        value = source_component_metadata_value(metadata, component)
        if value is None:
            return False
        return (
            FunctionOutputComponentIdentityAuthority.coerce_component_value(
                component,
                value,
            )
            == FunctionOutputComponentIdentityAuthority.coerce_component_value(
                component,
                group_key,
            )
        )

    @classmethod
    def record_metadata_with_runtime_plane_group(
        cls,
        plan: FunctionStepExecutionPlan,
        record: StoredRuntimeValue,
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        """Attach group-projected runtime-plane identity to record metadata."""
        if not plan.group_projects_runtime_plane:
            return metadata
        group_by_value = plan.group_by_value
        if group_by_value is None:
            return metadata
        component_metadata = dict(metadata.source_component_metadata or {})
        component_metadata[group_by_value] = record.key.scope.group_key
        return metadata.with_source_provenance(
            metadata.source_provenance.with_source_component_metadata(
                component_metadata
            ),
        )

    @classmethod
    def record_metadata_with_runtime_group_key(
        cls,
        plan: FunctionStepExecutionPlan,
        record: StoredRuntimeValue,
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        """Attach runtime group identity when metadata marks one missing axis."""
        metadata = cls.record_metadata_with_runtime_plane_group(
            plan,
            record,
            metadata,
        )
        if plan.group_by_value is not None:
            return metadata
        group_key = record.key.scope.group_key
        if group_key is None:
            return metadata
        component = cls.single_null_source_component(metadata.source_component_metadata)
        if component is None:
            return metadata
        component_metadata = dict(metadata.source_component_metadata or {})
        component_metadata = with_source_component_metadata(
            component_metadata,
            component,
            group_key,
        )
        return metadata.with_source_provenance(
            metadata.source_provenance.with_source_component_metadata(
                component_metadata
            ),
        )

    @staticmethod
    def single_null_source_component(
        metadata: Mapping[str, object] | None,
    ) -> AllComponents | None:
        """Return the only OpenHCS component explicitly marked as missing."""
        if metadata is None:
            return None
        missing_components: list[AllComponents] = []
        for field, value in SourceMetadataRoleView(metadata).scalar_items():
            if value is not None:
                continue
            component = source_metadata_component(field)
            if component is None or component in missing_components:
                continue
            missing_components.append(component)
        if len(missing_components) != 1:
            return None
        return missing_components[0]

    @classmethod
    def record_source_descriptor(
        cls,
        context: "ProcessingContext",
        plan: FunctionStepExecutionPlan,
        record: StoredRuntimeValue,
        materialization_spec: MaterializationSpec,
    ) -> ArtifactRecordSourceDescriptor | None:
        """Return source filename stem and identity for one artifact record."""
        metadata = cls.record_payload_metadata(record)
        if metadata is None:
            return None
        metadata = cls.record_metadata_with_runtime_group_key(
            plan,
            record,
            metadata,
        )
        parser_context = FunctionOutputParserContext.from_processing_context(context)
        use_filename_identity = materialization_spec.uses_filename_source_identity(
            record.value.data
        )
        if use_filename_identity:
            identity = FunctionOutputIdentityAuthority.filename_identity_from_metadata(
                parser_context.parser,
                metadata,
                fallback_identity_path=record.path,
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
                filename_stem = Path(
                    FunctionOutputPathAuthority.filename_for_identity(
                        parser_context.parser,
                        identity,
                    )
                ).stem
            except ValueError:
                logger.debug(
                    "Artifact record metadata identity is not filename-complete; "
                    "falling back to planned output lineage.",
                    exc_info=True,
                )
            else:
                return ArtifactRecordSourceDescriptor(
                    filename_stem=filename_stem,
                    source_identity=cls.record_source_identity_from_metadata(
                        metadata,
                        identity,
                        use_filename_identity=use_filename_identity,
                    ),
                )
        source_path = metadata.source_provenance.scalar_source_identity.path
        if source_path is None:
            return None
        return ArtifactRecordSourceDescriptor(
            filename_stem=Path(source_path).stem,
            source_identity=cls.record_source_identity_from_metadata(
                metadata,
                identity,
                use_filename_identity=use_filename_identity,
            ),
        )

    @classmethod
    def record_source_identity_from_metadata(
        cls,
        metadata: ImagePayloadMetadata,
        identity: FunctionOutputIdentity | None,
        *,
        use_filename_identity: bool = False,
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

    @staticmethod
    def produced_memory_paths(
        context: "ProcessingContext",
        plan: FunctionStepExecutionPlan,
    ) -> list[str]:
        """Return current-step output paths without scanning a shared directory."""
        return [
            str(path if path.is_absolute() else plan.output_dir / path)
            for path in (
                Path(item)
                for item in step_output_manifest(context).produced_paths_for(plan)
            )
        ]

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
    plan: FunctionStepExecutionPlan,
    output_plan: ArtifactOutputPlan,
) -> tuple[StoredRuntimeValue, ...]:
    """Return actually produced memory records for one planned artifact output."""
    if not output_plan.group_keys:
        raise RuntimeError(
            f"Artifact output plan '{output_plan.name}' has no group keys."
        )

    planned_path_by_group = {
        group_key: output_plan.for_group(group_key).path
        for group_key in output_plan.group_keys
    }
    planned_paths = frozenset(planned_path_by_group.values())
    group_order = {
        group_key: index
        for index, group_key in enumerate(output_plan.group_keys)
    }
    record_sort_items = []
    for record in store.find(
        name=output_plan.name,
        kind=output_plan.kind,
        axis_id=plan.axis_id,
    ):
        if record.backend != Backend.MEMORY.value or record.path not in planned_paths:
            continue

        group_key = record.key.scope.group_key
        if group_key not in planned_path_by_group:
            raise RuntimeError(
                f"Runtime artifact record for planned output '{output_plan.name}' "
                f"has unplanned group key {group_key!r}."
            )
        expected_path = planned_path_by_group[group_key]
        if record.path != expected_path:
            raise RuntimeError(
                f"Runtime artifact record for planned output '{output_plan.name}' "
                f"uses path {record.path!r}; expected {expected_path!r} for "
                f"group key {group_key!r}."
            )
        record_sort_items.append((group_order[group_key], record))

    if not record_sort_items:
        raise RuntimeError(
            f"Missing RuntimeValueStore record for planned artifact materialization "
            f"'{output_plan.name}' ({output_plan.kind.value}) on axis "
            f"'{plan.axis_id}'."
        )
    return tuple(
        record
        for _, record in sorted(record_sort_items, key=lambda item: item[0])
    )

def planned_materialization_preview(
    *,
    context: "ProcessingContext",
    plan: FunctionStepExecutionPlan,
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
    plan: FunctionStepExecutionPlan,
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
    )
    base_path = str(plan.artifact_analysis_output_dir / descriptor.filename)
    return PlannedArtifactMaterializationPath(
        group_key=output_plan.single_group_key,
        base_path=base_path,
        candidate_paths=materialization_spec.candidate_paths(base_path),
    )

def materialize_artifact_outputs(
    filemanager: "FileManager",
    plan: FunctionStepExecutionPlan,
    target_plan: ArtifactMaterializationTargetPlan,
    context: "ProcessingContext",
) -> None:
    """Materialize planned artifact outputs to persistent and streaming backends."""
    from openhcs.processing.materialization import materialize

    backend_plan = target_plan.backend_plan(plan, context)
    analysis_output_dir = plan.artifact_analysis_output_dir
    images_dir = plan.artifact_images_dir
    store = context.runtime_value_store

    for output_key, output_plan in plan.artifact_outputs.items():
        if output_plan.materialization is None and output_plan.kind is ArtifactKind.SPECIAL:
            continue

        records = actual_materialization_records(
            store=store,
            plan=plan,
            output_plan=output_plan,
        )

        for record in records:
            dict_key = record.key.scope.group_key

            filemanager.ensure_directory(
                Path(record.path).parent, record.backend
            )
            data = record.value.materialization_payload()
            mat_spec = resolve_artifact_materialization_spec(
                output_plan,
                record.value,
            )
            if mat_spec is None:
                continue

            output_descriptor = AnalysisOutputDescriptorAuthority.build(
                output_key,
                plan,
                context,
                dict_key,
                artifact_path=record.path,
                record=record,
                materialization_spec=mat_spec,
            )
            analysis_path = analysis_output_dir / output_descriptor.filename
            stream_output_paths = mat_spec.candidate_paths(str(analysis_path))
            backends = backend_plan.backends_for(
                filemanager=filemanager,
                stream_output_paths=stream_output_paths,
            )
            if not backends:
                continue
            materialize(
                mat_spec,
                data,
                str(analysis_path),
                filemanager,
                backends,
                backend_plan.backend_kwargs(
                    source_metadata_items=(
                        ArtifactStreamSourceMetadataAuthority.metadata_items(
                            materialization_spec=mat_spec,
                            data=data,
                            fallback_source_identity=(
                                output_descriptor.source_identity
                            ),
                        )
                    ),
                    producer_identity=(
                        FunctionStepOutputProducerIdentityAuthority.build(
                            FunctionStepOutputProducerIdentityRequest(
                                plan=plan,
                                output_kind="artifact",
                                output_key=output_key,
                                artifact_kind=output_plan.kind.value,
                            )
                        )
                    ),
                    context=context,
                    filemanager=filemanager,
                    images_dir=images_dir,
                    stream_output_paths=stream_output_paths,
                ),
                context=context,
                artifact_source_identity=output_descriptor.source_identity,
                variable_components=plan.variable_components,
            )
