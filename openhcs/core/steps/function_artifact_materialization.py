"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamBackendKwargs
from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.artifact_materialization_policy import (
    resolve_artifact_materialization_spec,
)
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
    require_runtime_value_store,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    SourceComponentMetadata,
    image_payload_metadata,
)
from openhcs.core.source_matching import source_component_metadata_value
from openhcs.core.steps.function_output_identity import FunctionOutputParserContext
from openhcs.core.steps.function_output_manifest import step_output_manifest
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.stream_component_semantics import (
    OpenHCSViewerStreamRequestAuthority,
    OpenHCSViewerStreamSourceScope,
    StreamSourceComponentMetadataItems,
)
from openhcs.core.streaming_config_factory import StreamingViewerSurface
from openhcs.processing.materialization.core import BackendKwargs

if TYPE_CHECKING:
    from polystore.filemanager import FileManager
    from openhcs.core.context.processing_context import ProcessingContext


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationBackendPlan:
    """Persistent backends and streaming payload builders for artifact output."""

    persistent_backend_kwargs: BackendKwargs
    streaming_viewer_surfaces: Mapping[str, StreamingViewerSurface]

    @property
    def backends(self) -> list[str]:
        return [
            *self.persistent_backend_kwargs,
            *self.streaming_viewer_surfaces,
        ]


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
        return {self.backend: {}}


class StreamingOnlyArtifactMaterializationTargetPlan(ArtifactMaterializationTargetPlan):
    """Target policy for viewer streams with no persistent artifact files."""

    target_key = "streaming_only"

    def persistent_backend_kwargs(self) -> BackendKwargs:
        return {}


@dataclass(frozen=True, slots=True)
class ArtifactAnalysisOutputDescriptor:
    """Filename plus source component metadata for an artifact materialization."""

    filename: str
    artifact_source_component_metadata: SourceComponentMetadata | None


@dataclass(frozen=True, slots=True)
class ArtifactRecordSourceDescriptor:
    """Scalar source identity for an artifact record, only when the record has one."""

    record: StoredRuntimeValue | None

    @property
    def payload_metadata(self) -> ImagePayloadMetadata | None:
        if self.record is None:
            return None
        return image_payload_metadata(self.record.value.data)

    @property
    def has_multiplane_provenance(self) -> bool:
        metadata = self.payload_metadata
        if metadata is None:
            return False
        return metadata.source_image_provenance_planes.count > 1

    def source_path(self) -> str | None:
        """Return a scalar source path without collapsing multi-plane provenance."""
        metadata = self.payload_metadata
        if metadata is None:
            return None
        if metadata.source_path:
            return str(metadata.source_path)
        if self.has_multiplane_provenance:
            return None
        for source_path in metadata.source_image_provenance_planes.paths:
            if source_path:
                return str(source_path)
        return None

    def source_component_metadata(
        self,
        *,
        fallback: SourceComponentMetadata | None = None,
    ) -> SourceComponentMetadata | None:
        """Return scalar source metadata without borrowing from the first plane."""
        metadata = self.payload_metadata
        if metadata is None:
            return fallback
        if metadata.source_component_metadata is not None:
            return AnalysisOutputDescriptorAuthority.merge_source_component_metadata(
                metadata.source_component_metadata,
                fallback,
            )
        if self.has_multiplane_provenance:
            return fallback
        candidate = None
        for component_metadata in metadata.source_image_provenance_planes.component_metadata:
            if component_metadata is not None:
                candidate = component_metadata
                break
        return AnalysisOutputDescriptorAuthority.merge_source_component_metadata(
            candidate,
            fallback,
        )


class AnalysisOutputDescriptorAuthority:
    """Own analysis artifact filenames and their source component metadata."""

    @classmethod
    def build(
        cls,
        output_key: str,
        plan: FunctionStepExecutionPlan,
        dict_key: str | None = None,
        context: "ProcessingContext | None" = None,
        artifact_path: str | None = None,
        record: StoredRuntimeValue | None = None,
    ) -> ArtifactAnalysisOutputDescriptor:
        """Build analysis output identity from the artifact's source metadata."""
        record_source = ArtifactRecordSourceDescriptor(record)
        source_path = record_source.source_path()
        if source_path is not None:
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{Path(source_path).stem}_{output_key}_step{plan.pipeline_position}.roi.zip",
                artifact_source_component_metadata=record_source.source_component_metadata(
                    fallback=cls.source_component_metadata_for_path(context, source_path),
                ),
            )

        if artifact_path is not None:
            source_component_metadata = cls.source_component_metadata_for_path(
                context,
                artifact_path,
            )
            if source_component_metadata is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    artifact_source_component_metadata=source_component_metadata,
                )

        memory_paths = cls.produced_memory_paths(context, plan)

        if not memory_paths:
            if dict_key is not None and artifact_path is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    artifact_source_component_metadata=None,
                )
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{plan.axis_id}_{output_key}_step{plan.pipeline_position}.roi.zip",
                artifact_source_component_metadata=None,
            )

        if dict_key and context:
            parser_context = FunctionOutputParserContext.from_processing_context(
                context
            )
            filtered_paths = []
            for path in memory_paths:
                metadata = parser_context.parse_path_metadata(path)
                if metadata and str(metadata.get("channel")) == str(dict_key):
                    filtered_paths.append(path)

            if filtered_paths:
                memory_paths = filtered_paths

        base_filename = Path(memory_paths[0]).stem
        return ArtifactAnalysisOutputDescriptor(
            filename=f"{base_filename}_{output_key}_step{plan.pipeline_position}.roi.zip",
            artifact_source_component_metadata=cls.source_component_metadata_for_path(
                context,
                memory_paths[0],
            ),
        )

    @staticmethod
    def produced_memory_paths(
        context: "ProcessingContext | None",
        plan: FunctionStepExecutionPlan,
    ) -> list[str]:
        """Return current-step output paths without scanning a shared directory."""
        if context is None:
            return []
        return [
            str(path if path.is_absolute() else plan.output_dir / path)
            for path in (
                Path(item)
                for item in step_output_manifest(context).produced_paths_for(plan)
            )
        ]

    @staticmethod
    def merge_source_component_metadata(
        primary: SourceComponentMetadata | None,
        fallback: SourceComponentMetadata | None,
    ) -> SourceComponentMetadata | None:
        """Merge payload metadata with parser-derived axis metadata."""
        if primary is None:
            return fallback
        if fallback is None:
            return primary

        merged = dict(primary)
        for component in AllComponents:
            if source_component_metadata_value(merged, component) is not None:
                continue
            fallback_value = source_component_metadata_value(fallback, component)
            if fallback_value is not None:
                merged[component.value] = fallback_value
        return merged

    @staticmethod
    def source_component_metadata_for_path(
        context: "ProcessingContext | None",
        path: str | Path | None,
    ) -> SourceComponentMetadata | None:
        """Return microscope component metadata for a source path when available."""
        if context is None or path is None:
            return None
        return FunctionOutputParserContext.from_processing_context(
            context
        ).parse_path_metadata(path)


STREAMING_BACKENDS = frozenset(
    (
        Backend.NAPARI_STREAM.value,
        Backend.FIJI_STREAM.value,
    )
)


class BackendKwargsWithStreamIdentityAuthority:
    """Attach stream identity and source-plane metadata to streaming backends."""

    @staticmethod
    def build(
        backend_plan: ArtifactMaterializationBackendPlan,
        artifact_source_component_metadata: SourceComponentMetadata | None,
        producer_identity: StreamProducerIdentity,
        context: "ProcessingContext",
        images_dir: str,
    ) -> BackendKwargs:
        result: BackendKwargs = {
            backend: dict(kwargs)
            for backend, kwargs in backend_plan.persistent_backend_kwargs.items()
        }
        source_metadata_items = StreamSourceComponentMetadataItems.from_values(
            (artifact_source_component_metadata,)
        )
        for backend, viewer_surface in backend_plan.streaming_viewer_surfaces.items():
            if backend not in STREAMING_BACKENDS:
                raise ValueError(
                    f"Unsupported streaming backend in artifact plan: {backend}"
                )
            viewer_request = OpenHCSViewerStreamRequestAuthority.from_source_scope(
                OpenHCSViewerStreamSourceScope.from_viewer_surface(
                    viewer_surface,
                    context=context,
                    producer_identity=producer_identity,
                    source_metadata_items=source_metadata_items,
                ),
                images_dir=images_dir,
            )
            result[backend] = ViewerStreamBackendKwargs(viewer_request).to_kwargs()
        return result


class ArtifactStreamIdentityAuthority:
    """Build stable stream producer identities for artifact outputs."""

    @staticmethod
    def build(
        plan: FunctionStepExecutionPlan,
        output_key: str,
        output_plan: ArtifactOutputPlan,
    ) -> StreamProducerIdentity:
        return StreamProducerIdentity.pipeline_output(
            output_kind="artifact",
            output_key=output_key,
            step_name=plan.step_name,
            pipeline_position=plan.pipeline_position,
            step_scope_id=plan.step_scope_id,
            artifact_kind=output_plan.kind.value,
        )


class PlannedArtifactPathsAuthority:
    """Return every compiler-planned memory path for one artifact output."""

    @staticmethod
    def paths(output_plan: ArtifactOutputPlan) -> frozenset[str]:
        paths = {output_plan.path}
        if output_plan.paths_by_group:
            paths.update(output_plan.paths_by_group.values())
        return frozenset(paths)


class MaterializationRecordSortKeyAuthority:
    """Sort runtime artifact records by compiler group order."""

    @staticmethod
    def key(
        record: StoredRuntimeValue,
        output_plan: ArtifactOutputPlan,
    ) -> tuple[int, str]:
        group_keys = output_plan.group_keys
        if group_keys is None:
            group_keys = (None,)
        group_order = {
            group_key: index
            for index, group_key in enumerate(group_keys)
        }
        group_key = record.key.scope.group_key
        if group_key in group_order:
            group_index = group_order[group_key]
        else:
            group_index = len(group_order)
        if group_key is None:
            group_name = ""
        else:
            group_name = str(group_key)
        return group_index, group_name


class ActualMaterializationRecordsAuthority:
    """Resolve records actually produced for one planned output."""

    @staticmethod
    def records(
        *,
        context: "ProcessingContext",
        plan: FunctionStepExecutionPlan,
        output_plan: ArtifactOutputPlan,
    ) -> tuple[StoredRuntimeValue, ...]:
        store = require_runtime_value_store(context, owner_name="context")
        planned_paths = PlannedArtifactPathsAuthority.paths(output_plan)
        records = tuple(
            record
            for record in store.find(
                name=output_plan.name,
                kind=output_plan.kind,
                axis_id=plan.axis_id,
            )
            if (
                record.backend == Backend.MEMORY.value
                and record.path in planned_paths
            )
        )
        if not records:
            raise RuntimeError(
                f"Missing RuntimeValueStore record for planned artifact materialization "
                f"'{output_plan.name}' ({output_plan.kind.value}) on axis "
                f"'{plan.axis_id}'."
            )
        return tuple(
            sorted(
                records,
                key=lambda record: MaterializationRecordSortKeyAuthority.key(
                    record,
                    output_plan,
                ),
            )
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
    backends = backend_plan.backends

    if not backends:
        return

    analysis_output_dir = plan.artifact_analysis_output_dir
    images_dir = plan.artifact_images_dir

    filemanager._materialization_context = {"images_dir": images_dir}

    for output_key, output_plan in plan.artifact_outputs.items():
        if output_plan.materialization is None and output_plan.kind is ArtifactKind.SPECIAL:
            continue

        records = ActualMaterializationRecordsAuthority.records(
            context=context,
            plan=plan,
            output_plan=output_plan,
        )
        for record in records:
            dict_key = record.key.scope.group_key

            filemanager.ensure_directory(
                Path(record.path).parent, record.backend
            )
            data = record.value.data
            mat_spec = resolve_artifact_materialization_spec(
                output_plan,
                record.value,
            )
            if mat_spec is None:
                continue

            output_descriptor = AnalysisOutputDescriptorAuthority.build(
                output_key,
                plan,
                dict_key,
                context,
                artifact_path=record.path,
                record=record,
            )
            analysis_path = analysis_output_dir / output_descriptor.filename
            materialize(
                mat_spec,
                data,
                str(analysis_path),
                filemanager,
                backends,
                BackendKwargsWithStreamIdentityAuthority.build(
                    backend_plan,
                    output_descriptor.artifact_source_component_metadata,
                    ArtifactStreamIdentityAuthority.build(
                        plan,
                        output_key,
                        output_plan,
                    ),
                    context,
                    images_dir,
                ),
                context=context,
            )
