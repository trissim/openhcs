"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta
from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.artifact_materialization_policy import (
    resolve_artifact_materialization_spec,
)
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
    require_runtime_value_store,
)
from openhcs.core.runtime_values import image_payload_metadata
from openhcs.core.source_matching import source_component_metadata_value
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan


logger = logging.getLogger(__name__)


class ArtifactMaterializationTargetPlan(ABC, metaclass=AutoRegisterMeta):
    """Nominal target policy for artifact materialization destinations."""

    __registry_key__ = "target_key"
    __skip_if_no_key__ = True
    target_key: ClassVar[str | None] = None

    def backend_kwargs(
        self,
        plan: FunctionStepExecutionPlan,
        context: Any,
    ) -> dict[str, dict[str, Any]]:
        backend_kwargs = self.persistent_backend_kwargs()
        for config in plan.streaming_configs:
            backend_kwargs[config.backend.value] = config.get_streaming_kwargs(context)
        return backend_kwargs

    @abstractmethod
    def persistent_backend_kwargs(self) -> dict[str, dict[str, Any]]:
        """Return persistent materialization backends owned by this policy."""


@dataclass(frozen=True, slots=True)
class PersistentArtifactMaterializationTargetPlan(ArtifactMaterializationTargetPlan):
    """Target policy for persistent files plus any enabled viewer streams."""

    target_key = "persistent"
    backend: str

    def persistent_backend_kwargs(self) -> dict[str, dict[str, Any]]:
        return {self.backend: {}}


class StreamingOnlyArtifactMaterializationTargetPlan(ArtifactMaterializationTargetPlan):
    """Target policy for viewer streams with no persistent artifact files."""

    target_key = "streaming_only"

    def persistent_backend_kwargs(self) -> dict[str, dict[str, Any]]:
        return {}


@dataclass(frozen=True, slots=True)
class ArtifactAnalysisOutputDescriptor:
    """Filename plus source component metadata for an artifact materialization."""

    filename: str
    component_metadata: Mapping[str, Any] | None


class AnalysisOutputDescriptorAuthority:
    """Own analysis artifact filenames and their source component metadata."""

    @classmethod
    def build(
        cls,
        output_key: str,
        plan: FunctionStepExecutionPlan,
        dict_key: str | None = None,
        context: Any = None,
        artifact_path: str | None = None,
        record: StoredRuntimeValue | None = None,
    ) -> ArtifactAnalysisOutputDescriptor:
        """Build analysis output identity from the artifact's source metadata."""
        source_path = cls.source_path_for_record(record)
        if source_path is not None:
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{Path(source_path).stem}_{output_key}_step{plan.pipeline_position}.roi.zip",
                component_metadata=cls.source_component_metadata_for_record(
                    record,
                    fallback=cls.component_metadata_for_path(context, source_path),
                ),
            )

        if artifact_path is not None:
            component_metadata = cls.component_metadata_for_path(context, artifact_path)
            if component_metadata is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    component_metadata=component_metadata,
                )

        memory_paths = plan.get_paths_for_axis(plan.output_dir, Backend.MEMORY.value)

        if not memory_paths:
            if dict_key is not None and artifact_path is not None:
                return ArtifactAnalysisOutputDescriptor(
                    filename=f"{Path(artifact_path).stem}.roi.zip",
                    component_metadata=None,
                )
            return ArtifactAnalysisOutputDescriptor(
                filename=f"{plan.axis_id}_{output_key}_step{plan.pipeline_position}.roi.zip",
                component_metadata=None,
            )

        if dict_key and context:
            parser = context.microscope_handler.parser
            filtered_paths = []
            for path in memory_paths:
                metadata = parser.parse_filename(Path(path).name)
                if metadata and str(metadata.get("channel")) == str(dict_key):
                    filtered_paths.append(path)

            if filtered_paths:
                memory_paths = filtered_paths

        base_filename = Path(memory_paths[0]).stem
        return ArtifactAnalysisOutputDescriptor(
            filename=f"{base_filename}_{output_key}_step{plan.pipeline_position}.roi.zip",
            component_metadata=cls.component_metadata_for_path(context, memory_paths[0]),
        )

    @staticmethod
    def source_path_for_record(record: StoredRuntimeValue | None) -> str | None:
        """Return the source image path carried by a runtime artifact payload."""
        if record is None:
            return None
        metadata = image_payload_metadata(record.value.data)
        if metadata.source_path:
            return str(metadata.source_path)
        for source_path in metadata.channel_source_paths:
            if source_path:
                return str(source_path)
        return None

    @staticmethod
    def source_component_metadata_for_record(
        record: StoredRuntimeValue | None,
        *,
        fallback: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        """Return source component metadata carried by a runtime artifact payload."""
        candidate: Mapping[str, Any] | None = None
        if record is None:
            return fallback
        payload_metadata = image_payload_metadata(record.value.data)
        if payload_metadata.source_component_metadata is not None:
            candidate = payload_metadata.source_component_metadata
        else:
            for component_metadata in payload_metadata.channel_source_component_metadata:
                if component_metadata is not None:
                    candidate = component_metadata
                    break
        return AnalysisOutputDescriptorAuthority.merge_source_component_metadata(
            candidate,
            fallback,
        )

    @staticmethod
    def merge_source_component_metadata(
        primary: Mapping[str, Any] | None,
        fallback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
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
    def component_metadata_for_path(
        context: Any,
        path: str | Path | None,
    ) -> Mapping[str, Any] | None:
        """Return microscope component metadata for a source path when available."""
        if context is None or path is None:
            return None
        return context.microscope_handler.parser.parse_filename(Path(path).name)


def _backend_kwargs_with_component_metadata(
    backend_kwargs: Mapping[str, Mapping[str, Any]],
    component_metadata: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Attach source component metadata to each materialization backend."""
    if component_metadata is None:
        return {backend: dict(kwargs) for backend, kwargs in backend_kwargs.items()}
    return {
        backend: {**kwargs, "component_metadata": dict(component_metadata)}
        for backend, kwargs in backend_kwargs.items()
    }


def _planned_artifact_paths(output_plan: ArtifactOutputPlan) -> frozenset[str]:
    """Return every compiler-planned memory path for one artifact output."""
    paths = {output_plan.path}
    paths.update((output_plan.paths_by_group or {}).values())
    return frozenset(paths)


def _sort_key_for_record(
    record: StoredRuntimeValue,
    output_plan: ArtifactOutputPlan,
) -> tuple[int, str]:
    group_order = {
        group_key: index
        for index, group_key in enumerate(output_plan.group_keys or (None,))
    }
    group_key = record.key.scope.group_key
    return (
        group_order.get(group_key, len(group_order)),
        "" if group_key is None else str(group_key),
    )


def _actual_materialization_records(
    *,
    context: Any,
    plan: FunctionStepExecutionPlan,
    output_plan: ArtifactOutputPlan,
) -> tuple[StoredRuntimeValue, ...]:
    """Resolve records actually produced for one planned output."""
    store = require_runtime_value_store(context, owner_name="context")
    planned_paths = _planned_artifact_paths(output_plan)
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
            key=lambda record: _sort_key_for_record(record, output_plan),
        )
    )


def materialize_artifact_outputs(
    filemanager: Any,
    plan: FunctionStepExecutionPlan,
    target_plan: ArtifactMaterializationTargetPlan,
    context: Any,
) -> None:
    """Materialize planned artifact outputs to persistent and streaming backends."""
    from openhcs.processing.materialization import materialize

    backend_kwargs = target_plan.backend_kwargs(plan, context)
    backends = list(backend_kwargs)

    if not backends:
        return

    analysis_output_dir = plan.artifact_analysis_output_dir
    images_dir = plan.artifact_images_dir

    for kwargs in backend_kwargs.values():
        kwargs["images_dir"] = images_dir
        kwargs["source"] = plan.step_name

    filemanager._materialization_context = {"images_dir": images_dir}

    for output_key, output_plan in plan.artifact_outputs.items():
        if output_plan.materialization is None and output_plan.kind is ArtifactKind.SPECIAL:
            continue

        records = _actual_materialization_records(
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
                _backend_kwargs_with_component_metadata(
                    backend_kwargs,
                    output_descriptor.component_metadata,
                ),
                context=context,
            )
