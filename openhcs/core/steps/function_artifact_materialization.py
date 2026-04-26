"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactOutputPlan
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    StoredRuntimeValue,
    require_runtime_value_store,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan


logger = logging.getLogger(__name__)


def _build_analysis_filename(
    output_key: str,
    plan: FunctionStepExecutionPlan,
    dict_key: str | None = None,
    context: Any = None,
) -> str:
    """Build an analysis result filename from the first matching image path."""
    memory_paths = plan.get_paths_for_axis(plan.output_dir, Backend.MEMORY.value)

    if not memory_paths:
        return f"{plan.axis_id}_{output_key}_step{plan.pipeline_position}.roi.zip"

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
    return f"{base_filename}_{output_key}_step{plan.pipeline_position}.roi.zip"


def _resolve_materializer_inputs(
    mat_spec: Any,
    *,
    dict_key: str | None,
    plan: FunctionStepExecutionPlan,
    filemanager: Any,
    context: Any,
) -> dict[str, Any]:
    """Resolve materializer-declared image inputs for one artifact invocation."""
    options = getattr(mat_spec, "options", {}) or {}
    inputs_spec = options.get("inputs") or {}
    if not inputs_spec:
        return {}
    if not isinstance(inputs_spec, dict):
        raise ValueError(
            f"MaterializationSpec.options['inputs'] must be a dict, got {type(inputs_spec)}"
        )

    resolved: dict[str, Any] = {}
    for input_name, input_desc in inputs_spec.items():
        if not isinstance(input_desc, dict):
            raise ValueError(
                f"Materialization input '{input_name}' must be a dict, got {type(input_desc)}"
            )

        kind = input_desc.get("kind")
        if kind != "image_slices":
            raise ValueError(
                f"Unsupported materialization input kind for '{input_name}': {kind}. "
                "Supported kinds: 'image_slices'."
            )

        source = input_desc.get("source")
        if source == "step_input":
            source_dir = plan.input_dir
            source_backend = plan.read_backend
        elif source == "step_output":
            source_dir = plan.output_dir
            source_backend = Backend.MEMORY.value
        else:
            raise ValueError(
                f"Unsupported materialization input source for '{input_name}': {source}. "
                "Supported sources: 'step_input', 'step_output'."
            )

        paths = plan.get_paths_for_axis(source_dir, source_backend)
        if dict_key is not None:
            paths = _filter_group_materializer_paths(
                input_name=input_name,
                input_desc=input_desc,
                paths=paths,
                dict_key=dict_key,
                plan=plan,
                context=context,
            )

        if not paths:
            raise ValueError(
                f"Materialization input '{input_name}' resolved to 0 paths "
                f"(source={source}, dir={source_dir}, backend={source_backend}, group={dict_key})."
            )

        resolved[input_name] = filemanager.load_batch(paths, source_backend)

    return resolved


def _filter_group_materializer_paths(
    *,
    input_name: str,
    input_desc: Mapping[str, Any],
    paths: list[str],
    dict_key: str,
    plan: FunctionStepExecutionPlan,
    context: Any,
) -> list[str]:
    """Filter materializer input paths to the current dict/group invocation."""
    group_by_key = input_desc.get("group_by")
    if group_by_key is None:
        group_by_key = plan.group_by_value

    if group_by_key is None:
        raise ValueError(
            f"Cannot resolve materialization input '{input_name}' for group '{dict_key}': "
            "no group_by specified in the input spec and the step has no group_by."
        )
    if context is None:
        raise ValueError(
            f"Cannot resolve materialization input '{input_name}' for group '{dict_key}': "
            "context is required for filename parsing."
        )

    parser = context.microscope_handler.parser
    return [
        path
        for path in paths
        if (
            (metadata := parser.parse_filename(Path(path).name))
            and str(metadata.get(group_by_key)) == str(dict_key)
        )
    ]


def _resolve_group_artifact_path(
    output_plan: ArtifactOutputPlan,
    dict_key: str | None,
) -> str:
    """Resolve the memory VFS path for one artifact output/group pair."""
    paths_by_group = output_plan.paths_by_group or {}
    if dict_key is None:
        return paths_by_group.get(None, output_plan.path)
    if dict_key in paths_by_group:
        return paths_by_group[dict_key]
    if None in paths_by_group:
        return paths_by_group[None]

    from openhcs.core.pipeline.path_planner import PipelinePathPlanner

    return PipelinePathPlanner.build_dict_pattern_path(output_plan.path, dict_key)


def _resolve_materialization_record(
    *,
    context: Any,
    plan: FunctionStepExecutionPlan,
    output_plan: ArtifactOutputPlan,
    dict_key: str | None,
) -> StoredRuntimeValue:
    """Resolve the typed runtime record for one planned artifact materialization."""
    channel_path = _resolve_group_artifact_path(output_plan, dict_key)
    store = require_runtime_value_store(
        context,
        owner_name="context",
    )
    query = RuntimeArtifactQuery.by_location(
        name=output_plan.name,
        kind=output_plan.kind,
        axis_id=plan.axis_id,
        location=RuntimeArtifactLocation(
            path=channel_path,
            backend=Backend.MEMORY.value,
        ),
    )
    record = store.resolve(
        query,
        purpose="planned artifact materialization",
    )
    if not context.filemanager.exists(record.path, record.backend):
        raise RuntimeError(
            f"RuntimeValueStore has record for artifact '{output_plan.name}' at "
            f"'{record.path}' ({record.backend}), but the VFS payload is missing."
        )
    return record


def materialize_artifact_outputs(
    filemanager: Any,
    plan: FunctionStepExecutionPlan,
    backend: str,
    context: Any,
) -> None:
    """Materialize planned artifact outputs to persistent and streaming backends."""
    from openhcs.processing.materialization import materialize

    backends = [backend]
    backend_kwargs: dict[str, dict[str, Any]] = {backend: {}}

    for config in plan.streaming_configs:
        backends.append(config.backend.value)
        backend_kwargs[config.backend.value] = config.get_streaming_kwargs(context)

    analysis_output_dir = plan.artifact_analysis_output_dir
    images_dir = plan.artifact_images_dir

    for kwargs in backend_kwargs.values():
        kwargs["images_dir"] = images_dir
        kwargs["source"] = plan.step_name

    filemanager._materialization_context = {"images_dir": images_dir}

    for output_key, output_plan in plan.artifact_outputs.items():
        mat_spec = output_plan.materialization
        if not mat_spec:
            continue

        group_keys = output_plan.group_keys or [None]

        for dict_key in group_keys:
            record = _resolve_materialization_record(
                context=context,
                plan=plan,
                output_plan=output_plan,
                dict_key=dict_key,
            )

            filemanager.ensure_directory(
                Path(record.path).parent, record.backend
            )
            data = filemanager.load(record.path, record.backend)

            filename = _build_analysis_filename(
                output_key, plan, dict_key, context
            )
            analysis_path = analysis_output_dir / filename
            extra_inputs = _resolve_materializer_inputs(
                mat_spec,
                dict_key=dict_key,
                plan=plan,
                filemanager=filemanager,
                context=context,
            )
            materialize(
                mat_spec,
                data,
                str(analysis_path),
                filemanager,
                backends,
                backend_kwargs,
                context=context,
                extra_inputs=extra_inputs,
            )
