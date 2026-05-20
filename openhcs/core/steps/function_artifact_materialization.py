"""Artifact materialization helpers for FunctionStep."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta
from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.artifact_materialization_policy import (
    resolve_artifact_materialization_spec,
)
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
    require_runtime_value_store,
)
from openhcs.core.registry_strategies import str_enum_member_with_payload
from openhcs.core.runtime_semantics import coerce_enum
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from nominal_refactor_advisor.descriptor_algebra import AliasProperty


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
class MaterializerInputLocation:
    """Resolved path universe for one materializer input source."""

    directory: Path
    backend: str


MaterializerInputLocationResolver = Callable[
    [FunctionStepExecutionPlan],
    MaterializerInputLocation,
]


def step_input_materializer_location(
    plan: FunctionStepExecutionPlan,
) -> MaterializerInputLocation:
    """Resolve materializer inputs from the step input universe."""
    return MaterializerInputLocation(plan.input_dir, plan.read_backend)


def step_output_materializer_location(
    plan: FunctionStepExecutionPlan,
) -> MaterializerInputLocation:
    """Resolve materializer inputs from the step output universe."""
    return MaterializerInputLocation(plan.output_dir, Backend.MEMORY.value)


class MaterializerInputKind(str, Enum):
    """Supported extra input payload shapes for artifact materializers."""

    IMAGE_SLICES = "image_slices"


class MaterializerInputSource(str, Enum):
    """Source universe for extra inputs passed to artifact materializers."""

    def __new__(
        cls,
        value: str,
        location_resolver: MaterializerInputLocationResolver,
    ):
        return str_enum_member_with_payload(
            cls,
            value,
            payload_attribute="_location_resolver",
            payload=location_resolver,
        )

    STEP_INPUT = ("step_input", step_input_materializer_location)
    STEP_OUTPUT = ("step_output", step_output_materializer_location)
    location_resolver = AliasProperty[MaterializerInputLocationResolver](
        "_location_resolver"
    )

    def location_for(
        self,
        plan: FunctionStepExecutionPlan,
    ) -> MaterializerInputLocation:
        """Resolve this source against the compiled step execution plan."""
        return self.location_resolver(plan)


@dataclass(frozen=True, slots=True)
class MaterializerInputSpec:
    """Validated descriptor for one materializer-declared input."""

    name: str
    kind: MaterializerInputKind
    source: MaterializerInputSource
    group_by: str | None = None

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Mapping[str, Any],
    ) -> "MaterializerInputSpec":
        """Build a typed materializer input descriptor from legacy options."""
        kind = coerce_enum(
            MaterializerInputKind,
            raw.get("kind"),
            f"Materialization input {name!r} kind",
        )
        source = coerce_enum(
            MaterializerInputSource,
            raw.get("source"),
            f"Materialization input {name!r} source",
        )
        group_by = raw.get("group_by")
        return cls(
            name=str(name),
            kind=kind,
            source=source,
            group_by=None if group_by is None else str(group_by),
        )

    def require_image_slices(self) -> None:
        """Fail loudly when a materializer asks for an unsupported input kind."""
        if self.kind is not MaterializerInputKind.IMAGE_SLICES:
            supported = ", ".join(repr(kind.value) for kind in MaterializerInputKind)
            raise ValueError(
                f"Unsupported materialization input kind for {self.name!r}: "
                f"{self.kind.value!r}. Supported kinds: {supported}."
            )


def _build_analysis_filename(
    output_key: str,
    plan: FunctionStepExecutionPlan,
    dict_key: str | None = None,
    context: Any = None,
    artifact_path: str | None = None,
) -> str:
    """Build an analysis result filename from the first matching image path."""
    memory_paths = plan.get_paths_for_axis(plan.output_dir, Backend.MEMORY.value)

    if not memory_paths:
        if dict_key is not None and artifact_path is not None:
            return f"{Path(artifact_path).stem}.roi.zip"
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

        input_spec = MaterializerInputSpec.from_mapping(input_name, input_desc)
        input_spec.require_image_slices()
        location = input_spec.source.location_for(plan)

        paths = plan.get_paths_for_axis(location.directory, location.backend)
        if dict_key is not None:
            paths = _filter_group_materializer_paths(
                input_spec=input_spec,
                paths=paths,
                dict_key=dict_key,
                plan=plan,
                context=context,
            )

        if not paths:
            raise ValueError(
                f"Materialization input '{input_name}' resolved to 0 paths "
                f"(source={input_spec.source.value}, dir={location.directory}, "
                f"backend={location.backend}, group={dict_key})."
            )

        resolved[input_name] = filemanager.load_batch(paths, location.backend)

    return resolved


def _filter_group_materializer_paths(
    *,
    input_spec: MaterializerInputSpec,
    paths: list[str],
    dict_key: str,
    plan: FunctionStepExecutionPlan,
    context: Any,
) -> list[str]:
    """Filter materializer input paths to the current dict/group invocation."""
    group_by_key = input_spec.group_by
    if group_by_key is None:
        group_by_key = plan.group_by_value

    if group_by_key is None:
        raise ValueError(
            f"Cannot resolve materialization input '{input_spec.name}' for group '{dict_key}': "
            "no group_by specified in the input spec and the step has no group_by."
        )
    if context is None:
        raise ValueError(
            f"Cannot resolve materialization input '{input_spec.name}' for group '{dict_key}': "
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

            filename = _build_analysis_filename(
                output_key,
                plan,
                dict_key,
                context,
                artifact_path=record.path,
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
