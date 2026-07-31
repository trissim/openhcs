"""Source-backed Official30 contributions for the visible lab-meeting sprint.

The benchmark manifest owns which cases opt into this showcase and their exact
biological presentation target. Pipeline bodies remain owned by their ``.cppipe``
sources and are imported through the public CellProfiler interoperability API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from polystore.streaming.identity import StreamProducerIdentity

from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    load_comparison_cases,
)
from benchmark.contracts.comparison_manifest import (
    ComparisonManifest,
    JSONValue,
    ManifestCasePayload,
)
from openhcs.constants.constants import AllComponents
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import ArtifactType, MeasurementsArtifactType
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyWellFilterConfig,
)
from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
)
from openhcs.core.source_matching import source_component_metadata_values
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.plate_workspace import (
    prepare_cellprofiler_input_workspace,
)
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL30_MANIFEST_PATH = (
    REPOSITORY_ROOT / "benchmark/manifests/official30_portable_axis1.json"
)


def _required_text(mapping: Mapping[str, JSONValue], key: str) -> str:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise ValueError(
            f"Official30 lab-meeting declaration is missing {key!r}."
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Official30 lab-meeting field {key!r} must be text.")
    return value


@dataclass(frozen=True, slots=True)
class Official30PresentationDeclaration:
    """Typed serialization owner for one manifest-declared visual target."""

    output_kind: str
    output_key: str
    artifact_type: type[ArtifactType]
    step_name: str

    @classmethod
    def from_payload(cls, payload: JSONValue) -> "Official30PresentationDeclaration":
        """Validate and type one manifest presentation object."""
        if not isinstance(payload, Mapping):
            raise ValueError("Official30 lab-meeting presentation must be an object.")
        output_kind = _required_text(payload, "output_kind")
        if output_kind not in (
            AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
            FunctionStepOutputProducerIdentityRequest.ARTIFACT_OUTPUT_KIND,
        ):
            raise ValueError(
                f"Official30 visual target has unsupported output kind {output_kind!r}."
            )
        artifact_type = ArtifactType.coerce(_required_text(payload, "artifact_kind"))
        if artifact_type is MeasurementsArtifactType:
            raise ValueError(
                "Official30 visual target cannot be a Measurements artifact."
            )
        return cls(
            output_kind=output_kind,
            output_key=_required_text(payload, "output_key"),
            artifact_type=artifact_type,
            step_name=_required_text(payload, "step_name"),
        )

    def producer_identity(self) -> StreamProducerIdentity:
        """Build the declaration-time producer identity for this target."""
        projection_key = (
            AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND
            if self.output_kind == AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND
            else self.output_key
        )
        return StreamProducerIdentity.pipeline_output(
            output_kind=self.output_kind,
            output_key=self.output_key,
            projection_key=projection_key,
            step_name=self.step_name,
            pipeline_position=None,
            artifact_kind=self.artifact_type.require_value(),
        )


@dataclass(frozen=True, slots=True)
class Official30LabMeetingDeclaration:
    """Typed Official30 benchmark case plus its optional demo extension."""

    comparison_case: CellProfilerComparisonCase
    title: str
    biological_question: str
    presentation: Official30PresentationDeclaration

    @classmethod
    def from_payload(
        cls,
        comparison_case: CellProfilerComparisonCase,
        payload: JSONValue,
    ) -> "Official30LabMeetingDeclaration":
        """Validate one manifest extension against its typed benchmark case."""
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Official30 case {comparison_case.name!r} lab_meeting_demo "
                "must be an object."
            )
        try:
            presentation = payload["presentation"]
        except KeyError as exc:
            raise ValueError(
                f"Official30 case {comparison_case.name!r} has no presentation "
                "declaration."
            ) from exc
        return cls(
            comparison_case=comparison_case,
            title=_required_text(payload, "title"),
            biological_question=_required_text(payload, "biological_question"),
            presentation=Official30PresentationDeclaration.from_payload(presentation),
        )


def _manifest_declarations() -> tuple[Official30LabMeetingDeclaration, ...]:
    """Load typed benchmark cases and their validated demo extensions."""
    comparison_cases = load_comparison_cases(OFFICIAL30_MANIFEST_PATH)
    manifest = ComparisonManifest.load(
        OFFICIAL30_MANIFEST_PATH,
        materialize_roots=False,
    )
    try:
        raw_cases = manifest.payload["cases"]
    except KeyError as exc:
        raise ValueError("Official30 manifest must declare cases.") from exc
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("Official30 manifest cases must be a sequence.")

    declarations: list[Official30LabMeetingDeclaration] = []
    for raw_case, comparison_case in zip(
        raw_cases,
        comparison_cases,
        strict=True,
    ):
        if not isinstance(raw_case, Mapping):
            raise ValueError(
                f"Official30 manifest case must be an object: {raw_case!r}"
            )
        case_payload: ManifestCasePayload = raw_case
        case_name = _required_text(case_payload, "name")
        if case_name != comparison_case.name:
            raise ValueError(
                "Official30 typed/raw manifest case order diverged: "
                f"{case_name!r} != {comparison_case.name!r}."
            )
        if "lab_meeting_demo" not in case_payload:
            continue
        declarations.append(
            Official30LabMeetingDeclaration.from_payload(
                comparison_case,
                case_payload["lab_meeting_demo"],
            )
        )
    if not declarations:
        raise ValueError("Official30 manifest declares no lab-meeting demos.")
    return tuple(declarations)


def _stream_only_declared_step(
    steps: Sequence[FunctionStep],
    identity: StreamProducerIdentity,
) -> tuple[FunctionStep, ...]:
    matches = tuple(
        index for index, step in enumerate(steps) if step.name == identity.step_name
    )
    if len(matches) != 1:
        raise ValueError(
            "Official30 lab-meeting presentation step must be unique; observed "
            f"{len(matches)} for {identity.step_name!r}."
        )
    target_index = matches[0]
    streamed_steps = deepcopy(tuple(steps))
    for index, step in enumerate(streamed_steps):
        step.napari_streaming_config = LazyNapariStreamingConfig(
            enabled=index == target_index,
            persistent=True,
            port=5888,
        )
    return streamed_steps


def _materialized_well_labels(
    prepared: InputWorkspacePreparationResult,
) -> str | list[str]:
    """Project exact execution wells from the prepared workspace authority."""

    if prepared.materialization is None:
        raise ValueError(
            "Official30 lab-meeting inputs must materialize a source-binding "
            "workspace with exact component metadata."
        )
    labels = sorted(
        {
            value
            for metadata in prepared.materialization.source_metadata.values()
            for value in source_component_metadata_values(
                metadata,
                AllComponents.WELL,
            )
        }
    )
    if not labels:
        raise ValueError(
            "Official30 lab-meeting workspace declares no exact well labels."
        )
    return labels[0] if len(labels) == 1 else labels


def official30_lab_meeting_demo_contributions(
    *,
    session_root: Path,
) -> tuple[PipelineDemoContribution, ...]:
    """Import manifest-selected Official30 stories without copied pipeline code."""

    resolved_session_root = session_root.expanduser().resolve()
    contributions: list[PipelineDemoContribution] = []
    for declaration in _manifest_declarations():
        comparison_case = declaration.comparison_case
        case_name = comparison_case.name
        cppipe_path = comparison_case.cppipe_path.expanduser().resolve()
        source_dataset = comparison_case.dataset_path.expanduser().resolve()
        if not cppipe_path.is_file() or not source_dataset.is_dir():
            raise FileNotFoundError(
                f"Official30 lab-meeting source is missing for {case_name!r}: "
                f"{cppipe_path}, {source_dataset}."
            )
        well_filter_config = comparison_case.well_filter_config
        if well_filter_config is None:
            raise ValueError(
                f"Official30 lab-meeting case {case_name!r} must declare a "
                "benchmark well filter."
            )
        identity = declaration.presentation.producer_identity()
        prepared = prepare_cellprofiler_input_workspace(
            InputWorkspacePreparationRequest(
                selected_path=source_dataset,
                selected_pipeline_path=cppipe_path,
                workspace_root=resolved_session_root / "plates" / declaration.title,
            )
        )
        if prepared.pipeline_import_error is not None:
            raise ValueError(prepared.pipeline_import_error.message)
        if prepared.pipeline_steps is None or prepared.pipeline_config is None:
            raise RuntimeError(
                f"Official30 input preparation produced no pipeline for {case_name!r}."
            )
        pipeline_steps = prepared.pipeline_steps
        pipeline_config = prepared.pipeline_config
        plate_path = prepared.execution_plate_path
        output_root = (
            resolved_session_root / "outputs" / f"official30_{case_name.casefold()}"
        )
        pipeline_config = replace(
            pipeline_config,
            well_filter_config=LazyWellFilterConfig(
                well_filter=_materialized_well_labels(prepared),
                well_filter_mode=well_filter_config.well_filter_mode,
            ),
            path_planning_config=LazyPathPlanningConfig(
                well_filter=0,
                global_output_folder=output_root,
            ),
            materialization_results_path=output_root / "results",
            materialize_runtime_artifacts=True,
        )

        contributions.append(
            PipelineDemoContribution(
                demo_id=f"official30_{case_name.casefold()}",
                title=declaration.title,
                plate_path=plate_path,
                pipeline_config=pipeline_config,
                pipeline_steps=_stream_only_declared_step(
                    pipeline_steps,
                    identity,
                ),
                presentation_identity=identity,
                biological_question=declaration.biological_question,
            )
        )
    return tuple(contributions)
