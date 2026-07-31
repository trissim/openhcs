from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.demos import official30_lab_meeting as demo
from openhcs.core.config import PipelineConfig
from openhcs.core.input_workspace import (
    InputWorkspacePreparationRequest,
    InputWorkspacePreparationResult,
)
from openhcs.core.source_binding_workspace import (
    SourceBindingWorkspaceMaterialization,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution


def test_official30_lab_meeting_contributions_import_manifest_owned_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    examples_root = tmp_path / "cellprofiler_examples"
    preparation_requests: list[InputWorkspacePreparationRequest] = []
    target_steps = {
        "ExampleCometAssay.cppipe": "OverlayOutlines",
        "ExampleWoundHealing.cppipe": "IdentifyPrimaryObjects",
    }
    for case_name, cppipe_name in (
        ("ExampleCometAssay", "ExampleCometAssay.cppipe"),
        ("ExampleWoundHealing", "ExampleWoundHealing.cppipe"),
    ):
        case_root = examples_root / case_name
        (case_root / "images").mkdir(parents=True)
        (case_root / cppipe_name).write_text("CellProfiler Pipeline", encoding="utf-8")
        (case_root / "images" / f"{case_name}.tif").write_bytes(b"image")
        (case_root / "images" / "openhcs_metadata.json").write_text(
            "stale",
            encoding="utf-8",
        )

    def fake_prepare(
        request: InputWorkspacePreparationRequest,
    ) -> InputWorkspacePreparationResult:
        preparation_requests.append(request)
        assert request.selected_pipeline_path is not None
        assert request.workspace_root is not None
        execution_plate_path = request.workspace_root / "returned_execution"
        execution_plate_path.mkdir(parents=True)
        (execution_plate_path / "openhcs_metadata.json").write_text(
            "prepared",
            encoding="utf-8",
        )
        target_name = target_steps[request.selected_pipeline_path.name]
        return InputWorkspacePreparationResult(
            original_source_root=request.selected_path,
            execution_plate_path=execution_plate_path,
            pipeline_path=request.selected_pipeline_path,
            pipeline_steps=[
                FunctionStep(name="Preparation", func=percentile_normalize),
                FunctionStep(name=target_name, func=percentile_normalize),
            ],
            pipeline_config=PipelineConfig(),
            materialization=SourceBindingWorkspaceMaterialization(
                source_root=request.selected_path,
                workspace_root=execution_plate_path,
                metadata_path=execution_plate_path / "openhcs_metadata.json",
                plane_mappings={},
                artifact_mappings={},
                source_metadata={
                    "A01_s001_w1_z001_t001.tif": {"well": "A01"},
                },
            ),
        )

    monkeypatch.setenv("CELLPROFILER_EXAMPLES_ROOT", str(examples_root))
    monkeypatch.setenv("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "0")
    monkeypatch.setattr(demo, "prepare_cellprofiler_input_workspace", fake_prepare)

    contributions = demo.official30_lab_meeting_demo_contributions(
        session_root=tmp_path / "session",
    )

    assert [contribution.demo_id for contribution in contributions] == [
        "official30_examplecometassay",
        "official30_examplewoundhealing",
    ]
    assert [contribution.title for contribution in contributions] == [
        "DNA-damage comet morphology and tail intensity",
        "Collective-migration wound closure",
    ]
    assert all(contribution.biological_question for contribution in contributions)
    assert all(
        isinstance(contribution, PipelineDemoContribution)
        for contribution in contributions
    )
    assert len(preparation_requests) == 2
    expected_axis_filters = ("A01", "A01")
    for contribution, expected_axis_filter in zip(
        contributions,
        expected_axis_filters,
        strict=True,
    ):
        streamed = [
            step
            for step in contribution.pipeline_steps
            if step.napari_streaming_config.enabled
        ]
        assert [step.name for step in streamed] == [
            contribution.presentation_identity.step_name
        ]
        assert (
            contribution.pipeline_config.well_filter_config.well_filter
            == expected_axis_filter
        )
        assert contribution.pipeline_config.path_planning_config.well_filter == 0
        assert contribution.pipeline_config.materialize_runtime_artifacts is True
        assert contribution.prepare is None
        request = preparation_requests.pop(0)
        assert request.selected_path.name == "images"
        assert request.selected_pipeline_path is not None
        assert request.selected_pipeline_path.suffix == ".cppipe"
        assert contribution.plate_path == request.workspace_root / "returned_execution"
        assert (contribution.plate_path / "openhcs_metadata.json").is_file()
        assert not tuple(contribution.plate_path.glob("*.tif"))

    comet, wound = contributions
    assert comet.presentation_identity.output_kind == "main"
    assert comet.presentation_identity.output_key == "CometOutline"
    assert comet.presentation_identity.projection_key == "main"
    assert comet.presentation_identity.artifact_kind == "image"
    assert wound.presentation_identity.output_kind == "artifact"
    assert wound.presentation_identity.output_key == "Tissue"
    assert wound.presentation_identity.artifact_kind == "object_labels"


@pytest.mark.parametrize(
    ("presentation", "message"),
    (
        (
            {
                "artifact_kind": "unknown",
                "output_key": "Result",
                "output_kind": "artifact",
                "step_name": "Measure",
            },
            "Unknown artifact type",
        ),
        (
            {
                "artifact_kind": "measurements",
                "output_key": "Measurements",
                "output_kind": "artifact",
                "step_name": "Measure",
            },
            "cannot be a Measurements artifact",
        ),
        (
            {
                "artifact_kind": "image",
                "output_key": "Result",
                "output_kind": "unknown",
                "step_name": "Measure",
            },
            "unsupported output kind",
        ),
    ),
)
def test_official30_presentation_declaration_rejects_invalid_semantics(
    presentation: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        demo.Official30PresentationDeclaration.from_payload(presentation)
