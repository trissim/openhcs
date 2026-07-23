from __future__ import annotations

from dataclasses import replace

import pytest
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution


def _identity(
    *,
    step_name: str = "Normalize",
    output_key: str = "main",
) -> StreamProducerIdentity:
    return StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key=output_key,
        projection_key="main",
        step_name=step_name,
        pipeline_position=None,
    )


def _contribution(tmp_path, **changes) -> PipelineDemoContribution:
    values = {
        "demo_id": "normalize",
        "title": "Normalize one field",
        "plate_path": tmp_path / "plate",
        "pipeline_config": PipelineConfig(),
        "pipeline_steps": (FunctionStep(name="Normalize", func=percentile_normalize),),
        "presentation_identity": _identity(),
        "supporting_presentation_identities": (_identity(output_key="normalized"),),
        "biological_question": "How does normalization change this field?",
    }
    values.update(changes)
    return PipelineDemoContribution(**values)


def test_pipeline_demo_contribution_owns_visual_declarations(tmp_path):
    contribution = _contribution(tmp_path)

    assert contribution.presentation_identities == (
        contribution.presentation_identity,
        *contribution.supporting_presentation_identities,
    )
    assert contribution.pipeline_steps[0].name == "Normalize"


def test_pipeline_demo_contribution_rejects_mutable_or_foreign_steps(tmp_path):
    with pytest.raises(TypeError, match="tuple of FunctionStep"):
        _contribution(
            tmp_path,
            pipeline_steps=[FunctionStep(name="Normalize", func=percentile_normalize)],
        )

    with pytest.raises(ValueError, match="exactly one"):
        _contribution(
            tmp_path,
            presentation_identity=_identity(step_name="Missing"),
        )


def test_pipeline_demo_contribution_rejects_runtime_contextual_identity(tmp_path):
    identity = replace(
        _identity(),
        pipeline_position=0,
        step_scope_id="submission::step_0",
    )

    with pytest.raises(ValueError, match="without compiled/runtime scope"):
        _contribution(tmp_path, presentation_identity=identity)
