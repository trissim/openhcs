from dataclasses import fields

import pytest

from openhcs.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import ProcessingConfig, StepMaterializationConfig
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep


def _identity(image):
    return image


class StateStub:
    def __init__(self, scope_id="plate::functionstep_0"):
        self.scope_id = scope_id

    def to_object(self):
        raise AssertionError("StepSnapshot must not call ObjectState.to_object()")


def test_step_snapshot_reads_semantics_from_resolved_step_without_object_conversion():
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                selector=SourceSelector(
                    components=(ComponentSelector("channel", "1"),)
                ),
            ),
        )
    )
    step = FunctionStep(
        func=_identity,
        name="identity",
        source_bindings=source_bindings,
        processing_config=ProcessingConfig(
            variable_components=[VariableComponents.SITE],
            group_by=None,
            input_source=InputSource.PIPELINE_START,
        ),
        step_materialization_config=StepMaterializationConfig(enabled=False),
    )
    state = StateStub()

    snapshot = build_step_snapshots([step], {0: state})[0]

    assert [field.name for field in fields(StepSnapshot)] == [
        "index",
        "scope_id",
        "step",
    ]
    assert snapshot.scope_id == "plate::functionstep_0"
    assert snapshot.step is step
    assert snapshot.step.source_bindings is source_bindings
    assert snapshot.step.processing_config.input_source is InputSource.PIPELINE_START
    assert snapshot.step.processing_config.variable_components == [
        VariableComponents.SITE
    ]


def test_build_step_snapshots_requires_matching_objectstate():
    step = FunctionStep(func=_identity, name="missing")

    with pytest.raises(ValueError, match="Missing ObjectState"):
        build_step_snapshots([step], {})
