from types import SimpleNamespace

import pytest

from objectstate import get_base_type_for_lazy

from openhcs.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    DtypeConfig,
    ProcessingConfig,
    StepMaterializationConfig,
    WellFilterMode,
)
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
    def __init__(self, values, scope_id="plate::functionstep_0"):
        self.values = values
        self.scope_id = scope_id

    def find_path_for_type(self, owner_type):
        owner_base = get_base_type_for_lazy(owner_type) or owner_type
        for path, value in self.values.items():
            if "." in path:
                continue
            value_base = get_base_type_for_lazy(type(value)) or type(value)
            if value_base is owner_base:
                return path
        return None

    def get_saved_resolved_value(self, path):
        return self.values[path]

    def to_saved_resolved_object(self):
        return SimpleNamespace(
            **{
                path: value
                for path, value in self.values.items()
                if "." not in path
            }
        )

    def to_object(self):
        raise AssertionError("StepSnapshot must not call ObjectState.to_object()")


def _state_values(**overrides):
    source_bindings = StepSourceBindingsConfig(bindings=(
                    NamedSourceBinding(
                        alias="OrigBlue",
                        selector=SourceSelector(
                            components=(ComponentSelector("channel", "1"),)
                        ),
                    ),
                ))
    values = {
        "enabled": True,
        "source_bindings": source_bindings,
        "processing_config": ProcessingConfig(
            variable_components=[VariableComponents.SITE],
            group_by=None,
            input_source=InputSource.PIPELINE_START,
        ),
        "dtype_config": DtypeConfig(),
        "step_materialization_config": StepMaterializationConfig(enabled=False),
    }
    values.update(overrides)
    return values


def test_step_snapshot_captures_saved_values_without_object_conversion():
    step = FunctionStep(
        func=_identity,
        name="identity",
    )
    state = StateStub(_state_values())

    snapshot = StepSnapshot.from_resolved_step(
        index=0,
        step=step,
        step_state=state,
    )

    assert snapshot.name == "identity"
    assert snapshot.scope_id == "plate::functionstep_0"
    assert snapshot.step_type == "FunctionStep"
    assert snapshot.enabled is True
    assert snapshot.is_function_step is True
    assert snapshot.func is _identity
    assert snapshot.source_bindings == state.values["source_bindings"]
    assert snapshot.input_source is InputSource.PIPELINE_START
    assert snapshot.variable_components == [VariableComponents.SITE]


def test_step_snapshot_captures_well_filter_roots():
    step = FunctionStep(func=_identity, name="filtered")
    state = StateStub(
        _state_values(
            **{
                "step_materialization_config": StepMaterializationConfig(
                    enabled=False,
                    well_filter=["A01"],
                    well_filter_mode=WellFilterMode.INCLUDE,
                ),
            }
        ),
    )

    snapshot = StepSnapshot.from_resolved_step(
        index=2,
        step=step,
        step_state=state,
    )

    assert len(snapshot.well_filters) == 1
    assert snapshot.well_filters[0].config.well_filter == ["A01"]
    assert snapshot.well_filters[0].config.well_filter_mode is WellFilterMode.INCLUDE


def test_build_step_snapshots_requires_matching_objectstate():
    step = FunctionStep(func=_identity, name="missing")

    with pytest.raises(ValueError, match="Missing ObjectState"):
        build_step_snapshots([step], {})
