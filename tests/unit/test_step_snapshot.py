from types import MappingProxyType, SimpleNamespace

import pytest

from openhcs.constants.input_source import InputSource
from openhcs.core.config import StepMaterializationConfig, WellFilterMode
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    GroupedSourceBindings,
    NamedSourceBinding,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep


def _identity(image):
    return image


class StateStub:
    def __init__(self, values, path_to_type=None, scope_id="plate::functionstep_0"):
        self.values = values
        self._path_to_type = path_to_type or {}
        self.scope_id = scope_id

    def get_saved_resolved_value(self, path):
        return self.values[path]

    def to_object(self):
        raise AssertionError("StepSnapshot must not call ObjectState.to_object()")


def _state_values(**overrides):
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigBlue",
                        selector=SourceSelector(
                            components=(ComponentSelector("channel", "1"),)
                        ),
                    ),
                )
            ),
        )
    )
    values = {
        "enabled": True,
        "source_bindings": source_bindings,
        "processing_config.variable_components": ("site",),
        "processing_config.group_by": None,
        "processing_config.input_source": InputSource.PIPELINE_START,
        "processing_config": SimpleNamespace(name="processing"),
        "step_materialization_config": SimpleNamespace(enabled=False),
        "dtype_config": SimpleNamespace(name="dtype"),
    }
    values.update(overrides)
    return values


def test_step_snapshot_captures_saved_values_without_object_conversion():
    step = FunctionStep(func=_identity, name="identity")
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
    assert snapshot.variable_components == ("site",)
    assert isinstance(snapshot.injectable_values, MappingProxyType)
    assert snapshot.injectable_values["enabled"] is True
    assert snapshot.injectable_values["dtype_config"].name == "dtype"


def test_step_snapshot_captures_well_filter_roots():
    step = FunctionStep(func=_identity, name="filtered")
    state = StateStub(
        _state_values(
            **{
                "step_materialization_config.well_filter": ["A01"],
                "step_materialization_config.well_filter_mode": WellFilterMode.INCLUDE,
            }
        ),
        {"step_materialization_config": StepMaterializationConfig},
    )

    snapshot = StepSnapshot.from_resolved_step(
        index=2,
        step=step,
        step_state=state,
    )

    assert len(snapshot.well_filters) == 1
    assert snapshot.well_filters[0].root == "step_materialization_config"
    assert snapshot.well_filters[0].well_filter == ["A01"]
    assert snapshot.well_filters[0].well_filter_mode is WellFilterMode.INCLUDE


def test_build_step_snapshots_requires_matching_objectstate():
    step = FunctionStep(func=_identity, name="missing")

    with pytest.raises(ValueError, match="Missing ObjectState"):
        build_step_snapshots([step], {})
