from openhcs.pyqt_gui.services.time_travel_navigation import (
    should_include_time_travel_scope,
)
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from pyqt_reactive.services.function_navigation import build_function_token_field_path


class RuntimeCallable:
    def __call__(self, image, threshold: int = 1):
        return image


def _reset_registry() -> None:
    ObjectStateRegistry._states.clear()
    ObjectStateRegistry._time_travel_limbo.clear()
    ObjectStateRegistry._graveyard.clear()
    ObjectStateRegistry._snapshots.clear()
    ObjectStateRegistry._timelines.clear()
    ObjectStateRegistry._current_timeline = "main"
    ObjectStateRegistry._current_head = None
    ObjectStateRegistry._in_time_travel = False
    ObjectStateRegistry._atomic_depth = 0
    ObjectStateRegistry._atomic_label = None
    ObjectStateRegistry._atomic_triggering_scope = None


def test_time_travel_navigation_filters_unrelated_dirty_scopes() -> None:
    triggering_scope = "/tmp/plate::functionstep_5"

    assert should_include_time_travel_scope(
        changed_scope_id=triggering_scope,
        triggering_scope=triggering_scope,
    )
    assert should_include_time_travel_scope(
        changed_scope_id=f"{triggering_scope}::cellprofilerruntimecallable_0",
        triggering_scope=triggering_scope,
    )
    assert not should_include_time_travel_scope(
        changed_scope_id="/tmp/plate::functionstep_9",
        triggering_scope=triggering_scope,
    )


def test_time_travel_navigation_keeps_legacy_global_refresh_when_scope_unknown() -> None:
    assert should_include_time_travel_scope(
        changed_scope_id="/tmp/plate::functionstep_9",
        triggering_scope=None,
    )


def test_time_travel_function_scope_reopens_parent_step_with_parent_state() -> None:
    _reset_registry()
    step_scope = "/tmp/plate::functionstep_5"
    function_scope = f"{step_scope}::cellprofilerruntimecallable_0"
    step_state = ObjectState(
        FunctionStep(func=RuntimeCallable(), name="Crop"),
        scope_id=step_scope,
    )
    function_state = ObjectState(
        RuntimeCallable(),
        scope_id=function_scope,
        parent_state=step_state,
        initial_values={"threshold": 3},
    )
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    ObjectStateRegistry.register(function_state, _skip_snapshot=True)

    requests = OpenHCSMainWindow._build_time_travel_window_requests(
        OpenHCSMainWindow.__new__(OpenHCSMainWindow),
        [(function_scope, function_state)],
        step_scope,
    )

    request = requests[step_scope]
    assert request.scope_id == step_scope
    assert request.object_state is step_state
    assert request.target is not None
    assert request.target.to_field_path() == build_function_token_field_path(
        "cellprofilerruntimecallable_0"
    )
