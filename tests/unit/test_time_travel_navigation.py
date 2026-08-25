from openhcs.pyqt_gui.services.time_travel_navigation import (
    TimeTravelSourceScope,
    make_field_path_target,
    make_function_token_target,
    should_replace_navigation_target,
    should_include_time_travel_scope,
)
from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from pyqt_reactive.services.function_navigation import build_function_token_field_path
from pyqt_reactive.services.window_navigation import (
    NavigationWaitReason,
    RegisteredWindowNavigationReadiness,
    RegisteredWindowNavigationRequest,
    WindowNavigationDriver,
    WindowNavigationResult,
)
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindowNavigationDriver


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
        TimeTravelSourceScope(
            changed_scope_id=triggering_scope,
            triggering_scope=triggering_scope,
        )
    )
    assert not should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id=f"{triggering_scope}::cellprofilerruntimecallable_0",
            triggering_scope=triggering_scope,
        )
    )
    assert not should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id="/tmp/plate::functionstep_9",
            triggering_scope=triggering_scope,
        )
    )


def test_time_travel_navigation_keeps_legacy_global_refresh_when_scope_unknown() -> (
    None
):
    assert should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id="/tmp/plate::functionstep_9",
            triggering_scope=None,
        )
    )


def test_time_travel_navigation_includes_parent_step_for_function_trigger() -> None:
    step_scope = "/tmp/plate::functionstep_5"
    function_scope = f"{step_scope}::cellprofilerruntimecallable_0"

    assert should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id=function_scope,
            triggering_scope=function_scope,
        )
    )
    assert should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id=step_scope,
            triggering_scope=function_scope,
        )
    )


def test_time_travel_plate_scope_does_not_open_step_descendants() -> None:
    plate_scope = "/tmp/plate"

    assert not should_include_time_travel_scope(
        TimeTravelSourceScope(
            changed_scope_id=f"{plate_scope}::functionstep_5",
            triggering_scope=plate_scope,
        )
    )


def test_time_travel_unknown_trigger_does_not_create_missing_windows(
    monkeypatch,
) -> None:
    _reset_registry()
    step_scope = "/tmp/plate::functionstep_5"
    step_state = ObjectState(
        FunctionStep(func=RuntimeCallable(), name="Crop"),
        scope_id=step_scope,
    )
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)

    requests = []

    def fake_navigate(request):
        requests.append(request)
        return WindowNavigationResult(
            request=request,
            window=None,
            focused=False,
            created=False,
            window_scope_id=request.scope_id,
        )

    from pyqt_reactive.services.scope_window_navigation import (
        ScopeWindowNavigationService,
    )

    monkeypatch.setattr(
        ScopeWindowNavigationService,
        "navigate",
        staticmethod(fake_navigate),
    )
    scheduled: list[object] = []
    monkeypatch.setattr(
        OpenHCSMainWindow,
        "_defer_time_travel_navigation",
        lambda _self, callback: scheduled.append(callback),
    )

    OpenHCSMainWindow._on_time_travel_complete(
        OpenHCSMainWindow.__new__(OpenHCSMainWindow),
        [(step_scope, step_state)],
        None,
    )

    assert requests == []
    assert len(scheduled) == 1
    scheduled[0]()

    assert len(requests) == 1
    assert requests[0].create_if_missing is False


def test_time_travel_navigation_runs_after_later_refresh_callbacks(monkeypatch) -> None:
    _reset_registry()
    step_scope = "/tmp/plate::functionstep_5"
    field_path = "source_bindings.source_filters"
    step_state = ObjectState(
        FunctionStep(func=RuntimeCallable(), name="Crop"),
        scope_id=step_scope,
    )
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    step_state._last_changed_field = field_path

    scheduled: list[object] = []
    monkeypatch.setattr(
        OpenHCSMainWindow,
        "_defer_time_travel_navigation",
        lambda _self, callback: scheduled.append(callback),
    )

    class TabWidget:
        def __init__(self) -> None:
            self.indices: list[int] = []

        def setCurrentIndex(self, index: int) -> None:
            self.indices.append(index)

    class StepWindow:
        def __init__(self) -> None:
            self.tab_widget = TabWidget()

    window = StepWindow()
    navigate_requests = []

    def fake_navigate(request):
        navigate_requests.append(request)
        return WindowNavigationResult(
            request=request,
            window=window,
            focused=True,
            created=False,
            window_scope_id=request.scope_id,
        )

    from pyqt_reactive.services.scope_window_navigation import (
        ScopeWindowNavigationService,
    )

    monkeypatch.setattr(
        ScopeWindowNavigationService,
        "navigate",
        staticmethod(fake_navigate),
    )

    monkeypatch.setattr(
        OpenHCSMainWindow,
        "_select_tab_for_time_travel",
        lambda _self, scope_id, target: window.tab_widget.setCurrentIndex(0),
    )

    main_window = OpenHCSMainWindow.__new__(OpenHCSMainWindow)
    OpenHCSMainWindow._on_time_travel_complete(
        main_window,
        [(step_scope, step_state)],
        step_scope,
    )

    assert navigate_requests == []
    window.tab_widget.setCurrentIndex(1)

    scheduled[0]()

    assert len(navigate_requests) == 1
    assert navigate_requests[0].field_path == field_path
    assert window.tab_widget.indices == [1, 0]


def test_time_travel_navigation_defer_runs_after_same_wave_refresh(monkeypatch) -> None:
    callbacks: list[object] = []

    def fake_single_shot(delay, callback) -> None:
        assert delay == 0
        callbacks.append(callback)

    from PyQt6.QtCore import QTimer

    monkeypatch.setattr(QTimer, "singleShot", staticmethod(fake_single_shot))

    order: list[str] = []
    main_window = OpenHCSMainWindow.__new__(OpenHCSMainWindow)
    OpenHCSMainWindow._defer_time_travel_navigation(
        main_window,
        lambda: order.append("navigate"),
    )
    QTimer.singleShot(0, lambda: order.append("refresh"))

    while callbacks:
        callback = callbacks.pop(0)
        callback()

    assert order == ["refresh", "navigate"]


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
        function_scope,
    )

    request = requests[step_scope]
    assert request.scope_id == step_scope
    assert request.object_state is step_state
    assert request.target is not None
    assert request.target.to_field_path() == build_function_token_field_path(
        "cellprofilerruntimecallable_0"
    )


def test_time_travel_step_source_bindings_child_targets_step_settings() -> None:
    _reset_registry()
    step_scope = "/tmp/plate::functionstep_5"
    field_path = "source_bindings.source_filters"
    step_state = ObjectState(
        FunctionStep(func=RuntimeCallable(), name="Crop"),
        scope_id=step_scope,
    )
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    step_state._last_changed_field = field_path

    requests = OpenHCSMainWindow._build_time_travel_window_requests(
        OpenHCSMainWindow.__new__(OpenHCSMainWindow),
        [(step_scope, step_state)],
        step_scope,
    )

    request = requests[step_scope]
    assert request.target is not None
    assert not request.target.is_function_target
    assert request.target.to_field_path() == field_path


def test_dual_editor_navigation_uses_step_editor_readiness_for_step_fields() -> None:
    class StepDriver(WindowNavigationDriver):
        def __init__(self) -> None:
            self.executed: list[str | None] = []
            self.callbacks: list[object] = []

        def readiness(self, request):
            return RegisteredWindowNavigationReadiness(
                wait_reason=NavigationWaitReason.FIELD_TARGET
            )

        def register_readiness_callback(self, request, callback):
            del request
            self.callbacks.append(callback)
            return True

        def execute(self, request):
            self.executed.append(request.field_path)

    class StepEditor:
        def __init__(self, driver: StepDriver) -> None:
            self.driver = driver

        def window_navigation_driver(self):
            return self.driver

    class TabWidget:
        def __init__(self) -> None:
            self.indices: list[int] = []

        def setCurrentIndex(self, index: int) -> None:
            self.indices.append(index)

    class Owner:
        def __init__(self, step_driver: StepDriver) -> None:
            self.step_editor = StepEditor(step_driver)
            self.func_editor = None
            self.tab_widget = TabWidget()

    step_driver = StepDriver()
    owner = Owner(step_driver)
    driver = DualEditorWindowNavigationDriver(owner)
    request = RegisteredWindowNavigationRequest(
        window=None,
        requested_scope_id="step_scope",
        field_path="source_bindings.source_filters",
    )

    assert driver.readiness(request).wait_reason is NavigationWaitReason.FIELD_TARGET

    def retry_callback() -> None:
        pass

    assert driver.register_readiness_callback(request, retry_callback)
    assert step_driver.callbacks == [retry_callback]

    driver.prepare(request)
    driver.execute(request)

    assert owner.tab_widget.indices == [0]
    assert step_driver.executed == ["source_bindings.source_filters"]


def test_dual_editor_function_navigation_does_not_use_step_build_callbacks(
    qapp,
) -> None:
    from PyQt6.QtWidgets import QWidget

    class StepDriver(WindowNavigationDriver):
        def __init__(self) -> None:
            self.callbacks: list[object] = []

        def register_readiness_callback(self, request, callback):
            del request
            self.callbacks.append(callback)
            return True

    class StepEditor:
        def __init__(self, driver: StepDriver) -> None:
            self.driver = driver

        def window_navigation_driver(self):
            return self.driver

    class FunctionEditor:
        def __init__(self) -> None:
            self.selected: list[str] = []

        def select_and_scroll_to_field(self, field_path: str) -> None:
            self.selected.append(field_path)

    class TabWidget:
        def __init__(self) -> None:
            self.indices: list[int] = []

        def setCurrentIndex(self, index: int) -> None:
            self.indices.append(index)

    class Owner:
        def __init__(self) -> None:
            self.step_driver = StepDriver()
            self.step_editor = StepEditor(self.step_driver)
            self.func_editor = FunctionEditor()
            self.tab_widget = TabWidget()

    owner = Owner()
    driver = DualEditorWindowNavigationDriver(owner)
    window = QWidget()
    request = RegisteredWindowNavigationRequest(
        window=window,
        requested_scope_id="step_scope",
        field_path="func.threshold",
    )

    assert driver.accepts(request)
    assert not driver.register_readiness_callback(request, lambda: None)

    driver.prepare(request)
    driver.execute(request)

    assert owner.tab_widget.indices == [1]
    assert owner.func_editor.selected == ["func.threshold"]
    assert owner.step_driver.callbacks == []
    window.close()


def test_time_travel_step_field_target_replaces_stale_function_target() -> None:
    field_target = make_field_path_target("source_bindings.source_filters")
    function_target = make_function_token_target("function_0")

    assert should_replace_navigation_target(function_target, field_target)
    assert not should_replace_navigation_target(field_target, function_target)


def test_time_travel_unknown_trigger_prefers_source_bindings_field_over_function_scope() -> (
    None
):
    _reset_registry()
    step_scope = "/tmp/plate::functionstep_5"
    function_scope = f"{step_scope}::function_0"
    field_path = "source_bindings.source_filters"
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
    step_state._last_changed_field = field_path

    main_window = OpenHCSMainWindow.__new__(OpenHCSMainWindow)

    function_first = OpenHCSMainWindow._build_time_travel_window_requests(
        main_window,
        [(function_scope, function_state), (step_scope, step_state)],
        None,
    )
    step_first = OpenHCSMainWindow._build_time_travel_window_requests(
        main_window,
        [(step_scope, step_state), (function_scope, function_state)],
        None,
    )

    assert function_first[step_scope].target is not None
    assert function_first[step_scope].target.to_field_path() == field_path
    assert not function_first[step_scope].target.is_function_target
    assert step_first[step_scope].target is not None
    assert step_first[step_scope].target.to_field_path() == field_path
    assert not step_first[step_scope].target.is_function_target
