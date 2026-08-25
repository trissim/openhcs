from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import MethodType, SimpleNamespace

import pytest
from PyQt6.QtWidgets import QWidget
from pyqt_reactive.services.async_operation_executor import (
    AsyncOperationExecutorClosedError,
)

import openhcs.pyqt_gui.main as main_module
from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.config import (
    GuiLogLevel,
    PyQtGuiRuntimeContext,
    get_default_ui_config,
)
from openhcs.pyqt_gui.main import MainWindowUiServices, OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowLifecycleWorkflow,
    MainWindowUiBridgeLifecycle,
)
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


def _visible_leaf_paths(value: object, prefix: str = "") -> tuple[str, ...]:
    """Derive the actual Configure OpenHCS leaf inventory from dataclasses."""

    if not is_dataclass(value):
        return (prefix,)
    paths: list[str] = []
    for declaration in fields(value):
        if declaration.metadata.get("ui_hidden"):
            continue
        child_prefix = f"{prefix}.{declaration.name}" if prefix else declaration.name
        paths.extend(
            _visible_leaf_paths(
                getattr(value, declaration.name),
                child_prefix,
            )
        )
    return tuple(paths)


@dataclass
class _ConfigAwareStub:
    calls: int = 0
    last_config: GlobalPipelineConfig | None = None

    def on_config_changed(self, new_config: GlobalPipelineConfig) -> None:
        self.calls += 1
        self.last_config = new_config


@dataclass
class _ServiceAdapterStub:
    calls: int = 0
    last_config: GlobalPipelineConfig | None = None

    def set_global_config(self, config: GlobalPipelineConfig) -> None:
        self.calls += 1
        self.last_config = config


@dataclass
class _LifecycleWorkflowStub:
    calls: int = 0
    last_config: GlobalPipelineConfig | None = None

    def propagate_config(self, new_config: GlobalPipelineConfig) -> None:
        self.calls += 1
        self.last_config = new_config


def test_on_config_changed_propagates_to_embedded_widgets() -> None:
    service_adapter = _ServiceAdapterStub()
    lifecycle_workflow = _LifecycleWorkflowStub()

    main_like = type("MainLike", (), {})()
    main_like.runtime_context = PyQtGuiRuntimeContext(get_default_ui_config())
    main_like.config_services = service_adapter
    main_like.lifecycle_workflow = lifecycle_workflow
    main_like.set_pipeline_runtime_config = lambda config: setattr(
        main_like,
        "runtime_context",
        main_like.runtime_context.with_pipeline_runtime(config),
    )

    new_config = GlobalPipelineConfig(num_workers=2)
    OpenHCSMainWindow.on_config_changed(main_like, new_config)

    assert main_like.runtime_context.pipeline_runtime == new_config
    assert service_adapter.calls == 1
    assert service_adapter.last_config == new_config

    assert lifecycle_workflow.calls == 1
    assert lifecycle_workflow.last_config == new_config


def test_service_adapter_uses_runtime_context_as_global_config_owner() -> None:
    main_window = type("MainWindow", (), {})()
    main_window.runtime_context = PyQtGuiRuntimeContext(get_default_ui_config())
    main_window.pipeline_runtime_config = main_window.runtime_context.pipeline_runtime

    def set_pipeline_runtime_config(config: GlobalPipelineConfig) -> None:
        main_window.pipeline_runtime_config = config

    main_window.set_pipeline_runtime_config = set_pipeline_runtime_config
    adapter = object.__new__(PyQtServiceAdapter)
    adapter.main_window = main_window
    new_config = GlobalPipelineConfig(num_workers=4)

    assert adapter.get_global_config() is main_window.pipeline_runtime_config
    adapter.set_global_config(new_config)

    assert main_window.pipeline_runtime_config is new_config
    assert not hasattr(main_window, "global_config")


def test_system_monitor_construction_receives_current_ui_config(monkeypatch) -> None:
    current = get_default_ui_config()

    class SystemMonitorProbe:
        def __init__(self, *, config) -> None:
            self.config = config

    monkeypatch.setattr(main_module, "SystemMonitorWidget", SystemMonitorProbe)
    services = type("Services", (), {"widget_gui_config": current})()

    monitor = MainWindowUiServices.create_system_monitor_widget(services)

    assert monitor.config is current.performance_monitor


def test_set_ui_config_propagates_one_exact_object_to_live_consumers() -> None:
    class ConfigConsumer:
        def __init__(self) -> None:
            self.config = None

        def set_ui_config(self, config) -> None:
            self.config = config

    class ZMQConsumer:
        def __init__(self) -> None:
            self.config = None
            self.ports = None
            self.progress_config = None

        def set_zmq_config(self, config, ports) -> None:
            self.config = config
            self.ports = ports

        def set_progress_config(self, config) -> None:
            self.progress_config = config

    class MonitorConsumer:
        def __init__(self) -> None:
            self.config = None

        def update_config(self, config) -> None:
            self.config = config

    class Signal:
        def __init__(self) -> None:
            self.value = None

        def emit(self, value) -> None:
            self.value = value

    current = get_default_ui_config()
    updated = replace(
        current,
        performance_monitor=replace(current.performance_monitor, update_fps=2.0),
        zmq=OpenHCSZMQConfig(default_port=8123),
    )
    main_like = type("MainLike", (), {})()
    main_like.runtime_context = PyQtGuiRuntimeContext(current)
    scheduled_operations = []
    main_like.window_services = SimpleNamespace(
        widget_gui_config=current,
        execute_async_operation=scheduled_operations.append,
    )
    main_like._prepare_execution_services = object()
    main_like.system_monitor = MonitorConsumer()
    main_like.plate_manager_widget = ConfigConsumer()
    main_like.zmq_manager_widget = ZMQConsumer()
    main_like.shortcut_lifecycle = SimpleNamespace(apply=lambda config: None)
    main_like._reconcile_ui_bridge = lambda config: None
    main_like.zmq_server_manager_ports_to_scan = lambda config=None: [8123, 5555]
    main_like.ui_config_changed = Signal()
    main_like._apply_ui_config_consumers = MethodType(
        OpenHCSMainWindow._apply_ui_config_consumers,
        main_like,
    )

    OpenHCSMainWindow.set_ui_config(main_like, updated)

    assert main_like.runtime_context.ui_config is updated
    assert main_like.window_services.widget_gui_config is updated
    assert main_like.system_monitor.config is updated.performance_monitor
    assert main_like.plate_manager_widget.config is updated
    assert main_like.zmq_manager_widget.config is updated.zmq
    assert main_like.zmq_manager_widget.progress_config is updated.progress
    assert main_like.ui_config_changed.value is updated
    assert scheduled_operations == [main_like._prepare_execution_services]


def test_set_ui_config_applies_changed_logging_declaration(monkeypatch) -> None:
    import openhcs.pyqt_gui.services.logging_config as logging_service

    current = get_default_ui_config()
    updated = replace(
        current,
        logging=replace(current.logging, level=GuiLogLevel.DEBUG),
    )
    applied = []
    monkeypatch.setattr(
        logging_service,
        "configure_gui_logging",
        applied.append,
    )
    main_like = SimpleNamespace(
        runtime_context=PyQtGuiRuntimeContext(current),
        window_services=SimpleNamespace(widget_gui_config=current),
        ui_config_changed=SimpleNamespace(emit=lambda _config: None),
        _apply_ui_config_consumers=lambda _config: None,
    )

    OpenHCSMainWindow.set_ui_config(main_like, updated)

    assert applied == [updated.logging]


def test_configure_openhcs_roots_reach_live_application_owners() -> None:
    """The exact saved roots reach their live and next-execution owners."""

    class Recorder:
        def __init__(self) -> None:
            self.values: list[object] = []

        def record(self, value=None, *additional_values, **named_values) -> None:
            if value is not None:
                self.values.append(value)
            self.values.extend(additional_values)
            self.values.extend(named_values.values())

    ui_config = get_default_ui_config()
    monitor = Recorder()
    progress = Recorder()
    plate_zmq = Recorder()
    shortcuts = Recorder()
    bridge = Recorder()
    manager_zmq = Recorder()

    plate_manager = SimpleNamespace(
        zmq_client_service=SimpleNamespace(set_config=plate_zmq.record),
        _batch_workflow_service=SimpleNamespace(update_progress_config=progress.record),
    )
    plate_manager.set_ui_config = MethodType(
        PlateManagerWidget.set_ui_config,
        plate_manager,
    )
    main_like = SimpleNamespace(
        system_monitor=SimpleNamespace(update_config=monitor.record),
        plate_manager_widget=plate_manager,
        shortcut_lifecycle=SimpleNamespace(apply=shortcuts.record),
        ui_bridge_lifecycle=SimpleNamespace(reconcile=bridge.record),
        zmq_manager_widget=SimpleNamespace(
            set_zmq_config=manager_zmq.record,
            set_progress_config=progress.record,
        ),
        _create_ui_bridge_server=lambda *_args: None,
        zmq_server_manager_ports_to_scan=lambda _config: (),
    )
    main_like._reconcile_ui_bridge = MethodType(
        OpenHCSMainWindow._reconcile_ui_bridge,
        main_like,
    )

    OpenHCSMainWindow._apply_ui_config_consumers(main_like, ui_config)

    live_owner_values = {
        id(value)
        for value in (
            *monitor.values,
            *progress.values,
            *plate_zmq.values,
            *shortcuts.values,
            *bridge.values,
            *manager_zmq.values,
        )
    }
    visible_component_owners = {
        declaration.name: getattr(ui_config, declaration.name)
        for declaration in fields(ui_config)
        if not declaration.metadata.get("ui_hidden")
        and is_dataclass(getattr(ui_config, declaration.name))
        and declaration.name != "logging"
    }
    ui_leaf_lifecycle = {
        path: (
            "live"
            if id(visible_component_owners[path.partition(".")[0]]) in live_owner_values
            else "unconsumed"
        )
        for path in _visible_leaf_paths(ui_config)
        if path.partition(".")[0] in visible_component_owners
    }
    assert ui_leaf_lifecycle
    assert set(ui_leaf_lifecycle.values()) == {"live"}
    assert is_dataclass(ui_config.logging)

    global_config = GlobalPipelineConfig()
    global_publication = Recorder()
    global_propagation = Recorder()
    global_like = SimpleNamespace(
        runtime_context=PyQtGuiRuntimeContext(ui_config),
        config_services=SimpleNamespace(set_global_config=global_publication.record),
        lifecycle_workflow=SimpleNamespace(propagate_config=global_propagation.record),
    )
    global_like.set_pipeline_runtime_config = MethodType(
        OpenHCSMainWindow.set_pipeline_runtime_config,
        global_like,
    )

    OpenHCSMainWindow.on_config_changed(global_like, global_config)

    assert global_like.runtime_context.pipeline_runtime is global_config
    assert global_publication.values == [global_config]
    assert global_propagation.values == [global_config]
    assert _visible_leaf_paths(global_config)


def test_set_ui_config_restores_previous_consumers_before_rejecting_update() -> None:
    class Signal:
        def __init__(self) -> None:
            self.values = []

        def emit(self, value) -> None:
            self.values.append(value)

    current = get_default_ui_config()
    updated = replace(
        current,
        performance_monitor=replace(current.performance_monitor, update_fps=2.0),
    )
    applied = []
    main_like = type("MainLike", (), {})()
    main_like.runtime_context = PyQtGuiRuntimeContext(current)
    main_like.window_services = SimpleNamespace(widget_gui_config=current)
    main_like.ui_config_changed = Signal()

    def apply_consumers(config) -> None:
        applied.append(config)
        if config is updated:
            raise RuntimeError("consumer rejected update")

    main_like._apply_ui_config_consumers = apply_consumers

    with pytest.raises(RuntimeError, match="consumer rejected update"):
        OpenHCSMainWindow.set_ui_config(main_like, updated)

    assert applied == [updated, current]
    assert main_like.runtime_context.ui_config is current
    assert main_like.window_services.widget_gui_config is current
    assert main_like.ui_config_changed.values == []


def test_lifecycle_workflow_propagates_config_to_embedded_widgets(qapp) -> None:
    plate_manager = _ConfigAwareStub()
    pipeline_editor = _ConfigAwareStub()
    progress_bar = type("ProgressBar", (), {})()

    workflow = MainWindowLifecycleWorkflow(
        main_window=QWidget(),
        embedded_widgets=SimpleNamespace(
            require_plate_manager=lambda: plate_manager,
            require_pipeline_editor=lambda: pipeline_editor,
        ),
        floating_windows={},
        status_progress_bar=progress_bar,
        ui_bridge_lifecycle=MainWindowUiBridgeLifecycle(),
        ui_services=SimpleNamespace(close=lambda: None),
    )

    new_config = GlobalPipelineConfig(num_workers=3)
    workflow.propagate_config(new_config)

    assert plate_manager.calls == 1
    assert plate_manager.last_config == new_config
    assert pipeline_editor.calls == 1
    assert pipeline_editor.last_config == new_config


def test_lifecycle_workflow_projects_runtime_progress_without_retaining_state(
    qapp,
) -> None:
    from PyQt6.QtWidgets import QProgressBar

    progress_bar = QProgressBar()
    workflow = MainWindowLifecycleWorkflow(
        main_window=QWidget(),
        embedded_widgets=SimpleNamespace(),
        floating_windows={},
        status_progress_bar=progress_bar,
        ui_bridge_lifecycle=MainWindowUiBridgeLifecycle(),
        ui_services=SimpleNamespace(close=lambda: None),
    )

    workflow.runtime_progress_changed(
        SimpleNamespace(overall_percent=37.6, has_active_work=True)
    )

    assert progress_bar.minimum() == 0
    assert progress_bar.maximum() == 100
    assert progress_bar.value() == 38
    assert not progress_bar.isHidden()

    workflow.runtime_progress_changed(
        SimpleNamespace(overall_percent=100.0, has_active_work=False)
    )

    assert progress_bar.value() == 100
    assert not progress_bar.isVisible()


def test_lifecycle_workflow_cleans_embedded_resource_owners_before_qt_teardown(
    qapp,
    monkeypatch,
) -> None:
    calls = []
    main_window = QWidget()
    embedded_widgets = SimpleNamespace(
        require_system_monitor=lambda: SimpleNamespace(
            stop_monitoring=lambda: calls.append("system_monitor")
        ),
        require_plate_manager=lambda: SimpleNamespace(
            cleanup=lambda: calls.append("plate_manager")
        ),
        require_zmq_manager=lambda: SimpleNamespace(
            cleanup=lambda: calls.append("zmq_manager")
        ),
    )
    bridge_lifecycle = SimpleNamespace(close=lambda: calls.append("ui_bridge"))
    monkeypatch.setattr(main_module.WindowManager, "get_open_scopes", lambda: ())
    monkeypatch.setattr(
        main_module.QApplication,
        "topLevelWidgets",
        lambda: [main_window],
    )

    MainWindowLifecycleWorkflow(
        main_window=main_window,
        embedded_widgets=embedded_widgets,
        floating_windows={},
        status_progress_bar=SimpleNamespace(),
        ui_bridge_lifecycle=bridge_lifecycle,
        ui_services=SimpleNamespace(close=lambda: calls.append("async_services")),
    ).close()

    assert calls == [
        "ui_bridge",
        "system_monitor",
        "plate_manager",
        "zmq_manager",
        "async_services",
    ]


def test_lifecycle_workflow_attempts_every_owner_before_reporting_failures(
    qapp,
    monkeypatch,
) -> None:
    calls: list[str] = []
    main_window = QWidget()

    def fail(name: str) -> None:
        calls.append(name)
        raise RuntimeError(name)

    embedded_widgets = SimpleNamespace(
        require_system_monitor=lambda: SimpleNamespace(
            stop_monitoring=lambda: fail("system_monitor")
        ),
        require_plate_manager=lambda: SimpleNamespace(
            cleanup=lambda: fail("plate_manager")
        ),
        require_zmq_manager=lambda: SimpleNamespace(
            cleanup=lambda: calls.append("zmq_manager")
        ),
    )
    floating_window = SimpleNamespace(
        close=lambda: fail("floating_window"),
        deleteLater=lambda: calls.append("floating_delete"),
    )
    top_level = SimpleNamespace(close=lambda: calls.append("top_level"))
    monkeypatch.setattr(
        main_module.WindowManager,
        "get_open_scopes",
        lambda: ("managed",),
    )
    monkeypatch.setattr(
        main_module.WindowManager,
        "close_window",
        lambda _scope_id: calls.append("managed_window"),
    )
    monkeypatch.setattr(
        main_module.QApplication,
        "topLevelWidgets",
        lambda: [main_window, top_level],
    )

    workflow = MainWindowLifecycleWorkflow(
        main_window=main_window,
        embedded_widgets=embedded_widgets,
        floating_windows={"floating": floating_window},
        status_progress_bar=SimpleNamespace(),
        ui_bridge_lifecycle=SimpleNamespace(close=lambda: fail("ui_bridge")),
        ui_services=SimpleNamespace(close=lambda: calls.append("async_services")),
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        workflow.close()

    assert [str(error) for error in exc_info.value.exceptions] == [
        "ui_bridge",
        "system_monitor",
        "plate_manager",
        "floating_window",
    ]
    assert calls == [
        "ui_bridge",
        "system_monitor",
        "plate_manager",
        "zmq_manager",
        "managed_window",
        "floating_window",
        "floating_delete",
        "top_level",
        "async_services",
    ]
    assert workflow.floating_windows == {}


def test_main_close_attempts_every_top_level_owner_after_other_failures(
    qapp,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fail(name: str) -> None:
        calls.append(name)
        raise RuntimeError(name)

    event = SimpleNamespace(accept=lambda: calls.append("accept"))
    main_like = SimpleNamespace(
        dock_layout_store=SimpleNamespace(save=lambda _window: fail("layout")),
        shortcut_lifecycle=SimpleNamespace(close=lambda: fail("shortcuts")),
        lifecycle_workflow=SimpleNamespace(close=lambda: fail("resources")),
    )
    monkeypatch.setattr(
        "PyQt6.QtCore.QTimer.singleShot",
        lambda _delay, _callback: calls.append("quit_scheduled"),
    )

    OpenHCSMainWindow.closeEvent(main_like, event)

    assert calls == [
        "layout",
        "shortcuts",
        "resources",
        "accept",
        "quit_scheduled",
    ]


def test_pyqt_service_adapter_owns_and_closes_async_operations(qapp) -> None:
    adapter = PyQtServiceAdapter(QWidget())

    async def operation() -> str:
        return "done"

    future = adapter.execute_async_operation(operation)

    assert future.result(timeout=1.0) == "done"
    assert adapter.close() is None
    with pytest.raises(AsyncOperationExecutorClosedError):
        adapter.execute_async_operation(operation)
