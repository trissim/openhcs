from __future__ import annotations

from dataclasses import dataclass, replace
from types import MethodType, SimpleNamespace

import openhcs.pyqt_gui.main as main_module
import pytest
from PyQt6.QtWidgets import QWidget
from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.config import (
    PyQtGuiRuntimeContext,
    get_default_ui_config,
)
from openhcs.pyqt_gui.main import MainWindowUiServices, OpenHCSMainWindow
from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowEmbeddedWidgets,
    MainWindowLifecycleWorkflow,
    MainWindowUiBridgeLifecycle,
)


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
    main_like.set_pipeline_runtime_config = (
        lambda config: setattr(
            main_like,
            "runtime_context",
            main_like.runtime_context.with_pipeline_runtime(config),
        )
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

        def set_zmq_config(self, config, ports) -> None:
            self.config = config
            self.ports = ports

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
    main_like.window_services = type("Services", (), {})()
    main_like.window_services.widget_gui_config = current
    main_like.system_monitor = MonitorConsumer()
    main_like.plate_manager_widget = ConfigConsumer()
    main_like.zmq_manager_widget = ZMQConsumer()
    main_like.shortcut_lifecycle = SimpleNamespace(apply=lambda config: None)
    main_like._reconcile_ui_bridge = lambda config: None
    main_like.zmq_server_manager_ports_to_scan = (
        lambda config=None: [8123, 5555]
    )
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
    assert main_like.ui_config_changed.value is updated


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


def test_lifecycle_workflow_propagates_config_to_embedded_widgets() -> None:
    plate_manager = _ConfigAwareStub()
    pipeline_editor = _ConfigAwareStub()
    progress_bar = type("ProgressBar", (), {})()

    workflow = MainWindowLifecycleWorkflow(
        main_window=QWidget(),
        embedded_widgets=MainWindowEmbeddedWidgets(
            plate_manager=plate_manager,
            pipeline_editor=pipeline_editor,
        ),
        floating_windows={},
        status_progress_bar=progress_bar,
        ui_bridge_lifecycle=MainWindowUiBridgeLifecycle(),
    )

    new_config = GlobalPipelineConfig(num_workers=3)
    workflow.propagate_config(new_config)

    assert plate_manager.calls == 1
    assert plate_manager.last_config == new_config
    assert pipeline_editor.calls == 1
    assert pipeline_editor.last_config == new_config
