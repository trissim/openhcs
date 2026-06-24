from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.main import OpenHCSMainWindow
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
    main_like.global_config = GlobalPipelineConfig()
    main_like.config_services = service_adapter
    main_like.lifecycle_workflow = lifecycle_workflow
    main_like.set_pipeline_runtime_config = (
        lambda config: setattr(main_like, "global_config", config)
    )

    new_config = GlobalPipelineConfig(num_workers=2)
    OpenHCSMainWindow.on_config_changed(main_like, new_config)

    assert main_like.global_config == new_config
    assert service_adapter.calls == 1
    assert service_adapter.last_config == new_config

    assert lifecycle_workflow.calls == 1
    assert lifecycle_workflow.last_config == new_config


def test_lifecycle_workflow_propagates_config_to_embedded_widgets() -> None:
    plate_manager = _ConfigAwareStub()
    pipeline_editor = _ConfigAwareStub()
    progress_bar = type("ProgressBar", (), {})()

    workflow = MainWindowLifecycleWorkflow(
        main_window=object(),
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
