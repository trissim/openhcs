from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QKeyEvent

from openhcs.core.config_cache import ConfigCacheSpec
from openhcs.pyqt_gui import config as config_module
from openhcs.pyqt_gui.config import (
    AgentUiBridgeConfig,
    ProgressUIConfig,
    ShortcutConfig,
    UIConfig,
    get_default_ui_config,
    load_cached_ui_config_sync,
    save_ui_config_sync,
)
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowShortcutLifecycle,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_workflow_components import (
    BatchWorkflowComponents,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)


def test_ui_config_cache_round_trip_applies_environment_at_load(
    tmp_path,
    monkeypatch,
) -> None:
    spec = ConfigCacheSpec(
        config_type=UIConfig,
        cache_file=tmp_path / "ui-config.cache",
    )
    monkeypatch.setattr(config_module, "ui_config_cache_spec", lambda: spec)
    persisted = replace(
        get_default_ui_config(),
        check_for_updates_on_startup=False,
        progress=ProgressUIConfig(update_fps=17.0),
        agent_bridge=replace(
            get_default_ui_config().agent_bridge,
            enabled=False,
            host="persisted-host",
            port=7997,
        ),
    )

    assert save_ui_config_sync(persisted) is True
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_HOST", "environment-host")
    restored = load_cached_ui_config_sync()

    assert type(restored) is UIConfig
    assert restored.check_for_updates_on_startup is False
    assert restored.progress == ProgressUIConfig(update_fps=17.0)
    assert restored.agent_bridge.host == "environment-host"
    assert restored.agent_bridge.port == 7997
    assert restored.agent_bridge.enabled is False


def test_agent_bridge_config_rejects_invalid_declared_values() -> None:
    with pytest.raises(ValueError, match="host"):
        AgentUiBridgeConfig(host="")
    with pytest.raises(TypeError, match="port"):
        AgentUiBridgeConfig(port=True)
    with pytest.raises(TypeError, match="transport_mode"):
        AgentUiBridgeConfig(transport_mode="invalid")
    with pytest.raises(TypeError, match="persistent"):
        AgentUiBridgeConfig(persistent=1)
    with pytest.raises(TypeError, match="enabled"):
        AgentUiBridgeConfig(enabled=1)
    with pytest.raises(ValueError, match="bridge_instance_id"):
        AgentUiBridgeConfig(bridge_instance_id="")
    with pytest.raises(TypeError, match="descriptor_file_path"):
        AgentUiBridgeConfig(descriptor_file_path=3)
    with pytest.raises(ValueError, match="poll_timeout_ms"):
        AgentUiBridgeConfig(poll_timeout_ms=0)
    with pytest.raises(ValueError, match="shutdown_timeout_seconds"):
        AgentUiBridgeConfig(shutdown_timeout_seconds=0)


def test_shortcut_config_rejects_noncanonical_qt_sequences() -> None:
    with pytest.raises(ValueError, match="show_plate_manager"):
        ShortcutConfig(show_plate_manager="NoSuchKey")


def test_shortcut_lifecycle_rebinds_live_qactions(qapp) -> None:
    action = QAction("Plate Manager")
    lifecycle = MainWindowShortcutLifecycle(qapp)
    lifecycle.bind_menu_action(
        lambda config: config.show_plate_manager,
        action,
    )

    try:
        lifecycle.apply(
            replace(
                ShortcutConfig(),
                show_plate_manager="Ctrl+1",
            )
        )
        assert action.shortcut().toString() == "Ctrl+1"

        lifecycle.apply(
            replace(
                ShortcutConfig(),
                show_plate_manager="Alt+2",
            )
        )
        assert action.shortcut().toString() == "Alt+2"
    finally:
        lifecycle.close()


def test_shortcut_lifecycle_rejects_duplicates_before_rebinding(qapp) -> None:
    first = QAction("First")
    second = QAction("Second")
    lifecycle = MainWindowShortcutLifecycle(qapp)
    lifecycle.bind_menu_action(lambda config: config.show_plate_manager, first)
    lifecycle.bind_menu_action(lambda config: config.show_pipeline_editor, second)
    lifecycle.apply(ShortcutConfig())
    original = (first.shortcut().toString(), second.shortcut().toString())

    try:
        with pytest.raises(ValueError, match="must be unique"):
            lifecycle.apply(
                replace(
                    ShortcutConfig(),
                    show_plate_manager="Ctrl+1",
                    show_pipeline_editor="Ctrl+1",
                )
            )

        assert (first.shortcut().toString(), second.shortcut().toString()) == original
    finally:
        lifecycle.close()


def test_time_travel_shortcut_accepts_any_configured_qt_key(qapp) -> None:
    calls: list[str] = []
    lifecycle = MainWindowShortcutLifecycle(qapp)
    lifecycle.bind_time_travel_command(
        lambda config: config.time_travel_back,
        "Back",
        lambda: calls.append("back"),
    )

    try:
        lifecycle.apply(
            replace(
                ShortcutConfig(),
                time_travel_back="Ctrl+B",
            )
        )
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_B,
            Qt.KeyboardModifier.ControlModifier,
        )

        assert lifecycle._event_filter.eventFilter(None, event) is True
        assert calls == ["back"]
    finally:
        lifecycle.close()


def test_progress_config_updates_materialized_timer_owner() -> None:
    intervals: list[int] = []
    service = object.__new__(ProgressWorkflowService)
    service._config = ProgressUIConfig(update_fps=30.0)
    service._progress_coalesce_timer = SimpleNamespace(
        setInterval=intervals.append,
    )
    updated = ProgressUIConfig(update_fps=20.0)

    service.update_config(updated)

    assert service._config is updated
    assert intervals == [50]


def test_progress_config_updates_lazy_component_and_live_service() -> None:
    applied: list[ProgressUIConfig] = []
    components = object.__new__(BatchWorkflowComponents)
    components.progress_config = ProgressUIConfig(update_fps=30.0)
    components._progress_workflow = SimpleNamespace(
        update_config=applied.append,
    )
    updated = ProgressUIConfig(update_fps=10.0)

    components.update_progress_config(updated)

    assert components.progress_config is updated
    assert applied == [updated]


def test_removed_ui_config_and_lifecycle_mirrors_do_not_recur() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    config_path = repository_root / "openhcs" / "pyqt_gui" / "config.py"
    main_path = repository_root / "openhcs" / "pyqt_gui" / "main.py"
    workflow_path = (
        repository_root
        / "openhcs"
        / "pyqt_gui"
        / "services"
        / "main_window_workflows.py"
    )
    tui_cache_path = (
        repository_root
        / "openhcs"
        / "textual_tui"
        / "services"
        / "config_cache_adapter.py"
    )
    config_tree = ast.parse(config_path.read_text())
    main_tree = ast.parse(main_path.read_text())
    workflow_tree = ast.parse(workflow_path.read_text())
    tui_tree = ast.parse(tui_cache_path.read_text())

    config_classes = {
        node.name for node in ast.walk(config_tree) if isinstance(node, ast.ClassDef)
    }
    assert {
        "Shortcut",
        "StyleConfig",
        "PlotTheme",
        "WindowConfig",
        "LoggingConfig",
    }.isdisjoint(config_classes)
    config_functions = {
        node.name
        for node in ast.walk(config_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "create_high_performance_config",
        "create_low_resource_config",
    }.isdisjoint(config_functions)

    main_names = {node.id for node in ast.walk(main_tree) if isinstance(node, ast.Name)}
    main_functions = {
        node.name
        for node in ast.walk(main_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "QSettings" not in main_names
    assert "auto_save_timer" not in main_names
    assert {"save_window_state", "restore_window_state"}.isdisjoint(main_functions)

    workflow_classes = {
        node.name for node in ast.walk(workflow_tree) if isinstance(node, ast.ClassDef)
    }
    assert "MainWindowPersistenceSurface" not in workflow_classes

    tui_functions = {
        node.name
        for node in ast.walk(tui_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "get_tui_config_cache",
        "clear_global_config_cache",
        "get_global_config_cache_info",
    }.isdisjoint(tui_functions)

    assert not (
        repository_root
        / "openhcs"
        / "pyqt_gui"
        / "services"
        / "config_cache_adapter.py"
    ).exists()
    bridge_source = (
        repository_root / "openhcs" / "pyqt_gui" / "services" / "ui_bridge_server.py"
    ).read_text()
    assert "DEFAULT_UI_BRIDGE_HOST" not in bridge_source
    assert "DEFAULT_UI_BRIDGE_TRANSPORT" not in bridge_source


def test_false_editor_ui_config_channel_does_not_recur() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    source_paths = (
        repository_root / "openhcs" / "pyqt_gui" / "widgets" / "pipeline_editor.py",
        repository_root
        / "openhcs"
        / "pyqt_gui"
        / "widgets"
        / "step_parameter_editor.py",
        repository_root / "openhcs" / "pyqt_gui" / "windows" / "dual_editor_window.py",
    )
    class_names = {
        "PipelineEditorWidget",
        "StepParameterEditorWidget",
        "DualEditorWindow",
    }

    for path in source_paths:
        tree = ast.parse(path.read_text())
        for class_node in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name in class_names
        ):
            initializers = [
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ]
            assert len(initializers) == 1
            initializer = initializers[0]
            argument_names = {
                argument.arg
                for argument in (
                    *initializer.args.posonlyargs,
                    *initializer.args.args,
                    *initializer.args.kwonlyargs,
                )
            }
            assert "gui_config" not in argument_names
            assert not any(
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == "gui_config"
                for node in ast.walk(class_node)
            )

    main_tree = ast.parse(
        (repository_root / "openhcs" / "pyqt_gui" / "main.py").read_text()
    )
    assert not any(
        isinstance(node, ast.Call)
        and (isinstance(node.func, ast.Name) and node.func.id == "PipelineEditorWidget")
        and any(keyword.arg == "gui_config" for keyword in node.keywords)
        for node in ast.walk(main_tree)
    )
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "gui_config"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "pipeline_editor_widget"
            for target in node.targets
        )
        for node in ast.walk(main_tree)
    )


def test_objectstate_current_global_compatibility_alias_does_not_recur() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    objectstate_root = repository_root / "external" / "ObjectState"
    offenders = [
        path
        for source_root in (
            objectstate_root / "src",
            objectstate_root / "tests",
        )
        for path in source_root.rglob("*.py")
        if "set_current_global_config" in path.read_text()
    ]

    assert offenders == []
