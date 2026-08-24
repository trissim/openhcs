"""Focused pure tests for the MCP thesis live-rehearsal script."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from openhcs.core.config_cache import ConfigCacheSpec, load_config_sync
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.pyqt_gui.services.ui_agent_bridge import UiCodeDocumentSourcePolicy

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "mcp_thesis_demo_live.py"
)
SPEC = importlib.util.spec_from_file_location("mcp_thesis_demo_live", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def test_direct_script_entrypoint_activates_source_checkout_externals() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=SCRIPT_PATH.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the OpenHCS MCP thesis live-demo rehearsal" in completed.stdout


def rehearsal_context(
    tmp_path: Path,
    *,
    napari_port: int = 5001,
    zmq_port: int = 7777,
) -> demo.RunContext:
    return demo.RunContext(
        index=1,
        run_id="test-run",
        run_dir=tmp_path,
        plate_dir=tmp_path / "plate",
        output_plate_dir=tmp_path / "outputs",
        source_path=tmp_path / "orchestrator_config.py",
        descriptor_dir=tmp_path / "ui_bridge",
        napari_port=napari_port,
        zmq_port=zmq_port,
        viewer_timeout_ms=2000,
    )


def test_reused_ui_descriptor_comes_from_the_exact_live_owner(tmp_path: Path) -> None:
    owner = rehearsal_context(tmp_path)
    descriptor = tmp_path / "ui_bridge" / "owned.json"
    descriptor.parent.mkdir()
    descriptor.write_text("{}", encoding="utf-8")
    owner.descriptor_path = descriptor

    class LiveProcess:
        @staticmethod
        def poll() -> None:
            return None

    owner.ui_process = LiveProcess()

    assert demo.require_owned_ui_descriptor(owner) == descriptor


def test_reused_ui_descriptor_rejects_an_exited_owner(tmp_path: Path) -> None:
    owner = rehearsal_context(tmp_path)
    descriptor = tmp_path / "owned.json"
    descriptor.write_text("{}", encoding="utf-8")
    owner.descriptor_path = descriptor

    class ExitedProcess:
        @staticmethod
        def poll() -> int:
            return 1

    owner.ui_process = ExitedProcess()

    with pytest.raises(demo.RehearsalFailure, match="exited before reuse"):
        demo.require_owned_ui_descriptor(owner)


def test_run_directory_resolves_output_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    run_dir = demo.resolved_run_directory(
        Path("mcp_outputs/thesis_demo/live/session"),
        3,
    )

    assert run_dir == tmp_path / "mcp_outputs/thesis_demo/live/session/run_03"
    assert (run_dir / "outputs").is_absolute()


def test_process_detection_requires_an_exact_argv_sequence() -> None:
    module_launch = (
        "/workspace/.venv/bin/python",
        "-m",
        "openhcs.pyqt_gui",
        "--log-level",
        "WARNING",
    )
    monitoring_shell = (
        "/usr/bin/zsh",
        "-c",
        "ps -ef | rg '-m|openhcs.pyqt_gui'",
    )
    sequence = ("-m", "openhcs.pyqt_gui")

    assert demo.argv_contains_sequence(module_launch, sequence)
    assert not demo.argv_contains_sequence(monitoring_shell, sequence)


def test_live_process_conflicts_reject_an_existing_ui_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        demo,
        "matching_pids",
        lambda sequence: [1234] if sequence == ("-m", "openhcs.pyqt_gui") else [],
    )

    with pytest.raises(demo.RehearsalFailure, match="PyQt UI"):
        demo.assert_no_live_process_conflicts(rehearsal_context(tmp_path))


def test_isolated_ui_port_allows_only_the_existing_ui_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checked_ports: list[int] = []
    monkeypatch.setattr(
        demo,
        "assert_isolated_ui_bridge_available",
        checked_ports.append,
    )
    monkeypatch.setattr(
        demo,
        "matching_pids",
        lambda sequence: [1234] if sequence == ("-m", "openhcs.pyqt_gui") else [],
    )
    monkeypatch.setattr(
        demo, "assert_owned_execution_endpoint_available", lambda _ctx: None
    )
    monkeypatch.setattr(
        demo, "assert_owned_viewer_endpoint_available", lambda _ctx: None
    )

    demo.assert_no_live_process_conflicts(
        rehearsal_context(tmp_path),
        isolated_ui_bridge_port=7999,
    )

    assert checked_ports == [7999]


def test_unrelated_runtime_processes_do_not_claim_requested_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checked: list[tuple[str, int]] = []
    monkeypatch.setattr(demo, "assert_isolated_ui_bridge_available", lambda _port: None)
    monkeypatch.setattr(
        demo,
        "matching_pids",
        lambda sequence: (
            [4321]
            if sequence == ("-m", "openhcs.runtime.zmq_execution_server_launcher")
            else []
        ),
    )
    monkeypatch.setattr(
        demo,
        "assert_owned_execution_endpoint_available",
        lambda ctx: checked.append(("execution", ctx.zmq_port)),
    )
    monkeypatch.setattr(
        demo,
        "assert_owned_viewer_endpoint_available",
        lambda ctx: checked.append(("viewer", ctx.napari_port)),
    )

    demo.assert_no_live_process_conflicts(
        rehearsal_context(tmp_path, napari_port=5001, zmq_port=7788),
        isolated_ui_bridge_port=7999,
    )

    assert checked == [("execution", 7788), ("viewer", 5001)]


def test_isolated_ui_bridge_checks_data_and_control_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned_ports: list[int] = []
    checked_ports: list[int] = []

    def cleanup_stale(endpoint, config) -> frozenset[int]:
        cleaned_ports.extend(sorted(endpoint.port_pair(config).ports))
        return frozenset()

    def occupied_ports(endpoint, config) -> frozenset[int]:
        checked_ports.extend(sorted(endpoint.port_pair(config).ports))
        return frozenset()

    monkeypatch.setattr(
        demo.TransportEndpoint,
        "cleanup_stale_addresses",
        cleanup_stale,
    )
    monkeypatch.setattr(demo.TransportEndpoint, "occupied_ports", occupied_ports)

    demo.assert_isolated_ui_bridge_available(7999)

    assert cleaned_ports == [7999, 8999]
    assert checked_ports == [7999, 8999]


def test_isolated_ui_bridge_rejects_an_occupied_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        demo.TransportEndpoint,
        "cleanup_stale_addresses",
        lambda _endpoint, _config: frozenset(),
    )
    monkeypatch.setattr(
        demo.TransportEndpoint,
        "occupied_ports",
        lambda endpoint, config: frozenset((endpoint.port_pair(config).control_port,)),
    )

    with pytest.raises(demo.RehearsalFailure, match="8999"):
        demo.assert_isolated_ui_bridge_available(7999)


def test_isolated_ui_bridge_rejects_an_invalid_control_port() -> None:
    with pytest.raises(demo.RehearsalFailure, match="66000"):
        demo.assert_isolated_ui_bridge_available(65000)


def test_owned_ui_receives_the_explicit_isolated_bridge_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    popen_calls: list[dict[str, object]] = []

    def fake_popen(*args, **kwargs):
        popen_calls.append({"args": args, **kwargs})
        return object()

    monkeypatch.setattr(demo.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        demo.TransportEndpoint,
        "cleanup_stale_addresses",
        lambda *_args, **_kwargs: frozenset(),
    )
    monkeypatch.setattr(
        demo.TransportEndpoint,
        "occupied_ports",
        lambda *_args, **_kwargs: frozenset(),
    )
    context = demo.RunContext(
        index=1,
        run_id="isolated-ui",
        run_dir=tmp_path,
        plate_dir=tmp_path,
        output_plate_dir=tmp_path / "outputs",
        source_path=tmp_path / "orchestrator_config.py",
        descriptor_dir=tmp_path / "ui_bridge",
        napari_port=5001,
        zmq_port=7777,
        viewer_timeout_ms=2000,
    )

    demo.start_ui(context, isolated_bridge_port=7999)

    assert popen_calls[0]["env"]["OPENHCS_UI_BRIDGE_PORT"] == "7999"
    assert popen_calls[0]["env"]["OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR"] == str(
        context.descriptor_dir
    )
    config_cache_file = demo.owned_ui_config_cache_path(context)
    assert popen_calls[0]["env"][
        demo.UIConfigCacheEnvironment.cache_file_path_key
    ] == str(config_cache_file)
    assert popen_calls[0]["env"]["XDG_DATA_HOME"] == str(
        (context.run_dir / "xdg-data").resolve()
    )
    persisted = load_config_sync(
        ConfigCacheSpec(config_type=demo.UIConfig, cache_file=config_cache_file)
    )
    assert persisted is not None
    assert persisted.check_for_updates_on_startup is False
    assert persisted.zmq.default_port == context.zmq_port
    assert context.owns_execution_endpoint


def test_owned_execution_endpoint_uses_generic_exact_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def shutdown_endpoint_on_port(**kwargs):
        calls.append(kwargs)
        return type(
            "ShutdownResult",
            (),
            {"succeeded": True, "endpoint_terminated": True},
        )()

    monkeypatch.setattr(
        demo.ZMQClient,
        "shutdown_endpoint_on_port",
        shutdown_endpoint_on_port,
    )
    context = demo.RunContext(
        index=1,
        run_id="owned-runtime",
        run_dir=tmp_path,
        plate_dir=tmp_path,
        output_plate_dir=tmp_path / "outputs",
        source_path=tmp_path / "orchestrator_config.py",
        descriptor_dir=tmp_path / "ui_bridge",
        napari_port=5001,
        zmq_port=7788,
        viewer_timeout_ms=2000,
        owns_execution_endpoint=True,
    )

    demo.stop_owned_execution_endpoint(context)

    assert calls[0]["port"] == 7788
    assert calls[0]["mode"] is demo.EndpointShutdownMode.FORCE
    assert calls[0]["config"].default_port == 7788
    assert not context.owns_execution_endpoint


def test_owned_ui_replaces_the_demo_side_execution_launcher() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)

    assert "start_zmq" not in {
        node.name for node in module.body if isinstance(node, ast.FunctionDef)
    }
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "zmq_process"
        for node in ast.walk(module)
    )


def test_runtime_wait_scans_the_requested_execution_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def command_json(_ctx, _label, command, **_kwargs):
        commands.append(command)
        return {
            "results": [
                {
                    "payloads": [
                        {
                            "servers": [
                                {
                                    "connection": {"port": 7788},
                                    "reachable": True,
                                    "ready": True,
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(demo, "command_json", command_json)
    context = demo.RunContext(
        index=1,
        run_id="runtime-wait",
        run_dir=tmp_path,
        plate_dir=tmp_path,
        output_plate_dir=tmp_path / "outputs",
        source_path=tmp_path / "orchestrator_config.py",
        descriptor_dir=tmp_path / "ui_bridge",
        napari_port=5001,
        zmq_port=7788,
        viewer_timeout_ms=2000,
    )

    demo.wait_for_runtime(context, 1.0)

    assert commands == [
        demo.mcp_cmd(
            "runtime-scan",
            "7788",
            "--timeout-seconds",
            "20",
        )
    ]


def test_config_document_read_opens_its_declared_window_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = rehearsal_context(tmp_path)
    context.descriptor_path = tmp_path / "bridge.json"
    calls: list[tuple[str, list[str]]] = []
    selected_tabs: list[tuple[str, str]] = []

    def response(tool: str, payload: dict[str, object]) -> dict[str, object]:
        return {
            "results": [
                {
                    "tool": tool,
                    "payloads": [payload],
                }
            ]
        }

    def command_json(_ctx, label, command, **_kwargs):
        calls.append((label, command))
        if label == "open_global_config_for_code_document":
            return response(
                demo.UiNavigateWindowCapability.name,
                {"focused": True, "navigated": False},
            )
        if label == "global_config_code_documents":
            return response(
                "openhcs_ui_list_code_documents",
                {
                    "documents": [
                        {
                            "document_id": "window_code_document:global_config",
                            "widget_id": "global_config",
                            "writable": True,
                        }
                    ]
                },
            )
        if label == "inspect_ui_config_document":
            return response(
                "openhcs_ui_get_code_document",
                {"source": "config = object()\n"},
            )
        raise AssertionError(label)

    monkeypatch.setattr(demo, "command_json", command_json)
    monkeypatch.setattr(
        demo,
        "tree_for_window",
        lambda *_args, **_kwargs: {"actionable_widgets": []},
    )
    monkeypatch.setattr(
        demo,
        "select_structured_tab",
        lambda _ctx, *, window_id, tab_label, **_kwargs: selected_tabs.append(
            (window_id, tab_label)
        ),
    )

    source = demo.exact_config_document_source(
        context,
        window_id=demo.OpenHCSUiWindowId.global_config,
        config_type=demo.UIConfig,
    )

    assert source == "config = object()\n"
    assert [label for label, _command in calls] == [
        "open_global_config_for_code_document",
        "global_config_code_documents",
        "inspect_ui_config_document",
    ]
    navigation_arguments = json.loads(calls[0][1][3])
    assert navigation_arguments["window_id"] == demo.OpenHCSUiWindowId.global_config
    assert navigation_arguments["create_if_missing"] is True
    assert selected_tabs == [
        (demo.OpenHCSUiWindowId.global_config, demo.UIConfig.__name__)
    ]


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (
            {"statistics_scope": "bounded_sample", "sample_shape": [1, 8, 8]},
            64,
        ),
        (
            {
                "statistics_scope": "source_resolution",
                "resolution_shape": [256, 256],
            },
            65_536,
        ),
        ({"statistics_scope": "unknown", "sample_shape": [8, 8]}, None),
    ],
)
def test_source_statistics_count_uses_the_declared_scope(
    sample: dict[str, object],
    expected: int | None,
) -> None:
    assert demo.source_statistics_element_count(sample) == expected


def test_owned_napari_viewer_uses_control_endpoint_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Endpoint:
        released = False

        def in_use(self) -> bool:
            return not self.released

        def wait_until_released(self, *, timeout: float) -> bool:
            assert timeout == demo.OWNED_VIEWER_SHUTDOWN_TIMEOUT_SECONDS
            return False

        def ping(self, *, timeout_ms: int, require_ready: bool) -> bool:
            assert (timeout_ms, require_ready) == (200, False)
            return False

        def force_release_addresses(self) -> None:
            self.released = True

    endpoint = Endpoint()
    request_arguments: list[dict[str, object]] = []

    class Request:
        def __init__(self, **kwargs: object) -> None:
            request_arguments.append(kwargs)

        def send(self) -> Request:
            return self

        def succeeded(self) -> bool:
            return True

    monkeypatch.setattr(demo, "ViewerTransportEndpoint", lambda **kwargs: kwargs)
    monkeypatch.setattr(demo, "ViewerRuntimeEndpoint", lambda **kwargs: endpoint)
    monkeypatch.setattr(demo, "ViewerControlMessageRequest", Request)
    context = demo.RunContext(
        index=1,
        run_id="viewer-cleanup",
        run_dir=tmp_path,
        plate_dir=tmp_path,
        output_plate_dir=tmp_path / "outputs",
        source_path=tmp_path / "orchestrator_config.py",
        descriptor_dir=tmp_path / "ui_bridge",
        napari_port=5001,
        zmq_port=7777,
        viewer_timeout_ms=2000,
        owns_napari_viewer=True,
    )

    demo.stop_owned_viewer(context)

    assert not context.owns_napari_viewer
    assert endpoint.released
    assert request_arguments == [
        {
            "endpoint": endpoint,
            "message_type": "force_shutdown",
            "timeout": 2.0,
        }
    ]


def test_artifact_names_come_from_public_function_contract() -> None:
    detail = {
        "runtime_contract": {
            "artifact_outputs": [
                {"name": "measurements", "kind": "measurements"},
                {"name": "objects", "kind": "object_labels"},
            ]
        }
    }

    assert (
        demo.artifact_name_from_function_detail(detail, artifact_kind="measurements")
        == "measurements"
    )
    assert (
        demo.artifact_name_from_function_detail(detail, artifact_kind="object_labels")
        == "objects"
    )


def test_missing_contract_artifact_is_a_rehearsal_failure() -> None:
    with pytest.raises(demo.RehearsalFailure, match="measurements"):
        demo.artifact_name_from_function_detail(
            {"runtime_contract": {"artifact_outputs": []}},
            artifact_kind="measurements",
        )


def test_query_records_use_the_selected_plate_result_envelope() -> None:
    records = demo.records_from_query_payload(
        {"query": {"records": [{"relative_path": "result.csv"}, "skip"]}}
    )

    assert records == [{"relative_path": "result.csv"}]


def test_window_catalog_uses_flat_canonical_window_ids() -> None:
    assert demo.window_ids_from_catalog(
        [
            {"window_id": demo.OpenHCSUiWindowId.plate_manager},
            {"window_id": demo.OpenHCSUiWindowId.image_browser},
            {"identity": {"window_id": "legacy-nested-shape"}},
        ]
    ) == frozenset(
        {
            demo.OpenHCSUiWindowId.plate_manager,
            demo.OpenHCSUiWindowId.image_browser,
        }
    )


def test_roi_payload_alignment_requires_shared_equal_components() -> None:
    images = [{"components": {"well": "A01", "site": "1"}}]
    aligned_rois = [{"components": {"well": "A01", "site": "1"}}]
    mismatched_rois = [{"components": {"well": "B01", "site": "1"}}]

    assert demo.payload_components_align(images, aligned_rois)
    assert not demo.payload_components_align(images, mismatched_rois)


def test_final_demo_layers_require_exact_producer_identity() -> None:
    final_layer = {
        "route_key": "opaque-final-route",
        "producer_identities": (
            {
                "origin": "pipeline",
                "pipeline_position": demo.FINAL_DEMO_STEP_ROUTE_INDEX,
            },
        ),
    }
    misleading_layer = {
        "route_key": f"step_{demo.FINAL_DEMO_STEP_ROUTE_INDEX}",
        "title": f"{demo.FINAL_DEMO_STEP_DISPLAY_INDEX}. Cell Counting",
        "producer_identities": (
            {
                "origin": "pipeline",
                "pipeline_position": 0,
            },
        ),
    }

    assert demo.final_demo_layers([misleading_layer, final_layer]) == [final_layer]


def test_final_demo_layers_do_not_fallback_to_presentation_strings() -> None:
    with pytest.raises(demo.RehearsalFailure, match="pipeline position"):
        demo.final_demo_layers(
            [
                {
                    "route_key": f"step_{demo.FINAL_DEMO_STEP_ROUTE_INDEX}",
                    "title": f"{demo.FINAL_DEMO_STEP_DISPLAY_INDEX}. Cell Counting",
                }
            ]
        )


def test_structured_tab_target_does_not_require_hidden_tab_content() -> None:
    path_id, target_index = demo.structured_tab_target(
        {
            "root": {
                "class_name": "ActionTabbedWindowBody",
                "path": [],
                "children": [
                    {
                        "class_name": "QTabBar",
                        "path": [0],
                        "child_index": 0,
                        "children": [],
                    },
                    {
                        "class_name": "QStackedWidget",
                        "path": [1],
                        "child_index": 1,
                        "children": [
                            {
                                "class_name": "StepParameterEditorWidget",
                                "path": [1, 0],
                                "child_index": 0,
                                "children": [],
                            },
                            {
                                "class_name": "FunctionListEditorWidget",
                                "path": [1, 1],
                                "child_index": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
            },
            "actionable_widgets": [
                {
                    "class_name": "QTabBar",
                    "path": [0],
                    "path_id": "0",
                    "action_kinds": ["tab_selector"],
                    "item_count": 3,
                    "item_texts": [
                        "Step Settings",
                        "Function Pattern",
                        "Artifacts",
                    ],
                }
            ],
        },
        "Artifacts",
    )

    assert (path_id, target_index) == ("0", 2)


def test_component_selector_requests_full_bounded_tree(monkeypatch) -> None:
    calls = []

    def tree_for_window(_ctx, _scope_id, *, label, max_nodes=400):
        calls.append((label, max_nodes))
        if len(calls) == 1:
            return {"actionable_widgets": []}
        raise RuntimeError("stop after full-tree request")

    monkeypatch.setattr(demo, "navigate_exact_field", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(demo, "tree_for_window", tree_for_window)
    monkeypatch.setattr(demo, "select_structured_tab", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="full-tree request"):
        demo.exercise_component_selector(
            object(),
            scope_id="step-7",
            phase="reverted",
        )

    assert calls == [
        ("reverted_inspect_function_pattern_tab", 400),
        ("reverted_inspect_component_selector_button", 2000),
    ]


def test_component_selector_uses_function_editor_action_bar_structure() -> None:
    action = demo.function_component_button_action(
        {
            "root": {
                "class_name": "QWidget",
                "path": [1],
                "children": [
                    {
                        "class_name": demo.DetachableActionBar.__name__,
                        "object_name": "func_action_buttons_container",
                        "path": [1, 0],
                        "visible": True,
                        "children": [],
                    }
                ],
            },
            "actionable_widgets": [
                {
                    "class_name": "QPushButton",
                    "path": [1, 0, index],
                    "path_id": f"1.0.{index}",
                    "action_kinds": ["button"],
                    "enabled": True,
                }
                for index in range(5)
            ],
        }
    )

    assert action["path_id"] == "1.0.2"


def test_plate_viewer_tabs_derive_metadata_target_from_image_widget() -> None:
    assert demo.plate_viewer_tab_targets(
        {
            "root": {
                "class_name": "PlateViewerWindow",
                "path": [],
                "children": [
                    {
                        "class_name": "QTabBar",
                        "path": [0],
                        "child_index": 0,
                        "children": [],
                    },
                    {
                        "class_name": "QStackedWidget",
                        "path": [1],
                        "child_index": 1,
                        "children": [
                            {
                                "class_name": demo.ImageBrowserWidget.__name__,
                                "path": [1, 0],
                                "child_index": 1,
                                "visible": True,
                                "children": [],
                            },
                            {
                                "class_name": "QWidget",
                                "path": [1, 1],
                                "child_index": 1,
                                "children": [],
                            },
                        ],
                    },
                ],
            },
            "actionable_widgets": [
                {
                    "class_name": "QTabBar",
                    "path": [0],
                    "path_id": "0",
                    "action_kinds": ["tab_selector"],
                    "item_count": 2,
                    "current_index": 0,
                }
            ],
        }
    ) == ("0", 0, 1)


def test_widget_action_acceptance_polls_exact_operation_outcome(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        demo,
        "wait_for_ui_operation",
        lambda ctx, **kwargs: calls.append((ctx, kwargs)),
    )
    context = object()

    demo.require_ui_mutation_completed(
        context,
        {
            "invoked": False,
            "receipt": {
                "accepted": True,
                "bridge_operation_id": "operation-1",
            },
        },
        action_label="Metadata tab",
        completed=False,
        expected_outcome="invoked",
    )

    assert calls == [
        (
            context,
            {
                "operation_id": "operation-1",
                "expected_outcome": "invoked",
                "timeout": 30,
            },
        )
    ]


def test_widget_action_immediate_completion_does_not_poll(monkeypatch) -> None:
    monkeypatch.setattr(
        demo,
        "wait_for_ui_operation",
        lambda *_args, **_kwargs: pytest.fail("completed action must not poll"),
    )

    demo.require_ui_mutation_completed(
        object(),
        {"invoked": True},
        action_label="Metadata tab",
        completed=True,
        expected_outcome="invoked",
    )


def test_ui_operation_wait_is_one_public_terminal_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = demo.RunContext(
        index=1,
        run_id="poll-interval",
        run_dir=tmp_path,
        plate_dir=tmp_path / "plate",
        output_plate_dir=tmp_path / "output",
        source_path=tmp_path / "source.py",
        descriptor_dir=tmp_path / "bridge",
        napari_port=5555,
        zmq_port=7777,
        viewer_timeout_ms=2000,
        descriptor_path=tmp_path / "bridge.json",
    )
    calls: list[tuple[str, list[str], float]] = []

    def command_json(_ctx, label, command, *, timeout):
        calls.append((label, command, timeout))
        return {
            "results": [
                {
                    "tool": "openhcs_ui_wait_for_operation_receipt",
                    "payloads": [
                        {
                            "status": "completed",
                            "outcome": "invoked",
                            "started_at_unix": 1.0,
                            "completed_at_unix": 2.0,
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(demo, "command_json", command_json)

    result = demo.wait_for_ui_operation(
        context,
        operation_id="operation-1",
        expected_outcome="invoked",
        timeout=1,
    )

    assert result["outcome"] == "invoked"
    assert result["started_at_unix"] == 1.0
    assert result["completed_at_unix"] == 2.0
    assert len(calls) == 1
    label, command, command_timeout = calls[0]
    assert label == "ui_operation_wait_operation-1"
    assert command_timeout == 11.0
    assert command[:2] == ["--timeout-seconds", "6.0"]
    joined_command = " ".join(command)
    assert "openhcs_ui_wait_for_operation_receipt" in joined_command
    assert "openhcs_ui_get_operation_status" not in joined_command
    assert "poll_interval_seconds" not in joined_command


def test_plate_action_waits_for_accepted_receipt(monkeypatch) -> None:
    action = demo.PlateManagerAction.VIEW_METADATA
    monkeypatch.setattr(
        demo,
        "plate_action_summary",
        lambda _ctx, _action: {
            "target_scope_ids": ["plate"],
            "selection_revision_token": "revision",
        },
    )
    monkeypatch.setattr(
        demo,
        "ui_connection_arguments",
        lambda _ctx: {"descriptor_file_path": "bridge.json", "timeout_ms": 2000},
    )
    monkeypatch.setattr(
        demo,
        "command_json",
        lambda *_args, **_kwargs: {
            "results": [
                {
                    "tool": demo.UiInvokeActionCapability.name,
                    "payloads": [
                        {
                            "status": "accepted",
                            "receipt": {
                                "accepted": True,
                                "bridge_operation_id": "operation-1",
                            },
                        }
                    ],
                }
            ]
        },
    )
    polls = []
    monkeypatch.setattr(
        demo,
        "wait_for_ui_operation",
        lambda ctx, **kwargs: polls.append((ctx, kwargs)),
    )
    context = object()

    result = demo.invoke_plate_action(context, action)

    assert result["status"] == "accepted"
    assert polls == [
        (
            context,
            {
                "operation_id": "operation-1",
                "expected_outcome": "accepted",
                "timeout": 30,
            },
        )
    ]


def test_official30_source_target_uses_discovered_python_section() -> None:
    document_id, section_id = demo.official30_python_source_target(
        [
            {
                "document": {"document_id": "official30"},
                "section": {"section_id": "case", "title": "ExampleHuman"},
            },
            {
                "document": {"document_id": "official30"},
                "section": {
                    "section_id": "case-openhcs-python",
                    "title": "ExampleHuman OpenHCS Python",
                },
            },
        ]
    )

    assert (document_id, section_id) == ("official30", "case-openhcs-python")


def test_mcp_command_includes_descriptor_connection() -> None:
    descriptor = Path("/tmp/ui_bridge.json")

    command = demo.mcp_cmd("windows", descriptor=descriptor)

    assert command == [
        "windows",
        "--descriptor-file-path",
        str(descriptor),
        "--timeout-ms",
        str(demo.UI_BRIDGE_COMMAND_TIMEOUT_MS),
        "--json",
    ]


def test_execution_runtime_discovery_uses_typed_server_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = rehearsal_context(tmp_path, zmq_port=7777)
    response = {
        "results": [
            {
                "tool": "openhcs_scan_runtime_servers",
                "payloads": [
                    {
                        "servers": [
                            {
                                "server": "ZMQExecutionServer",
                                "reachable": True,
                                "ready": True,
                            }
                        ]
                    }
                ],
            }
        ]
    }
    labels: list[str] = []

    def command_json(_ctx, label, _command, **_kwargs):
        labels.append(label)
        return response

    monkeypatch.setattr(demo, "command_json", command_json)

    demo.verify_execution_runtime_discovery(context)

    assert labels == ["runtime_scan_declared_roles"]


def test_demo_source_saves_source_bindings_through_pipeline_config(
    tmp_path: Path,
) -> None:
    context = demo.RunContext(
        index=1,
        run_id="test",
        run_dir=tmp_path,
        plate_dir=tmp_path / "plate",
        output_plate_dir=tmp_path / "output",
        source_path=tmp_path / "source.py",
        descriptor_dir=tmp_path / "bridge",
        napari_port=5555,
        zmq_port=7777,
        viewer_timeout_ms=2000,
        source_channel_values=("DNA_RAW", "AGP_RAW"),
    )

    baseline_source = demo.demo_source(context, demo.BASELINE_SOURCE_BINDING)
    edited_source = demo.demo_source(context, demo.EDITED_SOURCE_BINDING)

    assert "LazySourceBindingsConfig" in baseline_source
    assert "SourceBindingsConfig()" not in baseline_source
    assert demo.BASELINE_SOURCE_ALIAS in baseline_source
    assert demo.EDITED_SOURCE_ALIAS in edited_source
    assert "group_by=GroupBy.CHANNEL" in baseline_source
    assert "pipeline_config =" not in baseline_source
    assert "pipeline_steps =" not in baseline_source
    assert UiCodeDocumentSourcePolicy().validate(baseline_source) == ()

    states = []
    for source in (baseline_source, edited_source):
        namespace: dict[str, object] = {}
        exec(compile(source, str(context.source_path), "exec"), namespace)
        pipeline_config = namespace["per_plate_configs"][str(context.plate_dir)]
        pipeline_namespace: dict[str, object] = {}
        projected_source = demo.pipeline_document_source(source)
        assert {
            error.message
            for error in UiCodeDocumentSourcePolicy().validate(projected_source)
        } == {
            "Unexpected assignment target: pipeline_config",
            "Unexpected assignment target: pipeline_steps",
        }
        exec(
            compile(projected_source, "pipeline_document.py", "exec"),
            pipeline_namespace,
        )
        assert (
            pipeline_namespace["pipeline_config"]
            is pipeline_namespace["per_plate_configs"][str(context.plate_dir)]
        )
        assert (
            pipeline_namespace["pipeline_steps"]
            is pipeline_namespace["pipeline_data"][str(context.plate_dir)]
        )
        for step in pipeline_namespace["pipeline_steps"]:
            compile_function_pattern(step.func, {}, {})
        bindings = pipeline_config.source_bindings_config.bindings
        states.append(
            (
                type(pipeline_config.source_bindings_config).__name__,
                bindings[0].alias,
                bindings[0].selector.components[0].value,
                bindings[1].selector.components[0].value,
                bindings[0].component_identity[0].value,
            )
        )

    assert states == [
        (
            "LazySourceBindingsConfig",
            demo.BASELINE_SOURCE_ALIAS,
            "DNA_RAW",
            "AGP_RAW",
            demo.BASELINE_CHANNEL_IDENTITY,
        ),
        (
            "LazySourceBindingsConfig",
            demo.EDITED_SOURCE_ALIAS,
            "DNA_RAW",
            "AGP_RAW",
            demo.EDITED_CHANNEL_IDENTITY,
        ),
    ]


def test_applied_document_state_uses_reconstructed_public_objects(
    tmp_path: Path,
) -> None:
    context = demo.RunContext(
        index=1,
        run_id="typed-readback",
        run_dir=tmp_path,
        plate_dir=tmp_path / "plate",
        output_plate_dir=tmp_path / "output",
        source_path=tmp_path / "source.py",
        descriptor_dir=tmp_path / "bridge",
        napari_port=5555,
        zmq_port=7777,
        viewer_timeout_ms=2000,
        source_channel_values=("1", "2"),
    )
    source = demo.demo_source(context, demo.BASELINE_SOURCE_BINDING)

    demo.assert_applied_document_state(
        context,
        source.replace("# MCP source-binding state: baseline\n", ""),
        demo.BASELINE_SOURCE_BINDING,
    )


def test_metadata_cycle_requires_three_rebuilds_without_physical_drift() -> None:
    def metadata(revision: str, channel: str = "1") -> dict[str, object]:
        return {
            "plate_state_revision": revision,
            "detected_microscope_type": "openhcsdata",
            "handler_class": "OpenHCSMicroscopeHandler",
            "metadata_handler_class": "OpenHCSMetadataHandler",
            "components": [
                {
                    "component": "channel",
                    "values": [{"key": channel, "label": None}],
                }
            ],
        }

    baseline = metadata("baseline")
    edited = metadata("edited")
    reverted = metadata("reverted")

    demo.assert_rebuilt_metadata_cycle(baseline, edited, reverted)
    with pytest.raises(demo.RehearsalFailure, match="did not rebuild"):
        demo.assert_rebuilt_metadata_cycle(baseline, baseline, reverted)
    with pytest.raises(demo.RehearsalFailure, match="physical plate component"):
        demo.assert_rebuilt_metadata_cycle(baseline, metadata("edited", "2"), reverted)


def test_final_workflow_state_revision_uses_last_typed_state_event() -> None:
    assert (
        demo.final_workflow_state_revision(
            {
                "results": [
                    {
                        "tool": demo.UiGetStateSurfaceCapability.name,
                        "payloads": [{"current_revision_token": "baseline"}],
                    },
                    {"tool": "workflow", "payloads": [{}]},
                    {
                        "tool": demo.UiGetStateSurfaceCapability.name,
                        "payloads": [{"current_revision_token": "rebuilt"}],
                    },
                ]
            }
        )
        == "rebuilt"
    )


def test_execution_config_comes_from_active_ui_document() -> None:
    source = """from openhcs.pyqt_gui.config import UIConfig
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

config = UIConfig(
    zmq=OpenHCSZMQConfig(
        default_port=7788,
        server_info_timeout_ms=637,
    )
)
"""

    config = demo.execution_config_from_ui_document(
        source,
        expected_port=7788,
    )

    assert config.server_info_timeout_ms == 637
    assert type(config) is demo.OpenHCSZMQConfig


@pytest.mark.parametrize("advertised_scope", ["", demo.OpenHCSUiWindowId.global_config])
def test_managed_window_action_uses_advertised_global_window_scope(
    monkeypatch: pytest.MonkeyPatch,
    advertised_scope: str,
) -> None:
    monkeypatch.setattr(
        demo,
        "command_json",
        lambda *_args, **_kwargs: {
            "results": [
                {
                    "tool": demo.UiListActionsCapability.name,
                    "payloads": [
                        {
                            "actions": [
                                {
                                    "widget_id": (
                                        demo.ManagedWindowWidgetIdentity.require_value()
                                    ),
                                    "action_id": (
                                        demo.ManagedWindowAction.SAVE_WITHOUT_CLOSE.value
                                    ),
                                    "enabled": True,
                                    "target_scope_ids": [advertised_scope],
                                }
                            ]
                        }
                    ],
                }
            ]
        },
    )

    context = type("Context", (), {"descriptor_path": Path("/tmp/bridge.json")})()
    summary, target_scope_id = demo.managed_window_action_target(
        context,
        demo.ManagedWindowAction.SAVE_WITHOUT_CLOSE,
        window_id=demo.OpenHCSUiWindowId.global_config,
    )

    assert summary["target_scope_ids"] == [advertised_scope]
    assert target_scope_id == advertised_scope


def test_runtime_lock_refuses_a_concurrent_holder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        demo,
        "official_runtime_lock_path",
        lambda: tmp_path / "runtime.lock",
    )

    def context(run_id: str) -> demo.RunContext:
        return demo.RunContext(
            index=1,
            run_id=run_id,
            run_dir=tmp_path / run_id,
            plate_dir=tmp_path / "plate",
            output_plate_dir=tmp_path / "output",
            source_path=tmp_path / "source.py",
            descriptor_dir=tmp_path / "bridge",
            napari_port=5555,
            zmq_port=7777,
            viewer_timeout_ms=2000,
        )

    first = context("first")
    second = context("second")
    demo.acquire_runtime_lock(first)
    try:
        with pytest.raises(demo.RehearsalFailure, match="owned by another process"):
            demo.acquire_runtime_lock(second)
    finally:
        demo.release_runtime_lock(first)

    demo.acquire_runtime_lock(second)
    demo.release_runtime_lock(second)


def test_runtime_lock_path_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert demo.official_runtime_lock_path() == (
        tmp_path / "openhcs" / "official30-runtime.lock"
    )


def test_runtime_lock_path_rejects_relative_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")

    with pytest.raises(demo.RehearsalFailure, match="absolute"):
        demo.official_runtime_lock_path()


def test_results_projection_requires_declared_measurement_and_populated_table() -> None:
    projection = demo.results_measurement_projection(
        {
            "root": {
                "class_name": demo.LiveMeasurementsWindow.__name__,
                "visible": True,
                "children": [
                    {
                        "class_name": "QListWidget",
                        "object_name": "LiveResultsEntryList",
                        "visible": True,
                        "item_count": 1,
                        "current_text": "Cell Counting: cell_count (A01)",
                        "children": [],
                    },
                    {
                        "class_name": "QLabel",
                        "text": "1 row(s) | 2 column(s) | axis A01",
                        "visible": True,
                        "children": [],
                    },
                    {
                        "class_name": "QTableWidget",
                        "object_name": "LiveResultsTable",
                        "visible": True,
                        "children": [],
                    },
                ],
            }
        },
        measurement_name="cell_count",
    )

    assert projection["entry_count"] == 1
    assert projection["status"].startswith("1 row(s)")
    with pytest.raises(demo.RehearsalFailure, match="declared measurement"):
        demo.results_measurement_projection(
            {
                "root": {
                    "class_name": demo.LiveMeasurementsWindow.__name__,
                    "visible": True,
                    "children": [
                        {
                            "class_name": "QListWidget",
                            "object_name": "LiveResultsEntryList",
                            "visible": True,
                            "item_count": 1,
                            "current_text": "other",
                            "children": [],
                        },
                        {
                            "class_name": "QLabel",
                            "text": "1 row(s) | 2 column(s)",
                            "visible": True,
                            "children": [],
                        },
                        {
                            "class_name": "QTableWidget",
                            "object_name": "LiveResultsTable",
                            "visible": True,
                            "children": [],
                        },
                    ],
                }
            },
            measurement_name="cell_count",
        )


def test_forbidden_fuzzy_ui_helpers_are_absent() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "def action_matches_terms" not in source
    assert "def open_metadata_surface" not in source
    assert "def artifact_action_for_tree" not in source


def test_authoring_context_evidence_requires_exact_untruncated_next_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "results": [
            {
                "tool": demo.GetAuthoringContextCapability.name,
                "payloads": [
                    {
                        "schema_version": "1",
                        "kind": "first_use",
                        "content": "Choose intent.\nNext: call openhcs_get_authoring_context.",
                    }
                ],
            }
        ]
    }
    commands: list[list[str]] = []

    def command_json(_ctx, _label, command, **_kwargs):
        commands.append(command)
        return payload

    monkeypatch.setattr(demo, "command_json", command_json)

    evidence = demo.inspect_authoring_context(
        object(),
        kind="first_use",
        label="first_use",
    )

    assert evidence == {
        "kind": "first_use",
        "schema_version": "1",
        "character_count": 56,
        "next_action": "Next: call openhcs_get_authoring_context.",
        "truncated": False,
    }
    assert commands[0][commands[0].index("--max-chars") + 1] == "16000"

    payload["results"][0]["payloads"][0]["content"] += "\n...<truncated 10 chars>"
    with pytest.raises(demo.RehearsalFailure, match="truncated"):
        demo.inspect_authoring_context(
            object(),
            kind="first_use",
            label="first_use",
        )


def test_source_inspection_uses_virtual_identity_and_bounded_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    responses = iter(
        (
            {
                "results": [
                    {
                        "tool": demo.InspectPlatePathCapability.name,
                        "payloads": [
                            {
                                "plate_path": str(tmp_path),
                                "status": "ok",
                                "detected_microscope_type": "openhcsdata",
                                "handler_class": "OpenHCSMicroscopeHandler",
                                "image_files": {
                                    "count": 2,
                                    "sampled_records": [
                                        {
                                            "virtual_path": "A01_w1.tif",
                                            "source_path": "/physical/A01_w1.tif",
                                        }
                                    ],
                                },
                                "parse_summary": {
                                    "attempted_file_count": 2,
                                    "parsed_file_count": 2,
                                },
                                "components": [
                                    {
                                        "component": "channel",
                                        "count": 2,
                                        "values": [
                                            {"key": "DNA"},
                                            {"key": "AGP"},
                                        ],
                                    }
                                ],
                                "source_diagnostics": [],
                                "workflow_advice": {
                                    "ingestion_route": "detected_handler",
                                    "ingestion_owner": "openhcsdata",
                                    "source_binding_role": "semantic_selection",
                                    "ui_code_document_id": (
                                        demo.ORCHESTRATOR_DOCUMENT_ID
                                    ),
                                },
                            }
                        ],
                    }
                ]
            },
            {
                "results": [
                    {
                        "tool": demo.SamplePlateImageCapability.name,
                        "payloads": [
                            {
                                "virtual_path": "A01_w1.tif",
                                "source_path": "/physical/A01_w1.tif",
                                "shape": [1, 1024, 1024],
                                "resolution_shape": [1, 256, 256],
                                "selected_resolution_index": 2,
                                "resolution_count": 3,
                                "downsample_yx": [4.0, 4.0],
                                "statistics_scope": "source_resolution",
                                "sample_shape": [1, 8, 8],
                                "sample_included": True,
                                "dtype": "uint16",
                                "minimum": 1,
                                "maximum": 4,
                                "mean": 2.5,
                            }
                        ],
                    }
                ]
            },
        )
    )
    commands: list[list[str]] = []

    def command_json(_ctx, _label, command, **_kwargs):
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(demo, "command_json", command_json)
    context = type("Context", (), {"plate_dir": tmp_path})()

    evidence = demo.inspect_source_plate_and_sample(context)

    assert evidence["sample"]["sample_element_count"] == 64
    assert evidence["sample"]["statistics_element_count"] == 65_536
    assert (
        evidence["sample"]["statistics_element_budget"]
        == demo.SOURCE_RESOLUTION_STATISTICS_MAX_ELEMENTS
    )
    assert evidence["sample"]["selected_resolution_index"] == 2
    assert evidence["inspection"]["channel_values"] == ["DNA", "AGP"]
    assert commands[0][commands[0].index("--timeout-seconds") + 1] == "40"
    assert commands[1][0] == demo.SamplePlateImageCapability.cli_command
    assert "--resolution-index" not in commands[1]
    assert commands[1][commands[1].index("--max-array-elements") + 1] == "64"
    assert commands[1][commands[1].index("--max-auto-resolution-size") + 1] == "1024"


def test_viewer_sampling_enforces_the_element_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "results": [
            {
                "tool": demo.SampleViewerWindowImageCapability.name,
                "payloads": [
                    {
                        "observed": True,
                        "sample_protocol_supported": True,
                        "record_count": 1,
                        "sample_included_count": 1,
                        "records": [
                            {
                                "payload_route_key": "image:0",
                                "layer_route_key": "image",
                                "path": "virtual/A01.tif",
                                "summary": {"shape": [64, 64], "nonzero_count": 9},
                                "array_value_summary": {
                                    "included": True,
                                    "shape": [8, 8],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(demo, "command_json", lambda *_args, **_kwargs: payload)
    context = type("Context", (), {"napari_port": 5555})()

    evidence = demo.sample_viewer_image_bounded(
        context,
        route_key="image",
        axis_indices={"well": 0},
        viewer_timeout_ms=2000,
    )

    assert evidence["records"][0]["sample_element_count"] == 64
    payload["results"][0]["payloads"][0]["records"][0]["array_value_summary"][
        "shape"
    ] = [9, 8]
    with pytest.raises(demo.RehearsalFailure, match="exceeded"):
        demo.sample_viewer_image_bounded(
            context,
            route_key="image",
            axis_indices={"well": 0},
            viewer_timeout_ms=2000,
        )


def test_artifact_plan_uses_the_same_ui_authored_pipeline_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    payload = {
        "results": [
            {
                "tool": demo.InspectPipelineSourceArtifactPlanCapability.name,
                "payloads": [
                    {
                        "axis_count": 1,
                        "step_count": 1,
                        "progress_event_count": 2,
                        "source_workspace": {
                            "file_count": 1,
                            "truncated_file_count": 0,
                            "files": [
                                {
                                    "virtual_path": "virtual/A01.tif",
                                    "source_path": "/physical/A01.tif",
                                }
                            ],
                        },
                        "steps": [
                            {
                                "artifact_outputs": [
                                    {
                                        "name": "measurements",
                                        "kind": "measurements",
                                        "materialization": {
                                            "persistent_backend": "disk"
                                        },
                                    },
                                    {
                                        "name": "objects",
                                        "kind": "object_labels",
                                        "materialization": {
                                            "persistent_backend": "memory"
                                        },
                                    },
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
    }

    def command_json(_ctx, _label, command, **_kwargs):
        commands.append(command)
        return payload

    monkeypatch.setattr(demo, "command_json", command_json)
    context = type(
        "Context",
        (),
        {
            "plate_dir": tmp_path / "plate",
            "source_path": tmp_path / "source.py",
            "run_dir": tmp_path,
        },
    )()
    contracts = demo.DemoArtifactContracts(
        function_id="openhcs:function",
        measurement_name="measurements",
        roi_name="objects",
    )

    evidence = demo.inspect_compiled_artifact_plan(
        context,
        contracts,
        axis_id="A01",
    )

    assert evidence["required_artifacts"] == [
        {"name": "measurements", "kind": "measurements"},
        {"name": "objects", "kind": "object_labels"},
    ]
    assert evidence["materialization_backends"] == ["disk", "memory"]
    assert commands == [
        demo.mcp_cmd(
            demo.InspectPipelineSourceArtifactPlanCapability.cli_command,
            context.plate_dir,
            "--source-file",
            demo.pipeline_document_source_path(context),
            "--axis-filter",
            "A01",
        )
    ]


def complete_objective_report() -> dict[str, object]:
    contexts = {
        kind: {
            "kind": kind,
            "schema_version": "1",
            "character_count": 100,
            "next_action": "Next: openhcs_public_action",
            "truncated": False,
        }
        for kind in (
            "first_use",
            "ui_visible_workflow",
            "pipeline",
            "viewer_review",
        )
    }

    def authored(alias: str) -> dict[str, object]:
        return {
            "step_count": 2,
            "source_bindings": [
                {
                    "alias": alias,
                    "selector_components": [{"component": "channel", "value": "1"}],
                },
                {
                    "alias": "AGP",
                    "selector_components": [{"component": "channel", "value": "2"}],
                },
            ],
            "processing_semantics": [
                {"group_by": None, "variable_components": ["z_index"]},
                {"group_by": "channel", "variable_components": []},
            ],
            "config_families": {
                "pipeline": "PipelineConfig",
                "path_planning": "LazyPathPlanningConfig",
                "source_bindings": "LazySourceBindingsConfig",
                "napari_streaming": "LazyNapariStreamingConfig",
            },
        }

    surfaces = {
        "image_browser": {},
        "metadata_browser": {},
        "function_list_editor": {},
        "group_by_component_selector": {},
    }
    return {
        "authoring_contexts": contexts,
        "source_inspection": {
            "inspection": {
                "image_count": 2,
                "parsed_image_count": 2,
                "channel_values": ["1", "2"],
            },
            "sample": {
                "sample_element_count": 64,
                "statistics_element_count": 64,
                "statistics_element_budget": 64,
            },
        },
        "guided_tour": {
            "knowledge_document_count": 3,
            "function_contract_id": "openhcs:function",
            "official_source": {"complete_pipeline_document": True},
        },
        "authoring_schemas": {
            "capability": demo.DescribeConfigSchemaCapability.name,
            "probe_count": (
                len(demo.AUTHORING_SCHEMA_ROOT_PROBES)
                + len(demo.AUTHORING_SCHEMA_PROBES)
            ),
            "probes": [
                {
                    "requested_config_type": config_type,
                    "path_prefix": path_prefix,
                }
                for config_type, path_prefix, _required_paths in (
                    tuple(
                        (config_type, None, required_paths)
                        for config_type, required_paths in (
                            demo.AUTHORING_SCHEMA_ROOT_PROBES
                        )
                    )
                    + demo.AUTHORING_SCHEMA_PROBES
                )
            ],
        },
        "pipeline_authoring": {
            "baseline": authored("DNA"),
            "edited": authored("DNA_EDITED"),
            "reverted": authored("DNA"),
        },
        "rebuilt_metadata": {
            "baseline": {"plate_state_revision": "r1"},
            "edited": {"plate_state_revision": "r2"},
            "reverted": {"plate_state_revision": "r3"},
            "edited_views": {"surfaces": surfaces},
            "reverted_views": {"surfaces": surfaces},
        },
        "contracts": {
            "function_id": "openhcs:function",
            "measurement_name": "measurements",
            "roi_name": "objects",
        },
        "artifact_plan": {
            "artifact_outputs": [
                {"name": "measurements", "kind": "measurements"},
                {"name": "objects", "kind": "object_labels"},
            ],
            "materialization_contract_count": 2,
            "source_workspace": {
                "file_count": 3,
                "virtual_paths": ["virtual/A01.tif"],
                "source_paths": ["/physical/A01.tif"],
                "truncated_file_count": 2,
            },
        },
        "execution": {
            "terminal_status": "complete",
            "compile_state_revision": "compile-r1",
            "execution_state_revision": "run-r1",
            "output_plate_root": "/output",
        },
        "measurement_snapshot": {"row_count": 1},
        "results_window": {"entry_count": 1},
        "roi_streaming": {"roi_count": 1},
        "viewer": {
            "payload_record_count": 1,
            "bounded_sample": {
                "records": [{"sample_element_count": 64}],
            },
        },
    }


def test_objective_evidence_maps_every_public_workflow_requirement() -> None:
    evidence = demo.build_objective_evidence(complete_objective_report())

    assert evidence["complete"] is True
    assert evidence["requirement_count"] == 9
    assert {row["requirement"] for row in evidence["requirements"]} == {
        "progressive_task_specific_onboarding",
        "authoritative_hierarchical_authoring_schemas",
        "source_inspection_and_safe_sampling",
        "ui_visible_hierarchical_configuration",
        "typed_source_bindings_and_multidimensional_pipeline",
        "compile_plan_and_artifact_contracts",
        "storage_independent_execution_and_results",
        "viewer_streaming_and_bounded_visual_validation",
        "no_hidden_repository_knowledge",
    }


def test_objective_evidence_rejects_source_statistics_above_the_budget() -> None:
    report = complete_objective_report()
    report["source_inspection"]["sample"]["statistics_element_count"] = 65

    with pytest.raises(demo.RehearsalFailure, match="bounded pixel evidence"):
        demo.build_objective_evidence(report)


@pytest.mark.parametrize(
    "missing_key",
    (
        "authoring_contexts",
        "source_inspection",
        "guided_tour",
        "authoring_schemas",
        "pipeline_authoring",
        "rebuilt_metadata",
        "contracts",
        "artifact_plan",
        "execution",
        "measurement_snapshot",
        "results_window",
        "roi_streaming",
        "viewer",
    ),
)
def test_objective_evidence_fails_closed_for_every_stage(missing_key: str) -> None:
    report = deepcopy(complete_objective_report())
    del report[missing_key]

    with pytest.raises(demo.RehearsalFailure, match="evidence|validation|incomplete"):
        demo.build_objective_evidence(report)
