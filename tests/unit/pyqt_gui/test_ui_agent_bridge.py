from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from openhcs.agent.dto.ui_bridge import (
    UiActionInvokeRequest,
    UiBridgeConfirmationRequirement,
    UiBridgeConnectionSpec,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentId,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationRequest,
    UiStateSurfaceId,
    UiStateSurfaceRequest,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.ui_bridge_service import (
    UiBridgeConnectionResolution,
    UiBridgeService,
)
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_agent_bridge import UiAgentBridgeService
from openhcs.pyqt_gui.services.ui_bridge_server import (
    UiBridgeControlServer,
    UiBridgeDescriptorPathRequest,
    UiBridgeServerAuthSeed,
    UiBridgeServerConfig,
    UiBridgeServerIdentitySeed,
)
from openhcs.core.progress.projection import ExecutionRuntimeProjection
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ExecutionBatchRuntime,
    ManagerExecutionState,
)
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerAction
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionRoute,
)


DOCUMENT_ID = UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value
PLATE_SCOPE_ID = "plate-1"
PLATE_NAME = "plate 1"
ALL_SELECTION_MODE = UiCodeDocumentSelectionMode.ALL.value
SELECTED_SELECTION_MODE = UiCodeDocumentSelectionMode.SELECTED.value
BRIDGE_INSTANCE_ID = "bridge-test"
BRIDGE_AUTH_TOKEN = "secret-token"
VALID_SOURCE = (
    f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
    f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
)


class FakeEmptySelectionPolicy(str, Enum):
    ERROR = "error"


def _bridge_server_config(directory_path: Path) -> UiBridgeServerConfig:
    return UiBridgeServerConfig(
        descriptor_path_request=UiBridgeDescriptorPathRequest(
            directory_path=directory_path
        ),
        identity_seed=UiBridgeServerIdentitySeed(BRIDGE_INSTANCE_ID),
        auth_seed=UiBridgeServerAuthSeed(BRIDGE_AUTH_TOKEN),
    )


def _json_payload_values(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _json_payload_values(value)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _json_payload_values(value)
        return
    yield payload


@dataclass
class Dummy:
    x: int = 1


@dataclass(frozen=True, slots=True)
class FakeRow:
    scope_id: str
    name: str
    plate_root: str = f"/tmp/{PLATE_SCOPE_ID}"
    cppipe_path: str | None = None


@dataclass(frozen=True, slots=True)
class FakeCodeDocumentContext:
    source: str
    selected_scope_ids: tuple[str, ...]


class FakeOperations:
    def __init__(self, state: ObjectState | None = None) -> None:
        self.state = state
        self.pre_count = 0
        self.post_count = 0
        self.applied_namespaces: list[dict] = []

    @contextmanager
    def patch_lazy_constructors(self):
        yield

    def migrate_code_namespace(self, code, error, namespace):
        del code, error, namespace
        return None

    def apply_code_namespace(self, namespace: dict) -> bool:
        self.applied_namespaces.append(namespace)
        if self.state is not None:
            self.state.update_parameter("x", self.state.parameters["x"] + 1)
        return True

    def pre_code_execution(self) -> None:
        self.pre_count += 1

    def post_code_execution(self) -> None:
        self.post_count += 1
        ObjectStateRegistry.increment_token()


@dataclass(frozen=True, slots=True)
class FakeButton:
    enabled: bool = True

    def isEnabled(self) -> bool:
        return self.enabled


class FakeServiceAdapter:
    def execute_async_operation(self, operation):
        raise AssertionError(f"Unexpected async operation in test: {operation!r}")


class InlineDispatcher:
    def call(self, callback, *, timeout_ms: int = 5000):
        del timeout_ms
        return callback()


class FakePlateManager:
    BUTTON_CONFIGS = [
        ("Code", PlateManagerAction.CODE_PLATE.value, "Generate Python code"),
    ]
    ACTION_ROUTES = {
        PlateManagerAction.CODE_PLATE: WidgetActionRoute(
            PlateManagerAction.CODE_PLATE,
            lambda widget: widget.action_code_plate,
        ),
    }

    def __init__(
        self,
        *,
        selected: tuple[FakeRow, ...] = (),
        plates: tuple[FakeRow, ...] = (FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
        operations: FakeOperations | None = None,
    ) -> None:
        self.selected = list(selected)
        self.plates = list(plates)
        self.operations = operations or FakeOperations()
        self.execution_state = ManagerExecutionState.IDLE
        self.plate_execution_ids = {}
        self.runtime_progress_projection = ExecutionRuntimeProjection()
        self.plate_terminal_activity_status = ExecutionBatchRuntime()
        self.plate_init_pending = set()
        self.plate_compile_pending = set()
        self.plate_compiled_data = {}
        self.execution_server_info = None
        self.service_adapter = FakeServiceAdapter()
        self.buttons = {
            PlateManagerAction.CODE_PLATE.value: FakeButton(),
        }
        self.code_action_count = 0

    def get_selected_items(self):
        return list(self.selected)

    def orchestrator_code_document_context(
        self,
        *,
        selection_mode: str = SELECTED_SELECTION_MODE,
        empty_selection_policy: str = FakeEmptySelectionPolicy.ERROR.value,
    ) -> FakeCodeDocumentContext:
        rows_by_mode = {
            UiCodeDocumentSelectionMode.ALL: self.plates,
            UiCodeDocumentSelectionMode.SELECTED: self.selected,
        }
        rows = rows_by_mode[UiCodeDocumentSelectionMode(selection_mode)]
        if (
            not rows
            and FakeEmptySelectionPolicy(empty_selection_policy)
            is FakeEmptySelectionPolicy.ERROR
        ):
            raise ValueError("No plates selected.")
        return FakeCodeDocumentContext(
            source=VALID_SOURCE,
            selected_scope_ids=tuple(row.scope_id for row in rows),
        )

    def code_document_execution_operations(self) -> FakeOperations:
        return self.operations

    def action_code_plate(self) -> None:
        self.code_action_count += 1


@pytest.fixture(autouse=True)
def reset_object_state_registry():
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
    ObjectStateRegistry._token = 0


def test_atomic_success_does_not_record_snapshot_on_failure() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    with pytest.raises(RuntimeError):
        with ObjectStateRegistry.atomic_success("edit failing", state.scope_id):
            state.update_parameter("x", 2)
            raise RuntimeError("boom")

    assert ObjectStateRegistry.get_branch_history() == []


def test_selected_read_fails_loudly_when_no_plate_is_selected() -> None:
    bridge = UiAgentBridgeService(plate_manager=FakePlateManager())

    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=SELECTED_SELECTION_MODE,
        )
    )

    assert document.errors
    assert document.errors[0].code == "ui_code_document_read_failed"


def test_all_read_returns_source_hash_and_revision() -> None:
    bridge = UiAgentBridgeService(plate_manager=FakePlateManager())

    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    assert document.source == VALID_SOURCE
    assert document.size_bytes == len(VALID_SOURCE.encode("utf-8"))
    assert document.sha256
    assert document.current_revision_token
    assert document.selected_scope_ids == (PLATE_SCOPE_ID,)


def test_plate_manager_state_surface_projects_runtime_row_status() -> None:
    manager = FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),))
    manager.plate_compile_pending.add(PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(plate_manager=manager)

    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
        )
    )
    poll_state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
            selection_mode=ALL_SELECTION_MODE,
            base_revision_token=state.current_revision_token,
        )
    )

    assert state.summary.surface_id == UiStateSurfaceId.PLATE_MANAGER.value
    row = state.payload["rows"][0]
    assert row["plate_scope_id"] == PLATE_SCOPE_ID
    assert row["status_prefix"] == "⏳ Compile"
    assert row["compile_pending"] is True
    assert row["selected"] is True
    assert poll_state.unchanged is True


def test_plate_manager_action_catalog_token_can_guard_invoke() -> None:
    manager = FakePlateManager(
        selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),),
    )
    bridge = UiAgentBridgeService(plate_manager=manager)

    action = bridge.list_actions().actions[0]
    accepted = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=action.identity.widget_id,
            action_id=action.identity.action_id,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token=action.selection_revision_token,
        )
    )
    stale = bridge.invoke_action(
        UiActionInvokeRequest(
            widget_id=action.identity.widget_id,
            action_id=action.identity.action_id,
            selected_scope_ids=action.target_scope_ids,
            observed_selection_revision_token="stale-token",
        )
    )

    assert action.selection_revision_token
    assert accepted.status == "accepted"
    assert accepted.selection_revision_token == action.selection_revision_token
    assert manager.code_action_count == 1
    assert stale.status == "rejected"
    assert stale.errors
    assert stale.errors[0].code == "stale_ui_action_revision"


def test_validation_rejects_side_effecting_source_before_execution() -> None:
    bridge = UiAgentBridgeService(plate_manager=FakePlateManager())

    result = bridge.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source="open('/tmp/openhcs-mcp-side-effect', 'w')\n",
        )
    )

    assert not result.valid
    assert result.errors
    assert result.errors[0].code == "unsafe_statement"


def test_validation_rejects_legacy_pipeline_config_assignment() -> None:
    bridge = UiAgentBridgeService(plate_manager=FakePlateManager())

    result = bridge.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source=(
                "from openhcs.core.config import PipelineConfig\n"
                f"plate_paths = ['{PLATE_SCOPE_ID}']\n"
                "pipeline_config = PipelineConfig()\n"
                f"pipeline_data = {{'{PLATE_SCOPE_ID}': []}}\n"
            ),
        )
    )

    assert not result.valid
    assert any(error.code == "unexpected_assignment" for error in result.errors)


def test_apply_creates_baseline_and_edit_snapshot() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    operations = FakeOperations(state)
    bridge = UiAgentBridgeService(
        plate_manager=FakePlateManager(operations=operations)
    )
    document = bridge.get_document(
        UiCodeDocumentRequest(
            document_id=DOCUMENT_ID,
            selection_mode=ALL_SELECTION_MODE,
        )
    )

    result = bridge.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=VALID_SOURCE,
            base_revision_token=document.current_revision_token,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    history = ObjectStateRegistry.get_branch_history()
    assert result.applied
    assert operations.pre_count == 1
    assert operations.post_count == 1
    assert [snapshot.label for snapshot in history] == [
        "init",
        f"edit {DOCUMENT_ID} via MCP [{PLATE_SCOPE_ID}]",
    ]
    assert result.pre_apply_snapshot is None
    assert result.post_apply_snapshot is not None


def test_bridge_lists_and_restores_snapshots() -> None:
    state = ObjectState(Dummy(), scope_id=PLATE_SCOPE_ID)
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    ObjectStateRegistry.record_snapshot("before", scope_id=PLATE_SCOPE_ID)
    before_id = ObjectStateRegistry.get_branch_history()[-1].id
    state.update_parameter("x", 2)
    ObjectStateRegistry.record_snapshot("after", scope_id=PLATE_SCOPE_ID)
    bridge = UiAgentBridgeService(plate_manager=FakePlateManager())

    catalog = bridge.list_snapshots(UiSnapshotListRequest())
    restore = bridge.restore_snapshot(
        UiSnapshotRestoreRequest(
            snapshot_id=before_id,
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
        )
    )

    assert f"before [{PLATE_SCOPE_ID}]" in [
        snapshot.label for snapshot in catalog.snapshots
    ]
    assert f"after [{PLATE_SCOPE_ID}]" in [
        snapshot.label for snapshot in catalog.snapshots
    ]
    assert restore.restored
    assert restore.current_snapshot is not None
    assert restore.current_snapshot.snapshot_id == before_id


def test_ui_bridge_control_server_round_trips_documents_through_descriptor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    bridge = UiAgentBridgeService(
        plate_manager=FakePlateManager(selected=(FakeRow(PLATE_SCOPE_ID, PLATE_NAME),)),
        dispatcher=InlineDispatcher(),
    )
    server = UiBridgeControlServer(
        bridge,
        _bridge_server_config(tmp_path),
    )

    binding = server.start()
    try:
        service = UiBridgeService()

        status = service.status()
        catalog = service.list_documents()
        state_catalog = service.list_state_surfaces()
        document = service.get_document(
            UiCodeDocumentRequest(
                document_id=DOCUMENT_ID,
                selection_mode=ALL_SELECTION_MODE,
            )
        )
        state = service.get_state_surface(
            UiStateSurfaceRequest(
                surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
                selection_mode=ALL_SELECTION_MODE,
            )
        )

        assert status.reachable is True
        assert status.auth_required is True
        assert status.descriptor_status == "ok"
        assert status.bridge_instance_id == BRIDGE_INSTANCE_ID
        assert BRIDGE_AUTH_TOKEN not in set(_json_payload_values(to_jsonable(status)))
        assert catalog.documents[0].document_id == DOCUMENT_ID
        assert state_catalog.surfaces[0].surface_id == UiStateSurfaceId.PLATE_MANAGER.value
        assert document.source == VALID_SOURCE
        assert state.payload["rows"][0]["plate_scope_id"] == PLATE_SCOPE_ID
        assert binding.descriptor_file_path.exists()
    finally:
        server.stop()

    assert not binding.descriptor_file_path.exists()


def test_ui_bridge_control_server_preserves_bad_auth_error(tmp_path: Path) -> None:
    bridge = UiAgentBridgeService(dispatcher=InlineDispatcher())
    server = UiBridgeControlServer(
        bridge,
        _bridge_server_config(tmp_path),
    )

    binding = server.start()
    try:
        bad_connection = UiBridgeConnectionSpec(
            host=binding.connection.host,
            port=binding.connection.port,
            transport_mode=binding.connection.transport_mode,
            auth_token="wrong-token",
        )
        service = UiBridgeService(
            descriptor_resolver=_StaticUiBridgeDescriptorResolver(bad_connection)
        )

        catalog = service.list_documents()

        assert catalog.documents == ()
        assert catalog.errors[0].code == "ui_bridge_auth_failed"
    finally:
        server.stop()


class _StaticUiBridgeDescriptorResolver:
    def __init__(self, connection: UiBridgeConnectionSpec) -> None:
        self._connection = connection

    def resolve(
        self,
        connection: UiBridgeConnectionSpec | None,
    ) -> UiBridgeConnectionResolution:
        del connection
        return UiBridgeConnectionResolution(connection=self._connection)
