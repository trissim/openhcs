from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from openhcs.agent.dto.common import AgentResourceRef, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeConfirmationRequirement,
    UiBridgeConnectionSpec,
    UiCodeDocumentId,
    UiBridgeOperationIdentity,
    UiBridgeOperationRef,
    UiBridgeOperationRoute,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentIdentity,
    UiCodeDocumentRequest,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiMutationReceipt,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiSelectedPlateWorkflowKind,
    UiSelectedPlateWorkflowRequest,
    UiSelectedPlateWorkflowResult,
    UiStateSurfaceId,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceIdentity,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRef,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiTimeTravelHeadRequest,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowIdentity,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowOpenPolicy,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWindowSummary,
    UiWidgetRect,
    UiWidgetTreeNode,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
)
from openhcs.agent.services.ui_bridge_service import (
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeGatewayABC,
    UiBridgeService,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
)


DOCUMENT_ID = UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value
STATE_SURFACE_ID = UiStateSurfaceId.PLATE_MANAGER.value
WIDGET_ID = "plate_manager"
PLATE_SCOPE_ID = "plate-1"
PLATE_NAME = "plate 1"
BRIDGE_ID = "bridge-1"
AUTH_TOKEN = "secret"
WINDOW_ID = "main"
PLATE_MANAGER_STATE_PAYLOAD_SCHEMA = "openhcs.ui.plate_manager_state.v1"


class FakeSnapshotRestoreResult:
    @staticmethod
    def restored() -> UiSnapshotRestoreResult:
        return UiSnapshotRestoreResult(
            schema_version=SCHEMA_VERSION,
            restored=True,
            target_snapshot=None,
            current_snapshot=None,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorFile:
    path: Path
    bridge_id: str
    token: str

    def write(self) -> Path:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "bridge_protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
            "bridge_instance_id": self.bridge_id,
            "pid": os.getpid(),
            "started_at_unix": 1.0,
            "connection": {
                "host": "127.0.0.1",
                "port": 7888,
                "transport_mode": "tcp",
                "persistent": True,
            },
            "auth_token": self.token,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.path.chmod(0o600)
        return self.path


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


def _state_surface_document(state: UiPlateManagerState) -> UiStateSurfaceDocument:
    payload = to_jsonable(state)
    if not isinstance(payload, dict):
        raise TypeError("Plate manager test state did not serialize to an object.")
    return UiStateSurfaceDocument(
        schema_version=state.schema_version,
        summary=state.summary,
        payload_schema=PLATE_MANAGER_STATE_PAYLOAD_SCHEMA,
        payload=payload,
        current_revision_token=state.current_revision_token,
        current_snapshot=state.current_snapshot,
        selection_mode=state.selection_mode,
        selected_scope_ids=state.selected_scope_ids,
        unchanged=state.unchanged,
        warnings=state.warnings,
        errors=state.errors,
    )


def _snapshot_ref(
    snapshot_id: str,
    *,
    index: int,
    label: str,
    is_current: bool,
    is_head: bool,
) -> UiSnapshotRef:
    return UiSnapshotRef(
        schema_version=SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        index=index,
        branch="main",
        parent_snapshot_id=None,
        timestamp_unix=1.0 + index,
        timestamp=f"2026-06-25T11:38:5{index}.000",
        label=label,
        num_states=1,
        is_current=is_current,
        is_head=is_head,
        uri=f"openhcs://ui/snapshots/{snapshot_id}",
    )


class _FakeUiBridgeGateway(UiBridgeGatewayABC):
    def __init__(self) -> None:
        self.connections: list[UiBridgeConnectionSpec] = []
        self.restore_requests: list[UiSnapshotRestoreRequest] = []
        self.scope_requests: list[UiObjectStateScopeListRequest] = []
        self.navigate_requests: list[UiWindowNavigateRequest] = []
        self.close_requests: list[UiWindowCloseRequest] = []
        self.snapshot_requests: list[UiWindowSnapshotRequest] = []
        self.widget_tree_requests: list[UiWidgetTreeRequest] = []
        self.selected_plate_workflow_requests: list[UiSelectedPlateWorkflowRequest] = []

    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        self.connections.append(connection)
        return UiBridgeStatus(
            schema_version=SCHEMA_VERSION,
            reachable=True,
            bridge_instance_id=connection.bridge_instance_id,
            connection=connection,
            descriptor_file_path=connection.descriptor_file_path,
        )

    def list_documents(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiCodeDocumentCatalog:
        self.connections.append(connection)
        return UiCodeDocumentCatalog(
            schema_version=SCHEMA_VERSION,
            documents=(
                UiCodeDocumentSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiCodeDocumentIdentity(
                        document_id=DOCUMENT_ID
                    ),
                    title="Plate manager orchestrator config",
                    widget_id=WIDGET_ID,
                    readable=True,
                    writable=True,
                    supported_selection_modes=("selected", "all"),
                    current_selection_count=1,
                    total_scope_count=2,
                ),
            ),
        )

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiStateSurfaceCatalog:
        self.connections.append(connection)
        return UiStateSurfaceCatalog(
            schema_version=SCHEMA_VERSION,
            surfaces=(
                UiStateSurfaceSummary(
                    schema_version=SCHEMA_VERSION,
                    identity=UiStateSurfaceIdentity(surface_id=STATE_SURFACE_ID),
                    title="Plate manager state",
                    widget_id=WIDGET_ID,
                    readable=True,
                    supported_selection_modes=("selected", "all"),
                    current_selection_count=1,
                    total_scope_count=2,
                ),
            ),
        )

    def list_actions(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiActionCatalog:
        self.connections.append(connection)
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=(),
        )

    def invoke_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiActionInvokeRequest,
    ) -> UiActionInvokeResult:
        self.connections.append(connection)
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=request.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=True,
            ),
            target_scope_ids=request.selected_scope_ids,
        )

    def selected_plate_workflow(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSelectedPlateWorkflowRequest,
    ) -> UiSelectedPlateWorkflowResult:
        self.connections.append(connection)
        self.selected_plate_workflow_requests.append(request)
        return UiSelectedPlateWorkflowResult(
            schema_version=SCHEMA_VERSION,
            workflow=request.workflow,
            action_result=UiActionInvokeResult(
                schema_version=SCHEMA_VERSION,
                identity=UiActionIdentity(
                    widget_id=WIDGET_ID,
                    action_id=request.workflow.value,
                ),
                status=UiActionInvocationStatus.ACCEPTED.value,
                receipt=UiMutationReceipt(
                    request_token=request.request_token,
                    accepted=True,
                ),
                target_scope_ids=request.selected_scope_ids,
            ),
        )

    def list_windows(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiWindowCatalog:
        self.connections.append(connection)
        return UiWindowCatalog(
            schema_version=SCHEMA_VERSION,
            windows=(),
        )

    def focus_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowFocusRequest,
    ) -> UiWindowFocusResult:
        self.connections.append(connection)
        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=False,
        )

    def navigate_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        self.connections.append(connection)
        self.navigate_requests.append(request)
        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=True,
            navigated=request.field_path is not None or request.item_id is not None,
            created=False,
        )

    def close_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowCloseRequest,
    ) -> UiWindowCloseResult:
        self.connections.append(connection)
        self.close_requests.append(request)
        return UiWindowCloseResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            closed=True,
        )

    def snapshot_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        self.connections.append(connection)
        self.snapshot_requests.append(request)
        return UiWindowSnapshotResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            captured=True,
            resource=AgentResourceRef(
                uri="file:///tmp/openhcs-window.png",
                title="Main window",
                mime_type="image/png",
                path="/tmp/openhcs-window.png",
                size_bytes=123,
                sha256="abc123",
            ),
            summary=UiWindowSummary(
                schema_version=SCHEMA_VERSION,
                identity=UiWindowIdentity(window_id=request.window_id),
                title="Main window",
                window_kind="embedded",
                visible=True,
                focusable=True,
            ),
            width=320,
            height=200,
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
        )

    def widget_tree(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetTreeRequest,
    ) -> UiWidgetTreeResult:
        self.connections.append(connection)
        self.widget_tree_requests.append(request)
        return UiWidgetTreeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            projected=True,
            root=UiWidgetTreeNode(
                path=(),
                path_id="root",
                child_index=None,
                class_name="QWidget",
                object_name="main",
                visible=True,
                enabled=True,
                geometry=UiWidgetRect(x=0, y=0, width=320, height=200),
                global_geometry=UiWidgetRect(x=10, y=20, width=320, height=200),
                tool_tip="",
                status_tip="",
                whats_this="",
                window_title="Main window",
                accessible_name="",
                accessible_description="",
                text=None,
                text_truncated=False,
                title=None,
                action_kinds=(),
                clickable=False,
                actionable=False,
                checkable=None,
                checked=None,
                current_index=None,
                current_text=None,
                item_count=None,
                children=(
                    UiWidgetTreeNode(
                        path=(0,),
                        path_id="root/0",
                        child_index=0,
                        class_name="QPushButton",
                        object_name="compile_button",
                        visible=True,
                        enabled=True,
                        geometry=UiWidgetRect(x=8, y=160, width=72, height=24),
                        global_geometry=UiWidgetRect(
                            x=18,
                            y=180,
                            width=72,
                            height=24,
                        ),
                        tool_tip="Compile selected plate",
                        status_tip="",
                        whats_this="",
                        window_title="Main window",
                        accessible_name="Compile",
                        accessible_description="Compile selected plate",
                        text="Compile",
                        text_truncated=False,
                        title=None,
                        action_kinds=("click",),
                        clickable=True,
                        actionable=True,
                        checkable=False,
                        checked=False,
                        current_index=None,
                        current_text=None,
                        item_count=None,
                        children=(),
                    ),
                ),
            ),
            summary=UiWindowSummary(
                schema_version=SCHEMA_VERSION,
                identity=UiWindowIdentity(window_id=request.window_id),
                title="Main window",
                window_kind="embedded",
                visible=True,
                focusable=True,
            ),
            widget_count=2,
            actionable_count=1,
        )

    def list_object_state_scopes(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        self.connections.append(connection)
        self.scope_requests.append(request)
        return UiObjectStateScopeCatalog(
            schema_version=SCHEMA_VERSION,
            object_state_token=1,
            current_branch="main",
            current_snapshot_index=-1,
            active=False,
            scopes=(),
        )

    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument:
        self.connections.append(connection)
        summary = UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiCodeDocumentIdentity(document_id=request.document_id),
            title="Plate manager orchestrator config",
            widget_id=WIDGET_ID,
            readable=True,
            writable=True,
        )
        return UiCodeDocument(
            schema_version=SCHEMA_VERSION,
            summary=summary,
            source="plate_paths = []",
            mime_type="text/x-python",
            size_bytes=16,
            sha256="abc",
            current_revision_token="rev-1",
            current_snapshot=None,
            selection_mode=request.resolved_selection_mode("selected"),
            selected_scope_ids=(),
        )

    def get_state_surface(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument:
        self.connections.append(connection)
        summary = UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiStateSurfaceIdentity(surface_id=request.surface_id),
            title="Plate manager state",
            widget_id=WIDGET_ID,
            readable=True,
        )
        return _state_surface_document(
            UiPlateManagerState(
                schema_version=SCHEMA_VERSION,
                summary=summary,
                selection_mode=request.resolved_selection_mode("all"),
                selected_scope_ids=(PLATE_SCOPE_ID,),
                object_state_token=1,
                manager_execution_state="idle",
                rows=(
                    UiPlateManagerRowState(
                        plate_scope_id=PLATE_SCOPE_ID,
                        name=PLATE_NAME,
                        plate_root=f"/tmp/{PLATE_SCOPE_ID}",
                        cppipe_path=None,
                        selected=True,
                        initialized=True,
                        compiled=False,
                        init_pending=False,
                        compile_pending=False,
                        execution_active=False,
                        status_prefix="✓ Init",
                        orchestrator_state="ready",
                        execution_id=None,
                        terminal_status=None,
                        runtime_state=None,
                        runtime_percent=None,
                        queue_position=None,
                    ),
                ),
                current_revision_token="state-rev-1",
                current_snapshot=None,
                unchanged=request.base_revision_token == "state-rev-1",
            )
        )

    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        self.connections.append(connection)
        return UiCodeDocumentValidationResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            valid=True,
            normalized_scope_ids=(PLATE_SCOPE_ID,),
        )

    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        self.connections.append(connection)
        pre_snapshot = _snapshot_ref(
            snapshot_id="snap-1",
            index=1,
            label="before apply",
            is_current=False,
            is_head=False,
        )
        post_snapshot = _snapshot_ref(
            snapshot_id="snap-2",
            index=2,
            label="after apply",
            is_current=True,
            is_head=True,
        )
        return UiCodeDocumentApplyResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            applied=True,
            base_revision_token=request.base_revision_token,
            outcome="applied",
            operation_id="op-1",
            new_revision_token="rev-2",
            current_revision_token="rev-2",
            current_snapshot=post_snapshot,
            undo_snapshot=pre_snapshot,
            pre_apply_snapshot=pre_snapshot,
            post_apply_snapshot=post_snapshot,
        )

    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        self.connections.append(connection)
        return UiSnapshotCatalog(
            schema_version=SCHEMA_VERSION,
            current_branch="main",
            current_snapshot_index=-1,
            object_state_token=1,
            active=False,
            snapshots=(),
            branches=(),
        )

    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        self.connections.append(connection)
        self.restore_requests.append(request)
        return FakeSnapshotRestoreResult.restored()

    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        self.connections.append(connection)
        return FakeSnapshotRestoreResult.restored()

    def list_branches(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBranchCatalog:
        self.connections.append(connection)
        return UiBranchCatalog(
            schema_version=SCHEMA_VERSION,
            current_branch="main",
            branches=(),
        )

    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        self.connections.append(connection)
        return FakeSnapshotRestoreResult.restored()

    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        operation_id: str,
    ) -> UiBridgeOperationRef:
        self.connections.append(connection)
        return UiBridgeOperationRef(
            schema_version=SCHEMA_VERSION,
            identity=UiBridgeOperationIdentity(
                operation_id=operation_id,
                route=UiBridgeOperationRoute(operation_name="apply_document"),
            ),
            status="complete",
            started_at_unix=1.0,
            completed_at_unix=2.0,
            outcome="applied",
        )


def test_default_ui_bridge_service_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.errors[0].code == "ui_bridge_unavailable"


def test_descriptor_resolution_uses_token_without_exposing_it(monkeypatch, tmp_path):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor))
    gateway = _FakeUiBridgeGateway()
    service = UiBridgeService(gateway=gateway)

    status = service.status()
    payload = to_jsonable(status)

    assert status.reachable is True
    assert gateway.connections[0].auth_token == AUTH_TOKEN
    assert status.descriptor_status == "ok"
    assert AUTH_TOKEN not in set(_json_payload_values(payload))


def test_descriptor_resolver_rejects_world_readable_file(monkeypatch, tmp_path):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    descriptor.chmod(0o644)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor))

    status = UiBridgeService(gateway=_FakeUiBridgeGateway()).status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert status.errors[0].code == "stale_ui_bridge_descriptor"


def test_descriptor_resolver_reports_ambiguous_live_descriptors(monkeypatch, tmp_path):
    first_bridge_id = "bridge-one"
    second_bridge_id = "bridge-two"
    UiBridgeDescriptorFile(
        tmp_path / "ui_bridge_one.json",
        first_bridge_id,
        token="one",
    ).write()
    UiBridgeDescriptorFile(
        tmp_path / "ui_bridge_two.json",
        second_bridge_id,
        token="two",
    ).write()
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", raising=False)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    status = UiBridgeService(gateway=_FakeUiBridgeGateway()).status()

    assert status.reachable is False
    assert status.descriptor_status == "ambiguous_ui_bridge"
    assert {descriptor.bridge_instance_id for descriptor in status.descriptors} == {
        first_bridge_id,
        second_bridge_id,
    }


def test_service_forwards_fake_gateway_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    service = UiBridgeService(gateway=gateway)
    connection = service.connection_from_args(
        host="127.0.0.1",
        port=9999,
        auth_token="token",
    )

    catalog = service.list_documents(connection)
    state_catalog = service.list_state_surfaces(connection)
    document = service.get_document(
        UiCodeDocumentRequest(document_id=DOCUMENT_ID),
        connection,
    )
    state = service.get_state_surface(
        UiStateSurfaceRequest(surface_id=STATE_SURFACE_ID),
        connection,
    )
    polled_state = service.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=STATE_SURFACE_ID,
            base_revision_token=state.current_revision_token,
        ),
        connection,
    )
    object_state_scopes = service.list_object_state_scopes(
        UiObjectStateScopeListRequest(
            include_fields=True,
            field_limit=25,
            field_offset=5,
        ),
        connection,
    )
    validation = service.validate_document(
        UiCodeDocumentValidationRequest(
            document_id=DOCUMENT_ID,
            source=document.source,
        ),
        connection,
    )
    apply_result = service.apply_document(
        UiCodeDocumentApplyRequest(
            document_id=DOCUMENT_ID,
            source=document.source,
            base_revision_token=document.current_revision_token,
        ),
        connection,
    )
    close_result = service.close_window(
        UiWindowCloseRequest(window_id=WINDOW_ID),
        connection,
    )
    workflow_result = service.selected_plate_workflow(
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            selected_scope_ids=(PLATE_SCOPE_ID,),
        ),
        connection,
    )
    widget_tree = service.widget_tree(
        UiWidgetTreeRequest(
            window_id=WINDOW_ID,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
            maximum_text_length=128,
        ),
        connection,
    )
    operation = service.get_operation_status("op-1", connection)

    assert catalog.documents[0].current_selection_count == 1
    assert state_catalog.surfaces[0].surface_id == STATE_SURFACE_ID
    assert state.payload["rows"][0]["status_prefix"] == "✓ Init"
    assert polled_state.unchanged is True
    assert object_state_scopes.object_state_token == 1
    assert gateway.scope_requests == [
        UiObjectStateScopeListRequest(
            include_fields=True,
            field_limit=25,
            field_offset=5,
        )
    ]
    assert validation.valid is True
    assert apply_result.applied is True
    assert apply_result.current_revision_token == "rev-2"
    assert apply_result.current_snapshot is not None
    assert apply_result.current_snapshot.snapshot_id == "snap-2"
    assert apply_result.undo_snapshot is not None
    assert apply_result.undo_snapshot.snapshot_id == "snap-1"
    assert apply_result.pre_apply_snapshot == apply_result.undo_snapshot
    assert apply_result.post_apply_snapshot == apply_result.current_snapshot
    assert close_result.closed is True
    assert gateway.close_requests == [
        UiWindowCloseRequest(window_id=WINDOW_ID)
    ]
    assert workflow_result.action_result.status == UiActionInvocationStatus.ACCEPTED.value
    assert gateway.selected_plate_workflow_requests == [
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            selected_scope_ids=(PLATE_SCOPE_ID,),
        )
    ]
    assert widget_tree.projected is True
    assert widget_tree.root is not None
    assert widget_tree.root.path_id == "root"
    assert widget_tree.widget_count == 2
    assert widget_tree.actionable_count == 1
    assert len(widget_tree.root.children) == 1
    compile_button = widget_tree.root.children[0]
    assert compile_button.class_name == "QPushButton"
    assert compile_button.text == "Compile"
    assert compile_button.clickable is True
    assert compile_button.actionable is True
    assert compile_button.action_kinds == ("click",)
    assert compile_button.global_geometry == UiWidgetRect(
        x=18,
        y=180,
        width=72,
        height=24,
    )
    assert gateway.widget_tree_requests == [
        UiWidgetTreeRequest(
            window_id=WINDOW_ID,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
            maximum_text_length=128,
        )
    ]
    assert operation.status == "complete"
    assert all(sent.auth_token == "token" for sent in gateway.connections)


def test_restore_request_rejects_multiple_selectors(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    service = UiBridgeService(gateway=gateway)

    result = service.restore_snapshot(
        UiSnapshotRestoreRequest(snapshot_id="snap-1", index=0)
    )

    assert result.restored is False
    assert result.errors[0].code == "invalid_snapshot_restore_request"
    assert gateway.restore_requests == []


def test_restore_request_preserves_confirmation_and_auto_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    service = UiBridgeService(gateway=gateway)
    connection = service.connection_from_args(port=9999, auth_token="token")

    result = service.restore_snapshot(
        UiSnapshotRestoreRequest(
            snapshot_id="snap-1",
            confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(False),
            allow_auto_branch=True,
        ),
        connection,
    )

    assert result.restored is True
    assert gateway.restore_requests[0].confirmation_is_required() is False
    assert gateway.restore_requests[0].allow_auto_branch is True


def test_snapshot_window_forwards_request_and_resource(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    service = UiBridgeService(gateway=gateway)
    connection = service.connection_from_args(port=9999, auth_token="token")

    result = service.snapshot_window(
        UiWindowSnapshotRequest(
            window_id=WINDOW_ID,
            output_dir_path=str(tmp_path),
            capture_scope=WindowSnapshotCaptureScope.WINDOW,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
        ),
        connection,
    )

    assert result.captured is True
    assert result.resource is not None
    assert result.resource.mime_type == "image/png"
    assert result.width == 320
    assert result.height == 200
    assert result.capture_scope is WindowSnapshotCaptureScope.WINDOW
    assert gateway.snapshot_requests == [
        UiWindowSnapshotRequest(
            window_id=WINDOW_ID,
            output_dir_path=str(tmp_path),
            capture_scope=WindowSnapshotCaptureScope.WINDOW,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
        ),
    ]
    assert gateway.connections[-1].auth_token == "token"
