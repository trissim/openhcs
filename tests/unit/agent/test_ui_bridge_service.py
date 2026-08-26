from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope
from zmqruntime import EndpointApplication, EndpointApplicationCompatibilityError

from openhcs.agent.dto.common import SCHEMA_VERSION, AgentError, AgentResourceRef
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeOperationStatus,
    UiBridgeConfirmationRequirement,
    UiBridgeConnectionSpec,
    UiBridgeOperationIdentity,
    UiBridgeOperationRef,
    UiBridgeOperationRoute,
    UiBridgeOperationStatusRequest,
    UiBridgeOperationWaitRequest,
    UiBridgeRequestEnvelope,
    UiBridgeResponseEnvelope,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentId,
    UiCodeDocumentIdentity,
    UiCodeDocumentRequest,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiMutationReceipt,
    UiObjectStateFieldHelpQuery,
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldListQuery,
    UiObjectStateFieldMutationRequest,
    UiObjectStateFieldMutationResult,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeSummary,
    UiObjectStateValuePreview,
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiSelectedPlateWorkflowKind,
    UiSelectedPlateWorkflowRequest,
    UiSelectedPlateWorkflowResult,
    UiSemanticAddress,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRef,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceId,
    UiStateSurfaceIdentity,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiTimeTravelHeadRequest,
    UiWidgetActionInvokeRequest,
    UiWidgetActionInvokeResult,
    UiWidgetActionSummary,
    UiWidgetRect,
    UiWidgetTreeNode,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
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
)
from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority
from openhcs.agent.services.ui_bridge_service import (
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeDescriptorDirectoryAuthority,
    UiBridgeGatewayABC,
    UiBridgeGatewayResponseError,
    UiBridgeProcessAdvertisedDescriptorCatalog,
    UiBridgeService,
)
from openhcs.agent.services.ui_bridge_transport import UiBridgeControlClient
from openhcs.runtime.viewer_protocol import ViewerLaunchContextMode
from openhcs.runtime.zmq_application import OPENHCS_ENDPOINT_APPLICATION
from openhcs.serialization.json import to_jsonable

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
            "application": to_jsonable(OPENHCS_ENDPOINT_APPLICATION),
            "bridge_instance_id": self.bridge_id,
            "pid": os.getpid(),
            "started_at_unix": time.time(),
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
        self.widget_action_requests: list[UiWidgetActionInvokeRequest] = []
        self.field_mutation_requests: list[UiObjectStateFieldMutationRequest] = []
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
                    identity=UiCodeDocumentIdentity(document_id=DOCUMENT_ID),
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
        compile_action = UiWidgetActionSummary(
            path=(0,),
            path_id="0",
            child_index=0,
            class_name="QPushButton",
            object_name="compile_button",
            accessible_name="Compile",
            accessible_description="Compile selected plate",
            label="Compile",
            visible=True,
            enabled=True,
            geometry=UiWidgetRect(x=8, y=160, width=72, height=24),
            global_geometry=UiWidgetRect(x=18, y=180, width=72, height=24),
            action_kinds=("button",),
            clickable=True,
            checkable=False,
            checked=False,
            current_index=None,
            current_text=None,
            item_count=None,
            tool_tip="Compile selected plate",
        )
        return UiWidgetTreeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            projected=True,
            actionable_widgets=(compile_action,),
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
                        path_id="0",
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
                        action_kinds=("button",),
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
            returned_actionable_count=1,
            include_tree=request.include_tree,
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

    def describe_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldHelpRequest,
    ) -> UiObjectStateFieldHelpResult:
        self.connections.append(connection)
        return UiObjectStateFieldHelpResult(
            schema_version=SCHEMA_VERSION,
            address=request,
            parameter_name=request.field_path.rsplit(".", 1)[-1],
            summary="field help",
            description="field docs",
        )

    def mutate_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldMutationRequest,
    ) -> UiObjectStateFieldMutationResult:
        self.connections.append(connection)
        self.field_mutation_requests.append(request)
        return UiObjectStateFieldMutationResult(
            schema_version=SCHEMA_VERSION,
            address=request,
            mutated=True,
            reset=request.reset,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
        )

    def invoke_widget_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        self.connections.append(connection)
        self.widget_action_requests.append(request)
        return UiWidgetActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            path_id=request.path_id,
            action_kind=request.action_kind,
            invoked=True,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
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
            receipt=UiMutationReceipt.accepted_for(
                request.request_token,
                bridge_operation_id="op-1",
            ),
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
        request: UiBridgeOperationStatusRequest,
    ) -> UiBridgeOperationRef:
        self.connections.append(connection)
        return UiBridgeOperationRef(
            schema_version=SCHEMA_VERSION,
            identity=UiBridgeOperationIdentity(
                operation_id=request.operation_id,
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
    assert "OPENHCS_UI_BRIDGE_DESCRIPTOR" in status.errors[0].hint
    assert str(tmp_path) in status.errors[0].hint


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


def test_descriptor_catalog_uses_exact_environment_selector(monkeypatch, tmp_path):
    selected_directory = tmp_path / "selected"
    selected_directory.mkdir()
    selected_descriptor = UiBridgeDescriptorFile(
        selected_directory / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    discovered_directory = tmp_path / "discovered"
    discovered_directory.mkdir()
    UiBridgeDescriptorFile(
        discovered_directory / "ui_bridge_other.json",
        "other-bridge",
        token="other-token",
    ).write()
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(selected_descriptor))
    monkeypatch.setenv(
        "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR",
        str(discovered_directory),
    )
    monkeypatch.setattr(
        UiBridgeProcessAdvertisedDescriptorCatalog,
        "descriptor_paths",
        classmethod(lambda cls: ()),
    )

    catalog = UiBridgeService().list_bridges()

    assert [bridge.bridge_instance_id for bridge in catalog.bridges] == [BRIDGE_ID]
    assert catalog.bridges[0].descriptor_file_path == str(selected_descriptor.resolve())
    assert catalog.errors == ()


def test_ui_bridge_service_resolves_projected_graphical_viewer_launch_context(
    monkeypatch,
    tmp_path,
):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor))

    monkeypatch.setattr(
        "openhcs.agent.runtime_platform."
        "LinuxAgentRuntimePlatformAuthority._process_environment",
        lambda self, pid: (
            {
                "DISPLAY": ":19",
                "XDG_RUNTIME_DIR": "/run/user/1000",
                "OPENHCS_CPU_ONLY": "true",
                "SECRET_TOKEN": "do-not-forward",
            }
            if pid == os.getpid()
            else None
        ),
    )

    launch_context = UiBridgeService(path_policy=object()).viewer_launch_context()

    assert launch_context.mode is ViewerLaunchContextMode.PROJECTED_GRAPHICAL_SESSION
    assert launch_context.environment_overlay == {
        "DISPLAY": ":19",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "OPENHCS_CPU_ONLY": "true",
    }
    assert "SECRET_TOKEN" not in launch_context.environment_overlay


def test_descriptor_resolver_rejects_world_readable_file(monkeypatch, tmp_path):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    descriptor.chmod(0o644)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor))

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert status.errors[0].code == "stale_ui_bridge_descriptor"


def test_descriptor_reader_uses_declared_dataclass_fields_as_exact_schema(
    monkeypatch,
    tmp_path,
):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["parallel_schema_field"] = "must fail"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert "undeclared field" in status.errors[0].message


def test_descriptor_reader_constructs_transport_enum_from_declared_type(
    monkeypatch,
    tmp_path,
):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["connection"]["transport_mode"] = "invalid"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert "invalid" in status.errors[0].message


def test_descriptor_reader_rejects_mismatched_openhcs_application(
    monkeypatch,
    tmp_path,
):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["application"]["version"] = "0.7.22"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert "Endpoint application mismatch" in status.errors[0].message


def test_ui_bridge_client_rejects_mismatched_server_application() -> None:
    request = UiBridgeRequestEnvelope(
        schema_version=SCHEMA_VERSION,
        bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
        application=OPENHCS_ENDPOINT_APPLICATION,
        request_id="request-1",
        operation="status",
        auth_token=None,
        payload={},
    )
    response = UiBridgeResponseEnvelope(
        schema_version=SCHEMA_VERSION,
        bridge_protocol_version=UI_BRIDGE_PROTOCOL_VERSION,
        application=EndpointApplication(identifier="openhcs", version="0.7.22"),
        request_id=request.request_id,
        ok=True,
        payload={},
    )

    with pytest.raises(EndpointApplicationCompatibilityError):
        UiBridgeControlClient._validate_response(request, response)


def test_descriptor_resolver_rejects_dead_process(monkeypatch, tmp_path):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))
    monkeypatch.setattr(
        AgentRuntimePlatformAuthority,
        "process_started_at_unix",
        staticmethod(lambda _pid: None),
    )

    status = UiBridgeService(gateway=_FakeUiBridgeGateway()).status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert status.errors[0].code == "stale_ui_bridge_descriptor"
    assert "not running" in status.errors[0].message


def test_descriptor_resolver_rejects_reused_process_identity(monkeypatch, tmp_path):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["started_at_unix"] = 10.0
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))
    monkeypatch.setattr(
        AgentRuntimePlatformAuthority,
        "process_started_at_unix",
        staticmethod(lambda pid: 20.0 if pid == os.getpid() else None),
    )

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert "process identity is stale" in status.errors[0].message


def test_status_rechecks_descriptor_process_liveness_after_gateway_call(
    monkeypatch,
    tmp_path,
):
    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    descriptor_started_at = json.loads(descriptor_path.read_text(encoding="utf-8"))[
        "started_at_unix"
    ]
    process_start_times = iter((descriptor_started_at - 1.0, None))
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))
    monkeypatch.setattr(
        AgentRuntimePlatformAuthority,
        "process_started_at_unix",
        staticmethod(lambda _pid: next(process_start_times)),
    )

    status = UiBridgeService(gateway=_FakeUiBridgeGateway()).status()

    assert status.reachable is False
    assert status.descriptor_status == "stale_ui_bridge_descriptor"
    assert status.errors[0].code == "stale_ui_bridge_descriptor"
    assert "not running" in status.errors[0].message


def test_status_rejects_response_from_non_owner_endpoint(monkeypatch, tmp_path):
    class MismatchedIdentityGateway(_FakeUiBridgeGateway):
        def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
            result = super().status(connection)
            return UiBridgeStatus(
                schema_version=result.schema_version,
                reachable=result.reachable,
                bridge_instance_id="different-bridge",
                connection=result.connection,
                descriptor_file_path=result.descriptor_file_path,
            )

    descriptor_path = UiBridgeDescriptorFile(
        tmp_path / "bridge.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    ).write()
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", str(descriptor_path))

    status = UiBridgeService(gateway=MismatchedIdentityGateway()).status()

    assert status.reachable is False
    assert status.descriptor_status == "ui_bridge_endpoint_identity_mismatch"
    assert status.errors[0].code == "ui_bridge_endpoint_identity_mismatch"
    assert "does not own" in status.errors[0].message


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
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    service = UiBridgeService()
    status = service.status()
    launch_context = service.viewer_launch_context()

    assert status.reachable is False
    assert status.descriptor_status == "ambiguous_ui_bridge"
    assert {descriptor.bridge_instance_id for descriptor in status.descriptors} == {
        first_bridge_id,
        second_bridge_id,
    }
    assert launch_context.mode is ViewerLaunchContextMode.HEADLESS


def test_descriptor_resolver_reports_live_descriptors_for_missing_instance(
    monkeypatch,
    tmp_path,
):
    live_bridge_id = "live-bridge"
    UiBridgeDescriptorFile(
        tmp_path / "ui_bridge_live.json",
        live_bridge_id,
        token=AUTH_TOKEN,
    ).write()
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", raising=False)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    status = UiBridgeService(gateway=_FakeUiBridgeGateway()).status(
        UiBridgeConnectionSpec(bridge_instance_id="stale-bridge")
    )
    payload = to_jsonable(status)

    assert status.reachable is False
    assert status.descriptor_status == "ui_bridge_descriptor_not_found"
    assert status.errors[0].code == "ui_bridge_descriptor_not_found"
    assert "bridge_instance_id" in status.errors[0].hint
    assert [descriptor.bridge_instance_id for descriptor in status.descriptors] == [
        live_bridge_id,
    ]
    assert AUTH_TOKEN not in set(_json_payload_values(payload))


def test_descriptor_resolution_uses_process_advertised_descriptor(
    monkeypatch,
    tmp_path,
):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "custom" / "ui_bridge_agent.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    )
    descriptor.path.parent.mkdir()
    descriptor_path = descriptor.write()
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "1234"
    process_dir.mkdir(parents=True)
    process_dir.joinpath("environ").write_bytes(
        b"OTHER=value\0OPENHCS_UI_BRIDGE_DESCRIPTOR="
        + os.fsencode(descriptor_path)
        + b"\0"
    )
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", raising=False)
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", raising=False)
    monkeypatch.setattr(
        UiBridgeProcessAdvertisedDescriptorCatalog,
        "proc_root",
        proc_root,
    )
    monkeypatch.setattr(
        UiBridgeDescriptorDirectoryAuthority,
        "descriptor_dirs",
        classmethod(lambda cls: (tmp_path / "empty-descriptor-dir",)),
    )
    gateway = _FakeUiBridgeGateway()

    status = UiBridgeService(gateway=gateway).status()

    assert status.reachable is True
    assert status.descriptor_status == "ok"
    assert status.descriptor_file_path == str(descriptor_path.resolve())
    assert gateway.connections[0].auth_token == AUTH_TOKEN


def test_configured_descriptor_directory_disables_process_advertised_fallback(
    monkeypatch,
    tmp_path,
):
    descriptor = UiBridgeDescriptorFile(
        tmp_path / "custom" / "ui_bridge_agent.json",
        BRIDGE_ID,
        token=AUTH_TOKEN,
    )
    descriptor.path.parent.mkdir()
    descriptor.write()
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "1234"
    process_dir.mkdir(parents=True)
    process_dir.joinpath("environ").write_bytes(
        b"OPENHCS_UI_BRIDGE_DESCRIPTOR=" + os.fsencode(descriptor.path) + b"\0"
    )
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR", raising=False)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(
        UiBridgeProcessAdvertisedDescriptorCatalog,
        "proc_root",
        proc_root,
    )

    status = UiBridgeService().status()

    assert status.reachable is False
    assert status.errors[0].code == "ui_bridge_unavailable"


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
    object_state_field_help = service.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="global_config",
            field_path="napari_display_config.colormap",
        ),
        connection,
    )
    object_state_field_mutation = service.mutate_object_state_field(
        UiObjectStateFieldMutationRequest(
            object_state_scope_id="global_config",
            field_path="napari_display_config.colormap",
            value="gray",
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
            include_tree=True,
        ),
        connection,
    )
    widget_action = service.invoke_widget_action(
        UiWidgetActionInvokeRequest(
            window_id=WINDOW_ID,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
            path_id="0",
            action_kind="button",
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
    assert object_state_field_help.parameter_name == "colormap"
    assert object_state_field_help.description == "field docs"
    assert object_state_field_mutation.mutated is True
    assert object_state_field_mutation.receipt.accepted is True
    assert gateway.field_mutation_requests == [
        UiObjectStateFieldMutationRequest(
            object_state_scope_id="global_config",
            field_path="napari_display_config.colormap",
            value="gray",
        )
    ]
    assert validation.valid is True
    assert apply_result.applied is True
    assert apply_result.receipt.accepted is True
    assert apply_result.receipt.bridge_operation_id == "op-1"
    assert apply_result.current_revision_token == "rev-2"
    assert apply_result.current_snapshot is not None
    assert apply_result.current_snapshot.snapshot_id == "snap-2"
    assert apply_result.undo_snapshot is not None
    assert apply_result.undo_snapshot.snapshot_id == "snap-1"
    assert apply_result.pre_apply_snapshot == apply_result.undo_snapshot
    assert apply_result.post_apply_snapshot == apply_result.current_snapshot
    assert close_result.closed is True
    assert gateway.close_requests == [UiWindowCloseRequest(window_id=WINDOW_ID)]
    assert (
        workflow_result.action_result.status == UiActionInvocationStatus.ACCEPTED.value
    )
    assert gateway.selected_plate_workflow_requests == [
        UiSelectedPlateWorkflowRequest(
            workflow=UiSelectedPlateWorkflowKind.COMPILE,
            selected_scope_ids=(PLATE_SCOPE_ID,),
        )
    ]
    assert widget_tree.projected is True
    assert widget_tree.actionable_widgets[0].label == "Compile"
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
    assert compile_button.action_kinds == ("button",)
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
            include_tree=True,
        )
    ]
    assert widget_action.invoked is True
    assert widget_action.receipt.accepted is True
    assert gateway.widget_action_requests == [
        UiWidgetActionInvokeRequest(
            window_id=WINDOW_ID,
            open_policy=UiWindowOpenPolicy(create_if_missing=False),
            path_id="0",
            action_kind="button",
        )
    ]
    assert operation.status == "complete"
    assert all(sent.auth_token == "token" for sent in gateway.connections)


def test_list_object_state_scopes_filters_requested_scope_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    class _ScopeFilteringGateway(_FakeUiBridgeGateway):
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
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="global_config",
                        ),
                        object_type="GlobalPipelineConfig",
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                    ),
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id=PLATE_SCOPE_ID,
                        ),
                        object_type="PipelineOrchestrator",
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                    ),
                ),
            )

    gateway = _ScopeFilteringGateway()
    service = UiBridgeService(gateway=gateway)

    result = service.list_object_state_scopes(
        UiObjectStateScopeListRequest(scope_ids=(PLATE_SCOPE_ID,)),
    )

    assert [scope.identity.object_state_scope_id for scope in result.scopes] == [
        PLATE_SCOPE_ID
    ]
    assert gateway.scope_requests[0].scope_ids == (PLATE_SCOPE_ID,)


def test_get_object_state_fields_projects_query_from_scope_catalog(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    def field_summary(
        field_path: str,
        *,
        inherited_value: bool = False,
    ) -> UiObjectStateFieldSummary:
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id="global_config",
                field_path=field_path,
            ),
            field_name=field_path.rsplit(".", 1)[-1],
            container_path=(field_path.rsplit(".", 1)[0] if "." in field_path else ""),
            object_state_path_type="openhcs.core.config.NapariStreamingConfig",
            raw_value_type="None" if inherited_value else "bool",
            resolved_value_type="bool",
            dirty=False,
            signature_diff=False,
            last_changed=False,
            raw_value_preview=UiObjectStateValuePreview(
                type_name="None",
                is_none=True,
                text="None",
            ),
            resolved_value_preview=UiObjectStateValuePreview(
                type_name="bool",
                is_none=False,
                text="False",
            ),
            raw_value_is_none=inherited_value,
            resolved_value_is_none=False,
            inherited_value=inherited_value,
        )

    class _FieldProjectionGateway(_FakeUiBridgeGateway):
        def list_object_state_scopes(
            self,
            connection: UiBridgeConnectionSpec,
            request: UiObjectStateScopeListRequest,
        ) -> UiObjectStateScopeCatalog:
            self.connections.append(connection)
            self.scope_requests.append(request)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=12,
                current_branch="main",
                current_snapshot_index=-1,
                active=False,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="global_config",
                        ),
                        object_type="GlobalPipelineConfig",
                        parameter_count=3,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            field_summary(
                                "napari_streaming_config",
                                inherited_value=True,
                            ),
                            field_summary(
                                "napari_streaming_config.enabled",
                                inherited_value=True,
                            ),
                            field_summary("napari_streaming_config.port"),
                        ),
                    ),
                ),
            )

    gateway = _FieldProjectionGateway()
    service = UiBridgeService(gateway=gateway)

    result = service.get_object_state_fields(
        UiObjectStateFieldListQuery.from_fields(
            scope_ids=("global_config",),
            field_path_contains=("napari_streaming_config",),
            field_filter="semantic",
        )
    )

    assert result.field_filter == "semantic"
    assert result.matched_field_count == 1
    assert result.returned_field_count == 1
    assert result.scopes[0].fields[0].field_path == "napari_streaming_config.enabled"
    assert gateway.scope_requests[0].scope_ids == ("global_config",)
    assert gateway.scope_requests[0].field_paths == ()
    assert (
        gateway.scope_requests[0].field_limit
        == UiObjectStateFieldListQuery.source_query_scan_limit
    )


def test_unsupported_ui_bridge_operation_error_mentions_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))

    class _StaleUiBridgeGateway(_FakeUiBridgeGateway):
        def describe_object_state_field(
            self,
            connection: UiBridgeConnectionSpec,
            request: UiObjectStateFieldHelpRequest,
        ) -> UiObjectStateFieldHelpResult:
            del connection, request
            raise UiBridgeGatewayResponseError(
                errors=(
                    AgentError(
                        code="unsupported_ui_bridge_operation",
                        message=(
                            "Unsupported UI bridge operation: "
                            "describe_object_state_field"
                        ),
                        exception_type="UiBridgeUnsupportedOperationError",
                    ),
                )
            )

    service = UiBridgeService(gateway=_StaleUiBridgeGateway())
    connection = service.connection_from_args(port=9999, auth_token="token")

    result = service.describe_object_state_field(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="global_config",
            field_path="napari_display_config.colormap",
        ),
        connection,
    )

    assert result.errors
    error = result.errors[0]
    assert error.code == "unsupported_ui_bridge_operation"
    assert error.hint is not None
    assert "Restart the UI or UI bridge process" in error.hint
    assert "current OpenHCS source" in error.hint


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


def test_wait_for_operation_receipt_uses_gateway_terminal_wait_owner(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    statuses = iter(("running", "completed"))
    calls: list[UiBridgeOperationStatusRequest] = []

    def get_operation_status(connection, request):
        gateway.connections.append(connection)
        calls.append(request)
        status = next(statuses)
        return UiBridgeOperationRef(
            schema_version=SCHEMA_VERSION,
            identity=UiBridgeOperationIdentity(
                operation_id=request.operation_id,
                route=UiBridgeOperationRoute(operation_name="apply_document"),
            ),
            status=status,
            started_at_unix=1.0,
            completed_at_unix=2.0 if status == "completed" else None,
            outcome="applied" if status == "completed" else None,
        )

    monkeypatch.setattr(gateway, "get_operation_status", get_operation_status)
    monkeypatch.setattr(
        "openhcs.agent.services.ui_bridge_service.time.sleep",
        lambda _seconds: None,
    )
    service = UiBridgeService(gateway=gateway)
    connection = service.connection_from_args(port=9999, auth_token="token")

    result = service.wait_for_operation_receipt(
        UiBridgeOperationWaitRequest(
            operation_id="op-1",
            timeout_seconds=1.0,
            poll_interval_seconds=0.1,
        ),
        connection,
    )

    assert result.status == "completed"
    assert result.outcome == "applied"
    assert result.completed_at_unix == 2.0
    assert calls == [
        UiBridgeOperationStatusRequest(operation_id="op-1"),
        UiBridgeOperationStatusRequest(operation_id="op-1"),
    ]
    assert [item.auth_token for item in gateway.connections] == ["token", "token"]


def test_wait_for_operation_receipt_returns_fresh_running_ref_with_timeout_error(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", str(tmp_path))
    gateway = _FakeUiBridgeGateway()
    monotonic_values = iter((10.0, 10.0))
    running = UiBridgeOperationRef(
        schema_version=SCHEMA_VERSION,
        identity=UiBridgeOperationIdentity(
            operation_id="op-1",
            route=UiBridgeOperationRoute(operation_name="apply_document"),
        ),
        status="running",
        started_at_unix=7.0,
    )
    monkeypatch.setattr(
        gateway,
        "get_operation_status",
        lambda _connection, _request: running,
    )
    monkeypatch.setattr(
        "openhcs.agent.services.ui_bridge_service.time.monotonic",
        lambda: next(monotonic_values),
    )
    service = UiBridgeService(gateway=gateway)
    connection = service.connection_from_args(port=9999, auth_token="token")

    result = service.wait_for_operation_receipt(
        UiBridgeOperationWaitRequest(
            operation_id="op-1",
            timeout_seconds=0.0,
            poll_interval_seconds=0.1,
        ),
        connection,
    )

    assert result.status == "running"
    assert result.started_at_unix == 7.0
    assert result.errors[0].code == "ui_bridge_operation_wait_timeout"
    assert "openhcs_ui_wait_for_operation_receipt" in result.errors[0].hint
    assert "does not establish domain workflow completion" in result.errors[0].hint
    assert result.identity is running.identity


def test_wait_for_operation_receipt_request_rejects_unbounded_controls():
    invalid_controls = (
        {"timeout_seconds": -0.1},
        {"timeout_seconds": 120.1},
        {"poll_interval_seconds": 0.0},
        {"poll_interval_seconds": 5.1},
    )

    for controls in invalid_controls:
        try:
            UiBridgeOperationWaitRequest(operation_id="op-1", **controls)
        except ValueError:
            continue
        raise AssertionError(f"Unbounded wait controls were accepted: {controls}")


def test_ui_bridge_operation_status_owns_completion_selection():
    choices = {
        "active": object(),
        "succeeded": object(),
        "failed": object(),
    }

    assert (
        UiBridgeOperationStatus.RUNNING.select_completion(**choices)
        is choices["active"]
    )
    assert (
        UiBridgeOperationStatus.COMPLETED.select_completion(**choices)
        is choices["succeeded"]
    )
    for status in (
        UiBridgeOperationStatus.FAILED,
        UiBridgeOperationStatus.NOT_FOUND,
        UiBridgeOperationStatus.UNAVAILABLE,
    ):
        assert status.select_completion(**choices) is choices["failed"]

    assert UiBridgeOperationStatus.RUNNING.is_terminal is False
    assert UiBridgeOperationStatus.COMPLETED.is_terminal is True
    assert UiBridgeOperationStatus.RUNNING.live_overview_severity == "info"
    assert UiBridgeOperationStatus.FAILED.live_overview_severity == "error"


def test_ui_requests_own_mcp_tool_argument_projection():
    state_request = UiStateSurfaceRequest.from_fields(
        surface_id=STATE_SURFACE_ID,
        selection_mode="selected",
        base_revision_token="rev-1",
    )
    workflow_request = UiSelectedPlateWorkflowRequest.from_fields(
        workflow=UiSelectedPlateWorkflowKind.COMPILE,
        target_scope_ids=[PLATE_SCOPE_ID],
        observed_selection_revision_token="selection-1",
        request_token="request-1",
        require_confirmation=True,
    )
    wait_request = UiBridgeOperationWaitRequest.from_fields(
        operation_id="operation-1",
        timeout_seconds=12.5,
        poll_interval_seconds=0.25,
    )

    assert state_request.as_tool_arguments() == {
        "surface_id": STATE_SURFACE_ID,
        "selection_mode": "selected",
        "base_revision_token": "rev-1",
    }
    assert workflow_request.as_tool_arguments() == {
        "workflow": "compile_plate",
        "target_scope_ids": [PLATE_SCOPE_ID],
        "observed_selection_revision_token": "selection-1",
        "request_token": "request-1",
        "require_confirmation": True,
    }
    assert wait_request.as_tool_arguments() == {
        "operation_id": "operation-1",
        "timeout_seconds": 12.5,
        "poll_interval_seconds": 0.25,
    }

    request_projections = (
        UiCodeDocumentRequest.from_fields(document_id="pipeline-editor"),
        UiCodeDocumentValidationRequest.from_fields(
            document_id="pipeline-editor",
            source="pipeline = []\n",
        ),
        UiCodeDocumentApplyRequest.from_fields(
            document_id="pipeline-editor",
            source="pipeline = []\n",
            base_revision_token="rev-1",
        ),
        UiActionInvokeRequest.from_fields(
            widget_id="plate_manager",
            action_id="compile_plate",
        ),
        UiWidgetActionInvokeRequest.from_fields(
            window_id="pipeline_editor",
            path_id="function_0.enabled",
            action_kind="toggle",
        ),
        UiWidgetTreeRequest.from_fields(window_id="pipeline_editor"),
        UiWindowSnapshotRequest.from_fields(window_id="pipeline_editor"),
        UiObjectStateScopeListRequest.from_fields(),
        UiObjectStateFieldListQuery.from_fields(),
        UiObjectStateFieldHelpQuery.from_fields(
            field_path="pipeline_config.num_workers"
        ),
        UiObjectStateFieldMutationRequest.from_fields(
            object_state_scope_id=PLATE_SCOPE_ID,
            field_path="pipeline_config.num_workers",
            value=4,
        ),
    )

    for request in (
        state_request,
        workflow_request,
        wait_request,
        *request_projections,
    ):
        assert set(request.as_tool_arguments()) == set(
            inspect.signature(type(request).from_fields).parameters
        )
