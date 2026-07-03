"""Agent service boundary for a running OpenHCS PyQt UI bridge."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from functools import singledispatch
from os import environ
from pathlib import Path
from typing import ClassVar, Generic, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import AgentError, JsonObject, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError
from openhcs.agent.dto.ui_bridge import (
    UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchSwitchRequest,
    UiBridgeConnectionFields,
    UiBridgeConnectionSpec,
    UiBridgeCatalog,
    UiBridgeDescriptorFile,
    UiBridgeDescriptorSummary,
    UiBridgeOperationIdentity,
    UiBridgeOperationRef,
    UiBridgeOperationStatusRequest,
    UiBridgeOperationStatus,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentIdentity,
    UiCodeDocumentRequest,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiMutationReceipt,
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldListQuery,
    UiObjectStateFieldListResult,
    UiObjectStateFieldMutationRequest,
    UiObjectStateFieldMutationResult,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiSelectedPlateWorkflowRequest,
    UiSelectedPlateWorkflowResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceIdentity,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiTimeTravelHeadRequest,
    UI_BRIDGE_UNKNOWN_WIDGET,
    UiWidgetActionInvokeRequest,
    UiWidgetActionInvokeResult,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
)
from openhcs.agent.services.object_state_field_projection import (
    ObjectStateFieldListProjector,
)


UI_BRIDGE_PROTOCOL_VERSION = "openhcs.ui_bridge.v1"
DEFAULT_UI_BRIDGE_TIMEOUT_MS = 5000
DEFAULT_UI_BRIDGE_CONNECTION_SPEC = UiBridgeConnectionSpec(
    timeout_ms=DEFAULT_UI_BRIDGE_TIMEOUT_MS
)
UNAVAILABLE_UI_CODE_DOCUMENT_TITLE = "Unavailable UI code document"
UNAVAILABLE_UI_STATE_SURFACE_TITLE = "Unavailable UI state surface"
UiBridgeResultT = TypeVar("UiBridgeResultT")
UiBridgeRequestT = TypeVar("UiBridgeRequestT")
UiBridgeResponseT = TypeVar("UiBridgeResponseT")


@dataclass(frozen=True, slots=True)
class UiBridgeGatewayMethod(Generic[UiBridgeResponseT]):
    """Nominal reference to a UI bridge gateway method."""

    method: Callable

    @property
    def name(self) -> str:
        return self.method.__name__


@dataclass(frozen=True, slots=True)
class UiBridgeNoPayloadGatewayMethod(UiBridgeGatewayMethod[UiBridgeResponseT]):
    """Gateway method that accepts only a connection payload."""

    method: Callable[["UiBridgeGatewayABC", UiBridgeConnectionSpec], UiBridgeResponseT]
    call: Callable[["UiBridgeGatewayABC", UiBridgeConnectionSpec], UiBridgeResponseT]

    def invoke(
        self,
        gateway: "UiBridgeGatewayABC",
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeResponseT:
        return self.call(gateway, connection)


@dataclass(frozen=True, slots=True)
class UiBridgePayloadGatewayMethod(
    UiBridgeGatewayMethod[UiBridgeResponseT],
    Generic[UiBridgeRequestT, UiBridgeResponseT],
):
    """Gateway method that accepts a typed request payload."""

    method: Callable[
        ["UiBridgeGatewayABC", UiBridgeConnectionSpec, UiBridgeRequestT],
        UiBridgeResponseT,
    ]
    call: Callable[
        ["UiBridgeGatewayABC", UiBridgeConnectionSpec, UiBridgeRequestT],
        UiBridgeResponseT,
    ]

    def invoke(
        self,
        gateway: "UiBridgeGatewayABC",
        connection: UiBridgeConnectionSpec,
        request: UiBridgeRequestT,
    ) -> UiBridgeResponseT:
        return self.call(gateway, connection, request)


class UiBridgeFeature(str, Enum):
    """Status feature tags projected from UI bridge operation declarations."""

    UI_CODE_DOCUMENTS = "ui_code_documents"
    UI_STATE_SURFACES = "ui_state_surfaces"
    UI_ACTIONS = "ui_actions"
    UI_WINDOWS = "ui_windows"
    UI_WINDOW_NAVIGATION = "ui_window_navigation"
    UI_WINDOW_SNAPSHOTS = "ui_window_snapshots"
    SELECTED_PLATE_WORKFLOWS = "selected_plate_workflows"
    WIDGET_TREE_PROJECTION = "widget_tree_projection"
    WIDGET_ACTION_INVOCATION = "widget_action_invocation"
    OBJECTSTATE_SCOPES = "objectstate_scopes"
    OBJECTSTATE_FIELD_MUTATION = "objectstate_field_mutation"
    OBJECTSTATE_SNAPSHOTS = "objectstate_snapshots"
    OBJECTSTATE_BRANCHES = "objectstate_branches"
    OPERATION_STATUS = "operation_status"


def _ui_bridge_operation_registry_key(
    _class_name: str,
    operation_type: type,
) -> str | None:
    gateway_method = operation_type.gateway_method
    if isinstance(gateway_method, UiBridgeGatewayMethod):
        return gateway_method.name
    return None


class UiBridgeOperationContractABC(ABC, metaclass=AutoRegisterMeta):
    """Registered UI bridge operation contract declaration."""

    __registry_key__ = "name"
    __key_extractor__ = _ui_bridge_operation_registry_key
    __skip_if_no_key__ = True

    name: ClassVar[str | None] = None
    gateway_method: ClassVar[UiBridgeGatewayMethod | None] = None
    response_type: ClassVar[type]
    requires_auth: ClassVar[bool] = True
    request_type: ClassVar[type | None] = None
    bridge_features: ClassVar[tuple[UiBridgeFeature, ...]] = ()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None and cls.gateway_method is not None:
            cls.name = cls.gateway_method.name

    @classmethod
    def for_name(cls, operation_name: str) -> "UiBridgeOperationContract":
        try:
            return cls.__registry__[operation_name]
        except KeyError as exc:
            raise KeyError(f"Unknown UI bridge operation: {operation_name}") from exc

    @classmethod
    def require_name(cls) -> str:
        if cls.name is None:
            raise ValueError(f"UI bridge operation {cls.__qualname__} has no name.")
        return cls.name

    @classmethod
    def supported_operation_names(cls) -> tuple[str, ...]:
        return tuple(cls.__registry__)

    @classmethod
    def supported_bridge_features(cls) -> tuple[str, ...]:
        return tuple(
            feature.value
            for feature in dict.fromkeys(
                feature
                for operation_type in cls.__registry__.values()
                for feature in operation_type.bridge_features
            )
        )


class UiBridgeNoPayloadOperationContract(
    UiBridgeOperationContractABC,
    Generic[UiBridgeResponseT],
):
    """Typed UI bridge operation whose gateway method accepts no request payload."""

    request_type: ClassVar[None] = None
    gateway_method: ClassVar[UiBridgeNoPayloadGatewayMethod[UiBridgeResponseT]]

    @classmethod
    def invoke_with_payload(
        cls,
        gateway: "UiBridgeGatewayABC",
        connection: UiBridgeConnectionSpec,
        payload: None,
    ) -> UiBridgeResponseT:
        return cls.gateway_method.invoke(gateway, connection)

    @classmethod
    def decode_request_payload(
        cls,
        payload: JsonObject,
        decoder: Callable[[type[UiBridgeRequestT], JsonObject], UiBridgeRequestT],
    ) -> None:
        del decoder
        if payload:
            raise ValueError(
                f"UI bridge operation {cls.name!r} does not accept a payload."
            )
        return None

    @classmethod
    def validate_request_payload(cls, payload: None) -> None:
        if payload is not None:
            raise TypeError(
                f"UI bridge operation {cls.name!r} does not accept a payload."
            )


class UiBridgePayloadOperationContract(
    UiBridgeOperationContractABC,
    Generic[UiBridgeRequestT, UiBridgeResponseT],
):
    """Typed UI bridge operation whose gateway method accepts a request payload."""

    request_type: ClassVar[type[UiBridgeRequestT]]
    gateway_method: ClassVar[
        UiBridgePayloadGatewayMethod[UiBridgeRequestT, UiBridgeResponseT]
    ]

    @classmethod
    def invoke_with_payload(
        cls,
        gateway: "UiBridgeGatewayABC",
        connection: UiBridgeConnectionSpec,
        payload: UiBridgeRequestT,
    ) -> UiBridgeResponseT:
        return cls.gateway_method.invoke(gateway, connection, payload)

    @classmethod
    def decode_request_payload(
        cls,
        payload: JsonObject,
        decoder: Callable[[type[UiBridgeRequestT], JsonObject], UiBridgeRequestT],
    ) -> UiBridgeRequestT:
        return decoder(cls.request_type, payload)

    @classmethod
    def validate_request_payload(cls, payload: UiBridgeRequestT) -> None:
        if not isinstance(payload, cls.request_type):
            raise TypeError(
                f"UI bridge operation {cls.name!r} requires "
                f"{cls.request_type.__name__} payload."
            )


UiBridgeOperationContract: TypeAlias = (
    type[UiBridgeNoPayloadOperationContract] | type[UiBridgePayloadOperationContract]
)


class UiBridgeDescriptorDirectoryAuthority:
    """Filesystem location policy for live UI bridge descriptors."""

    UI_BRIDGE_DESCRIPTOR_SUBDIR = Path("openhcs") / "ui-bridge"

    @staticmethod
    def default_descriptor_dir() -> Path:
        return UiBridgeDescriptorDirectoryAuthority.descriptor_dirs()[0]

    @classmethod
    def descriptor_dirs(cls) -> tuple[Path, ...]:
        configured = environ.get("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR")
        if configured:
            return (Path(configured).expanduser(),)

        candidates: list[Path] = []
        runtime_dir = environ.get("XDG_RUNTIME_DIR")
        if runtime_dir:
            candidates.append(Path(runtime_dir).expanduser() / cls.UI_BRIDGE_DESCRIPTOR_SUBDIR)
        candidates.append(Path(f"/run/user/{os.getuid()}") / cls.UI_BRIDGE_DESCRIPTOR_SUBDIR)
        candidates.append(Path(tempfile.gettempdir()) / f"openhcs-ui-bridge-{os.getuid()}")
        return tuple(dict.fromkeys(candidates))


class UiBridgeGatewayABC(ABC, metaclass=AutoRegisterMeta):
    """Transport boundary for querying a running OpenHCS UI bridge."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: str | None = None

    @classmethod
    def registered_types(cls) -> tuple[type["UiBridgeGatewayABC"], ...]:
        return tuple(cls.__registry__.values())

    @abstractmethod
    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        raise NotImplementedError

    @abstractmethod
    def list_documents(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiCodeDocumentCatalog:
        raise NotImplementedError

    @abstractmethod
    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiStateSurfaceCatalog:
        raise NotImplementedError

    @abstractmethod
    def list_actions(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiActionCatalog:
        raise NotImplementedError

    @abstractmethod
    def list_windows(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiWindowCatalog:
        raise NotImplementedError

    @abstractmethod
    def list_object_state_scopes(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        raise NotImplementedError

    @abstractmethod
    def describe_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldHelpRequest,
    ) -> UiObjectStateFieldHelpResult:
        raise NotImplementedError

    @abstractmethod
    def mutate_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldMutationRequest,
    ) -> UiObjectStateFieldMutationResult:
        raise NotImplementedError

    @abstractmethod
    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument:
        raise NotImplementedError

    @abstractmethod
    def get_state_surface(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument:
        raise NotImplementedError

    @abstractmethod
    def invoke_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiActionInvokeRequest,
    ) -> UiActionInvokeResult:
        raise NotImplementedError

    @abstractmethod
    def focus_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowFocusRequest,
    ) -> UiWindowFocusResult:
        raise NotImplementedError

    @abstractmethod
    def navigate_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        raise NotImplementedError

    @abstractmethod
    def close_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowCloseRequest,
    ) -> UiWindowCloseResult:
        raise NotImplementedError

    @abstractmethod
    def snapshot_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        raise NotImplementedError

    @abstractmethod
    def widget_tree(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetTreeRequest,
    ) -> UiWidgetTreeResult:
        raise NotImplementedError

    @abstractmethod
    def invoke_widget_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        raise NotImplementedError

    @abstractmethod
    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        raise NotImplementedError

    @abstractmethod
    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        raise NotImplementedError

    @abstractmethod
    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        raise NotImplementedError

    @abstractmethod
    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        raise NotImplementedError

    @abstractmethod
    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        raise NotImplementedError

    @abstractmethod
    def list_branches(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBranchCatalog:
        raise NotImplementedError

    @abstractmethod
    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        raise NotImplementedError

    @abstractmethod
    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBridgeOperationStatusRequest,
    ) -> UiBridgeOperationRef:
        raise NotImplementedError

    @abstractmethod
    def selected_plate_workflow(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSelectedPlateWorkflowRequest,
    ) -> UiSelectedPlateWorkflowResult:
        raise NotImplementedError


class UiBridgeStatusOperation(UiBridgeNoPayloadOperationContract[UiBridgeStatus]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.status,
        lambda gateway, connection: gateway.status(connection),
    )
    response_type = UiBridgeStatus
    requires_auth = False


class UiBridgeListDocumentsOperation(UiBridgeNoPayloadOperationContract[UiCodeDocumentCatalog]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.list_documents,
        lambda gateway, connection: gateway.list_documents(connection),
    )
    response_type = UiCodeDocumentCatalog
    bridge_features = (UiBridgeFeature.UI_CODE_DOCUMENTS,)


class UiBridgeListStateSurfacesOperation(UiBridgeNoPayloadOperationContract[UiStateSurfaceCatalog]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.list_state_surfaces,
        lambda gateway, connection: gateway.list_state_surfaces(connection),
    )
    response_type = UiStateSurfaceCatalog
    bridge_features = (UiBridgeFeature.UI_STATE_SURFACES,)


class UiBridgeListActionsOperation(UiBridgeNoPayloadOperationContract[UiActionCatalog]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.list_actions,
        lambda gateway, connection: gateway.list_actions(connection),
    )
    response_type = UiActionCatalog
    bridge_features = (UiBridgeFeature.UI_ACTIONS,)


class UiBridgeListWindowsOperation(UiBridgeNoPayloadOperationContract[UiWindowCatalog]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.list_windows,
        lambda gateway, connection: gateway.list_windows(connection),
    )
    response_type = UiWindowCatalog
    bridge_features = (UiBridgeFeature.UI_WINDOWS,)


class UiBridgeListObjectStateScopesOperation(
    UiBridgePayloadOperationContract[
        UiObjectStateScopeListRequest,
        UiObjectStateScopeCatalog,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.list_object_state_scopes,
        lambda gateway, connection, request: gateway.list_object_state_scopes(
            connection, request
        ),
    )
    request_type = UiObjectStateScopeListRequest
    response_type = UiObjectStateScopeCatalog
    bridge_features = (UiBridgeFeature.OBJECTSTATE_SCOPES,)


class UiBridgeDescribeObjectStateFieldOperation(
    UiBridgePayloadOperationContract[
        UiObjectStateFieldHelpRequest,
        UiObjectStateFieldHelpResult,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.describe_object_state_field,
        lambda gateway, connection, request: gateway.describe_object_state_field(
            connection, request
        ),
    )
    request_type = UiObjectStateFieldHelpRequest
    response_type = UiObjectStateFieldHelpResult
    bridge_features = (UiBridgeFeature.OBJECTSTATE_SCOPES,)


class UiBridgeMutateObjectStateFieldOperation(
    UiBridgePayloadOperationContract[
        UiObjectStateFieldMutationRequest,
        UiObjectStateFieldMutationResult,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.mutate_object_state_field,
        lambda gateway, connection, request: gateway.mutate_object_state_field(
            connection, request
        ),
    )
    request_type = UiObjectStateFieldMutationRequest
    response_type = UiObjectStateFieldMutationResult
    bridge_features = (UiBridgeFeature.OBJECTSTATE_FIELD_MUTATION,)


class UiBridgeGetDocumentOperation(
    UiBridgePayloadOperationContract[UiCodeDocumentRequest, UiCodeDocument]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.get_document,
        lambda gateway, connection, request: gateway.get_document(
            connection, request
        ),
    )
    request_type = UiCodeDocumentRequest
    response_type = UiCodeDocument
    bridge_features = (UiBridgeFeature.UI_CODE_DOCUMENTS,)


class UiBridgeGetStateSurfaceOperation(
    UiBridgePayloadOperationContract[UiStateSurfaceRequest, UiStateSurfaceDocument]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.get_state_surface,
        lambda gateway, connection, request: gateway.get_state_surface(
            connection, request
        ),
    )
    request_type = UiStateSurfaceRequest
    response_type = UiStateSurfaceDocument
    bridge_features = (UiBridgeFeature.UI_STATE_SURFACES,)


class UiBridgeInvokeActionOperation(
    UiBridgePayloadOperationContract[UiActionInvokeRequest, UiActionInvokeResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.invoke_action,
        lambda gateway, connection, request: gateway.invoke_action(
            connection, request
        ),
    )
    request_type = UiActionInvokeRequest
    response_type = UiActionInvokeResult
    bridge_features = (UiBridgeFeature.UI_ACTIONS,)


class UiBridgeFocusWindowOperation(
    UiBridgePayloadOperationContract[UiWindowFocusRequest, UiWindowFocusResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.focus_window,
        lambda gateway, connection, request: gateway.focus_window(
            connection, request
        ),
    )
    request_type = UiWindowFocusRequest
    response_type = UiWindowFocusResult
    bridge_features = (UiBridgeFeature.UI_WINDOWS,)


class UiBridgeNavigateWindowOperation(
    UiBridgePayloadOperationContract[UiWindowNavigateRequest, UiWindowNavigateResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.navigate_window,
        lambda gateway, connection, request: gateway.navigate_window(
            connection, request
        ),
    )
    request_type = UiWindowNavigateRequest
    response_type = UiWindowNavigateResult
    bridge_features = (UiBridgeFeature.UI_WINDOW_NAVIGATION,)


class UiBridgeCloseWindowOperation(
    UiBridgePayloadOperationContract[UiWindowCloseRequest, UiWindowCloseResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.close_window,
        lambda gateway, connection, request: gateway.close_window(
            connection, request
        ),
    )
    request_type = UiWindowCloseRequest
    response_type = UiWindowCloseResult
    bridge_features = (UiBridgeFeature.UI_WINDOWS,)


class UiBridgeSnapshotWindowOperation(
    UiBridgePayloadOperationContract[UiWindowSnapshotRequest, UiWindowSnapshotResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.snapshot_window,
        lambda gateway, connection, request: gateway.snapshot_window(
            connection, request
        ),
    )
    request_type = UiWindowSnapshotRequest
    response_type = UiWindowSnapshotResult
    bridge_features = (UiBridgeFeature.UI_WINDOW_SNAPSHOTS,)


class UiBridgeWidgetTreeOperation(
    UiBridgePayloadOperationContract[UiWidgetTreeRequest, UiWidgetTreeResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.widget_tree,
        lambda gateway, connection, request: gateway.widget_tree(
            connection, request
        ),
    )
    request_type = UiWidgetTreeRequest
    response_type = UiWidgetTreeResult
    bridge_features = (UiBridgeFeature.WIDGET_TREE_PROJECTION,)


class UiBridgeInvokeWidgetActionOperation(
    UiBridgePayloadOperationContract[
        UiWidgetActionInvokeRequest,
        UiWidgetActionInvokeResult,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.invoke_widget_action,
        lambda gateway, connection, request: gateway.invoke_widget_action(
            connection, request
        ),
    )
    request_type = UiWidgetActionInvokeRequest
    response_type = UiWidgetActionInvokeResult
    bridge_features = (UiBridgeFeature.WIDGET_ACTION_INVOCATION,)


class UiBridgeValidateDocumentOperation(
    UiBridgePayloadOperationContract[
        UiCodeDocumentValidationRequest,
        UiCodeDocumentValidationResult,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.validate_document,
        lambda gateway, connection, request: gateway.validate_document(
            connection, request
        ),
    )
    request_type = UiCodeDocumentValidationRequest
    response_type = UiCodeDocumentValidationResult
    bridge_features = (UiBridgeFeature.UI_CODE_DOCUMENTS,)


class UiBridgeApplyDocumentOperation(
    UiBridgePayloadOperationContract[UiCodeDocumentApplyRequest, UiCodeDocumentApplyResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.apply_document,
        lambda gateway, connection, request: gateway.apply_document(
            connection, request
        ),
    )
    request_type = UiCodeDocumentApplyRequest
    response_type = UiCodeDocumentApplyResult
    bridge_features = (UiBridgeFeature.UI_CODE_DOCUMENTS,)


class UiBridgeListSnapshotsOperation(
    UiBridgePayloadOperationContract[UiSnapshotListRequest, UiSnapshotCatalog]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.list_snapshots,
        lambda gateway, connection, request: gateway.list_snapshots(
            connection, request
        ),
    )
    request_type = UiSnapshotListRequest
    response_type = UiSnapshotCatalog
    bridge_features = (UiBridgeFeature.OBJECTSTATE_SNAPSHOTS,)


class UiBridgeRestoreSnapshotOperation(
    UiBridgePayloadOperationContract[UiSnapshotRestoreRequest, UiSnapshotRestoreResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.restore_snapshot,
        lambda gateway, connection, request: gateway.restore_snapshot(
            connection, request
        ),
    )
    request_type = UiSnapshotRestoreRequest
    response_type = UiSnapshotRestoreResult
    bridge_features = (UiBridgeFeature.OBJECTSTATE_SNAPSHOTS,)


class UiBridgeTimeTravelHeadOperation(
    UiBridgePayloadOperationContract[UiTimeTravelHeadRequest, UiSnapshotRestoreResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.time_travel_head,
        lambda gateway, connection, request: gateway.time_travel_head(
            connection, request
        ),
    )
    request_type = UiTimeTravelHeadRequest
    response_type = UiSnapshotRestoreResult
    bridge_features = (UiBridgeFeature.OBJECTSTATE_SNAPSHOTS,)


class UiBridgeListBranchesOperation(UiBridgeNoPayloadOperationContract[UiBranchCatalog]):
    gateway_method = UiBridgeNoPayloadGatewayMethod(
        UiBridgeGatewayABC.list_branches,
        lambda gateway, connection: gateway.list_branches(connection),
    )
    response_type = UiBranchCatalog
    bridge_features = (UiBridgeFeature.OBJECTSTATE_BRANCHES,)


class UiBridgeSwitchBranchOperation(
    UiBridgePayloadOperationContract[UiBranchSwitchRequest, UiSnapshotRestoreResult]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.switch_branch,
        lambda gateway, connection, request: gateway.switch_branch(
            connection, request
        ),
    )
    request_type = UiBranchSwitchRequest
    response_type = UiSnapshotRestoreResult
    bridge_features = (UiBridgeFeature.OBJECTSTATE_BRANCHES,)


class UiBridgeGetOperationStatusOperation(
    UiBridgePayloadOperationContract[UiBridgeOperationStatusRequest, UiBridgeOperationRef]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.get_operation_status,
        lambda gateway, connection, request: gateway.get_operation_status(
            connection, request
        ),
    )
    request_type = UiBridgeOperationStatusRequest
    response_type = UiBridgeOperationRef
    bridge_features = (UiBridgeFeature.OPERATION_STATUS,)


class UiBridgeSelectedPlateWorkflowOperation(
    UiBridgePayloadOperationContract[
        UiSelectedPlateWorkflowRequest,
        UiSelectedPlateWorkflowResult,
    ]
):
    gateway_method = UiBridgePayloadGatewayMethod(
        UiBridgeGatewayABC.selected_plate_workflow,
        lambda gateway, connection, request: gateway.selected_plate_workflow(
            connection, request
        ),
    )
    request_type = UiSelectedPlateWorkflowRequest
    response_type = UiSelectedPlateWorkflowResult
    bridge_features = (UiBridgeFeature.SELECTED_PLATE_WORKFLOWS,)


class UnavailableUiBridgeGateway(UiBridgeGatewayABC):
    """Gateway used until the PyQt bridge transport is wired."""

    registry_key = UiBridgeOperationStatus.UNAVAILABLE.value

    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        return UiBridgeStatus(
            schema_version=SCHEMA_VERSION,
            reachable=False,
            connection=_public_connection(connection),
            descriptor_file_path=connection.descriptor_file_path,
            errors=(
                AgentError(
                    code="ui_bridge_unavailable",
                    message="No running OpenHCS UI bridge gateway is configured.",
                    hint="Start OpenHCS with the UI bridge enabled.",
                ),
            ),
        )

    def list_documents(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiCodeDocumentCatalog:
        raise UiBridgeGatewayUnavailableError

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiStateSurfaceCatalog:
        raise UiBridgeGatewayUnavailableError

    def list_actions(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiActionCatalog:
        raise UiBridgeGatewayUnavailableError

    def list_windows(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiWindowCatalog:
        raise UiBridgeGatewayUnavailableError

    def list_object_state_scopes(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        raise UiBridgeGatewayUnavailableError

    def describe_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldHelpRequest,
    ) -> UiObjectStateFieldHelpResult:
        raise UiBridgeGatewayUnavailableError

    def mutate_object_state_field(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateFieldMutationRequest,
    ) -> UiObjectStateFieldMutationResult:
        raise UiBridgeGatewayUnavailableError

    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument:
        raise UiBridgeGatewayUnavailableError

    def get_state_surface(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument:
        raise UiBridgeGatewayUnavailableError

    def invoke_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiActionInvokeRequest,
    ) -> UiActionInvokeResult:
        raise UiBridgeGatewayUnavailableError

    def focus_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowFocusRequest,
    ) -> UiWindowFocusResult:
        raise UiBridgeGatewayUnavailableError

    def navigate_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        raise UiBridgeGatewayUnavailableError

    def close_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowCloseRequest,
    ) -> UiWindowCloseResult:
        raise UiBridgeGatewayUnavailableError

    def snapshot_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        raise UiBridgeGatewayUnavailableError

    def widget_tree(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetTreeRequest,
    ) -> UiWidgetTreeResult:
        raise UiBridgeGatewayUnavailableError

    def invoke_widget_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        raise UiBridgeGatewayUnavailableError

    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        raise UiBridgeGatewayUnavailableError

    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        raise UiBridgeGatewayUnavailableError

    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        raise UiBridgeGatewayUnavailableError

    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        raise UiBridgeGatewayUnavailableError

    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        raise UiBridgeGatewayUnavailableError

    def list_branches(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBranchCatalog:
        raise UiBridgeGatewayUnavailableError

    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        raise UiBridgeGatewayUnavailableError

    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBridgeOperationStatusRequest,
    ) -> UiBridgeOperationRef:
        raise UiBridgeGatewayUnavailableError

    def selected_plate_workflow(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSelectedPlateWorkflowRequest,
    ) -> UiSelectedPlateWorkflowResult:
        raise UiBridgeGatewayUnavailableError


class UiBridgeGatewayErrorABC(ABC, metaclass=AutoRegisterMeta):
    """Nominal projection contract for gateway-originated bridge failures."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    @abstractmethod
    def agent_errors(self, fallback_code: str) -> tuple[AgentError, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class UiBridgeGatewayUnavailableError(ConnectionError, UiBridgeGatewayErrorABC):
    registry_key = "unavailable"

    def __str__(self) -> str:
        return "No running OpenHCS UI bridge gateway is configured."

    def agent_errors(self, fallback_code: str) -> tuple[AgentError, ...]:
        del fallback_code
        return (
            AgentError.from_exception(
                "ui_bridge_unavailable",
                self,
                hint=self.discovery_hint(),
            ),
        )

    @staticmethod
    def discovery_hint() -> str:
        searched_dirs = ", ".join(
            str(path)
            for path in UiBridgeDescriptorDirectoryAuthority.descriptor_dirs()
        )
        return (
            "Pass descriptor_file_path, set OPENHCS_UI_BRIDGE_DESCRIPTOR, set "
            "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR, or restart the UI so its bridge "
            f"descriptor is written to one of the searched directories: {searched_dirs}."
        )


@dataclass(frozen=True, slots=True)
class UiBridgeGatewayResponseError(RuntimeError, UiBridgeGatewayErrorABC):
    registry_key = "response"

    errors: tuple[AgentError, ...]

    def __str__(self) -> str:
        if not self.errors:
            return "UI bridge returned an error response."
        return "; ".join(error.message for error in self.errors)

    def agent_errors(self, fallback_code: str) -> tuple[AgentError, ...]:
        del fallback_code
        return tuple(self._with_restart_hint(error) for error in self.errors)

    @staticmethod
    def _with_restart_hint(error: AgentError) -> AgentError:
        if error.code != "unsupported_ui_bridge_operation":
            return error
        if error.hint:
            return error
        return replace(
            error,
            hint=(
                "The running OpenHCS UI bridge does not expose this operation. "
                "Restart the UI or UI bridge process so it imports the current "
                "OpenHCS source, then retry the MCP call."
            ),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeGatewayTimeoutError(TimeoutError, UiBridgeGatewayErrorABC):
    registry_key = "timeout"

    operation: str
    timeout_ms: int

    def __str__(self) -> str:
        return (
            f"UI bridge operation {self.operation!r} timed out after "
            f"{self.timeout_ms}ms."
        )

    def agent_errors(self, fallback_code: str) -> tuple[AgentError, ...]:
        del fallback_code
        return (
            AgentError(
                code="ui_bridge_timeout",
                message=str(self),
                hint=(
                    "The running UI may be blocked or busy; retry after the UI "
                    "event loop is responsive."
                ),
                exception_type=type(self).__name__,
            ),
        )


@singledispatch
def ui_bridge_gateway_errors(
    exception: Exception,
    fallback_code: str,
) -> tuple[AgentError, ...]:
    """Project gateway exceptions into agent-facing errors."""

    return (AgentError.from_exception(fallback_code, exception),)


@ui_bridge_gateway_errors.register
def _unavailable_gateway_errors(
    exception: UiBridgeGatewayUnavailableError,
    fallback_code: str,
) -> tuple[AgentError, ...]:
    return exception.agent_errors(fallback_code)


@ui_bridge_gateway_errors.register
def _response_gateway_errors(
    exception: UiBridgeGatewayResponseError,
    fallback_code: str,
) -> tuple[AgentError, ...]:
    return exception.agent_errors(fallback_code)


@ui_bridge_gateway_errors.register
def _timeout_gateway_errors(
    exception: UiBridgeGatewayTimeoutError,
    fallback_code: str,
) -> tuple[AgentError, ...]:
    return exception.agent_errors(fallback_code)


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorResolution:
    status: str | None = None
    summaries: tuple[UiBridgeDescriptorSummary, ...] = ()

    def project_status(
        self,
        status_result: UiBridgeStatus,
        *,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeStatus:
        descriptor_status = status_result.descriptor_status
        if self.status is not None:
            descriptor_status = self.status
        descriptors = status_result.descriptors
        if self.summaries:
            descriptors = self.summaries
        return replace(
            status_result,
            connection=_public_connection(connection),
            descriptor_status=descriptor_status,
            descriptors=descriptors,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeConnectionResolution(UiBridgeConnectionSpec):
    descriptor: UiBridgeDescriptorResolution = UiBridgeDescriptorResolution()
    errors: tuple[AgentError, ...] = ()

    @classmethod
    def from_connection(
        cls,
        connection: UiBridgeConnectionSpec,
        *,
        descriptor: UiBridgeDescriptorResolution = UiBridgeDescriptorResolution(),
        errors: tuple[AgentError, ...] = (),
    ) -> "UiBridgeConnectionResolution":
        return cls(
            host=connection.host,
            port=connection.port,
            transport_mode=connection.transport_mode,
            persistent=connection.persistent,
            timeout_ms=connection.timeout_ms,
            auth_token=connection.auth_token,
            descriptor_file_path=connection.descriptor_file_path,
            bridge_instance_id=connection.bridge_instance_id,
            descriptor=descriptor,
            errors=errors,
        )

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorReadResult:
    descriptor: UiBridgeDescriptorFile | None
    path: Path
    errors: tuple[AgentError, ...] = ()
    stale_process_descriptor: bool = False

    @property
    def ok(self) -> bool:
        return self.descriptor is not None and not self.errors


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorProcessGoneError(ValueError):
    pid: int

    def __str__(self) -> str:
        return f"UI bridge process is not running: {self.pid}"


class DescriptorSetCardinality(Enum):
    NONE = "none"
    ONE = "one"
    MANY = "many"


@dataclass(frozen=True, slots=True)
class LiveUiBridgeDescriptorSet:
    descriptors: tuple[UiBridgeDescriptorFile, ...]

    @property
    def cardinality(self) -> DescriptorSetCardinality:
        count = len(self.descriptors)
        if count == 0:
            return DescriptorSetCardinality.NONE
        if count == 1:
            return DescriptorSetCardinality.ONE
        return DescriptorSetCardinality.MANY

    def only_descriptor(self) -> UiBridgeDescriptorFile:
        if self.cardinality is not DescriptorSetCardinality.ONE:
            raise ValueError("Live UI bridge descriptor set does not contain exactly one descriptor.")
        return self.descriptors[0]


class UiBridgeDescriptorSummaryBuilder:
    """Build public descriptor summaries from token-bearing descriptor files."""

    @staticmethod
    def summary(
        descriptor: UiBridgeDescriptorFile,
        status: str,
    ) -> UiBridgeDescriptorSummary:
        return UiBridgeDescriptorSummary(
            schema_version=descriptor.schema_version,
            bridge_instance_id=descriptor.bridge_instance_id,
            pid=descriptor.pid,
            started_at_unix=descriptor.started_at_unix,
            descriptor_file_path=descriptor.descriptor_file_path,
            status=status,
            connection=descriptor.connection,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeEnvironment(UiBridgeConnectionFields):

    @classmethod
    def current(cls) -> "UiBridgeEnvironment":
        return cls.from_values(
            host=_env_text("OPENHCS_UI_BRIDGE_HOST"),
            port=_env_int("OPENHCS_UI_BRIDGE_PORT"),
            transport_mode=_env_text("OPENHCS_UI_BRIDGE_TRANSPORT_MODE"),
            timeout_ms=_env_int("OPENHCS_UI_BRIDGE_TIMEOUT_MS"),
            auth_token=_env_text("OPENHCS_UI_BRIDGE_AUTH_TOKEN"),
        )

    def apply(self, connection: UiBridgeConnectionSpec) -> UiBridgeConnectionSpec:
        return UiBridgeConnectionSpec.from_fields(
            self,
            defaults=connection,
        )


class DescriptorSetResolutionRunner(ABC, metaclass=AutoRegisterMeta):
    """Registered resolver behavior for live UI bridge descriptor cardinality."""

    __registry_key__ = "cardinality"
    __skip_if_no_key__ = True

    cardinality: ClassVar[DescriptorSetCardinality | None] = None

    @classmethod
    def for_cardinality(
        cls,
        cardinality: DescriptorSetCardinality,
    ) -> "DescriptorSetResolutionRunner":
        return cls.__registry__[cardinality]()

    @abstractmethod
    def resolve(
        self,
        resolver: "UiBridgeDescriptorResolver",
        descriptor_set: LiveUiBridgeDescriptorSet,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        raise NotImplementedError


class NoDescriptorSetResolutionRunner(DescriptorSetResolutionRunner):
    cardinality = DescriptorSetCardinality.NONE

    def resolve(
        self,
        resolver: "UiBridgeDescriptorResolver",
        descriptor_set: LiveUiBridgeDescriptorSet,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        del resolver, descriptor_set
        return UiBridgeConnectionResolution.from_connection(
            UiBridgeEnvironment.current().apply(connection),
            descriptor=UiBridgeDescriptorResolution(),
        )


class SingleDescriptorSetResolutionRunner(DescriptorSetResolutionRunner):
    cardinality = DescriptorSetCardinality.ONE

    def resolve(
        self,
        resolver: "UiBridgeDescriptorResolver",
        descriptor_set: LiveUiBridgeDescriptorSet,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        return resolver._connection_from_descriptor(
            descriptor_set.only_descriptor(),
            connection,
            "ok",
        )


class AmbiguousDescriptorSetResolutionRunner(DescriptorSetResolutionRunner):
    cardinality = DescriptorSetCardinality.MANY

    def resolve(
        self,
        resolver: "UiBridgeDescriptorResolver",
        descriptor_set: LiveUiBridgeDescriptorSet,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        return UiBridgeConnectionResolution.from_connection(
            connection,
            descriptor=UiBridgeDescriptorResolution(
                status="ambiguous_ui_bridge",
                summaries=tuple(
                    UiBridgeDescriptorSummaryBuilder.summary(descriptor, "live")
                    for descriptor in descriptor_set.descriptors
                ),
            ),
            errors=(
                AgentError(
                    code="ambiguous_ui_bridge",
                    message="Multiple running OpenHCS UI bridge descriptors were found.",
                    hint="Provide descriptor_file_path or bridge_instance_id.",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorPayload:
    payload: JsonObject
    path: Path

    def required_text(self, key: str) -> str:
        return JsonDescriptorValueAuthority.text(self.payload[key])

    def required_int(self, key: str) -> int:
        return JsonDescriptorValueAuthority.integer(self.payload[key])

    def required_float(self, key: str) -> float:
        return JsonDescriptorValueAuthority.floating(self.payload[key])

    def required_bool(self, key: str) -> bool:
        return JsonDescriptorValueAuthority.boolean(self.payload[key])

    def required_object(self, key: str) -> JsonObject:
        return JsonDescriptorValueAuthority.json_object(self.payload[key])

    def optional_text(self, key: str) -> str | None:
        if key not in self.payload:
            return None
        return JsonDescriptorValueAuthority.optional_text(self.payload[key])


class JsonDescriptorValueAuthority:
    """Typed extraction rules for descriptor JSON payload values."""

    @staticmethod
    def text(value: JsonValue) -> str:
        if isinstance(value, str):
            return value
        raise TypeError(f"Expected JSON string, got {type(value).__name__}")

    @staticmethod
    def integer(value: JsonValue) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise TypeError(f"Expected JSON integer, got {type(value).__name__}")

    @staticmethod
    def floating(value: JsonValue) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"Expected JSON number, got {type(value).__name__}")

    @staticmethod
    def boolean(value: JsonValue) -> bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Expected JSON boolean, got {type(value).__name__}")

    @staticmethod
    def json_object(value: JsonValue) -> JsonObject:
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    @classmethod
    def optional_text(cls, value: JsonValue) -> str | None:
        if value is None:
            return None
        return cls.text(value)


class UiBridgeDescriptorReader:
    """Read, parse, and validate one UI bridge descriptor file."""

    @classmethod
    def read(cls, path: Path) -> UiBridgeDescriptorReadResult:
        resolved_path = path.expanduser().resolve(strict=False)
        try:
            cls._validate_descriptor_file_path(resolved_path)
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("UI bridge descriptor must be a JSON object.")
            descriptor = cls._descriptor_from_payload(
                UiBridgeDescriptorPayload(payload, resolved_path)
            )
            cls._validate_descriptor_process(descriptor)
        except Exception as exc:
            return UiBridgeDescriptorReadResult(
                descriptor=None,
                path=resolved_path,
                errors=(AgentError.from_exception("stale_ui_bridge_descriptor", exc),),
                stale_process_descriptor=isinstance(
                    exc,
                    UiBridgeDescriptorProcessGoneError,
                ),
            )
        return UiBridgeDescriptorReadResult(descriptor=descriptor, path=resolved_path)

    @classmethod
    def _descriptor_from_payload(
        cls,
        descriptor_payload: UiBridgeDescriptorPayload,
    ) -> UiBridgeDescriptorFile:
        del cls
        required = (
            "schema_version",
            "bridge_protocol_version",
            "bridge_instance_id",
            "pid",
            "started_at_unix",
            "connection",
            "auth_token",
        )
        missing = tuple(key for key in required if key not in descriptor_payload.payload)
        if missing:
            raise ValueError(f"UI bridge descriptor is missing keys: {', '.join(missing)}")
        protocol_version = descriptor_payload.required_text("bridge_protocol_version")
        if protocol_version != UI_BRIDGE_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported UI bridge protocol version: {protocol_version}")
        connection_payload = UiBridgeDescriptorPayload(
            payload=descriptor_payload.required_object("connection"),
            path=descriptor_payload.path,
        )
        return UiBridgeDescriptorFile(
            schema_version=descriptor_payload.required_text("schema_version"),
            bridge_protocol_version=protocol_version,
            bridge_instance_id=descriptor_payload.required_text("bridge_instance_id"),
            pid=descriptor_payload.required_int("pid"),
            started_at_unix=descriptor_payload.required_float("started_at_unix"),
            connection=ExecutionConnectionSpec(
                host=connection_payload.required_text("host"),
                port=connection_payload.required_int("port"),
                transport_mode=connection_payload.optional_text("transport_mode"),
                persistent=connection_payload.required_bool("persistent"),
            ),
            auth_token=descriptor_payload.required_text("auth_token"),
            descriptor_file_path=str(descriptor_payload.path),
        )

    @staticmethod
    def _validate_descriptor_file_path(path: Path) -> None:
        stat_result = path.stat()
        uid = os.getuid()
        if stat_result.st_uid != uid:
            raise PermissionError("UI bridge descriptor is not owned by the current user.")
        if stat_result.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError("UI bridge descriptor must not be group/world accessible.")
        parent_stat = path.parent.stat()
        parent_mode = parent_stat.st_mode
        parent_is_sticky = bool(parent_mode & stat.S_ISVTX)
        if parent_mode & (stat.S_IWGRP | stat.S_IWOTH) and not parent_is_sticky:
            raise PermissionError(
                "UI bridge descriptor parent directory is writable by other users."
            )

    @staticmethod
    def _validate_descriptor_process(descriptor: UiBridgeDescriptorFile) -> None:
        try:
            os.kill(descriptor.pid, 0)
        except ProcessLookupError as exc:
            raise UiBridgeDescriptorProcessGoneError(descriptor.pid) from exc
        except PermissionError:
            return


class UiBridgeDescriptorDirectoryCatalog:
    """Read live descriptor sets and public descriptor catalogs from the runtime dir."""

    @classmethod
    def live_descriptors(cls) -> tuple[UiBridgeDescriptorFile, ...]:
        descriptors: list[UiBridgeDescriptorFile] = []
        for result in cls._read_descriptor_results():
            if result.ok and result.descriptor is not None:
                descriptors.append(result.descriptor)
        return tuple(descriptors)

    @classmethod
    def descriptor_catalog(cls) -> UiBridgeCatalog:
        descriptors: list[UiBridgeDescriptorSummary] = []
        errors: list[AgentError] = []
        for result in cls._read_descriptor_results():
            if result.descriptor is not None and not result.errors:
                descriptors.append(
                    UiBridgeDescriptorSummaryBuilder.summary(result.descriptor, "live")
                )
                continue
            errors.extend(result.errors)
        return UiBridgeCatalog(
            schema_version=SCHEMA_VERSION,
            bridges=tuple(descriptors),
            errors=tuple(errors),
        )

    @classmethod
    def _read_descriptor_results(cls) -> tuple[UiBridgeDescriptorReadResult, ...]:
        results: list[UiBridgeDescriptorReadResult] = []
        for directory in UiBridgeDescriptorDirectoryAuthority.descriptor_dirs():
            if not directory.exists():
                continue
            for path in sorted(directory.glob("ui_bridge_*.json")):
                result = UiBridgeDescriptorReader.read(path)
                if result.stale_process_descriptor:
                    cls._remove_stale_process_descriptor(result.path)
                    continue
                results.append(result)
        if cls._has_live_descriptor(results):
            return tuple(results)
        if environ.get("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR"):
            return tuple(results)
        cls._extend_with_process_advertised_descriptors(results)
        return tuple(results)

    @staticmethod
    def _has_live_descriptor(results: list[UiBridgeDescriptorReadResult]) -> bool:
        return any(result.ok and result.descriptor is not None for result in results)

    @classmethod
    def _extend_with_process_advertised_descriptors(
        cls,
        results: list[UiBridgeDescriptorReadResult],
    ) -> None:
        seen_paths = {result.path for result in results}
        for path in UiBridgeProcessAdvertisedDescriptorCatalog.descriptor_paths():
            resolved_path = path.expanduser().resolve(strict=False)
            if resolved_path in seen_paths:
                continue
            results.append(UiBridgeDescriptorReader.read(resolved_path))
            seen_paths.add(resolved_path)

    @staticmethod
    def _remove_stale_process_descriptor(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return


class UiBridgeProcessAdvertisedDescriptorCatalog:
    """Find descriptor files explicitly advertised by running local UI processes."""

    proc_root = Path("/proc")
    descriptor_environment_name = "OPENHCS_UI_BRIDGE_DESCRIPTOR"

    @classmethod
    def descriptor_paths(cls) -> tuple[Path, ...]:
        paths: list[Path] = []
        for process_dir in cls._process_dirs():
            descriptor_path = cls._descriptor_path_from_process(process_dir)
            if descriptor_path is not None:
                paths.append(descriptor_path)
        return tuple(dict.fromkeys(paths))

    @classmethod
    def _process_dirs(cls) -> tuple[Path, ...]:
        try:
            return tuple(
                path
                for path in cls.proc_root.iterdir()
                if path.name.isdigit()
            )
        except (FileNotFoundError, PermissionError, OSError):
            return ()

    @classmethod
    def _descriptor_path_from_process(cls, process_dir: Path) -> Path | None:
        try:
            environment_payload = (process_dir / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            return None
        return cls._descriptor_path_from_environment(environment_payload)

    @classmethod
    def _descriptor_path_from_environment(cls, payload: bytes) -> Path | None:
        prefix = f"{cls.descriptor_environment_name}=".encode()
        for entry in payload.split(b"\0"):
            if not entry.startswith(prefix):
                continue
            value = entry.removeprefix(prefix)
            if not value:
                return None
            return Path(os.fsdecode(value))
        return None


class UiBridgeDescriptorResolver:
    """Resolve UI bridge descriptors without widening the general path policy."""

    def resolve(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        if connection.descriptor_file_path is not None:
            return self._resolve_explicit_file(Path(connection.descriptor_file_path), connection)

        env_descriptor = environ.get("OPENHCS_UI_BRIDGE_DESCRIPTOR")
        if env_descriptor:
            return self._resolve_explicit_file(Path(env_descriptor), connection)

        if connection.bridge_instance_id is not None:
            return self._resolve_instance(connection.bridge_instance_id, connection)

        descriptor_set = LiveUiBridgeDescriptorSet(
            UiBridgeDescriptorDirectoryCatalog.live_descriptors()
        )
        return DescriptorSetResolutionRunner.for_cardinality(
            descriptor_set.cardinality
        ).resolve(self, descriptor_set, connection)

    def _resolve_explicit_file(
        self,
        path: Path,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        result = UiBridgeDescriptorReader.read(path)
        if not result.ok or result.descriptor is None:
            return UiBridgeConnectionResolution.from_connection(
                replace(connection, descriptor_file_path=str(result.path)),
                descriptor=UiBridgeDescriptorResolution(
                    status="stale_ui_bridge_descriptor",
                ),
                errors=result.errors,
            )
        return self._connection_from_descriptor(result.descriptor, connection, "ok")

    def _resolve_instance(
        self,
        bridge_instance_id: str,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        live_descriptors = UiBridgeDescriptorDirectoryCatalog.live_descriptors()
        matches = tuple(
            descriptor
            for descriptor in live_descriptors
            if descriptor.bridge_instance_id == bridge_instance_id
        )
        if not matches:
            return UiBridgeConnectionResolution.from_connection(
                connection,
                descriptor=UiBridgeDescriptorResolution(
                    status="ui_bridge_descriptor_not_found",
                    summaries=tuple(
                        UiBridgeDescriptorSummaryBuilder.summary(descriptor, "live")
                        for descriptor in live_descriptors
                    ),
                ),
                errors=(
                    AgentError(
                        code="ui_bridge_descriptor_not_found",
                        message=f"No live OpenHCS UI bridge descriptor matches {bridge_instance_id!r}.",
                        hint=(
                            "Use one of the returned descriptors' bridge_instance_id "
                            "values, pass descriptor_file_path, or omit both when "
                            "exactly one live bridge is available."
                        ),
                    ),
                ),
            )
        return self._connection_from_descriptor(matches[0], connection, "ok")

    def _connection_from_descriptor(
        self,
        descriptor: UiBridgeDescriptorFile,
        connection: UiBridgeConnectionSpec,
        status: str,
    ) -> UiBridgeConnectionResolution:
        descriptor_connection = UiBridgeConnectionSpec.from_fields(
            UiBridgeConnectionFields.from_descriptor(descriptor),
            defaults=connection,
        )
        return UiBridgeConnectionResolution.from_connection(
            descriptor_connection,
            descriptor=UiBridgeDescriptorResolution(
                status=status,
                summaries=(UiBridgeDescriptorSummaryBuilder.summary(descriptor, status),),
            ),
        )


DEFAULT_UI_BRIDGE_DESCRIPTOR_RESOLVER = UiBridgeDescriptorResolver()


class UiBridgeService:
    """Expose running-UI code documents and ObjectState snapshots to agents."""

    def __init__(
        self,
        gateway: UiBridgeGatewayABC | None = None,
        descriptor_resolver: UiBridgeDescriptorResolver = DEFAULT_UI_BRIDGE_DESCRIPTOR_RESOLVER,
        path_policy: AgentPathPolicy | None = None,
    ) -> None:
        if gateway is None:
            from openhcs.agent.services.ui_bridge_transport import ZMQUiBridgeGateway

            gateway = ZMQUiBridgeGateway()
        self._gateway = gateway
        self._descriptor_resolver = descriptor_resolver
        self._path_policy = path_policy or AgentPathPolicy.from_environment()

    def connection_from_args(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        transport_mode: str | None = None,
        timeout_ms: int | None = None,
        auth_token: str | None = None,
        descriptor_file_path: str | None = None,
        bridge_instance_id: str | None = None,
        persistent: bool = True,
    ) -> UiBridgeConnectionSpec:
        return self.connection_from_fields(
            UiBridgeConnectionFields.from_values(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
                timeout_ms=timeout_ms,
                auth_token=auth_token,
                descriptor_file_path=descriptor_file_path,
                bridge_instance_id=bridge_instance_id,
            )
        )

    def connection_from_fields(
        self,
        fields: UiBridgeConnectionFields,
    ) -> UiBridgeConnectionSpec:
        return UiBridgeConnectionSpec.from_fields(
            fields,
            defaults=UiBridgeConnectionSpec(timeout_ms=DEFAULT_UI_BRIDGE_TIMEOUT_MS),
        )

    def _dispatch_gateway(
        self,
        *,
        connection: UiBridgeConnectionSpec,
        call: Callable[[UiBridgeConnectionResolution], UiBridgeResultT],
        error_result: Callable[[tuple[AgentError, ...]], UiBridgeResultT],
        unavailable_error_code: str = "ui_bridge_unavailable",
    ) -> UiBridgeResultT:
        resolution = self._descriptor_resolver.resolve(connection)
        if not resolution.ok:
            return error_result(resolution.errors)
        try:
            return call(resolution)
        except Exception as exc:
            return error_result(self._gateway_errors(unavailable_error_code, exc))

    def list_bridges(self) -> UiBridgeCatalog:
        return UiBridgeDescriptorDirectoryCatalog.descriptor_catalog()

    def status(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBridgeStatus:
        resolution = self._descriptor_resolver.resolve(connection)
        if not resolution.ok:
            return self._status_from_resolution(resolution)
        try:
            status_result = self._gateway.status(resolution)
        except Exception as exc:
            return self._status_from_resolution(
                UiBridgeConnectionResolution.from_connection(
                    resolution,
                    descriptor=resolution.descriptor,
                    errors=self._gateway_errors("ui_bridge_unreachable", exc),
                )
            )
        return resolution.descriptor.project_status(status_result, connection=resolution)

    def list_documents(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=self._gateway.list_documents,
            error_result=lambda errors: UiCodeDocumentCatalog(
                SCHEMA_VERSION,
                documents=(),
                errors=errors,
            ),
        )

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiStateSurfaceCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=self._gateway.list_state_surfaces,
            error_result=lambda errors: UiStateSurfaceCatalog(
                SCHEMA_VERSION,
                surfaces=(),
                errors=errors,
            ),
        )

    def list_actions(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiActionCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=self._gateway.list_actions,
            error_result=lambda errors: UiActionCatalog(
                SCHEMA_VERSION,
                actions=(),
                errors=errors,
            ),
        )

    def list_windows(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=self._gateway.list_windows,
            error_result=lambda errors: UiWindowCatalog(
                schema_version=SCHEMA_VERSION,
                windows=(),
                errors=errors,
            ),
        )

    def list_object_state_scopes(
        self,
        request: UiObjectStateScopeListRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiObjectStateScopeCatalog:
        catalog = self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.list_object_state_scopes(
                resolution,
                request,
            ),
            error_result=self._object_state_scope_catalog_error,
        )
        return request.filtered_catalog(catalog)

    def get_object_state_fields(
        self,
        query: UiObjectStateFieldListQuery,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiObjectStateFieldListResult:
        catalog = self.list_object_state_scopes(
            query.scope_list_request(),
            connection,
        )
        return ObjectStateFieldListProjector.project_catalog(query, catalog)

    def describe_object_state_field(
        self,
        request: UiObjectStateFieldHelpRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiObjectStateFieldHelpResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.describe_object_state_field(
                resolution,
                request,
            ),
            error_result=lambda errors: UiObjectStateFieldHelpResult(
                schema_version=SCHEMA_VERSION,
                address=request,
                errors=errors,
            ),
        )

    def mutate_object_state_field(
        self,
        request: UiObjectStateFieldMutationRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiObjectStateFieldMutationResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.mutate_object_state_field(
                resolution,
                request,
            ),
            error_result=lambda errors: UiObjectStateFieldMutationResult(
                schema_version=SCHEMA_VERSION,
                address=request,
                mutated=False,
                reset=request.reset,
                receipt=UiMutationReceipt.rejected_for(request.request_token),
                errors=errors,
            ),
        )

    def get_document(
        self,
        request: UiCodeDocumentRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocument:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.get_document(resolution, request),
            error_result=lambda errors: self._document_error(
                request,
                errors,
            ),
        )

    def get_state_surface(
        self,
        request: UiStateSurfaceRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiStateSurfaceDocument:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.get_state_surface(
                resolution,
                request,
            ),
            error_result=lambda errors: self._state_surface_error(
                request,
                errors,
            ),
        )

    def invoke_action(
        self,
        request: UiActionInvokeRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiActionInvokeResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.invoke_action(resolution, request),
            error_result=lambda errors: self._action_error(
                request,
                errors,
            ),
        )

    def selected_plate_workflow(
        self,
        request: UiSelectedPlateWorkflowRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSelectedPlateWorkflowResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.selected_plate_workflow(
                resolution,
                request,
            ),
            error_result=lambda errors: self._selected_plate_workflow_error(
                request,
                errors,
            ),
        )

    def focus_window(
        self,
        request: UiWindowFocusRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowFocusResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.focus_window(resolution, request),
            error_result=lambda errors: self._window_focus_error(
                request,
                errors,
            ),
        )

    def navigate_window(
        self,
        request: UiWindowNavigateRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowNavigateResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.navigate_window(resolution, request),
            error_result=lambda errors: self._window_navigate_error(
                request,
                errors,
            ),
        )

    def close_window(
        self,
        request: UiWindowCloseRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowCloseResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.close_window(resolution, request),
            error_result=lambda errors: self._window_close_error(
                request,
                errors,
            ),
        )

    def snapshot_window(
        self,
        request: UiWindowSnapshotRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowSnapshotResult:
        try:
            request = self._writable_snapshot_request(request)
        except AgentPathPolicyError as exc:
            return self._window_snapshot_error(request, (exc.to_agent_error(),))
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.snapshot_window(
                resolution,
                request,
            ),
            error_result=lambda errors: self._window_snapshot_error(
                request,
                errors,
            ),
        )

    def _writable_snapshot_request(
        self,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotRequest:
        return replace(
            request,
            output_dir_path=str(
                self._path_policy.assert_writable(request.output_dir_path)
            ),
        )

    def widget_tree(
        self,
        request: UiWidgetTreeRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWidgetTreeResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.widget_tree(resolution, request),
            error_result=lambda errors: self._widget_tree_error(
                request,
                errors,
            ),
        )

    def invoke_widget_action(
        self,
        request: UiWidgetActionInvokeRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWidgetActionInvokeResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.invoke_widget_action(
                resolution,
                request,
            ),
            error_result=lambda errors: self._widget_action_error(
                request,
                errors,
            ),
        )

    def validate_document(
        self,
        request: UiCodeDocumentValidationRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentValidationResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.validate_document(
                resolution,
                request,
            ),
            error_result=lambda errors: UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=errors,
            ),
        )

    def apply_document(
        self,
        request: UiCodeDocumentApplyRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentApplyResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.apply_document(
                resolution,
                request,
            ),
            error_result=lambda errors: UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=False,
                base_revision_token=request.base_revision_token,
                receipt=UiMutationReceipt.rejected_for(request.request_token),
                errors=errors,
            ),
        )

    def list_snapshots(
        self,
        request: UiSnapshotListRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.list_snapshots(
                resolution,
                request,
            ),
            error_result=self._snapshot_catalog_error,
        )

    def restore_snapshot(
        self,
        request: UiSnapshotRestoreRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotRestoreResult:
        selector_count = sum(
            selector is not None
            for selector in (request.snapshot_id, request.index, request.branch)
        )
        if selector_count != 1:
            return self._restore_error(
                (
                    AgentError(
                        code="invalid_snapshot_restore_request",
                        message="Exactly one snapshot restore selector is required.",
                    ),
                )
            )
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.restore_snapshot(
                resolution,
                request,
            ),
            error_result=self._restore_error,
        )

    def time_travel_head(
        self,
        request: UiTimeTravelHeadRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotRestoreResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.time_travel_head(
                resolution,
                request,
            ),
            error_result=self._restore_error,
        )

    def list_branches(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBranchCatalog:
        return self._dispatch_gateway(
            connection=connection,
            call=self._gateway.list_branches,
            error_result=lambda errors: UiBranchCatalog(
                SCHEMA_VERSION,
                current_branch="",
                branches=(),
                errors=errors,
            ),
        )

    def switch_branch(
        self,
        request: UiBranchSwitchRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotRestoreResult:
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.switch_branch(resolution, request),
            error_result=self._restore_error,
        )

    def get_operation_status(
        self,
        operation_id: str,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBridgeOperationRef:
        request = UiBridgeOperationStatusRequest(operation_id=operation_id)
        return self._dispatch_gateway(
            connection=connection,
            call=lambda resolution: self._gateway.get_operation_status(
                resolution,
                request,
            ),
            error_result=lambda errors: UiBridgeOperationRef(
                schema_version=SCHEMA_VERSION,
                identity=UiBridgeOperationIdentity(
                    operation_id=operation_id,
                    route=UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
                ),
                status=UiBridgeOperationStatus.UNAVAILABLE.value,
                started_at_unix=0.0,
                errors=errors,
            ),
        )

    @staticmethod
    def _status_from_resolution(
        resolution: UiBridgeConnectionResolution,
    ) -> UiBridgeStatus:
        return UiBridgeStatus(
            schema_version=SCHEMA_VERSION,
            reachable=False,
            connection=_public_connection(resolution),
            descriptor_file_path=resolution.descriptor_file_path,
            descriptor_status=resolution.descriptor.status,
            descriptors=resolution.descriptor.summaries,
            errors=resolution.errors,
        )

    @staticmethod
    def _document_error(
        request: UiCodeDocumentRequest,
        errors: tuple[AgentError, ...],
    ) -> UiCodeDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.SELECTED
        )
        summary = UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiCodeDocumentIdentity(document_id=request.document_id),
            title=UNAVAILABLE_UI_CODE_DOCUMENT_TITLE,
            widget_id=UI_BRIDGE_UNKNOWN_WIDGET,
            readable=False,
            writable=False,
        )
        return UiCodeDocument(
            schema_version=SCHEMA_VERSION,
            summary=summary,
            source="",
            mime_type="text/x-python",
            size_bytes=0,
            sha256="",
            current_revision_token=None,
            current_snapshot=None,
            selection_mode=selection_mode,
            selected_scope_ids=(),
            errors=errors,
        )

    @staticmethod
    def _state_surface_error(
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        summary = UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiStateSurfaceIdentity(surface_id=request.surface_id),
            title=UNAVAILABLE_UI_STATE_SURFACE_TITLE,
            widget_id=UI_BRIDGE_UNKNOWN_WIDGET,
            readable=False,
        )
        return UiStateSurfaceDocument(
            schema_version=SCHEMA_VERSION,
            summary=summary,
            payload_schema="openhcs.ui.unavailable_state_surface.v1",
            payload={},
            selection_mode=selection_mode,
            selected_scope_ids=(),
            current_revision_token=None,
            current_snapshot=None,
            errors=errors,
        )

    @staticmethod
    def _action_error(
        request: UiActionInvokeRequest,
        errors: tuple[AgentError, ...],
    ) -> UiActionInvokeResult:
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=request.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.UNAVAILABLE.value,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            errors=errors,
        )

    @staticmethod
    def _selected_plate_workflow_error(
        request: UiSelectedPlateWorkflowRequest,
        errors: tuple[AgentError, ...],
    ) -> UiSelectedPlateWorkflowResult:
        return UiSelectedPlateWorkflowResult(
            schema_version=SCHEMA_VERSION,
            workflow=request.workflow,
            action_result=UiActionInvokeResult(
                schema_version=SCHEMA_VERSION,
                identity=UiActionIdentity(
                    widget_id=UI_BRIDGE_UNKNOWN_WIDGET,
                    action_id=request.workflow.value,
                ),
                status=UiActionInvocationStatus.UNAVAILABLE.value,
                receipt=UiMutationReceipt.rejected_for(request.request_token),
                errors=errors,
            ),
            errors=errors,
        )

    @staticmethod
    def _window_focus_error(
        request: UiWindowFocusRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWindowFocusResult:
        return UiWindowFocusResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=False,
            errors=errors,
        )

    @staticmethod
    def _window_navigate_error(
        request: UiWindowNavigateRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWindowNavigateResult:
        return UiWindowNavigateResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            focused=False,
            navigated=False,
            created=False,
            errors=errors,
        )

    @staticmethod
    def _window_close_error(
        request: UiWindowCloseRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWindowCloseResult:
        return UiWindowCloseResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            closed=False,
            errors=errors,
        )

    @staticmethod
    def _window_snapshot_error(
        request: UiWindowSnapshotRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWindowSnapshotResult:
        return UiWindowSnapshotResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            output_dir_path=request.output_dir_path,
            capture_scope=request.capture_scope,
            captured=False,
            errors=errors,
        )

    @staticmethod
    def _widget_tree_error(
        request: UiWidgetTreeRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWidgetTreeResult:
        return UiWidgetTreeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            projected=False,
            errors=errors,
        )

    @staticmethod
    def _widget_action_error(
        request: UiWidgetActionInvokeRequest,
        errors: tuple[AgentError, ...],
    ) -> UiWidgetActionInvokeResult:
        return UiWidgetActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            window_id=request.window_id,
            path_id=request.path_id,
            action_kind=request.action_kind,
            invoked=False,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            errors=errors,
        )

    @staticmethod
    def _object_state_scope_catalog_error(
        errors: tuple[AgentError, ...],
    ) -> UiObjectStateScopeCatalog:
        return UiObjectStateScopeCatalog(
            schema_version=SCHEMA_VERSION,
            object_state_token=0,
            current_branch="",
            current_snapshot_index=-1,
            active=False,
            scopes=(),
            errors=errors,
        )

    @staticmethod
    def _snapshot_catalog_error(errors: tuple[AgentError, ...]) -> UiSnapshotCatalog:
        return UiSnapshotCatalog(
            schema_version=SCHEMA_VERSION,
            current_branch="",
            current_snapshot_index=-1,
            object_state_token=0,
            active=False,
            snapshots=(),
            branches=(),
            errors=errors,
        )

    @staticmethod
    def _restore_error(errors: tuple[AgentError, ...]) -> UiSnapshotRestoreResult:
        return UiSnapshotRestoreResult(
            schema_version=SCHEMA_VERSION,
            restored=False,
            target_snapshot=None,
            current_snapshot=None,
            errors=errors,
        )

    @staticmethod
    def _gateway_errors(code: str, exception: Exception) -> tuple[AgentError, ...]:
        return ui_bridge_gateway_errors(exception, code)


def _env_text(name: str) -> str | None:
    if name not in environ:
        return None
    value = environ[name]
    if value == "":
        return None
    return value


def _env_int(name: str) -> int | None:
    value = _env_text(name)
    if value is None:
        return None
    return int(value)


def _public_connection(connection: ExecutionConnectionSpec) -> ExecutionConnectionSpec:
    return ExecutionConnectionSpec(
        host=connection.host,
        port=connection.port,
        transport_mode=connection.transport_mode,
        persistent=connection.persistent,
    )
