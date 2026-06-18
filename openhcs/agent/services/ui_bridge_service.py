"""Agent service boundary for a running OpenHCS PyQt UI bridge."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from os import environ
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import AgentError, JsonObject, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.execution import ExecutionConnectionSpec
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
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
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
    UiTimeTravelRuntimeState,
    UI_BRIDGE_UNKNOWN_WIDGET,
    UiWindowCatalog,
    UiWindowFocusRequest,
    UiWindowFocusResult,
)


UI_BRIDGE_PROTOCOL_VERSION = "openhcs.ui_bridge.v1"
DEFAULT_UI_BRIDGE_TIMEOUT_MS = 5000
DEFAULT_UI_BRIDGE_CONNECTION_SPEC = UiBridgeConnectionSpec(
    timeout_ms=DEFAULT_UI_BRIDGE_TIMEOUT_MS
)
UNAVAILABLE_UI_CODE_DOCUMENT_TITLE = "Unavailable UI code document"
UNAVAILABLE_UI_STATE_SURFACE_TITLE = "Unavailable UI state surface"


class UiBridgeDescriptorDirectoryAuthority:
    """Filesystem location policy for live UI bridge descriptors."""

    @staticmethod
    def default_descriptor_dir() -> Path:
        configured = environ.get("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR")
        if configured:
            return Path(configured).expanduser()
        runtime_dir = environ.get("XDG_RUNTIME_DIR")
        if runtime_dir:
            return Path(runtime_dir).expanduser() / "openhcs" / "ui-bridge"
        return Path(tempfile.gettempdir()) / f"openhcs-ui-bridge-{os.getuid()}"


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
        operation_id: str,
    ) -> UiBridgeOperationRef:
        raise NotImplementedError


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
        operation_id: str,
    ) -> UiBridgeOperationRef:
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
        return (AgentError.from_exception("ui_bridge_unavailable", self),)


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
        return self.errors


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
class UiBridgeConnectionResolution:
    connection: UiBridgeConnectionSpec
    descriptor: UiBridgeDescriptorResolution = UiBridgeDescriptorResolution()
    errors: tuple[AgentError, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def project_status(self, status_result: UiBridgeStatus) -> UiBridgeStatus:
        return self.descriptor.project_status(
            status_result,
            connection=self.connection,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeDescriptorReadResult:
    descriptor: UiBridgeDescriptorFile | None
    path: Path
    errors: tuple[AgentError, ...] = ()

    @property
    def ok(self) -> bool:
        return self.descriptor is not None and not self.errors


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
class UiBridgeEnvironment:
    fields: UiBridgeConnectionFields

    @classmethod
    def current(cls) -> "UiBridgeEnvironment":
        return cls(
            fields=UiBridgeConnectionFields.from_values(
                host=_env_text("OPENHCS_UI_BRIDGE_HOST"),
                port=_env_int("OPENHCS_UI_BRIDGE_PORT"),
                transport_mode=_env_text("OPENHCS_UI_BRIDGE_TRANSPORT_MODE"),
                timeout_ms=_env_int("OPENHCS_UI_BRIDGE_TIMEOUT_MS"),
                auth_token=_env_text("OPENHCS_UI_BRIDGE_AUTH_TOKEN"),
            )
        )

    def apply(self, connection: UiBridgeConnectionSpec) -> UiBridgeConnectionSpec:
        return UiBridgeConnectionSpec.from_fields(
            self.fields,
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
        return UiBridgeConnectionResolution(
            connection=UiBridgeEnvironment.current().apply(connection),
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
        return UiBridgeConnectionResolution(
            connection=connection,
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
            raise ValueError(f"UI bridge process is not running: {descriptor.pid}") from exc
        except PermissionError:
            return


class UiBridgeDescriptorDirectoryCatalog:
    """Read live descriptor sets and public descriptor catalogs from the runtime dir."""

    @classmethod
    def live_descriptors(cls) -> tuple[UiBridgeDescriptorFile, ...]:
        descriptors: list[UiBridgeDescriptorFile] = []
        directory = UiBridgeDescriptorDirectoryAuthority.default_descriptor_dir()
        if not directory.exists():
            return ()
        for path in sorted(directory.glob("ui_bridge_*.json")):
            result = UiBridgeDescriptorReader.read(path)
            if result.ok and result.descriptor is not None:
                descriptors.append(result.descriptor)
        return tuple(descriptors)

    @classmethod
    def descriptor_catalog(cls) -> UiBridgeCatalog:
        descriptors: list[UiBridgeDescriptorSummary] = []
        errors: list[AgentError] = []
        directory = UiBridgeDescriptorDirectoryAuthority.default_descriptor_dir()
        if not directory.exists():
            return UiBridgeCatalog(schema_version=SCHEMA_VERSION)
        for path in sorted(directory.glob("ui_bridge_*.json")):
            result = UiBridgeDescriptorReader.read(path)
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
            return UiBridgeConnectionResolution(
                connection=replace(connection, descriptor_file_path=str(result.path)),
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
        matches = tuple(
            descriptor
            for descriptor in UiBridgeDescriptorDirectoryCatalog.live_descriptors()
            if descriptor.bridge_instance_id == bridge_instance_id
        )
        if not matches:
            return UiBridgeConnectionResolution(
                connection=connection,
                descriptor=UiBridgeDescriptorResolution(
                    status="ui_bridge_descriptor_not_found",
                ),
                errors=(
                    AgentError(
                        code="ui_bridge_descriptor_not_found",
                        message=f"No live OpenHCS UI bridge descriptor matches {bridge_instance_id!r}.",
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
        return UiBridgeConnectionResolution(
            connection=descriptor_connection,
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
    ) -> None:
        if gateway is None:
            from openhcs.agent.services.ui_bridge_transport import ZMQUiBridgeGateway

            gateway = ZMQUiBridgeGateway()
        self._gateway = gateway
        self._descriptor_resolver = descriptor_resolver

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

    def list_bridges(self) -> UiBridgeCatalog:
        return UiBridgeDescriptorDirectoryCatalog.descriptor_catalog()

    def status(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBridgeStatus:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._status_from_resolution(resolution)
        try:
            status_result = self._gateway.status(resolution.connection)
        except Exception as exc:
            return self._status_from_resolution(
                UiBridgeConnectionResolution(
                    connection=resolution.connection,
                    descriptor=resolution.descriptor,
                    errors=self._gateway_errors("ui_bridge_unreachable", exc),
                )
            )
        return resolution.project_status(status_result)

    def list_documents(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiCodeDocumentCatalog(SCHEMA_VERSION, documents=(), errors=resolution.errors)
        try:
            return self._gateway.list_documents(resolution.connection)
        except Exception as exc:
            return UiCodeDocumentCatalog(
                SCHEMA_VERSION,
                documents=(),
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiStateSurfaceCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiStateSurfaceCatalog(
                SCHEMA_VERSION,
                surfaces=(),
                errors=resolution.errors,
            )
        try:
            return self._gateway.list_state_surfaces(resolution.connection)
        except Exception as exc:
            return UiStateSurfaceCatalog(
                SCHEMA_VERSION,
                surfaces=(),
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def list_actions(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiActionCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiActionCatalog(
                SCHEMA_VERSION,
                actions=(),
                errors=resolution.errors,
            )
        try:
            return self._gateway.list_actions(resolution.connection)
        except Exception as exc:
            return UiActionCatalog(
                SCHEMA_VERSION,
                actions=(),
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def list_windows(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiWindowCatalog(
                schema_version=SCHEMA_VERSION,
                windows=(),
                errors=resolution.errors,
            )
        try:
            return self._gateway.list_windows(resolution.connection)
        except Exception as exc:
            return UiWindowCatalog(
                schema_version=SCHEMA_VERSION,
                windows=(),
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def list_object_state_scopes(
        self,
        request: UiObjectStateScopeListRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiObjectStateScopeCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._object_state_scope_catalog_error(resolution.errors)
        try:
            return self._gateway.list_object_state_scopes(
                resolution.connection,
                request,
            )
        except Exception as exc:
            return self._object_state_scope_catalog_error(
                self._gateway_errors("ui_bridge_unavailable", exc)
            )

    def get_document(
        self,
        request: UiCodeDocumentRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocument:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._document_error(request, resolution.errors)
        try:
            return self._gateway.get_document(resolution.connection, request)
        except Exception as exc:
            return self._document_error(
                request,
                self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def get_state_surface(
        self,
        request: UiStateSurfaceRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiStateSurfaceDocument:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._state_surface_error(request, resolution.errors)
        try:
            return self._gateway.get_state_surface(resolution.connection, request)
        except Exception as exc:
            return self._state_surface_error(
                request,
                self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def invoke_action(
        self,
        request: UiActionInvokeRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiActionInvokeResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._action_error(request, resolution.errors)
        try:
            return self._gateway.invoke_action(resolution.connection, request)
        except Exception as exc:
            return self._action_error(
                request,
                self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def focus_window(
        self,
        request: UiWindowFocusRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiWindowFocusResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._window_focus_error(request, resolution.errors)
        try:
            return self._gateway.focus_window(resolution.connection, request)
        except Exception as exc:
            return self._window_focus_error(
                request,
                self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def validate_document(
        self,
        request: UiCodeDocumentValidationRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentValidationResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=resolution.errors,
            )
        try:
            return self._gateway.validate_document(resolution.connection, request)
        except Exception as exc:
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def apply_document(
        self,
        request: UiCodeDocumentApplyRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiCodeDocumentApplyResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=False,
                base_revision_token=request.base_revision_token,
                errors=resolution.errors,
            )
        try:
            return self._gateway.apply_document(resolution.connection, request)
        except Exception as exc:
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=False,
                base_revision_token=request.base_revision_token,
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def list_snapshots(
        self,
        request: UiSnapshotListRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._snapshot_catalog_error(resolution.errors)
        try:
            return self._gateway.list_snapshots(resolution.connection, request)
        except Exception as exc:
            return self._snapshot_catalog_error(
                self._gateway_errors("ui_bridge_unavailable", exc)
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
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._restore_error(resolution.errors)
        try:
            return self._gateway.restore_snapshot(resolution.connection, request)
        except Exception as exc:
            return self._restore_error(
                self._gateway_errors("ui_bridge_unavailable", exc)
            )

    def time_travel_head(
        self,
        request: UiTimeTravelHeadRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotRestoreResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._restore_error(resolution.errors)
        try:
            return self._gateway.time_travel_head(resolution.connection, request)
        except Exception as exc:
            return self._restore_error(
                self._gateway_errors("ui_bridge_unavailable", exc)
            )

    def list_branches(
        self,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBranchCatalog:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiBranchCatalog(SCHEMA_VERSION, current_branch="", branches=(), errors=resolution.errors)
        try:
            return self._gateway.list_branches(resolution.connection)
        except Exception as exc:
            return UiBranchCatalog(
                SCHEMA_VERSION,
                current_branch="",
                branches=(),
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def switch_branch(
        self,
        request: UiBranchSwitchRequest,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiSnapshotRestoreResult:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return self._restore_error(resolution.errors)
        try:
            return self._gateway.switch_branch(resolution.connection, request)
        except Exception as exc:
            return self._restore_error(
                self._gateway_errors("ui_bridge_unavailable", exc)
            )

    def get_operation_status(
        self,
        operation_id: str,
        connection: UiBridgeConnectionSpec = DEFAULT_UI_BRIDGE_CONNECTION_SPEC,
    ) -> UiBridgeOperationRef:
        resolution = self._resolve(connection)
        if not resolution.ok:
            return UiBridgeOperationRef(
                schema_version=SCHEMA_VERSION,
                identity=UiBridgeOperationIdentity(
                    operation_id=operation_id,
                    route=UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
                ),
                status=UiBridgeOperationStatus.UNAVAILABLE.value,
                started_at_unix=0.0,
                errors=resolution.errors,
            )
        try:
            return self._gateway.get_operation_status(resolution.connection, operation_id)
        except Exception as exc:
            return UiBridgeOperationRef(
                schema_version=SCHEMA_VERSION,
                identity=UiBridgeOperationIdentity(
                    operation_id=operation_id,
                    route=UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
                ),
                status=UiBridgeOperationStatus.UNAVAILABLE.value,
                started_at_unix=0.0,
                errors=self._gateway_errors("ui_bridge_unavailable", exc),
            )

    def _resolve(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiBridgeConnectionResolution:
        return self._descriptor_resolver.resolve(connection)

    @staticmethod
    def _status_from_resolution(
        resolution: UiBridgeConnectionResolution,
    ) -> UiBridgeStatus:
        return UiBridgeStatus(
            schema_version=SCHEMA_VERSION,
            reachable=False,
            connection=_public_connection(resolution.connection),
            descriptor_file_path=resolution.connection.descriptor_file_path,
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
            current_revision_token="",
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
            current_revision_token="",
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
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=False,
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
    def _object_state_scope_catalog_error(
        errors: tuple[AgentError, ...],
    ) -> UiObjectStateScopeCatalog:
        return UiObjectStateScopeCatalog(
            schema_version=SCHEMA_VERSION,
            object_state_token=0,
            current_branch="",
            current_snapshot_index=-1,
            time_travel_state=UiTimeTravelRuntimeState(active=False),
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
            time_travel_state=UiTimeTravelRuntimeState(active=False),
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
        if isinstance(exception, UiBridgeGatewayErrorABC):
            return exception.agent_errors(code)
        return (AgentError.from_exception(code, exception),)


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
