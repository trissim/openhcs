"""Typed debug/test-mode substrate for OpenHCS runtime execution."""

from __future__ import annotations

import json
import hashlib
import shutil
import time
import traceback
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping
from uuid import uuid4

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactPlan
from openhcs.core.function_patterns import CompiledFunctionInvocation
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.vfs_protocol import FileManagerLike

if TYPE_CHECKING:
    from openhcs.core.debug_views import DebugViewModel
    from openhcs.core.pipeline.compiler import CompiledRuntimeEnvironmentPlan
    from openhcs.core.runtime_stores import RuntimeValueStore


class DebugEventType(Enum):
    """Runtime boundary represented by one debug event."""

    BEFORE_INVOCATION = "before_invocation"
    AFTER_INVOCATION = "after_invocation"
    EXCEPTION = "exception"


class DebugCommandType(Enum):
    """User-facing debug/test-mode command."""

    TOGGLE = "toggle"
    STEP = "step"
    RUN = "run"
    RUN_TO_PAUSE = "run_to_pause"
    RESTART = "restart"
    CHOOSE_SOURCE_GROUP = "choose_source_group"
    RANDOM_SOURCE_GROUP = "random_source_group"
    STOP = "stop"


class DebugCommandDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one debug command."""

    __registry_key__ = "command_type"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[DebugCommandType, type["DebugCommandDeclarationBase"]]
    ] = {}

    command_type: ClassVar[DebugCommandType | None] = None
    advances_one_boundary: ClassVar[bool] = False

    @classmethod
    def require_command_type(cls) -> DebugCommandType:
        if cls.command_type is None:
            raise TypeError(f"{cls.__name__} does not declare a debug command.")
        return cls.command_type

    @classmethod
    def for_command_type(
        cls,
        command_type: DebugCommandType,
    ) -> type["DebugCommandDeclarationBase"]:
        return cls.__registry__[command_type]


class AdvancesOneDebugBoundary:
    """Trait for debug commands that release exactly one boundary."""

    advances_one_boundary: ClassVar[bool] = True


class ToggleDebugCommand(DebugCommandDeclarationBase):
    command_type = DebugCommandType.TOGGLE


class StepDebugCommand(AdvancesOneDebugBoundary, DebugCommandDeclarationBase):
    command_type = DebugCommandType.STEP


class RunDebugCommand(DebugCommandDeclarationBase):
    command_type = DebugCommandType.RUN


class RunToPauseDebugCommand(AdvancesOneDebugBoundary, DebugCommandDeclarationBase):
    command_type = DebugCommandType.RUN_TO_PAUSE


class RestartDebugCommand(AdvancesOneDebugBoundary, DebugCommandDeclarationBase):
    command_type = DebugCommandType.RESTART


class ChooseSourceGroupDebugCommand(DebugCommandDeclarationBase):
    command_type = DebugCommandType.CHOOSE_SOURCE_GROUP


class RandomSourceGroupDebugCommand(DebugCommandDeclarationBase):
    command_type = DebugCommandType.RANDOM_SOURCE_GROUP


class StopDebugCommand(DebugCommandDeclarationBase):
    command_type = DebugCommandType.STOP


class DebugBoundaryOutcome(str, Enum):
    """Runtime outcome represented by one debug boundary event."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class DebugTimelineNodeState(str, Enum):
    """Debugger timeline state derived from a debug boundary event."""

    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class DebugBoundaryEventDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one debug event type."""

    __registry_key__ = "event_type"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[DebugEventType, type["DebugBoundaryEventDeclarationBase"]]
    ] = {}

    event_type: ClassVar[DebugEventType | None] = None
    progress_status: ClassVar[ProgressStatus]
    boundary_outcome: ClassVar[DebugBoundaryOutcome]
    timeline_node_state: ClassVar[DebugTimelineNodeState]
    reports_output_artifacts: ClassVar[bool] = False

    @classmethod
    def require_event_type(cls) -> DebugEventType:
        if cls.event_type is None:
            raise TypeError(f"{cls.__name__} does not declare a debug event type.")
        return cls.event_type

    @classmethod
    def for_event_type(
        cls,
        event_type: DebugEventType,
    ) -> type["DebugBoundaryEventDeclarationBase"]:
        return cls.__registry__[event_type]


class StartedDebugBoundary:
    """Trait for before-invocation debug boundary semantics."""

    progress_status: ClassVar[ProgressStatus] = ProgressStatus.STARTED
    boundary_outcome: ClassVar[DebugBoundaryOutcome] = DebugBoundaryOutcome.STARTED
    timeline_node_state: ClassVar[DebugTimelineNodeState] = (
        DebugTimelineNodeState.STARTED
    )


class CompletedDebugBoundary:
    """Trait for completed debug boundary semantics."""

    progress_status: ClassVar[ProgressStatus] = ProgressStatus.SUCCESS
    boundary_outcome: ClassVar[DebugBoundaryOutcome] = DebugBoundaryOutcome.COMPLETED
    timeline_node_state: ClassVar[DebugTimelineNodeState] = (
        DebugTimelineNodeState.COMPLETED
    )


class FailedDebugBoundary:
    """Trait for failed debug boundary semantics."""

    progress_status: ClassVar[ProgressStatus] = ProgressStatus.ERROR
    boundary_outcome: ClassVar[DebugBoundaryOutcome] = DebugBoundaryOutcome.FAILED
    timeline_node_state: ClassVar[DebugTimelineNodeState] = DebugTimelineNodeState.FAILED


class ReportsOutputArtifactsDebugBoundary:
    """Trait for boundary events that can report produced artifacts."""

    reports_output_artifacts: ClassVar[bool] = True


class BeforeInvocationDebugBoundary(
    StartedDebugBoundary,
    DebugBoundaryEventDeclarationBase,
):
    event_type = DebugEventType.BEFORE_INVOCATION


class AfterInvocationDebugBoundary(
    ReportsOutputArtifactsDebugBoundary,
    CompletedDebugBoundary,
    DebugBoundaryEventDeclarationBase,
):
    event_type = DebugEventType.AFTER_INVOCATION


class ExceptionDebugBoundary(
    ReportsOutputArtifactsDebugBoundary,
    FailedDebugBoundary,
    DebugBoundaryEventDeclarationBase,
):
    event_type = DebugEventType.EXCEPTION


class DebugControlMessageType(Enum):
    """OpenHCS-specific debug control requests sent over the ZMQ control socket."""

    READ_SNAPSHOT = "openhcs_debug_read_snapshot"
    WORKER_COMMAND = "openhcs_debug_worker_command"
    EXPORT_ARTIFACT = "openhcs_debug_export_artifact"
    INSPECT_RUNTIME = "openhcs_debug_inspect_runtime"


class DebugPausedWorkerState(Enum):
    """Lifecycle state for one persistent debug worker controller."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class DebugPausedWorkerStatus:
    """Serializable status for a paused persistent debug worker."""

    debug_session_id: str
    state: DebugPausedWorkerState
    cursor: DebugCursor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debug_session_id": self.debug_session_id,
            "state": self.state.value,
            "cursor": (
                None if self.cursor is None else DebugJsonCodec.cursor_to_record(self.cursor)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DebugPausedWorkerStatus":
        cursor = payload.get("cursor")
        return cls(
            debug_session_id=str(payload["debug_session_id"]),
            state=DebugPausedWorkerState(str(payload["state"])),
            cursor=None if cursor is None else DebugJsonCodec.cursor_from_record(cursor),
        )

    @classmethod
    def from_controller(
        cls,
        controller: "DebugPausedWorkerController",
    ) -> "DebugPausedWorkerStatus":
        return cls(
            debug_session_id=controller.debug_session_id,
            state=controller._state,
            cursor=controller._cursor,
        )


class DebugReplayMode(Enum):
    """How debug replay should reuse already prepared execution state."""

    COLD = "cold"
    WARM_ARTIFACT = "warm_artifact"
    PERSISTENT_PAUSED_WORKER = "persistent_paused_worker"

    @property
    def retains_compile_artifact(self) -> bool:
        return self is not DebugReplayMode.COLD


@dataclass(frozen=True, slots=True)
class DebugCommand:
    """Typed GUI command for a debug/test-mode session."""

    command_type: DebugCommandType
    debug_session_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.command_type, DebugCommandType):
            raise TypeError(
                "DebugCommand.command_type must be DebugCommandType, "
                f"got {type(self.command_type).__name__}."
            )


@dataclass(frozen=True, slots=True)
class DebugCursor:
    """Invocation-aware debug cursor."""

    step_index: int
    step_scope_id: str | None
    group_key: str | None
    invocation_key: str | None
    pattern_group_identity: str | None = None

    @classmethod
    def from_invocation(
        cls,
        *,
        step_index: int,
        step_scope_id: str | None,
        invocation: CompiledFunctionInvocation,
        pattern_group_identity: str | None = None,
    ) -> "DebugCursor":
        return cls(
            step_index=step_index,
            step_scope_id=step_scope_id,
            group_key=invocation.key.group_key,
            invocation_key=DebugCursor.invocation_key_text(invocation),
            pattern_group_identity=pattern_group_identity,
        )

    @staticmethod
    def invocation_key_text(invocation: CompiledFunctionInvocation) -> str:
        key = invocation.key
        return f"{key.group_key}:{key.position}:{key.function_name}"

    @staticmethod
    def invocation_key_parts_text(
        *,
        group_key: str,
        position: int,
        function_name: str,
    ) -> str:
        return f"{group_key}:{position}:{function_name}"

    def matches_invocation_key_parts(
        self,
        *,
        group_key: str,
        position: int,
        function_name: str,
    ) -> bool:
        if self.invocation_key is None:
            return False
        return (
            self.group_key == group_key
            and self.invocation_key
            == self.invocation_key_parts_text(
                group_key=group_key,
                position=position,
                function_name=function_name,
            )
        )


class DebugJsonCodec:
    """JSON-record codec derived from debug dataclass field authority."""

    @staticmethod
    def cursor_to_record(cursor: DebugCursor) -> dict[str, Any]:
        return DebugJsonCodec.dataclass_record(cursor)

    @staticmethod
    def cursor_from_record(record: Mapping[str, Any]) -> DebugCursor:
        cursor_record = dict(record)
        cursor_record["step_index"] = int(cursor_record["step_index"])
        return DebugJsonCodec.dataclass_from_record(DebugCursor, cursor_record)

    @staticmethod
    def dataclass_record(value: object) -> dict[str, Any]:
        if not is_dataclass(value):
            raise TypeError(
                "DebugJsonCodec.dataclass_record requires a dataclass instance, "
                f"got {type(value).__name__}."
            )
        return asdict(value)

    @staticmethod
    def dataclass_from_record(dataclass_type: type, record: Mapping[str, Any]):
        if not is_dataclass(dataclass_type):
            raise TypeError(
                "DebugJsonCodec.dataclass_from_record requires a dataclass type, "
                f"got {dataclass_type!r}."
            )
        missing_fields = tuple(
            field_info.name
            for field_info in fields(dataclass_type)
            if field_info.name not in record
        )
        if missing_fields:
            raise KeyError(
                f"Debug record missing dataclass fields: {', '.join(missing_fields)}"
            )
        return dataclass_type(
            **{
                field_info.name: record[field_info.name]
                for field_info in fields(dataclass_type)
            }
        )


@dataclass(frozen=True, slots=True)
class DebugArtifactRef:
    """Reference to a debug artifact or preview payload."""

    kind: ArtifactKind
    name: str
    cursor: DebugCursor
    storage_ref: str
    storage_backend: str | None = None
    preview_ref: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    identity: "DebugArtifactIdentity | None" = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError(
                "DebugArtifactRef.kind must be ArtifactKind, "
                f"got {type(self.kind).__name__}."
            )
        if not self.name:
            raise ValueError("DebugArtifactRef.name cannot be empty.")
        if not self.storage_ref:
            raise ValueError("DebugArtifactRef.storage_ref cannot be empty.")

    @classmethod
    def from_artifact_plan(
        cls,
        *,
        plan: ArtifactPlan,
        cursor: DebugCursor,
    ) -> "DebugArtifactRef":
        """Build a metadata-only debug reference from a compiled artifact plan."""

        return cls(
            kind=plan.kind,
            name=plan.name,
            cursor=cursor,
            storage_ref=plan.path,
            storage_backend="memory",
            identity=DebugArtifactIdentity.from_artifact_plan(plan),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "cursor": DebugJsonCodec.cursor_to_record(self.cursor),
            "storage_ref": self.storage_ref,
            "storage_backend": self.storage_backend,
            "preview_ref": self.preview_ref,
            "shape": self.shape,
            "dtype": self.dtype,
            "identity": None if self.identity is None else self.identity.to_json_dict(),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "DebugArtifactRef":
        shape = data.get("shape")
        return cls(
            kind=ArtifactKind(data["kind"]),
            name=str(data["name"]),
            cursor=DebugJsonCodec.cursor_from_record(data["cursor"]),
            storage_ref=str(data["storage_ref"]),
            storage_backend=data.get("storage_backend"),
            preview_ref=data.get("preview_ref"),
            shape=None if shape is None else tuple(int(value) for value in shape),
            dtype=data.get("dtype"),
            identity=(
                None
                if data.get("identity") is None
                else DebugArtifactIdentity.from_json_dict(data["identity"])
            ),
            content_digest=data.get("content_digest"),
        )

    def with_content_digest(self, digest: str | None) -> "DebugArtifactRef":
        return DebugArtifactRef(
            kind=self.kind,
            name=self.name,
            cursor=self.cursor,
            storage_ref=self.storage_ref,
            storage_backend=self.storage_backend,
            preview_ref=self.preview_ref,
            shape=self.shape,
            dtype=self.dtype,
            identity=self.identity,
            content_digest=digest,
        )


@dataclass(frozen=True, slots=True)
class DebugArtifactIdentity:
    """Stable logical identity for matching debug artifacts across replay runs."""

    name: str
    kind: ArtifactKind
    group_keys: tuple[str | None, ...] = (None,)
    producer_step_index: str | None = None
    producer_step_scope_id: str | None = None
    producer_step_name: str | None = None
    settings_digest: str | None = None

    @classmethod
    def from_artifact_plan(cls, plan: ArtifactPlan) -> "DebugArtifactIdentity":
        output_plan = plan if isinstance(plan, ArtifactOutputPlan) else None
        return cls(
            name=plan.name,
            kind=plan.kind,
            group_keys=tuple(plan.group_keys or (None,)),
            producer_step_index=(
                None
                if output_plan is None or output_plan.producer_step_index is None
                else str(output_plan.producer_step_index)
            ),
            producer_step_scope_id=(
                None if output_plan is None else output_plan.producer_step_scope_id
            ),
            producer_step_name=(
                None if output_plan is None else output_plan.producer_step_name
            ),
            settings_digest=cls.settings_digest_for_plan(plan),
        )

    def matches(self, other: "DebugArtifactIdentity") -> bool:
        return (
            self.name == other.name
            and self.kind is other.kind
            and self.group_keys == other.group_keys
            and self.producer_step_index == other.producer_step_index
            and self.producer_step_scope_id == other.producer_step_scope_id
            and self.producer_step_name == other.producer_step_name
            and self.settings_digest == other.settings_digest
        )

    @staticmethod
    def settings_digest_for_plan(plan: ArtifactPlan) -> str | None:
        if not isinstance(plan, ArtifactOutputPlan):
            return None
        materialization = plan.materialization
        if materialization is None:
            return None
        payload = json.dumps(
            {
                "materialization": repr(materialization),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "group_keys": list(self.group_keys),
            "producer_step_index": self.producer_step_index,
            "producer_step_scope_id": self.producer_step_scope_id,
            "producer_step_name": self.producer_step_name,
            "settings_digest": self.settings_digest,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "DebugArtifactIdentity":
        return cls(
            name=str(data["name"]),
            kind=ArtifactKind(data["kind"]),
            group_keys=tuple(data.get("group_keys", (None,))),
            producer_step_index=data.get("producer_step_index"),
            producer_step_scope_id=data.get("producer_step_scope_id"),
            producer_step_name=data.get("producer_step_name"),
            settings_digest=data.get("settings_digest"),
        )


@dataclass(frozen=True, slots=True)
class DebugArtifactRefProjection:
    """Typed debug projection over compiled artifact plans."""

    refs: tuple[DebugArtifactRef, ...] = ()

    @classmethod
    def from_artifact_plans(
        cls,
        *,
        artifact_plans: Mapping[str, ArtifactPlan],
        cursor: DebugCursor,
    ) -> "DebugArtifactRefProjection":
        return cls(
            refs=tuple(
                DebugArtifactRef.from_artifact_plan(plan=plan, cursor=cursor)
                for plan in artifact_plans.values()
            )
        )

    def of_kind(self, kind: ArtifactKind) -> tuple[DebugArtifactRef, ...]:
        resolved_kind = kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
        return tuple(ref for ref in self.refs if ref.kind is resolved_kind)

    @property
    def measurement_refs(self) -> tuple[DebugArtifactRef, ...]:
        return self.of_kind(ArtifactKind.MEASUREMENTS)

    @property
    def relationship_refs(self) -> tuple[DebugArtifactRef, ...]:
        return self.of_kind(ArtifactKind.RELATIONSHIPS)


@dataclass(frozen=True, slots=True)
class DebugInvocationParameter:
    """JSON-safe parameter projection for one compiled function invocation."""

    name: str
    value_repr: str

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, Any],
    ) -> tuple["DebugInvocationParameter", ...]:
        return tuple(
            cls(name=str(name), value_repr=repr(value))
            for name, value in sorted(kwargs.items(), key=lambda item: str(item[0]))
        )

    def to_json_dict(self) -> dict[str, str]:
        return DebugJsonCodec.dataclass_record(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "DebugInvocationParameter":
        return cls(name=str(data["name"]), value_repr=str(data["value_repr"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class DebugBoundaryState(ABC, metaclass=AutoRegisterMeta):
    """Common cursor/artifact state shared by debug events and snapshots."""

    __registry_key__ = "boundary_kind"
    __skip_if_no_key__ = True
    boundary_kind: ClassVar[str | None] = None

    cursor: DebugCursor
    step_name: str
    callable_name: str | None = None
    axis_id: str | None = None
    input_artifact_refs: tuple[DebugArtifactRef, ...] = ()
    output_artifact_refs: tuple[DebugArtifactRef, ...] = ()
    preview_refs: tuple[DebugArtifactRef, ...] = ()
    measurement_refs: tuple[DebugArtifactRef, ...] = ()
    relationship_refs: tuple[DebugArtifactRef, ...] = ()
    timing_seconds: float | None = None
    exception: str | None = None
    invocation_parameters: tuple[DebugInvocationParameter, ...] = ()


@dataclass(frozen=True, slots=True)
class DebugSnapshot(DebugBoundaryState):
    """Snapshot metadata for one debug execution boundary."""

    boundary_kind: ClassVar[str] = "snapshot"

    snapshot_id: str
    source_paths: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "cursor": DebugJsonCodec.cursor_to_record(self.cursor),
            "step_name": self.step_name,
            "callable_name": self.callable_name,
            "axis_id": self.axis_id,
            "source_paths": list(self.source_paths),
            "input_artifact_refs": [
                ref.to_json_dict() for ref in self.input_artifact_refs
            ],
            "output_artifact_refs": [
                ref.to_json_dict() for ref in self.output_artifact_refs
            ],
            "preview_refs": [ref.to_json_dict() for ref in self.preview_refs],
            "measurement_refs": [
                ref.to_json_dict() for ref in self.measurement_refs
            ],
            "relationship_refs": [
                ref.to_json_dict() for ref in self.relationship_refs
            ],
            "timing_seconds": self.timing_seconds,
            "exception": self.exception,
            "invocation_parameters": [
                parameter.to_json_dict()
                for parameter in self.invocation_parameters
            ],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "DebugSnapshot":
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            cursor=DebugJsonCodec.cursor_from_record(data["cursor"]),
            step_name=str(data["step_name"]),
            callable_name=data.get("callable_name"),
            axis_id=data.get("axis_id"),
            source_paths=tuple(str(path) for path in data.get("source_paths", ())),
            input_artifact_refs=tuple(
                DebugArtifactRef.from_json_dict(ref)
                for ref in data.get("input_artifact_refs", ())
            ),
            output_artifact_refs=tuple(
                DebugArtifactRef.from_json_dict(ref)
                for ref in data.get("output_artifact_refs", ())
            ),
            preview_refs=tuple(
                DebugArtifactRef.from_json_dict(ref)
                for ref in data.get("preview_refs", ())
            ),
            measurement_refs=tuple(
                DebugArtifactRef.from_json_dict(ref)
                for ref in data.get("measurement_refs", ())
            ),
            relationship_refs=tuple(
                DebugArtifactRef.from_json_dict(ref)
                for ref in data.get("relationship_refs", ())
            ),
            timing_seconds=data.get("timing_seconds"),
            exception=data.get("exception"),
            invocation_parameters=tuple(
                DebugInvocationParameter.from_json_dict(parameter)
                for parameter in data.get("invocation_parameters", ())
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DebugSessionRequest(ABC, metaclass=AutoRegisterMeta):
    """Base request carrying a non-empty debug session id."""

    __registry_key__ = "request_kind"
    __skip_if_no_key__ = True
    request_kind: ClassVar[str | None] = None

    debug_session_id: str

    def __post_init__(self) -> None:
        self.validate_required_non_empty_values()

    @property
    def required_non_empty_values(self) -> tuple[tuple[str, str | None], ...]:
        return (("debug_session_id", self.debug_session_id),)

    def validate_required_non_empty_values(self) -> None:
        for field_name, value in self.required_non_empty_values:
            if not value:
                raise ValueError(f"{field_name} cannot be empty.")


@dataclass(frozen=True, slots=True)
class DebugSnapshotReadRequest(DebugSessionRequest):
    """Control-channel request for reading one debug snapshot from the server."""

    request_kind: ClassVar[str] = "snapshot_read"

    snapshot_id: str
    snapshot_store_ref: str
    snapshot_store_backend: str | None = None

    @property
    def required_non_empty_values(self) -> tuple[tuple[str, str | None], ...]:
        return (
            ("debug_session_id", self.debug_session_id),
            ("snapshot_id", self.snapshot_id),
            ("snapshot_store_ref", self.snapshot_store_ref),
        )


@dataclass(frozen=True, slots=True)
class DebugSnapshotReadControlPayload:
    """Wire payload for one debug snapshot read control request."""

    debug_session_id: str
    snapshot_id: str
    snapshot_store_ref: str
    snapshot_store_backend: str | None = None
    message_type: DebugControlMessageType = DebugControlMessageType.READ_SNAPSHOT

    @classmethod
    def from_request(
        cls,
        request: DebugSnapshotReadRequest,
    ) -> "DebugSnapshotReadControlPayload":
        return cls(
            debug_session_id=request.debug_session_id,
            snapshot_id=request.snapshot_id,
            snapshot_store_ref=request.snapshot_store_ref,
            snapshot_store_backend=request.snapshot_store_backend,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugSnapshotReadControlPayload":
        message_type = DebugControlMessageType(str(payload["type"]))
        if message_type is not DebugControlMessageType.READ_SNAPSHOT:
            raise ValueError(
                f"Unsupported debug snapshot read control type: {message_type.value!r}."
            )
        return cls(
            message_type=message_type,
            debug_session_id=str(payload["debug_session_id"]),
            snapshot_id=str(payload["snapshot_id"]),
            snapshot_store_ref=str(payload["snapshot_store_ref"]),
            snapshot_store_backend=payload.get("snapshot_store_backend"),
        )

    def to_request(self) -> DebugSnapshotReadRequest:
        return DebugSnapshotReadRequest(
            debug_session_id=self.debug_session_id,
            snapshot_id=self.snapshot_id,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "debug_session_id": self.debug_session_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_store_ref": self.snapshot_store_ref,
            "snapshot_store_backend": self.snapshot_store_backend,
        }


@dataclass(frozen=True, slots=True)
class DebugSnapshotReadResponse:
    """Control-channel response carrying snapshot metadata only."""

    snapshot: DebugSnapshot

    def to_control_response(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "snapshot": self.snapshot.to_json_dict(),
        }

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugSnapshotReadResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        return cls(snapshot=DebugSnapshot.from_json_dict(payload["snapshot"]))


@dataclass(frozen=True, slots=True)
class DebugArtifactExportRequest(DebugSessionRequest):
    """Control-channel request for server-side artifact export/materialization."""

    request_kind: ClassVar[str] = "artifact_export"

    artifact_ref: DebugArtifactRef
    export_root: str
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None

    @property
    def required_non_empty_values(self) -> tuple[tuple[str, str | None], ...]:
        return (
            ("debug_session_id", self.debug_session_id),
            ("export_root", self.export_root),
        )


@dataclass(frozen=True, slots=True)
class DebugArtifactExportControlPayload:
    """Wire payload for server-side artifact export/materialization."""

    debug_session_id: str
    artifact_ref: DebugArtifactRef
    export_root: str
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    message_type: DebugControlMessageType = DebugControlMessageType.EXPORT_ARTIFACT

    @classmethod
    def from_request(
        cls,
        request: DebugArtifactExportRequest,
    ) -> "DebugArtifactExportControlPayload":
        return cls(
            debug_session_id=request.debug_session_id,
            artifact_ref=request.artifact_ref,
            export_root=request.export_root,
            snapshot_store_ref=request.snapshot_store_ref,
            snapshot_store_backend=request.snapshot_store_backend,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugArtifactExportControlPayload":
        message_type = DebugControlMessageType(str(payload["type"]))
        if message_type is not DebugControlMessageType.EXPORT_ARTIFACT:
            raise ValueError(
                f"Unsupported debug artifact export type: {message_type.value!r}."
            )
        return cls(
            message_type=message_type,
            debug_session_id=str(payload["debug_session_id"]),
            artifact_ref=DebugArtifactRef.from_json_dict(payload["artifact_ref"]),
            export_root=str(payload["export_root"]),
            snapshot_store_ref=payload.get("snapshot_store_ref"),
            snapshot_store_backend=payload.get("snapshot_store_backend"),
        )

    def to_request(self) -> DebugArtifactExportRequest:
        return DebugArtifactExportRequest(
            debug_session_id=self.debug_session_id,
            artifact_ref=self.artifact_ref,
            export_root=self.export_root,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "debug_session_id": self.debug_session_id,
            "artifact_ref": self.artifact_ref.to_json_dict(),
            "export_root": self.export_root,
            "snapshot_store_ref": self.snapshot_store_ref,
            "snapshot_store_backend": self.snapshot_store_backend,
        }


@dataclass(frozen=True, slots=True)
class DebugArtifactExportResponse:
    """Control-channel response containing the server-side export path."""

    exported_ref: str

    def to_control_response(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "exported_ref": self.exported_ref,
        }

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugArtifactExportResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        return cls(exported_ref=str(payload["exported_ref"]))


@dataclass(frozen=True, slots=True)
class DebugRuntimeInspectionRequest(DebugSessionRequest):
    """Control-channel request for a paused worker runtime inspection view."""

    request_kind: ClassVar[str] = "runtime_inspection"


@dataclass(frozen=True, slots=True)
class DebugRuntimeInspectionControlPayload:
    """Wire payload for live runtime inspection requests."""

    debug_session_id: str
    message_type: DebugControlMessageType = DebugControlMessageType.INSPECT_RUNTIME

    @classmethod
    def from_request(
        cls,
        request: DebugRuntimeInspectionRequest,
    ) -> "DebugRuntimeInspectionControlPayload":
        return cls(debug_session_id=request.debug_session_id)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugRuntimeInspectionControlPayload":
        message_type = DebugControlMessageType(str(payload["type"]))
        if message_type is not DebugControlMessageType.INSPECT_RUNTIME:
            raise ValueError(
                "DebugRuntimeInspectionControlPayload requires "
                f"{DebugControlMessageType.INSPECT_RUNTIME.value}, got "
                f"{message_type.value}."
            )
        return cls(
            message_type=message_type,
            debug_session_id=str(payload["debug_session_id"]),
        )

    def to_request(self) -> DebugRuntimeInspectionRequest:
        return DebugRuntimeInspectionRequest(debug_session_id=self.debug_session_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "debug_session_id": self.debug_session_id,
        }


@dataclass(frozen=True, slots=True)
class DebugRuntimeInspectionResponse:
    """Control-channel response containing a renderer-independent debug view."""

    view_model: "DebugViewModel"

    def to_control_response(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "view_model": self.view_model.to_json_dict(),
        }

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugRuntimeInspectionResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        from openhcs.core.debug_views import DebugViewModel

        return cls(view_model=DebugViewModel.from_json_dict(payload["view_model"]))


@dataclass(frozen=True, slots=True)
class DebugArtifactExportPlan:
    """Filesystem export plan executed in the worker/server namespace."""

    artifact_ref: DebugArtifactRef
    export_root: Path
    filemanager: FileManagerLike | None = None

    @property
    def source_path(self) -> Path:
        return Path(self.artifact_ref.preview_ref or self.artifact_ref.storage_ref)

    @property
    def destination_path(self) -> Path:
        suffix = self.source_path.suffix
        return self.export_root / f"{self.artifact_ref.name}{suffix}"

    def export(self) -> Path:
        if self.artifact_ref.storage_backend is not None and self.filemanager is not None:
            return self._export_vfs_payload()
        source_path = self.source_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Debug artifact source does not exist: {source_path}"
            )
        self.export_root.mkdir(parents=True, exist_ok=True)
        destination_path = self.destination_path
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)
        return destination_path

    def _export_vfs_payload(self) -> Path:
        self.export_root.mkdir(parents=True, exist_ok=True)
        destination_path = self.destination_path
        payload = self.filemanager.load(
            self.artifact_ref.storage_ref,
            self.artifact_ref.storage_backend,
        )
        if isinstance(payload, bytes):
            destination_path.write_bytes(payload)
        elif isinstance(payload, str):
            destination_path.write_text(payload, encoding="utf-8")
        else:
            destination_path.write_text(json.dumps(payload, default=repr), encoding="utf-8")
        return destination_path


@dataclass(frozen=True, slots=True)
class DebugWarmReplayArtifactAvailability:
    """Availability result for one warm-replay artifact output."""

    artifact_ref: DebugArtifactRef
    available: bool
    hydrated_from: str | None = None


@dataclass(frozen=True, slots=True)
class DebugWarmReplayArtifactReusePlan:
    """Nominal validation/hydration plan for skipped warm-replay outputs."""

    artifact_refs: tuple[DebugArtifactRef, ...]
    snapshot_store: "DebugSnapshotStore | None" = None

    @classmethod
    def from_artifact_plans(
        cls,
        *,
        artifact_plans: Mapping[str, ArtifactPlan],
        cursor: DebugCursor,
        snapshot_store: "DebugSnapshotStore | None" = None,
    ) -> "DebugWarmReplayArtifactReusePlan":
        return cls(
            artifact_refs=DebugArtifactRefProjection.from_artifact_plans(
                artifact_plans=artifact_plans,
                cursor=cursor,
            ).refs,
            snapshot_store=snapshot_store,
        )

    def prepare(self, context: object) -> tuple[DebugWarmReplayArtifactAvailability, ...]:
        hydration_context = DebugArtifactHydrationContext.from_execution_context(context)
        return tuple(
            self._prepare_ref(hydration_context, artifact_ref)
            for artifact_ref in self.artifact_refs
        )

    def require_available(self, context: object) -> None:
        results = self.prepare(context)
        missing = tuple(result.artifact_ref for result in results if not result.available)
        if missing:
            missing_text = ", ".join(
                f"{ref.kind.value}:{ref.name}@{ref.storage_ref}" for ref in missing
            )
            raise RuntimeError(
                "Warm debug replay cannot skip upstream step because expected "
                f"artifact outputs are unavailable: {missing_text}."
            )

    def _prepare_ref(
        self,
        hydration_context: "DebugArtifactHydrationContext",
        artifact_ref: DebugArtifactRef,
    ) -> DebugWarmReplayArtifactAvailability:
        if hydration_context.artifact_available(artifact_ref):
            return DebugWarmReplayArtifactAvailability(artifact_ref, available=True)
        source_ref = self._matching_snapshot_ref(artifact_ref)
        if source_ref is not None and hydration_context.hydrate(
            source_ref=source_ref,
            destination_ref=artifact_ref,
        ):
            return DebugWarmReplayArtifactAvailability(
                artifact_ref,
                available=True,
                hydrated_from=source_ref.storage_ref,
            )
        return DebugWarmReplayArtifactAvailability(artifact_ref, available=False)

    def _matching_snapshot_ref(
        self,
        artifact_ref: DebugArtifactRef,
    ) -> DebugArtifactRef | None:
        if self.snapshot_store is None or artifact_ref.identity is None:
            return None
        for snapshot_id in reversed(self.snapshot_store.list_snapshot_ids()):
            snapshot = self.snapshot_store.read_snapshot(snapshot_id)
            for candidate in snapshot.output_artifact_refs:
                if (
                    candidate.identity is not None
                    and artifact_ref.identity.matches(candidate.identity)
                    and candidate.storage_ref != artifact_ref.storage_ref
                ):
                    return candidate
        return None


@dataclass(frozen=True, slots=True)
class DebugArtifactHydrationContext:
    """Runtime storage context used to hydrate warm replay artifacts."""

    filemanager: FileManagerLike | None = None

    @classmethod
    def from_execution_context(
        cls,
        context: object,
    ) -> "DebugArtifactHydrationContext":
        if isinstance(context, DebugExecutionContext):
            return cls(filemanager=context.filemanager)
        return cls()

    def artifact_available(self, artifact_ref: DebugArtifactRef) -> bool:
        if artifact_ref.storage_backend is not None and self.filemanager is not None:
            try:
                return self.filemanager.exists(
                    artifact_ref.storage_ref,
                    artifact_ref.storage_backend,
                )
            except Exception:
                return False
        return Path(artifact_ref.storage_ref).exists()

    def content_digest(self, artifact_ref: DebugArtifactRef) -> str | None:
        if artifact_ref.storage_backend is not None and self.filemanager is not None:
            vfs_digest = self._vfs_content_digest(artifact_ref)
            if vfs_digest is not None:
                return vfs_digest
        return self._local_content_digest(artifact_ref.storage_ref)

    def content_matches(self, artifact_ref: DebugArtifactRef) -> bool:
        if artifact_ref.content_digest is None:
            return True
        return self.content_digest(artifact_ref) == artifact_ref.content_digest

    def snapshot_with_content_digests(
        self,
        snapshot: "DebugSnapshot",
    ) -> "DebugSnapshot":
        return DebugSnapshot(
            snapshot_id=snapshot.snapshot_id,
            cursor=snapshot.cursor,
            step_name=snapshot.step_name,
            callable_name=snapshot.callable_name,
            axis_id=snapshot.axis_id,
            source_paths=snapshot.source_paths,
            input_artifact_refs=tuple(
                self.ref_with_content_digest(ref)
                for ref in snapshot.input_artifact_refs
            ),
            output_artifact_refs=tuple(
                self.ref_with_content_digest(ref)
                for ref in snapshot.output_artifact_refs
            ),
            preview_refs=tuple(
                self.ref_with_content_digest(ref)
                for ref in snapshot.preview_refs
            ),
            measurement_refs=tuple(
                self.ref_with_content_digest(ref)
                for ref in snapshot.measurement_refs
            ),
            relationship_refs=tuple(
                self.ref_with_content_digest(ref)
                for ref in snapshot.relationship_refs
            ),
            timing_seconds=snapshot.timing_seconds,
            exception=snapshot.exception,
            invocation_parameters=snapshot.invocation_parameters,
        )

    def ref_with_content_digest(
        self,
        artifact_ref: DebugArtifactRef,
    ) -> DebugArtifactRef:
        if artifact_ref.content_digest is not None:
            return artifact_ref
        return artifact_ref.with_content_digest(self.content_digest(artifact_ref))

    def hydrate(
        self,
        *,
        source_ref: DebugArtifactRef,
        destination_ref: DebugArtifactRef,
    ) -> bool:
        if (
            source_ref.storage_backend is not None
            and destination_ref.storage_backend is not None
            and self.filemanager is not None
            and self.content_matches(source_ref)
            and self._hydrate_vfs_artifact(
                source_ref=source_ref,
                destination_ref=destination_ref,
            )
        ):
            return True
        if not self.content_matches(source_ref):
            return False
        return self._hydrate_local_artifact(
            source_ref=source_ref.storage_ref,
            destination_ref=destination_ref.storage_ref,
        )

    def _hydrate_vfs_artifact(
        self,
        *,
        source_ref: DebugArtifactRef,
        destination_ref: DebugArtifactRef,
    ) -> bool:
        if self.filemanager is None:
            return False
        if not self.filemanager.exists(source_ref.storage_ref, source_ref.storage_backend):
            return False
        if self.filemanager.is_dir(source_ref.storage_ref, source_ref.storage_backend):
            return self._hydrate_vfs_directory(
                source_ref=source_ref,
                destination_ref=destination_ref,
            )
        payload = self.filemanager.load(
            source_ref.storage_ref,
            source_ref.storage_backend,
        )
        self.filemanager.save(
            payload,
            destination_ref.storage_ref,
            destination_ref.storage_backend,
        )
        return True

    def _hydrate_vfs_directory(
        self,
        *,
        source_ref: DebugArtifactRef,
        destination_ref: DebugArtifactRef,
    ) -> bool:
        if self.filemanager is None:
            return False
        self.filemanager.ensure_directory(
            destination_ref.storage_ref,
            destination_ref.storage_backend,
        )
        source_root = Path(source_ref.storage_ref)
        destination_root = Path(destination_ref.storage_ref)
        for source_file in self.filemanager.list_files(
            source_ref.storage_ref,
            source_ref.storage_backend,
            recursive=True,
        ):
            relative_path = Path(source_file).relative_to(source_root)
            payload = self.filemanager.load(source_file, source_ref.storage_backend)
            self.filemanager.save(
                payload,
                destination_root / relative_path,
                destination_ref.storage_backend,
            )
        return True

    @staticmethod
    def _hydrate_local_artifact(
        *,
        source_ref: str,
        destination_ref: str,
    ) -> bool:
        source_path = Path(source_ref)
        if not source_path.exists():
            return False
        destination_path = Path(destination_ref)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)
        return True

    def _vfs_content_digest(self, artifact_ref: DebugArtifactRef) -> str | None:
        if self.filemanager is None or artifact_ref.storage_backend is None:
            return None
        if not self.filemanager.exists(artifact_ref.storage_ref, artifact_ref.storage_backend):
            return None
        if self.filemanager.is_dir(artifact_ref.storage_ref, artifact_ref.storage_backend):
            source_root = Path(artifact_ref.storage_ref)
            digest = hashlib.sha256()
            for source_file in self.filemanager.list_files(
                artifact_ref.storage_ref,
                artifact_ref.storage_backend,
                recursive=True,
            ):
                relative_path = str(Path(source_file).relative_to(source_root))
                digest.update(relative_path.encode("utf-8"))
                payload = self.filemanager.load(source_file, artifact_ref.storage_backend)
                digest.update(self._payload_digest_bytes(payload))
            return digest.hexdigest()
        payload = self.filemanager.load(
            artifact_ref.storage_ref,
            artifact_ref.storage_backend,
        )
        return hashlib.sha256(self._payload_digest_bytes(payload)).hexdigest()

    @staticmethod
    def _local_content_digest(storage_ref: str) -> str | None:
        path = Path(storage_ref)
        if not path.exists():
            return None
        if path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                digest.update(str(child.relative_to(path)).encode("utf-8"))
                digest.update(child.read_bytes())
            return digest.hexdigest()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _payload_digest_bytes(payload: object) -> bytes:
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DebugWorkerCommandRequest(DebugSessionRequest):
    """Control-channel request for a persistent paused debug worker."""

    request_kind: ClassVar[str] = "worker_command"

    command_type: DebugCommandType

    def __post_init__(self) -> None:
        DebugSessionRequest.__post_init__(self)
        if not isinstance(self.command_type, DebugCommandType):
            raise TypeError(
                "DebugWorkerCommandRequest.command_type must be DebugCommandType, "
                f"got {type(self.command_type).__name__}."
            )


@dataclass(frozen=True, slots=True)
class DebugWorkerCommandControlPayload:
    """Wire payload for one persistent debug-worker command."""

    debug_session_id: str
    command_type: DebugCommandType
    message_type: DebugControlMessageType = DebugControlMessageType.WORKER_COMMAND

    @classmethod
    def from_request(
        cls,
        request: DebugWorkerCommandRequest,
    ) -> "DebugWorkerCommandControlPayload":
        return cls(
            debug_session_id=request.debug_session_id,
            command_type=request.command_type,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugWorkerCommandControlPayload":
        message_type = DebugControlMessageType(str(payload["type"]))
        if message_type is not DebugControlMessageType.WORKER_COMMAND:
            raise ValueError(
                f"Unsupported debug worker control type: {message_type.value!r}."
            )
        return cls(
            message_type=message_type,
            debug_session_id=str(payload["debug_session_id"]),
            command_type=DebugCommandType(str(payload["command_type"])),
        )

    def to_request(self) -> DebugWorkerCommandRequest:
        return DebugWorkerCommandRequest(
            debug_session_id=self.debug_session_id,
            command_type=self.command_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "debug_session_id": self.debug_session_id,
            "command_type": self.command_type.value,
        }


@dataclass(frozen=True, slots=True)
class DebugWorkerCommandResponse:
    """Control-channel response for a persistent debug-worker command."""

    status: DebugPausedWorkerStatus

    def to_control_response(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "worker": self.status.to_dict(),
        }

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "DebugWorkerCommandResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        return cls(status=DebugPausedWorkerStatus.from_dict(payload["worker"]))


@dataclass(frozen=True, slots=True)
class DebugEvent(DebugBoundaryState):
    """Lightweight debug event emitted from runtime boundaries."""

    boundary_kind: ClassVar[str] = "event"

    event_type: DebugEventType
    traceback_text: str | None = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def for_invocation(
        cls,
        *,
        event_type: DebugEventType,
        cursor: DebugCursor,
        step_name: str,
        callable_name: str | None,
        axis_id: str | None,
        input_artifacts: DebugArtifactRefProjection,
        output_artifacts: DebugArtifactRefProjection = DebugArtifactRefProjection(),
        exception: BaseException | None = None,
        timing_seconds: float | None = None,
        invocation_parameters: tuple[DebugInvocationParameter, ...] = (),
    ) -> "DebugEvent":
        if event_type is DebugEventType.EXCEPTION and exception is None:
            raise ValueError("Exception debug events require an exception.")
        if event_type is not DebugEventType.EXCEPTION and exception is not None:
            raise ValueError("Only exception debug events can carry an exception.")

        event_declaration = DebugBoundaryEventDeclarationBase.for_event_type(event_type)
        reported_output_artifacts = (
            output_artifacts.refs if event_declaration.reports_output_artifacts else ()
        )
        event_ref_projection = (
            output_artifacts
            if event_declaration.reports_output_artifacts
            else input_artifacts
        )
        exception_text = None
        traceback_text = None
        if exception is not None:
            exception_text = f"{type(exception).__name__}: {exception}"
            traceback_text = traceback.format_exc()
        return cls(
            event_type=event_type,
            cursor=cursor,
            step_name=step_name,
            callable_name=callable_name,
            axis_id=axis_id,
            input_artifact_refs=input_artifacts.refs,
            output_artifact_refs=reported_output_artifacts,
            measurement_refs=event_ref_projection.measurement_refs,
            relationship_refs=event_ref_projection.relationship_refs,
            timing_seconds=timing_seconds,
            exception=exception_text,
            traceback_text=traceback_text,
            invocation_parameters=invocation_parameters,
        )

    def to_snapshot(self, *, snapshot_id: str) -> DebugSnapshot:
        """Build a metadata-only snapshot for this runtime event."""

        return DebugSnapshot(
            snapshot_id=snapshot_id,
            cursor=self.cursor,
            step_name=self.step_name,
            callable_name=self.callable_name,
            axis_id=self.axis_id,
            timing_seconds=self.timing_seconds,
            input_artifact_refs=self.input_artifact_refs,
            output_artifact_refs=self.output_artifact_refs,
            preview_refs=self.preview_refs,
            measurement_refs=self.measurement_refs,
            relationship_refs=self.relationship_refs,
            exception=self.exception,
            invocation_parameters=self.invocation_parameters,
        )


@dataclass(frozen=True, slots=True)
class DebugProgressContext:
    """Small debug context intended for ProgressEvent.context transport."""

    progress_context_discriminator: ClassVar[str] = "debug_session_id"

    debug_session_id: str
    snapshot_id: str | None
    cursor: DebugCursor
    event_type: DebugEventType
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None

    @classmethod
    def from_event(
        cls,
        *,
        debug_session_id: str,
        event: DebugEvent,
        snapshot_id: str | None = None,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
    ) -> "DebugProgressContext":
        if not debug_session_id:
            raise ValueError("debug_session_id cannot be empty.")
        return cls(
            debug_session_id=debug_session_id,
            snapshot_id=snapshot_id,
            cursor=event.cursor,
            event_type=event.event_type,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        )

    def to_progress_context(self) -> dict[str, Any]:
        return {
            "debug_session_id": self.debug_session_id,
            "snapshot_id": self.snapshot_id,
            "cursor": DebugJsonCodec.cursor_to_record(self.cursor),
            "event_type": self.event_type.value,
            "snapshot_store_ref": self.snapshot_store_ref,
            "snapshot_store_backend": self.snapshot_store_backend,
        }

    @classmethod
    def from_progress_context(
        cls,
        context: Mapping[str, Any],
    ) -> "DebugProgressContext":
        return cls(
            debug_session_id=str(context["debug_session_id"]),
            snapshot_id=context.get("snapshot_id"),
            cursor=DebugJsonCodec.cursor_from_record(context["cursor"]),
            event_type=DebugEventType(context["event_type"]),
            snapshot_store_ref=context.get("snapshot_store_ref"),
            snapshot_store_backend=context.get("snapshot_store_backend"),
        )

    @classmethod
    def is_progress_context(cls, context: Mapping[str, Any]) -> bool:
        return cls.progress_context_discriminator in context


@dataclass(frozen=True, slots=True)
class DebugProgressEventRequest(DebugSessionRequest):
    """Request for one lightweight ProgressEvent announcing debug state."""

    request_kind: ClassVar[str] = "progress_event"

    debug_event: DebugEvent
    execution_id: str
    plate_id: str
    snapshot_id: str | None = None
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    completed: int = 0
    total: int = 1
    percent: float = 0.0
    worker_slot: str | None = None
    owned_wells: tuple[str, ...] | None = None
    timestamp: float = field(default_factory=time.time)
    pid: int = 0

    @property
    def required_non_empty_values(self) -> tuple[tuple[str, str | None], ...]:
        return (
            ("debug_session_id", self.debug_session_id),
            ("execution_id", self.execution_id),
            ("plate_id", self.plate_id),
        )

    def progress_context(self) -> DebugProgressContext:
        return DebugProgressContext.from_event(
            debug_session_id=self.debug_session_id,
            event=self.debug_event,
            snapshot_id=self.snapshot_id,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
        )

    def to_progress_event(self) -> ProgressEvent:
        if self.debug_event.axis_id is None:
            raise ValueError(
                "DebugProgressEventRequest requires debug_event.axis_id."
            )
        return ProgressEvent(
            identity=ProgressIdentity(
                execution_id=self.execution_id,
                plate_id=self.plate_id,
                axis_id=self.debug_event.axis_id,
                step_name=self.debug_event.step_name,
            ),
            phase=ProgressPhase.PATTERN_GROUP,
            status=DebugBoundaryEventDeclarationBase.for_event_type(
                self.debug_event.event_type
            ).progress_status,
            percent=self.percent,
            completed=self.completed,
            total=self.total,
            timestamp=self.timestamp,
            pid=self.pid,
            message=self.debug_event.event_type.value,
            context=self.progress_context().to_progress_context(),
            error=self.debug_event.exception,
            traceback=self.debug_event.traceback_text,
            worker_slot=self.worker_slot,
            owned_wells=(
                None if self.owned_wells is None else list(self.owned_wells)
            ),
        )


class DebugPausedWorkerController:
    """Thread-shared pause/resume controller for persistent debug replay."""

    def __init__(self, *, debug_session_id: str) -> None:
        if not debug_session_id:
            raise ValueError("debug_session_id cannot be empty.")
        self.debug_session_id = debug_session_id
        self._condition = threading.Condition()
        self._state = DebugPausedWorkerState.RUNNING
        self._cursor: DebugCursor | None = None
        self._context: DebugExecutionContext | None = None
        self._step_permits = 0
        self._continuous = False

    @property
    def status(self) -> DebugPausedWorkerStatus:
        with self._condition:
            return DebugPausedWorkerStatus.from_controller(self)

    def bind_context(self, context: "DebugExecutionContext") -> None:
        with self._condition:
            self._context = context

    def runtime_inspection_view(self) -> "DebugViewModel":
        with self._condition:
            if self._context is None:
                raise RuntimeError("Debug worker context is not available.")
            if self._state is not DebugPausedWorkerState.PAUSED:
                raise RuntimeError(
                    "Runtime inspection requires a paused debug worker."
                )
            from openhcs.core.debug_views import DebugViewModel

            return DebugViewModel.from_runtime_value_store(
                self._context.runtime_value_store,
                title=f"Runtime Values ({self.debug_session_id[:8]})",
            )

    def apply_command(self, command_type: DebugCommandType) -> DebugPausedWorkerStatus:
        with self._condition:
            if command_type is DebugCommandType.STOP:
                self._state = DebugPausedWorkerState.STOPPED
                self._condition.notify_all()
            elif command_type is DebugCommandType.RUN:
                self._continuous = True
                self._state = DebugPausedWorkerState.RUNNING
                self._condition.notify_all()
            elif DebugCommandDeclarationBase.for_command_type(
                command_type
            ).advances_one_boundary:
                self._continuous = False
                self._step_permits += 1
                self._state = DebugPausedWorkerState.RUNNING
                self._condition.notify_all()
            else:
                self._state = DebugPausedWorkerState.PAUSED
            return DebugPausedWorkerStatus.from_controller(self)

    def wait_at_boundary(self, event: DebugEvent) -> None:
        if event.event_type is not DebugEventType.AFTER_INVOCATION:
            return
        with self._condition:
            self._cursor = event.cursor
            if self._continuous:
                self._state = DebugPausedWorkerState.RUNNING
                return
            if self._step_permits > 0:
                self._step_permits -= 1
                self._state = DebugPausedWorkerState.RUNNING
                return
            self._state = DebugPausedWorkerState.PAUSED
            while self._state is DebugPausedWorkerState.PAUSED:
                self._condition.wait()
            if self._state is DebugPausedWorkerState.STOPPED:
                raise DebugWorkerStopped("Persistent debug worker was stopped.")


class DebugWorkerStopped(RuntimeError):
    """Raised inside a persistent worker when the controller receives STOP."""


class DebugPausedWorkerRegistry:
    """Process-local registry for persistent debug worker controllers."""

    _controllers: ClassVar[dict[str, DebugPausedWorkerController]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def controller_for(cls, debug_session_id: str) -> DebugPausedWorkerController:
        with cls._lock:
            controller = cls._controllers.get(debug_session_id)
            if controller is None:
                controller = DebugPausedWorkerController(
                    debug_session_id=debug_session_id
                )
                cls._controllers[debug_session_id] = controller
            return controller

    @classmethod
    def remove(cls, debug_session_id: str) -> None:
        with cls._lock:
            cls._controllers.pop(debug_session_id, None)


@dataclass(frozen=True, slots=True)
class DebugSession:
    """Controller identity for one short-lived debug/test-mode execution."""

    debug_session_id: str
    execution_id: str | None = None
    plate_id: str | None = None
    axis_id: str | None = None
    command_type: DebugCommandType | None = None
    selected_source_group: str | None = None
    cursor: DebugCursor | None = None
    breakpoints: tuple[DebugCursor, ...] = ()
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    dirty_from_cursor: DebugCursor | None = None

    @classmethod
    def create(
        cls,
        *,
        execution_id: str | None = None,
        plate_id: str | None = None,
        axis_id: str | None = None,
        command_type: DebugCommandType | None = None,
    ) -> "DebugSession":
        return cls(
            debug_session_id=str(uuid4()),
            execution_id=execution_id,
            plate_id=plate_id,
            axis_id=axis_id,
            command_type=command_type,
        )

    def with_command(self, command_type: DebugCommandType) -> "DebugSession":
        return DebugSession(
            debug_session_id=self.debug_session_id,
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            axis_id=self.axis_id,
            command_type=command_type,
            selected_source_group=self.selected_source_group,
            cursor=self.cursor,
            breakpoints=self.breakpoints,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
            dirty_from_cursor=self.dirty_from_cursor,
        )

    def with_snapshot_store(
        self,
        *,
        snapshot_store_ref: str | None,
        snapshot_store_backend: str | None,
        axis_id: str | None = None,
    ) -> "DebugSession":
        return DebugSession(
            debug_session_id=self.debug_session_id,
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            axis_id=self.axis_id if axis_id is None else axis_id,
            command_type=self.command_type,
            selected_source_group=self.selected_source_group,
            cursor=self.cursor,
            breakpoints=self.breakpoints,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
            dirty_from_cursor=self.dirty_from_cursor,
        )

    def with_cursor(self, cursor: DebugCursor) -> "DebugSession":
        return DebugSession(
            debug_session_id=self.debug_session_id,
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            axis_id=self.axis_id,
            command_type=self.command_type,
            selected_source_group=self.selected_source_group,
            cursor=cursor,
            breakpoints=self.breakpoints,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
            dirty_from_cursor=self.dirty_from_cursor,
        )

    def mark_dirty_from_cursor(self) -> "DebugSession":
        if self.cursor is None:
            return self
        return DebugSession(
            debug_session_id=self.debug_session_id,
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            axis_id=self.axis_id,
            command_type=self.command_type,
            selected_source_group=self.selected_source_group,
            cursor=self.cursor,
            breakpoints=self.breakpoints,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
            dirty_from_cursor=self.cursor,
        )


@dataclass(frozen=True, slots=True)
class DebugTerminalSummary:
    """Terminal UI/debug summary for one completed debug session."""

    debug_session_id: str
    plate_id: str
    terminal_status: str
    cursor: DebugCursor | None = None
    command_type: DebugCommandType | None = None
    axis_id: str | None = None
    snapshot_id: str | None = None
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    step_name: str | None = None
    callable_name: str | None = None
    completed_at_unix: float | None = None

    @classmethod
    def from_session(
        cls,
        session: DebugSession,
        *,
        terminal_status: str,
        completed_at_unix: float | None = None,
    ) -> "DebugTerminalSummary":
        if session.plate_id is None:
            raise ValueError("DebugTerminalSummary requires session.plate_id.")
        return cls(
            debug_session_id=session.debug_session_id,
            plate_id=session.plate_id,
            terminal_status=terminal_status,
            cursor=session.cursor,
            command_type=session.command_type,
            axis_id=session.axis_id,
            snapshot_store_ref=session.snapshot_store_ref,
            snapshot_store_backend=session.snapshot_store_backend,
            completed_at_unix=completed_at_unix,
        )

    def with_snapshot(
        self,
        *,
        snapshot: DebugSnapshot | None,
        snapshot_id: str | None,
        snapshot_store_ref: str | None,
        snapshot_store_backend: str | None,
    ) -> "DebugTerminalSummary":
        return DebugTerminalSummary(
            debug_session_id=self.debug_session_id,
            plate_id=self.plate_id,
            terminal_status=self.terminal_status,
            cursor=self.cursor if snapshot is None else snapshot.cursor,
            command_type=self.command_type,
            axis_id=self.axis_id if snapshot is None else snapshot.axis_id,
            snapshot_id=snapshot_id,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
            step_name=self.step_name if snapshot is None else snapshot.step_name,
            callable_name=(
                self.callable_name if snapshot is None else snapshot.callable_name
            ),
            completed_at_unix=self.completed_at_unix,
        )


@dataclass(slots=True)
class DebugExecutionConfig:
    """Serializable debug execution settings carried by existing run requests."""

    CONFIG_PARAMS_KEY: ClassVar[str] = "debug_execution"

    debug_session_id: str
    snapshot_store_ref: str | None = None
    snapshot_store_backend: str | None = None
    command_type: DebugCommandType = DebugCommandType.RUN
    selected_source_group: str | None = None
    pause_step_indices: tuple[int, ...] = ()
    start_step_index: int = 0
    start_after_invocation_key: str | None = None
    replay_mode: DebugReplayMode = DebugReplayMode.COLD

    def __post_init__(self) -> None:
        if not self.debug_session_id:
            raise ValueError("DebugExecutionConfig.debug_session_id cannot be empty.")
        if not isinstance(self.command_type, DebugCommandType):
            self.command_type = DebugCommandType(str(self.command_type))
        self.pause_step_indices = tuple(
            int(index) for index in self.pause_step_indices
        )
        if not isinstance(self.replay_mode, DebugReplayMode):
            self.replay_mode = DebugReplayMode(str(self.replay_mode))
        self.start_step_index = int(self.start_step_index)
        if self.start_step_index < 0:
            raise ValueError("start_step_index cannot be negative.")

    def to_config_params(self) -> dict[str, Any]:
        return {
            self.CONFIG_PARAMS_KEY: {
                "debug_session_id": self.debug_session_id,
                "snapshot_store_ref": self.snapshot_store_ref,
                "snapshot_store_backend": self.snapshot_store_backend,
                "command_type": self.command_type.value,
                "selected_source_group": self.selected_source_group,
                "pause_step_indices": list(self.pause_step_indices),
                "start_step_index": self.start_step_index,
                "start_after_invocation_key": self.start_after_invocation_key,
                "replay_mode": self.replay_mode.value,
            }
        }

    def compile_cache_config_params(self) -> dict[str, Any]:
        """Return debug config params containing only compile-relevant state."""

        return DebugExecutionConfig(
            debug_session_id=self.debug_session_id,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
            command_type=DebugCommandType.RUN,
            selected_source_group=self.selected_source_group,
            pause_step_indices=self.pause_step_indices,
            replay_mode=self.replay_mode,
        ).to_config_params()

    @classmethod
    def compatibility_config_params(
        cls,
        config_params: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize debug replay config for compile-artifact compatibility checks."""

        if not config_params:
            return None
        payload = config_params.get(cls.CONFIG_PARAMS_KEY)
        if payload is None:
            return dict(config_params)
        normalized = dict(config_params)
        normalized.update(cls.from_payload(payload).compile_cache_config_params())
        return normalized

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DebugExecutionConfig":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "debug_execution config must be a mapping, "
                f"got {type(payload).__name__}."
            )
        return cls(
            debug_session_id=str(payload["debug_session_id"]),
            snapshot_store_ref=payload.get("snapshot_store_ref"),
            snapshot_store_backend=payload.get("snapshot_store_backend"),
            command_type=DebugCommandType(
                payload.get("command_type", DebugCommandType.RUN.value)
            ),
            selected_source_group=payload.get("selected_source_group"),
            pause_step_indices=tuple(payload.get("pause_step_indices", ())),
            start_step_index=int(payload.get("start_step_index", 0)),
            start_after_invocation_key=payload.get("start_after_invocation_key"),
            replay_mode=DebugReplayMode(
                payload.get("replay_mode", DebugReplayMode.COLD.value)
            ),
        )


class DebugInvocationExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Command-specific invocation execution window for bounded debug runs."""

    __registry_key__ = "command_type"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[DebugCommandType, type["DebugInvocationExecutionStrategy"]]
    ] = {}

    command_type: ClassVar[DebugCommandType | None] = None

    def __init__(self, config: DebugExecutionConfig | None = None) -> None:
        self.config = config

    @classmethod
    def for_config(
        cls,
        config: DebugExecutionConfig,
    ) -> "DebugInvocationExecutionStrategy":
        strategy_type = cls.__registry__.get(
            config.command_type,
            NeverStopDebugInvocationExecutionStrategy,
        )
        return strategy_type(config)

    def should_skip_invocation(self, cursor: DebugCursor) -> bool:
        del cursor
        return False

    def should_stop_after_invocation(self, event: DebugEvent) -> bool:
        del event
        return False


class NeverStopDebugInvocationExecutionStrategy(DebugInvocationExecutionStrategy):
    """Default invocation strategy for commands that stop at step boundaries."""


class StepDebugInvocationExecutionStrategy(DebugInvocationExecutionStrategy):
    """Execute exactly one compiled invocation for the debug STEP command."""

    command_type = DebugCommandType.STEP

    def __init__(self, config: DebugExecutionConfig | None = None) -> None:
        super().__init__(config)
        self._executed_invocations = 0
        self._waiting_for_start_after = bool(
            config and config.start_after_invocation_key
        )

    def should_skip_invocation(self, cursor: DebugCursor) -> bool:
        if self.config is None or not self._waiting_for_start_after:
            return False
        if cursor.invocation_key == self.config.start_after_invocation_key:
            self._waiting_for_start_after = False
        return True

    def should_stop_after_invocation(self, event: DebugEvent) -> bool:
        if event.event_type is not DebugEventType.AFTER_INVOCATION:
            return False
        self._executed_invocations += 1
        return self._executed_invocations >= 1


class DebugStepStopStrategy(ABC, metaclass=AutoRegisterMeta):
    """Command-specific policy for stopping bounded debug execution after a step."""

    __registry_key__ = "command_type"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[DebugCommandType, type["DebugStepStopStrategy"]]] = {}

    command_type: ClassVar[DebugCommandType | None] = None

    def __init__(self, config: DebugExecutionConfig | None = None) -> None:
        self.config = config

    @classmethod
    def for_config(
        cls,
        config: DebugExecutionConfig,
    ) -> "DebugStepStopStrategy":
        strategy_type = cls.__registry__.get(
            config.command_type,
            NeverStopDebugStepStrategy,
        )
        return strategy_type(config)

    @abstractmethod
    def should_stop_after_step(
        self,
        *,
        step_index: int,
        step_name: str,
    ) -> bool:
        """Return whether execution should stop after this completed step."""


class NeverStopDebugStepStrategy(DebugStepStopStrategy):
    """Default command strategy for unbounded debug run commands."""

    def should_stop_after_step(
        self,
        *,
        step_index: int,
        step_name: str,
    ) -> bool:
        del step_index, step_name
        return False


class StepDebugStepStopStrategy(DebugStepStopStrategy):
    """Stop after one completed step."""

    command_type = DebugCommandType.STEP

    def should_stop_after_step(
        self,
        *,
        step_index: int,
        step_name: str,
    ) -> bool:
        del step_name
        if self.config is None:
            raise RuntimeError("Step debug strategy requires a config.")
        return step_index == self.config.start_step_index


class RunToPauseDebugStepStopStrategy(DebugStepStopStrategy):
    """Stop when a completed step is marked as a debug pause boundary."""

    command_type = DebugCommandType.RUN_TO_PAUSE

    def should_stop_after_step(
        self,
        *,
        step_index: int,
        step_name: str,
    ) -> bool:
        del step_name
        if self.config is None:
            raise RuntimeError("Run-to-pause debug strategy requires a config.")
        return step_index in self.config.pause_step_indices


class DebugExecutionPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for installing per-run debug behavior on contexts."""

    __registry_key__ = "policy_kind"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["DebugExecutionPolicy"]]] = {}

    policy_kind: ClassVar[str | None] = None

    @classmethod
    def from_config_params(
        cls,
        config_params: Mapping[str, Any] | None,
    ) -> "DebugExecutionPolicy":
        payload = (
            None
            if not config_params
            else config_params.get(DebugExecutionConfig.CONFIG_PARAMS_KEY)
        )
        if payload is None:
            return NoOpDebugExecutionPolicy()
        return ProgressDebugExecutionPolicy(
            DebugExecutionConfig.from_payload(payload)
        )

    @classmethod
    def from_config(
        cls,
        config: DebugExecutionConfig | None,
    ) -> "DebugExecutionPolicy":
        if config is None:
            return NoOpDebugExecutionPolicy()
        return ProgressDebugExecutionPolicy(config)

    @abstractmethod
    def install_context_sink(
        self,
        request: DebugSinkInstallRequest,
    ) -> None:
        """Install this policy's debug sink on a worker context."""

    def step_stop_strategy(self) -> DebugStepStopStrategy:
        """Return the command-specific step-stop strategy for this policy."""

        return NeverStopDebugStepStrategy()

    def should_execute_step(self, step_index: int) -> bool:
        """Return whether a pipeline step should execute for this debug run."""

        del step_index
        return True

    def should_reuse_step_outputs(self, step_index: int) -> bool:
        """Return whether skipped step outputs are expected to be materialized."""

        del step_index
        return False

    def prepare_reused_step_outputs(
        self,
        *,
        step_index: int,
        step_name: str,
        step_scope_id: str | None,
        context: object,
        artifact_outputs: Mapping[str, ArtifactPlan],
    ) -> None:
        """Validate/hydrate outputs for a skipped warm-replay step."""

        del step_index, step_name, step_scope_id, context, artifact_outputs

    @abstractmethod
    def axis_filter_for_available(
        self,
        available_axis_ids: tuple[str, ...],
    ) -> list[str]:
        """Return the axis IDs this execution policy should compile/run."""

    def compile_worker_count(self, configured_num_workers: int) -> int:
        """Return the worker count the compiler should use for assignments."""

        return configured_num_workers

    def compiled_runtime_environment(
        self,
        runtime_environment: "CompiledRuntimeEnvironmentPlan",
    ) -> "CompiledRuntimeEnvironmentPlan":
        """Return the runtime environment plan selected by this policy."""

        return runtime_environment


class DebugExecutionContext(ABC):
    """Execution context owned by a running pipeline worker."""

    @property
    @abstractmethod
    def filemanager(self) -> FileManagerLike:
        """Return the FileManager used by the active execution context."""

    @property
    @abstractmethod
    def debug_event_sink(self) -> "DebugEventSink":
        """Return the debug sink installed for the active execution."""

    @abstractmethod
    def install_debug_event_sink(self, debug_event_sink: "DebugEventSink") -> None:
        """Install the debug sink selected for this execution context."""

    @property
    @abstractmethod
    def runtime_value_store(self) -> "RuntimeValueStore":
        """Return the RuntimeValueStore owned by the active execution context."""


@dataclass(frozen=True, slots=True)
class DebugSinkInstallRequest:
    """Execution identity needed to install one worker debug sink."""

    context: DebugExecutionContext
    execution_id: str
    plate_id: str
    worker_slot: str
    owned_wells: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, DebugExecutionContext):
            raise TypeError(
                "DebugSinkInstallRequest.context must implement DebugExecutionContext."
            )
        for field_name, value in (
            ("execution_id", self.execution_id),
            ("plate_id", self.plate_id),
            ("worker_slot", self.worker_slot),
        ):
            if not value:
                raise ValueError(f"{field_name} cannot be empty.")


class NoOpDebugExecutionPolicy(DebugExecutionPolicy):
    """Debug execution policy for normal non-debug runs."""

    policy_kind = "noop"

    def install_context_sink(
        self,
        request: DebugSinkInstallRequest,
    ) -> None:
        del request

    def axis_filter_for_available(
        self,
        available_axis_ids: tuple[str, ...],
    ) -> list[str]:
        return list(available_axis_ids)


@dataclass(frozen=True, slots=True)
class ProgressDebugExecutionPolicy(DebugExecutionPolicy):
    """Debug execution policy that emits runtime events as ProgressEvents."""

    policy_kind = "progress"

    config: DebugExecutionConfig

    def install_context_sink(
        self,
        request: DebugSinkInstallRequest,
    ) -> None:
        from openhcs.core.progress import emit_event
        invocation_strategy = DebugInvocationExecutionStrategy.for_config(self.config)
        pause_controller = (
            DebugPausedWorkerRegistry.controller_for(self.config.debug_session_id)
            if self.config.replay_mode is DebugReplayMode.PERSISTENT_PAUSED_WORKER
            else None
        )
        if pause_controller is not None:
            pause_controller.bind_context(request.context)

        if self.config.snapshot_store_ref is not None:
            debug_event_sink: DebugEventSink = LocalSnapshotProgressDebugEventSink(
                debug_session_id=self.config.debug_session_id,
                execution_id=request.execution_id,
                plate_id=request.plate_id,
                emit_progress=emit_event,
                snapshot_store=self.snapshot_store_for_context(request.context),
                snapshot_store_ref=self.config.snapshot_store_ref,
                snapshot_store_backend=self.config.snapshot_store_backend,
                worker_slot=request.worker_slot,
                owned_wells=request.owned_wells,
                invocation_strategy=invocation_strategy,
                pause_controller=pause_controller,
            )
        else:
            debug_event_sink = ProgressDebugEventSink(
                debug_session_id=self.config.debug_session_id,
                execution_id=request.execution_id,
                plate_id=request.plate_id,
                emit_progress=emit_event,
                worker_slot=request.worker_slot,
                owned_wells=request.owned_wells,
                invocation_strategy=invocation_strategy,
                pause_controller=pause_controller,
            )
        request.context.install_debug_event_sink(debug_event_sink)

    def step_stop_strategy(self) -> DebugStepStopStrategy:
        return DebugStepStopStrategy.for_config(self.config)

    def should_execute_step(self, step_index: int) -> bool:
        return step_index >= self.config.start_step_index

    def should_reuse_step_outputs(self, step_index: int) -> bool:
        return (
            self.config.replay_mode.retains_compile_artifact
            and step_index < self.config.start_step_index
        )

    def prepare_reused_step_outputs(
        self,
        *,
        step_index: int,
        step_name: str,
        step_scope_id: str | None,
        context: object,
        artifact_outputs: Mapping[str, ArtifactPlan],
    ) -> None:
        if not self.should_reuse_step_outputs(step_index):
            return
        snapshot_store = (
            self.snapshot_store_for_context(context)
            if self.config.snapshot_store_ref is not None
            else None
        )
        DebugWarmReplayArtifactReusePlan.from_artifact_plans(
            artifact_plans=artifact_outputs,
            cursor=DebugCursor(
                step_index=step_index,
                step_scope_id=step_scope_id,
                group_key=None,
                invocation_key=None,
            ),
            snapshot_store=snapshot_store,
        ).require_available(context)

    def axis_filter_for_available(
        self,
        available_axis_ids: tuple[str, ...],
    ) -> list[str]:
        if self.config.selected_source_group is not None:
            return [self.config.selected_source_group]
        return [] if not available_axis_ids else [available_axis_ids[0]]

    def compile_worker_count(self, configured_num_workers: int) -> int:
        if self.config.replay_mode is DebugReplayMode.PERSISTENT_PAUSED_WORKER:
            return 1
        return configured_num_workers

    def compiled_runtime_environment(
        self,
        runtime_environment: "CompiledRuntimeEnvironmentPlan",
    ) -> "CompiledRuntimeEnvironmentPlan":
        if self.config.replay_mode is not DebugReplayMode.PERSISTENT_PAUSED_WORKER:
            return runtime_environment
        return runtime_environment.with_execution_shape(
            use_threading=True,
            configured_num_workers=1,
        )

    def snapshot_store_for_context(
        self,
        context: DebugExecutionContext,
    ) -> "DebugSnapshotStore":
        if self.config.snapshot_store_ref is None:
            raise RuntimeError("snapshot_store_ref is required for debug snapshot storage.")
        if self.config.snapshot_store_backend is None:
            return LocalDebugSnapshotStore(
                root_path=Path(self.config.snapshot_store_ref),
                debug_session_id=self.config.debug_session_id,
            )
        return FileManagerDebugSnapshotStore(
            filemanager=context.filemanager,
            backend=self.config.snapshot_store_backend,
            root_path=self.config.snapshot_store_ref,
            debug_session_id=self.config.debug_session_id,
        )


class DebugEventSink(metaclass=AutoRegisterMeta):
    """Runtime sink for debug/test-mode events."""

    __registry_key__ = "sink_kind"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["DebugEventSink"]]] = {}

    sink_kind: ClassVar[str | None] = None

    @classmethod
    def registered_types(cls) -> tuple[type["DebugEventSink"], ...]:
        return tuple(cls.__registry__.values())

    def record(self, event: DebugEvent) -> None:
        """Record one debug event."""

    def captures_invocation_events(self) -> bool:
        """Return whether invocation debug events need to be constructed."""
        return True

    def should_skip_invocation(self, cursor: DebugCursor) -> bool:
        del cursor
        return False

    def should_stop_after_invocation(self, event: DebugEvent) -> bool:
        del event
        return False


class NoOpDebugEventSink(DebugEventSink):
    """Default sink used by production execution."""

    sink_kind = "noop"

    def captures_invocation_events(self) -> bool:
        return False

    def record(self, event: DebugEvent) -> None:
        del event


class RecordingDebugEventSink(DebugEventSink):
    """In-memory sink for tests and local diagnostics."""

    sink_kind = "recording"

    def __init__(self) -> None:
        self.events: list[DebugEvent] = []

    def record(self, event: DebugEvent) -> None:
        self.events.append(event)


class ProgressDebugEventSink(DebugEventSink):
    """Debug sink that emits lightweight ProgressEvent records."""

    sink_kind = "progress"

    def __init__(
        self,
        *,
        debug_session_id: str,
        execution_id: str,
        plate_id: str,
        emit_progress: Callable[[ProgressEvent], None],
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
        worker_slot: str | None = None,
        owned_wells: tuple[str, ...] | None = None,
        invocation_strategy: DebugInvocationExecutionStrategy | None = None,
        pause_controller: DebugPausedWorkerController | None = None,
    ) -> None:
        if not debug_session_id:
            raise ValueError("debug_session_id cannot be empty.")
        if not execution_id:
            raise ValueError("execution_id cannot be empty.")
        if not plate_id:
            raise ValueError("plate_id cannot be empty.")
        self.debug_session_id = debug_session_id
        self.execution_id = execution_id
        self.plate_id = plate_id
        self.emit_progress = emit_progress
        self.snapshot_store_ref = snapshot_store_ref
        self.snapshot_store_backend = snapshot_store_backend
        self.worker_slot = worker_slot
        self.owned_wells = owned_wells
        self.invocation_strategy = invocation_strategy or NeverStopDebugInvocationExecutionStrategy()
        self.pause_controller = pause_controller

    def record(self, event: DebugEvent) -> None:
        self.emit_progress(
            DebugProgressEventRequest(
                debug_session_id=self.debug_session_id,
                debug_event=event,
                execution_id=self.execution_id,
                plate_id=self.plate_id,
                snapshot_store_ref=self.snapshot_store_ref,
                snapshot_store_backend=self.snapshot_store_backend,
                worker_slot=self.worker_slot,
                owned_wells=self.owned_wells,
            ).to_progress_event()
        )
        if self.pause_controller is not None:
            self.pause_controller.wait_at_boundary(event)

    def should_skip_invocation(self, cursor: DebugCursor) -> bool:
        return self.invocation_strategy.should_skip_invocation(cursor)

    def should_stop_after_invocation(self, event: DebugEvent) -> bool:
        return self.invocation_strategy.should_stop_after_invocation(event)


class LocalSnapshotProgressDebugEventSink(DebugEventSink):
    """Debug sink that writes local snapshots and emits lightweight progress."""

    sink_kind = "local_snapshot_progress"

    def __init__(
        self,
        *,
        debug_session_id: str,
        execution_id: str,
        plate_id: str,
        emit_progress: Callable[[ProgressEvent], None],
        snapshot_store: "DebugSnapshotStore",
        snapshot_store_ref: str,
        snapshot_store_backend: str | None = None,
        worker_slot: str | None = None,
        owned_wells: tuple[str, ...] | None = None,
        invocation_strategy: DebugInvocationExecutionStrategy | None = None,
        pause_controller: DebugPausedWorkerController | None = None,
    ) -> None:
        if snapshot_store.debug_session_id != debug_session_id:
            raise ValueError(
                "snapshot_store.debug_session_id must match debug_session_id."
            )
        self.debug_session_id = debug_session_id
        self.execution_id = execution_id
        self.plate_id = plate_id
        self.emit_progress = emit_progress
        self.snapshot_store = snapshot_store
        self.snapshot_store_ref = snapshot_store_ref
        self.snapshot_store_backend = snapshot_store_backend
        self.worker_slot = worker_slot
        self.owned_wells = owned_wells
        self.invocation_strategy = invocation_strategy or NeverStopDebugInvocationExecutionStrategy()
        self.pause_controller = pause_controller

    def record(self, event: DebugEvent) -> None:
        snapshot_id = str(uuid4())
        self.snapshot_store.write_snapshot(event.to_snapshot(snapshot_id=snapshot_id))
        self.emit_progress(
            DebugProgressEventRequest(
                debug_session_id=self.debug_session_id,
                debug_event=event,
                execution_id=self.execution_id,
                plate_id=self.plate_id,
                snapshot_id=snapshot_id,
                snapshot_store_ref=self.snapshot_store_ref,
                snapshot_store_backend=self.snapshot_store_backend,
                worker_slot=self.worker_slot,
                owned_wells=self.owned_wells,
            ).to_progress_event()
        )
        if self.pause_controller is not None:
            self.pause_controller.wait_at_boundary(event)

    def should_skip_invocation(self, cursor: DebugCursor) -> bool:
        return self.invocation_strategy.should_skip_invocation(cursor)

    def should_stop_after_invocation(self, event: DebugEvent) -> bool:
        return self.invocation_strategy.should_stop_after_invocation(event)


@dataclass(slots=True)
class DebugStoreManifest:
    """Serializable manifest for local filesystem/VFS-backed debug snapshots."""

    debug_session_id: str
    snapshots: Mapping[str, DebugSnapshot] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        self.snapshots = MappingProxyType(dict(self.snapshots))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "debug_session_id": self.debug_session_id,
            "snapshot_ids": sorted(self.snapshots),
        }


class DebugSnapshotStore(ABC, metaclass=AutoRegisterMeta):
    """Store abstraction for debug snapshot metadata."""

    __registry_key__ = "store_kind"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["DebugSnapshotStore"]]] = {}

    store_kind: ClassVar[str | None] = None
    debug_session_id: str

    @classmethod
    def registered_types(cls) -> tuple[type["DebugSnapshotStore"], ...]:
        return tuple(cls.__registry__.values())

    @abstractmethod
    def write_snapshot(self, snapshot: DebugSnapshot) -> object:
        """Write one snapshot and return the concrete storage reference."""

    @abstractmethod
    def read_snapshot(self, snapshot_id: str) -> DebugSnapshot:
        """Read one snapshot by ID."""

    @abstractmethod
    def list_snapshot_ids(self) -> tuple[str, ...]:
        """List snapshot IDs available in this store."""

    def manifest(self) -> DebugStoreManifest:
        return DebugStoreManifest(
            debug_session_id=self.debug_session_id,
            snapshots=MappingProxyType(
                {
                    snapshot_id: self.read_snapshot(snapshot_id)
                    for snapshot_id in self.list_snapshot_ids()
                }
            ),
        )

    @abstractmethod
    def write_manifest(self) -> object:
        """Persist the current manifest and return the concrete storage reference."""


@dataclass(slots=True)
class LocalDebugSnapshotStore(DebugSnapshotStore):
    """Local filesystem-backed debug snapshot metadata store."""

    store_kind = "local"

    root_path: Path
    debug_session_id: str

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)

    @classmethod
    def for_session(
        cls,
        *,
        root_path: str | Path,
        session: DebugSession,
    ) -> "LocalDebugSnapshotStore":
        return cls(root_path=Path(root_path), debug_session_id=session.debug_session_id)

    @property
    def session_path(self) -> Path:
        return self.root_path / self.debug_session_id

    @property
    def manifest_path(self) -> Path:
        return self.session_path / "manifest.json"

    def snapshot_path(self, snapshot_id: str) -> Path:
        if not snapshot_id:
            raise ValueError("snapshot_id cannot be empty.")
        return self.session_path / f"{snapshot_id}.json"

    def write_snapshot(self, snapshot: DebugSnapshot) -> Path:
        snapshot = DebugArtifactHydrationContext().snapshot_with_content_digests(
            snapshot
        )
        self.session_path.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_path(snapshot.snapshot_id)
        path.write_text(
            json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.write_manifest()
        return path

    def read_snapshot(self, snapshot_id: str) -> DebugSnapshot:
        return DebugSnapshot.from_json_dict(
            json.loads(self.snapshot_path(snapshot_id).read_text(encoding="utf-8"))
        )

    def list_snapshot_ids(self) -> tuple[str, ...]:
        if not self.session_path.exists():
            return ()
        return tuple(
            path.stem
            for path in sorted(self.session_path.glob("*.json"))
            if path.name != self.manifest_path.name
        )

    def write_manifest(self) -> Path:
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest().to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.manifest_path


@dataclass(frozen=True, slots=True)
class FileManagerDebugSnapshotStore(DebugSnapshotStore):
    """FileManager/VFS-backed debug snapshot metadata store."""

    store_kind = "filemanager"

    filemanager: FileManagerLike
    backend: str
    root_path: str | Path
    debug_session_id: str

    @property
    def session_path(self) -> str:
        return str(Path(self.root_path) / self.debug_session_id)

    @property
    def manifest_path(self) -> str:
        return str(Path(self.session_path) / "manifest.json")

    def snapshot_path(self, snapshot_id: str) -> str:
        if not snapshot_id:
            raise ValueError("snapshot_id cannot be empty.")
        return str(Path(self.session_path) / f"{snapshot_id}.json")

    def write_snapshot(self, snapshot: DebugSnapshot) -> str:
        snapshot = DebugArtifactHydrationContext(
            filemanager=self.filemanager,
        ).snapshot_with_content_digests(snapshot)
        self.filemanager.ensure_directory(self.session_path, self.backend)
        path = self.snapshot_path(snapshot.snapshot_id)
        self.filemanager.save(snapshot.to_json_dict(), path, self.backend)
        self.write_manifest()
        return path

    def read_snapshot(self, snapshot_id: str) -> DebugSnapshot:
        payload = self.filemanager.load(
            self.snapshot_path(snapshot_id),
            self.backend,
        )
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Debug snapshot store payload must be a mapping or JSON string, "
                f"got {type(payload).__name__}."
            )
        return DebugSnapshot.from_json_dict(payload)

    def list_snapshot_ids(self) -> tuple[str, ...]:
        if not self.filemanager.exists(self.session_path, self.backend):
            return ()
        return tuple(
            Path(path).stem
            for path in sorted(
                str(path)
                for path in self.filemanager.list_files(
                    self.session_path,
                    self.backend,
                    recursive=False,
                )
            )
            if Path(path).name != "manifest.json"
        )

    def write_manifest(self) -> str:
        self.filemanager.ensure_directory(self.session_path, self.backend)
        self.filemanager.save(
            self.manifest().to_json_dict(),
            self.manifest_path,
            self.backend,
        )
        return self.manifest_path


NO_OP_DEBUG_EVENT_SINK = NoOpDebugEventSink()


def debug_event_sink_from_context(context: DebugExecutionContext) -> DebugEventSink:
    if not isinstance(context, DebugExecutionContext):
        raise TypeError(
            "Function runtime context must implement DebugExecutionContext."
        )
    sink = context.debug_event_sink
    if not isinstance(sink, DebugEventSink):
        raise TypeError(
            "ProcessingContext.debug_event_sink must be DebugEventSink, "
            f"got {type(sink).__name__}."
        )
    return sink


def is_debug_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    ) or name in {"NO_OP_DEBUG_EVENT_SINK", "debug_event_sink_from_context"}


__all__ = tuple(
    name for name, value in globals().items() if is_debug_export(name, value)
)
