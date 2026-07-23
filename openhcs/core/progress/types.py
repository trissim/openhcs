"""Immutable progress types following OpenHCS patterns."""

from abc import ABC
from dataclasses import dataclass, replace as dataclass_replace
from enum import Enum
from typing import ClassVar, Dict, Any, Mapping, Optional, List, Protocol
import time

from metaclass_registry import AutoRegisterMeta
from zmqruntime.messages import TaskProgress

# =============================================================================
# ProgressPhase Enum - Unifies TaskPhase + AxisPhase
# =============================================================================


class ProgressPhase(Enum):
    """Progress phases - unified phase vocabulary.

    Extends ZMQRuntime's TaskPhase with OpenHCS-specific phases.
    """

    # Generic phases (from TaskPhase)
    INIT = "init"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Compilation phases
    COMPILE = "compile"

    # Execution phases
    AXIS_STARTED = "axis_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    PATTERN_GROUP = "pattern_group"
    AXIS_COMPLETED = "axis_completed"
    VIEWER_SETTLEMENT = "viewer_settlement"

    # Error phases
    AXIS_ERROR = "axis_error"

    def __str__(self):
        """String representation for logging."""
        return self.value


class ProgressChannelRole(Enum):
    """Nominal role for semantic progress channels."""

    CONTROL = "control"
    EXECUTION = "execution"


class ProgressChannel(Enum):
    """Semantic channel for phase-specific progress streams."""

    INIT = "init"
    COMPILE = "compile"
    PIPELINE = "pipeline"
    STEP = "step"

    def __str__(self):
        return self.value


# =============================================================================
# ProgressStatus Enum - Unifies TaskStatus + AxisStatus
# =============================================================================


class ProgressStatus(Enum):
    """Progress status - unified status vocabulary.

    Extends ZMQRuntime's TaskStatus with OpenHCS-specific statuses.
    """

    # Generic statuses (from TaskStatus)
    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # OpenHCS-specific statuses
    ERROR = "error"
    QUEUED = "queued"

    def __str__(self):
        """String representation for logging."""
        return self.value


class ProgressChannelDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one progress channel."""

    __registry_key__ = "channel"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[ProgressChannel, type["ProgressChannelDeclarationBase"]]
    ] = {}

    channel: ClassVar[ProgressChannel | None] = None
    role: ClassVar[ProgressChannelRole]

    @classmethod
    def require_channel(cls) -> ProgressChannel:
        if cls.channel is None:
            raise TypeError(f"{cls.__name__} does not declare a progress channel.")
        return cls.channel

    @classmethod
    def for_channel(
        cls,
        channel: ProgressChannel,
    ) -> type["ProgressChannelDeclarationBase"]:
        return cls.__registry__[channel]


class ControlProgressChannel:
    """Trait for progress channels that control setup/compile lifecycle."""

    role: ClassVar[ProgressChannelRole] = ProgressChannelRole.CONTROL


class ExecutionProgressChannel:
    """Trait for progress channels that represent execution lifecycle."""

    role: ClassVar[ProgressChannelRole] = ProgressChannelRole.EXECUTION


class InitProgressChannel(ControlProgressChannel, ProgressChannelDeclarationBase):
    channel = ProgressChannel.INIT


class CompileProgressChannel(ControlProgressChannel, ProgressChannelDeclarationBase):
    channel = ProgressChannel.COMPILE


class PipelineProgressChannel(ExecutionProgressChannel, ProgressChannelDeclarationBase):
    channel = ProgressChannel.PIPELINE


class StepProgressChannel(ExecutionProgressChannel, ProgressChannelDeclarationBase):
    channel = ProgressChannel.STEP


class ProgressPhaseDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one progress phase."""

    __registry_key__ = "phase"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[ProgressPhase, type["ProgressPhaseDeclarationBase"]]] = {}

    phase: ClassVar[ProgressPhase | None] = None
    channel: ClassVar[type[ProgressChannelDeclarationBase]]
    is_terminal: ClassVar[bool] = False
    is_failure: ClassVar[bool] = False
    is_success_terminal: ClassVar[bool] = False

    @classmethod
    def require_phase(cls) -> ProgressPhase:
        if cls.phase is None:
            raise TypeError(f"{cls.__name__} does not declare a progress phase.")
        return cls.phase

    @classmethod
    def for_phase(
        cls,
        phase: ProgressPhase,
    ) -> type["ProgressPhaseDeclarationBase"]:
        return cls.__registry__[phase]


class TerminalProgressEvent:
    """Trait for progress declarations that close an event lifecycle."""

    is_terminal: ClassVar[bool] = True


class FailureProgressEvent(TerminalProgressEvent):
    """Trait for terminal progress declarations that represent failure."""

    is_failure: ClassVar[bool] = True


class SuccessTerminalProgressPhase(TerminalProgressEvent):
    """Trait for terminal progress phases that represent successful completion."""

    is_success_terminal: ClassVar[bool] = True


class InitChannelProgressPhase:
    """Trait for progress phases carried on the init channel."""

    channel: ClassVar[type[ProgressChannelDeclarationBase]] = InitProgressChannel


class CompileChannelProgressPhase:
    """Trait for progress phases carried on the compile channel."""

    channel: ClassVar[type[ProgressChannelDeclarationBase]] = CompileProgressChannel


class PipelineChannelProgressPhase:
    """Trait for progress phases carried on the pipeline execution channel."""

    channel: ClassVar[type[ProgressChannelDeclarationBase]] = PipelineProgressChannel


class StepChannelProgressPhase:
    """Trait for progress phases carried on the step execution channel."""

    channel: ClassVar[type[ProgressChannelDeclarationBase]] = StepProgressChannel


class InitProgressPhase(InitChannelProgressPhase, ProgressPhaseDeclarationBase):
    phase = ProgressPhase.INIT


class QueuedProgressPhase(PipelineChannelProgressPhase, ProgressPhaseDeclarationBase):
    phase = ProgressPhase.QUEUED


class RunningProgressPhase(PipelineChannelProgressPhase, ProgressPhaseDeclarationBase):
    phase = ProgressPhase.RUNNING


class SuccessProgressPhase(
    SuccessTerminalProgressPhase,
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.SUCCESS


class FailedProgressPhase(
    FailureProgressEvent,
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.FAILED


class CancelledProgressPhase(
    FailureProgressEvent,
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.CANCELLED


class CompileProgressPhase(CompileChannelProgressPhase, ProgressPhaseDeclarationBase):
    phase = ProgressPhase.COMPILE


class AxisStartedProgressPhase(
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.AXIS_STARTED


class StepStartedProgressPhase(
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.STEP_STARTED


class StepCompletedProgressPhase(
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.STEP_COMPLETED


class PatternGroupProgressPhase(StepChannelProgressPhase, ProgressPhaseDeclarationBase):
    phase = ProgressPhase.PATTERN_GROUP


class AxisCompletedProgressPhase(
    SuccessTerminalProgressPhase,
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.AXIS_COMPLETED


class ViewerSettlementProgressPhase(
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    """Parent-observed progress while streamed viewer state is materialized."""

    phase = ProgressPhase.VIEWER_SETTLEMENT


class AxisErrorProgressPhase(
    FailureProgressEvent,
    PipelineChannelProgressPhase,
    ProgressPhaseDeclarationBase,
):
    phase = ProgressPhase.AXIS_ERROR


class ProgressStatusDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one progress status."""

    __registry_key__ = "status"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[ProgressStatus, type["ProgressStatusDeclarationBase"]]
    ] = {}

    status: ClassVar[ProgressStatus | None] = None
    is_terminal: ClassVar[bool] = False
    is_failure: ClassVar[bool] = False

    @classmethod
    def require_status(cls) -> ProgressStatus:
        if cls.status is None:
            raise TypeError(f"{cls.__name__} does not declare a progress status.")
        return cls.status

    @classmethod
    def for_status(
        cls,
        status: ProgressStatus,
    ) -> type["ProgressStatusDeclarationBase"]:
        return cls.__registry__[status]


class PendingProgressStatus(ProgressStatusDeclarationBase):
    status = ProgressStatus.PENDING


class StartedProgressStatus(ProgressStatusDeclarationBase):
    status = ProgressStatus.STARTED


class RunningProgressStatus(ProgressStatusDeclarationBase):
    status = ProgressStatus.RUNNING


class SuccessProgressStatus(TerminalProgressEvent, ProgressStatusDeclarationBase):
    status = ProgressStatus.SUCCESS


class FailedProgressStatus(FailureProgressEvent, ProgressStatusDeclarationBase):
    status = ProgressStatus.FAILED


class CancelledProgressStatus(FailureProgressEvent, ProgressStatusDeclarationBase):
    status = ProgressStatus.CANCELLED


class ErrorProgressStatus(FailureProgressEvent, ProgressStatusDeclarationBase):
    status = ProgressStatus.ERROR


class QueuedProgressStatus(ProgressStatusDeclarationBase):
    status = ProgressStatus.QUEUED


def phase_channel(phase: ProgressPhase) -> ProgressChannel:
    """Classify phase to semantic channel."""
    return ProgressPhaseDeclarationBase.for_phase(phase).channel.require_channel()


def progress_channel_role(channel: ProgressChannel) -> ProgressChannelRole:
    """Return the nominal role for a progress channel."""
    return ProgressChannelDeclarationBase.for_channel(channel).role


def is_terminal_event(event: "ProgressEvent") -> bool:
    """True when the event is terminal."""
    return (
        ProgressPhaseDeclarationBase.for_phase(event.phase).is_terminal
        or ProgressStatusDeclarationBase.for_status(event.status).is_terminal
    )


def is_execution_phase(phase: ProgressPhase) -> bool:
    """True when phase belongs to execution tree."""
    return progress_channel_role(phase_channel(phase)) is ProgressChannelRole.EXECUTION


def is_failure_event(event: "ProgressEvent") -> bool:
    """True when event represents a failure state."""
    return (
        ProgressPhaseDeclarationBase.for_phase(event.phase).is_failure
        or ProgressStatusDeclarationBase.for_status(event.status).is_failure
    )


def is_success_terminal_event(event: "ProgressEvent") -> bool:
    """True when event represents successful terminal completion."""
    return ProgressPhaseDeclarationBase.for_phase(event.phase).is_success_terminal


# =============================================================================
# ProgressEvent Frozen Dataclass - Single Source of Truth
# =============================================================================


@dataclass(frozen=True)
class ProgressIdentity:
    """Nominal identity for one progress event."""

    execution_id: str
    plate_id: str
    axis_id: str
    step_name: str

    @classmethod
    def from_transport_fields(cls, data: Dict[str, Any]) -> "ProgressIdentity":
        return cls(
            execution_id=str(data["execution_id"]),
            plate_id=str(data["plate_id"]),
            axis_id=str(data["axis_id"]),
            step_name=str(data["step_name"]),
        )


class ProgressQueue(Protocol):
    """Queue contract for serialized progress updates."""

    def put(self, progress_update: dict) -> None:
        """Enqueue a serialized progress update."""


@dataclass(frozen=True, slots=True)
class ProgressExecutionContext:
    """Execution identity carried with progress queue setup."""

    execution_id: str
    plate_id: str

    @classmethod
    def from_transport_fields(
        cls,
        data: Mapping[str, object],
    ) -> "ProgressExecutionContext":
        return cls(
            execution_id=str(data["execution_id"]),
            plate_id=str(data["plate_id"]),
        )

    @classmethod
    def from_value(
        cls,
        value: "ProgressExecutionContext | Mapping[str, object]",
    ) -> "ProgressExecutionContext":
        if isinstance(value, cls):
            return value
        return cls.from_transport_fields(value)

    def to_transport_fields(self) -> dict[str, str]:
        return {
            "execution_id": self.execution_id,
            "plate_id": self.plate_id,
        }

    def identity_for_event(self, *, axis_id: str, step_name: str) -> ProgressIdentity:
        """Return a progress-event identity scoped by this execution."""

        return ProgressIdentity(
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            axis_id=axis_id,
            step_name=step_name,
        )


@dataclass(frozen=True)
class ProgressEventPayload:
    """Nominal payload for constructing progress events."""

    identity: ProgressIdentity
    phase: ProgressPhase
    status: ProgressStatus
    percent: float
    completed: int = 0
    total: int = 1
    error: Optional[str] = None
    traceback: Optional[str] = None
    total_wells: Optional[List[str]] = None
    worker_assignments: Optional[Dict[str, List[str]]] = None
    worker_slot: Optional[str] = None
    owned_wells: Optional[List[str]] = None
    message: Optional[str] = None
    component: Optional[str] = None
    pattern: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def to_event(self, *, timestamp: float, pid: int) -> "ProgressEvent":
        return ProgressEvent(
            identity=self.identity,
            phase=self.phase,
            status=self.status,
            percent=self.percent,
            completed=self.completed,
            total=self.total,
            timestamp=timestamp,
            pid=pid,
            error=self.error,
            traceback=self.traceback,
            total_wells=self.total_wells,
            worker_assignments=self.worker_assignments,
            worker_slot=self.worker_slot,
            owned_wells=self.owned_wells,
            message=self.message,
            component=self.component,
            pattern=self.pattern,
            context=self.context,
        )


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable progress event - single source of truth.

    Replaces dict-based progress payloads with validated, immutable data.
    Uses frozen=True to ensure thread-safety and prevent accidental mutation.

    All fields are explicit and typed - no generic metadata dict.
    """

    # Required core identifiers
    identity: ProgressIdentity

    # Progress tracking
    phase: ProgressPhase
    status: ProgressStatus
    percent: float
    completed: int
    total: int

    # Metadata (timestamp, PID)
    timestamp: float
    pid: int

    # Optional error information
    error: Optional[str] = None
    traceback: Optional[str] = None

    # Optional application-specific fields
    total_wells: Optional[List[str]] = None
    worker_assignments: Optional[Dict[str, List[str]]] = None
    worker_slot: Optional[str] = None
    owned_wells: Optional[List[str]] = None
    message: Optional[str] = None  # General message field (e.g., error messages)
    component: Optional[str] = None  # Component value for pattern group progress
    pattern: Optional[str] = None  # Pattern value for pattern group progress
    context: Optional[Dict[str, Any]] = None  # Generic context for arbitrary data
    step_names: Optional[List[str]] = None  # Step names for the pipeline

    def __post_init__(self) -> None:
        self._validate()

    @property
    def execution_id(self) -> str:
        return self.identity.execution_id

    @property
    def plate_id(self) -> str:
        return self.identity.plate_id

    @property
    def axis_id(self) -> str:
        return self.identity.axis_id

    @property
    def step_name(self) -> str:
        return self.identity.step_name

    def _validate(self):
        """Validate invariants (fail-loud principle)."""
        # Validate percent range
        if not (0.0 <= self.percent <= 100.0):
            raise ValueError(f"percent must be in [0.0, 100.0], got {self.percent}")

        # Validate completed <= total
        if self.completed > self.total:
            raise ValueError(
                f"completed ({self.completed}) cannot exceed total ({self.total})"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressEvent":
        """Create ProgressEvent from dict (for ZMQ transport).

        Converts string phase/status to enums for type safety.

        Args:
            data: Dictionary with progress data (from ZMQ message)

        Returns:
            ProgressEvent instance

        Raises:
            KeyError: If required fields missing
            ValueError: If phase/status strings invalid
            TypeError: If field types invalid
        """
        # Validate generic transport invariants using zmqruntime primitive
        TaskProgress.from_dict(data)

        # Validate OpenHCS-required fields
        required_fields = {
            "execution_id",
            "plate_id",
            "axis_id",
            "step_name",
            "phase",
            "status",
            "percent",
            "completed",
            "total",
            "timestamp",
            "pid",
        }
        missing = required_fields - set(data.keys())
        if missing:
            raise KeyError(
                f"Missing required fields: {missing}. Got keys: {list(data.keys())}"
            )

        # Convert phase string to enum
        phase_str = data["phase"]
        try:
            phase = ProgressPhase(phase_str)
        except ValueError:
            raise ValueError(
                f"Invalid phase '{phase_str}'. Valid phases: "
                f"{[p.value for p in ProgressPhase]}"
            )

        # Convert status string to enum
        status_str = data["status"]
        try:
            status = ProgressStatus(status_str)
        except ValueError:
            raise ValueError(
                f"Invalid status '{status_str}'. Valid statuses: "
                f"{[s.value for s in ProgressStatus]}"
            )

        # Create event with all fields (optional fields use .get())
        return cls(
            identity=ProgressIdentity.from_transport_fields(data),
            phase=phase,
            status=status,
            percent=float(data["percent"]),
            completed=int(data["completed"]),
            total=int(data["total"]),
            timestamp=float(data["timestamp"]),
            pid=int(data["pid"]),
            error=data.get("error"),
            traceback=data.get("traceback"),
            total_wells=data.get("total_wells"),
            worker_assignments=data.get("worker_assignments"),
            worker_slot=data.get("worker_slot"),
            owned_wells=data.get("owned_wells"),
            message=data.get("message"),
            component=data.get("component"),
            pattern=data.get("pattern"),
            context=data.get("context"),
            step_names=data.get("step_names"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for ZMQ transport).

        Converts enums to strings for JSON serialization.
        Only includes optional fields if they are not None.

        Returns:
            Dictionary representation of this event
        """
        result = {
            "execution_id": self.execution_id,
            "plate_id": self.plate_id,
            "axis_id": self.axis_id,
            "step_name": self.step_name,
            "phase": self.phase.value,  # Enum → string
            "status": self.status.value,  # Enum → string
            "percent": self.percent,
            "completed": self.completed,
            "total": self.total,
            "timestamp": self.timestamp,
            "pid": self.pid,
        }

        # Add optional fields if present
        if self.error is not None:
            result["error"] = self.error
        if self.traceback is not None:
            result["traceback"] = self.traceback
        if self.total_wells is not None:
            result["total_wells"] = self.total_wells
        if self.worker_assignments is not None:
            result["worker_assignments"] = self.worker_assignments
        if self.worker_slot is not None:
            result["worker_slot"] = self.worker_slot
        if self.owned_wells is not None:
            result["owned_wells"] = self.owned_wells
        if self.message is not None:
            result["message"] = self.message
        if self.component is not None:
            result["component"] = self.component
        if self.pattern is not None:
            result["pattern"] = self.pattern
        if self.context is not None:
            result["context"] = self.context
        if self.step_names is not None:
            result["step_names"] = self.step_names

        return result

    def with_worker_topology(
        self,
        *,
        worker_assignments: Dict[str, List[str]],
        total_wells: List[str],
    ) -> "ProgressEvent":
        """Return this event with execution topology attached."""
        return dataclass_replace(
            self,
            worker_assignments=worker_assignments,
            total_wells=total_wells,
        )

    def is_complete(self) -> bool:
        """Check if this event represents a completed/terminal state.

        Returns:
            True if the event is in a terminal phase or status
        """
        return is_terminal_event(self)


# =============================================================================
# Utility Functions
# =============================================================================


def create_event(payload: ProgressEventPayload) -> ProgressEvent:
    """Convenience function to create ProgressEvent with defaults.

    Automatically sets timestamp and pid for caller.

    Args:
        payload: Nominal progress payload.

    Returns:
        ProgressEvent instance with timestamp and pid set

    Raises:
        ValueError: If validation fails
    """
    return payload.to_event(
        timestamp=time.time(),
        pid=__import__("os").getpid(),
    )
