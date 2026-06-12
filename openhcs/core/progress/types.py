"""Immutable progress types following OpenHCS patterns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import time
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

    def __new__(cls, value: str, role: ProgressChannelRole):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._role = role
        return obj

    INIT = ("init", ProgressChannelRole.CONTROL)
    COMPILE = ("compile", ProgressChannelRole.CONTROL)
    PIPELINE = ("pipeline", ProgressChannelRole.EXECUTION)
    STEP = ("step", ProgressChannelRole.EXECUTION)

    @property
    def role(self) -> ProgressChannelRole:
        return self._role

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


class ProgressSemanticsABC(ABC):
    """Nominal contract for progress phase semantics."""

    @abstractmethod
    def channel_for_phase(self, phase: ProgressPhase) -> ProgressChannel:
        """Classify phase into a semantic channel."""

    @abstractmethod
    def is_terminal(self, event: "ProgressEvent") -> bool:
        """Return True when event is terminal."""

    @abstractmethod
    def is_execution_phase(self, phase: ProgressPhase) -> bool:
        """Return True when phase belongs to execution."""


class ProgressSemantics(ProgressSemanticsABC):
    """Single source of truth for phase semantics."""

    _PHASE_TO_CHANNEL = {
        ProgressPhase.INIT: ProgressChannel.INIT,
        ProgressPhase.QUEUED: ProgressChannel.PIPELINE,
        ProgressPhase.RUNNING: ProgressChannel.PIPELINE,
        ProgressPhase.SUCCESS: ProgressChannel.PIPELINE,
        ProgressPhase.FAILED: ProgressChannel.PIPELINE,
        ProgressPhase.CANCELLED: ProgressChannel.PIPELINE,
        ProgressPhase.COMPILE: ProgressChannel.COMPILE,
        ProgressPhase.AXIS_STARTED: ProgressChannel.PIPELINE,
        ProgressPhase.STEP_STARTED: ProgressChannel.PIPELINE,
        ProgressPhase.STEP_COMPLETED: ProgressChannel.PIPELINE,
        ProgressPhase.PATTERN_GROUP: ProgressChannel.STEP,
        ProgressPhase.AXIS_COMPLETED: ProgressChannel.PIPELINE,
        ProgressPhase.AXIS_ERROR: ProgressChannel.PIPELINE,
    }
    _TERMINAL_PHASES = {
        ProgressPhase.SUCCESS,
        ProgressPhase.FAILED,
        ProgressPhase.CANCELLED,
        ProgressPhase.AXIS_COMPLETED,
        ProgressPhase.AXIS_ERROR,
    }
    _TERMINAL_STATUSES = {
        ProgressStatus.SUCCESS,
        ProgressStatus.FAILED,
        ProgressStatus.CANCELLED,
        ProgressStatus.ERROR,
    }

    def channel_for_phase(self, phase: ProgressPhase) -> ProgressChannel:
        return self._PHASE_TO_CHANNEL[phase]

    def is_terminal(self, event: "ProgressEvent") -> bool:
        return (
            event.phase in self._TERMINAL_PHASES
            or event.status in self._TERMINAL_STATUSES
        )

    def is_execution_phase(self, phase: ProgressPhase) -> bool:
        channel = self.channel_for_phase(phase)
        return channel.role is ProgressChannelRole.EXECUTION


_PROGRESS_SEMANTICS = ProgressSemantics()
_FAILURE_STATUSES = {
    ProgressStatus.FAILED,
    ProgressStatus.ERROR,
    ProgressStatus.CANCELLED,
}
_FAILURE_PHASES = {
    ProgressPhase.FAILED,
    ProgressPhase.CANCELLED,
    ProgressPhase.AXIS_ERROR,
}
_SUCCESS_TERMINAL_PHASES = {
    ProgressPhase.SUCCESS,
    ProgressPhase.AXIS_COMPLETED,
}


def phase_channel(phase: ProgressPhase) -> ProgressChannel:
    """Classify phase to semantic channel."""
    return _PROGRESS_SEMANTICS.channel_for_phase(phase)


def is_terminal_event(event: "ProgressEvent") -> bool:
    """True when the event is terminal."""
    return _PROGRESS_SEMANTICS.is_terminal(event)


def is_execution_phase(phase: ProgressPhase) -> bool:
    """True when phase belongs to execution tree."""
    return _PROGRESS_SEMANTICS.is_execution_phase(phase)


def is_failure_event(event: "ProgressEvent") -> bool:
    """True when event represents a failure state."""
    return event.status in _FAILURE_STATUSES or event.phase in _FAILURE_PHASES


def is_success_terminal_event(event: "ProgressEvent") -> bool:
    """True when event represents successful terminal completion."""
    return event.phase in _SUCCESS_TERMINAL_PHASES


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

    def replace(
        self,
        *,
        execution_id: str | None = None,
        plate_id: str | None = None,
        axis_id: str | None = None,
        step_name: str | None = None,
    ) -> "ProgressIdentity":
        return ProgressIdentity(
            execution_id=self.execution_id if execution_id is None else str(execution_id),
            plate_id=self.plate_id if plate_id is None else str(plate_id),
            axis_id=self.axis_id if axis_id is None else str(axis_id),
            step_name=self.step_name if step_name is None else str(step_name),
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


@dataclass(frozen=True, init=False)
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

    def __init__(
        self,
        *,
        phase: ProgressPhase,
        status: ProgressStatus,
        percent: float,
        completed: int,
        total: int,
        timestamp: float,
        pid: int,
        identity: ProgressIdentity | None = None,
        execution_id: str | None = None,
        plate_id: str | None = None,
        axis_id: str | None = None,
        step_name: str | None = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None,
        total_wells: Optional[List[str]] = None,
        worker_assignments: Optional[Dict[str, List[str]]] = None,
        worker_slot: Optional[str] = None,
        owned_wells: Optional[List[str]] = None,
        message: Optional[str] = None,
        component: Optional[str] = None,
        pattern: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        step_names: Optional[List[str]] = None,
    ) -> None:
        if identity is None:
            missing = [
                name
                for name, value in (
                    ("execution_id", execution_id),
                    ("plate_id", plate_id),
                    ("axis_id", axis_id),
                    ("step_name", step_name),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "ProgressEvent requires identity or identity fields; "
                    f"missing {missing}"
                )
            identity = ProgressIdentity(
                execution_id=str(execution_id),
                plate_id=str(plate_id),
                axis_id=str(axis_id),
                step_name=str(step_name),
            )
        else:
            overrides = {
                "execution_id": execution_id,
                "plate_id": plate_id,
                "axis_id": axis_id,
                "step_name": step_name,
            }
            mismatches = [
                name
                for name, value in overrides.items()
                if value is not None and str(value) != getattr(identity, name)
            ]
            if mismatches:
                raise ValueError(
                    "ProgressEvent identity fields conflict with identity: "
                    f"{mismatches}"
                )

        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "percent", percent)
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "pid", pid)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "traceback", traceback)
        object.__setattr__(self, "total_wells", total_wells)
        object.__setattr__(self, "worker_assignments", worker_assignments)
        object.__setattr__(self, "worker_slot", worker_slot)
        object.__setattr__(self, "owned_wells", owned_wells)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "step_names", step_names)
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

    def replace(self, **kwargs) -> "ProgressEvent":
        """Create a copy with replaced fields (immutable update pattern).

        Returns:
            New ProgressEvent with specified fields replaced
        """
        values = {
            "identity": self.identity,
            "phase": self.phase,
            "status": self.status,
            "percent": self.percent,
            "completed": self.completed,
            "total": self.total,
            "timestamp": self.timestamp,
            "pid": self.pid,
            "error": self.error,
            "traceback": self.traceback,
            "total_wells": self.total_wells,
            "worker_assignments": self.worker_assignments,
            "worker_slot": self.worker_slot,
            "owned_wells": self.owned_wells,
            "message": self.message,
            "component": self.component,
            "pattern": self.pattern,
            "context": self.context,
            "step_names": self.step_names,
        }
        values["identity"] = self.identity.replace(
            execution_id=kwargs.pop("execution_id", None),
            plate_id=kwargs.pop("plate_id", None),
            axis_id=kwargs.pop("axis_id", None),
            step_name=kwargs.pop("step_name", None),
        )
        values.update(kwargs)
        return ProgressEvent(**values)

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
