"""Typed progress parsing for benchmark execution logs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Mapping

from openhcs.core.config import Backend
from openhcs.core.vfs_protocol import FileManagerLike


class BenchmarkProgressEventKind(Enum):
    """Semantic event kinds emitted by benchmark/OpenHCS logs."""

    CASE_START = auto()
    CASE_RESULT = auto()
    CASE_EXCEPTION = auto()
    STEP_START = auto()
    STEP_COMPLETE = auto()
    SOURCE_ANCHOR_FILTER = auto()
    WATCHDOG_TIMEOUT = auto()
    COMMAND_STATUS = auto()
    PROCESS_RESOURCE = auto()


@dataclass(frozen=True, slots=True)
class BenchmarkProgressEvent:
    """One parsed benchmark progress event."""

    kind: BenchmarkProgressEventKind
    line_number: int
    text: str
    case_name: str | None = None
    step_index: int | None = None
    step_name: str | None = None
    axis_id: str | None = None
    seconds: float | None = None
    execute_seconds: float | None = None
    before: int | None = None
    after: int | None = None
    success: bool | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None
    command_status: int | None = None
    max_rss_kb: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkCaseProgress:
    """Current progress summary for one benchmark case."""

    case_name: str
    started: bool = False
    finished: bool = False
    success: bool | None = None
    current_axis: str | None = None
    current_step_index: int | None = None
    current_step_name: str | None = None
    completed_axes: tuple[str, ...] = ()
    completed_step_count: int = 0
    metrics: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkProgressSnapshot:
    """Typed snapshot returned by an incremental progress poll."""

    path: Path
    parsed_until_line: int
    events: tuple[BenchmarkProgressEvent, ...]
    cases: Mapping[str, BenchmarkCaseProgress]
    active_case_name: str | None
    command_status: int | None
    max_rss_kb: int | None
    last_event: BenchmarkProgressEvent | None

    @property
    def active_case(self) -> BenchmarkCaseProgress | None:
        """Return the active case progress, if any."""
        if self.active_case_name is None:
            return None
        return self.cases.get(self.active_case_name)


_CASE_START_RE = re.compile(r"^CASE_START (?P<case>\S+)(?:\s+timeout=(?P<timeout>\S+))?")
_CASE_RESULT_RE = re.compile(
    r"^CASE_RESULT (?P<case>\S+) success=(?P<success>True|False)"
    r"(?: metrics=(?P<metrics>\{.*\}))?(?: error=(?P<error>.*))?$"
)
_CASE_EXCEPTION_RE = re.compile(r"^CASE_EXCEPTION (?P<case>\S+)\s+(?P<error>.*)$")
_STEP_START_RE = re.compile(
    r"Starting step '(?P<step>[^']+)' for axis (?P<axis>\S+)"
)
_STEP_COMPLETE_RE = re.compile(
    r"FunctionStep (?P<index>\d+) \((?P<step>[^)]+)\) completed for axis "
    r"(?P<axis>\S+) in (?P<seconds>[0-9.]+)s \(execute=(?P<execute>[0-9.]+)s"
)
_SOURCE_ANCHOR_FILTER_RE = re.compile(
    r"RUNTIME_PROFILE step_filter_source_anchors\b.*\bstep=(?P<index>\d+)"
    r"\s+step_name=(?P<step>\S+)\s+before=(?P<before>\d+)\s+after=(?P<after>\d+)"
)
_WATCHDOG_TIMEOUT_RE = re.compile(
    r"OpenHCS execution exceeded (?P<seconds>[0-9.]+)s watchdog"
)
_COMMAND_STATUS_RE = re.compile(r"^status=(?P<status>\d+)$")
_GNU_TIME_EXIT_STATUS_RE = re.compile(
    r"^(?:Exit status:|Command exited with non-zero status)\s+(?P<status>\d+)$"
)
_GNU_TIME_MAX_RSS_RE = re.compile(
    r"^\s*Maximum resident set size \(kbytes\):\s+(?P<rss>\d+)$"
)


def iter_progress_events(
    path: Path,
    *,
    start_line: int = 0,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
) -> Iterator[BenchmarkProgressEvent]:
    """Yield typed progress events from ``path`` after ``start_line``."""
    if start_line < 0:
        raise ValueError("start_line cannot be negative.")
    for line_number, line in _iter_lines(
        path,
        start_line=start_line,
        filemanager=filemanager,
        backend=backend,
    ):
        event = parse_progress_line(line, line_number=line_number)
        if event is not None:
            yield event


def summarize_progress(
    path: Path,
    *,
    start_line: int = 0,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
) -> BenchmarkProgressSnapshot:
    """Return a typed progress snapshot for all events after ``start_line``."""
    events: list[BenchmarkProgressEvent] = []
    case_builders: dict[str, _CaseProgressBuilder] = {}
    active_case_name: str | None = None
    command_status: int | None = None
    max_rss_kb: int | None = None
    parsed_until_line = start_line

    for line_number, line in _iter_lines(
        path,
        start_line=start_line,
        filemanager=filemanager,
        backend=backend,
    ):
        parsed_until_line = line_number
        event = parse_progress_line(line, line_number=line_number)
        if event is None:
            continue
        events.append(event)
        if event.kind is BenchmarkProgressEventKind.COMMAND_STATUS:
            command_status = event.command_status
            continue
        if event.kind is BenchmarkProgressEventKind.PROCESS_RESOURCE:
            max_rss_kb = event.max_rss_kb
            continue
        if event.case_name is not None:
            active_case_name = event.case_name
            builder = case_builders.setdefault(
                event.case_name,
                _CaseProgressBuilder(case_name=event.case_name),
            )
            builder.apply(event)
            continue
        if active_case_name is not None:
            case_builders[active_case_name].apply(event)

    cases = {
        case_name: builder.freeze()
        for case_name, builder in case_builders.items()
    }
    if active_case_name is not None and cases.get(active_case_name, None) is not None:
        if cases[active_case_name].finished:
            active_case_name = None

    return BenchmarkProgressSnapshot(
        path=path,
        parsed_until_line=parsed_until_line,
        events=tuple(events),
        cases=cases,
        active_case_name=active_case_name,
        command_status=command_status,
        max_rss_kb=max_rss_kb,
        last_event=events[-1] if events else None,
    )


def parse_progress_line(
    line: str,
    *,
    line_number: int,
) -> BenchmarkProgressEvent | None:
    """Parse a single log line into a typed progress event."""
    text = line.rstrip("\n")
    stripped = text.strip()
    if not stripped:
        return None

    if match := _CASE_START_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.CASE_START,
            line_number=line_number,
            text=text,
            case_name=match.group("case"),
        )
    if match := _CASE_RESULT_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.CASE_RESULT,
            line_number=line_number,
            text=text,
            case_name=match.group("case"),
            success=match.group("success") == "True",
            metrics=_parse_metrics(match.group("metrics")),
            error=_normalize_empty(match.group("error")),
        )
    if match := _CASE_EXCEPTION_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.CASE_EXCEPTION,
            line_number=line_number,
            text=text,
            case_name=match.group("case"),
            error=_normalize_empty(match.group("error")),
        )
    if match := _STEP_START_RE.search(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.STEP_START,
            line_number=line_number,
            text=text,
            step_name=match.group("step"),
            axis_id=match.group("axis"),
        )
    if match := _STEP_COMPLETE_RE.search(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.STEP_COMPLETE,
            line_number=line_number,
            text=text,
            step_index=int(match.group("index")),
            step_name=match.group("step"),
            axis_id=match.group("axis"),
            seconds=float(match.group("seconds")),
            execute_seconds=float(match.group("execute")),
        )
    if match := _SOURCE_ANCHOR_FILTER_RE.search(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.SOURCE_ANCHOR_FILTER,
            line_number=line_number,
            text=text,
            step_index=int(match.group("index")),
            step_name=match.group("step"),
            before=int(match.group("before")),
            after=int(match.group("after")),
        )
    if match := _WATCHDOG_TIMEOUT_RE.search(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.WATCHDOG_TIMEOUT,
            line_number=line_number,
            text=text,
            seconds=float(match.group("seconds")),
        )
    if match := _COMMAND_STATUS_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.COMMAND_STATUS,
            line_number=line_number,
            text=text,
            command_status=int(match.group("status")),
        )
    if match := _GNU_TIME_EXIT_STATUS_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.COMMAND_STATUS,
            line_number=line_number,
            text=text,
            command_status=int(match.group("status")),
        )
    if match := _GNU_TIME_MAX_RSS_RE.match(stripped):
        return BenchmarkProgressEvent(
            kind=BenchmarkProgressEventKind.PROCESS_RESOURCE,
            line_number=line_number,
            text=text,
            max_rss_kb=int(match.group("rss")),
        )
    return None


@dataclass(slots=True)
class _CaseProgressBuilder:
    case_name: str
    started: bool = False
    finished: bool = False
    success: bool | None = None
    current_axis: str | None = None
    current_step_index: int | None = None
    current_step_name: str | None = None
    completed_axes: set[str] = field(default_factory=set)
    completed_step_count: int = 0
    metrics: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None

    def apply(self, event: BenchmarkProgressEvent) -> None:
        if event.kind is BenchmarkProgressEventKind.CASE_START:
            self.started = True
        elif event.kind is BenchmarkProgressEventKind.CASE_RESULT:
            self.finished = True
            self.success = event.success
            self.metrics = event.metrics
            self.error = event.error
        elif event.kind is BenchmarkProgressEventKind.CASE_EXCEPTION:
            self.finished = True
            self.success = False
            self.error = event.error
        elif event.kind is BenchmarkProgressEventKind.STEP_START:
            self.current_axis = event.axis_id
            self.current_step_name = event.step_name
        elif event.kind is BenchmarkProgressEventKind.STEP_COMPLETE:
            self.current_axis = event.axis_id
            self.current_step_index = event.step_index
            self.current_step_name = event.step_name
            if event.axis_id is not None:
                self.completed_axes.add(event.axis_id)
            self.completed_step_count += 1
        elif event.kind is BenchmarkProgressEventKind.WATCHDOG_TIMEOUT:
            self.finished = True
            self.success = False
            self.error = f"OpenHCS execution exceeded {event.seconds}s watchdog"

    def freeze(self) -> BenchmarkCaseProgress:
        return BenchmarkCaseProgress(
            case_name=self.case_name,
            started=self.started,
            finished=self.finished,
            success=self.success,
            current_axis=self.current_axis,
            current_step_index=self.current_step_index,
            current_step_name=self.current_step_name,
            completed_axes=tuple(sorted(self.completed_axes)),
            completed_step_count=self.completed_step_count,
            metrics=self.metrics,
            error=self.error,
        )


def _iter_lines(
    path: Path,
    *,
    start_line: int,
    filemanager: FileManagerLike | None,
    backend: Backend,
) -> Iterator[tuple[int, str]]:
    if filemanager is not None:
        content = filemanager.load(path, backend.value)
        if not isinstance(content, str):
            raise TypeError("Progress logs loaded through FileManager must be text.")
        for line_number, line in enumerate(content.splitlines(), start=1):
            if line_number > start_line:
                yield line_number, line
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > start_line:
                yield line_number, line


def _parse_metrics(raw_metrics: str | None) -> Mapping[str, object]:
    if not raw_metrics:
        return {}
    try:
        parsed = ast.literal_eval(raw_metrics)
    except (SyntaxError, ValueError):
        return {"unparsed": raw_metrics}
    if isinstance(parsed, dict):
        return parsed
    return {"unparsed": raw_metrics}


def _normalize_empty(value: str | None) -> str | None:
    if value is None or value == "None":
        return None
    return value
