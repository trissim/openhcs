"""Typed phase timing for benchmark runs."""

from __future__ import annotations

import csv
import io
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterator

from openhcs.core.config import Backend
from openhcs.core.vfs_protocol import FileManagerLike


class BenchmarkPhase(Enum):
    """Semantic benchmark phases reported independently."""

    RESOLVE_SOURCE = auto()
    PARSE_CPPIPE = auto()
    COMPILE_DIALECT = auto()
    MATERIALIZE_SOURCE_SCHEMA = auto()
    INITIALIZE_RUNTIME = auto()
    COMPILE_OPENHCS = auto()
    EXECUTE_OPENHCS = auto()
    SUBMIT_OPENHCS = auto()
    WAIT_OPENHCS = auto()
    EXECUTE_NATIVE_CP = auto()
    VALIDATE_RUNTIME = auto()
    SNAPSHOT_OUTPUTS = auto()
    COMPARE_EQUIVALENCE = auto()
    READ_CACHE = auto()
    WRITE_CACHE = auto()


@dataclass(frozen=True, slots=True)
class PhaseTimingRecord:
    """One observed benchmark phase duration."""

    run_id: str
    pipeline_name: str
    tool: str
    phase: BenchmarkPhase
    seconds: float
    cached: bool = False

    def as_payload(self) -> dict[str, object]:
        """Return a JSON/CSV-stable record payload."""
        payload = asdict(self)
        payload["phase"] = self.phase.name
        return payload


class PhaseTimingTrace:
    """Append-only phase timing trace for one benchmark run."""

    def __init__(self, *, run_id: str, pipeline_name: str, tool: str) -> None:
        if not run_id:
            raise ValueError("PhaseTimingTrace.run_id cannot be empty.")
        if not pipeline_name:
            raise ValueError("PhaseTimingTrace.pipeline_name cannot be empty.")
        if not tool:
            raise ValueError("PhaseTimingTrace.tool cannot be empty.")
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.tool = tool
        self._records: list[PhaseTimingRecord] = []

    @property
    def records(self) -> tuple[PhaseTimingRecord, ...]:
        """Recorded phases in observation order."""
        return tuple(self._records)

    @contextmanager
    def phase(
        self,
        phase: BenchmarkPhase,
        *,
        cached: bool = False,
    ) -> Iterator[None]:
        """Time a benchmark phase and append its record."""
        normalized_phase = BenchmarkPhase(phase)
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                normalized_phase,
                seconds=time.perf_counter() - started_at,
                cached=cached,
            )

    def record(
        self,
        phase: BenchmarkPhase,
        *,
        seconds: float,
        cached: bool = False,
    ) -> None:
        """Append an externally measured phase duration."""
        if seconds < 0:
            raise ValueError("PhaseTimingRecord.seconds cannot be negative.")
        self._records.append(
            PhaseTimingRecord(
                run_id=self.run_id,
                pipeline_name=self.pipeline_name,
                tool=self.tool,
                phase=BenchmarkPhase(phase),
                seconds=float(seconds),
                cached=bool(cached),
            )
        )

    def payloads(self) -> tuple[dict[str, object], ...]:
        """Return JSON/CSV-stable payloads for all records."""
        return tuple(record.as_payload() for record in self._records)


def write_phase_timing_jsonl(
    path: Path,
    records: tuple[PhaseTimingRecord, ...],
    *,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
) -> None:
    """Write phase timing records as newline-delimited JSON."""
    content = "".join(
        json.dumps(record.as_payload(), sort_keys=True) + "\n"
        for record in records
    )
    _save_text(path, content, filemanager=filemanager, backend=backend)


def write_phase_timing_csv(
    path: Path,
    records: tuple[PhaseTimingRecord, ...],
    *,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
) -> None:
    """Write phase timing records as a long-table CSV."""
    fieldnames = ("run_id", "pipeline_name", "tool", "phase", "seconds", "cached")
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(record.as_payload())
    _save_text(path, handle.getvalue(), filemanager=filemanager, backend=backend)


def _save_text(
    path: Path,
    content: str,
    *,
    filemanager: FileManagerLike | None,
    backend: Backend,
) -> None:
    """Save text through FileManager when available, otherwise use local disk."""
    if filemanager is not None:
        filemanager.ensure_directory(path.parent, backend.value)
        filemanager.save(content, path, backend.value)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
