"""Worker-side profiling policies for orchestrator execution."""

from __future__ import annotations

import cProfile
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


WORKER_PROFILE_DIR_ENV = "OPENHCS_WORKER_PROFILE_DIR"


class WorkerProfilingPolicy(Protocol):
    """Policy boundary for optional worker-side execution profiling."""

    @contextmanager
    def profile(
        self,
        *,
        execution_id: str,
        plate_id: str,
        worker_slot: str,
        owned_wells: list[str],
    ) -> Iterator[None]:
        """Profile a worker execution region when the policy is active."""
        yield


@dataclass(frozen=True)
class DisabledWorkerProfilingPolicy:
    """No-op worker profiling policy."""

    @contextmanager
    def profile(
        self,
        *,
        execution_id: str,
        plate_id: str,
        worker_slot: str,
        owned_wells: list[str],
    ) -> Iterator[None]:
        with nullcontext():
            yield


@dataclass(frozen=True)
class CProfileWorkerProfilingPolicy:
    """Dump cProfile stats for worker execution regions."""

    output_dir: Path

    @classmethod
    def from_environment(cls) -> WorkerProfilingPolicy:
        profile_dir = os.environ.get(WORKER_PROFILE_DIR_ENV)
        if not profile_dir:
            return DisabledWorkerProfilingPolicy()
        return cls(Path(profile_dir))

    @contextmanager
    def profile(
        self,
        *,
        execution_id: str,
        plate_id: str,
        worker_slot: str,
        owned_wells: list[str],
    ) -> Iterator[None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            profiler.dump_stats(
                str(
                    self.output_dir
                    / self.profile_filename(
                        execution_id=execution_id,
                        plate_id=plate_id,
                        worker_slot=worker_slot,
                        owned_wells=owned_wells,
                    )
                )
            )

    def profile_filename(
        self,
        *,
        execution_id: str,
        plate_id: str,
        worker_slot: str,
        owned_wells: list[str],
    ) -> str:
        well_token = "all" if not owned_wells else "-".join(sorted(owned_wells))
        fields = (execution_id, plate_id, worker_slot, well_token)
        return "__".join(self.filename_component(field) for field in fields) + ".prof"

    @staticmethod
    def filename_component(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in value
        )
