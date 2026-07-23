"""Runtime profiling event sink shared by execution adapters."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
from typing import Any


PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"


@dataclass(frozen=True, slots=True)
class RuntimeProfileTimer:
    """Runtime-profile timer that owns disabled-profile elapsed semantics."""

    enabled: bool
    started_at: float

    @classmethod
    def start(cls) -> "RuntimeProfileTimer":
        """Start a timer under the current profile sink state."""
        if RuntimeProfileLogger.enabled():
            return cls(enabled=True, started_at=time.perf_counter())
        return cls(enabled=False, started_at=0.0)

    def elapsed(self) -> float:
        """Return elapsed seconds, or the declared disabled-profile value."""
        if not self.enabled:
            return 0.0
        return time.perf_counter() - self.started_at


class RuntimeProfileLogger:
    """Environment-gated writer for runtime profile events."""

    @staticmethod
    def enabled() -> bool:
        return os.environ.get(PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}

    @classmethod
    def log(
        cls,
        logger: logging.Logger,
        label: str,
        seconds: float,
        **fields: Any,
    ) -> None:
        if not cls.enabled():
            return
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
        if profile_path := os.environ.get(PROFILE_RUNTIME_PATH_ENV):
            with open(profile_path, "a", encoding="utf-8") as handle:
                handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")
