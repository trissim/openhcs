"""Runtime profiling helpers for CellProfiler integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging

from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerProfileFields,
    CellProfilerRuntimeValue,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeProfileEvent:
    """Deferred profile event with explicit structured fields."""

    name: str
    elapsed: float
    fields: CellProfilerProfileFields = ()


class CellProfilerRuntimeProfileLogger:
    """Runtime profile sink with one owner for environment-gated logging."""

    @staticmethod
    def enabled() -> bool:
        """Return whether CellProfiler runtime profile events will be emitted."""
        return RuntimeProfileLogger.enabled()

    @staticmethod
    def log_module_profile(
        label: str,
        seconds: float,
        **fields: CellProfilerRuntimeValue,
    ) -> None:
        RuntimeProfileLogger.log(logger, label, seconds, **fields)

    @classmethod
    def log_module_profile_deferred(
        cls,
        label: str,
        seconds: float,
        fields: Callable[[], Mapping[str, CellProfilerRuntimeValue]],
    ) -> None:
        """Emit a profile event whose fields are only built when profiling is on."""
        if not cls.enabled():
            return
        cls.log_module_profile(label, seconds, **dict(fields()))
