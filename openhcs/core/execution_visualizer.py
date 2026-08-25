"""Visualizer execution contract shared by orchestration and viewer runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhcs.runtime.viewer_protocol import (
        ViewerControlResponse,
        ViewerSettleProgress,
    )


class ExecutionVisualizerABC(ABC):
    """Application viewer contract consumed by compiled plate execution."""

    port: int
    persistent: bool

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the viewer endpoint is currently active."""

    @abstractmethod
    def clear_viewer_state(self) -> bool:
        """Clear all viewer state before a new execution."""

    @abstractmethod
    def settle_viewer_state(
        self,
        timeout: float = 30.0,
        *,
        progress_callback: Callable[["ViewerSettleProgress"], None] | None = None,
    ) -> bool:
        """Drain queued viewer updates before execution completes."""

    @abstractmethod
    def read_viewer_state(self, timeout: float = 30.0) -> "ViewerControlResponse":
        """Read typed viewer state while the settled endpoint is still active."""

    @abstractmethod
    def force_stop(self, timeout: float = 5.0) -> None:
        """Stop the viewer process and release its transport endpoint."""

    @abstractmethod
    def rollback_failed_bootstrap(self) -> None:
        """Release resources acquired by an incomplete viewer bootstrap."""
