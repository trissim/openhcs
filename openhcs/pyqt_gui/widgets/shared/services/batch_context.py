"""Shared execution context for PlateManager batch workflow services."""

from __future__ import annotations

from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from openhcs.core.config import GlobalPipelineConfig
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import (
    ZMQExecutionClientBoundary,
)

T = TypeVar("T")
RunBlockingCallable = Callable[[AbstractEventLoop, Callable[[], T]], Awaitable[T]]
ProgressClientConnector = Callable[[], Awaitable[ZMQExecutionClient]]
GlobalConfigProvider = Callable[[], GlobalPipelineConfig]


@dataclass(frozen=True, slots=True)
class BatchWorkflowContext:
    """Nominal carrier for shared execution support services."""

    zmq: ZMQExecutionClientBoundary
    global_config_provider: GlobalConfigProvider
    run_blocking: RunBlockingCallable
    connect_progress_client: ProgressClientConnector

    def global_config(self) -> GlobalPipelineConfig:
        return self.global_config_provider()
