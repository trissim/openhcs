"""OpenHCS GUI event-bus broadcast helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GuiEventBusBroadcaster:
    """Broadcasts OpenHCS GUI state changes when an event bus is present."""

    event_bus: Any

    def pipeline_changed(self, pipeline_steps: list) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit_pipeline_changed(pipeline_steps)
        logger.debug("Broadcasted pipeline_changed to event bus")

    def config_changed(self, config: Any) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit_config_changed(config)
        logger.debug("Broadcasted config_changed to event bus")
