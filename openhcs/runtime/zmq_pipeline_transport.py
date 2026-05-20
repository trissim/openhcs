"""Executable pipeline source transport for ZMQ requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ZMQPipelineCodeTransport:
    """Recover executable pipeline declarations from pycodified source."""

    source: str

    @classmethod
    def from_pipeline_source(
        cls,
        *,
        source: str,
        pipeline_steps: list[Any],
    ) -> "ZMQPipelineCodeTransport":
        return cls(source=source)

    @staticmethod
    def pipeline_from_namespace(namespace: Mapping[str, Any]) -> Any:
        return namespace.get("pipeline_steps")
