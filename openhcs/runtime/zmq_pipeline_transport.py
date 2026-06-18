"""Executable pipeline source transport for ZMQ requests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from metaclass_registry import AutoRegisterMeta

from openhcs.core.steps.abstract import AbstractStep

PipelineStepSequence = list[AbstractStep] | tuple[AbstractStep, ...]


@dataclass(frozen=True, slots=True)
class PipelineStepsBoundary(Sequence[AbstractStep]):
    """Nominal carrier for executable OpenHCS pipeline steps."""

    steps: Sequence[AbstractStep]

    def __getitem__(self, index):
        return self.steps[index]

    def __len__(self) -> int:
        return len(self.steps)


class PipelineStepsCarrier(ABC, metaclass=AutoRegisterMeta):
    """Mixin for records that expose executable OpenHCS pipeline steps."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @property
    @abstractmethod
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        raise NotImplementedError

    @property
    def pipeline_steps(self) -> Sequence[AbstractStep]:
        return self.pipeline_steps_boundary.steps


@dataclass(frozen=True, slots=True)
class ZMQPipelineSourcePayload(PipelineStepsCarrier):
    registry_key = "zmq_source_payload"

    source: str
    source_pipeline: PipelineStepsBoundary

    @property
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        return self.source_pipeline


@dataclass(frozen=True, slots=True)
class ZMQPipelineCodeTransport:
    """Recover executable pipeline declarations from pycodified source."""

    source: str

    @classmethod
    def from_pipeline_source(
        cls,
        payload: ZMQPipelineSourcePayload,
    ) -> "ZMQPipelineCodeTransport":
        return cls(source=payload.source)

    @staticmethod
    def pipeline_from_namespace(
        namespace: Mapping[str, PipelineStepSequence],
    ) -> PipelineStepsBoundary | None:
        if "pipeline_steps" not in namespace:
            return None
        return PipelineStepsBoundary(namespace["pipeline_steps"])
