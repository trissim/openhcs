"""Executable pipeline source transport for ZMQ requests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from metaclass_registry import AutoRegisterMeta

from openhcs.core.steps.abstract import AbstractStep

PipelineStepSequence = Sequence[AbstractStep]


@dataclass(frozen=True, slots=True)
class PipelineStepsBoundary(Sequence[AbstractStep]):
    """Nominal carrier for executable OpenHCS pipeline steps."""

    steps: list[AbstractStep]

    def __init__(self, steps: PipelineStepSequence) -> None:
        # Compilation resolves step ObjectState by replacing the submitted sequence
        # in-place, so the transport boundary owns the mutable list contract.
        from openhcs.core.pipeline import Pipeline

        if not isinstance(steps, Pipeline):
            object.__setattr__(self, "steps", list(steps))
            return

        object.__setattr__(
            self,
            "steps",
            Pipeline(
                steps=list(steps),
                name=steps.name,
                metadata=dict(steps.metadata),
                description=steps.description,
                step_scope_ids=steps.step_scope_ids,
            ),
        )

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


class PipelineSourceExport(str, Enum):
    """Named exports produced by pycodified pipeline transport source."""

    PIPELINE_STEPS = "pipeline_steps"


@dataclass(frozen=True, slots=True)
class PipelineStepsNamespaceProjection:
    """Read the executable pipeline export from an executed source namespace."""

    namespace: Mapping[str, PipelineStepSequence]

    def boundary_or_none(self) -> PipelineStepsBoundary | None:
        export_name = PipelineSourceExport.PIPELINE_STEPS.value
        if export_name not in self.namespace:
            return None
        return PipelineStepsBoundary(self.namespace[export_name])


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
