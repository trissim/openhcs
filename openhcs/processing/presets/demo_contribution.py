"""Authoritative contribution contract for composed pipeline demonstrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from polystore.streaming.identity import StreamProducerIdentity, StreamProducerOrigin

from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep


@dataclass(frozen=True, slots=True)
class PipelineDemoContribution:
    """One self-contained pipeline declaration accepted by demo schedulers."""

    demo_id: str
    title: str
    plate_path: Path
    pipeline_config: PipelineConfig
    pipeline_steps: tuple[FunctionStep, ...]
    presentation_identity: StreamProducerIdentity
    supporting_presentation_identities: tuple[StreamProducerIdentity, ...] = ()
    prepare: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    biological_question: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.demo_id, str) or not self.demo_id.strip():
            raise ValueError("Pipeline demo contribution demo_id must be non-empty.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Pipeline demo contribution title must be non-empty.")
        if not isinstance(self.plate_path, Path):
            raise TypeError("Pipeline demo contribution plate_path must be a Path.")
        if not isinstance(self.pipeline_config, PipelineConfig):
            raise TypeError(
                "Pipeline demo contribution pipeline_config must be PipelineConfig."
            )
        if (
            type(self.pipeline_steps) is not tuple
            or not self.pipeline_steps
            or not all(isinstance(step, FunctionStep) for step in self.pipeline_steps)
        ):
            raise TypeError(
                "Pipeline demo contribution pipeline_steps must be a non-empty "
                "tuple of FunctionStep declarations."
            )
        if type(self.supporting_presentation_identities) is not tuple:
            raise TypeError(
                "Pipeline demo supporting_presentation_identities must be a tuple."
            )
        if self.prepare is not None and not callable(self.prepare):
            raise TypeError("Pipeline demo contribution prepare must be callable.")
        if self.biological_question is not None and (
            not isinstance(self.biological_question, str)
            or not self.biological_question.strip()
        ):
            raise ValueError(
                "Pipeline demo contribution biological_question must be non-empty."
            )

        step_names = tuple(step.name for step in self.pipeline_steps)
        for identity in self.presentation_identities:
            self._validate_presentation_identity(identity, step_names)

    @property
    def presentation_identities(self) -> tuple[StreamProducerIdentity, ...]:
        """Return the primary visual followed by supporting visual declarations."""

        return (
            self.presentation_identity,
            *self.supporting_presentation_identities,
        )

    @staticmethod
    def _validate_presentation_identity(
        identity: StreamProducerIdentity,
        step_names: tuple[str, ...],
    ) -> None:
        if not isinstance(identity, StreamProducerIdentity):
            raise TypeError(
                "Pipeline demo presentation identities must be "
                "StreamProducerIdentity declarations."
            )
        if (
            identity.origin != StreamProducerOrigin.PIPELINE.value
            or not identity.step_name
            or identity.pipeline_position is not None
            or identity.step_scope_id is not None
            or identity.invocation_key is not None
        ):
            raise ValueError(
                "Pipeline demo presentation identities must be declaration-time "
                "pipeline producers without compiled/runtime scope."
            )
        if step_names.count(identity.step_name) != 1:
            raise ValueError(
                "Pipeline demo presentation identity must name exactly one "
                f"declared FunctionStep: {identity.step_name!r}."
            )
