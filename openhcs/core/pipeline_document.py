"""Canonical Python document contract for one OpenHCS pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from openhcs.core.config import PipelineConfig
from openhcs.core.pipeline_document_fields import PipelineDocumentField
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep


@dataclass(frozen=True, slots=True)
class PipelineDocument:
    """Normalized public configuration and steps for one pipeline."""

    pipeline_config: PipelineConfig
    pipeline_steps: list[FunctionStep]
    original_source: str | None = None


class PipelineDocumentAuthority:
    """Validate, parse, and render the canonical pipeline document shape."""

    HEADER = "# OpenHCS pipeline"

    @classmethod
    def from_values(
        cls,
        *,
        pipeline_config: PipelineConfig,
        pipeline_steps: Sequence[FunctionStep],
    ) -> PipelineDocument:
        """Build a document from typed values and normalize its steps."""

        cls._require_pipeline_config(pipeline_config)
        steps = list(pipeline_steps)
        cls._require_pipeline_steps(steps)
        normalized_steps = FunctionStepTransportAuthority.normalize_pipeline(steps)
        cls._require_pipeline_steps(normalized_steps)
        return PipelineDocument(
            pipeline_config=pipeline_config,
            pipeline_steps=normalized_steps,
        )

    @classmethod
    def from_namespace(
        cls,
        namespace: Mapping[str, object],
    ) -> PipelineDocument:
        """Read the two exact public assignments from an executed namespace."""

        config_field = PipelineDocumentField.PIPELINE_CONFIG.value
        steps_field = PipelineDocumentField.PIPELINE_STEPS.value
        if config_field not in namespace:
            raise ValueError(f"Pipeline document must define {config_field!r}.")
        if steps_field not in namespace:
            raise ValueError(f"Pipeline document must define {steps_field!r}.")

        pipeline_config = namespace[config_field]
        pipeline_steps = namespace[steps_field]
        cls._require_pipeline_config(pipeline_config)
        if type(pipeline_steps) is not list:
            raise TypeError(
                f"{steps_field} must be a list[FunctionStep], "
                f"got {type(pipeline_steps).__name__}."
            )
        return cls.from_values(
            pipeline_config=pipeline_config,
            pipeline_steps=pipeline_steps,
        )

    @classmethod
    def from_source(cls, source: str) -> PipelineDocument:
        """Execute Python source and read its canonical pipeline assignments."""

        namespace: dict[str, object] = {}
        code = compile(source, "<openhcs-pipeline-document>", "exec")
        exec(code, namespace)
        return replace(cls.from_namespace(namespace), original_source=source)

    @classmethod
    def execution_source(cls, document: PipelineDocument) -> str:
        """Return the reviewed source when present, otherwise canonical source."""

        if document.original_source is not None:
            return document.original_source
        return cls.render(document)

    @classmethod
    def render(
        cls,
        document: PipelineDocument,
        *,
        clean_mode: bool = True,
    ) -> str:
        """Render a validated pipeline document as reviewable Python source."""

        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, BlankLine, CodeBlock
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        normalized = cls.from_values(
            pipeline_config=document.pipeline_config,
            pipeline_steps=document.pipeline_steps,
        )
        body = CodeBlock.from_items(
            (
                Assignment(
                    PipelineDocumentField.PIPELINE_CONFIG.value,
                    normalized.pipeline_config,
                ),
                BlankLine(),
                Assignment(
                    PipelineDocumentField.PIPELINE_STEPS.value,
                    normalized.pipeline_steps,
                ),
            )
        )
        return OpenHCSPythonSourceDocument(
            body,
            header=cls.HEADER,
            clean_mode=clean_mode,
        ).render()

    @staticmethod
    def _require_pipeline_config(value: object) -> None:
        if not isinstance(value, PipelineConfig):
            raise TypeError(
                f"pipeline_config must be a PipelineConfig, got {type(value).__name__}."
            )

    @staticmethod
    def _require_pipeline_steps(values: Sequence[object]) -> None:
        for index, step in enumerate(values):
            if not isinstance(step, FunctionStep):
                raise TypeError(
                    f"pipeline_steps member {index} must be FunctionStep, "
                    f"got {type(step).__name__}."
                )
