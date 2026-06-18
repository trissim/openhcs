"""Draft pipeline authoring service for OpenHCS agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from itertools import count
from typing import TypeAlias

from openhcs.agent.dto.common import (
    AgentError,
    JsonObject,
    RenderedSource,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.pipeline import (
    FunctionSpecRef,
    FunctionStepSpec,
    PipelineConfigRefs,
    PipelineRef,
    PipelineSpec,
    PipelineValidationResult,
)
from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.agent.services.source_rendering_service import PythonSourceAssignmentKind
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.config import (
    LazyDtypeConfig,
    LazyFijiStreamingConfig,
    LazyNapariStreamingConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    LazyStepWellFilterConfig,
)
from openhcs.core.steps.function_step import FunctionStep


StepConfig: TypeAlias = (
    LazyDtypeConfig
    | LazyProcessingConfig
    | LazyStepWellFilterConfig
    | LazyStepMaterializationConfig
    | LazyNapariStreamingConfig
    | LazyFijiStreamingConfig
)
DEFAULT_PIPELINE_CONFIG_REFS = PipelineConfigRefs()


class StepConfigKind(Enum):
    DTYPE = ("LazyDtypeConfig", LazyDtypeConfig)
    PROCESSING = ("LazyProcessingConfig", LazyProcessingConfig)
    STEP_WELL_FILTER = ("LazyStepWellFilterConfig", LazyStepWellFilterConfig)
    STEP_MATERIALIZATION = (
        "LazyStepMaterializationConfig",
        LazyStepMaterializationConfig,
    )
    NAPARI_STREAMING = ("LazyNapariStreamingConfig", LazyNapariStreamingConfig)
    FIJI_STREAMING = ("LazyFijiStreamingConfig", LazyFijiStreamingConfig)

    @property
    def config_class(self) -> type[StepConfig]:
        return self.value[1]

    @classmethod
    def from_patch(cls, patch: ConfigPatch) -> "StepConfigKind":
        for kind in cls:
            class_name, _config_class = kind.value
            if patch.config_type == class_name:
                return kind
        raise ValueError(f"Unknown OpenHCS step config type: {patch.config_type}")


class PipelineAuthoringService:
    """Create and render OpenHCS FunctionStep pipelines from nominal specs."""

    def __init__(
        self,
        function_catalog: FunctionCatalogService | None = None,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._pipelines: dict[str, PipelineSpec] = {}
        self._counter = count(1)
        self._step_counter = count(1)

    def create_pipeline(
        self,
        *,
        steps: Sequence[FunctionStepSpec] = (),
        config_refs: PipelineConfigRefs = DEFAULT_PIPELINE_CONFIG_REFS,
    ) -> PipelineRef:
        pipeline_id = f"pipeline-{next(self._counter)}"
        spec = PipelineSpec(
            schema_version=SCHEMA_VERSION,
            pipeline_id=pipeline_id,
            steps=tuple(steps),
            config_refs=config_refs,
        )
        self._pipelines[pipeline_id] = spec
        return PipelineRef(
            pipeline_id=pipeline_id,
            uri=f"openhcs://pipelines/{pipeline_id}",
        )

    def make_step_spec(
        self,
        *,
        function_id: str,
        name: str | None = None,
        kwargs: JsonObject | None = None,
        step_id: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        debug_pause: bool = False,
    ) -> FunctionStepSpec:
        if kwargs is None:
            function_kwargs = {}
        else:
            function_kwargs = dict(kwargs)
        function = FunctionSpecRef(
            function_id=function_id,
            kwargs=function_kwargs,
        )
        detail = self._function_catalog.get(function_id)
        return FunctionStepSpec(
            step_id=step_id or f"step-{next(self._step_counter)}",
            name=name or detail.entry.name,
            functions=(function,),
            description=description,
            enabled=enabled,
            debug_pause=debug_pause,
        )

    def get_pipeline(self, pipeline_ref: PipelineRef | str) -> PipelineSpec:
        pipeline_id = _pipeline_id(pipeline_ref)
        try:
            return self._pipelines[pipeline_id]
        except KeyError as exc:
            raise KeyError(f"Unknown OpenHCS pipeline_id: {pipeline_id}") from exc

    def add_step(
        self,
        pipeline_ref: PipelineRef | str,
        step_spec: FunctionStepSpec,
        *,
        index: int | None = None,
    ) -> PipelineSpec:
        current = self.get_pipeline(pipeline_ref)
        steps = list(current.steps)
        if any(step.step_id == step_spec.step_id for step in steps):
            raise ValueError(f"Duplicate step_id in pipeline: {step_spec.step_id}")
        if index is None:
            steps.append(step_spec)
        else:
            steps.insert(index, step_spec)
        updated = replace(current, steps=tuple(steps))
        self._pipelines[updated.pipeline_id] = updated
        return updated

    def validate(self, pipeline_ref: PipelineRef | str) -> PipelineValidationResult:
        ref = self._ref(pipeline_ref)
        try:
            self.to_function_steps(ref)
        except Exception as exc:
            return PipelineValidationResult(
                schema_version=SCHEMA_VERSION,
                valid=False,
                pipeline_ref=ref,
                errors=(
                    AgentError.from_exception("pipeline_invalid", exc),
                ),
            )
        return PipelineValidationResult(
            schema_version=SCHEMA_VERSION,
            valid=True,
            pipeline_ref=ref,
        )

    def to_function_steps(self, pipeline_ref: PipelineRef | str) -> list[FunctionStep]:
        spec = self.get_pipeline(pipeline_ref)
        steps = [self._to_function_step(step_spec) for step_spec in spec.steps]
        return FunctionStepTransportAuthority.normalize_pipeline(steps)

    def render_source(
        self,
        pipeline_ref: PipelineRef | str,
        *,
        clean: bool = True,
    ) -> RenderedSource:
        steps = self.to_function_steps(pipeline_ref)
        return RenderedSource(
            schema_version=SCHEMA_VERSION,
            title=f"{_pipeline_id(pipeline_ref)} source",
            source=PythonSourceAssignmentKind.PIPELINE_STEPS.assignment(steps, clean).render(),
        )

    def _to_function_step(self, step_spec: FunctionStepSpec) -> FunctionStep:
        if not step_spec.functions:
            raise ValueError(f"FunctionStepSpec {step_spec.step_id!r} has no functions")
        if step_spec.source_bindings is not None:
            raise ValueError(
                "source_bindings mapping is not accepted by the v1 agent API; "
                "use a nominal source-binding DTO when that API is added."
            )

        step_kwargs = {
            "name": step_spec.name,
            "description": step_spec.description,
            "enabled": step_spec.enabled,
            "debug_pause": step_spec.debug_pause,
        }
        step_kwargs.update(_step_config_kwargs(step_spec))
        return FunctionStep(
            func=self._function_spec(step_spec.functions),
            **step_kwargs,
        )

    def _function_spec(self, refs: tuple[FunctionSpecRef, ...]):
        specs = [self._function_spec_item(ref) for ref in refs]
        if len(specs) == 1:
            return specs[0]
        return specs

    def _function_spec_item(self, ref: FunctionSpecRef):
        if ref.runtime_options:
            raise ValueError(
                "runtime_options are not accepted by the v1 agent API until "
                "nominal RuntimeInvocationOptions DTOs are added."
            )
        func = self._function_catalog.resolve(ref.function_id)
        kwargs = dict(ref.kwargs)
        if not kwargs:
            return func
        return (func, kwargs)

    def _ref(self, pipeline_ref: PipelineRef | str) -> PipelineRef:
        pipeline_id = _pipeline_id(pipeline_ref)
        self.get_pipeline(pipeline_id)
        return PipelineRef(
            pipeline_id=pipeline_id,
            uri=f"openhcs://pipelines/{pipeline_id}",
        )


def _pipeline_id(pipeline_ref: PipelineRef | str) -> str:
    return pipeline_ref.pipeline_id if isinstance(pipeline_ref, PipelineRef) else pipeline_ref


def _step_config_kwargs(step_spec: FunctionStepSpec) -> dict[str, StepConfig]:
    patch_by_kwarg = {
        "dtype_config": step_spec.dtype_config,
        "processing_config": step_spec.processing_config,
        "step_well_filter_config": step_spec.step_well_filter_config,
        "step_materialization_config": step_spec.step_materialization_config,
        "napari_streaming_config": step_spec.napari_streaming_config,
        "fiji_streaming_config": step_spec.fiji_streaming_config,
    }
    return {
        kwarg: _instantiate_config_patch(patch)
        for kwarg, patch in patch_by_kwarg.items()
        if patch is not None
    }


def _instantiate_config_patch(patch: ConfigPatch) -> StepConfig:
    return StepConfigKind.from_patch(patch).config_class(**dict(patch.values))
